"""wdb_pairfold -- SELECT a, b, COUNT(*) AS c FROM t WHERE b <> '' GROUP BY a, b ORDER BY c DESC LIMIT k
(ClickBench Q14: SearchEngineID x SearchPhrase). THE PLIST PATH: the big key's position lists hand
each candidate's rows; candidates in descending count, the partner's code gathered at their rows,
a certificate that no excluded b can host a better pair. THE BAR (2026-09-29): the top candidates are
selected (wdb_kernels.topk_bar), not sorted out of every b. A LIMIT with tied counts has more than one
legal answer, so an answer is checked for LEGALITY against DuckDB's full grouping: every row's count
is that pair's true count, rows descend by count, and the counts are exactly the true top-k counts."""
import sys, os, uuid, tempfile, shutil, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_pairfold, wdb_kernels
from wdb_db import Database
from wdb_catalog import Catalog
from wdb_engine import Segment

TMP = tempfile.gettempdir()


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def _db(n=600_000, seed=14):
    """se: 16 engines, 70% 0, in short runs (mean 2) -- the shape that elects blocked zstd (tag 3),
    as the kit's SearchEngineID does (iid values elect tag 8, long runs tag 10); p: ~87% empty,
    the rest uniform phrases (tag 8, as the kit's SearchPhrase; a Zipf mix elects tag 3), with
    a small heavy head so the top counts are distinct and many pairs tie at small counts"""
    rng = np.random.default_rng(seed)
    m = n // 2 + 10
    se = np.repeat(np.where(rng.random(m) < 0.7, 0, rng.integers(1, 16, m)),
                   rng.geometric(0.5, m))[:n].astype(np.int64)
    ph = rng.integers(1, 40_001, n)
    hv = rng.random(n) < 0.02
    ph[hv] = rng.integers(1, 30, int(hv.sum()))
    p = np.array(['phrase %05d' % v for v in ph], dtype=object)
    p[rng.random(n) < 0.87] = ''
    df = pd.DataFrame({'se': se, 'p': p, 'x': rng.integers(0, 1000, n).astype(np.int64)})
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'pf_{t}'); pq = os.path.join(TMP, f'pf_{t}.parquet')
    df.to_parquet(pq, index=False)
    os.makedirs(d); Catalog.create(d)
    out = os.path.join(d, 'hits_0.wdb')
    with _env(WDB_SEQ_NARROW_OK='0'):
        wdb_encode.encode(pq, out, stream=True)
    seg = Segment(out); cat = Catalog.open(d)
    tn = {0: 'int', 1: 'str', 2: 'float'}
    cat.data['tables']['hits'] = {'schema': [[c, tn.get(seg.cols[c].get('dt'), 'str')]
                                             for c in wdb_encode.input_column_order(pq, seg.order)],
                                  'segments': ['hits_0.wdb'], 'mode': 'segment'}
    cat.save()
    return d, pq, seg


def _legal(rows, truth, k):
    """rows (a, b, c) are a legal top-k of truth {(a, b): count}"""
    want = sorted(truth.values(), reverse=True)[:k]
    cs = [int(r[2]) for r in rows]
    assert len(rows) == len(want), (len(rows), len(want))
    assert cs == sorted(cs, reverse=True), cs
    assert cs == want, (cs, want)
    for a, b, c in rows:
        b = b.decode() if isinstance(b, bytes) else b
        assert truth.get((int(a), b)) == int(c), (a, b, c, truth.get((int(a), b)))
    assert len({(int(a), b) for a, b, c in rows}) == len(rows)


def test_q14_shape_legal_against_duck():
    d, pq, seg = _db()
    try:
        enc = {c: seg.cols[c].get('code_enc') for c in ('se', 'p')}
        assert enc == {'se': 3, 'p': 8}, ('toy no longer encodes like the kit', enc)
        db = Database.open(d)
        truth = {(int(a), b): int(c) for a, b, c in duckdb.connect().execute(
            f"SELECT se, p, COUNT(*) FROM '{pq}' WHERE p <> '' GROUP BY se, p").fetchall()}
        bars = []
        bar0 = wdb_kernels.topk_bar
        def _bar(cnt, m):
            bars.append(int(m)); return bar0(cnt, m)
        wdb_kernels.topk_bar = _bar
        try:
            # k = 20000: 4k candidates exceed every b present -> the whole census, no exclusion bar
            for k in (1, 10, 100, 1000, 20000):
                sql = "SELECT se, p, COUNT(*) AS c FROM hits WHERE p <> '' GROUP BY se, p ORDER BY c DESC LIMIT %d" % k
                h0 = wdb_pairfold._HITS; nb = len(bars)
                rows = db.run(sql)[0]
                assert wdb_pairfold._HITS == h0 + 1, ('pairfold did not serve', enc, sql)
                assert len(bars) > nb, ('the plist bar path did not serve', sql)
                _legal(rows, truth, min(k, len(truth)))
                # THE OVERLAP and THE COLD READ off (codes_band after the census, plist by faults):
                # the same path, the same codes -- the same rows, in the same order
                sw = (wdb_pairfold._OVERLAP[0], wdb_pairfold._PFWARM[0])
                wdb_pairfold._OVERLAP[0] = wdb_pairfold._PFWARM[0] = False
                try:
                    rows0 = db.run(sql)[0]
                finally:
                    wdb_pairfold._OVERLAP[0], wdb_pairfold._PFWARM[0] = sw
                assert [tuple(r) for r in rows0] == [tuple(r) for r in rows], ('switches changed the answer', k)
        finally:
            wdb_kernels.topk_bar = bar0
        print('encodings', enc, 'bars', bars)
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def _db30(n=600_000, seed=30):
    """Q30's shape on a toy: p the filter (~87% empty, tag 8 as above); a a 6-value key, 80% one value;
    s a 6-value partner (the small pair space -> the dense fold); w a ~60K-value partner whose 0 holds
    half the rows (6 x 60K > DENSE_MAX -> the buckets); x, y the summed columns"""
    rng = np.random.default_rng(seed)
    p = np.array(['phrase %05d' % v for v in rng.integers(1, 40_001, n)], dtype=object)
    p[rng.random(n) < 0.87] = ''
    a = np.where(rng.random(n) < 0.8, 0, rng.integers(1, 6, n)).astype(np.int64)
    s = np.where(rng.random(n) < 0.7, 3, rng.integers(0, 6, n)).astype(np.int64)
    w = np.where(rng.random(n) < 0.5, 0, rng.integers(1, 60_000, n)).astype(np.int64)
    # n (~300 values) x q (~500): a small key past one byte, dense (150K slots); n x m (~1000): past one byte, buckets
    nn = np.where(rng.random(n) < 0.4, 7, rng.integers(0, 300, n)).astype(np.int64)
    q = rng.integers(0, 500, n).astype(np.int64)
    m = np.where(rng.random(n) < 0.6, 0, rng.integers(0, 1000, n)).astype(np.int64)
    u = rng.permutation(n).astype(np.int64)                 # a near-unique partner (WatchID's profile)
    df = pd.DataFrame({'p': p, 'a': a, 's': s, 'w': w, 'n': nn, 'q': q, 'm': m, 'u': u,
                       'x': rng.integers(0, 2, n).astype(np.int64),
                       'y': rng.integers(0, 3000, n).astype(np.int64)})
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'pf30_{t}'); pq = os.path.join(TMP, f'pf30_{t}.parquet')
    df.to_parquet(pq, index=False)
    os.makedirs(d); Catalog.create(d)
    out = os.path.join(d, 'hits_0.wdb')
    with _env(WDB_SEQ_NARROW_OK='0'):
        wdb_encode.encode(pq, out, stream=True)
    seg = Segment(out); cat = Catalog.open(d)
    tn = {0: 'int', 1: 'str', 2: 'float'}
    cat.data['tables']['hits'] = {'schema': [[c, tn.get(seg.cols[c].get('dt'), 'str')]
                                             for c in wdb_encode.input_column_order(pq, seg.order)],
                                  'segments': ['hits_0.wdb'], 'mode': 'segment'}
    cat.save()
    return d, pq, seg


def _legal5(rows, truth, k):
    """rows (a, b, c, sum x, avg y) are a legal top-k of truth {(a, b): (c, sum x, avg y)}"""
    want = sorted((v[0] for v in truth.values()), reverse=True)[:k]
    cs = [int(r[2]) for r in rows]
    assert cs == want, (cs, want)
    for a, b, c, sx, ay in rows:
        tc, tsx, tay = truth[(int(a), int(b))]
        assert (int(c), int(sx)) == (tc, tsx), ((a, b), (c, sx), (tc, tsx))
        assert abs(float(ay) - tay) < 1e-9 * max(1.0, abs(tay)), ((a, b), ay, tay)
    assert len({(int(r[0]), int(r[1])) for r in rows}) == len(rows)


def test_q30_shape_dense_and_buckets_legal_against_duck():
    """the dense fold (small pair space) and the buckets (wide partner) both serve, both legal; the dense
    fold switched off, the buckets answer the small pair space too, with the same counts"""
    d, pq, seg = _db30()
    try:
        db = Database.open(d)
        con = duckdb.connect()
        cases = []
        for where in (" WHERE p <> ''", ''):          # '' is Q32's form: no filter, every row
            # (first key, second key, the dense fold expected): a 6-value key with a 6- and a 60K-value partner;
            # a ~300-value key (past one byte) with a ~500 (dense) and a ~1000-value partner (buckets)
            cases += [('a', 's', where, True), ('a', 'w', where, False), ('n', 'q', where, True), ('m', 'n', where, False)]
        for ka, part, where, want_dense in cases:
            truth = {(int(a), int(b)): (int(c), int(sx), float(ay)) for a, b, c, sx, ay in con.execute(
                f"SELECT {ka}, {part}, COUNT(*), SUM(x), AVG(y) FROM '{pq}'{where} GROUP BY {ka}, {part}").fetchall()}
            for k in (1, 10, 100):
                sql = ("SELECT %s, %s, COUNT(*) AS c, SUM(x), AVG(y) FROM hits%s GROUP BY %s, %s "
                       "ORDER BY c DESC LIMIT %d" % (ka, part, where, ka, part, k))
                h0, dn0 = wdb_pairfold._HITS, wdb_pairfold._DENSED
                rows = db.run(sql)[0]
                assert wdb_pairfold._HITS == h0 + 1, ('pairfold did not serve', sql)
                dense = wdb_pairfold._DENSED == dn0 + 1
                assert dense == want_dense, ('wrong branch', ka, part, where, dense)
                _legal5(rows, truth, min(k, len(truth)))
                wdb_pairfold._DENSE[0] = False
                try:
                    rows0 = db.run(sql)[0]
                finally:
                    wdb_pairfold._DENSE[0] = True
                assert wdb_pairfold._DENSED == dn0 + int(dense), 'the switch did not turn the dense fold off'
                _legal5(rows0, truth, min(k, len(truth)))
                assert [int(r[2]) for r in rows0] == [int(r[2]) for r in rows]
        # an unfiltered COUNT-only pair stays gridwalk's (its composed tally), not pairfold's
        h0 = wdb_pairfold._HITS
        db.run("SELECT a, s, COUNT(*) AS c FROM hits GROUP BY a, s ORDER BY c DESC LIMIT 10")
        assert wdb_pairfold._HITS == h0, 'pairfold took an unfiltered COUNT-only pair'
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_near_unique_partner_is_handed_on():
    """a partner with a value per ~row (WatchID) is not pairfold's: the pair is handed on (pairtop's walk),
    and the answer that comes back is still a legal top-k"""
    d, pq, seg = _db30()
    try:
        db = Database.open(d)
        truth = {(int(a), int(b)): (int(c), int(sx), float(ay)) for a, b, c, sx, ay in duckdb.connect().execute(
            f"SELECT a, u, COUNT(*), SUM(x), AVG(y) FROM '{pq}' WHERE p <> '' GROUP BY a, u").fetchall()}
        h0 = wdb_pairfold._HITS
        rows = db.run("SELECT a, u, COUNT(*) AS c, SUM(x), AVG(y) FROM hits WHERE p <> '' GROUP BY a, u "
                      "ORDER BY c DESC LIMIT 10")[0]
        assert wdb_pairfold._HITS == h0, 'pairfold kept a near-unique partner'
        _legal5(rows, truth, 10)
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_warm_plist_reads_cold_pages_once():
    """warm_plist: evicted position lists are read (bytes > 0), then found resident (0); the offsets
    and lists read through the map afterwards equal the file's own bytes"""
    import wdb_funnel as F
    d, pq, seg = _db(n=200_000, seed=3)
    try:
        offs, pos = F._plist(seg, 'p')          # (mapped, only the header page touched)
        p = F._plist_path(seg, 'p')
        raw = open(p, 'rb').read()              # the truth, by read(): nothing mapped is touched
        V9 = int(np.frombuffer(raw[:8], np.int64)[0])
        P0 = 8 + 8 * (V9 + 1)
        top = np.argsort(np.diff(np.frombuffer(raw[8:P0], np.int64)))[::-1][:5]
        fd = os.open(p, os.O_RDONLY)
        os.fsync(fd)                            # a just-written file's dirty pages cannot be dropped
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        n1 = F.warm_plist(seg, 'p', offsets=True, codes=top)
        n2 = F.warm_plist(seg, 'p', offsets=True, codes=top)
        assert n1 > 0, ('the evicted lists were not read', n1)
        assert n2 == 0, ('resident lists were read again', n2)
        assert np.array_equal(np.frombuffer(raw[8:P0], np.int64), offs)
        for c9 in top.tolist():
            a, b = int(offs[c9]), int(offs[c9 + 1])
            assert np.array_equal(np.frombuffer(raw[P0 + 4 * a:P0 + 4 * b], np.uint32), pos[a:b])
        print('warm_plist cold bytes', n1)
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)
