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
        finally:
            wdb_kernels.topk_bar = bar0
        print('encodings', enc, 'bars', bars)
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)
