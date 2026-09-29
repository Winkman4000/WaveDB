"""LOAD_ANSWERS (wdb_blockstats._ANSWERS; WDB_LOAD_ANSWERS=0): the load statistics may steer but never
BE an answer. With it off: no SUM/AVG from block sums (COUNT and MIN/MAX from per-block counts and
zone maps stay), no per-value counts as a GROUP BY's result or a filtered COUNT, no repeat list as a
pair board -- yet the planner may still read the per-value counts as estimates (plan=True). Every
answer equals DuckDB with the switch on and off. And with sidecars off the load writes no pair
tables (the differentiator shelves obey the switch)."""
import sys, os, uuid, tempfile, shutil, contextlib, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_blockstats, wdb_sidecar
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


def _db(n=200_000, seed=5, sidecars='off'):
    rng = np.random.default_rng(seed)
    # near-unique random 64-bit ids (a WatchID's shape: not a sequence, so not mode 4 -- a mode-4
    # column's codes are positions, and GROUP BY on one with repeated values is a separate open bug)
    w = rng.integers(1 << 40, 1 << 62, n, dtype=np.int64)
    dup = rng.choice(n, 8, replace=False); w[dup[:4]] = w[dup[4:]]  # four repeated values, eight rows
    ipd = rng.integers(0, 50, n)
    ipd[dup[4:]] = ipd[dup[:4]]                                     # ...and each repeat keeps its ip: the
                                                                    # pair board's top counts are 2
    df = pd.DataFrame({
        'w': w, 'ip': ipd.astype(np.int64),
        'a': np.where(rng.random(n) < 0.8, 0, rng.integers(1, 12, n)).astype(np.int64),
        'x': rng.integers(0, 1000, n).astype(np.int64),
        'rw': rng.integers(300, 2000, n).astype(np.int64)})
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'la_{t}'); pq = os.path.join(TMP, f'la_{t}.parquet')
    df.to_parquet(pq, index=False)
    os.makedirs(d); Catalog.create(d)
    wdb_sidecar.set_setting(d, sidecars)
    out = os.path.join(d, 'hits_0.wdb')
    with _env(WDB_SEQ_NARROW_OK='0', WDB_SIDECARS='1' if sidecars == 'on' else '0'):   # (the suite exports 1)
        wdb_encode.encode(pq, out, stream=True)
    seg = Segment(out); cat = Catalog.open(d)
    tn = {0: 'int', 1: 'str', 2: 'float'}
    cat.data['tables']['hits'] = {'schema': [[c, tn.get(seg.cols[c].get('dt'), 'str')]
                                             for c in wdb_encode.input_column_order(pq, seg.order)],
                                  'segments': ['hits_0.wdb'], 'mode': 'segment'}
    cat.save()
    return d, pq


QS = [
    ("SELECT SUM(x), COUNT(*), AVG(rw) FROM hits", 'exact'),
    ("SELECT AVG(w) FROM hits", 'exact'),
    ("SELECT MIN(x), MAX(x), COUNT(x) FROM hits", 'exact'),
    ("SELECT COUNT(*) FROM hits WHERE a <> 0", 'exact'),
    ("SELECT a, COUNT(*) AS c FROM hits WHERE a <> 0 GROUP BY a ORDER BY c DESC", 'exact'),
    ("SELECT SUM(rw), SUM(rw + 1), SUM(rw + 2), SUM(rw + 3) FROM hits", 'exact'),
    ("SELECT w, ip, COUNT(*) AS c, SUM(x), AVG(rw) FROM hits GROUP BY w, ip ORDER BY c DESC LIMIT 10", 'top'),
]


def _check(rows, sql, pq, kind):
    con = duckdb.connect()
    if kind == 'exact':
        want = con.execute(sql.replace('FROM hits', f"FROM '{pq}'")).fetchall()
        got = [tuple(r) for r in rows]
        assert len(got) == len(want), (sql, got, want)
        for g, e in zip(got, want):
            for gv, ev in zip(g, e):
                assert abs(float(gv) - float(ev)) < 1e-6 * max(1.0, abs(float(ev))), (sql, g, e)
        return
    full = sql.replace('FROM hits', f"FROM '{pq}'").split(' ORDER BY')[0]
    truth = {(int(r[0]), int(r[1])): int(r[2]) for r in con.execute(full).fetchall()}
    want = sorted(truth.values(), reverse=True)[:10]
    assert [int(r[2]) for r in rows] == want, (sql, [r[2] for r in rows], want)
    for r in rows:
        assert truth[(int(r[0]), int(r[1]))] == int(r[2]), (sql, r)


def test_answers_equal_duck_with_the_switch_on_and_off():
    d, pq = _db()
    sw = wdb_blockstats._ANSWERS[0]
    try:
        seg = Segment(os.path.join(d, 'hits_0.wdb'))
        assert wdb_blockstats._load_npz(seg) is not None, 'the load wrote no statistics'
        assert seg.cols['w'].get('mode') != 4, ('the toy id became a sequence', seg.cols['w'].get('mode'))
        for on in (True, False):
            wdb_blockstats._ANSWERS[0] = on
            db = Database.open(d)
            for sql, kind in QS:
                h0 = wdb_blockstats._HITS
                rows = db.run(sql)[0]
                _check(rows, sql, pq, kind)
                if not on and ('SUM(' in sql or 'AVG(' in sql) and 'GROUP' not in sql:
                    assert wdb_blockstats._HITS == h0, ('block sums answered with LOAD_ANSWERS off', sql)
                if on and sql.startswith('SELECT SUM(x), COUNT(*)'):
                    assert wdb_blockstats._HITS == h0 + 1, 'block sums did not serve with LOAD_ANSWERS on'
            assert wdb_blockstats.rep_from_load(seg, 'w') is None if not on else True
            assert (wdb_blockstats.vcnt_from_load(seg, 'a') is None) == (not on)
            assert wdb_blockstats.vcnt_from_load(seg, 'a', plan=True) is not None   # estimates stay
        wdb_blockstats._ANSWERS[0] = True
        assert wdb_blockstats.rep_from_load(seg, 'w') is not None, 'the load found no repeat list'
    finally:
        wdb_blockstats._ANSWERS[0] = sw
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_sidecars_off_load_writes_no_pair_tables():
    d, pq = _db(n=120_000, seed=6, sidecars='off')
    try:
        derived = [os.path.basename(f) for f in glob.glob(os.path.join(d, 'hits_0.wdb.*'))
                   if f.endswith(('.pt2', '.ptrep'))]
        assert derived == [], derived
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)
