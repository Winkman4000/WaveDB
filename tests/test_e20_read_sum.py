"""THE READ-VALUES SUM through the back-reference (2026-10-22, Q29's RefererHash): SUM / AVG of an enc-20 integer column
by weighing each block's literals with the rows that lead back to them and reading value x weight from the unzipped
dictionary -- no census. Oracle: the census fold, an exact Python sum, and DuckDB through SQL."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_engine as EN, controller
from wdb_engine import Segment
from wdb_db import Database

TMP = tempfile.gettempdir()
ENV = dict(WDB_HASH_COLS='h', WDB_E20_BR='4096', WDB_I2CHUNK='1024', WDB_I2CHUNK_MIN='1000', WDB_SEQ_NARROW_OK='0')
_FIX = None


def _col(seed=5):
    rng = np.random.default_rng(seed)
    n, V = 400_000, 100_000
    pool = rng.integers(-(1 << 62), 1 << 62, V, dtype=np.int64)
    x = pool[rng.integers(0, V, n)]
    near = rng.random(n) < 0.5                                  # back-references, some chained
    back = np.maximum(0, np.arange(n) - rng.integers(1, 40, n))
    x[near] = x[back[near]]
    return x


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    x = _col()
    keep = {k: os.environ.get(k) for k in ENV}
    os.environ.update(ENV)
    try:
        d = os.path.join(TMP, 'e20rs_%s' % uuid.uuid4().hex[:8]); os.makedirs(d)
        pq = os.path.join(d, 't.parquet'); pd.DataFrame({'h': x}).to_parquet(pq, index=False)
        db = Database.create(d); db.cat.add_table('t', [['h', 'int']])
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb')
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    con = duckdb.connect(); con.execute("CREATE TABLE t AS SELECT * FROM read_parquet('%s')" % pq)
    _FIX = (x, db, con, Segment(db.cat.segment_paths('t')[0]))
    return _FIX


def test_equals_the_census_and_the_exact_sum():
    x, _db, _con, seg = _fixture()
    c = seg.cols['h']
    assert c['code_enc'] == 20 and c.get('i2ch') is not None and c.get('mode') == 2, (c.get('code_enc'), c.get('mode'))
    old = EN._E20RS_MIN[0]
    try:
        EN._E20RS_MIN[0] = 0
        nd = int(c.get('n_dict', c['V']))
        r = seg.e20_read_sum('h')
        assert r is not None
        assert r == (int(x.astype(object).sum()), x.size)
        assert r[0] == seg.i2_fold_sum('h', seg.raw_census('h', nd))
        EN._E20RS_MIN[0] = nd + 1                               # under the size line: declines
        assert seg.e20_read_sum('h') is None
        EN._E20RS_MIN[0] = 0; EN._E20RS[0] = False              # switch off: declines
        assert seg.e20_read_sum('h') is None
    finally:
        EN._E20RS_MIN[0] = old; EN._E20RS[0] = True


def test_sql_sum_avg_affine():
    x, db, con, _seg = _fixture()
    old = EN._E20RS_MIN[0]
    try:
        EN._E20RS_MIN[0] = 0
        for q in ("SELECT SUM(h) FROM t", "SELECT SUM(h), SUM(h + 1), SUM(h + 7), COUNT(*) FROM t"):
            h0 = EN._E20RS_HITS
            g = db.run(q); g = g[0] if isinstance(g, tuple) else g
            e = con.execute(q).fetchall()
            assert controller._SERVED[0] == 'fused_agg' and EN._E20RS_HITS > h0, (q, controller._SERVED[0])
            assert [tuple(int(v) for v in r) for r in g] == [tuple(int(v) for v in r) for r in e], (q, g, e)
    finally:
        EN._E20RS_MIN[0] = old
