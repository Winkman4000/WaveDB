"""THE BOUND HUNT (wdb_bound, behind fused_agg's peel door): GROUP BY several keys ORDER BY COUNT(*) DESC LIMIT k
over a composite too wide for the dense table -- a group's count never exceeds any one of its key values' row
counts, so rows whose smallest key-value count is below the threshold are dropped before grouping, and the
k-th count proves the cut (2026-10-13, the Q18 board: tripletop's double bound, made general).
Checked against DuckDB with total orders; the hunt is asserted taken, and asserted to decline where the bounds
cannot prune."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, controller
from wdb_db import Database
import wdb_bound

N = 400_000
_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'bnd_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    # a, b: heavy heads and long tails (users, phrases); u: wide and flat (no bound prunes it); x: a measure
    _CON.execute("""CREATE TABLE t AS SELECT i AS id,
        CAST(floor(60000 * pow(((i * 7919) %% 100003) / 100003.0, 4)) AS BIGINT) AS a,
        CAST(floor(30000 * pow(((i * 104729) %% 99991) / 99991.0, 5)) AS BIGINT) AS b,
        (i * 31) %% 150001 AS u, (i * 17) %% 1000 AS x, i %% 3 AS r,
        CASE WHEN (i * 7) %% 100 < 95 THEN 0 ELSE (i * 13) %% 50000 END AS h,
        CASE WHEN (i * 11) %% 100 < 90 THEN 7 ELSE (i * 3) %% 20000 END AS g,
        TIMESTAMP '2013-07-01 00:00:00' + INTERVAL ((i * 7907) %% 86400) SECOND AS ts
        FROM range(%d) t(i)""" % N)
    _DB = Database.create(d)
    pq = os.path.join(d, 't.parquet')
    _CON.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    _DB.cat.add_table('t', [['id', 'int'], ['a', 'int'], ['b', 'int'], ['u', 'int'], ['x', 'int'], ['r', 'int'],
                            ['h', 'int'], ['g', 'int'], ['ts', 'datetime']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None: os.environ[k] = v
    _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON


def _rows(res):
    return [tuple(round(c, 6) if isinstance(c, float) else c for c in r) for r in (res[0] if isinstance(res, tuple) else res)]


def _exact(q, hunted=True):
    db, con = _fixture()
    h0 = wdb_bound._HITS
    got = _rows(db.run(q)); served = controller._SERVED[0]
    want = _rows(con.execute(q).fetchall())
    assert got == want, (q, served, got[:3], want[:3])
    if hunted is not None:
        assert (wdb_bound._HITS == h0 + 1) == hunted, (q, served, wdb_bound._HITS - h0)


def test_q18_shape():
    _exact("SELECT a, extract(minute FROM ts) AS m, b, COUNT(*) FROM t GROUP BY a, m, b "
           "ORDER BY COUNT(*) DESC, a, m, b LIMIT 10")

def test_two_keys_with_a_where_and_a_sum():
    """(pairagg, an earlier read, answers this shape today: the answer is the test)"""
    _exact("SELECT a, b, COUNT(*) AS c, SUM(x) FROM t WHERE r <> 1 GROUP BY a, b ORDER BY c DESC, a, b LIMIT 15",
           hunted=None)
    _exact("SELECT a, minute(ts) AS m, b, COUNT(*) AS c, SUM(x) FROM t WHERE r <> 1 GROUP BY a, minute(ts), b "
           "ORDER BY c DESC, a, m, b LIMIT 15")

def test_min_max_avg_beside_the_count():
    _exact("SELECT b, a, MIN(x), MAX(x), AVG(x), COUNT(*) AS c FROM t GROUP BY b, a ORDER BY c DESC, b, a LIMIT 8")

def test_flat_keys():
    """u x minute x: no key value is heavy, so no bound prunes -- every row is kept and grouped, still exact"""
    _exact("SELECT u, minute(ts) AS m, x, COUNT(*) AS c FROM t GROUP BY u, minute(ts), x "
           "ORDER BY c DESC, u, m, x LIMIT 10")

def test_not_count_ordered_is_not_hunted():
    _exact("SELECT a, b, SUM(x) AS s FROM t GROUP BY a, b ORDER BY s DESC, a, b LIMIT 10", hunted=False)

def test_the_switch():
    old = os.environ.get('WDB_BOUND')
    os.environ['WDB_BOUND'] = '0'
    try:
        _exact("SELECT a, b, COUNT(*) AS c FROM t GROUP BY a, b ORDER BY c DESC, a, b LIMIT 10", hunted=False)
    finally:
        if old is None: os.environ.pop('WDB_BOUND', None)
        else: os.environ['WDB_BOUND'] = old

def test_every_row_kept():
    """a threshold of 1 keeps every row: grouped whole, no filter -- still exact"""
    old = wdb_bound.LEAN
    wdb_bound.LEAN = 0.0
    try:
        _exact("SELECT b, minute(ts) AS m, a, COUNT(*) AS c, SUM(x) FROM t GROUP BY b, minute(ts), a "
               "ORDER BY c DESC, b, m, a LIMIT 10")
    finally:
        wdb_bound.LEAN = old

def test_dense_cells():
    """a x minute: 3.6M composite cells -- past the fused kernel's dense table, under the hunt's: one census of
    the cells, the rows revisited only for the SUM"""
    _exact("SELECT a, minute(ts) AS m, COUNT(*) AS c, SUM(x), MAX(x) FROM t WHERE r <> 2 GROUP BY a, minute(ts) "
           "ORDER BY c DESC, a, m LIMIT 12")


def test_heavy_split():
    """h holds 95% on one value, g 90%: the rows on both heavy values differ only in the minute -- counted as 60
    dense cells; the rest grouped exactly"""
    h0 = wdb_bound._HEAVY
    _exact("SELECT h, minute(ts) AS m, g, COUNT(*) AS c FROM t GROUP BY h, m, g ORDER BY c DESC, h, m, g LIMIT 10")
    _exact("SELECT h, minute(ts) AS m, a, COUNT(*) AS c FROM t GROUP BY h, m, a ORDER BY c DESC, h, m, a LIMIT 10")
    assert wdb_bound._HEAVY == h0 + 2
    _exact("SELECT g, h, COUNT(*) AS c FROM t WHERE r <> 0 GROUP BY g, h ORDER BY c DESC, g, h LIMIT 10", hunted=None)


def test_flat_scan():
    """u x minute x, ts x x: no bound prunes (the bounds keep every row) -- the flat scan answers: scatter by the
    evenest key, sort each bucket in cache, every run one group"""
    f0 = wdb_bound._FLAT
    _exact("SELECT u, minute(ts) AS m, x, COUNT(*) AS c FROM t GROUP BY u, minute(ts), x "
           "ORDER BY c DESC, u, m, x LIMIT 10")
    _exact("SELECT ts, minute(ts) AS m, x, COUNT(*) AS c FROM t GROUP BY ts, minute(ts), x "
           "ORDER BY c DESC, ts, m, x LIMIT 10")
    _exact("SELECT r, u, x, COUNT(*) AS c FROM t GROUP BY r, u, x ORDER BY c DESC, r, u, x LIMIT 25")
    assert wdb_bound._FLAT == f0 + 3, wdb_bound._FLAT - f0
    old = os.environ.get('WDB_FLAT')
    os.environ['WDB_FLAT'] = '0'
    try:
        _exact("SELECT u, minute(ts) AS m, x, COUNT(*) AS c FROM t GROUP BY u, minute(ts), x "
               "ORDER BY c DESC, u, m, x LIMIT 10")
        assert wdb_bound._FLAT == f0 + 3
    finally:
        if old is None: os.environ.pop('WDB_FLAT', None)
        else: os.environ['WDB_FLAT'] = old


def test_flat_count_kernel():
    """the kernel against np.unique, including the census cap (counts past it settled exactly)"""
    import numpy as np, wdb_kernels as WK
    rng = np.random.default_rng(7)
    n = 300_000
    a = (rng.zipf(1.3, n) % 70_000).astype(np.int64)
    r = rng.integers(0, 40, n).astype(np.uint32)
    ku, kr, offs = WK.gd_pass1(a, r, np.int64(5), np.int64(8))
    P = (a.astype(np.uint64) << np.uint64(32)) | r.astype(np.uint64)
    u, c = np.unique(P, return_counts=True)
    for k in (1, 10, 100):
        ck = np.sort(c)[::-1][k - 1]
        want = dict(zip(u[c >= ck].tolist(), c[c >= ck].tolist()))
        for cap in (1 << 16, 4, 1):
            pk, pc = WK.flat_count(ku, kr, offs, k, cap=cap)
            assert dict(zip(pk.tolist(), pc.tolist())) == want, (k, cap)
