"""THE OFFSET WINDOW: fused_agg answers LIMIT + OFFSET rows in ORDER BY order and drops the first OFFSET of
them (wdb_join.table_agg), so an OFFSET query no longer skips the composer for the general scan -- the funnel
family's board variants fell 0.7-1.6 s there where DuckDB took 0.1 s (2026-10-12). Checked against DuckDB:
exactly where the order is total, and as SQL defines it where ties reach the window (real groups whose order
values equal DuckDB's at the same positions)."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, controller
from wdb_db import Database

_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'foff_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    # site: 40 values (a selective equality); a x b: ~1000 x 7 groups; u: near-unique (the peel's road)
    _CON.execute("""CREATE TABLE t AS SELECT i AS id, i % 40 AS site, (i * 7) % 1000 AS a, (i * 3) % 7 AS b,
        CASE WHEN i % 97 = 0 AND i > 0 THEN (i - 1) * 5 ELSE i * 5 END AS u,
        i % 2 AS r, (i * 31) % 500 AS x FROM range(400000) t(i)""")
    _DB = Database.create(d)
    pq = os.path.join(d, 't.parquet')
    _CON.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    _DB.cat.add_table('t', [[c, 'int'] for c in ('id', 'site', 'a', 'b', 'u', 'r', 'x')])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None: os.environ[k] = v
    _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON


def _rows(res):
    return [tuple(r) for r in (res[0] if isinstance(res, tuple) else res)]


def _exact(q, fused=True):
    db, con = _fixture()
    got = _rows(db.run(q)); served = controller._SERVED[0]
    want = [tuple(r) for r in con.execute(q).fetchall()]
    assert got == want, (q, served, got[:4], want[:4])
    if fused:
        assert served == 'fused_agg', (q, served)


def _tied(q, order_cols, keys, fused=True):
    """ties may reach the window: real groups, distinct, DuckDB's order values at the same positions"""
    db, con = _fixture()
    got = _rows(db.run(q)); served = controller._SERVED[0]
    want = [tuple(r) for r in con.execute(q).fetchall()]
    full = {tuple(r[i] for i in keys): tuple(r) for r in con.execute(q.split(' ORDER BY ')[0].split(' LIMIT ')[0].split(' OFFSET ')[0]).fetchall()}
    assert len(got) == len(want), (q, len(got), len(want))
    assert len({tuple(r[i] for i in keys) for r in got}) == len(got), ('duplicate groups', q)
    assert all(full.get(tuple(r[i] for i in keys)) == r for r in got), ('not a real group row', q, got[:3])
    if order_cols is not None:
        assert [tuple(r[i] for i in order_cols) for r in got] == [tuple(r[i] for i in order_cols) for r in want], (q, got[:4], want[:4])
    if fused:
        assert served == 'fused_agg', (q, served)


def test_offset_total_order():
    _exact("SELECT a, b, COUNT(*) AS c, SUM(x) FROM t WHERE site + r = 7 GROUP BY a, b ORDER BY c DESC, a, b LIMIT 10 OFFSET 100")

def test_offset_ties_in_the_window():
    _tied("SELECT a, b, COUNT(*) AS c FROM t WHERE site + r = 7 GROUP BY a, b ORDER BY c DESC LIMIT 10 OFFSET 100", [2], (0, 1))

def test_offset_ascending_sum_with_key_tiebreak():
    _exact("SELECT a, SUM(x) AS s, COUNT(*) FROM t WHERE site <> 3 GROUP BY a ORDER BY s, a LIMIT 7 OFFSET 30")

def test_offset_ordinal_and_expression_terms():
    _exact("SELECT a, b, MAX(x), COUNT(*) FROM t GROUP BY a, b ORDER BY 3 DESC, COUNT(*), a, b LIMIT 5 OFFSET 40")

def test_offset_without_order():
    _tied("SELECT a, b, COUNT(*) AS c FROM t WHERE site + r = 1 GROUP BY a, b LIMIT 10 OFFSET 20", None, (0, 1))

def test_offset_without_limit():
    _exact("SELECT a, COUNT(*) AS c FROM t WHERE site + r = 2 GROUP BY a ORDER BY c DESC, a OFFSET 990")

def test_offset_past_the_end():
    _exact("SELECT a, COUNT(*) AS c FROM t WHERE site + r = 2 GROUP BY a ORDER BY c DESC, a LIMIT 10 OFFSET 5000")

def test_offset_with_having():
    _exact("SELECT a, COUNT(*) AS c FROM t GROUP BY a HAVING COUNT(*) > 399 ORDER BY c DESC, a LIMIT 10 OFFSET 15")

def test_offset_through_the_peel():
    """a near-unique key: the singleton peel answers LIMIT + OFFSET, the window slices"""
    _exact("SELECT u, COUNT(*) AS c, SUM(x) FROM t WHERE r + site > 3 GROUP BY u ORDER BY c DESC, u LIMIT 10 OFFSET 3")
    _tied("SELECT u, b, COUNT(*) AS c FROM t GROUP BY u, b ORDER BY c ASC LIMIT 10 OFFSET 1000", [2], (0, 1))

def test_offset_through_the_early_pick():
    _tied("SELECT a, b, COUNT(*) AS c, SUM(x) FROM t GROUP BY a, b ORDER BY c ASC LIMIT 10 OFFSET 2000", [2], (0, 1))

def test_offset_equality_where_still_served():
    """an equality WHERE: wherescan applies OFFSET itself and answers first -- the answer is the test"""
    _exact("SELECT a, b, COUNT(*) AS c, SUM(x) FROM t WHERE site = 7 AND r = 0 GROUP BY a, b ORDER BY c DESC, a, b LIMIT 10 OFFSET 100", fused=False)
