"""wherescan's count-desc emission breaks ties on its group keys ascending IN KEY ORDER (the composite).
A trailing ORDER BY key list is honored only when it is a prefix of that order: ORDER BY c DESC, w over
keys (a, w) broke the count ties on a instead and cut the wrong rows at the LIMIT -- the shifting board's
G on hits (GROUP BY ClientIP, WatchID ... ORDER BY c DESC, WatchID), 2026-10-12. Checked against DuckDB."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, controller
from wdb_db import Database

_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'wstb_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    # a: 1000 values; w: distinct except every 101st row repeats its neighbour (real count-2 groups)
    _CON.execute("""CREATE TABLE t AS SELECT i AS id, i % 17 AS region, (i * 7) % 1000 AS a,
        CASE WHEN i % 101 = 0 AND i > 0 THEN ((i - 1) * 2654435761) % 4294967291
             ELSE (i * 2654435761) % 4294967291 END AS w,
        i % 2 AS r FROM range(300000) t(i)""")
    _DB = Database.create(d)
    pq = os.path.join(d, 't.parquet')
    _CON.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    _DB.cat.add_table('t', [[c, 'int'] for c in ('id', 'region', 'a', 'w', 'r')])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None: os.environ[k] = v
    _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON


def _exact(q):
    db, con = _fixture()
    got = [tuple(r) for r in db.run(q)[0]]
    served = controller._SERVED[0]
    want = [tuple(r) for r in con.execute(q).fetchall()]
    assert got == want, (q, served, got, want)
    return served


def test_tiebreak_on_the_second_key_alone():
    _exact("SELECT a, w, COUNT(*) AS c, SUM(r) FROM t WHERE region = 3 GROUP BY a, w ORDER BY c DESC, w LIMIT 10")

def test_tiebreak_keys_out_of_key_order():
    _exact("SELECT a, w, COUNT(*) AS c FROM t WHERE region = 3 GROUP BY a, w ORDER BY c DESC, w, a LIMIT 10")

def test_tiebreak_in_key_order_still_served_here():
    s = _exact("SELECT a, w, COUNT(*) AS c, SUM(r) FROM t WHERE region = 3 GROUP BY a, w ORDER BY c DESC, a, w LIMIT 10")
    assert s == 'wherescan', s

def test_tiebreak_prefix_of_key_order():
    """ORDER BY c DESC, a: ties past a are any order -- check the order values only"""
    db, con = _fixture()
    q = "SELECT a, w, COUNT(*) AS c FROM t WHERE region = 3 GROUP BY a, w ORDER BY c DESC, a LIMIT 10"
    got = [tuple(r) for r in db.run(q)[0]]
    full = {(r[0], r[1]): r for r in con.execute(q.split(' ORDER BY ')[0]).fetchall()}
    want = [(r[2], r[0]) for r in con.execute(q).fetchall()]
    assert [(r[2], r[0]) for r in got] == want, (got, want)
    assert all(full[(r[0], r[1])] == r for r in got)
