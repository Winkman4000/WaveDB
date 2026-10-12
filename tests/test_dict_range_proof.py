"""THE DICTIONARY-RANGE PROOF: a WHERE conjunct no stored value can satisfy (IsLink < 0 on a 0/1 column) makes a
grouped query empty, proven from each column's sorted dictionary ends before any row is touched -- the board's
Q36 variant grouped 100M rows for 9.4 s, then filtered every one out (2026-10-12). Every answer against DuckDB;
the boundary cases (flag <= 0, flag < 1, an OR with one impossible side) must NOT be proven empty."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, wdb_join, controller
from wdb_db import Database

_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'drp_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    _CON.execute("""CREATE TABLE t AS SELECT i AS id, i % 2 AS flag, (i * 7) % 100000 AS big, (i % 50) / 4.0 AS f,
        DATE '2013-07-01' + (i % 30)::INT AS dd, CASE WHEN i % 10 = 0 THEN NULL ELSE i % 7 END AS nn,
        (i * 3) % 11 AS g FROM range(300000) t(i)""")
    _DB = Database.create(d)
    pq = os.path.join(d, 't.parquet')
    _CON.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    _DB.cat.add_table('t', [['id', 'int'], ['flag', 'int'], ['big', 'int'], ['f', 'float'], ['dd', 'date'], ['nn', 'int'], ['g', 'int']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None: os.environ[k] = v
    _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON


def _rows(res):
    return sorted(tuple(r) for r in (res[0] if isinstance(res, tuple) else res))


def _check(where, proven):
    db, con = _fixture()
    q = "SELECT g, COUNT(*) AS c, SUM(big) FROM t WHERE %s GROUP BY g ORDER BY c DESC, g" % where
    h0 = wdb_join._VETO_HITS
    got = _rows(db.run(q))
    want = sorted(tuple(r) for r in con.execute(q).fetchall())
    assert got == want, (where, got[:3], want[:3])
    if controller._SERVED[0] != 'fused_agg' and proven:
        return                                   # an earlier read answered it (an equality: wherescan) -- right either way
    assert (wdb_join._VETO_HITS == h0 + 1) == proven, ('proof fired' if not proven else 'proof missed', where)


def test_impossible_comparisons():
    for w in ("flag < 0", "flag > 1", "flag >= 2", "flag <= -1", "big > 100000", "f < 0", "f > 12.25",
              "0 > flag", "flag = 5", "dd < DATE '2013-06-01'", "dd > '2013-08-15'"):
        _check(w, True)

def test_impossible_in_and_between():
    _check("flag IN (-1, 2, 7)", True)
    _check("big BETWEEN 200000 AND 300000", True)
    _check("g BETWEEN 5 AND 3", True)

def test_one_impossible_conjunct_empties_the_and():
    _check("g = 3 AND (big > 5 AND flag < 0)", True)

def test_nullable_column_still_proven():
    _check("nn > 6", True)

def test_boundaries_are_not_proven():
    for w in ("flag <= 0", "flag < 1", "flag >= 1", "big >= 99999", "f >= 12.25", "flag IN (-1, 1)",
              "dd >= '2013-07-30'", "nn <> 3", "big BETWEEN 99999 AND 300000"):
        _check(w, False)

def test_or_with_an_impossible_side_is_not_proven():
    _check("flag < 0 OR g = 2", False)

def test_ungrouped_aggregate_is_left_alone():
    db, con = _fixture()
    q = "SELECT COUNT(*), SUM(big) FROM t WHERE flag < 0"
    h0 = wdb_join._VETO_HITS
    assert _rows(db.run(q)) == sorted(tuple(r) for r in con.execute(q).fetchall())
    assert wdb_join._VETO_HITS == h0
