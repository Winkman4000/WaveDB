"""The fused engine's emit picks exactly k groups in array space whenever SQL lets it:
ORDER BY any projected term -- alias, bare column, the aggregate expression itself
(COUNT(*), MAX(x)), or an ordinal -- and LIMIT with no ORDER BY at all. Matching ORDER BY
terms by .name once missed every aggregate term (COUNT(*) is named '*', MAX(x) ''), so
tie-heavy orders (ORDER BY COUNT(*) ASC LIMIT 1) sent millions of groups to a Python sort:
the shifting board's 16-22 s cliffs (2026-10-12). Answers are checked against DuckDB as
SQL defines them: with ties at the cut, any tied groups are legal, so a result must be
k real groups whose order values equal DuckDB's top-k order values."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, wdb_join
from wdb_db import Database

_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'foe_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    # a x b: 97 x 13 pairs, ~40 rows each -> heavy ties on COUNT(*) and MAX
    _CON.execute("CREATE TABLE t AS SELECT i AS id, i % 97 AS a, (i * 7) % 13 AS b, "
                 "((i * 31) % 50) * 1.5 AS amt FROM range(50000) t(i)")
    _DB = Database.create(d)
    desc = _CON.execute("DESCRIBE t").fetchall()
    wt = {'BIGINT': 'int', 'INTEGER': 'int', 'DOUBLE': 'float'}
    sel = ", ".join(("CAST(%s AS DOUBLE) AS %s" % (c[0], c[0])) if c[1].startswith('DECIMAL') else c[0] for c in desc)
    pq = os.path.join(d, 't.parquet')
    _CON.execute("COPY (SELECT %s FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % (sel, pq))
    _DB.cat.add_table('t', [[c[0], 'float' if c[1].startswith('DECIMAL') else wt[c[1]]] for c in desc])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON

def _n(v):
    return round(float(v), 6) if isinstance(v, float) else v

def _check(q, order_cols, k, keys=(0, 1)):
    """k real groups, the right order values, and the emit picked them in array space"""
    db, con = _fixture()
    f0, e0 = wdb_join._FAST_HITS, wdb_join._TOPK_EMIT_HITS
    got = [tuple(_n(c) for c in r) for r in db.run(q)[0]]
    assert wdb_join._FAST_HITS == f0 + 1, 'fused path not taken: ' + q
    assert wdb_join._TOPK_EMIT_HITS == e0 + 1, 'emit did not pick k in array space: ' + q
    full = q.split(' ORDER BY ')[0].split(' LIMIT ')[0]
    allrows = {tuple(r[i] for i in keys): tuple(_n(c) for c in r) for r in con.execute(full).fetchall()}
    assert len(got) == k, (q, got)
    assert len({tuple(r[i] for i in keys) for r in got}) == k, ('duplicate groups', q, got)
    for r in got:
        assert allrows.get(tuple(r[i] for i in keys)) == r, ('not a real group row', q, r)
    if order_cols is not None:
        want = [tuple(_n(r[i]) for i in order_cols) for r in con.execute(q).fetchall()]
        assert [tuple(r[i] for i in order_cols) for r in got] == want, (q, got, want)

def test_order_by_count_star_asc_ties():
    _check("SELECT a, b, COUNT(*), SUM(amt) FROM t GROUP BY a, b ORDER BY COUNT(*) ASC LIMIT 1", [2], 1)

def test_order_by_count_star_desc():
    _check("SELECT a, b, COUNT(*), SUM(amt) FROM t GROUP BY a, b ORDER BY COUNT(*) DESC LIMIT 5", [2], 5)

def test_order_by_max_unaliased():
    _check("SELECT a, b, MAX(amt) FROM t GROUP BY a, b ORDER BY MAX(amt) DESC LIMIT 4", [2], 4)

def test_order_by_ordinal():
    _check("SELECT a, b, SUM(amt) FROM t GROUP BY a, b ORDER BY 3 DESC LIMIT 3", [2], 3)

def test_order_by_count_then_keys_exact():
    _check("SELECT a, b, COUNT(*), SUM(amt) FROM t GROUP BY a, b ORDER BY COUNT(*) DESC, a, b LIMIT 6", [0, 1, 2], 6)

def test_limit_without_order():
    _check("SELECT a, b, COUNT(*), SUM(amt) FROM t GROUP BY a, b LIMIT 10", None, 10)


# ---- THE EARLY PICK: the k rows are chosen before keys are decoded and the governor is asked ----
_U = None
def _unique_fixture():
    """a near-unique key: every id its own group, so ORDER BY COUNT(*) ASC ties all of them at 1"""
    global _U
    if _U is not None:
        return _U
    con = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'foeu_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    con.execute("CREATE TABLE t AS SELECT i AS id, (i * 7919) % 1000003 AS u, i % 5 AS k, "
                "((i * 31) % 50) * 1.5 AS amt FROM range(300000) t(i)")
    db = Database.create(d)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db.cat.add_table('t', [['id', 'int'], ['u', 'int'], ['k', 'int'], ['amt', 'float']])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb')
    _U = (db, con)
    return _U


def test_early_pick_under_a_small_governor_budget():
    """300K tied groups as Python rows would exceed a 1 MB budget; the 5 rows asked for do not.
    Before the early pick the governor was asked about every tied group and refused."""
    db, con = _unique_fixture()
    q = "SELECT u, k, COUNT(*) AS c, SUM(amt) FROM t GROUP BY u, k ORDER BY c ASC LIMIT 5"
    old = os.environ.get('WDB_WORK_MB')
    os.environ['WDB_WORK_MB'] = '1'
    try:
        e0 = wdb_join._TOPK_EMIT_HITS
        got = [tuple(_n(c) for c in r) for r in db.run(q)[0]]
        assert wdb_join._TOPK_EMIT_HITS == e0 + 1
    finally:
        if old is None: os.environ.pop('WDB_WORK_MB', None)
        else: os.environ['WDB_WORK_MB'] = old
    full = {(r[0], r[1]): tuple(_n(c) for c in r) for r in
            con.execute("SELECT u, k, COUNT(*) AS c, SUM(amt) FROM t GROUP BY u, k").fetchall()}
    assert len(got) == 5 and len({(r[0], r[1]) for r in got}) == 5
    assert all(full[(r[0], r[1])] == r and r[2] == 1 for r in got), got


def test_early_pick_with_a_key_tiebreak_is_exact():
    """a key in the ORDER BY leaves the pick to the late road (it needs decoded keys): exact order"""
    db, con = _unique_fixture()
    q = "SELECT u, k, COUNT(*) AS c FROM t GROUP BY u, k ORDER BY c ASC, u, k LIMIT 7"
    got = [tuple(r) for r in db.run(q)[0]]
    want = [tuple(r) for r in con.execute(q).fetchall()]
    assert got == want, (got, want)
