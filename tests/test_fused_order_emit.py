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
