"""THE WITH READ TWICE, RUN ONCE (2026-11-02): Q15's shape -- a CTE joined to its base table and filtered to its own
MAX (or MIN) -- answers like DuckDB, with an extra base condition and a projected expression too, and the once-run
is taken; switched off it fails as before (no such table); a CTE used another way is not touched."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

Q = ("WITH rev AS (SELECT o_custkey AS cust_no, SUM(o_totalprice) AS total FROM orders WHERE o_orderdate >= DATE '1995-01-01' "
     "AND o_orderdate < DATE '1995-04-01' GROUP BY o_custkey) "
     "SELECT c_custkey, c_name, c_phone, total FROM customer, rev WHERE c_custkey = cust_no AND total = (SELECT %s(total) FROM rev) "
     "ORDER BY c_custkey")


def _rows(db, q):
    r = db.run(q)
    return r[0] if isinstance(r, tuple) else r


def test_answers():
    import wdb_cte as C, controller, test_join_fast as J
    db, con = J._fixture()
    h0 = C._CTE_ONCE_HITS[0]
    qs = [Q % 'MAX', Q % 'MIN',
          (Q % 'MAX').replace('ORDER BY c_custkey', "AND c_mktsegment <> 'NONE' ORDER BY c_custkey"),
          (Q % 'MAX').replace('c_phone, total', 'c_phone, total * 2 AS t2')]
    for q in qs:
        controller.plan_epoch_bump()
        exp = J._norm([tuple(r) for r in con.execute(q).fetchall()])
        got = J._norm(_rows(db, q))
        assert exp and len(got) == len(exp), (q, got, exp)
        for g, e in zip(got, exp):
            assert J._eq(g, e), (q, g, e)
    assert C._CTE_ONCE_HITS[0] >= h0 + len(qs), 'the once-run was not taken'


def test_switch_and_declines():
    import sqlglot, wdb_cte as C, test_join_fast as J
    db, con = J._fixture()
    old = C._CTE_ONCE[0]
    try:
        C._CTE_ONCE[0] = False
        assert C.once_rewrite(db, sqlglot.parse_one(Q % 'MAX', read='duckdb')) is None
    finally:
        C._CTE_ONCE[0] = old
    other = ("WITH rev AS (SELECT o_custkey AS cust_no, SUM(o_totalprice) AS total FROM orders GROUP BY o_custkey) "
             "SELECT c_custkey, total FROM customer, rev WHERE c_custkey = cust_no AND total > 1000 ORDER BY c_custkey")
    assert C.once_rewrite(db, sqlglot.parse_one(other, read='duckdb')) is None          # no extreme: not this shape
