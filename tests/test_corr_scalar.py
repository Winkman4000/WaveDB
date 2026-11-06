"""THE VALUE PER KEY (2026-11-02): WHERE comparisons against a correlated scalar subquery -- AVG with and without a
constant factor, SUM, COUNT(*), MIN, MAX; the subquery on either side; the outer key a parent (orders) and a
grandparent (customer); grouped and scalar (Q17's SUM / 7.0) -- answer like DuckDB with the value per key on and
off (off: the old decline), and the value per key is taken."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

LO = "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
QS = [
    "SELECT o.o_orderpriority, COUNT(*), SUM(l.l_extendedprice) " + LO + "WHERE o.o_orderstatus = 'F' AND "
    "l.l_quantity < (SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_orderkey = o.o_orderkey) GROUP BY o.o_orderpriority",
    "SELECT o.o_orderpriority, COUNT(*) " + LO + "WHERE l.l_quantity < (SELECT AVG(l_quantity) FROM lineitem WHERE l_orderkey = o.o_orderkey) "
    "GROUP BY o.o_orderpriority",
    "SELECT o.o_orderstatus, COUNT(*) " + LO + "WHERE l.l_extendedprice * 3 > (SELECT SUM(l_extendedprice) FROM lineitem WHERE l_orderkey = o.o_orderkey) "
    "GROUP BY o.o_orderstatus",
    "SELECT o.o_orderstatus, COUNT(*) " + LO + "WHERE l.l_linenumber >= (SELECT COUNT(*) FROM lineitem WHERE l_orderkey = o.o_orderkey) "
    "GROUP BY o.o_orderstatus",
    "SELECT l.l_returnflag, COUNT(*) " + LO + "WHERE l.l_quantity = (SELECT MAX(l_quantity) FROM lineitem WHERE l_orderkey = o.o_orderkey) "
    "GROUP BY l.l_returnflag",
    "SELECT l.l_returnflag, COUNT(*) " + LO + "WHERE (SELECT MIN(l_quantity) FROM lineitem WHERE o.o_orderkey = l_orderkey) >= l.l_quantity "
    "GROUP BY l.l_returnflag",
    "SELECT c.c_mktsegment, COUNT(*), SUM(l.l_quantity) " + LO + "JOIN customer c ON o.o_custkey = c.c_custkey "
    "WHERE l.l_shipmode = 'AIR' AND o.o_totalprice > (SELECT AVG(o_totalprice) FROM orders WHERE o_custkey = c.c_custkey) GROUP BY c.c_mktsegment",
]
Q17 = ("SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly FROM lineitem, orders WHERE o_orderkey = l_orderkey AND o_orderpriority = '1-URGENT' "
       "AND l_quantity < (SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_orderkey = o_orderkey)")


def test_answers_both_ways():
    import wdb_join as WJ, controller, test_join_fast as J
    db, con = J._fixture()
    old = WJ._CORR_SCALAR[0]
    try:
        h0 = WJ._CORR_SCALAR_HITS[0]
        WJ._CORR_SCALAR[0] = True
        for q in QS:
            J._match(q, expect_fast=False)
        exp = con.execute(Q17).fetchall()[0][0]
        r = db.run(Q17)
        got = r[0][0][0] if isinstance(r, tuple) else (r[0][0] if isinstance(r, list) else r)
        assert abs(float(got) - float(exp)) <= 1e-6 * max(1.0, abs(float(exp))), (got, exp)
        assert WJ._CORR_SCALAR_HITS[0] >= h0 + len(QS) + 1, 'the value per key was not taken every time'
        WJ._CORR_SCALAR[0] = False                            # off: the old decline (no answer from the fast path)
        h1 = WJ._CORR_SCALAR_HITS[0]
        try:
            db.run(QS[0])
        except Exception:
            pass
        assert WJ._CORR_SCALAR_HITS[0] == h1
    finally:
        WJ._CORR_SCALAR[0] = old


def test_spec_declines():
    import sqlglot, wdb_join as WJ, test_join_fast as J
    db, con = J._fixture()
    def spec(sql):
        return WJ._corr_scalar_spec(db, sqlglot.parse_one(sql, read='duckdb'))
    assert spec("SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_orderkey = o_orderkey") is not None
    assert spec("SELECT AVG(l_quantity) FROM lineitem WHERE l_orderkey = o_orderkey AND l_quantity > 3") is None   # its own filter
    assert spec("SELECT AVG(l_quantity) FROM lineitem, orders WHERE l_orderkey = o_orderkey") is None             # two tables
    assert spec("SELECT AVG(l_quantity) + MAX(l_quantity) FROM lineitem WHERE l_orderkey = o_orderkey") is None   # two aggregates
    assert spec("SELECT COUNT(DISTINCT l_partkey) FROM lineitem WHERE l_orderkey = o_orderkey") is None
    assert spec("SELECT AVG(l_quantity) FROM lineitem WHERE l_orderkey = l_partkey") is None                      # not correlated
