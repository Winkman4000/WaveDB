"""THE JOIN'S OWN SCALARS (2026-11-02): join queries with an uncorrelated scalar subquery -- Q11's shape (HAVING over
the same joins times a fraction), a WHERE comparison, the subquery on the left, a subquery with no row -- answer like
DuckDB, and the substitution is taken; a correlated subquery (Q17's) is left to the value per key; switched off the
join query fails as before."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

LO = "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
QS = [
    "SELECT o.o_orderpriority, SUM(l.l_extendedprice * l.l_quantity) AS v " + LO + "WHERE o.o_orderstatus = 'F' GROUP BY o.o_orderpriority "
    "HAVING SUM(l.l_extendedprice * l.l_quantity) > (SELECT SUM(l_extendedprice * l_quantity) * 0.19 FROM lineitem, orders "
    "WHERE l_orderkey = o_orderkey AND o_orderstatus = 'F') ORDER BY v DESC",
    "SELECT o.o_orderpriority, COUNT(*) " + LO + "WHERE o.o_totalprice > (SELECT AVG(o_totalprice) FROM orders) GROUP BY o.o_orderpriority",
    "SELECT o.o_orderstatus, COUNT(*) " + LO + "WHERE (SELECT AVG(l_quantity) FROM lineitem) < l.l_quantity GROUP BY o.o_orderstatus",
    "SELECT o.o_orderstatus, COUNT(*) " + LO + "WHERE l.l_quantity > (SELECT MAX(l_quantity) FROM lineitem WHERE l_quantity > 1000) "
    "GROUP BY o.o_orderstatus",
]


def test_answers():
    import wdb_subquery as S, controller, test_join_fast as J
    db, con = J._fixture()
    h0 = S._JOIN_SCALARS_HITS[0]
    for q in QS:
        controller.plan_epoch_bump()
        exp = J._norm([tuple(r) for r in con.execute(q).fetchall()])
        r = db.run(q)
        got = J._norm(r[0] if isinstance(r, tuple) else r)
        k = lambda t: tuple((x is None, str(x)) for x in t)
        assert len(got) == len(exp), (q, got, exp)
        for g, e in zip(sorted(got, key=k), sorted(exp, key=k)):
            assert J._eq(g, e), (q, g, e)
    assert S._JOIN_SCALARS_HITS[0] >= h0 + len(QS) - 1, 'the substitution was not taken'


def test_correlated_left_and_switch():
    import sqlglot, wdb_subquery as S, test_join_fast as J
    db, con = J._fixture()
    corr = ("SELECT o.o_orderpriority, COUNT(*) " + LO + "WHERE l.l_quantity < (SELECT AVG(l_quantity) FROM lineitem "
            "WHERE l_orderkey = o.o_orderkey) GROUP BY o.o_orderpriority")
    assert S.join_scalars(db, sqlglot.parse_one(corr, read='duckdb')) is None
    old = S._JOIN_SCALARS[0]
    try:
        S._JOIN_SCALARS[0] = False
        assert S.join_scalars(db, sqlglot.parse_one(QS[1], read='duckdb')) is None
    finally:
        S._JOIN_SCALARS[0] = old
