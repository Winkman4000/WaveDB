"""THE SAME ROWS, ONE WALK (c119, 2026-11-03, Q11): a join GROUP BY whose HAVING compares its SUM against a scalar
subquery summing the same expression over the same tables and conditions answers like DuckDB from one walk -- with
aliases and reordered conditions, a parent group key, the subquery on the left, ORDER BY + LIMIT, nothing passing;
a subquery over different rows is left to the join's own scalars; switched off, the same answers."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

W = "FROM lineitem, orders WHERE l_orderkey = o_orderkey AND o_orderstatus = 'F'"
QS = [
    ("SELECT l_orderkey, SUM(l_extendedprice * l_quantity) AS v " + W + " GROUP BY l_orderkey "
     "HAVING SUM(l_extendedprice * l_quantity) > (SELECT SUM(l_extendedprice * l_quantity) * 0.0002 " + W + ") ORDER BY v DESC", 1),
    ("SELECT l.l_orderkey, SUM(l.l_extendedprice * l.l_quantity) AS v FROM lineitem l, orders o "
     "WHERE l.l_orderkey = o.o_orderkey AND o.o_orderstatus = 'F' GROUP BY l.l_orderkey "
     "HAVING SUM(l.l_extendedprice * l.l_quantity) > (SELECT SUM(l_extendedprice * l_quantity) * 0.0002 FROM orders, lineitem "
     "WHERE o_orderstatus = 'F' AND o_orderkey = l_orderkey) ORDER BY v DESC, l.l_orderkey", 1),
    ("SELECT o_orderpriority, SUM(l_quantity) AS q, COUNT(*) AS c FROM lineitem, orders WHERE l_orderkey = o_orderkey "
     "GROUP BY o_orderpriority HAVING SUM(l_quantity) >= (SELECT SUM(l_quantity) / 6 FROM lineitem, orders "
     "WHERE l_orderkey = o_orderkey) ORDER BY q DESC", 1),
    ("SELECT l_orderkey, SUM(l_extendedprice) AS v " + W + " GROUP BY l_orderkey "
     "HAVING (SELECT SUM(l_extendedprice) * 0.0003 " + W + ") < SUM(l_extendedprice) ORDER BY v DESC LIMIT 5", 1),
    ("SELECT l_orderkey, SUM(l_extendedprice) AS v " + W + " GROUP BY l_orderkey "
     "HAVING SUM(l_extendedprice) > (SELECT SUM(l_extendedprice) * 10 " + W + ")", None),
]
OTHER = ("SELECT l_orderkey, SUM(l_extendedprice) AS v " + W + " GROUP BY l_orderkey "
         "HAVING SUM(l_extendedprice) > (SELECT SUM(l_extendedprice) * 0.0003 FROM lineitem "
         "WHERE l_quantity > 10) ORDER BY v DESC")


def _check(db, con, q, vcol):
    import controller, test_join_fast as J
    controller.plan_epoch_bump()
    exp = J._norm([tuple(r) for r in con.execute(q).fetchall()])
    r = db.run(q)
    got = J._norm(r[0] if isinstance(r, tuple) else r)
    assert len(got) == len(exp), (q, len(got), len(exp))
    k = lambda t: tuple((x is None, str(x)) for x in t)
    for g, e in zip(sorted(got, key=k), sorted(exp, key=k)):
        assert J._eq(g, e), (q, g, e)
    if vcol is not None:                        # ORDER BY the sum: the same values in the same order
        for g, e in zip(got, exp):
            assert math.isclose(float(g[vcol]), float(e[vcol]), rel_tol=1e-9), (q, g, e)


def test_same_rows():
    import wdb_subquery as S, test_join_fast as J
    db, con = J._fixture()
    h0 = S._SAME_ROWS_HITS[0]
    for q, vcol in QS:
        _check(db, con, q, vcol)
    assert S._SAME_ROWS_HITS[0] == h0 + len(QS), 'the one walk was not taken'


def test_other_rows_and_switch():
    import wdb_subquery as S, test_join_fast as J
    db, con = J._fixture()
    h0 = S._SAME_ROWS_HITS[0]; j0 = S._JOIN_SCALARS_HITS[0]
    _check(db, con, OTHER, 1)                   # different rows: the join's own scalars answer it
    assert S._SAME_ROWS_HITS[0] == h0 and S._JOIN_SCALARS_HITS[0] == j0 + 1
    old = S._SAME_ROWS[0]
    try:
        S._SAME_ROWS[0] = False
        # the old road (the join's own scalars) answers these three; QS[2] (SUM / 6 into a parent-key group: compose
        # Nullif) and QS[3] (the subquery on the left: the row HAVING reads only a column there) are its known gaps
        for q, vcol in (QS[0], QS[1], QS[4]):
            _check(db, con, q, vcol)
        assert S._SAME_ROWS_HITS[0] == h0
    finally:
        S._SAME_ROWS[0] = old
