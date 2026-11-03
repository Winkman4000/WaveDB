"""THE EXACT WALK and THE ORDER IN ARRAYS (c120, 2026-11-03; Zoom.lean): the same-rows HAVING is cut from exact
whole-unit group sums (scale read off each factor's dictionary, width checked, the zoom over never-decreasing keys,
the cut in whole numbers) and answers like DuckDB; switched off, the float road gives the same answers. ORDER BY over
aggregates / COUNT with no LIMIT is ordered on the arrays -- the same order DuckDB gives; a key column in the ORDER BY
keeps the row sort; switched off, the same answers."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

W = "FROM lineitem, orders WHERE l_orderkey = o_orderkey AND o_orderstatus = 'F'"
EXACT = [
    ("SELECT l_orderkey, SUM(l_extendedprice * l_quantity) AS v " + W + " GROUP BY l_orderkey "
     "HAVING SUM(l_extendedprice * l_quantity) > (SELECT SUM(l_extendedprice * l_quantity) * 0.0002 " + W + ") ORDER BY v DESC", 1),
    ("SELECT l_orderkey, SUM(l_extendedprice) AS v " + W + " GROUP BY l_orderkey "
     "HAVING SUM(l_extendedprice) >= (SELECT SUM(l_extendedprice) / 2500 " + W + ") ORDER BY v DESC, l_orderkey", 1),
    ("SELECT l_orderkey, SUM(l_quantity) AS q " + W + " GROUP BY l_orderkey "
     "HAVING SUM(l_quantity) < (SELECT SUM(l_quantity) * 0.00001 " + W + ")", None),
]
ORDERQ = [
    ("SELECT o_orderpriority, SUM(l_extendedprice) AS v, COUNT(*) AS c FROM lineitem, orders "
     "WHERE l_orderkey = o_orderkey GROUP BY o_orderpriority ORDER BY v DESC", 1),
    ("SELECT l_orderkey, COUNT(*) AS c, SUM(l_quantity) AS q " + W + " GROUP BY l_orderkey ORDER BY c, q DESC", None),
]
KEYORD = ("SELECT o_orderpriority, SUM(l_quantity) AS q FROM lineitem, orders WHERE l_orderkey = o_orderkey "
          "GROUP BY o_orderpriority ORDER BY o_orderpriority")


def _check(db, con, q, vcol, ordered=False):
    import controller, test_join_fast as J
    controller.plan_epoch_bump()
    exp = J._norm([tuple(r) for r in con.execute(q).fetchall()])
    r = db.run(q)
    got = J._norm(r[0] if isinstance(r, tuple) else r)
    assert len(got) == len(exp), (q, len(got), len(exp))
    k = lambda t: tuple((x is None, str(x)) for x in t)
    for g, e in zip(sorted(got, key=k), sorted(exp, key=k)):
        assert J._eq(g, e), (q, g, e)
    if vcol is not None:
        for g, e in zip(got, exp):
            assert math.isclose(float(g[vcol]), float(e[vcol]), rel_tol=1e-9), (q, g, e)
    return got


def test_exact_walk():
    import wdb_join as WJ, test_join_fast as J
    db, con = J._fixture()
    h0 = WJ._EXACT_WALK_HITS[0]
    for q, vcol in EXACT:
        _check(db, con, q, vcol)
    assert WJ._EXACT_WALK_HITS[0] == h0 + len(EXACT), 'the exact walk was not taken'
    old = WJ._EXACT_WALK[0]
    try:
        WJ._EXACT_WALK[0] = False
        for q, vcol in EXACT:
            _check(db, con, q, vcol)
        assert WJ._EXACT_WALK_HITS[0] == h0 + len(EXACT)
    finally:
        WJ._EXACT_WALK[0] = old


def test_order_in_arrays():
    import wdb_join as WJ, test_join_fast as J
    db, con = J._fixture()
    h0 = WJ._ORDER_ARR_HITS[0]
    for q, vcol in ORDERQ:
        got = _check(db, con, q, vcol)
        if vcol is None:                        # ORDER BY c, q DESC: the rows come out in that order
            for a, b in zip(got, got[1:]):
                assert (a[1], -a[2]) <= (b[1], -b[2]), (a, b)
    assert WJ._ORDER_ARR_HITS[0] == h0 + len(ORDERQ), 'the order in arrays was not taken'
    h1 = WJ._ORDER_ARR_HITS[0]
    got = _check(db, con, KEYORD, None)         # a key column: the row sort
    assert [r[0] for r in got] == sorted(r[0] for r in got)
    assert WJ._ORDER_ARR_HITS[0] == h1
    old = WJ._ORDER_ARR[0]
    try:
        WJ._ORDER_ARR[0] = False
        for q, vcol in ORDERQ:
            _check(db, con, q, vcol)
        assert WJ._ORDER_ARR_HITS[0] == h1
    finally:
        WJ._ORDER_ARR[0] = old
