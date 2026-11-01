"""ONE GATHER PER ARRAY (2026-11-01): _at_rows' parallel gather equals numpy's a[rows] for every numeric and bool
dtype; a query naming the same parent column several times (Q12's CASE) gathers it once at the survivors, and
answers like DuckDB with the memo on and off."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np


def test_parallel_gather_equals_numpy():
    import wdb_join as WJ
    rng = np.random.default_rng(1101)
    rows = np.sort(rng.choice(200_000, 90_000, replace=False)).astype(np.int64)
    for dt in (np.uint8, np.uint16, np.int32, np.int64, np.float64, np.bool_):
        a = (rng.random(200_000) < 0.5) if dt is np.bool_ else rng.integers(0, 200, 200_000).astype(dt)
        got = WJ._at_rows(a, rows)
        assert got.dtype == a.dtype and np.array_equal(got, a[rows]), dt
    assert np.array_equal(WJ._at_rows(a, rows[:10]), a[rows[:10]])


def test_join_answers_both_ways():
    import wdb_join as WJ, controller, test_join_fast as J
    db, con = J._fixture()
    q = ("SELECT l_shipmode, SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS hi, "
         "SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS lo FROM orders, lineitem "
         "WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL', 'SHIP') AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate "
         "AND l_receiptdate >= DATE '1994-01-01' AND l_receiptdate < DATE '1995-01-01' GROUP BY l_shipmode ORDER BY l_shipmode")
    q2 = ("SELECT o.o_orderpriority, SUM(CASE WHEN c.c_mktsegment = 'BUILDING' THEN l.l_quantity ELSE 0 END), "
          "SUM(CASE WHEN c.c_mktsegment <> 'BUILDING' THEN l.l_quantity ELSE 0 END) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "JOIN customer c ON o.o_custkey = c.c_custkey WHERE l.l_quantity > 30 GROUP BY o.o_orderpriority")
    old = (WJ._RW_MEMO[0], WJ._RW_PAR_MIN)
    try:
        WJ._RW_PAR_MIN = 1                                    # the fixture is small: take the parallel gather
        h0 = WJ._RW_MEMO_HITS[0]
        for qq in (q, q2):
            for on in (True, False):
                WJ._RW_MEMO[0] = on
                controller.plan_epoch_bump()
                J._match(qq, ordered=(qq is q), expect_fast=False)
        assert WJ._RW_MEMO_HITS[0] > h0, 'a repeated parent column was gathered again'
    finally:
        WJ._RW_MEMO[0], WJ._RW_PAR_MIN = old
        controller.plan_epoch_bump()
