"""THE LEFTOVER IN FIRST (2026-11-01): after the plane handoff a leftover 'fact text column IN (literals)' keeps the
survivors by code before the per-row leftovers -- Q12's shape (a receipt-year band, a ship-mode IN, two date-vs-date
comparisons) answers like DuckDB with it on and off, including a literal the dictionary lacks, and the keep is taken."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))


def test_join_answers_both_ways():
    import wdb_join as WJ, controller, test_join_fast as J
    db, con = J._fixture()
    q12 = ("SELECT l_shipmode, SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS hi, "
           "SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS lo FROM orders, lineitem "
           "WHERE o_orderkey = l_orderkey AND l_shipmode IN (%s) AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate "
           "AND l_receiptdate >= DATE '%d-01-01' AND l_receiptdate < DATE '%d-01-01' GROUP BY l_shipmode ORDER BY l_shipmode")
    qs = [q12 % ("'MAIL', 'SHIP'", 1994, 1995), q12 % ("'AIR', 'NOT A MODE'", 1996, 1997), q12 % ("'TRUCK'", 1993, 1994)]
    old = WJ._LEFT_FIRST[0]
    try:
        h0 = WJ._LEFT_FIRST_HITS[0]
        for q in qs:
            for on in (True, False):
                WJ._LEFT_FIRST[0] = on
                J._match(q, ordered=True, expect_fast=False)
        assert WJ._LEFT_FIRST_HITS[0] > h0, 'the leftover IN never kept the survivors first'
    finally:
        WJ._LEFT_FIRST[0] = old
