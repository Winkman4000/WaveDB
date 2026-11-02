"""THE PARENT KEEP WAITS FOR THE ROAD (2026-11-02): joins whose only keeps sit on fact-adjacent parents (one parent,
two parents, a grandparent above a parent) answer like DuckDB with the wait on and off, and the wait is taken."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))


def test_join_answers_both_ways():
    import wdb_join as WJ, controller, test_join_fast as J
    db, con = J._fixture()
    qs = ["SELECT o.o_orderpriority, SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "WHERE o.o_orderpriority = '1-URGENT' GROUP BY o.o_orderpriority",
          "SELECT o.o_orderstatus, COUNT(*), SUM(l.l_quantity) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "WHERE o.o_orderstatus = 'P' GROUP BY o.o_orderstatus",
          "SELECT n.n_name, COUNT(*), SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey "
          "WHERE n.n_name = 'PERU' GROUP BY n.n_name",
          "SELECT c.c_mktsegment, COUNT(*) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "JOIN customer c ON o.o_custkey = c.c_custkey WHERE c.c_mktsegment = 'BUILDING' AND o.o_orderpriority = '2-HIGH' "
          "GROUP BY c.c_mktsegment"]
    old = WJ._PARENT_WAIT[0]
    try:
        h0 = WJ._PARENT_WAIT_HITS[0]
        for q in qs:
            for on in (True, False):
                WJ._PARENT_WAIT[0] = on
                controller.plan_epoch_bump()
                J._match(q, expect_fast=False)
        assert WJ._PARENT_WAIT_HITS[0] > h0, 'the parent keep never waited'
    finally:
        WJ._PARENT_WAIT[0] = old
        controller.plan_epoch_bump()
