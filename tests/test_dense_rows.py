"""THE DENSE SPACE PAYS BY THE ROWS (2026-11-01): RT.dense_multigroup_pays keeps the dense composite space under the
floor and while it is under half the rows, and always when switched off. A joined GROUP BY whose composite space
far outgrows its rows answers like DuckDB with the line on and off, and the factorise is taken."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))


def test_line():
    import wdb_measure_runtime as RT
    old = RT._DENSE_ROWS[0]
    try:
        RT._DENSE_ROWS[0] = True
        assert RT.dense_multigroup_pays(1000, 10)                      # under the floor: dense
        assert RT.dense_multigroup_pays(RT.DENSE_SPACE_FLOOR - 1, 1)
        assert not RT.dense_multigroup_pays(1_578_750, 1_463_770)       # TPC-H Q7's group step
        assert RT.dense_multigroup_pays(1_578_750, 10_000_000)
        assert RT.dense_multigroup_pays(100_000, 200_000) and not RT.dense_multigroup_pays(100_001, 200_000)
        assert not RT.dense_multigroup_pays(100_000, 0)
        RT._DENSE_ROWS[0] = False
        assert RT.dense_multigroup_pays(1_578_750, 1)
    finally:
        RT._DENSE_ROWS[0] = old


def test_join_answers_both_ways():
    import wdb_join as WJ, wdb_measure_runtime as RT, controller, test_join_fast as J
    db, con = J._fixture()
    qs = ["SELECT o.o_custkey, l.l_suppkey, l.l_linenumber, COUNT(*), SUM(l.l_quantity) FROM lineitem l "
          "JOIN orders o ON l.l_orderkey = o.o_orderkey WHERE l.l_quantity > 10 GROUP BY o.o_custkey, l.l_suppkey, l.l_linenumber",
          "SELECT c.c_nationkey, l.l_suppkey, l.l_partkey, SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "JOIN customer c ON o.o_custkey = c.c_custkey WHERE l.l_shipmode = 'RAIL' GROUP BY c.c_nationkey, l.l_suppkey, l.l_partkey"]
    old = RT._DENSE_ROWS[0]
    try:
        h0 = WJ._DENSE_ROWS_HITS[0]
        for q in qs:
            for on in (True, False):
                RT._DENSE_ROWS[0] = on
                controller.plan_epoch_bump()
                J._match(q, expect_fast=False)
        assert WJ._DENSE_ROWS_HITS[0] > h0, 'the factorise by the rows was never taken'
    finally:
        RT._DENSE_ROWS[0] = old
        controller.plan_epoch_bump()
