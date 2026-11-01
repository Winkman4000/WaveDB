"""THE ROADS IN ONE PASS (2026-11-01): wdb_kernels.fuse_roads_rows equals numpy's flatnonzero(mask & keep1[p1]
(& keep2[p2])) for one and two roads, int32 and int64 pointers, empty / full / sparse masks and keeps, sizes across
the 2^20 chunk. Joins with a fact keep and a fact-adjacent parent keep answer like DuckDB with it on and off, and
the one pass is taken."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np


def test_equals_numpy():
    import wdb_kernels as WK
    rng = np.random.default_rng(1101)
    checked = 0
    for n in (7, 50_000, (1 << 20) + 12_345):
        A, B = 3_000, 40
        for pdt in (np.int32, np.int64):
            p1 = rng.integers(0, A, n).astype(pdt); p2 = rng.integers(0, B, n).astype(pdt)
            for mk in (rng.random(n) < 0.3, np.zeros(n, bool), np.ones(n, bool)):
                k1 = rng.random(A) < 0.5; k2 = rng.random(B) < 0.2
                assert np.array_equal(WK.fuse_roads_rows(mk, [(p1, k1)]), np.flatnonzero(mk & k1[p1]))
                assert np.array_equal(WK.fuse_roads_rows(mk, [(p1, k1), (p2, k2)]), np.flatnonzero(mk & k1[p1] & k2[p2]))
                checked += 2
    assert checked == 36


def test_join_answers_both_ways():
    import wdb_join as WJ, controller, test_join_fast as J
    db, con = J._fixture()
    qs = ["SELECT l.l_returnflag, COUNT(*), SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "WHERE l.l_quantity > 20 AND o.o_orderstatus = 'F' GROUP BY l.l_returnflag",
          "SELECT c.c_mktsegment, COUNT(*), SUM(l.l_quantity) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "JOIN customer c ON o.o_custkey = c.c_custkey WHERE l.l_shipmode = 'AIR' AND o.o_orderpriority = '1-URGENT' "
          "AND c.c_mktsegment = 'BUILDING' GROUP BY c.c_mktsegment",
          "SELECT o.o_orderpriority, COUNT(*) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "WHERE l.l_discount > 0.05 AND o.o_totalprice > 200000 GROUP BY o.o_orderpriority"]
    old = WJ._FUSE_ROADS[0]
    try:
        h0 = WJ._FUSE_ROADS_HITS[0]
        for q in qs:
            for on in (True, False):
                WJ._FUSE_ROADS[0] = on
                controller.plan_epoch_bump()
                J._match(q, expect_fast=False)
        assert WJ._FUSE_ROADS_HITS[0] > h0, 'the one pass was never taken'
    finally:
        WJ._FUSE_ROADS[0] = old
        controller.plan_epoch_bump()
