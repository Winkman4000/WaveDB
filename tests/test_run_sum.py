"""THE RUNS SUM THEMSELVES (2026-11-02): wdb_kernels.road_run_sums equals np.bincount(road, weights=vals[codes]) bit for
bit (non-integer weights too: every run is summed by one thread in row order) and np.bincount(road) for counts --
short and long runs, a run longer than a chunk, parents with no rows, sizes across the line; None for a road that
steps down, a code outside the values, under the line, switched off. Q18's shape answers like DuckDB with it on and
off, and the run sums are taken."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np


def test_equals_bincount():
    import wdb_kernels as WK
    rng = np.random.default_rng(1102)
    for n, N in ((70_000, 20_000), (1_000_003, 300_000), (2_000_000, 7)):
        road = np.sort(rng.integers(0, N, n)).astype(np.int32)
        if n == 1_000_003:
            road[200_000:900_000] = road[200_000]                  # one run far longer than a chunk
            road = np.sort(road)
        vals = rng.random(50) * 1e3                                 # non-integer weights
        codes = rng.integers(0, 50, n).astype(np.uint8)
        got = WK.road_run_sums(road, codes, vals, N)
        assert got is not None and np.array_equal(got, np.bincount(road, weights=vals[codes], minlength=N)), n
        cnt = WK.road_run_sums(road, None, None, N)
        assert cnt is not None and np.array_equal(cnt, np.bincount(road, minlength=N).astype(np.float64)), n
    road = np.sort(rng.integers(0, 1000, 100_000)).astype(np.int64)
    codes = rng.integers(0, 50, 100_000).astype(np.uint8)
    assert WK.road_run_sums(road[::-1].copy(), codes, vals, 1000) is None            # steps down
    bad = codes.copy(); bad[77_777] = 60
    assert WK.road_run_sums(road, bad, vals, 1000) is None                           # a code outside the values
    assert WK.road_run_sums(road[:100], codes[:100], vals, 1000) is None             # under the line
    old = WK._RUN_SUM[0]
    try:
        WK._RUN_SUM[0] = False
        assert WK.road_run_sums(road, codes, vals, 1000) is None
    finally:
        WK._RUN_SUM[0] = old


def test_q18_shape_both_ways():
    import wdb_kernels as WK, wdb_join as WJ, controller, test_join_fast as J
    db, con = J._fixture()
    q = ("SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, SUM(l_quantity) FROM customer, orders, lineitem "
         "WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > %d) "
         "AND c_custkey = o_custkey AND o_orderkey = l_orderkey GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice "
         "ORDER BY o_totalprice DESC, o_orderdate LIMIT 100")
    old = (WK._RUN_SUM[0], WK._RUN_SUM_MIN)
    try:
        WK._RUN_SUM_MIN = 1                                   # the fixture is small: take the run sums
        h0 = WJ._RUN_SUM_HITS[0]
        for qty in (250, 300):
            for on in (True, False):
                WK._RUN_SUM[0] = on
                controller.plan_epoch_bump()
                J._match(q % qty, ordered=True, expect_fast=False)
        assert WJ._RUN_SUM_HITS[0] > h0, 'the run sums were never taken'
    finally:
        WK._RUN_SUM[0], WK._RUN_SUM_MIN = old
        controller.plan_epoch_bump()
