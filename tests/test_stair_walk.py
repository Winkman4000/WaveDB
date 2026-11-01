"""THE STAIRS WALKED (2026-11-01): wdb_kernels.stairs_at equals np.searchsorted(st, rows, side='right') -- sorted rows
with repeats, rows before the first step and past the last, no steps, unsorted rows and short row lists (numpy
answers those), the switch off. A join grouping by a staircase key (code_enc 2, asserted) answers like DuckDB with
the walk on and off, and the walk is taken."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np


def test_equals_searchsorted():
    import wdb_kernels as WK
    rng = np.random.default_rng(1101)
    N = 3_000_000
    for S in (0, 1, 50, 200_000):
        st = np.sort(rng.choice(np.arange(1, N), S, replace=False)).astype(np.int64)
        for n in (10, 70_000, 600_001):
            rows = np.sort(rng.integers(0, N + 5, n)).astype(np.int64)       # repeats, past the last step
            rows[:3] = 0                                                       # before the first step
            exp = np.searchsorted(st, rows, side='right')
            assert np.array_equal(WK.stairs_at(st, rows), exp), (S, n)
            sh = rng.permutation(rows)
            assert np.array_equal(WK.stairs_at(st, sh), np.searchsorted(st, sh, side='right')), (S, n, 'unsorted')
    st = np.sort(rng.choice(N, 1000, replace=False)).astype(np.int64)
    rows = np.sort(rng.integers(0, N, 100_000)).astype(np.int32)
    assert np.array_equal(WK.stairs_at(st, rows), np.searchsorted(st, rows, side='right'))
    old = WK._STAIR_WALK[0]
    try:
        WK._STAIR_WALK[0] = False
        assert np.array_equal(WK.stairs_at(st, rows), np.searchsorted(st, rows, side='right'))
    finally:
        WK._STAIR_WALK[0] = old


def test_join_answers_both_ways():
    import wdb_kernels as WK, controller, test_join_fast as J, wdb_join as WJ
    db, con = J._fixture()
    seg, _ = WJ._solo_segment(db, 'lineitem')
    pc = db.cat.phys_map('lineitem').get('l_orderkey', 'l_orderkey')
    assert seg.cols[pc].get('code_enc') == 2, 'the fixture key must be a staircase (else the test is vacuous)'
    qs = ["SELECT l.l_orderkey, o.o_orderdate, SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "WHERE l.l_shipdate > DATE '1995-03-15' AND o.o_orderdate < DATE '1995-03-15' GROUP BY l.l_orderkey, o.o_orderdate",
          "SELECT l.l_orderkey, COUNT(*) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "WHERE l.l_quantity > 45 GROUP BY l.l_orderkey"]
    calls = [0]
    orig = WK._stairs_walk_nb
    def counted(*a):
        calls[0] += 1
        return orig(*a)
    old = (WK._STAIR_WALK[0], WK._STAIR_WALK_MIN)
    try:
        WK._stairs_walk_nb = counted
        WK._STAIR_WALK_MIN = 1                                # the fixture is small: take the walk
        for q in qs:
            for on in (True, False):
                WK._STAIR_WALK[0] = on
                controller.plan_epoch_bump()
                J._match(q, expect_fast=False)
        assert calls[0] > 0, 'the walk was never taken'
    finally:
        WK._stairs_walk_nb = orig
        WK._STAIR_WALK[0], WK._STAIR_WALK_MIN = old
        controller.plan_epoch_bump()
