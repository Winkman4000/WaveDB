"""THE CLUSTER ORDER BY COUNTING (2026-09-26): wdb_kernels.counting_order must give exactly the
permutation np.lexsort((key,)) gives -- equal keys keep their row order -- for every key width and
sign, and decline (None) where it cannot: a span past its limit, floats, u64."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import wdb_kernels as K


def _same(key, dt=np.int64):
    got = K.counting_order(key, dt)
    assert got is not None
    assert got.dtype == dt
    assert np.array_equal(got, np.lexsort((np.asarray(key),)))


def test_ties_keep_row_order():
    rng = np.random.default_rng(1)
    _same(rng.integers(0, 50, 200000).astype(np.int64))
    _same(rng.integers(0, 50, 200000).astype(np.int64), np.int32)


def test_narrow_signed_keys_do_not_wrap():
    rng = np.random.default_rng(2)
    _same(rng.integers(-100, 101, 50000).astype(np.int8))
    _same(rng.integers(-30000, 30001, 50000).astype(np.int16))
    _same(rng.integers(-(1 << 20), 1 << 20, 50000).astype(np.int32))
    _same(rng.integers(0, 60000, 50000).astype(np.uint16))
    _same(rng.integers(3_000_000_000, 3_000_100_000, 50000).astype(np.uint32))


def test_clock_keys():
    rng = np.random.default_rng(3)
    s = rng.integers(1_372_000_000, 1_372_000_000 + 2_592_000, 300000).astype(np.int64)
    _same(s)
    _same(s.astype('datetime64[s]'))


def test_edges():
    _same(np.zeros(0, np.int64))
    _same(np.array([7], np.int64))
    _same(np.full(1000, 5, np.int32))


def test_declines():
    assert K.counting_order(np.array([0, K.COUNTING_ORDER_MAX_SPAN + 1], np.int64)) is None
    assert K.counting_order(np.array([0.5, 1.5])) is None
    assert K.counting_order(np.array([1, 2], np.uint64)) is None
