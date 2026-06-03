"""Boundary, overflow, and tiny-size corpus. These are the adversarial integer inputs a
delta/sequence codec must survive -- int64 extremes, ranges whose gaps overflow int64, huge
strides, and N in {0,1,2}. Also empirically settles whether cumsum(diff(x)) round-trips at
extremes via two's-complement wraparound (which determines whether mode-4's reconstruct is
safe there). A failure here on current code is a real pre-existing finding, not noise."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from helpers import roundtrip, assert_lossless

IMIN, IMAX = np.iinfo(np.int64).min, np.iinfo(np.int64).max

def test_int64_extremes_mode0():
    df = pd.DataFrame({'x': np.array([IMIN, IMAX, 0, -1, 1, IMIN+1, IMAX-1, 0]*30, dtype=np.int64)})
    seg, pq = roundtrip(df); assert_lossless(seg, pq, 'x')

def test_int64_extreme_range_highcard():
    # high-card RANDOM values spanning BOTH extremes -> mode 2; the sorted-dict gap between
    # the low and high clusters overflows int64. Verifies diff/cumsum cancellation in the
    # mode-2 delta path. (Affine extreme ranges are covered by the mode-4 seqcodec tests.)
    rng = np.random.default_rng(3)
    lo = IMIN + rng.integers(0, 10**9, 30000).astype(np.int64)
    hi = IMAX - rng.integers(0, 10**9, 30000).astype(np.int64)
    vals = np.unique(np.concatenate([lo, hi]))
    df = pd.DataFrame({'x': vals})
    seg, pq = roundtrip(df)
    m = assert_lossless(seg, pq, 'x')
    assert m == 2, f"expected mode 2 for high-card random extreme range, got {m}"

def test_large_stride_near_overflow():
    # base + i*stride where the total span is enormous but each value fits int64
    n = 70000; stride = (IMAX // n) - 1
    df = pd.DataFrame({'x': (np.arange(n, dtype=np.int64) * stride) + IMIN//2})
    seg, pq = roundtrip(df); assert_lossless(seg, pq, 'x')

def test_monotonic_with_huge_jump():
    # mostly stride 1, then one gap so large the delta overflows if computed naively
    a = np.arange(60000, dtype=np.int64)
    a[30000:] = a[30000:] + (IMAX - 100000)   # giant jump in the middle, still distinct & sorted
    df = pd.DataFrame({'x': np.unique(a)})
    seg, pq = roundtrip(df); assert_lossless(seg, pq, 'x')

def test_empty_column():
    df = pd.DataFrame({'x': pd.array([], dtype='Int64')})
    seg, pq = roundtrip(df); assert_lossless(seg, pq, 'x')

def test_single_row_int():
    df = pd.DataFrame({'x': np.array([12345], dtype=np.int64)})
    seg, pq = roundtrip(df); assert_lossless(seg, pq, 'x')

def test_two_rows_int():
    df = pd.DataFrame({'x': np.array([IMIN, IMAX], dtype=np.int64)})
    seg, pq = roundtrip(df); assert_lossless(seg, pq, 'x')

def test_single_row_string():
    df = pd.DataFrame({'x': ['only']}); seg, pq = roundtrip(df); assert_lossless(seg, pq, 'x')

def test_two_distinct_alternating():
    df = pd.DataFrame({'x': np.array([7, -7]*5000, dtype=np.int64)})
    seg, pq = roundtrip(df); assert_lossless(seg, pq, 'x')

def test_all_same_extreme():
    df = pd.DataFrame({'x': np.full(1000, IMIN, dtype=np.int64)})
    seg, pq = roundtrip(df); assert_lossless(seg, pq, 'x')

def test_datetime_epoch_extremes():
    # very early and very late timestamps (large int64 epoch range), high-card -> mode 2
    base = np.datetime64('1970-01-01T00:00:00','s').astype('int64')
    vals = np.unique(np.concatenate([np.arange(30000), np.arange(10**9, 10**9+30000)])).astype('datetime64[s]')
    df = pd.DataFrame({'x': vals}); seg, pq = roundtrip(df); assert_lossless(seg, pq, 'x')

def test_wraparound_cancellation_direct():
    # the underlying numpy guarantee mode-4/mode-2 lean on: cumsum(diff(x)) == x in int64
    # even when intermediate deltas overflow. Pin it so a numpy change can't silently break us.
    x = np.array([IMIN, 0, IMAX, -1, IMAX//2, IMIN//3], dtype=np.int64)
    d = np.diff(x)
    recon = np.empty_like(x); recon[0] = x[0]; np.cumsum(d, out=recon[1:]); recon[1:] += x[0]
    assert np.array_equal(recon, x), "int64 diff/cumsum no longer cancels -- codec assumption broken"
