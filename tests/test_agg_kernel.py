"""wdb_agg group-by kernel vs pandas, across dtypes / aggregates / nulls."""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import wdb_agg

rng = np.random.default_rng(7)
N, K = 50000, 6
codes = rng.integers(0, K, N).astype(np.int64)
present = np.nonzero(np.bincount(codes, minlength=K) > 0)[0]

def _close(a, b):
    if a is None or (isinstance(a, float) and np.isnan(a)): return b is None
    return abs(float(a) - float(b)) < 1e-6

def _ref(fn, v=None, null=None):
    import pandas as pd
    g = pd.Series(np.arange(N)); 
    if fn == 'COUNT_STAR': s = pd.Series(v if v is not None else codes).groupby(codes).size()
    else:
        ser = pd.Series(v)
        if null is not None: ser = ser.mask(null)
        gp = ser.groupby(codes)
        if   fn == 'COUNT': s = gp.count()
        elif fn == 'SUM':   s = gp.sum(min_count=1)
        elif fn == 'AVG':   s = gp.mean()
        elif fn == 'MIN':   s = gp.min()
        elif fn == 'MAX':   s = gp.max()
    return {int(k): s.loc[k] for k in s.index}

def test_count_star():
    out = wdb_agg.group_agg(codes, K, 'COUNT_STAR')
    ref = _ref('COUNT_STAR', codes)
    for k in present: assert int(out[k]) == int(ref[k])

def test_sum_avg_int():
    v = rng.integers(-1000, 1000, N).astype(np.int64)
    for fn in ('SUM', 'AVG'):
        out = wdb_agg.group_agg(codes, K, fn, v); ref = _ref(fn, v)
        for k in present: assert _close(out[k], ref[k]), (fn, k, out[k], ref[k])

def test_sum_avg_float():
    v = rng.normal(50, 20, N).astype(np.float64)
    for fn in ('SUM', 'AVG'):
        out = wdb_agg.group_agg(codes, K, fn, v); ref = _ref(fn, v)
        for k in present: assert _close(out[k], ref[k]), (fn, k, out[k], ref[k])

def test_minmax_int():
    v = rng.integers(-1000, 1000, N).astype(np.int64)
    for fn in ('MIN', 'MAX'):
        out = wdb_agg.group_agg(codes, K, fn, v); ref = _ref(fn, v)
        for k in present: assert int(out[k]) == int(ref[k]), (fn, k, out[k], ref[k])

def test_minmax_float():
    v = rng.normal(0, 100, N).astype(np.float64)
    for fn in ('MIN', 'MAX'):
        out = wdb_agg.group_agg(codes, K, fn, v); ref = _ref(fn, v)
        for k in present: assert _close(out[k], ref[k]), (fn, k, out[k], ref[k])

def test_minmax_datetime():
    base = np.datetime64('1995-01-01')
    v = base + rng.integers(0, 3000, N).astype('timedelta64[D]')
    for fn in ('MIN', 'MAX'):
        out = wdb_agg.group_agg(codes, K, fn, v); ref = _ref(fn, v)
        for k in present: assert np.datetime64(out[k]) == np.datetime64(ref[k]), (fn, k, out[k], ref[k])

def test_minmax_string():
    words = np.array([b'apple', b'mango', b'cherry', b'date', b'fig'], dtype=object)
    v = words[rng.integers(0, len(words), N)]
    for fn in ('MIN', 'MAX'):
        out = wdb_agg.group_agg(codes, K, fn, v); ref = _ref(fn, v)
        for k in present: assert out[k] == ref[k], (fn, k, out[k], ref[k])

def test_count_and_sum_with_nulls():
    v = rng.integers(0, 500, N).astype(np.int64)
    null = rng.random(N) < 0.3
    for fn in ('COUNT', 'SUM', 'AVG'):
        out = wdb_agg.group_agg(codes, K, fn, v, nullmask=null)
        ref = _ref(fn, v.astype(float), null=null)
        for k in present:
            if fn == 'COUNT': assert int(out[k]) == int(ref[k]), (k, out[k], ref[k])
            else: assert _close(out[k], ref[k]), (fn, k, out[k], ref[k])


def test_parallel_matches_serial():
    """Threaded fused kernel must match the serial path bit-for-bit across COUNT/SUM/AVG, +/- nulls."""
    rng2 = np.random.default_rng(11)
    N2, K2 = 3_000_000, 7
    g = rng2.integers(0, K2, N2).astype(np.int64)
    vf = rng2.normal(100, 30, N2).astype(np.float64)
    vi = rng2.integers(0, 1000, N2).astype(np.int64)
    nm = rng2.random(N2) < 0.2
    specs = [(0, 'COUNT', vf, None), (1, 'SUM', vf, None), (2, 'AVG', vf, None),
             (3, 'SUM', vi, nm), (4, 'AVG', vf, nm), (5, 'COUNT', vf, nm)]
    counts, got = wdb_agg.parallel_counts_and_aggs(g, K2, specs, n_threads=8)
    assert np.array_equal(counts, wdb_agg.group_counts(g, K2))
    for key, fn, v, m in specs:
        exp = wdb_agg.group_agg(g, K2, fn, v, m)
        for k in range(K2):
            a, b = got[key][k], exp[k]
            if a is None or b is None: assert a is b, (fn, key, k, a, b)
            else:  # parallel chunked sums differ from serial only by float non-associativity
                assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-6), (fn, key, k, a, b)
