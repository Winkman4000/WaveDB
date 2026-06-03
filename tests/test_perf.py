"""Anomaly detection: measured speed vs theoretical projection.

Philosophy: absolute ms is noisy and machine-specific, so we DON'T assert on it.
We assert on the SCALING LAW - a property of the algorithm's complexity class,
derived from our theoretical framework. A linear op's time must ~double when the
data doubles; if it trends toward 4x, the op went superlinear (regression / edge case).

Each test prints projected-vs-measured so deviations are visible as optimization
opportunities even when within tolerance. Deviation = signal worth investigating
(e.g. the scatter_add-into-25-bins case: model said fast, reality was 40x slower).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from helpers import roundtrip

def _bestof(fn, k=5):
    b = 9e9
    for _ in range(k):
        t = time.time(); fn(); b = min(b, time.time() - t)
    return b

def _scaling_exponent(sizes, times):
    # fit time ~ size^p in log-log; p is the empirical complexity exponent
    lx = np.log(np.array(sizes, float)); ly = np.log(np.array(times, float))
    return float(np.polyfit(lx, ly, 1)[0])

def _check_scaling(label, make_op, sizes, the_exp, lo, hi):
    times = []
    for n in sizes:
        op = make_op(n)
        times.append(_bestof(op))
    p = _scaling_exponent(sizes, times)
    rates = [f"{n//1000}k:{t*1000:.0f}ms" for n, t in zip(sizes, times)]
    print(f"    [{label}] theory=O(n^{the_exp}) measured_exp={p:.2f} band=[{lo},{hi}]  {' '.join(rates)}")
    assert lo <= p <= hi, (f"{label}: scaling exponent {p:.2f} outside [{lo},{hi}] "
                           f"(theory O(n^{the_exp})) - op may have changed complexity class")
    return p

# ---- codes() decode: bit-unpack is O(rows). exponent must be ~1.0 ----
def test_codes_decode_is_linear():
    sizes = [1_000_000, 2_000_000, 4_000_000]
    def mk(n):
        seg, _ = roundtrip(pd.DataFrame({'x': np.random.default_rng(0).permutation(n).astype(np.int64)}))
        def op():
            seg._codes.clear(); seg.codes('x')
        return op
    _check_scaling("codes() decode", mk, sizes, 1, 0.80, 1.40)

# ---- values() decode+gather: also O(rows). exponent ~1.0 ----
def test_values_decode_is_linear():
    sizes = [1_000_000, 2_000_000, 4_000_000]
    def mk(n):
        seg, _ = roundtrip(pd.DataFrame({'x': np.random.default_rng(0).permutation(n).astype(np.int64)}))
        def op():
            seg._codes.clear()
            if 'intvals' in seg.cols['x']: seg.cols['x']['intvals'] = None
            seg.values('x')
        return op
    _check_scaling("values() decode", mk, sizes, 1, 0.80, 1.40)

# ---- coarse throughput floor: decode must stay in the right order of magnitude ----
# wide band (only flags catastrophes like the 40x scatter_add regression), never noise.
def test_decode_throughput_floor():
    n = 4_000_000
    seg, _ = roundtrip(pd.DataFrame({'x': np.random.default_rng(0).permutation(n).astype(np.int64)}))
    t = _bestof(lambda: (seg._codes.clear(), seg.codes('x'))[1])
    rate = n / t / 1e6
    PROJECTED = 20.0  # M rows/s, measured baseline on this machine's bit-unpack path
    floor = PROJECTED / 4  # flag only if >4x slower than projection
    print(f"    [decode throughput] projected~{PROJECTED}M/s measured={rate:.0f}M/s floor={floor:.0f}M/s")
    assert rate >= floor, (f"decode {rate:.0f}M/s far below projected {PROJECTED}M/s "
                           f"- catastrophic regression (cf. scatter_add 40x case)")
