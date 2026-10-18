"""The census atoms (2026-10-18): bincount_par with its per-thread boards padded apart equals np.bincount for small
and large dictionaries; topk_bar on a plateau (many counts tied at the k-th) equals the stable reference order --
value descending, then index ascending."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import wdb_kernels as WK


def test_bincount_par_small_and_large_dictionaries():
    rng = np.random.default_rng(5)
    for V in (1, 2, 3, 15, 16, 17, 255, 4096, 300_000, (64 << 20) + 5, 100_000_000):
        codes = rng.integers(0, V, 5_000_000).astype(np.uint32 if V > 255 else np.uint8)
        if V > (64 << 20):
            codes[:1000] = V - 1; codes[1000:2000] = 0           # both ends of the dictionary
        got = np.asarray(WK.bincount_par(codes, V))
        assert np.array_equal(got, np.bincount(codes, minlength=V)), V


def _ref(a, k):
    return np.argsort(-a, kind='stable')[:k]


def test_topk_bar_plateaus():
    rng = np.random.default_rng(6)
    cases = [np.ones(2_000_000, np.int64),                                  # every count 1 (a near-unique key)
             np.concatenate([np.full(1_000_000, 2), np.ones(3_000_000)]).astype(np.int64),
             rng.integers(1, 4, 3_000_000).astype(np.int64),                 # three values, huge ties
             np.where(rng.random(2_000_000) < 0.001, rng.integers(5, 50, 2_000_000), 1).astype(np.int64),
             rng.integers(0, 1_000_000, 500_000).astype(np.int64)]           # few ties
    a = np.ones(1_000_000, np.int64); a[[5, 900_000, 77]] = [9, 9, 3]; cases.append(a)
    a = rng.integers(0, 3, 3_000_000).astype(np.int64); a[[10, 20]] = [(1 << 21), (1 << 22)]; cases.append(a)   # past the cap
    a = np.zeros(2_000_000, np.int64); a[rng.choice(2_000_000, 40, replace=False)] = (1 << 21) + 5; cases.append(a)
    cases.append(rng.normal(0, 5, 2_000_000))                         # not a census: the bar path
    cases.append(rng.integers(-50, 50, 2_000_000).astype(np.int64))
    for a in cases:
        for k in (1, 3, 10, 25):
            got = np.asarray(WK.topk_bar(a, k))
            assert np.array_equal(got, _ref(a, k)), (a[:5], k, got[:5], _ref(a, k)[:5])


def _ref_any(a, k):
    """value descending, index ascending, for any dtype (no negation: an unsigned array would wrap)"""
    idx = np.arange(a.size)
    w = a.astype(np.float64) if a.dtype.kind == 'f' else a.astype(np.int64)   # every case here fits int64
    return np.lexsort((idx, -w))[:k]


def test_topk_bar_cap_edges_and_dtypes():
    """the 4096-bin census: ties straddling the cap, the k-th exactly at it, unsigned and narrow counts, a float
    range under one (the old int() of max and min called it a constant), bools"""
    rng = np.random.default_rng(7)
    C = WK.BAR_CAP
    cases = []
    a = rng.integers(0, 50, 200_000).astype(np.int64); a[rng.choice(200_000, 30, replace=False)] = C; cases.append(a)
    a = rng.integers(0, 50, 200_000).astype(np.int64); a[[3, 9, 70_000]] = [C - 1, C, C + 1]; cases.append(a)
    a = rng.integers(C - 3, C + 3, 200_000).astype(np.int64); cases.append(a)               # every value near the cap
    a = rng.integers(0, 1 << 30, 200_000).astype(np.uint32); a[[1, 2]] = (1 << 32) - 1; cases.append(a)
    cases.append(rng.integers(0, 7, 300_000).astype(np.uint8))
    a = rng.integers(0, 3, 300_000).astype(np.int32); a[100] = 1 << 30; cases.append(a)
    cases.append(rng.random(200_000) * 0.5)
    cases.append(rng.random(200_000) < 0.0001)
    for a in cases:
        for k in (1, 3, 10, 40):
            got = np.asarray(WK.topk_bar(a, k))
            ref = _ref_any(a, k)
            assert np.array_equal(got, ref), (a.dtype, k, got[:5], ref[:5])
