"""Tests for wdb_bsi_kernels: bitmap-walk aggregates vs numpy reductions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import wdb_bsi_kernels as K


def _setup(N, dens, seed):
    rng = np.random.default_rng(seed)
    mask = rng.random(N) < dens
    packed = np.packbits(mask)
    a = rng.random(N) * 100.0
    b = rng.random(N)
    g = rng.integers(0, 4, size=N)
    return mask, packed, a, b, g


def test_count_matches():
    for N in (1000, 4096, 6007):
        mask, packed, a, b, g = _setup(N, 0.3, N)
        assert K.bw_count(packed, N) == int(mask.sum()), N


def test_sum1_sum2_match():
    mask, packed, a, b, g = _setup(5000, 0.2, 11)
    assert abs(K.bw_sum1(packed, a, 5000) - a[mask].sum()) < 1e-6
    assert abs(K.bw_sum2(packed, a, b, 5000) - (a * b)[mask].sum()) < 1e-6


def test_group_sum_and_count_match():
    N = 7000
    mask, packed, a, b, g = _setup(N, 0.25, 5)
    gs = K.bw_group_sum1(packed, a, g, 4, N)
    gc = K.bw_group_count(packed, g, 4, N)
    exp_s = np.bincount(g[mask], a[mask], minlength=4)
    exp_c = np.bincount(g[mask], minlength=4)
    assert np.allclose(gs, exp_s) and np.array_equal(gc, exp_c)


def test_pad_bits_ignored():
    # N=13 -> last byte has 3 pad bits; force them set in the raw mask region
    N = 13
    mask = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    packed = np.packbits(mask)                      # pads with 0 to 16 bits
    a = np.arange(N, dtype=np.float64) + 1.0
    assert K.bw_count(packed, N) == int(mask.sum())
    assert abs(K.bw_sum1(packed, a, N) - a[mask].sum()) < 1e-9
