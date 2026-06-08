"""Tests for wdb_bsi: bit-sliced index + per-value bitmaps.
All synthetic + tiny; every predicate checked bit-exact vs brute-force numpy."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import wdb_bsi


def _mask(bm, N):
    return np.unpackbits(bm)[:N].astype(bool)


def test_bsi_ge_matches_bruteforce():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 50, size=1000)
    bsi = wdb_bsi.build_bsi(codes)
    for C in [0, 1, 7, 24, 49, 50, 100]:
        assert np.array_equal(_mask(bsi.ge(C), codes.size), codes >= C), C


def test_bsi_range_matches_bruteforce():
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 366, size=5000)
    bsi = wdb_bsi.build_bsi(codes)
    for lo, hi in [(0, 366), (100, 200), (5, 6), (300, 366), (50, 50)]:
        assert np.array_equal(_mask(bsi.range(lo, hi), codes.size),
                              (codes >= lo) & (codes < hi)), (lo, hi)


def test_bsi_eq_and_in():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 30, size=2000)
    bsi = wdb_bsi.build_bsi(codes)
    for C in [0, 5, 11, 29]:
        assert np.array_equal(_mask(bsi.eq(C), codes.size), codes == C), C
    s = [3, 7, 20, 29]
    assert np.array_equal(_mask(bsi.in_set(s), codes.size), np.isin(codes, s))


def test_bsi_high_cardinality_range():
    # cardinality far above any per-value-bitmap budget; range still O(B)
    rng = np.random.default_rng(4)
    codes = rng.integers(0, 2526, size=20000)
    bsi = wdb_bsi.build_bsi(codes)
    assert bsi.B == 12                                  # 2525 -> 12 bits
    assert np.array_equal(_mask(bsi.range(800, 1166), codes.size),
                          (codes >= 800) & (codes < 1166))


def test_value_bitmaps_in_matches_isin():
    rng = np.random.default_rng(5)
    codes = rng.integers(0, 11, size=3000)
    bms = wdb_bsi.build_value_bitmaps(codes, 11)
    s = [5, 6, 7]
    assert np.array_equal(_mask(wdb_bsi.vbm_in(bms, s), codes.size), np.isin(codes, s))
    assert np.array_equal(_mask(wdb_bsi.vbm_in(bms, []), codes.size),
                          np.zeros(codes.size, bool))


def test_combine_and_or_is_logical():
    rng = np.random.default_rng(6)
    a = rng.integers(0, 40, size=4000); b = rng.integers(0, 40, size=4000)
    ba = wdb_bsi.build_bsi(a); bb = wdb_bsi.build_bsi(b)
    ma, mb = ba.range(5, 8), bb.range(0, 24)
    assert np.array_equal(_mask(ma & mb, a.size), (a >= 5) & (a < 8) & (b < 24))
    assert np.array_equal(_mask(ma | mb, a.size), ((a >= 5) & (a < 8)) | (b < 24))


def test_pad_bits_held_zero():
    # N not a multiple of 8: nothing beyond N may register, popcount exact
    codes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)  # N=13
    bsi = wdb_bsi.build_bsi(codes)
    m = bsi.ge(0)                                       # all true
    assert wdb_bsi.popcount(m) == 13
    assert _mask(m, 13).all() and np.unpackbits(m).size == 16  # 3 pad bits
    assert wdb_bsi.popcount(bsi.range(3, 7)) == 4


def test_positions_match_where():
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 100, size=5000)
    bsi = wdb_bsi.build_bsi(codes)
    m = bsi.range(10, 20)
    assert np.array_equal(wdb_bsi.to_positions(m, codes.size),
                          np.where((codes >= 10) & (codes < 20))[0])


def test_single_value_column():
    codes = np.zeros(64, dtype=np.int64)               # max=0 -> B=1
    bsi = wdb_bsi.build_bsi(codes)
    assert bsi.B == 1
    assert wdb_bsi.popcount(bsi.eq(0)) == 64
    assert wdb_bsi.popcount(bsi.ge(1)) == 0
