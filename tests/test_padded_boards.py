"""THE PADDED BOARDS (2026-10-18): the per-thread count / sum boards are padded so no two threads' counters share
a cache line (at K = 2 an unpadded board cost mx_fold2 1.8 s against 18 ms, pf_dense 1.0 s against 32 ms). The
padding must never reach an answer: every padded kernel equals its numpy reference at the small widths where the
padding is most of the row, and at a width past it."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import wdb_kernels as WK
import wdb_lens


def test_padded_boards_equal_numpy():
    rng = np.random.default_rng(23)
    n = 300_001
    T = WK._nt()
    vc = rng.integers(0, 50, n).astype(np.uint8)
    vt = rng.integers(0, 1000, 50).astype(np.int64)
    dv = rng.random(50)
    L = rng.integers(0, 9, n).astype(np.int64)
    mask = rng.random(n) < 0.4
    for K in (1, 2, 3, 17, 300):
        kc = rng.integers(0, K, n).astype(np.uint16)
        c2 = rng.integers(0, 2, n).astype(np.uint8)
        ref = np.bincount(kc, minlength=K)
        assert np.array_equal(WK._count_codes_nb(kc, np.int64(K), T), ref), K
        assert np.array_equal(WK._grouped_sum_codes_nb(kc, vc, vt, np.int64(K), T),
                              np.bincount(kc, weights=vt[vc], minlength=K).astype(np.int64)), K
        assert np.array_equal(WK._masked_count_nb(kc, mask, np.int64(K), T), np.bincount(kc[mask], minlength=K)), K
        pair = kc.astype(np.int64) * 2 + c2
        assert np.array_equal(WK._grid2_count_nb(kc, c2, kc, np.int64(-1), np.int64(2), np.int64(2 * K), T),
                              np.bincount(pair, minlength=2 * K)), K
        out = np.zeros(K)
        WK._group_fold_dict_nb(kc, vc, dv, np.int64(K), out, T)
        assert np.allclose(out, np.bincount(kc, weights=dv[vc], minlength=K)), K
        c, s1, s2 = WK.mx_fold2(kc, vc, dv, vc, dv, K)
        assert c.size == K and np.array_equal(c, ref) and np.allclose(s1, np.bincount(kc, weights=dv[vc], minlength=K))
        pk = c2.astype(np.int64) + kc.astype(np.int64) * 2              # pf_dense's pair code b * VA + a
        c, s1, s2 = WK.pf_dense(None, c2, kc, vc, vc, dv, dv, 2, 2 * K)
        assert c.size == 2 * K and np.array_equal(c, np.bincount(pk, minlength=2 * K)), K
        assert np.allclose(s2, np.bincount(pk, weights=dv[vc], minlength=2 * K)), K
        perm = WK._part_scatter_nb(kc, np.int64(K), T)
        assert np.array_equal(perm, np.argsort(kc, kind='stable')), K
        S, C = wdb_lens.row_pour(kc, L, K, True, T)
        m = L > 0
        assert S.size == K and np.array_equal(C, np.bincount(kc[m], minlength=K)), K
        assert np.array_equal(S, np.bincount(kc[m], weights=L[m], minlength=K).astype(np.int64)), K
