"""Rung-3 fused numba kernel: correctness of fused_numba / numba_grouped vs the numpy group_agg paths.
Covers COUNT/SUM/AVG/MIN/MAX, a WHERE mask, multi-column (V>=2), whole-table (K=1), the serial path, and
the parallel path (>PARALLEL_THRESHOLD rows, small K -> exercises the cache-line padding + reduction).
If numba is unavailable the engine uses the numpy paths, so these checks are simply skipped."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import wdb_agg as A
import wdb_measure_runtime as RT

def _cmp(counts, finals, codes, K, specs, m=None):
    c = codes if m is None else codes[m]
    assert np.array_equal(counts, A.group_counts(c, K)), "counts mismatch"
    for key, fn, v in specs:
        vv = None if v is None else (v if m is None else v[m])
        ref = A.group_agg(c, K, fn, vv); got = finals[key]
        for k in range(K):
            a, b = got[k], ref[k]
            if a is None or b is None:
                assert (a is None) == (b is None), f"{fn} None mismatch at {k}"
            else:
                assert np.isclose(float(a), float(b), rtol=1e-9, atol=1e-6), f"{fn} value at {k}"

def _run(codes, K, spec_in, m=None):
    # spec_in: list of (key, fn, value_array_or_None)
    specs_all = [(key, fn, (('d', v) if v is not None else None), None) for key, fn, v in spec_in]
    group_op = ('d', codes.astype(np.int64))
    mask_op = ('d', m) if m is not None else None
    counts, finals = A.fused_numba(group_op, K, specs_all, mask_op, len(codes))
    _cmp(counts, finals, codes.astype(np.int64), K, spec_in, m)

def test_fused_serial_all_aggs():
    if not A.HAS_NUMBA: return
    rng = np.random.default_rng(1); n = 400_000; K = 7
    codes = rng.integers(0, K, n); price = rng.uniform(1, 1000, n); qty = rng.integers(1, 50, n).astype(float)
    _run(codes, K, [(0, 'SUM', price), (1, 'AVG', price), (2, 'MIN', price), (3, 'MAX', price)])
    _run(codes, K, [(0, 'SUM', price), (1, 'SUM', qty), (2, 'MIN', qty)])          # V>=2
    _run(codes, K, [(0, 'COUNT', price), (1, 'AVG', price)])

def test_fused_serial_with_mask():
    if not A.HAS_NUMBA: return
    rng = np.random.default_rng(2); n = 400_000; K = 5
    codes = rng.integers(0, K, n); price = rng.uniform(1, 1000, n); m = rng.random(n) < 0.55
    _run(codes, K, [(0, 'SUM', price), (1, 'MIN', price), (2, 'MAX', price)], m=m)
    _run(codes, K, [(0, 'SUM', price), (1, 'SUM', price.copy())], m=m)             # V>=2 + mask

def test_fused_whole_table():
    if not A.HAS_NUMBA: return
    rng = np.random.default_rng(3); n = 300_000
    codes = np.zeros(n, np.int64); price = rng.uniform(1, 1000, n)
    _run(codes, 1, [(0, 'SUM', price), (1, 'MIN', price), (2, 'MAX', price)])

def test_fused_parallel_small_K_padding():
    # >PARALLEL_THRESHOLD rows with K=3: the false-sharing case the cache-line padding fixes.
    if not A.HAS_NUMBA: return
    rng = np.random.default_rng(4); n = RT.PARALLEL_THRESHOLD + 100_000; K = 3
    codes = rng.integers(0, K, n); price = rng.uniform(1, 1000, n)
    _run(codes, K, [(0, 'SUM', price), (1, 'MIN', price), (2, 'MAX', price)])      # unmasked parallel
    m = rng.random(n) < 0.5
    _run(codes, K, [(0, 'SUM', price), (1, 'MAX', price)], m=m)                    # masked parallel

def test_fused_gathered_group():
    # gathered group (group code = pcodes[ptr[i]]), parallel path + mask, vs materialise-then-group_agg
    if not A.HAS_NUMBA: return
    rng = np.random.default_rng(5)
    n_parent = 50_000; n = RT.PARALLEL_THRESHOLD + 200_000; K = 2000
    pcodes = rng.integers(0, K, n_parent).astype(np.int64)
    ptr = rng.integers(0, n_parent, n).astype(np.int64)
    price = rng.uniform(1, 1000, n)
    gc = pcodes[ptr]                                   # reference gather
    counts, sums, _, _ = A.numba_grouped_g(pcodes, ptr, [price], K)
    assert np.array_equal(counts, A.group_counts(gc, K))
    ref_sum = A.group_agg(gc, K, 'SUM', price)
    for k in range(K):
        if ref_sum[k] is None: assert counts[k] == 0
        else: assert np.isclose(float(sums[0][k]), float(ref_sum[k]), rtol=1e-9, atol=1e-6)
    m = rng.random(n) < 0.5                              # masked gathered path
    counts2, _, _, _ = A.numba_grouped_g(pcodes, ptr, [price], K, mask=m)
    assert np.array_equal(counts2, A.group_counts(gc[m], K))

def test_decode_fused_vs_materialised():
    # decode-fused kernels (value = base[vcodes[i]]) must match the materialised base[vcodes] path,
    # for direct & gathered group, masked & unmasked, serial & parallel.
    if not A.HAS_NUMBA: return
    rng = np.random.default_rng(11)
    def same(got, ref):
        gc_, gs, gmn, gmx = got; rc, rs, rmn, rmx = ref
        return (np.array_equal(gc_, rc) and np.allclose(gs, rs)
                and np.allclose(gmn, rmn) and np.allclose(gmx, rmx))
    for n in (250_000, RT.PARALLEL_THRESHOLD + 120_000):     # serial then parallel
        K = 1500
        base = rng.uniform(1, 1000, 30_000).astype(np.float64)
        vcodes = rng.integers(0, len(base), n).astype(np.int64)
        val = base[vcodes]
        gc = rng.integers(0, K, n).astype(np.int64)
        m = rng.random(n) < 0.5
        assert same(A.numba_grouped_d(gc, base, vcodes, K), A.numba_grouped(gc, [val], K))
        assert same(A.numba_grouped_d(gc, base, vcodes, K, mask=m), A.numba_grouped(gc, [val], K, mask=m))
        npar = 20_000
        pcodes = rng.integers(0, K, npar).astype(np.int64); ptr = rng.integers(0, npar, n).astype(np.int64)
        assert same(A.numba_grouped_gd(pcodes, ptr, base, vcodes, K), A.numba_grouped_g(pcodes, ptr, [val], K))
        assert same(A.numba_grouped_gd(pcodes, ptr, base, vcodes, K, mask=m),
                    A.numba_grouped_g(pcodes, ptr, [val], K, mask=m))
