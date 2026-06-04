"""Code-based group-by aggregation kernel.

Operates on DENSE integer group codes (0..K-1) -- exactly what WaveDB already stores for dictionary
columns -- using numpy bincount for COUNT/SUM/AVG (one vectorized pass, no sort, no object keys) and
native reductions for MIN/MAX. This is the inner loop the join and single-table group-by share, and
the kernel we later fuse into C/SIMD. Everything here is integer-indexed and branch-light by design.

Each aggregate returns a numpy/object array of length K, indexed by group code; the caller keeps only
groups with a nonzero row count and maps codes back to key values.
"""
import numpy as np


def group_counts(codes, K):
    """Rows per group code (COUNT(*))."""
    return np.bincount(codes, minlength=K)


def group_agg(codes, K, fn, v=None, nullmask=None):
    """One aggregate over dense group codes 0..K-1.

    fn: 'COUNT_STAR' | 'COUNT' | 'SUM' | 'AVG' | 'MIN' | 'MAX'
    v:  value array (native dtype) for non-count aggregates
    nullmask: optional bool array, True where v is NULL (SQL aggregates skip NULLs)
    Returns array length K (object array with None for empty groups on SUM/AVG/MIN/MAX).
    """
    if fn == 'COUNT_STAR':
        return np.bincount(codes, minlength=K)
    if nullmask is not None:
        keep = ~nullmask
        codes = codes[keep]
        v = None if v is None else v[keep]
    if fn == 'COUNT':
        return np.bincount(codes, minlength=K)
    cnt = np.bincount(codes, minlength=K)
    if fn in ('SUM', 'AVG'):
        s = np.bincount(codes, weights=v.astype(np.float64), minlength=K)
        if fn == 'SUM':
            out = s.astype(object); out[cnt == 0] = None; return out
        out = np.full(K, None, dtype=object)
        nz = cnt > 0; out[nz] = s[nz] / cnt[nz]; return out
    if fn in ('MIN', 'MAX'):
        return _group_minmax(codes, v, K, cnt, fn)
    raise ValueError(f"unknown aggregate {fn!r}")


def _group_minmax(codes, v, K, cnt, fn):
    out = np.full(K, None, dtype=object)
    nz = np.nonzero(cnt > 0)[0]
    if v.dtype.kind in 'iuf' or v.dtype.kind == 'M':
        iv = v.view(np.int64) if v.dtype.kind == 'M' else v.astype(np.int64) if v.dtype.kind in 'iu' else None
        if v.dtype.kind == 'f':
            acc = np.full(K, np.inf if fn == 'MIN' else -np.inf, dtype=np.float64)
            (np.minimum.at if fn == 'MIN' else np.maximum.at)(acc, codes, v)
            for k in nz: out[k] = acc[k]
            return out
        init = np.iinfo(np.int64).max if fn == 'MIN' else np.iinfo(np.int64).min
        acc = np.full(K, init, dtype=np.int64)
        (np.minimum.at if fn == 'MIN' else np.maximum.at)(acc, codes, iv)
        res = acc.view(v.dtype) if v.dtype.kind == 'M' else acc
        for k in nz: out[k] = res[k]
        return out
    # object/string: grouped reduction (rare in hot joins)
    best = {}
    cmp = (lambda a, b: a < b) if fn == 'MIN' else (lambda a, b: a > b)
    for c, val in zip(codes, v):
        if c not in best or cmp(val, best[c]): best[c] = val
    for k, val in best.items(): out[k] = val
    return out


# ── Threaded fused aggregation (rung 2) ──────────────────────────────────────
# The bincount family (COUNT/SUM/AVG) is embarrassingly parallel over rows: each thread bincounts its
# row-chunk into a private K-vector, then we reduce. np.bincount releases the GIL, so threads run truly
# in parallel. This workload is memory-bandwidth-bound (measured: ~2.8x at 8 threads, plateauing past
# 4), so the win is real but bounded -- it is not an 8x. MIN/MAX stay on the serial path.
import os
from concurrent.futures import ThreadPoolExecutor

PARALLEL_THRESHOLD = 2_000_000     # rows; below this the serial path wins (dispatch + reduce overhead)
_NT = min(8, os.cpu_count() or 4)  # physical-core-ish; bandwidth-bound, so more threads don't help
_POOL = None

def _pool():
    global _POOL
    if _POOL is None: _POOL = ThreadPoolExecutor(max_workers=_NT)
    return _POOL


def _slice(op, lo, hi):
    """Resolve an operand for the row range [lo,hi). op is None (all-zero / absent), ('d', arr) for a
    direct child column arr[lo:hi], or ('g', parent, ptr) for a gather parent[ptr[lo:hi]]. The gather is
    done per-chunk inside the worker thread -- the fusion that avoids materialising the full gathered
    array and re-reading it (the 2.0x -> 2.8x win measured on the bandwidth-bound workload)."""
    if op is None: return None
    if op[0] == 'd': return op[1][lo:hi]
    return op[1][op[2][lo:hi]]


def fused_counts_and_aggs(group_op, K, specs, mask_op, n, n_threads=None):
    """Threaded one-pass group sizes + COUNT/SUM/AVG, gathering each chunk inside the worker.
    group_op: operand giving group codes per chunk, or None (whole-table -> single group 0).
    specs: list of (key, fn, value_op, nullmask_op), fn in COUNT/SUM/AVG.
    mask_op: operand giving a bool WHERE mask per chunk, or None.
    Returns (counts, {key: result-array length K}) identical in shape to the serial path."""
    T = n_threads or _NT
    bnd = np.linspace(0, n, T + 1).astype(np.intp)
    ranges = [(int(bnd[i]), int(bnd[i + 1])) for i in range(T) if int(bnd[i + 1]) > int(bnd[i])]

    def work(lh):
        lo, hi = lh
        gc = _slice(group_op, lo, hi)
        gc = np.zeros(hi - lo, dtype=np.int64) if gc is None else gc.astype(np.int64, copy=False)
        m = _slice(mask_op, lo, hi)
        if m is not None: gc = gc[m]
        size = np.bincount(gc, minlength=K)
        out = {}
        for key, fn, vop, nop in specs:
            nm = _slice(nop, lo, hi)
            if nm is not None and m is not None: nm = nm[m]
            if fn == 'COUNT':
                out[key] = size if nm is None else np.bincount(gc[~nm], minlength=K)
            else:  # SUM / AVG
                v = _slice(vop, lo, hi)
                if m is not None: v = v[m]
                if nm is None: gg, vv = gc, v
                else: keep = ~nm; gg = gc[keep]; vv = v[keep]
                s = np.bincount(gg, weights=vv.astype(np.float64, copy=False), minlength=K)
                c = size if nm is None else np.bincount(gg, minlength=K)
                out[key] = (s, c)
        return size, out

    parts = list(_pool().map(work, ranges))
    counts = np.zeros(K, dtype=np.int64)
    cnt_only = {}; sums = {}; cnts = {}
    for size, out in parts:
        counts += size
        for key, val in out.items():
            if isinstance(val, tuple):
                sv, cv = val; sums[key] = sums.get(key, 0) + sv; cnts[key] = cnts.get(key, 0) + cv
            else:
                cnt_only[key] = cnt_only.get(key, 0) + val

    finals = {}
    for key, fn, vop, nop in specs:
        if fn == 'COUNT':
            finals[key] = cnt_only[key]
        elif fn == 'SUM':
            o = sums[key].astype(object); o[cnts[key] == 0] = None; finals[key] = o
        else:  # AVG
            sv = sums[key]; cv = cnts[key]; o = np.full(K, None, dtype=object)
            nz = cv > 0; o[nz] = sv[nz] / cv[nz]; finals[key] = o
    return counts, finals


def parallel_counts_and_aggs(gcodes, K, specs, n_threads=None):
    """Pre-gathered convenience wrapper over fused_counts_and_aggs (used by tests and small callers).
    specs: (key, fn, v, nullmask) with v/nullmask already mask-applied direct arrays."""
    group_op = ('d', gcodes)
    fspecs = [(key, fn, (('d', v) if v is not None else None), (('d', nm) if nm is not None else None))
              for key, fn, v, nm in specs]
    return fused_counts_and_aggs(group_op, K, fspecs, None, len(gcodes), n_threads)


# ── Rung 3: fused single-pass grouped aggregation (numba) ────────────────────
# numpy does one pass PER reduction (a bincount for SUM, another for COUNT, a minimum.at for MIN, ...),
# which is why aggregate-dense and high-cardinality group-bys lose to DuckDB's single fused scan. This
# kernel does COUNT + per-column SUM/MIN/MAX in ONE pass over the (already gathered + masked) value
# columns, accumulating into dense per-group arrays -- sequential data reads, L2-resident accumulators,
# threaded with per-thread-local accumulators reduced at the end. numba is OPTIONAL: if it is not
# importable, HAS_NUMBA is False and the caller stays on the numpy paths above.
try:
    from numba import njit as _njit, prange as _prange, set_num_threads as _set_nt
    _set_nt(_NT)
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

if HAS_NUMBA:
    # V=1 (the common case -- a single value column, incl. MIN/MAX/AVG of ONE column): flat per-group
    # accumulators, no value matrix, no inner column loop. This is the shape the prototype clocked at
    # ~1.6ms on 6M; the generic V>=2 kernel below is measurably slower (3-D indexing + a length-1 inner
    # loop), so we only fall to it for genuine multi-column aggregates like SUM(a)+SUM(b).
    @_njit(cache=True)
    def _nb_g1_ser(gc, v, K):
        n = gc.shape[0]
        cnt = np.zeros(K, np.int64); s = np.zeros(K, np.float64)
        mn = np.full(K, np.inf); mx = np.full(K, -np.inf)
        for i in range(n):
            c = gc[i]; x = v[i]
            cnt[c] += 1; s[c] += x
            if x < mn[c]: mn[c] = x
            if x > mx[c]: mx[c] = x
        return cnt, s, mn, mx

    @_njit(parallel=True, cache=True)
    def _nb_g1_par(gc, v, K, NT, Kp):
        # Kp pads each thread\'s row past a cache line so the NT accumulator rows never share a line --
        # without it small K (e.g. K=3) ping-pongs lines across cores (measured 24.8ms -> 1.4ms).
        n = gc.shape[0]
        cnt = np.zeros((NT, Kp), np.int64); s = np.zeros((NT, Kp), np.float64)
        mn = np.full((NT, Kp), np.inf); mx = np.full((NT, Kp), -np.inf)
        chunk = (n + NT - 1) // NT
        for t in _prange(NT):
            lo = t * chunk; hi = min(lo + chunk, n)
            for i in range(lo, hi):
                c = gc[i]; x = v[i]
                cnt[t, c] += 1; s[t, c] += x
                if x < mn[t, c]: mn[t, c] = x
                if x > mx[t, c]: mx[t, c] = x
        return cnt, s, mn, mx

    @_njit(cache=True)
    def _nb_g1m_ser(gc, v, m, K):       # V=1 with a WHERE mask fused in (skip rows where ~m, no copy)
        n = gc.shape[0]
        cnt = np.zeros(K, np.int64); s = np.zeros(K, np.float64)
        mn = np.full(K, np.inf); mx = np.full(K, -np.inf)
        for i in range(n):
            if m[i]:
                c = gc[i]; x = v[i]
                cnt[c] += 1; s[c] += x
                if x < mn[c]: mn[c] = x
                if x > mx[c]: mx[c] = x
        return cnt, s, mn, mx

    @_njit(parallel=True, cache=True)
    def _nb_g1m_par(gc, v, m, K, NT, Kp):
        n = gc.shape[0]
        cnt = np.zeros((NT, Kp), np.int64); s = np.zeros((NT, Kp), np.float64)
        mn = np.full((NT, Kp), np.inf); mx = np.full((NT, Kp), -np.inf)
        chunk = (n + NT - 1) // NT
        for t in _prange(NT):
            lo = t * chunk; hi = min(lo + chunk, n)
            for i in range(lo, hi):
                if m[i]:
                    c = gc[i]; x = v[i]
                    cnt[t, c] += 1; s[t, c] += x
                    if x < mn[t, c]: mn[t, c] = x
                    if x > mx[t, c]: mx[t, c] = x
        return cnt, s, mn, mx

    @_njit(cache=True)
    def _nb_g1g_ser(pcodes, ptr, v, K):     # gathered group: c = pcodes[ptr[i]] fused, no materialised gather
        n = ptr.shape[0]
        cnt = np.zeros(K, np.int64); s = np.zeros(K, np.float64)
        mn = np.full(K, np.inf); mx = np.full(K, -np.inf)
        for i in range(n):
            c = pcodes[ptr[i]]; x = v[i]
            cnt[c] += 1; s[c] += x
            if x < mn[c]: mn[c] = x
            if x > mx[c]: mx[c] = x
        return cnt, s, mn, mx

    @_njit(parallel=True, cache=True)
    def _nb_g1g_par(pcodes, ptr, v, K, NT, Kp):
        n = ptr.shape[0]
        cnt = np.zeros((NT, Kp), np.int64); s = np.zeros((NT, Kp), np.float64)
        mn = np.full((NT, Kp), np.inf); mx = np.full((NT, Kp), -np.inf)
        chunk = (n + NT - 1) // NT
        for t in _prange(NT):
            lo = t * chunk; hi = min(lo + chunk, n)
            for i in range(lo, hi):
                c = pcodes[ptr[i]]; x = v[i]
                cnt[t, c] += 1; s[t, c] += x
                if x < mn[t, c]: mn[t, c] = x
                if x > mx[t, c]: mx[t, c] = x
        return cnt, s, mn, mx

    @_njit(cache=True)
    def _nb_g1gm_ser(pcodes, ptr, v, m, K):     # gathered group + fused WHERE mask
        n = ptr.shape[0]
        cnt = np.zeros(K, np.int64); s = np.zeros(K, np.float64)
        mn = np.full(K, np.inf); mx = np.full(K, -np.inf)
        for i in range(n):
            if m[i]:
                c = pcodes[ptr[i]]; x = v[i]
                cnt[c] += 1; s[c] += x
                if x < mn[c]: mn[c] = x
                if x > mx[c]: mx[c] = x
        return cnt, s, mn, mx

    @_njit(parallel=True, cache=True)
    def _nb_g1gm_par(pcodes, ptr, v, m, K, NT, Kp):
        n = ptr.shape[0]
        cnt = np.zeros((NT, Kp), np.int64); s = np.zeros((NT, Kp), np.float64)
        mn = np.full((NT, Kp), np.inf); mx = np.full((NT, Kp), -np.inf)
        chunk = (n + NT - 1) // NT
        for t in _prange(NT):
            lo = t * chunk; hi = min(lo + chunk, n)
            for i in range(lo, hi):
                if m[i]:
                    c = pcodes[ptr[i]]; x = v[i]
                    cnt[t, c] += 1; s[t, c] += x
                    if x < mn[t, c]: mn[t, c] = x
                    if x > mx[t, c]: mx[t, c] = x
        return cnt, s, mn, mx

    # ---- decode-fused: x = base[vcodes[i]] computed in the loop, never materialising the value array ----
    @_njit(cache=True)
    def _nb_g1d_ser(gc, base, vcodes, K):
        n = gc.shape[0]
        cnt = np.zeros(K, np.int64); s = np.zeros(K, np.float64)
        mn = np.full(K, np.inf); mx = np.full(K, -np.inf)
        for i in range(n):
            c = gc[i]; x = base[vcodes[i]]
            cnt[c] += 1; s[c] += x
            if x < mn[c]: mn[c] = x
            if x > mx[c]: mx[c] = x
        return cnt, s, mn, mx

    @_njit(parallel=True, cache=True)
    def _nb_g1d_par(gc, base, vcodes, K, NT, Kp):
        n = gc.shape[0]
        cnt = np.zeros((NT, Kp), np.int64); s = np.zeros((NT, Kp), np.float64)
        mn = np.full((NT, Kp), np.inf); mx = np.full((NT, Kp), -np.inf)
        chunk = (n + NT - 1) // NT
        for t in _prange(NT):
            lo = t * chunk; hi = min(lo + chunk, n)
            for i in range(lo, hi):
                c = gc[i]; x = base[vcodes[i]]
                cnt[t, c] += 1; s[t, c] += x
                if x < mn[t, c]: mn[t, c] = x
                if x > mx[t, c]: mx[t, c] = x
        return cnt, s, mn, mx

    @_njit(cache=True)
    def _nb_g1md_ser(gc, base, vcodes, m, K):
        n = gc.shape[0]
        cnt = np.zeros(K, np.int64); s = np.zeros(K, np.float64)
        mn = np.full(K, np.inf); mx = np.full(K, -np.inf)
        for i in range(n):
            if m[i]:
                c = gc[i]; x = base[vcodes[i]]
                cnt[c] += 1; s[c] += x
                if x < mn[c]: mn[c] = x
                if x > mx[c]: mx[c] = x
        return cnt, s, mn, mx

    @_njit(parallel=True, cache=True)
    def _nb_g1md_par(gc, base, vcodes, m, K, NT, Kp):
        n = gc.shape[0]
        cnt = np.zeros((NT, Kp), np.int64); s = np.zeros((NT, Kp), np.float64)
        mn = np.full((NT, Kp), np.inf); mx = np.full((NT, Kp), -np.inf)
        chunk = (n + NT - 1) // NT
        for t in _prange(NT):
            lo = t * chunk; hi = min(lo + chunk, n)
            for i in range(lo, hi):
                if m[i]:
                    c = gc[i]; x = base[vcodes[i]]
                    cnt[t, c] += 1; s[t, c] += x
                    if x < mn[t, c]: mn[t, c] = x
                    if x > mx[t, c]: mx[t, c] = x
        return cnt, s, mn, mx

    @_njit(cache=True)
    def _nb_g1gd_ser(pcodes, ptr, base, vcodes, K):     # gathered group + decoded value, both fused
        n = ptr.shape[0]
        cnt = np.zeros(K, np.int64); s = np.zeros(K, np.float64)
        mn = np.full(K, np.inf); mx = np.full(K, -np.inf)
        for i in range(n):
            c = pcodes[ptr[i]]; x = base[vcodes[i]]
            cnt[c] += 1; s[c] += x
            if x < mn[c]: mn[c] = x
            if x > mx[c]: mx[c] = x
        return cnt, s, mn, mx

    @_njit(parallel=True, cache=True)
    def _nb_g1gd_par(pcodes, ptr, base, vcodes, K, NT, Kp):
        n = ptr.shape[0]
        cnt = np.zeros((NT, Kp), np.int64); s = np.zeros((NT, Kp), np.float64)
        mn = np.full((NT, Kp), np.inf); mx = np.full((NT, Kp), -np.inf)
        chunk = (n + NT - 1) // NT
        for t in _prange(NT):
            lo = t * chunk; hi = min(lo + chunk, n)
            for i in range(lo, hi):
                c = pcodes[ptr[i]]; x = base[vcodes[i]]
                cnt[t, c] += 1; s[t, c] += x
                if x < mn[t, c]: mn[t, c] = x
                if x > mx[t, c]: mx[t, c] = x
        return cnt, s, mn, mx

    @_njit(cache=True)
    def _nb_g1gmd_ser(pcodes, ptr, base, vcodes, m, K):
        n = ptr.shape[0]
        cnt = np.zeros(K, np.int64); s = np.zeros(K, np.float64)
        mn = np.full(K, np.inf); mx = np.full(K, -np.inf)
        for i in range(n):
            if m[i]:
                c = pcodes[ptr[i]]; x = base[vcodes[i]]
                cnt[c] += 1; s[c] += x
                if x < mn[c]: mn[c] = x
                if x > mx[c]: mx[c] = x
        return cnt, s, mn, mx

    @_njit(parallel=True, cache=True)
    def _nb_g1gmd_par(pcodes, ptr, base, vcodes, m, K, NT, Kp):
        n = ptr.shape[0]
        cnt = np.zeros((NT, Kp), np.int64); s = np.zeros((NT, Kp), np.float64)
        mn = np.full((NT, Kp), np.inf); mx = np.full((NT, Kp), -np.inf)
        chunk = (n + NT - 1) // NT
        for t in _prange(NT):
            lo = t * chunk; hi = min(lo + chunk, n)
            for i in range(lo, hi):
                if m[i]:
                    c = pcodes[ptr[i]]; x = base[vcodes[i]]
                    cnt[t, c] += 1; s[t, c] += x
                    if x < mn[t, c]: mn[t, c] = x
                    if x > mx[t, c]: mx[t, c] = x
        return cnt, s, mn, mx

    @_njit(cache=True)
    def _nb_gN_ser(gc, vmat, K):
        n = gc.shape[0]; V = vmat.shape[1]
        cnt = np.zeros(K, np.int64); sums = np.zeros((V, K), np.float64)
        mins = np.full((V, K), np.inf); maxs = np.full((V, K), -np.inf)
        for i in range(n):
            c = gc[i]; cnt[c] += 1
            for j in range(V):
                v = vmat[i, j]; sums[j, c] += v
                if v < mins[j, c]: mins[j, c] = v
                if v > maxs[j, c]: maxs[j, c] = v
        return cnt, sums, mins, maxs

    @_njit(parallel=True, cache=True)
    def _nb_gN_par(gc, vmat, K, NT, Kp):
        n = gc.shape[0]; V = vmat.shape[1]
        cnt = np.zeros((NT, Kp), np.int64); sums = np.zeros((NT, V, Kp), np.float64)
        mins = np.full((NT, V, Kp), np.inf); maxs = np.full((NT, V, Kp), -np.inf)
        chunk = (n + NT - 1) // NT
        for t in _prange(NT):
            lo = t * chunk; hi = min(lo + chunk, n)
            for i in range(lo, hi):
                c = gc[i]; cnt[t, c] += 1
                for j in range(V):
                    v = vmat[i, j]; sums[t, j, c] += v
                    if v < mins[t, j, c]: mins[t, j, c] = v
                    if v > maxs[t, j, c]: maxs[t, j, c] = v
        return cnt, sums, mins, maxs

def numba_grouped(gc, cols, K, mask=None):
    """One fused pass over group codes + value columns (mask, if any, fused into the V=1 kernel so no
    boolean-index copy is made). cols: list of 1-D numeric arrays. Returns (count[K], sums[V,K],
    mins[V,K], maxs[V,K]). Serial below the parallel threshold."""
    V = len(cols); par = gc.shape[0] >= PARALLEL_THRESHOLD
    Kp = ((K + 7) // 8) * 8 + 8                 # cache-line padding for the per-thread accumulator rows
    if V == 1:
        v = cols[0]
        if mask is not None:
            if par:
                cnt, s, mn, mx = _nb_g1m_par(gc, v, mask, K, _NT, Kp)
                return (cnt[:, :K].sum(0), s[:, :K].sum(0)[None, :],
                        mn[:, :K].min(0)[None, :], mx[:, :K].max(0)[None, :])
            cnt, s, mn, mx = _nb_g1m_ser(gc, v, mask, K)
            return cnt, s[None, :], mn[None, :], mx[None, :]
        if par:
            cnt, s, mn, mx = _nb_g1_par(gc, v, K, _NT, Kp)
            return (cnt[:, :K].sum(0), s[:, :K].sum(0)[None, :],
                    mn[:, :K].min(0)[None, :], mx[:, :K].max(0)[None, :])
        cnt, s, mn, mx = _nb_g1_ser(gc, v, K)
        return cnt, s[None, :], mn[None, :], mx[None, :]
    vmat = np.empty((gc.shape[0], V), np.float64)      # V>=2: mask already applied by caller
    for j, c in enumerate(cols): vmat[:, j] = np.asarray(c).astype(np.float64, copy=False)
    if par:
        cnt, sums, mins, maxs = _nb_gN_par(gc, vmat, K, _NT, Kp)
        return cnt[:, :K].sum(0), sums[:, :, :K].sum(0), mins[:, :, :K].min(0), maxs[:, :, :K].max(0)
    return _nb_gN_ser(gc, vmat, K)


def numba_grouped_g(pcodes, ptr, cols, K, mask=None):
    """Gather-fused V=1: the group code is pcodes[ptr[i]] computed per row, instead of materialising the
    full pcodes[ptr] gather + int64 cast first (measured 11.7ms -> 2.0ms at 6M)."""
    v = cols[0]; par = ptr.shape[0] >= PARALLEL_THRESHOLD
    Kp = ((K + 7) // 8) * 8 + 8
    if mask is not None:
        if par:
            cnt, s, mn, mx = _nb_g1gm_par(pcodes, ptr, v, mask, K, _NT, Kp)
            return (cnt[:, :K].sum(0), s[:, :K].sum(0)[None, :],
                    mn[:, :K].min(0)[None, :], mx[:, :K].max(0)[None, :])
        cnt, s, mn, mx = _nb_g1gm_ser(pcodes, ptr, v, mask, K)
        return cnt, s[None, :], mn[None, :], mx[None, :]
    if par:
        cnt, s, mn, mx = _nb_g1g_par(pcodes, ptr, v, K, _NT, Kp)
        return (cnt[:, :K].sum(0), s[:, :K].sum(0)[None, :],
                mn[:, :K].min(0)[None, :], mx[:, :K].max(0)[None, :])
    cnt, s, mn, mx = _nb_g1g_ser(pcodes, ptr, v, K)
    return cnt, s[None, :], mn[None, :], mx[None, :]


def _reduce1(cnt, s, mn, mx, K):
    return (cnt[:, :K].sum(0), s[:, :K].sum(0)[None, :], mn[:, :K].min(0)[None, :], mx[:, :K].max(0)[None, :])

def numba_grouped_d(gc, base, vcodes, K, mask=None):
    """Direct group code gc[i], value decoded as base[vcodes[i]] -- decode fused, no value array."""
    par = gc.shape[0] >= PARALLEL_THRESHOLD; Kp = ((K + 7) // 8) * 8 + 8
    if mask is not None:
        if par: return _reduce1(*_nb_g1md_par(gc, base, vcodes, mask, K, _NT, Kp), K)
        cnt, s, mn, mx = _nb_g1md_ser(gc, base, vcodes, mask, K); return cnt, s[None, :], mn[None, :], mx[None, :]
    if par: return _reduce1(*_nb_g1d_par(gc, base, vcodes, K, _NT, Kp), K)
    cnt, s, mn, mx = _nb_g1d_ser(gc, base, vcodes, K); return cnt, s[None, :], mn[None, :], mx[None, :]

def numba_grouped_gd(pcodes, ptr, base, vcodes, K, mask=None):
    """Gathered group code pcodes[ptr[i]] AND value base[vcodes[i]] -- both gathers fused into one pass."""
    par = ptr.shape[0] >= PARALLEL_THRESHOLD; Kp = ((K + 7) // 8) * 8 + 8
    if mask is not None:
        if par: return _reduce1(*_nb_g1gmd_par(pcodes, ptr, base, vcodes, mask, K, _NT, Kp), K)
        cnt, s, mn, mx = _nb_g1gmd_ser(pcodes, ptr, base, vcodes, mask, K); return cnt, s[None, :], mn[None, :], mx[None, :]
    if par: return _reduce1(*_nb_g1gd_par(pcodes, ptr, base, vcodes, K, _NT, Kp), K)
    cnt, s, mn, mx = _nb_g1gd_ser(pcodes, ptr, base, vcodes, K); return cnt, s[None, :], mn[None, :], mx[None, :]


def _opkey(op):
    return (op[0], id(op[1])) + ((id(op[2]),) if op[0] in ('g', 'raw') else ())

def fused_numba(group_op, K, specs_all, mask_op, n):
    """Fused-kernel replacement for the numpy agg paths, for numeric non-null operands. Materialises the
    group codes and each distinct value column once (gather + mask), then computes every COUNT/SUM/AVG/
    MIN/MAX in a single pass. Returns (counts, {key: length-K array}) matching the other kernels."""
    m = _slice(mask_op, 0, n)
    if m is not None: m = np.ascontiguousarray(m)

    col_of = {}; vops = []
    for (key, fn, vop, nop) in specs_all:
        if fn in ('SUM', 'AVG', 'MIN', 'MAX'):
            k = _opkey(vop)
            if k not in col_of:
                col_of[k] = len(vops); vops.append(vop)
    V = len(vops); sums = mins = maxs = None
    def _mat(v): return np.ascontiguousarray(_slice(v, 0, n))
    def _gc():
        g = _slice(group_op, 0, n)
        g = np.zeros(n, dtype=np.int64) if g is None else g.astype(np.int64, copy=False)
        return np.ascontiguousarray(g, dtype=np.int64)   # full length -- mask is fused into the kernel
    gathered = group_op is not None and group_op[0] == 'g'

    if V == 1 and vops[0][0] == 'raw':
        # value is a plain dict column: fuse the base[vcodes[i]] decode into the pass (no value array)
        base = np.ascontiguousarray(vops[0][1]); vcodes = np.ascontiguousarray(vops[0][2])
        if gathered:
            pcodes = np.ascontiguousarray(group_op[1]); ptr = np.ascontiguousarray(group_op[2])
            counts, sums, mins, maxs = numba_grouped_gd(pcodes, ptr, base, vcodes, K, mask=m)
        else:
            counts, sums, mins, maxs = numba_grouped_d(_gc(), base, vcodes, K, mask=m)
    elif V == 1 and gathered:
        pcodes = np.ascontiguousarray(group_op[1]); ptr = np.ascontiguousarray(group_op[2])
        counts, sums, mins, maxs = numba_grouped_g(pcodes, ptr, [_mat(vops[0])], K, mask=m)
    elif V == 1:
        counts, sums, mins, maxs = numba_grouped(_gc(), [_mat(vops[0])], K, mask=m)
    elif V == 0:
        gc = _gc(); g = gc if m is None else gc[m]
        counts = np.bincount(g, minlength=K) if g.size else np.zeros(K, np.int64)
    else:                                                # V>=2 (rare): apply mask by indexing
        gc = _gc(); cols = [_mat(v) for v in vops]
        if m is not None: gc = gc[m]; cols = [c[m] for c in cols]
        if gc.size: counts, sums, mins, maxs = numba_grouped(gc, cols, K)
        else: counts = np.zeros(K, np.int64)

    finals = {}
    for (key, fn, vop, nop) in specs_all:
        if fn == 'COUNT':
            finals[key] = counts.copy()
        else:
            j = col_of[_opkey(vop)]; o = np.full(K, None, dtype=object); nz = counts > 0
            if sums is None: finals[key] = o; continue
            if fn == 'SUM':   o = sums[j].astype(object); o[counts == 0] = None
            elif fn == 'AVG': o[nz] = sums[j][nz] / counts[nz]
            elif fn == 'MIN': o[nz] = mins[j][nz]
            else:             o[nz] = maxs[j][nz]
            finals[key] = o
    return counts, finals
