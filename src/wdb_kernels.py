"""Compiled aggregation kernels (numba soft-dependency).

kway_topk: single-pass loser-tree merge of K sorted (key, count) runs with on-the-fly equal-key
accumulation feeding a running top-K min-heap -- the merged stream is never materialized. This is
the counting core of the disk-only GROUP BY path: measured on ClickBench Q16 (100M rows, 13.2M
non-norm keys in 14 runs) at ~0.21 s, completing a 0.70 s total vs DuckDB's 0.79 s best.

top10_i32: single-pass top-K over a dense int32 count table (the norm lane's per-code counts).

Without numba both fall back to numpy (pairwise sorted-merge tournament / argpartition) --
correct, ~2-4x slower.
"""
import numpy as np

try:
    from numba import njit
    import numba
    HAVE_NUMBA = True
except Exception:                                     # pragma: no cover
    HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f):
            return f
        return deco


@njit(nogil=True, cache=True)
def _kway_topk_nb(keys, vals, offs, K):
    nr = offs.size - 1
    cur = offs[:nr].copy()
    INF = np.int64(0x7FFFFFFFFFFFFFFF)
    P = 1
    while P < nr:
        P *= 2
    lk = np.full(P, INF, np.int64)
    for r in range(nr):
        lk[r] = keys[cur[r]] if cur[r] < offs[r + 1] else INF
    tree = np.full(2 * P, -1, np.int32)
    win = np.empty(2 * P, np.int32)
    for i in range(P):
        win[P + i] = i if i < nr else -1
    for i in range(P - 1, 0, -1):
        a, b = win[2 * i], win[2 * i + 1]
        ka = lk[a] if a >= 0 else INF
        kb = lk[b] if b >= 0 else INF
        if ka <= kb:
            win[i] = a; tree[i] = b
        else:
            win[i] = b; tree[i] = a
    topc = np.zeros(K, np.int64); topk = np.zeros(K, np.int64)
    curkey = np.int64(-1); acc = np.int64(0)
    w = win[1]
    while w >= 0 and lk[w] != INF:
        r = w
        k = lk[r]
        if k != curkey:
            if acc > topc[0]:
                topc[0] = acc; topk[0] = curkey
                j = 0
                while True:
                    l = 2 * j + 1; rt = 2 * j + 2; m = j
                    if l < K and topc[l] < topc[m]: m = l
                    if rt < K and topc[rt] < topc[m]: m = rt
                    if m == j: break
                    topc[j], topc[m] = topc[m], topc[j]
                    topk[j], topk[m] = topk[m], topk[j]
                    j = m
            curkey = k; acc = vals[cur[r]]
        else:
            acc += vals[cur[r]]
        cur[r] += 1
        lk[r] = keys[cur[r]] if cur[r] < offs[r + 1] else INF
        i = (P + r) >> 1
        wcur = r
        while i >= 1:
            lo = tree[i]
            klo = lk[lo] if lo >= 0 else INF
            if klo < (lk[wcur] if wcur >= 0 else INF):
                tree[i] = wcur; wcur = lo
            i >>= 1
        w = wcur
    if acc > topc[0]:
        topc[0] = acc; topk[0] = curkey
    return topc, topk


@njit(nogil=True, cache=True)
def top10_i32(tab, K):
    topc = np.zeros(K, np.int64); topk = np.zeros(K, np.int64)
    for i in range(tab.size):
        v = np.int64(tab[i])
        if v > topc[0]:
            topc[0] = v; topk[0] = i
            j = 0
            while True:
                l = 2 * j + 1; rt = 2 * j + 2; m = j
                if l < K and topc[l] < topc[m]: m = l
                if rt < K and topc[rt] < topc[m]: m = rt
                if m == j: break
                topc[j], topc[m] = topc[m], topc[j]
                topk[j], topk[m] = topk[m], topk[j]
                j = m
    return topc, topk


def kway_topk(keys, vals, offs, K):
    """Top-K (count, key) pairs from K sorted runs. Returns (counts, keys) unsorted heaps;
    zero-count slots are empty. Numba path streams; numpy fallback merges then partitions."""
    if HAVE_NUMBA:
        return _kway_topk_nb(keys, vals, offs, np.int64(K))
    parts = [(keys[offs[i]:offs[i + 1]], vals[offs[i]:offs[i + 1]])
             for i in range(offs.size - 1) if offs[i + 1] > offs[i]]
    while len(parts) > 1:
        nxt = []
        for i in range(0, len(parts) - 1, 2):
            k = np.concatenate([parts[i][0], parts[i + 1][0]])
            v = np.concatenate([parts[i][1], parts[i + 1][1]])
            o = np.argsort(k, kind='stable'); k = k[o]; v = v[o]
            new = np.ones(k.size, bool); new[1:] = k[1:] != k[:-1]
            idx = np.nonzero(new)[0]
            nxt.append((k[idx], np.add.reduceat(v, idx)))
        if len(parts) % 2:
            nxt.append(parts[-1])
        parts = nxt
    g, c = parts[0] if parts else (np.empty(0, np.int64), np.empty(0, np.int64))
    kk = min(K, g.size)
    if kk == 0:
        return np.zeros(K, np.int64), np.zeros(K, np.int64)
    ti = np.argpartition(-c, kk - 1)[:kk]
    tc = np.zeros(K, np.int64); tk = np.zeros(K, np.int64)
    tc[:kk] = c[ti]; tk[:kk] = g[ti]
    return tc, tk


@njit(nogil=True, parallel=True, cache=True)
def _part_scatter_nb(codes, K):
    """Stable parallel counting scatter: perm such that codes[perm] is grouped by code with
    original order preserved inside each group -- the fused motion, compiled. Two passes:
    per-chunk histograms -> exclusive global/chunk offsets -> stable scatter."""
    N = codes.size
    T = numba.get_num_threads()
    chunk = (N + T - 1) // T
    hist = np.zeros((T, K), np.int64)
    for t in numba.prange(T):
        lo = t * chunk
        hi = min(N, lo + chunk)
        for i in range(lo, hi):
            hist[t, codes[i]] += 1
    offs = np.zeros((T, K), np.int64)
    run = np.int64(0)
    for k in range(K):
        for t in range(T):
            offs[t, k] = run
            run += hist[t, k]
    perm = np.empty(N, np.int64)
    for t in numba.prange(T):
        lo = t * chunk
        hi = min(N, lo + chunk)
        cur = offs[t].copy()
        for i in range(lo, hi):
            c = codes[i]
            perm[cur[c]] = i
            cur[c] += 1
    return perm


def part_scatter(codes, K):
    """Stable grouping permutation by small-int key; numba parallel, numpy fallback."""
    if HAVE_NUMBA:
        return _part_scatter_nb(codes, np.int64(K))
    return np.argsort(codes, kind='stable')


@njit(nogil=True, parallel=True, cache=True)
def _seg_cumminmax_nb(vals, lane_start, lane_of, do_min):
    """Segmented cumulative min/max: running extreme within each lane (parallel over lanes)."""
    out = np.empty(vals.size, vals.dtype)
    L = lane_start.size
    for li in numba.prange(L):
        lo = lane_start[li]
        hi = lane_start[li + 1] if li + 1 < L else vals.size
        cur = vals[lo]
        out[lo] = cur
        for i in range(lo + 1, hi):
            v = vals[i]
            if (v < cur) == do_min and v != cur:
                cur = v
            elif do_min and v < cur:
                cur = v
            elif not do_min and v > cur:
                cur = v
            out[i] = cur
    return out


def seg_cummin(vals, lane_start):
    if HAVE_NUMBA:
        return _seg_cumminmax_nb(vals, lane_start, None, True)
    out = vals.copy()
    for li in range(lane_start.size):
        lo = lane_start[li]
        hi = lane_start[li + 1] if li + 1 < lane_start.size else vals.size
        out[lo:hi] = np.minimum.accumulate(vals[lo:hi])
    return out


def seg_cummax(vals, lane_start):
    if HAVE_NUMBA:
        return _seg_cumminmax_nb(vals, lane_start, None, False)
    out = vals.copy()
    for li in range(lane_start.size):
        lo = lane_start[li]
        hi = lane_start[li + 1] if li + 1 < lane_start.size else vals.size
        out[lo:hi] = np.maximum.accumulate(vals[lo:hi])
    return out


@njit(nogil=True, parallel=True, cache=True)
def _seg_slidext_nb(vals, lane_start, k, do_min):
    """Sliding window min/max over ROWS k PRECEDING .. CURRENT ROW, per lane: the classic
    monotonic deque, one deque per lane, parallel over lanes."""
    N = vals.size
    out = np.empty(N, vals.dtype)
    L = lane_start.size
    for li in numba.prange(L):
        lo = lane_start[li]
        hi = lane_start[li + 1] if li + 1 < L else N
        m = hi - lo
        dq = np.empty(m, np.int64)          # indices, front..back monotonic
        head = 0; tail = 0                   # deque in dq[head:tail]
        for i in range(lo, hi):
            lo_w = i - k
            while head < tail and dq[head] < lo_w:
                head += 1
            v = vals[i]
            if do_min:
                while head < tail and vals[dq[tail - 1]] >= v:
                    tail -= 1
            else:
                while head < tail and vals[dq[tail - 1]] <= v:
                    tail -= 1
            dq[tail] = i; tail += 1
            out[i] = vals[dq[head]]
    return out


def seg_slidmin(vals, lane_start, k):
    if HAVE_NUMBA:
        return _seg_slidext_nb(vals, lane_start, np.int64(k), True)
    out = vals.copy()
    for li in range(lane_start.size):
        lo = lane_start[li]
        hi = lane_start[li + 1] if li + 1 < lane_start.size else vals.size
        for i in range(lo, hi):
            out[i] = vals[max(lo, i - k):i + 1].min()
    return out


def seg_slidmax(vals, lane_start, k):
    if HAVE_NUMBA:
        return _seg_slidext_nb(vals, lane_start, np.int64(k), False)
    out = vals.copy()
    for li in range(lane_start.size):
        lo = lane_start[li]
        hi = lane_start[li + 1] if li + 1 < lane_start.size else vals.size
        for i in range(lo, hi):
            out[i] = vals[max(lo, i - k):i + 1].max()
    return out


def warm():
    """JIT-compile the kernels (call from prewarm; ~1 s once, cached on disk after)."""
    kway_topk(np.array([1, 2], np.int64), np.array([1, 1], np.int64),
              np.array([0, 1, 2], np.int64), 4)
    if HAVE_NUMBA:
        top10_i32(np.array([1, 2], np.int32), 4)
        part_scatter(np.array([1, 0, 1], np.int64), 2)
