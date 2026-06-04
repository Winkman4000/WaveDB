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
