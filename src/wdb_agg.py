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
