"""
wdb_radix — key-partitioned (radix) aggregation for high-cardinality single-key GROUP BY.

Why this exists: grouped_multi accumulates s[t, g] per row into per-thread (NT, K) arrays.
When K is large enough that NT*K*8 exceeds L3, those scattered writes fall off cache and the
group-by collapses (~6x slower on a random-order key). Sorted/cluster keys avoid this (sequential
accumulator walk) and already take the slice path; the hole is RANDOM-order high-card keys.

Fix: radix-partition the rows by the top bits of the key into P buckets, so each bucket's key
sub-range fits cache, then aggregate each bucket cache-resident. The partition is parallel (per-
thread histograms -> prefix sum -> disjoint scatter); the bucket pass writes disjoint K-slices so
it is race-free without atomics.

Scope: ONE non-gathered key, aggregates = COUNT (E=0) and/or a single SUM over a bare decoded slot
(E=1), no MIN/MAX, no mask, no predicate. E>=2 falls through to grouped_multi. The caller (wdb_join)
gates on this shape + cardinality. Group results are bit-identical to grouped_multi.
"""
import os
import numpy as np
from numba import njit, prange, get_num_threads

_P = 512   # partition fan-out: K/P keys per bucket stays inside L2


@njit(parallel=True, cache=True)
def _radix0(key, S, P, NT, K):                      # COUNT only
    N = key.size
    chunk = (N + NT - 1) // NT
    hist = np.zeros((NT, P), np.int64)
    for t in prange(NT):
        lo = t * chunk; hi = lo + chunk
        if hi > N: hi = N
        h = hist[t]
        for i in range(lo, hi): h[key[i] >> S] += 1
    base = np.zeros(P + 1, np.int64)
    for b in range(P):
        s = 0
        for t in range(NT): s += hist[t, b]
        base[b + 1] = base[b] + s
    starts = np.zeros((NT, P), np.int64)
    for b in range(P):
        acc = base[b]
        for t in range(NT):
            starts[t, b] = acc; acc += hist[t, b]
    kp = np.empty(N, key.dtype)
    for t in prange(NT):
        lo = t * chunk; hi = lo + chunk
        if hi > N: hi = N
        cur = starts[t].copy()
        for i in range(lo, hi):
            b = key[i] >> S; p = cur[b]; kp[p] = key[i]; cur[b] = p + 1
    counts = np.zeros(K, np.int64)
    for b in prange(P):
        for j in range(base[b], base[b + 1]): counts[kp[j]] += 1
    return counts


@njit(parallel=True, cache=True)
def _radix1(key, dvals, mcodes, S, P, NT, K):       # COUNT + SUM(dvals[mcodes]) -- decode folded in
    N = key.size
    chunk = (N + NT - 1) // NT
    hist = np.zeros((NT, P), np.int64)
    for t in prange(NT):
        lo = t * chunk; hi = lo + chunk
        if hi > N: hi = N
        h = hist[t]
        for i in range(lo, hi): h[key[i] >> S] += 1
    boff = np.zeros(P + 1, np.int64)
    for b in range(P):
        s = 0
        for t in range(NT): s += hist[t, b]
        boff[b + 1] = boff[b] + s
    starts = np.zeros((NT, P), np.int64)
    for b in range(P):
        acc = boff[b]
        for t in range(NT):
            starts[t, b] = acc; acc += hist[t, b]
    kp = np.empty(N, key.dtype)
    pp = np.empty(N, np.float64)
    for t in prange(NT):
        lo = t * chunk; hi = lo + chunk
        if hi > N: hi = N
        cur = starts[t].copy()
        for i in range(lo, hi):
            b = key[i] >> S; p = cur[b]
            kp[p] = key[i]; pp[p] = dvals[mcodes[i]]; cur[b] = p + 1   # gather measure value in-pass
    counts = np.zeros(K, np.int64)
    sums = np.zeros(K, np.float64)
    for b in prange(P):
        for j in range(boff[b], boff[b + 1]):
            g = kp[j]; counts[g] += 1; sums[g] += pp[j]
    return counts, sums


def _fanout(K):
    P = _P
    if K <= P: P = 1 << max(0, int(K).bit_length() - 1)
    if P < 1: P = 1
    S = 0
    while (K >> S) >= P: S += 1
    return S, P


def radix_grouped(key, K, measure, NT=None):
    """key: 1-D integer array of group ids in [0, K). measure: None for COUNT-only, or a tuple
    (dvals, mcodes) where the SUM value of row i is dvals[mcodes[i]] (dvals = typed dict as float64).
    Returns (counts[K], [sum[K]] or []) in group order. Caller gates shape/cardinality."""
    if NT is None: NT = get_num_threads()
    key = np.ascontiguousarray(key)
    S, P = _fanout(K)
    if measure is None:
        return _radix0(key, S, P, NT, K), []
    dvals, mcodes = measure
    dvals = np.ascontiguousarray(dvals, dtype=np.float64)
    mcodes = np.ascontiguousarray(mcodes)
    counts, sums = _radix1(key, dvals, mcodes, S, P, NT, K)
    return counts, [sums]


def _detect_l3():
    """Last-level cache size in bytes, read from sysfs; falls back to 96 MiB (this box's V-cache)."""
    try:
        base = '/sys/devices/system/cpu/cpu0/cache'
        best = 0
        for d in os.listdir(base):
            p = os.path.join(base, d)
            try:
                if open(os.path.join(p, 'level')).read().strip() != '3': continue
                s = open(os.path.join(p, 'size')).read().strip()
                mult = 1
                if s and s[-1] in 'Kk': mult, s = 1024, s[:-1]
                elif s and s[-1] in 'Mm': mult, s = 1024 * 1024, s[:-1]
                best = max(best, int(s) * mult)
            except Exception:
                continue
        if best: return best
    except Exception:
        pass
    return 96 * 1024 * 1024

L3_BYTES = _detect_l3()


def should_use(K, nt, l3_bytes=None):
    """Fire only when the replicated per-thread accumulators (counts + one sum, nt*K*16 bytes) would
    overflow the last-level cache — below that, grouped_multi is cache-resident and already optimal."""
    if l3_bytes is None: l3_bytes = L3_BYTES
    return K * nt * 16 > l3_bytes
