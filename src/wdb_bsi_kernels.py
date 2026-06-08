"""Numba bitmap-walk kernels: accumulate aggregates directly over the set bits
of a packed bitmap (np.packbits layout, MSB-first) -- no unpackbits, no position
array. Replaces unpackbits+nonzero (the ~7.4ms gather bottleneck) for the
measure step. Pad bits beyond N are 0 (guaranteed by wdb_bsi ops); the r<N
guard on the final partial byte is kept for safety.

Byte-walk: skip whole zero bytes (`if by`), expand only set bytes. At low
selectivity most bytes are zero so the outer skip dominates.
"""
import numpy as np
import numba


@numba.njit(cache=True, nogil=True)
def bw_count(packed, N):
    c = 0
    for i in range(packed.shape[0]):
        by = packed[i]
        if by:
            base = i << 3
            for j in range(8):
                if (by >> (7 - j)) & 1 and base + j < N:
                    c += 1
    return c


@numba.njit(cache=True, nogil=True)
def bw_sum1(packed, a, N):
    s = 0.0
    for i in range(packed.shape[0]):
        by = packed[i]
        if by:
            base = i << 3
            for j in range(8):
                if (by >> (7 - j)) & 1:
                    r = base + j
                    if r < N:
                        s += a[r]
    return s


@numba.njit(cache=True, nogil=True)
def bw_sum2(packed, a, b, N):
    """Sum of a[r]*b[r] over set bits (e.g. SUM(extendedprice*discount))."""
    s = 0.0
    for i in range(packed.shape[0]):
        by = packed[i]
        if by:
            base = i << 3
            for j in range(8):
                if (by >> (7 - j)) & 1:
                    r = base + j
                    if r < N:
                        s += a[r] * b[r]
    return s


@numba.njit(cache=True, nogil=True)
def bw_group_sum1(packed, a, gcodes, nb, N):
    """Per-group sum of a[r] over set bits, bucketed by gcodes[r] (0..nb-1)."""
    out = np.zeros(nb, np.float64)
    for i in range(packed.shape[0]):
        by = packed[i]
        if by:
            base = i << 3
            for j in range(8):
                if (by >> (7 - j)) & 1:
                    r = base + j
                    if r < N:
                        out[gcodes[r]] += a[r]
    return out


@numba.njit(cache=True, nogil=True)
def bw_group_count(packed, gcodes, nb, N):
    """Per-group count over set bits, bucketed by gcodes[r]."""
    out = np.zeros(nb, np.int64)
    for i in range(packed.shape[0]):
        by = packed[i]
        if by:
            base = i << 3
            for j in range(8):
                if (by >> (7 - j)) & 1:
                    r = base + j
                    if r < N:
                        out[gcodes[r]] += 1
    return out


def warmup():
    """Trigger JIT compilation on tiny inputs (idempotent)."""
    p = np.zeros(2, np.uint8); a = np.zeros(16, np.float64); g = np.zeros(16, np.int64)
    bw_count(p, 16); bw_sum1(p, a, 16); bw_sum2(p, a, a, 16)
    bw_group_sum1(p, a, g, 1, 16); bw_group_count(p, g, 1, 16)
