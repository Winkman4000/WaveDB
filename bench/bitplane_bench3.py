"""planes-only verdict bench: the four verbs on vertical-only storage.
1. eq-scan       (snowball)          -- the known 11x
2. window decode (13-tape transpose) -- Jackson's horizontal insert
3. CRUMB GATHER  (the holdout)       -- 738K scattered rows, planes vs bp0
4. top-k early stop                  -- tapes from row 0, stop at k hits
"""
import time
import numpy as np
from numba import njit, prange

N = 20_000_000
BITS = 13
V = 6000
rng = np.random.default_rng(99)


@njit(nogil=True, cache=True)
def _pop(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(nogil=True, parallel=True, cache=True)
def v_transpose_range(planes, bits, lo, hi, out):
    """Jackson's tapes: decode rows [lo,hi) from planes, word-block bit
    matrix un-rotation -- 64 values per plane-word column."""
    w_lo = lo // 64
    w_hi = (hi + 63) // 64
    for wb in prange(w_hi - w_lo):
        w = w_lo + wb
        r0 = w * 64
        colw = np.empty(bits, np.uint64)
        for p in range(bits):
            colw[p] = planes[p, w]
        for j in range(64):
            r = r0 + j
            if r < lo or r >= hi:
                continue
            v = np.int64(0)
            for p in range(bits):
                v = (v << 1) | np.int64((colw[p] >> np.uint64(j)) & np.uint64(1))
            out[r - lo] = v


@njit(nogil=True, parallel=True, cache=True)
def v_gather(planes, bits, rows, out):
    """The holdout: scattered gather from planes -- bits line-pulls/row."""
    n = rows.size
    for i in prange(n):
        r = rows[i]
        w = r // 64
        j = np.uint64(r % 64)
        v = np.int64(0)
        for p in range(bits):
            v = (v << 1) | np.int64((planes[p, w] >> j) & np.uint64(1))
        out[i] = v


@njit(nogil=True, parallel=True, cache=True)
def h_gather(buf, base, bits, rows, out):
    """The incumbent bp0_gather (word-wise horizontal)."""
    n = rows.size
    mask = (np.int64(1) << bits) - 1
    limit = buf.size - 8
    for i in prange(n):
        o = rows[i] * bits
        j = base + (o >> 3)
        if j <= limit:
            w = np.uint64(0)
            for t in range(8):
                w = (w << np.uint64(8)) | np.uint64(buf[j + t])
            out[i] = np.int64(w >> np.uint64(64 - (o & 7) - bits)) & mask
        else:
            out[i] = 0


@njit(nogil=True, cache=True)
def v_topk_walk(planes, bits, flag, k):
    """Early stop from row 0: transpose word blocks, test flag, stop at k
    matches. Returns (matches_found, rows_visited)."""
    nwords = planes.shape[1]
    found = 0
    for w in range(nwords):
        colw = np.empty(bits, np.uint64)
        for p in range(bits):
            colw[p] = planes[p, w]
        for j in range(64):
            v = np.int64(0)
            for p in range(bits):
                v = (v << 1) | np.int64((colw[p] >> np.uint64(j)) & np.uint64(1))
            if flag[v]:
                found += 1
                if found >= k:
                    return found, w * 64 + j + 1
    return found, nwords * 64


@njit(nogil=True, cache=True)
def h_topk_walk(buf, base, bits, n, flag, k):
    mask = (np.int64(1) << bits) - 1
    bo = 0
    p = base
    acc = np.uint64(0)
    nb = 0
    found = 0
    for i in range(n):
        while nb < bits:
            acc = (acc << np.uint64(8)) | np.uint64(buf[p])
            p += 1
            nb += 8
        nb -= bits
        if flag[np.int64(acc >> np.uint64(nb)) & mask]:
            found += 1
            if found >= k:
                return found, i + 1
        acc &= (np.uint64(1) << np.uint64(nb)) - np.uint64(1)
    return found, n


def build_planes(codes, bits):
    n = codes.size
    nwords = (n + 63) // 64
    planes = np.zeros((bits, nwords), np.uint64)
    pad = (-n) % 64
    for p in range(bits):
        bitcol = ((codes >> (bits - 1 - p)) & 1).astype(np.uint8)
        if pad:
            bitcol = np.concatenate([bitcol, np.zeros(pad, np.uint8)])
        planes[p] = np.frombuffer(np.packbits(bitcol, bitorder='little').tobytes(), np.uint64)
    return planes, nwords


def build_horiz(codes, bits):
    n = codes.size
    acc = np.zeros(n * bits, np.uint8)
    for bi in range(bits):
        acc[bi::bits] = (codes >> (bits - 1 - bi)) & 1
    return np.packbits(acc)


def race(fn_v, fn_h, reps=7):
    tv, th = [], []
    for _ in range(reps):
        t0 = time.perf_counter(); rv = fn_v(); tv.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); rh = fn_h(); th.append(time.perf_counter() - t0)
    return min(tv) * 1e3, min(th) * 1e3, rv, rh


if __name__ == '__main__':
    nv = N // 29 + 2
    vals = rng.integers(0, V, nv)
    reps = np.clip(rng.geometric(1 / 29, nv), 1, 400)
    codes = np.repeat(vals, reps)[:N].astype(np.int64)
    planes, nwords = build_planes(codes, BITS)
    hbuf = build_horiz(codes, BITS)

    # 2. WINDOW DECODE: 1M-row staircase window
    lo, hi = 7_000_000, 8_000_000
    ov = np.zeros(hi - lo, np.int64); oh = np.zeros(hi - lo, np.int64)
    v_transpose_range(planes, BITS, lo, hi, ov)
    from bitplane_bench import horiz_count_eq  # noqa: warm import path
    def hwin():
        h_gather(hbuf, 0, BITS, np.arange(lo, hi, dtype=np.int64), oh); return oh[0]
    def vwin():
        v_transpose_range(planes, BITS, lo, hi, ov); return ov[0]
    hwin()
    tv, th, _, _ = race(vwin, hwin)
    print('WINDOW 1M      vert %7.2fms  horiz %7.2fms  (%.1fx)  exact=%s'
          % (tv, th, th / tv, bool((ov == codes[lo:hi]).all() and (oh == codes[lo:hi]).all())), flush=True)

    # 3. THE CRUMB GATHER: 738K scattered rows (density 1/27)
    crumb = np.sort(rng.choice(N, 738_172, replace=False)).astype(np.int64)
    ovg = np.zeros(crumb.size, np.int64); ohg = np.zeros(crumb.size, np.int64)
    v_gather(planes, BITS, crumb, ovg); h_gather(hbuf, 0, BITS, crumb, ohg)
    tv, th, _, _ = race(lambda: (v_gather(planes, BITS, crumb, ovg), ovg[0])[1],
                        lambda: (h_gather(hbuf, 0, BITS, crumb, ohg), ohg[0])[1])
    print('CRUMB 738K     vert %7.2fms  horiz %7.2fms  (vert is %.1fx SLOWER)  exact=%s'
          % (tv, th, tv / th, bool((ovg == codes[crumb]).all() and (ohg == codes[crumb]).all())), flush=True)

    # dense-crumb face: same count, clustered in a 3M-row region
    crumb2 = np.sort(rng.choice(3_000_000, 738_172, replace=False)).astype(np.int64) + 5_000_000
    v_gather(planes, BITS, crumb2, ovg); h_gather(hbuf, 0, BITS, crumb2, ohg)
    tv, th, _, _ = race(lambda: (v_gather(planes, BITS, crumb2, ovg), ovg[0])[1],
                        lambda: (h_gather(hbuf, 0, BITS, crumb2, ohg), ohg[0])[1])
    print('CRUMB-DENSE    vert %7.2fms  horiz %7.2fms  (%.2fx)     exact=%s'
          % (tv, th, tv / th, bool((ovg == codes[crumb2]).all() and (ohg == codes[crumb2]).all())), flush=True)

    # 4. TOP-K EARLY STOP: flag ~1% of V, find first 10K matches
    flag = np.zeros(V, np.bool_)
    flag[rng.choice(V, V // 100, replace=False)] = True
    v_topk_walk(planes, BITS, flag, 10)
    h_topk_walk(hbuf, 0, BITS, N, flag, 10)
    tv, th, rv, rh = race(lambda: v_topk_walk(planes, BITS, flag, 10_000),
                          lambda: h_topk_walk(hbuf, 0, BITS, N, flag, 10_000))
    print('TOPK-10K       vert %7.2fms  horiz %7.2fms  (%.1fx)  visited v=%d h=%d agree=%s'
          % (tv, th, th / tv, rv[1], rh[1], rv[0] == rh[0]), flush=True)
    print('BENCH3 DONE', flush=True)
