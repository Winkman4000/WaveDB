"""bit-plane faces 2: RUNS (the snowball) and RANGE (native prefix prune)."""
import time
import numpy as np
from numba import njit, prange
from bitplane_bench import _popcnt, build, vplane_count_eq, horiz_count_eq, N, BITS, V, BLK

rng = np.random.default_rng(77)


@njit(nogil=True, cache=True)
def _range_block(planes, w0, w1, bits, mlb, mhb):
    ONES = ~np.uint64(0)
    sz = w1 - w0
    gt = np.zeros(sz, np.uint64)
    eqL = np.full(sz, ONES, np.uint64)
    lt = np.zeros(sz, np.uint64)
    eqH = np.full(sz, ONES, np.uint64)
    touched = 0
    for p in range(bits):
        ml = mlb[p]
        mh = mhb[p]
        touched += 1
        any_alive = np.uint64(0)
        for w in range(sz):
            pw = planes[p, w0 + w]
            gt[w] |= eqL[w] & pw & (~ml)         # lo-bit 0: a 1 escapes upward
            eqL[w] &= pw ^ (~ml)                 # stay equal to lo's prefix
            lt[w] |= eqH[w] & (~pw) & mh         # hi-bit 1: a 0 escapes downward
            eqH[w] &= pw ^ (~mh)                 # stay equal to hi's prefix
            any_alive |= (gt[w] | eqL[w]) & (lt[w] | eqH[w])
        if any_alive == np.uint64(0):
            return touched, 0
    cnt = 0
    for w in range(sz):
        cnt += _popcnt((gt[w] | eqL[w]) & (lt[w] | eqH[w]))
    return touched, cnt


@njit(nogil=True, parallel=True, cache=True)
def _vrange_drive(planes, nwords, bits, mlb, mhb, block_planes, counts):
    nblk = counts.size
    for b in prange(nblk):
        w0 = b * BLK
        w1 = min(nwords, w0 + BLK)
        t9, c9 = _range_block(planes, w0, w1, bits, mlb, mhb)
        block_planes[b] = t9
        counts[b] = c9


def vplane_count_range(planes, nwords, bits, mlb, mhb, block_planes):
    """COUNT lo <= code <= hi: branchless four-state walk, early exit per
    block; serial worker per block, prange driver, host-side sum."""
    nblk = (nwords + BLK - 1) // BLK
    counts = np.zeros(nblk, np.int64)
    _vrange_drive(planes, nwords, bits, mlb, mhb, block_planes, counts)
    return int(counts.sum())


@njit(nogil=True, parallel=True, cache=True)
def horiz_count_range(buf, base, bits, n, lo, hi):
    mask = (np.int64(1) << bits) - 1
    CH = 1 << 18
    nch = (n + CH - 1) // CH
    total = 0
    for cix in prange(nch):
        a = cix * CH
        b = min(n, a + CH)
        bo = a * bits
        p = base + (bo >> 3)
        acc = np.uint64(buf[p]) & np.uint64(0xFF >> (bo & 7))
        nb = 8 - (bo & 7)
        p += 1
        cnt = 0
        for i in range(a, b):
            while nb < bits:
                acc = (acc << np.uint64(8)) | np.uint64(buf[p])
                p += 1
                nb += 8
            nb -= bits
            v = np.int64(acc >> np.uint64(nb)) & mask
            if v >= lo and v <= hi:
                cnt += 1
            acc &= (np.uint64(1) << np.uint64(nb)) - np.uint64(1)
        total += cnt
    return total


def face_eq(name, codes, target):
    planes, nwords, hbuf = build(codes, BITS)
    ref = int((codes == target).sum())
    bp = np.zeros((nwords + BLK - 1) // BLK, np.int64)
    vplane_count_eq(planes, nwords, BITS, target, bp)
    horiz_count_eq(hbuf, 0, BITS, N, target)
    tv, th = [], []
    for _ in range(7):
        t0 = time.perf_counter(); cv = vplane_count_eq(planes, nwords, BITS, target, bp); tv.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); ch = horiz_count_eq(hbuf, 0, BITS, N, target); th.append(time.perf_counter() - t0)
    print('%-18s exact=%s | vert %6.2fms horiz %6.2fms (%5.1fx) | planes %.2f/%d (%.0f%% bytes)'
          % (name, cv == ref == ch, min(tv)*1e3, min(th)*1e3, min(th)/min(tv),
             bp.mean(), BITS, 100*bp.mean()/BITS), flush=True)


def face_range(name, codes, lo, hi):
    planes, nwords, hbuf = build(codes, BITS)
    ref = int(((codes >= lo) & (codes <= hi)).sum())
    bp = np.zeros((nwords + BLK - 1) // BLK, np.int64)
    mlb = np.array([~np.uint64(0) if (lo >> (BITS-1-p)) & 1 else np.uint64(0) for p in range(BITS)], np.uint64)
    mhb = np.array([~np.uint64(0) if (hi >> (BITS-1-p)) & 1 else np.uint64(0) for p in range(BITS)], np.uint64)
    vplane_count_range(planes, nwords, BITS, mlb, mhb, bp)
    horiz_count_range(hbuf, 0, BITS, N, lo, hi)
    tv, th = [], []
    for _ in range(7):
        t0 = time.perf_counter(); cv = vplane_count_range(planes, nwords, BITS, mlb, mhb, bp); tv.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); ch = horiz_count_range(hbuf, 0, BITS, N, lo, hi); th.append(time.perf_counter() - t0)
    print('%-18s exact=%s | vert %6.2fms horiz %6.2fms (%5.1fx) | planes %.2f/%d (%.0f%% bytes)'
          % (name, cv == ref == ch, min(tv)*1e3, min(th)*1e3, min(th)/min(tv),
             bp.mean(), BITS, 100*bp.mean()/BITS), flush=True)


if __name__ == '__main__':
    # THE RUNS FACE: mean run ~29, like the realm's enc-10 columns
    nv = N // 29 + 2
    vals = rng.integers(0, V, nv)
    reps = np.clip(rng.geometric(1/29, nv), 1, 400)
    runs = np.repeat(vals, reps)[:N].astype(np.int64)
    face_eq('runs29-eq62', runs, 62)
    # runs with a RARE target (block-killable): value present in ~0.02% rows
    rare = runs.copy()
    face_eq('runs29-eq-rare', rare, int(V - 7))
    # THE RANGE FACES
    uni = rng.integers(0, V, N).astype(np.int64)
    face_range('uni-range-tight', uni, 60, 70)
    face_range('uni-range-top', uni, V - 300, V)
    face_range('runs29-range', runs, 60, 70)
    print('BENCH2 DONE', flush=True)


@njit(nogil=True, cache=True)
def _eq_alist_block(planes, w0, w1, bits, target):
    """Word-granularity pruning: after each plane, only still-alive words
    walk on. Returns (words_touched, count) -- the snowball's own meter."""
    sz = w1 - w0
    m = np.full(sz, ~np.uint64(0), np.uint64)
    alive = np.empty(sz, np.int64)
    for w in range(sz):
        alive[w] = w
    na = sz
    touched = 0
    for p in range(bits):
        tbit = (target >> (bits - 1 - p)) & 1
        touched += na
        k = 0
        if tbit:
            for ai in range(na):
                w = alive[ai]
                m[w] &= planes[p, w0 + w]
                if m[w] != np.uint64(0):
                    alive[k] = w
                    k += 1
        else:
            for ai in range(na):
                w = alive[ai]
                m[w] &= ~planes[p, w0 + w]
                if m[w] != np.uint64(0):
                    alive[k] = w
                    k += 1
        na = k
        if na == 0:
            return touched, 0
    cnt = 0
    for ai in range(na):
        cnt += _popcnt(m[alive[ai]])
    return touched, cnt


@njit(nogil=True, parallel=True, cache=True)
def _valist_drive(planes, nwords, bits, target, wtouched, counts):
    nblk = counts.size
    for b in prange(nblk):
        w0 = b * BLK
        w1 = min(nwords, w0 + BLK)
        t9, c9 = _eq_alist_block(planes, w0, w1, bits, target)
        wtouched[b] = t9
        counts[b] = c9


def vplane_count_eq_alist(planes, nwords, bits, target):
    nblk = (nwords + BLK - 1) // BLK
    wt = np.zeros(nblk, np.int64)
    cn = np.zeros(nblk, np.int64)
    _valist_drive(planes, nwords, bits, target, wt, cn)
    return int(cn.sum()), float(wt.sum()) / (nwords * bits)


def face_eq3(name, codes, target):
    planes, nwords, hbuf = build(codes, BITS)
    ref = int((codes == target).sum())
    bp = np.zeros((nwords + BLK - 1) // BLK, np.int64)
    vplane_count_eq(planes, nwords, BITS, target, bp)
    vplane_count_eq_alist(planes, nwords, BITS, target)
    horiz_count_eq(hbuf, 0, BITS, N, target)
    tv, ta, th = [], [], []
    for _ in range(7):
        t0 = time.perf_counter(); cv = vplane_count_eq(planes, nwords, BITS, target, bp); tv.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); ca, frac = vplane_count_eq_alist(planes, nwords, BITS, target); ta.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); ch = horiz_count_eq(hbuf, 0, BITS, N, target); th.append(time.perf_counter() - t0)
    print('%-18s exact=%s | horiz %6.2f  vert-plain %5.2f  VERT-SNOWBALL %5.2fms (%5.1fx vs horiz) | words touched %4.1f%%'
          % (name, cv == ca == ref == ch, min(th)*1e3, min(tv)*1e3, min(ta)*1e3,
             min(th)/min(ta), 100*frac), flush=True)


if __name__ == '__main__':
    pass
