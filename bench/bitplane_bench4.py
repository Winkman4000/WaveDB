"""POD-SCALE planes-only verdict: N=100M, DRAM regime, proper 64x64
bit-matrix transpose for the tapes. Faces: 13-bit runs (CounterID-class)
and 25-bit zipf (URL-class)."""
import time
import numpy as np
from numba import njit, prange

N = 100_000_000
rng = np.random.default_rng(7)


@njit(nogil=True, cache=True)
def _pop(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(nogil=True, cache=True)
def _t64(a):
    """Hacker's Delight 64x64 bit-matrix transpose, in place."""
    j = np.uint64(32)
    m = np.uint64(0x00000000FFFFFFFF)
    while j != np.uint64(0):
        k = 0
        while k < 64:
            for i in range(k, k + int(j)):
                t = (a[i] ^ (a[i + int(j)] >> j)) & m
                a[i] ^= t
                a[i + int(j)] ^= (t << j)
            k = (k + int(j) * 2)
        j >>= np.uint64(1)
        m ^= (m << j)
    return a


@njit(nogil=True, parallel=True, cache=True)
def v_eq_alist(planes, nwords, bits, target, wt, cn, BLK):
    nblk = cn.size
    for b in prange(nblk):
        w0 = b * BLK
        w1 = min(nwords, w0 + BLK)
        sz = w1 - w0
        m = np.full(sz, ~np.uint64(0), np.uint64)
        alive = np.empty(sz, np.int64)
        for w in range(sz):
            alive[w] = w
        na = sz
        touched = 0
        dead = False
        for p in range(bits):
            tbit = (target >> (bits - 1 - p)) & 1
            touched += na
            k = 0
            if tbit:
                for ai in range(na):
                    w = alive[ai]
                    m[w] &= planes[p, w0 + w]
                    if m[w] != np.uint64(0):
                        alive[k] = w; k += 1
            else:
                for ai in range(na):
                    w = alive[ai]
                    m[w] &= ~planes[p, w0 + w]
                    if m[w] != np.uint64(0):
                        alive[k] = w; k += 1
            na = k
            if na == 0:
                dead = True
                break
        wt[b] = touched
        c = 0
        if not dead:
            for ai in range(na):
                c += _pop(m[alive[ai]])
        cn[b] = c


@njit(nogil=True, parallel=True, cache=True)
def h_scan_eq(buf, base, bits, n, target):
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
            if (np.int64(acc >> np.uint64(nb)) & mask) == target:
                cnt += 1
            acc &= (np.uint64(1) << np.uint64(nb)) - np.uint64(1)
        total += cnt
    return total


@njit(nogil=True, parallel=True, cache=True)
def v_window_t64(planes, bits, lo, hi, out):
    """Tapes with the REAL transpose: 64 rows per un-rotation."""
    w_lo = lo // 64
    w_hi = (hi + 63) // 64
    for wb in prange(w_hi - w_lo):
        w = w_lo + wb
        a = np.zeros(64, np.uint64)
        for p in range(bits):
            a[p] = planes[p, w]
        _t64(a)
        r0 = w * 64
        sh = np.uint64(64 - bits)
        for j in range(64):
            r = r0 + j
            if lo <= r < hi:
                out[r - lo] = np.int64(a[63 - j] >> sh)
    return 0


@njit(nogil=True, parallel=True, cache=True)
def v_gather(planes, bits, rows, out):
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
def v_topk_t64(planes, bits, flag, k):
    nwords = planes.shape[1]
    found = 0
    a = np.zeros(64, np.uint64)
    sh = np.uint64(64 - bits)
    for w in range(nwords):
        for p in range(bits):
            a[p] = planes[p, w]
        for p in range(bits, 64):
            a[p] = np.uint64(0)
        _t64(a)
        for j in range(64):
            if flag[np.int64(a[63 - j] >> sh)]:
                found += 1
                if found >= k:
                    return found, w * 64 + j + 1
    return found, nwords * 64


@njit(nogil=True, cache=True)
def h_topk(buf, base, bits, n, flag, k):
    mask = (np.int64(1) << bits) - 1
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
    acc = np.zeros(codes.size * bits, np.uint8)
    for bi in range(bits):
        acc[bi::bits] = (codes >> (bits - 1 - bi)) & 1
    return np.packbits(acc)


def race(fv, fh, reps=5):
    tv, th = [], []
    for _ in range(reps):
        t0 = time.perf_counter(); rv = fv(); tv.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); rh = fh(); th.append(time.perf_counter() - t0)
    return min(tv) * 1e3, min(th) * 1e3, rv, rh


def suite(tag, codes, bits, V, crumb):
    t0 = time.perf_counter()
    planes, nwords = build_planes(codes, bits)
    hbuf = build_horiz(codes, bits)
    print('%s built planes+horiz in %.0fs (each %.0fMB)' % (tag, time.perf_counter() - t0, planes.nbytes / 1e6), flush=True)
    BLK = 4096
    nblk = (nwords + BLK - 1) // BLK
    wt = np.zeros(nblk, np.int64); cn = np.zeros(nblk, np.int64)
    tgt = int(codes[12345])
    v_eq_alist(planes, nwords, bits, tgt, wt, cn, BLK)
    h_scan_eq(hbuf, 0, bits, N, tgt)
    tv, th, _, _ = race(lambda: (v_eq_alist(planes, nwords, bits, tgt, wt, cn, BLK), int(cn.sum()))[1],
                        lambda: h_scan_eq(hbuf, 0, bits, N, tgt))
    ref = int((codes == tgt).sum())
    print('%s SCAN-EQ    vert %8.2f horiz %8.2f (%5.1fx) words %4.1f%% exact=%s'
          % (tag, tv, th, th / tv, 100 * wt.sum() / (nwords * bits), int(cn.sum()) == ref), flush=True)
    lo, hi = N // 3, N // 3 + 2_000_000
    ov = np.zeros(hi - lo, np.int64); oh = np.zeros(hi - lo, np.int64)
    v_window_t64(planes, bits, lo, hi, ov)
    h_gather(hbuf, 0, bits, np.arange(lo, hi, dtype=np.int64), oh)
    tv, th, _, _ = race(lambda: v_window_t64(planes, bits, lo, hi, ov),
                        lambda: (h_gather(hbuf, 0, bits, np.arange(lo, hi, dtype=np.int64), oh), 0)[1])
    print('%s WINDOW-2M  vert %8.2f horiz %8.2f (%5.1fx) exact=%s'
          % (tag, tv, th, th / tv, bool((ov == codes[lo:hi]).all() and (oh == codes[lo:hi]).all())), flush=True)
    ovg = np.zeros(crumb.size, np.int64); ohg = np.zeros(crumb.size, np.int64)
    v_gather(planes, bits, crumb, ovg); h_gather(hbuf, 0, bits, crumb, ohg)
    tv, th, _, _ = race(lambda: (v_gather(planes, bits, crumb, ovg), 0)[1],
                        lambda: (h_gather(hbuf, 0, bits, crumb, ohg), 0)[1])
    print('%s CRUMB-738K vert %8.2f horiz %8.2f (vert %.2fx slower) exact=%s'
          % (tag, tv, th, tv / th, bool((ovg == codes[crumb]).all() and (ohg == codes[crumb]).all())), flush=True)
    flag = np.zeros(V, np.bool_)
    flag[rng.choice(V, max(2, V // 100), replace=False)] = True
    v_topk_t64(planes, bits, flag, 5)
    h_topk(hbuf, 0, bits, N, flag, 5)
    tv, th, rv, rh = race(lambda: v_topk_t64(planes, bits, flag, 10_000),
                          lambda: h_topk(hbuf, 0, bits, N, flag, 10_000), reps=3)
    print('%s TOPK-10K   vert %8.2f horiz %8.2f (%5.1fx) visited v=%d h=%d agree=%s'
          % (tag, tv, th, th / tv, rv[1], rh[1], rv[0] == rh[0]), flush=True)
    del planes, hbuf


if __name__ == '__main__':
    nv = N // 29 + 2
    vals = rng.integers(0, 6000, nv)
    reps = np.clip(rng.geometric(1 / 29, nv), 1, 400)
    c13 = np.repeat(vals, reps)[:N].astype(np.int64)
    if c13.size < N:
        c13 = np.concatenate([c13, c13[:N - c13.size]])
    crumb = np.sort(rng.choice(N, 738_172, replace=False)).astype(np.int64)
    suite('CID-13bit', c13, 13, 6000, crumb)
    del c13
    c25 = (np.clip(rng.zipf(1.3, N), 1, 20_000_000) - 1).astype(np.int64)
    suite('URL-25bit', c25, 25, 20_000_000, crumb)
    print('BENCH4 DONE', flush=True)
