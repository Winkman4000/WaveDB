"""bit-plane (vertical) scan vs horizontal word-wise scan -- Jackson's
progressive pruning, benched before any encoder ink.

Vertical layout: `bits` planes of N bits each (plane p = bit p of every
row, packed 64 rows/u64). Scan for code T: running match-word ANDs
~(plane ^ broadcast(bit_p(T))) plane by plane; a block whose match goes
all-zero SKIPS every remaining plane -- the dead-word snowball.
"""
import time
import numpy as np
from numba import njit, prange

N = 20_000_000
BITS = 13
V = 6000
BLK = 4096                                       # u64 words per pruning block
rng = np.random.default_rng(42)


@njit(nogil=True, cache=True)
def _popcnt(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(nogil=True, parallel=True, cache=True)
def vplane_count_eq(planes, nwords, bits, target, block_planes):
    """COUNT rows == target over vertical planes with per-block early exit.
    block_planes[b] records how many planes block b actually touched."""
    nblk = (nwords + BLK - 1) // BLK
    total = 0
    for b in prange(nblk):
        w0 = b * BLK
        w1 = min(nwords, w0 + BLK)
        # match starts all-ones; AND in agreement per plane, MSB first
        alive = True
        touched = 0
        m = np.full(w1 - w0, ~np.uint64(0), np.uint64)
        for p in range(bits):
            tbit = (target >> (bits - 1 - p)) & 1
            touched += 1
            any_alive = np.uint64(0)
            if tbit:
                for w in range(w0, w1):
                    m[w - w0] &= planes[p, w]
                    any_alive |= m[w - w0]
            else:
                for w in range(w0, w1):
                    m[w - w0] &= ~planes[p, w]
                    any_alive |= m[w - w0]
            if any_alive == np.uint64(0):
                alive = False
                break
        block_planes[b] = touched
        cnt = 0
        if alive:
            for w in range(w1 - w0):
                cnt += _popcnt(m[w])
        total += cnt
    return total


@njit(nogil=True, parallel=True, cache=True)
def horiz_count_eq(buf, base, bits, n, target):
    """The reigning atom: tile-reader horizontal scan, count == target."""
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


def build(codes, bits):
    n = codes.size
    nwords = (n + 63) // 64
    planes = np.zeros((bits, nwords), np.uint64)
    for p in range(bits):
        bitcol = ((codes >> (bits - 1 - p)) & 1).astype(np.uint8)
        planes[p] = np.packbits(bitcol, bitorder='little').view(np.uint64)[:nwords] \
            if bitcol.size % 64 == 0 else np.frombuffer(
                np.packbits(np.pad(bitcol, (0, (-n) % 64)), bitorder='little').tobytes(),
                np.uint64)
    # horizontal MSB-first pack for the incumbent
    acc = np.zeros(((n * bits + 7) // 8) * 8, np.uint8)
    for bi in range(bits):
        acc[bi::bits][:n] = (codes >> (bits - 1 - bi)) & 1
    hbuf = np.packbits(acc[:((n * bits + 7) // 8) * 8])
    return planes, nwords, hbuf


def bench(name, codes):
    planes, nwords, hbuf = build(codes, BITS)
    ref = int((codes == 62).sum())
    bp = np.zeros((nwords + BLK - 1) // BLK, np.int64)
    # warm both
    vplane_count_eq(planes, nwords, BITS, 62, bp)
    horiz_count_eq(hbuf, 0, BITS, N, 62)
    tv = []
    th = []
    for _ in range(7):
        t0 = time.perf_counter(); cv = vplane_count_eq(planes, nwords, BITS, 62, bp); tv.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); chz = horiz_count_eq(hbuf, 0, BITS, N, 62); th.append(time.perf_counter() - t0)
    mean_pl = float(bp.mean())
    frac = mean_pl / BITS
    print('%-16s exact=%s | vertical %6.1fms  horizontal %6.1fms  (%.1fx) | planes touched %.2f/%d (%.0f%% of bytes)'
          % (name, cv == ref == chz, min(tv) * 1e3, min(th) * 1e3, min(th) / min(tv),
             mean_pl, BITS, 100 * frac), flush=True)


if __name__ == '__main__':
    uni = rng.integers(0, V, N).astype(np.int64)
    bench('uniform-V6000', uni)
    z = np.clip(rng.zipf(1.5, N), 1, V).astype(np.int64) - 1
    bench('zipf-V6000', z)
    hot = uni.copy()
    hot[rng.random(N) < 0.10] = 62               # 10% selectivity face
    bench('hot10pct-62', hot)
    print('BENCH DONE', flush=True)
