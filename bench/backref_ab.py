"""SPEED A/B of Jackson's variable-width back-reference layout for URLHash against today's column.

Layout (per block of B rows, blocks start on a byte, a table of block starts):
  per row: 1 flag bit; flag 0 -> the value's `bits` bits; flag 1 -> 4-bit class k (gap has k+1 bits,
  leading 1 implicit) then the k low bits of the gap back to the previous copy of the value IN THE BLOCK.
Readers: pread the touched blocks (neighbours merged, 16 lanes), then one compiled pass decodes the
blocks in parallel. Shapes as before: q40 survivors, c62 counter 62 rows, full column. Fresh process
each, files evicted, cold then hot; every answer checked against today's numbers.
Usage: PYTHONPATH=src python bench/backref_ab.py DB_DIR build | ab [reps]
"""
import sys, os, time, json, subprocess, glob, hashlib
import numpy as np
import numba
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = '/workspace/url_br'
LISTS = '/workspace/url_lists'
NT = 16
BLOCKS = (4096, 65536)


@numba.njit(inline='always')
def _put(buf, p, v, n):
    for i in range(n):
        if (v >> i) & 1:
            q = p + i
            buf[q >> 3] |= np.uint8(1 << (q & 7))
    return p + n


@numba.njit(inline='always')
def _get(buf, p, n):
    by = p >> 3
    w = np.uint64(0)
    for k in range(5):
        w |= np.uint64(buf[by + k]) << np.uint64(8 * k)
    return np.int64((w >> np.uint64(p & 7)) & np.uint64((1 << n) - 1))


@numba.njit(parallel=True, cache=True)
def _encode(x, gap, B, bits, boff, buf):
    nb = boff.size - 1
    for b in numba.prange(nb):
        p = boff[b] * 8
        lo = b * B; hi = min(x.size, lo + B)
        for r in range(lo, hi):
            g = gap[r]
            if g <= 0:
                p = _put(buf, p, 0, 1)
                p = _put(buf, p, x[r], bits)
            else:
                k = 0
                while (g >> (k + 1)) > 0:
                    k += 1
                p = _put(buf, p, 1, 1)
                p = _put(buf, p, k, 4)
                p = _put(buf, p, g & ((1 << k) - 1), k)


@numba.njit(nogil=True, inline='always')
def _decode_block(buf, pbyte, n, bits, out):
    p = pbyte * 8
    for i in range(n):
        f = _get(buf, p, 1); p += 1
        if f == 0:
            out[i] = _get(buf, p, bits); p += bits
        else:
            k = _get(buf, p, 4); p += 4
            g = (1 << k) | _get(buf, p, k); p += k
            out[i] = out[i - g]


@numba.njit(parallel=True, nogil=True, cache=True)
def _decode_full(buf, base, blocks, boff, B, N, bits, res, dst):
    for j in numba.prange(blocks.size):
        b = blocks[j]
        n = min(B, N - b * B)
        _decode_block(buf, boff[b] - base, n, bits, res[dst[j]:dst[j] + n])


@numba.njit(parallel=True, nogil=True, cache=True)
def _decode_gather(buf, base, blocks, boff, B, N, bits, rows, bnd, res):
    for j in numba.prange(blocks.size):
        b = blocks[j]
        n = min(B, N - b * B)
        tmp = np.empty(n, np.int64)
        _decode_block(buf, boff[b] - base, n, bits, tmp)
        for i in range(bnd[j], bnd[j + 1]):
            res[i] = tmp[rows[i] - b * B]


def _open(dbdir):
    import link_ceiling as L
    return L._open(dbdir)


def build(dbdir):
    os.makedirs(OUT, exist_ok=True)
    db, seg = _open(dbdir)
    x = np.asarray(seg._raw_codes('URLHash')).astype(np.int64)
    N = x.size
    bits = max(1, int(int(seg.cols['URLHash']['V']) - 1).bit_length())
    o = np.argsort(x, kind='stable')
    same = np.r_[False, x[o][1:] == x[o][:-1]]
    prev = np.full(N, -1, np.int64)
    prev[o[same]] = o[np.flatnonzero(same) - 1]
    del o, same
    r = np.arange(N, dtype=np.int64)
    czl = int(seg.cols['URLHash']['czlen'])
    for B in BLOCKS:
        t = time.perf_counter()
        gap = np.where((prev >= 0) & (prev // B == r // B), r - prev, 0)
        k = np.zeros(N, np.int64)
        gg = gap.copy()
        while True:
            m = (gg >> (k + 1)) > 0
            if not m.any():
                break
            k[m] += 1
        rb = np.where(gap > 0, 1 + 4 + k, 1 + bits)
        nb = -(-N // B)
        bbits = np.add.reduceat(rb, np.arange(0, N, B))
        boff = np.r_[0, np.cumsum((bbits + 7) // 8)].astype(np.int64)
        buf = np.zeros(int(boff[-1]) + 8, np.uint8)
        _encode(x, gap, B, bits, boff, buf)
        buf.tofile(os.path.join(OUT, 'br%d.bin' % B))
        np.save(os.path.join(OUT, 'br%d_off.npy' % B), boff)
        res = np.empty(N, np.int64)
        blocks = np.arange(nb, dtype=np.int64)
        _decode_full(buf, 0, blocks, boff, B, N, bits, res, blocks * B)
        assert np.array_equal(res, x), ('round trip', B)
        print('block %d: ROUND TRIP OK. %.1f MB (%.1f bits/row, table %.2f MB) vs today %.1f MB; %.0f s'
              % (B, buf.size / 1e6, buf.size * 8 / N, boff.nbytes / 1e6, czl / 1e6, time.perf_counter() - t), flush=True)
    np.save(os.path.join(OUT, 'bits.npy'), np.array([bits]))


_P = [None]


def _pool():
    if _P[0] is None:
        from concurrent.futures import ThreadPoolExecutor
        _P[0] = ThreadPoolExecutor(NT)
    return _P[0]


def read_blocks(fd, boff, blocks):
    """pread the byte ranges of `blocks` (sorted): neighbours merged, split so every lane has work."""
    lo = int(boff[blocks[0]]); hi = int(boff[blocks[-1] + 1])
    buf = np.zeros(hi - lo + 8, np.uint8)
    brk = np.flatnonzero(np.diff(blocks) != 1) + 1
    total = int((boff[blocks + 1] - boff[blocks]).sum())
    cap = max(1 << 18, min(8 << 20, total // (2 * NT)))
    runs = []
    for g in np.split(blocks, brk):
        a, b = int(boff[g[0]]), int(boff[g[-1] + 1])
        while a < b:
            runs.append((a, min(b, a + cap))); a += cap

    def task(ab):
        a, b = ab
        d = os.pread(fd, b - a, a)
        buf[a - lo:a - lo + len(d)] = np.frombuffer(d, np.uint8)
    list(_pool().map(task, runs))
    return buf, lo


def one(dbdir, layout, shape):
    db, seg = _open(dbdir)
    N = int(seg.N)
    rows = None if shape == 'full' else np.load(os.path.join(LISTS, shape + '.npy'))
    bits = int(np.load(os.path.join(OUT, 'bits.npy'))[0])
    z = np.zeros(64, np.uint8); e = np.zeros(1, np.int64)
    _decode_full(z, 0, e[:0], e, 1, 0, bits, e, e)                       # kernels loaded
    _decode_gather(z, 0, e[:0], e, 1, 0, bits, e, e, e)
    _pool().submit(int).result()
    for p in glob.glob(seg.path + '*') + glob.glob(os.path.join(OUT, '*')):
        if os.path.isfile(p):
            fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)

    def run():
        if layout == 'today':
            return np.asarray(seg._raw_codes('URLHash') if rows is None else seg.codes_at('URLHash', rows))
        B = int(layout[2:])
        boff = np.load(os.path.join(OUT, 'br%d_off.npy' % B))
        fd = os.open(os.path.join(OUT, 'br%d.bin' % B), os.O_RDONLY)
        blocks = np.arange(boff.size - 1, dtype=np.int64) if rows is None else np.unique(rows // B)
        buf, base = read_blocks(fd, boff, blocks)
        os.close(fd)
        if rows is None:
            res = np.empty(N, np.int64)
            _decode_full(buf, base, blocks, boff, B, N, bits, res, blocks * B)
        else:
            res = np.empty(rows.size, np.int64)
            bnd = np.searchsorted(rows // B, np.r_[blocks, blocks[-1] + 1])
            _decode_gather(buf, base, blocks, boff, B, N, bits, rows, bnd, res)
        return res
    t = time.perf_counter(); res = run(); cold = (time.perf_counter() - t) * 1e3
    t = time.perf_counter(); run(); hot = (time.perf_counter() - t) * 1e3
    h = hashlib.md5(np.ascontiguousarray(np.asarray(res, np.int64)).tobytes()).hexdigest()
    return dict(cold=round(cold, 1), hot=round(hot, 1), md5=h)


if __name__ == '__main__':
    if sys.argv[1] == '--one':
        print(json.dumps(one(sys.argv[2], sys.argv[3], sys.argv[4]))); sys.exit(0)
    dbdir, mode = sys.argv[1], sys.argv[2]
    if mode == 'build':
        build(dbdir); sys.exit(0)
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    lays = ['today'] + ['br%d' % B for B in BLOCKS]
    for shape in ('q40', 'c62', 'full'):
        out = {}
        for rep in range(reps):
            for layout in lays:
                o = subprocess.run([sys.executable, __file__, '--one', dbdir, layout, shape], capture_output=True, text=True)
                if o.returncode:
                    print(o.stderr[-3000:]); raise SystemExit(1)
                out.setdefault(layout, []).append(json.loads(o.stdout.strip().splitlines()[-1]))
        same = len({r['md5'] for L in out.values() for r in L}) == 1
        med = lambda L, k: sorted(r[k] for r in L)[len(L) // 2]
        print('%-4s ' % shape + ' | '.join('%s cold %s med %.0f hot %.0f' % (l, sorted(r['cold'] for r in out[l]),
              med(out[l], 'cold'), med(out[l], 'hot')) for l in lays) + ' | same: %s' % same, flush=True)
        assert same, shape
