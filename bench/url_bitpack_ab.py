"""SPEED A/B: URLHash bitpacked (today's dictionary numbers, 25 bits a row, no frames) against today's
column (enc 3, 65,536-row zstd frames). Jumping straight to a row instead of unpacking its frame.

Readers for the bitpacked file:
  mm   mmap the file, 16 lanes pull each wanted row's bits (the kernel pages in what is touched)
  pr   pread only the 4 KB pages that hold wanted rows (neighbours merged, 16 lanes), then pull bits
Shapes (fresh process each, database opened first, files evicted, then a hot re-run):
  q40  Q40's 89,914 survivors   c62  counter 62's 738,172 rows   full  every row
Every answer is checked against today's numbers.
Usage: PYTHONPATH=src python bench/url_bitpack_ab.py DB_DIR build | ab [reps]
"""
import sys, os, time, json, subprocess, glob, hashlib
import numpy as np
import numba
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = '/workspace/url_bp'
LISTS = '/workspace/url_lists'          # the survivor / counter-62 row files from url_lists_ab.py
NT = 16
PG = 4096


@numba.njit(cache=True)
def _pack(codes, B, out):
    for i in range(codes.size):
        bit = i * B; by = bit >> 3; sh = bit & 7
        v = np.uint64(codes[i]) << np.uint64(sh)
        for k in range(5):
            out[by + k] |= np.uint8((v >> np.uint64(8 * k)) & np.uint64(255))


@numba.njit(parallel=True, nogil=True, cache=True)
def _pull(buf, base, rows, B, res):
    """res[i] = the B-bit number of row rows[i]; buf holds the file from byte `base` on."""
    m = (1 << B) - 1
    for i in numba.prange(rows.size):
        bit = rows[i] * B; by = (bit >> 3) - base; sh = bit & 7
        v = np.int64(buf[by]) | (np.int64(buf[by + 1]) << 8) | (np.int64(buf[by + 2]) << 16) \
            | (np.int64(buf[by + 3]) << 24) | (np.int64(buf[by + 4]) << 32)
        res[i] = (v >> sh) & m


def _open(dbdir):
    import link_ceiling as L
    return L._open(dbdir)


def build(dbdir):
    os.makedirs(OUT, exist_ok=True)
    db, seg = _open(dbdir)
    t = time.perf_counter()
    cc = np.asarray(seg._raw_codes('URLHash')).astype(np.int64)
    B = int(int(seg.cols['URLHash']['V']) - 1).bit_length()
    out = np.zeros((cc.size * B + 7) // 8 + 8, np.uint8)
    _pack(cc, B, out)
    out.tofile(os.path.join(OUT, 'bp.bin'))
    back = np.empty(cc.size, np.int64)
    _pull(out, 0, np.arange(cc.size, dtype=np.int64), B, back)
    assert np.array_equal(back, cc), 'round trip'
    np.save(os.path.join(OUT, 'B.npy'), np.array([B]))
    czl = int(seg.cols['URLHash']['czlen'])
    print('ROUND TRIP OK. bitpacked %d bits/row: %.1f MB vs today %.1f MB (+%.1f MB, %+.0f%%); %.0f s'
          % (B, out.size / 1e6, czl / 1e6, (out.size - czl) / 1e6, 100.0 * (out.size - czl) / czl,
             time.perf_counter() - t), flush=True)


_P = [None]


def _pool():
    if _P[0] is None:
        from concurrent.futures import ThreadPoolExecutor
        _P[0] = ThreadPoolExecutor(NT)
    return _P[0]


def read_pr(fd, rows, B):
    """pread only the pages holding wanted rows: neighbouring pages merged into runs, runs split so
    every lane gets work, all issued at once; then pull the bits."""
    b0 = (rows * B) >> 3
    pages = np.unique(np.r_[b0 // PG, (b0 + 4) // PG])
    lo, hi = int(pages[0]) * PG, (int(pages[-1]) + 1) * PG
    buf = np.zeros(hi - lo + 8, np.uint8)
    brk = np.flatnonzero(np.diff(pages) != 1) + 1
    runs = []
    total = pages.size * PG
    cap = max(1 << 18, min(8 << 20, total // (2 * NT)))
    for r in np.split(pages, brk):
        a, b = int(r[0]) * PG, (int(r[-1]) + 1) * PG
        while a < b:
            runs.append((a, min(b, a + cap))); a += cap

    def task(ab):
        a, b = ab
        data = os.pread(fd, b - a, a)
        buf[a - lo:a - lo + len(data)] = np.frombuffer(data, np.uint8)
    list(_pool().map(task, runs))
    res = np.empty(rows.size, np.int64)
    _pull(buf, lo, rows, B, res)
    return res, pages.size * PG


def one(dbdir, layout, shape):
    db, seg = _open(dbdir)
    N = int(seg.N)
    rows = np.arange(N, dtype=np.int64) if shape == 'full' else np.load(os.path.join(LISTS, shape + '.npy'))
    B = int(np.load(os.path.join(OUT, 'B.npy'))[0])
    _pull(np.zeros(16, np.uint8), 0, np.zeros(1, np.int64), B, np.zeros(1, np.int64))   # kernel loaded
    _pool().submit(int).result()
    for p in glob.glob(seg.path + '*') + glob.glob(os.path.join(OUT, '*')):
        if os.path.isfile(p):
            fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    info = {}

    def run():
        if layout == 'today':
            return np.asarray(seg._raw_codes('URLHash') if shape == 'full' else seg.codes_at('URLHash', rows))
        path = os.path.join(OUT, 'bp.bin')
        if layout == 'mm':
            mm = np.memmap(path, np.uint8, 'r')
            res = np.empty(rows.size, np.int64); _pull(mm, 0, rows, B, res)
            del mm
            return res
        fd = os.open(path, os.O_RDONLY)
        res, nb = read_pr(fd, rows, B); os.close(fd); info['read_mb'] = round(nb / 1e6, 1)
        return res
    t = time.perf_counter(); res = run(); cold = (time.perf_counter() - t) * 1e3
    t = time.perf_counter(); run(); hot = (time.perf_counter() - t) * 1e3
    h = hashlib.md5(np.ascontiguousarray(np.asarray(res, np.int64)).tobytes()).hexdigest()
    return dict(cold=round(cold, 1), hot=round(hot, 1), md5=h, **info)


if __name__ == '__main__':
    if sys.argv[1] == '--one':
        print(json.dumps(one(sys.argv[2], sys.argv[3], sys.argv[4]))); sys.exit(0)
    dbdir, mode = sys.argv[1], sys.argv[2]
    if mode == 'build':
        build(dbdir); sys.exit(0)
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    for shape in ('q40', 'c62', 'full'):
        out = {}
        for rep in range(reps):
            for layout in ('today', 'mm', 'pr'):
                o = subprocess.run([sys.executable, __file__, '--one', dbdir, layout, shape], capture_output=True, text=True)
                if o.returncode:
                    print(o.stderr[-3000:]); raise SystemExit(1)
                out.setdefault(layout, []).append(json.loads(o.stdout.strip().splitlines()[-1]))
        same = len({r['md5'] for L in out.values() for r in L}) == 1
        med = lambda L, k: sorted(r[k] for r in L)[len(L) // 2]
        print('%-4s cold today %s | mm %s | pr %s (reads %s MB) || medians cold %.0f / %.0f / %.0f, hot %.0f / %.0f / %.0f ms | same: %s'
              % (shape, sorted(r['cold'] for r in out['today']), sorted(r['cold'] for r in out['mm']),
                 sorted(r['cold'] for r in out['pr']), out['pr'][0].get('read_mb'),
                 med(out['today'], 'cold'), med(out['mm'], 'cold'), med(out['pr'], 'cold'),
                 med(out['today'], 'hot'), med(out['mm'], 'hot'), med(out['pr'], 'hot'), same), flush=True)
        assert same, shape
