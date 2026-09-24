"""SPEED A/B of Jackson's value-as-axis layout for URLHash against today's column (enc 3).

The new layout, per 65,536-row section, one zstd frame holding:
  flags   one bit per distinct URL in first-seen order: new (= the next global number) or returning
  ret     the returning URLs' global numbers (int32 gaps)
  stream  per URL in first-seen order: count, jump of its first row from the previous URL's first row,
          then jumps to its later rows -- variable width (1/2/3 bytes)
Global numbers are given out by first appearance; `perm` maps today's value-order number -> new
number (for literal lookups; its size is reported).

Shapes timed (fresh process each, database opened first, files evicted, then hot re-run):
  full  every row's number            today: seg._raw_codes      lists: walk every section
  q40   Q40's 89,914 survivor rows    today: seg.codes_at        lists: walk the touched sections
  c62   counter 62's 738,172 rows     today: seg.codes_at        lists: walk the touched sections
Every lists answer is checked against today's through perm (not timed).

Usage: PYTHONPATH=src python bench/url_lists_ab.py DB_DIR build
       PYTHONPATH=src python bench/url_lists_ab.py DB_DIR ab [reps]
"""
import sys, os, time, json, subprocess, glob
import numpy as np
import numba
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = '/workspace/url_lists'
FR = 65536
NT = 16


def venc(g):
    g = g.astype(np.int64)
    ln = 1 + (g >= 128) + (g >= 16384)
    b = np.stack([(g & 0x7F) | ((g >= 128) << 7), ((g >> 7) & 0x7F) | ((g >= 16384) << 7), g >> 14], 1)
    return b[np.arange(3)[None, :] < ln[:, None]].astype(np.uint8)


def enc_section(cc, seen_max):
    n = cc.size
    order = np.argsort(cc, kind='stable'); sc = cc[order]
    start = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
    cnt = np.diff(np.r_[start, n]); L = start.size
    fo = np.argsort(order[start], kind='stable')            # lists in first-seen order
    vals, cnts, fps = sc[start][fo], cnt[fo], order[start][fo]
    new = vals > seen_max
    assert np.array_equal(vals[new], seen_max + 1 + np.arange(int(new.sum()))), 'new numbers not consecutive'
    ret = vals[~new]
    rk = np.empty(L, np.int64); rk[fo] = np.arange(L)
    lid = np.repeat(np.arange(L), cnt)                      # list of each sorted row
    pos2 = order[np.argsort(rk[lid], kind='stable')]        # rows grouped by first-seen list, ascending
    gaps = np.diff(pos2, prepend=0)
    s2 = np.r_[0, np.cumsum(cnts)[:-1]]
    gaps[s2] = np.diff(fps, prepend=0)
    stream = venc(np.insert(gaps, s2, cnts))
    fl = np.packbits(new, bitorder='little')
    rd = np.diff(ret, prepend=0).astype(np.int32)
    blob = fl.tobytes() + rd.tobytes() + stream.tobytes()
    return blob, (L, int((~new).sum()), fl.size, int(seen_max + 1)), int(max(seen_max, vals.max()))


@numba.njit(nogil=True, cache=True)
def _walk(buf, nl, nret, flen, base, out):
    """One section: flags at 0, returning gaps after, stream after that. Writes every row's number."""
    r0 = flen
    p = flen + 4 * nret
    prev = 0; kn = base; kr = 0; racc = 0
    for i in range(nl):
        c = 0; sh = 0
        while True:
            b = np.int64(buf[p]); p += 1; c |= (b & 127) << sh; sh += 7
            if b < 128:
                break
        j = 0; sh = 0
        while True:
            b = np.int64(buf[p]); p += 1; j |= (b & 127) << sh; sh += 7
            if b < 128:
                break
        pos = prev + j; prev = pos
        if (buf[i >> 3] >> (i & 7)) & 1:
            num = kn; kn += 1
        else:
            q = r0 + 4 * kr
            d = np.int64(buf[q]) | (np.int64(buf[q + 1]) << 8) | (np.int64(buf[q + 2]) << 16) | (np.int64(buf[q + 3]) << 24)
            if d >= 2147483648:
                d -= 4294967296
            racc += d; num = racc; kr += 1
        out[pos] = num
        for t in range(c - 1):
            g = 0; sh = 0
            while True:
                b = np.int64(buf[p]); p += 1; g |= (b & 127) << sh; sh += 7
                if b < 128:
                    break
            pos += g; out[pos] = num


_P = [None]


def _pool():
    if _P[0] is None:
        from concurrent.futures import ThreadPoolExecutor
        _P[0] = ThreadPoolExecutor(NT)
    return _P[0]


def _meta():
    m = np.load(os.path.join(OUT, 'meta.npz'))
    return {k: m[k] for k in m.files}


def lists_decode(sections, meta, fd, want=None):
    """Numbers for every row of `sections` (want=None) or for rows `want` (sorted, global).
    Neighbouring sections are read in one pread (runs up to 8 MB); one task per run, 16 lanes."""
    import zstandard as zs
    from concurrent.futures import ThreadPoolExecutor
    off, clen, nl, nret, flen, base, nrow = (meta[k] for k in ('off', 'clen', 'nl', 'nret', 'flen', 'base', 'nrow'))
    runs = []; cur = []
    cap = min(8 << 20, max(1 << 18, int(clen[sections].sum()) // (2 * NT)))   # at least 2 tasks a lane
    for s in sections:
        if cur and (s != cur[-1] + 1 or off[s] + clen[s] - off[cur[0]] > cap):
            runs.append(cur); cur = []
        cur.append(int(s))
    if cur:
        runs.append(cur)
    if want is None:
        res = np.empty(int(nrow[sections].sum()), np.int32)
        dst = np.r_[0, np.cumsum(nrow[sections])]
        slot = {int(s): i for i, s in enumerate(sections)}
    else:
        res = np.empty(want.size, np.int32)
        sec_of = want // FR
        bnd = np.searchsorted(sec_of, np.arange(int(meta['off'].size) + 1))

    def task(run):
        dz = zs.ZstdDecompressor()
        a = int(off[run[0]]); b = int(off[run[-1]] + clen[run[-1]])
        raw = os.pread(fd, b - a, a)
        tmp = np.empty(FR, np.int32)
        for s in run:
            buf = np.frombuffer(dz.decompress(raw[int(off[s]) - a:int(off[s] + clen[s]) - a]), np.uint8)
            if want is None:
                i = slot[s]
                _walk(buf, int(nl[s]), int(nret[s]), int(flen[s]), int(base[s]), res[dst[i]:dst[i + 1]])
            else:
                _walk(buf, int(nl[s]), int(nret[s]), int(flen[s]), int(base[s]), tmp)
                lo, hi = bnd[s], bnd[s + 1]
                res[lo:hi] = tmp[want[lo:hi] - s * FR]

    list(_pool().map(task, runs))
    return res


def _open(dbdir):
    import link_ceiling as L
    return L._open(dbdir)


def build(dbdir):
    import link_ceiling as L, wdb_funnel
    import zstandard as zs
    os.makedirs(OUT, exist_ok=True)
    db, seg = _open(dbdir)
    N = int(seg.N); t = time.perf_counter()
    allc = np.asarray(seg._raw_codes('URLHash')).astype(np.int64)
    u, first = np.unique(allc, return_index=True)
    assert u.size == u[-1] + 1
    perm = np.empty(u.size, np.int64); perm[np.argsort(first, kind='stable')] = np.arange(u.size)
    new = perm[allc]
    np.save(os.path.join(OUT, 'perm.npy'), perm.astype(np.uint32))
    print('renumbered by first appearance: %.0f s' % (time.perf_counter() - t), flush=True)
    cz = zs.ZstdCompressor(level=9)
    nsec = -(-N // FR)
    M = {k: np.zeros(nsec, np.int64) for k in ('off', 'clen', 'nl', 'nret', 'flen', 'base', 'nrow')}
    seen = -1; pos = 0
    with open(os.path.join(OUT, 'blob.bin'), 'wb') as f:
        for s in range(nsec):
            cc = new[s * FR:min(N, (s + 1) * FR)]
            blob, (nl, nret, flen, base), seen = enc_section(cc, seen)
            z = cz.compress(blob); f.write(z)
            for k, v in (('off', pos), ('clen', len(z)), ('nl', nl), ('nret', nret), ('flen', flen),
                         ('base', base), ('nrow', cc.size)):
                M[k][s] = v
            pos += len(z)
    np.savez(os.path.join(OUT, 'meta.npz'), **M)
    print('encoded %d sections: %.0f s' % (nsec, time.perf_counter() - t), flush=True)
    fd = os.open(os.path.join(OUT, 'blob.bin'), os.O_RDONLY)
    back = lists_decode(np.arange(nsec), _meta(), fd)
    assert np.array_equal(back, new), 'full decode differs'
    pz = len(cz.compress(perm.astype(np.uint32).tobytes()))
    czl = int(seg.cols['URLHash']['czlen'])
    print('ROUND TRIP OK. lists %.1f MB (%.1f bits/row) vs today %.1f MB (%.1f bits/row); '
          'perm map %.1f MB zstd (%.1f bits/row)' % (pos / 1e6, pos * 8 / N, czl / 1e6, czl * 8 / N, pz / 1e6, pz * 8 / N))
    _, rows, _ = L.by_hand(seg, 'C')
    np.save(os.path.join(OUT, 'q40.npy'), np.asarray(rows, np.int64))
    c62 = wdb_funnel._code_of(seg, 'CounterID', 62)
    np.save(os.path.join(OUT, 'c62.npy'), np.asarray(wdb_funnel.positions(seg, 'CounterID', c62), np.int64))
    print('shapes saved', flush=True)


def _evict(seg):
    for p in glob.glob(seg.path + '*') + glob.glob(os.path.join(OUT, '*')):
        if os.path.isfile(p):
            fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)


def one(dbdir, layout, shape):
    import hashlib
    db, seg = _open(dbdir)
    rows = None if shape == 'full' else np.load(os.path.join(OUT, shape + '.npy'))
    _walk(np.zeros(8, np.uint8), 0, 0, 0, 0, np.zeros(1, np.int32))     # kernel loaded, as a real process has
    _pool().submit(int).result()
    _evict(seg)

    def run():
        if layout == 'today':
            return np.asarray(seg._raw_codes('URLHash')) if rows is None else np.asarray(seg.codes_at('URLHash', rows))
        meta = _meta()
        fd = os.open(os.path.join(OUT, 'blob.bin'), os.O_RDONLY)
        secs = np.arange(meta['off'].size) if rows is None else np.unique(rows // FR)
        r = lists_decode(secs, meta, fd, rows)
        os.close(fd)
        return r
    t = time.perf_counter(); res = run(); cold = (time.perf_counter() - t) * 1e3
    t = time.perf_counter(); run(); hot = (time.perf_counter() - t) * 1e3
    if layout == 'today':
        res = np.load(os.path.join(OUT, 'perm.npy'))[np.asarray(res, np.int64)]
    h = hashlib.md5(np.ascontiguousarray(np.asarray(res, np.int64)).tobytes()).hexdigest()
    return {'cold': round(cold, 1), 'hot': round(hot, 1), 'md5': h}


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
            for layout in ('today', 'lists'):
                o = subprocess.run([sys.executable, __file__, '--one', dbdir, layout, shape], capture_output=True, text=True)
                if o.returncode:
                    print(o.stderr[-3000:]); raise SystemExit(1)
                out.setdefault(layout, []).append(json.loads(o.stdout.strip().splitlines()[-1]))
        same = len({r['md5'] for L in out.values() for r in L}) == 1
        med = lambda L, k: sorted(r[k] for r in L)[len(L) // 2]
        print('%-4s today cold %s hot %.0f | lists cold %s hot %.0f | median cold %.0f -> %.0f ms | same answer: %s'
              % (shape, sorted(r['cold'] for r in out['today']), med(out['today'], 'hot'),
                 sorted(r['cold'] for r in out['lists']), med(out['lists'], 'hot'),
                 med(out['today'], 'cold'), med(out['lists'], 'cold'), same), flush=True)
        assert same, shape
