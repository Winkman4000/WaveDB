"""THE GAP PACKER (Jackson, 2026-09-24) -- a prototype, measured against zstd on the real dictionaries.

The zstd autopsy showed a sorted integer dictionary is half noise stored as itself (the low bytes
of each gap, raw literals) and half rules that say "the top bytes of this gap are small" -- about
one copy rule per value, chains up to 56 deep, half the frame run on average before a value is
ready. Here every value is IN PLACE by construction instead:

  per chunk of CH values: base (the first value), width w (bits of the chunk's largest gap), the
  gaps packed at w bits each -- gap j at bit j*w, reachable by arithmetic -- and a CHECKPOINT
  every K values (the value's offset from base), so any value is at most K-1 additions away.

Measured per column, on the values of the live database: size against zstd, full decode (warm,
CPU only: every chunk in parallel), and a single-value read. FAIL-LOUD: every decode is checked
value for value against the dictionary.

Usage: PYTHONPATH=src python bench/gap_pack.py DB_DIR [col ...]
"""
import sys, os, glob, time
import numpy as np
import numba
from numba import njit, prange
import wdb_engine

CH = 8192


@njit(cache=True)
def _nbits(x):
    b = 0
    while x != np.uint64(0):
        x >>= np.uint64(1); b += 1
    return b


@njit(cache=True)
def plan(vu, CH):
    n = vu.size; nch = (n + CH - 1) // CH
    w = np.zeros(nch, np.int64)
    for c in range(nch):
        lo = c * CH; hi = min(n, lo + CH); m = np.uint64(0)
        for i in range(lo + 1, hi):
            g = vu[i] - vu[i - 1]
            if g > m: m = g
        w[c] = max(1, _nbits(m))
    return w


@njit(cache=True, parallel=True)
def pack(vu, CH, K, w, bitstart, words, ckpt):
    n = vu.size; nch = w.size; nk = ckpt.shape[1]
    for c in prange(nch):
        lo = c * CH; hi = min(n, lo + CH); wc = np.uint64(w[c]); pos = bitstart[c]
        for k in range(nk):
            if lo + k * K < hi:
                ckpt[c, k] = vu[lo + k * K] - vu[lo]
        for i in range(lo + 1, hi):
            g = vu[i] - vu[i - 1]
            q = pos >> 6; r = np.uint64(pos & 63)
            words[q] |= g << r
            if r + wc > np.uint64(64):
                words[q + 1] |= g >> (np.uint64(64) - r)
            pos += w[c]


@njit(cache=True, inline='always')
def _get(words, pos, wc):
    q = pos >> 6; r = np.uint64(pos & 63)
    x = words[q] >> r
    if r + wc > np.uint64(64):
        x |= words[q + 1] << (np.uint64(64) - r)
    if wc < np.uint64(64):
        x &= (np.uint64(1) << wc) - np.uint64(1)
    return x


@njit(cache=True, parallel=True)
def decode_all(words, w, bitstart, base, n, CH, out):
    nch = w.size
    for c in prange(nch):
        lo = c * CH; hi = min(n, lo + CH); wc = np.uint64(w[c]); pos = bitstart[c]
        acc = base[c]; out[lo] = acc
        for i in range(lo + 1, hi):
            acc += _get(words, pos, wc); out[i] = acc; pos += w[c]


@njit(cache=True)
def decode_at(words, w, bitstart, base, ckpt, K, idx, out):
    for t in range(idx.size):
        i = idx[t]; c = i // 8192; j = i - c * 8192; k = j // K
        wc = np.uint64(w[c])
        acc = base[c] + ckpt[c, k]
        pos = bitstart[c] + (k * K) * w[c]
        for s in range(k * K + 1, j + 1):
            acc += _get(words, pos, wc); pos += w[c]
        out[t] = acc


def build(v, K):
    vu = v.view(np.uint64)
    w = plan(vu, CH)
    nper = np.minimum(CH, v.size - np.arange(w.size) * CH) - 1          # gaps per chunk
    bits = nper * w
    bitstart = np.zeros(w.size, np.int64)
    # each chunk starts on a word boundary (a real layout reads a chunk on its own)
    wordsz = (bits + 63) // 64
    bitstart[1:] = np.cumsum(wordsz)[:-1] * 64
    words = np.zeros(int(wordsz.sum()) + 1, np.uint64)
    nk = (CH + K - 1) // K
    ckpt = np.zeros((w.size, nk), np.uint64)
    pack(vu, CH, K, w, bitstart, words, ckpt)
    base = vu[::CH].copy()
    size = words.nbytes + w.size * (8 + 1 + 8) + ckpt.nbytes              # words + base, width, offset + checkpoints
    return dict(w=w, bitstart=bitstart, words=words, ckpt=ckpt, base=base, size=size, K=K)


def best(fn, r=3):
    ts = []
    for _ in range(r):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts) * 1e3


def zstd_point(seg, c, i):
    import zstandard as z
    ch = i // int(c['i2ch']); zo = c['i2zoffs']; b = c['i2base']
    raw = seg.read_span(b + int(zo[ch]), b + int(zo[ch + 1]))
    d = np.frombuffer(z.ZstdDecompressor().decompress(raw), np.int64)
    return int(np.cumsum(d)[i - ch * int(c['i2ch'])])


if __name__ == '__main__':
    db = sys.argv[1]
    cols = sys.argv[2:] or ['URLHash', 'RefererHash', 'UserID', 'WatchID', 'FUniqID', 'HID',
                            'ClientIP', 'RemoteIP', 'EventTime', 'LocalEventTime', 'ClientEventTime']
    seg = wdb_engine.Segment(glob.glob(os.path.join(db, '*.wdb'))[0])
    rng = np.random.default_rng(5)
    print('%-15s %10s | %9s | %-38s | %21s | %22s' % (
        'column', 'values', 'zstd MB', 'gap pack MB: no ckpt / K=256 / K=64 / K=16', 'full decode ms z / gap',
        'one value us z / K=64'))
    for nm in cols:
        c = seg.cols[nm]
        assert int(c['i2ch']) == CH, (nm, 'expects the 8192-value chunks of cbdb_i2')
        c['intvals'] = None
        v = np.ascontiguousarray(seg._dict_ints(c), dtype=np.int64)
        zmb = (int(c['i2zoffs'][-1]) + 4 * (len(c['i2zoffs']) - 1)) / 1e6
        # zstd full decode, warm (the data is in memory now): the engine's own reader
        def zfull():
            c['intvals'] = None; c['i2chunks'].clear(); seg._dict_ints(c)
        tz = best(zfull)
        packs = {K: build(v, K) for K in (CH, 256, 64, 16)}
        P = packs[64]
        out = np.empty_like(v).view(np.uint64)
        decode_all(P['words'], P['w'], P['bitstart'], P['base'], v.size, CH, out)   # compile + check
        assert np.array_equal(out.view(np.int64), v), (nm, 'full decode differs')
        tg = best(lambda: decode_all(P['words'], P['w'], P['bitstart'], P['base'], v.size, CH, out))
        idx = rng.integers(0, v.size, 2000).astype(np.int64)
        res = {}
        for K, Q in packs.items():
            o = np.empty(idx.size, np.uint64)
            decode_at(Q['words'], Q['w'], Q['bitstart'], Q['base'], Q['ckpt'], Q['K'], idx, o)
            assert np.array_equal(o.view(np.int64), v[idx]), (nm, K, 'point read differs')
            res[K] = best(lambda: decode_at(Q['words'], Q['w'], Q['bitstart'], Q['base'], Q['ckpt'], Q['K'], idx, o)) * 1e3 / idx.size
        c['intvals'] = None; c['i2chunks'].clear()
        few = idx[:200]
        for i in few[:3]: assert zstd_point(seg, c, int(i)) == int(v[i])
        tzp = best(lambda: [zstd_point(seg, c, int(i)) for i in few], 1) * 1e3 / few.size
        print('%-15s %10d | %9.2f | %8.2f / %6.2f / %6.2f / %6.2f  (w %4.1f bits) | %8.1f / %8.1f  | %8.1f / %6.2f  (no ckpt %.1f, K=16 %.2f)' % (
            nm, v.size, zmb, packs[CH]['size'] / 1e6, packs[256]['size'] / 1e6, packs[64]['size'] / 1e6,
            packs[16]['size'] / 1e6, float(P['w'].mean()), tz, tg, tzp, res[64], res[CH], res[16]), flush=True)
