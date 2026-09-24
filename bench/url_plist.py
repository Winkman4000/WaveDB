"""Jackson's value-as-axis layout: per section, no number per row -- for each distinct value, the list
of rows it sits on, written as jumps (variable width: 1 byte under 128, 2 under 16,384, else 3; no
jump exceeds the section). Plus a count per value and the section dictionary that says which value
each list belongs to. Bits per row, against the display layout (13.5 / 14.3 + dictionary).
Two list orders: by the section dictionary's number order, and by first appearance (then each list's
first row is a jump from the previous list's first row).
Usage: PYTHONPATH=src python bench/url_plist.py DB_DIR
"""
import sys, time
import numpy as np
sys.path.insert(0, 'bench')
import link_ceiling as L, wdb_funnel
try:
    import zstandard as _z
    _C = _z.ZstdCompressor(level=9); zlen = lambda b: len(_C.compress(b))
except ImportError:
    from compression import zstd as _z
    zlen = lambda b: len(_z.compress(b, level=9))


def vlen(g):
    return 1 + (g >= 128) + (g >= 16384)


def venc(g):
    g = g.astype(np.int64)
    b = np.stack([(g & 0x7F) | ((g >= 128) << 7), ((g >> 7) & 0x7F) | ((g >= 16384) << 7), g >> 14], 1)
    keep = np.arange(3)[None, :] < vlen(g)[:, None]
    return b[keep].astype(np.uint8).tobytes()


def section(cc):
    """-> [varint bytes, varint+zstd bytes, first-seen varint bytes, first-seen varint+zstd,
           dictionary (number order, gaps) zstd, dictionary (first-seen order, deltas) zstd]"""
    order = np.argsort(cc, kind='stable')          # rows grouped by value, ascending inside
    sc = cc[order]
    start = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
    cnt = np.diff(np.r_[start, sc.size])
    gaps = np.diff(order, prepend=0); gaps[start] = order[start]      # first row: its position
    a = venc(gaps) + venc(cnt)
    # first-appearance order of the lists: first rows become jumps from the previous list's first row
    fo = np.argsort(order[start], kind='stable')
    firsts = order[start][fo]
    g2 = gaps.copy(); g2[start] = 0
    j = np.diff(firsts, prepend=0)
    b = venc(np.r_[g2[np.setdiff1d(np.arange(sc.size), start)], j]) + venc(cnt[fo])
    u = sc[start]
    dnum = zlen(np.diff(u, prepend=0).astype(np.uint32).tobytes())
    dfirst = zlen(np.diff(u[fo], prepend=0).astype(np.int32).tobytes())
    return np.array([len(a), zlen(a), len(b), zlen(b), dnum, dfirst], float)


db, seg = L._open(sys.argv[1])
c = seg.cols['URLHash']; FR = int(c['BR']); N = int(seg.N); t = time.perf_counter()
allc = np.asarray(seg._raw_codes('URLHash')).astype(np.int64)
c62 = wdb_funnel._code_of(seg, 'CounterID', 62)
runs, _ = L._blocks_runs(seg, 'CounterID', c62)
ra, rb = runs[0][0] // FR, -(-runs[-1][1] // FR)
for numbering in ('value order', 'first appearance'):
    if numbering == 'first appearance':
        u, first = np.unique(allc, return_index=True)
        rank = np.empty(u.size, np.int64); rank[np.argsort(first, kind='stable')] = np.arange(u.size)
        allc = rank[np.searchsorted(u, allc)]
    if len(sys.argv) > 2:
        continue
    print('--- global numbers in %s ---' % numbering, flush=True)
    for name, lo, hi in (("counter 62's region", ra, rb), ('whole column', 0, -(-N // FR))):
        acc = np.zeros(6); rows = 0
        for f in range(lo, hi):
            cc = allc[f * FR:min(N, (f + 1) * FR)]
            acc += section(cc); rows += cc.size
        b = acc * 8 / rows
        print('%s: lists by number order: jumps %.1f raw / %.1f zstd, + dictionary %.1f = %.1f bits/row'
              % (name, b[0], b[1], b[4], min(b[0], b[1]) + b[4]))
        print('%s  lists by first seen:   jumps %.1f raw / %.1f zstd, + dictionary %.1f = %.1f bits/row  (%.0f s)'
              % (' ' * len(name), b[2], b[3], b[5], min(b[2], b[3]) + b[5], time.perf_counter() - t), flush=True)


# The first-seen dictionary, split: with global numbers in first-appearance order, a URL never seen in
# an earlier section is always the next number -- it costs one flag bit. Only returning URLs carry a
# number. (Run: url_plist.py DB_DIR split)
if len(sys.argv) > 2 and sys.argv[2] == 'split':
    seen_max = -1; flags = 0; ret_bytes = 0; rows = 0; nnew = 0; nret = 0
    lo, hi = ra, rb
    for f in range(0, -(-N // FR)):
        cc = allc[f * FR:min(N, (f + 1) * FR)]
        order = np.argsort(cc, kind='stable'); sc = cc[order]
        start = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
        u = sc[start][np.argsort(order[start], kind='stable')]      # the section's values, first seen order
        new = u > seen_max
        seen_max = max(seen_max, int(u.max()))
        if lo <= f < hi or len(sys.argv) > 3:
            flags += zlen(np.packbits(new).tobytes())
            ret_bytes += zlen(np.diff(u[~new], prepend=0).astype(np.int32).tobytes())
            rows += cc.size; nnew += int(new.sum()); nret += int((~new).sum())
    where = 'whole column' if len(sys.argv) > 3 else "counter 62's region"
    print('%s, first-seen dictionary split: new URLs %d (flag bits %.2f bits/row), returning %d (%.1f bits/row)'
          ' = %.1f bits/row' % (where, nnew, flags * 8 / rows, nret, ret_bytes * 8 / rows, (flags + ret_bytes) * 8 / rows))
