"""Jackson's section-numbered display: each section numbers its own distinct URLs (the display, per
row) and keeps a section dictionary that completes each number to the URL. Both compressed with zstd
at the engine's level, per section. What does it cost in bits per row, against today's column?
Dictionary forms: the global dictionary numbers (sorted, stored as gaps -- the linear rule), or the
full hashes (sorted, as gaps). Display numbering: by value order, or by first appearance.
Usage: PYTHONPATH=src python bench/url_local.py DB_DIR
"""
import sys, time
import numpy as np
sys.path.insert(0, 'bench')
import link_ceiling as L, wdb_funnel
try:
    import zstandard as _z
    _C = _z.ZstdCompressor(level=9)
    zlen = lambda b: len(_C.compress(b))
except ImportError:
    from compression import zstd as _z
    zlen = lambda b: len(_z.compress(b, level=9))

db, seg = L._open(sys.argv[1])
c = seg.cols['URLHash']; FR = int(c['BR']); N = int(seg.N)
print('today: %.1f bits/row (whole column, zstd over 4-byte numbers)' % (int(c['czlen']) * 8 / N))
t = time.perf_counter()
allc = np.asarray(seg._raw_codes('URLHash')).astype(np.int64)
c62 = wdb_funnel._code_of(seg, 'CounterID', 62)
runs, _ = L._blocks_runs(seg, 'CounterID', c62)
ra, rb = runs[0][0] // FR, -(-runs[-1][1] // FR)


def section(codes, hashes_of=None):
    u, first, inv = np.unique(codes, return_index=True, return_inverse=True)
    d = u.size
    w = np.uint8 if d <= 256 else (np.uint16 if d <= 65536 else np.uint32)
    disp_val = zlen(inv.astype(w).tobytes())
    rank = np.empty(d, np.int64); rank[np.argsort(first, kind='stable')] = np.arange(d)
    disp_first = zlen(rank[inv].astype(w).tobytes())
    gaps = np.diff(u, prepend=0).astype(np.uint32)
    dict_codes = zlen(gaps.tobytes())
    dict_hash = None
    if hashes_of is not None:
        hv = np.sort(np.asarray(seg.values_at('URLHash', u)).astype(np.int64).view(np.uint64))
        dict_hash = zlen(np.diff(hv, prepend=np.uint64(0)).tobytes())
    return d, disp_val, disp_first, dict_codes, dict_hash, zlen(codes.astype(np.uint32).tobytes())


for name, lo, hi, want_hash in (("counter 62's region", ra, rb, True), ('whole column', 0, -(-N // FR), False)):
    acc = np.zeros(6); rows = 0; ds = []
    for f in range(lo, hi):
        cc = allc[f * FR:min(N, (f + 1) * FR)]
        r = section(cc, want_hash if want_hash else None)
        ds.append(r[0]); rows += cc.size
        acc += [r[0], r[1], r[2], r[3], r[4] or 0, r[5]]
    b = lambda x: x * 8 / rows
    print('%s: %d sections of %d rows, distinct per section median %d' % (name, hi - lo, FR, int(np.median(ds))))
    print('   today, same sections recompressed: %.1f bits/row' % b(acc[5]))
    print('   display, numbered by value order:  %.1f bits/row' % b(acc[1]))
    print('   display, numbered by first seen:   %.1f bits/row' % b(acc[2]))
    print('   dictionary of global numbers (gaps): %.1f bits/row' % b(acc[3]))
    if want_hash:
        print('   dictionary of full hashes (gaps):    %.1f bits/row' % b(acc[4]))
    print('   (%.0f s so far)' % (time.perf_counter() - t), flush=True)


# THE LINEAR RULE: number the whole column's URLs in order of first appearance instead of by value.
# Each section's new URLs are then one unbroken run of numbers -- its dictionary is mostly runs.
if len(sys.argv) > 2 and sys.argv[2] == 'first':
    u, first = np.unique(allc, return_index=True)
    rank = np.empty(u.size, np.int64); rank[np.argsort(first, kind='stable')] = np.arange(u.size)
    pos = np.searchsorted(u, allc)
    allc = rank[pos]
    print('--- global numbers renumbered by first appearance ---')
    for name, lo, hi in (("counter 62's region", ra, rb), ('whole column', 0, -(-N // FR))):
        acc = np.zeros(6); rows = 0
        for f in range(lo, hi):
            cc = allc[f * FR:min(N, (f + 1) * FR)]
            r = section(cc); rows += cc.size
            acc += [r[0], r[1], r[2], r[3], 0, r[5]]
        b = lambda x: x * 8 / rows
        print('%s: 4-byte numbers as today %.1f | display (first seen) %.1f + dictionary (gaps) %.1f = %.1f bits/row'
              % (name, b(acc[5]), b(acc[2]), b(acc[3]), b(acc[2] + acc[3])), flush=True)
