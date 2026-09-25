"""Jackson's back-reference, variable width: per row, 1 flag bit, then either the full value (bits wide)
or the distance back to the previous copy of the same value, written short when the distance is short.
Rows live in blocks of B rows with a start table (8 bytes a block), and a distance may only point inside
its own block, so any row costs one jump plus decoding its block, and no block needs another.
Distance codes compared: Elias gamma (2*floor(log2 g)+1 bits) and a 4-bit length class + the bits.
Usage: PYTHONPATH=src python bench/backref_var.py DB_DIR COL
"""
import sys, glob
import numpy as np
import wdb_engine

seg = wdb_engine.Segment(glob.glob(sys.argv[1] + '/*.wdb')[0])
col = sys.argv[2]
x = np.asarray(seg._raw_codes(col)).astype(np.int64)
N = x.size
bits = max(1, int(int(seg.cols[col]['V']) - 1).bit_length())
o = np.argsort(x, kind='stable')
same = np.r_[False, x[o][1:] == x[o][:-1]]
prev = np.full(N, -1, np.int64)
prev[o[same]] = o[np.flatnonzero(same) - 1]
r = np.arange(N, dtype=np.int64)
del o, same


def blen(v):
    v = v.astype(np.uint64); n = np.zeros(v.shape, np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        m = v >= (np.uint64(1) << np.uint64(s))
        n[m] += s; v[m] >>= np.uint64(s)
    return n + (v > 0)


cz = int(seg.cols[col].get('czlen') or 0)
print('%s: %d rows, %d bits a value; today %.1f MB (%.1f bits/row); plain bitpack %.1f MB (%d bits/row)'
      % (col, N, bits, cz / 1e6, cz * 8 / N, N * bits / 8e6, bits))
for B in (256, 4096, 65536, 1 << 20):
    ok = (prev >= 0) & (prev // B == r // B)
    g = np.where(ok, r - prev, 1)
    L = blen(g)
    gamma = np.where(ok, 1 + 2 * (L - 1) + 1, 1 + bits)
    klass = np.where(ok, 1 + 4 + L, 1 + bits)
    tbl = (-(-N // B)) * 64
    for name, c in (('gamma', gamma), ('length class', klass)):
        tot = int(c.sum()) + tbl
        print('  block %8d rows: repeats inside the block %5.1f%% | %-12s %.1f MB (%.1f bits/row)'
              % (B, 100.0 * ok.mean(), name, tot / 8e6, tot / N))
