"""Jackson's back-reference slot: instead of the value, a row may hold "same as the row g back".
How far back is each row's previous occurrence of the same value? (never = first time seen)
Usage: PYTHONPATH=src python bench/backref_gaps.py DB_DIR COL
"""
import sys, glob
import numpy as np
import wdb_engine

seg = wdb_engine.Segment(glob.glob(sys.argv[1] + '/*.wdb')[0])
x = np.asarray(seg._raw_codes(sys.argv[2])).astype(np.int64)
N = x.size
o = np.argsort(x, kind='stable')                      # rows grouped by value, ascending inside
same = np.r_[False, x[o][1:] == x[o][:-1]]
gap = np.full(N, -1, np.int64)
gap[o[same]] = o[same] - o[np.flatnonzero(same) - 1]  # distance back to the previous occurrence
never = int((gap < 0).sum())
print('%s: %d rows, first time seen (no earlier copy) %d (%.1f%%)' % (sys.argv[2], N, never, 100.0 * never / N))
edges = [1, 2, 4, 16, 256, 4096, 65536, 1 << 20, 1 << 40]
lo = 1
for hi in edges[1:]:
    k = int(((gap >= lo) & (gap < hi)).sum())
    print('  previous copy %9d .. %-12d rows back: %5.1f%%  (a gap that fits in %2d bits)'
          % (lo, hi - 1, 100.0 * k / N, int(hi - 1).bit_length()))
    lo = hi
