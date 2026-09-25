"""Would bitpack-plus (enc 10) pay on a wide column? enc 10 today: 4,096-row blocks, each elects
plain bitpack or run tokens (u16 run length + u16 value) by the profit formula, only for codes <= 16
bits. Here the same election at any width -- run token = 16-bit length + a `bits`-wide value -- plus a
third choice, the linear pattern (a stretch where each code = the previous + a fixed step: 16-bit
length + start + step). Whichever is cheapest per block. Size against plain bitpack and today's zstd.
Usage: PYTHONPATH=src python bench/bplus_wide.py DB_DIR COL [COL ...]
"""
import sys, glob
import numpy as np
import wdb_engine

B = 4096
seg = wdb_engine.Segment(glob.glob(sys.argv[1] + '/*.wdb')[0])
N = int(seg.N)
for col in sys.argv[2:]:
    c = seg.cols[col]
    bits = max(1, int(int(c['V']) - 1).bit_length())
    x = np.asarray(seg._raw_codes(col)).astype(np.int64)
    nb = -(-N // B)
    rows = np.minimum(np.arange(B, N + B, B), N) - np.arange(0, N, B)
    blk = np.arange(N) // B
    # equal runs: a new run starts where the code changes or a block starts
    st_eq = np.r_[True, x[1:] != x[:-1]] | (np.arange(N) % B == 0)
    runs = np.bincount(blk[st_eq], minlength=nb)
    # linear stretches: a new stretch starts where the step changes (or a block starts)
    d = np.diff(x)
    st_ln = np.r_[True, True, d[1:] != d[:-1]] | (np.arange(N) % B == 0)
    st_ln[1:][np.arange(1, N) % B == 1] = True        # the second row of a block opens its step
    lins = np.bincount(blk[st_ln], minlength=nb)
    cost_bp = rows * bits
    cost_run = runs * (16 + bits)
    cost_lin = lins * (16 + 2 * bits)
    best = np.minimum(np.minimum(cost_bp, cost_run), cost_lin)
    pick = np.argmin(np.stack([cost_bp, cost_run, cost_lin]), 0)
    mb = lambda bits_: bits_.sum() / 8 / 1e6
    print('%-12s %2d bits: plain bitpack %.1f MB | bitpack-plus (runs + linear) %.1f MB (+%.1f MB directory) | '
          'today %.1f MB | blocks electing bitpack %d, runs %d, linear %d of %d; mean run %.2f rows'
          % (col, bits, mb(cost_bp), mb(best), nb * 8 / 1e6, int(c.get('czlen') or 0) / 1e6,
             int((pick == 0).sum()), int((pick == 1).sum()), int((pick == 2).sum()), nb, N / st_eq.sum()), flush=True)
