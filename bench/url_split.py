"""Jackson's display/dictionary split of a hash column: per section, keep only as many leading bits of
each hash as that section needs to tell its hashes apart (the display, per row); the rest of each
hash (the suffix) lives in the section's dictionary. How many bits does a section need?
Measured on counter 62's region of URLHash (frames 726..918), for two section sizes.
Usage: PYTHONPATH=src python bench/url_split.py DB_DIR
"""
import sys
import numpy as np
sys.path.insert(0, 'bench')
import link_ceiling as L, wdb_funnel

db, seg = L._open(sys.argv[1])
c62 = wdb_funnel._code_of(seg, 'CounterID', 62)
runs, _ = L._blocks_runs(seg, 'CounterID', c62)
FR = int(seg.cols['URLHash']['BR'])
a, b = runs[0][0] // FR * FR, min(int(seg.N), -(-runs[-1][1] // FR) * FR)
codes = np.asarray(seg.codes_at('URLHash', np.arange(a, b, dtype=np.int64))).astype(np.int64)
u, inv = np.unique(codes, return_inverse=True)
vals = np.asarray(seg.values_at('URLHash', u)).astype(np.int64).view(np.uint64)[inv]
print('region rows %d, distinct %d, dictionary code today %d bits'
      % (codes.size, u.size, int(seg.cols['URLHash']['V']).bit_length()))


def bitlen(x):
    x = x.astype(np.uint64); n = np.zeros(x.shape, np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        m = x >= (np.uint64(1) << np.uint64(s))
        n[m] += s; x[m] >>= np.uint64(s)
    return n + (x > 0)


def need_bits(v):
    """Fewest leading bits that tell every distinct hash in v apart."""
    s = np.unique(v)
    if s.size < 2:
        return 0
    return int((64 - bitlen(s[1:] ^ s[:-1])).max()) + 1


for S in (65536, 8192):
    bs, ds = [], []
    for i in range(0, vals.size, S):
        sec = vals[i:i + S]
        bs.append(need_bits(sec)); ds.append(np.unique(sec).size)
    bs, ds = np.array(bs), np.array(ds)
    dict_bits = float(((64 - bs) * ds).sum()) / vals.size        # suffixes, spread over the rows
    print('section %6d rows: %d sections, distinct per section median %d; display bits per row '
          'median %d, min %d, max %d; suffix dictionary adds %.1f bits/row'
          % (S, bs.size, int(np.median(ds)), int(np.median(bs)), bs.min(), bs.max(), dict_bits))
