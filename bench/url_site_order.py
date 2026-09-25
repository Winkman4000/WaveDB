"""Jackson's site-grouped URL dictionary: number the URLs grouped by the website (CounterID) that uses
them most, alphabetical inside each website, instead of alphabetical over the whole table. Measures:
  1. how many URLs belong to exactly one website (then a URL number alone says which website a row is)
  2. per-row URL numbers, zstd per 65,536-row frame: alphabetical vs site-grouped numbering
  3. the front-coded dictionary: suffix bytes (after the prefix shared with the previous entry) and
     their zstd size, alphabetical vs site-grouped order
Usage: PYTHONPATH=src python bench/url_site_order.py DB_DIR
"""
import sys, time
import numpy as np
import numba
import zstandard as zs
import wdb_db

t0 = time.perf_counter()
db = wdb_db.Database.open(sys.argv[1])
seg = db.open_segment(db.cat.segment_paths('hits')[0], 'hits')
cc = np.asarray(seg._raw_codes('CounterID')).astype(np.int64)
uc = np.asarray(seg._raw_codes('URL')).astype(np.int64)
V = int(seg.cols['URL']['V']); K = int(seg.cols['CounterID']['V'])
N = uc.size
# 1. websites per URL, and each URL's main website (most rows)
pair, pc = np.unique(uc * K + cc, return_counts=True)
pu, ps = pair // K, pair % K
nsite = np.bincount(pu, minlength=V)
o = np.lexsort((-pc, pu))                       # per URL, its biggest website first
first = np.r_[True, pu[o][1:] != pu[o][:-1]]
owner = np.full(V, -1, np.int64); owner[pu[o][first]] = ps[o][first]
single = nsite == 1
rows_single = int(single[uc].sum())
print('URLs %d; used by exactly one website %d (%.2f%%); rows whose URL is single-website %d of %d (%.2f%%)  [%.0f s]'
      % (V, int(single.sum()), 100.0 * single.mean(), rows_single, N, 100.0 * rows_single / N, time.perf_counter() - t0), flush=True)
# 2. the new numbering: by (main website, alphabetical) -- codes are alphabetical already
neworder = np.lexsort((np.arange(V), owner))    # new code k -> old code neworder[k]
rank = np.empty(V, np.int64); rank[neworder] = np.arange(V)
cz = zs.ZstdCompressor(level=9)
FR = 65536
def frames(x):
    return sum(len(cz.compress(x[i:i + FR].astype(np.uint32).tobytes())) for i in range(0, x.size, FR))
a = frames(uc); b = frames(rank[uc])
print('per-row URL numbers, zstd per 65,536 rows: alphabetical %.1f MB (%.1f bits/row), site-grouped %.1f MB (%.1f bits/row)  [%.0f s]'
      % (a / 1e6, a * 8 / N, b / 1e6, b * 8 / N, time.perf_counter() - t0), flush=True)
# 3. the front-coded dictionary in both orders
vals = seg.dict_vals('URL')
lens = np.fromiter((len(v) for v in vals), np.int64, count=V)
off = np.zeros(V + 1, np.int64); np.cumsum(lens, out=off[1:])
blob = np.frombuffer(b''.join(v if isinstance(v, (bytes, bytearray)) else v.encode() for v in vals), np.uint8)
print('dictionary: %d entries, %.1f MB of URL bytes  [%.0f s]' % (V, blob.size / 1e6, time.perf_counter() - t0), flush=True)


@numba.njit(cache=True)
def suffixes(blob, off, order, out_len, out_buf):
    """per entry in `order`: bytes shared with the previous entry, the suffix appended to out_buf"""
    p = 0
    prev = -1
    for k in range(order.size):
        i = order[k]
        a0, a1 = off[i], off[i + 1]
        s = 0
        if prev >= 0:
            b0, b1 = off[prev], off[prev + 1]
            m = min(a1 - a0, b1 - b0)
            while s < m and blob[a0 + s] == blob[b0 + s]:
                s += 1
        for j in range(a0 + s, a1):
            out_buf[p] = blob[j]; p += 1
        out_len[k] = a1 - a0 - s
        prev = i
    return p


for name, order in (('alphabetical', np.arange(V, dtype=np.int64)), ('site-grouped', neworder.astype(np.int64))):
    ol = np.empty(V, np.int64); ob = np.empty(blob.size, np.uint8)
    n = suffixes(blob, off, order, ol, ob)
    z = sum(len(cz.compress(ob[i:i + (64 << 20)].tobytes())) for i in range(0, n, 64 << 20))
    print('  %-13s suffix bytes %.1f MB (shared prefixes save %.1f%%), suffixes zstd %.1f MB  [%.0f s]'
          % (name, n / 1e6, 100.0 * (1 - n / blob.size), z / 1e6, time.perf_counter() - t0), flush=True)
