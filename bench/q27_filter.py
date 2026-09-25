"""Q27, Jackson's filter first: drop every website (CounterID) that cannot reach 100,000 views BEFORE any
length work. The load's census holds rows per CounterID, and a website's count with URL <> '' can never
exceed its total, so a total <= 100,000 rules it out without reading a row. Timed in isolation, cold
(fresh process, files evicted): the census read, the per-row website numbers, the keep mask.
Usage: PYTHONPATH=src python bench/q27_filter.py DB_DIR
"""
import sys, os, glob, time
import numpy as np
import wdb_db, wdb_blockstats

d = sys.argv[1]
db = wdb_db.Database.open(d)
seg = db.open_segment(db.cat.segment_paths('hits')[0], 'hits')
src = os.path.dirname(os.path.abspath(wdb_db.__file__))
for f in [f for f in glob.glob(d + '/*') if os.path.isfile(f)] + glob.glob(src + '/__pycache__/*.nb*'):
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
N = int(seg.N)
T = {}
t = time.perf_counter()
cnt = wdb_blockstats.vcnt_from_load(seg, 'CounterID')
T['1. read the stored views-per-website counts'] = (time.perf_counter() - t) * 1e3
t = time.perf_counter()
keep = cnt > 100000
T['2. mark websites with more than 100,000 views in total'] = (time.perf_counter() - t) * 1e3
t = time.perf_counter()
cc = np.asarray(seg._raw_codes('CounterID'))
T['3. unpack every row\'s website number (100M rows)'] = (time.perf_counter() - t) * 1e3
t = time.perf_counter()
rows = np.flatnonzero(keep[cc])
T['4. list the rows that belong to a kept website'] = (time.perf_counter() - t) * 1e3
for k, v in T.items():
    print('%-60s %7.1f ms' % (k, v))
print('%-60s %7.1f ms' % ('TOTAL', sum(T.values())))
print('websites: %d in all, %d kept (%.2f%%); rows kept %d of %d (%.1f%%)'
      % (cnt.size, int(keep.sum()), 100.0 * keep.sum() / cnt.size, rows.size, N, 100.0 * rows.size / N))
# for reference, not timed: how many websites pass the exact test (count with URL <> '' > 100,000)
import wdb_wherescan
z = wdb_wherescan._code_of(seg, 'URL', b'') if True else None
uc = np.asarray(seg._raw_codes('URL'))
exact = np.bincount(cc[uc != z], minlength=cnt.size) if z is not None else cnt
print('exact test (URL <> \'\'): %d websites pass; the stored totals kept %d' % (int((exact > 100000).sum()), int(keep.sum())))
