"""Jackson's two Q40 ideas, the numbers behind them:
 1. start from the most common URLs (stored counts) and ask whether any other URL can even compete
    with the answer's counts -- how long must the candidate list be, at each level the counts could
    be stored at (whole table, counter 62's region, counter 62's rows)?
 2. read only part of each value: with that candidate list, how many survivor rows does 1 or 2 bytes
    of the URL code already rule out (no candidate shares those bytes)?
Usage: PYTHONPATH=src python bench/url_topk.py DB_DIR
"""
import sys, time
import numpy as np
sys.path.insert(0, 'bench')
import link_ceiling as L, wdb_funnel

db, seg = L._open(sys.argv[1])
_, rows, _ = L.by_hand(seg, 'C')
V = int(seg.cols['URLHash']['V'])
su = np.asarray(seg.codes_at('URLHash', rows)).astype(np.int64)
scnt = np.bincount(su, minlength=V)
srt = np.sort(scnt[scnt > 0])[::-1]
TH = int(srt[109])                               # the 110th pile: OFFSET 100 LIMIT 10 ends here
print('survivor piles: top counts %s ... 101st-110th %s; the bar to compete: %d views'
      % (srt[:5].tolist(), srt[100:110].tolist(), TH))
top = np.argsort(-scnt, kind='stable')[:110]

t = time.perf_counter(); allc = np.asarray(seg._raw_codes('URLHash')).astype(np.int64)
print('(whole-column decode for the census: %.0f ms)' % ((time.perf_counter() - t) * 1e3))
c62 = wdb_funnel._code_of(seg, 'CounterID', 62)
runs, _ = L._blocks_runs(seg, 'CounterID', c62)
p62 = wdb_funnel.positions(seg, 'CounterID', c62)
levels = [('whole table', np.bincount(allc, minlength=V)),
          ("counter 62's region", np.bincount(np.concatenate([allc[a:b] for a, b in runs]), minlength=V)),
          ("counter 62's rows", np.bincount(allc[p62], minlength=V))]
for name, cnt in levels:
    need = int((cnt >= TH).sum())                # every URL that could reach the bar
    rank = np.empty(V, np.int64); rank[np.argsort(-cnt, kind='stable')] = np.arange(V)
    print('%-20s URLs with >= %d rows (could compete): %d; answer URLs rank at most #%d'
          % (name, TH, need, int(rank[top].max()) + 1))
    cand = np.flatnonzero(cnt >= TH)
    for nb in (1, 2, 3):
        m = (1 << (8 * nb)) - 1
        hit = np.zeros(m + 1, bool); hit[cand & m] = True
        left = int(hit[su & m].sum())
        print('    read %d byte(s) of each survivor: %d of %d rows still look like a candidate'
              % (nb, left, rows.size))
