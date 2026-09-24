"""The general form of the top-k start: no stored lists. Read the first bytes of every survivor's URL
code, count the piles those bytes make (each pile's count bounds every URL inside it), and only the
piles that reach the bar need the rest of their bytes. How many survivor rows is that?
Usage: PYTHONPATH=src python bench/url_buckets.py DB_DIR
"""
import sys
import numpy as np
sys.path.insert(0, 'bench')
import link_ceiling as L

db, seg = L._open(sys.argv[1])
_, rows, _ = L.by_hand(seg, 'C')
su = np.asarray(seg.codes_at('URLHash', rows)).astype(np.int64)
exact = np.sort(np.unique(su, return_counts=True)[1])[::-1]
TH = int(exact[109])
print('survivors %d, bar %d' % (rows.size, TH))
for nb in (1, 2, 3):
    for side, key in (('low', su & ((1 << (8 * nb)) - 1)), ('high', su >> (25 - 8 * nb))):
        k, inv, c = np.unique(key, return_inverse=True, return_counts=True)
        need = c[inv] >= TH                      # rows whose pile could still hold a competitor
        print('%d byte(s), %-4s: %6d piles; %6d rows (%4.1f%%) sit in piles reaching %d and need the rest'
              % (nb, side, k.size, int(need.sum()), 100.0 * need.mean(), TH))
