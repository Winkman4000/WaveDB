"""How many distinct URLs does each website use? (the 100 websites past Q27's cut, and all of them)
Usage: PYTHONPATH=src python bench/q27_distinct.py DB_DIR
"""
import sys
import numpy as np
import wdb_db

db = wdb_db.Database.open(sys.argv[1])
seg = db.open_segment(db.cat.segment_paths('hits')[0], 'hits')
cc = np.asarray(seg._raw_codes('CounterID')).astype(np.int64)
uc = np.asarray(seg._raw_codes('URL')).astype(np.int64)
ne = uc != 0
V = int(seg.cols['URL']['V'])
pair = np.unique(cc[ne] * V + uc[ne])
site = pair // V
d = np.bincount(site, minlength=int(seg.cols['CounterID']['V']))
rows = np.bincount(cc[ne], minlength=d.size)
k = rows > 100000
print('distinct (website, URL) pairs: %d over all websites; %d over the 100 kept; distinct URLs in the table %d'
      % (pair.size, int(d[k].sum()), V))
dk = np.sort(d[k])
print('distinct URLs per kept website: min %d, median %d, max %d; rows per distinct URL (kept) %.1f'
      % (dk[0], int(np.median(dk)), dk[-1], rows[k].sum() / d[k].sum()))
