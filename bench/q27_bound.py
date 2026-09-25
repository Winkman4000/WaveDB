"""Jackson's bound for Q27: a website's LONGEST non-empty URL is a ceiling on its average URL length.
Once the 25th-best average is known, any website whose longest URL is below it cannot be in the top 25.
How many of the 100 kept websites would that rule out? (numbers only, not timed)
Usage: PYTHONPATH=src python bench/q27_bound.py DB_DIR
"""
import sys
import numpy as np, pandas as pd
import wdb_db

db = wdb_db.Database.open(sys.argv[1])
seg = db.open_segment(db.cat.segment_paths('hits')[0], 'hits')
cc = np.asarray(seg._raw_codes('CounterID')).astype(np.int64)
uc = np.asarray(seg._raw_codes('URL')).astype(np.int64)
lens = np.asarray(seg.dict_charlens('URL')).astype(np.int64)
z = seg.fetch('URL', 0)
ne = uc != 0 if z in ('', b'') else np.ones(uc.size, bool)
df = pd.DataFrame({'c': cc[ne], 'l': lens[uc[ne]]})
g = df.groupby('c')['l'].agg(['count', 'sum', 'max', 'mean'])
g = g[g['count'] > 100000]
avg_sorted = np.sort(g['mean'].to_numpy())[::-1]
bar = avg_sorted[24]
out = g[g['max'] < bar]
print('kept websites %d; 25th-best average %.1f characters' % (len(g), bar))
print('longest URL per website: min %d, median %d, max %d' % (g['max'].min(), g['max'].median(), g['max'].max()))
print('websites whose longest URL is below the 25th-best average (ruled out by the bound): %d of %d, holding %d of %d rows'
      % (len(out), len(g), int(out['count'].sum()), int(g['count'].sum())))
q = np.percentile(g['max'], [10, 25, 50, 75, 90])
print('longest-URL percentiles (10/25/50/75/90): %s; averages: min %.1f, median %.1f, max %.1f'
      % (q.astype(int).tolist(), g['mean'].min(), g['mean'].median(), g['mean'].max()))
