"""Megaboard non-exact audit: different-correct-answers or actual bug?

Case 1/2 (t-bool, g-lower): classify the errors.
Case 3/4 (w-frame-avg, w-frame-max): THE TIEBREAK EXPERIMENT. Our lane order is file order
within EventTime ties (stable scatter). Force duck onto the identical order with
file_row_number as the final ORDER BY key inside OVER. Exact match => the megaboard
disagreement was pure tie-order (two correct answers). Mismatch => real bug.
"""
import sys, time
sys.path.insert(0, 'src')
import duckdb
from wdb_db import Database
import wdb_kernels
wdb_kernels.warm()

db = Database.open('/workspace/data/fjdb')
con = duckdb.connect()
PQ = '/workspace/data/hits.parquet'

def rows_of(r):
    return r[0] if isinstance(r, tuple) else r

# ---- case 1: t-bool ----
cols = db.cat.column_names('hits')
print('CASE t-bool: IsMobile in catalog?', 'IsMobile' in cols,
      '| catalog has %d cols' % len(cols), flush=True)

# ---- case 2: g-lower ----
try:
    db.run("SELECT LOWER(SearchPhrase) AS l, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY l ORDER BY c DESC LIMIT 10")
    print('CASE g-lower: ran?! unexpected', flush=True)
except NotImplementedError as e:
    print('CASE g-lower: LOUD DECLINE ->', str(e)[:80], flush=True)
except Exception as e:
    print('CASE g-lower: OTHER ERROR', type(e).__name__, str(e)[:80], flush=True)

# ---- cases 3/4: the tiebreak experiment ----
CTE = ("WITH hits AS (SELECT * REPLACE ((DATE '1970-01-01'+EventDate) AS EventDate,"
       "(TIMESTAMP '1970-01-01'+to_seconds(EventTime)) AS EventTime)"
       " FROM read_parquet('%s', file_row_number=true))" % PQ)

def experiment(label, wave_q, duck_q, keyfun):
    t = time.perf_counter(); w = rows_of(db.run(wave_q)); wt = time.perf_counter() - t
    t = time.perf_counter(); d = con.execute(CTE + ' ' + duck_q).fetchall(); dt = time.perf_counter() - t
    ws = sorted(keyfun(r) for r in w)
    ds = sorted(keyfun(r) for r in d)
    print('CASE %s: wave_rows=%d duck_tiebreak_rows=%d EXACT_MATCH=%s (wave %.1fs duck %.1fs)'
          % (label, len(w), len(d), ws == ds, wt, dt), flush=True)
    if ws != ds:
        both = set(ws) & set(ds)
        print('   overlap=%d wave_only=%d duck_only=%d'
              % (len(both), len(set(ws) - both), len(set(ds) - both)), flush=True)
        for x in list(set(ws) - set(ds))[:3]:
            print('   wave-only sample:', x, flush=True)
        for x in list(set(ds) - set(ws))[:3]:
            print('   duck-only sample:', x, flush=True)

experiment(
    'w-frame-avg',
    "SELECT UserID, AVG(ResolutionWidth) OVER (PARTITION BY UserID ORDER BY EventTime ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS ma FROM hits QUALIFY ma > 2000",
    "SELECT UserID, AVG(ResolutionWidth) OVER (PARTITION BY UserID ORDER BY EventTime, file_row_number ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS ma FROM hits QUALIFY ma > 2000",
    lambda r: (int(r[0]), round(float(r[1]), 4)))

experiment(
    'w-frame-max',
    "SELECT RegionID, MAX(ResolutionWidth) OVER (PARTITION BY RegionID ORDER BY EventTime ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS sx FROM hits QUALIFY sx = 0",
    "SELECT RegionID, MAX(ResolutionWidth) OVER (PARTITION BY RegionID ORDER BY EventTime, file_row_number ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS sx FROM hits QUALIFY sx = 0",
    lambda r: (int(r[0]), int(r[1])))

print('AUDIT COMPLETE', flush=True)
