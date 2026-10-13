"""THE Q13 BOARD, step 1: the column profiles of hits.

A profile = the column's passport (storage mode, code encoding, type) + its V class (how many distinct
values, in bands). Two columns with the same profile should travel the same roads at the same cost, so
the board times one representative per profile. Per column we also record the most common value and its
share -- the original Q13 filters its group key's most common value away (SearchPhrase <> '').

Usage: python bench/q13_profiles.py SRC DB_DIR HITS_PARQUET OUT_JSON
"""
import sys, os, json, time, collections
sys.path.insert(0, sys.argv[1])
import duckdb
from wdb_db import Database
import wdb_join

SRC, DBD, PQ, OUT = sys.argv[1:5]
T = {0: 'int', 1: 'str', 2: 'float', 3: 'time'}
HITS_CTE = ("WITH hits AS (SELECT * REPLACE ("
            "(DATE '1970-01-01' + EventDate) AS EventDate, "
            "(TIMESTAMP '1970-01-01' + to_seconds(EventTime)) AS EventTime) "
            "FROM read_parquet('%s'))" % PQ)


def vclass(V, N):
    if V >= 0.8 * N: return 'unique'
    if V <= 2: return 'binary'
    if V <= 256: return 'tiny'
    if V <= 65536: return 'small'
    if V <= 4194304: return 'mid'
    return 'large'


db = Database.open(DBD)
seg, _ = wdb_join._solo_segment(db, 'hits')
N = int(seg.N)
con = duckdb.connect()
cols = []
for nm in seg.order:
    m = seg.cols[nm]
    V = int(m['V'])
    t0 = time.perf_counter()
    top = con.execute(HITS_CTE + ' SELECT "%s", COUNT(*) AS c FROM hits GROUP BY 1 ORDER BY c DESC LIMIT 1' % nm).fetchone()
    tv = top[0]
    if hasattr(tv, 'isoformat'):
        tv = str(tv)
    rec = {'col': nm, 'mode': m.get('mode'), 'enc': m.get('code_enc'), 'dt': T.get(m.get('dt'), m.get('dt')),
           'V': V, 'has_null': bool(m.get('has_null')), 'vclass': vclass(V, N),
           'top': tv, 'top_share': round(top[1] / N, 4), 'duck_s': round(time.perf_counter() - t0, 2)}
    rec['passport'] = 'm%s/e%s/%s' % (rec['mode'], rec['enc'], rec['dt'])
    rec['profile'] = rec['passport'] + '/' + rec['vclass']
    cols.append(rec)
    print('%-22s %-14s V=%-10d %-7s top=%r (%.1f%%)' % (nm, rec['passport'], V, rec['vclass'], str(tv)[:30], 100 * rec['top_share']), flush=True)

prof = collections.defaultdict(list)
for r in cols:
    prof[r['profile']].append(r['col'])
json.dump({'N': N, 'columns': cols, 'profiles': prof}, open(OUT, 'w'), indent=1, default=str)
print('\ncolumns %d  passports %d  profiles %d' % (len(cols), len({r['passport'] for r in cols}), len(prof)))
for k, v in sorted(prof.items(), key=lambda kv: kv[0]):
    print('  %-26s %3d  %s' % (k, len(v), ', '.join(v[:6]) + (' ...' if len(v) > 6 else '')))
