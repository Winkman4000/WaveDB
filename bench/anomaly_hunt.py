"""
anomaly_hunt -- Jackson's fingerprint method, made permanent.

Profile every board query (window whales timed elsewhere), tally each engine
function's cost PER QUERY, and flag pairs where a function runs >=5x its own
median-across-queries. A strong function running hot for one query is either a
smuggled process or a conditional flaw -- diskpair at 900ms against its usual
60ms is how the stair-walk and grid2-distinct were found.

Usage (pod): /workspace/venv/bin/python3 bench/anomaly_hunt.py [fixture_dir]
"""
import sys, re, time, cProfile, pstats, json
sys.path.insert(0, '/workspace/WaveDB/src')
import numpy as np
from wdb_db import Database
import wdb_kernels; wdb_kernels.warm()

txt = open('/workspace/WaveDB/bench/megaboard_queries.py').read()
ALL = re.findall(r"\('([\w-]+)',\s*'[A-Z]',\s*\"(.+?)\"\)", txt, re.S)
SKIP = {'cq-aboveavg', 'w-partavg', 'w-runsum', 'w-frame-avg', 'w-q-mixed'}
db = Database.open(sys.argv[1] if len(sys.argv) > 1 else '/workspace/data/fjdb3')
db.run('SELECT COUNT(*) FROM hits')
per = {}                                  # func -> {query: ms}
for lab, q in ALL:
    if lab in SKIP:
        continue
    try:
        db.run(q)                         # warm
        pr = cProfile.Profile()
        pr.enable(); db.run(q); pr.disable()
        for (fn, ln, name), (cc, nc, tt, ct, callers) in pstats.Stats(pr).stats.items():
            if tt < 0.003:
                continue
            key = '%s:%d %s' % (fn.split('/')[-1], ln, name[:36])
            per.setdefault(key, {})[lab] = per.setdefault(key, {}).get(lab, 0) + tt * 1000
    except Exception as e:
        print('SKIP %s: %s' % (lab, str(e)[:40]))
out = []
for key, qs in per.items():
    if len(qs) < 4:
        continue
    vals = np.array(list(qs.values()))
    med = float(np.median(vals))
    if med < 1.0:
        med = 1.0
    for lab, ms in qs.items():
        if ms > max(5 * med, med + 60):
            out.append((ms - med, key, lab, ms, med, len(qs)))
out.sort(reverse=True)
print('%-46s %-14s %8s %8s %4s' % ('function', 'query', 'ms', 'typ_ms', 'nq'))
for excess, key, lab, ms, med, nq in out[:16]:
    print('%-46s %-14s %8.0f %8.0f %4d' % (key[:46], lab, ms, med, nq))
print('DONE')
