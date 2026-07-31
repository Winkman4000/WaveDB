import sys, re, time, cProfile, pstats
sys.path.insert(0, '/workspace/WaveDB/src')
sys.path.insert(0, '/workspace/WaveDB/bench')
import numpy as np
from wdb_db import Database
from megaboard_queries import QUERIES
import wdb_kernels; wdb_kernels.warm()

rs = []
for ln in open('/workspace/data/megaboard_v3_26.log', errors='ignore'):
    m = re.match(r'(\S+)\s+OK\s+wave=\s*([\d.]+)s duck=\s*([\d.]+)s x\s*([\d.]+)', ln)
    if m and float(m.group(4)) < 1.0:
        rs.append((m.group(1), float(m.group(2)), float(m.group(4))))
losers = {lab for lab, w, x in rs}
print('%d losers on v3.26:' % len(losers))
for lab, w, x in sorted(rs, key=lambda t: t[2]):
    print('  %-16s x%.2f  %.2fs' % (lab, x, w))
QS = {t[0]: t[2] for t in QUERIES if t[0] in losers}
db = Database.open('/workspace/data/fjdb3')
db.run('SELECT COUNT(*) FROM hits')
agg = {}
for lab, q in QS.items():
    try:
        db.run(q)
        pr = cProfile.Profile()
        pr.enable(); db.run(q); pr.disable()
        for (fn, ln_, name), (cc, nc, tt, ct, cal) in pstats.Stats(pr).stats.items():
            if tt < 0.004:
                continue
            key = '%s:%d %s' % (fn.split('/')[-1], ln_, name[:36])
            agg.setdefault(key, [0.0, {}])
            agg[key][0] += tt
            agg[key][1][lab] = tt
    except Exception as e:
        print('SKIP %s %s' % (lab, str(e)[:40]))
print()
print('COMMON DENOMINATORS (by total ms across losers):')
print('%-50s %8s %3s  heaviest' % ('function', 'total_ms', 'nq'))
for key, (tot, qs) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:14]:
    heav = sorted(qs.items(), key=lambda kv: -kv[1])[:3]
    print('%-50s %8.0f %3d  %s' % (key[:50], tot * 1000, len(qs),
          ' '.join('%s:%.0f' % (l, v * 1000) for l, v in heav)))
print('DONE')
