"""THE SCORE: the true-cold board on one database (bench/true_cold.py per query, fresh process each,
REPS runs, best kept), then ClickBench's score against the fair-trial referees -- the geometric mean of
(t + 10 ms) / (best + 10 ms), best over every engine -- plus totals and the queries each engine wins.
Usage: PYTHONPATH=src python bench/board_score.py DB_DIR QUERIES_SQL OUT_JSONL [reps]
"""
import sys, os, json, math, subprocess

DB, SQLF, OUT = sys.argv[1:4]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
HERE = os.path.dirname(os.path.abspath(__file__))
REF = '/workspace/session_2026-09-23/fair_%s.jsonl'
qs = [l.strip() for l in open(SQLF) if l.strip() and not l.strip().startswith('--')]
mine = {'cold': {}, 'hot': {}}
with open(OUT, 'w') as f:
    for rep in range(REPS):
        for q in range(len(qs)):
            p = subprocess.run([sys.executable, os.path.join(HERE, 'true_cold.py'), DB, SQLF, str(q)],
                               capture_output=True, text=True, timeout=900)
            ln = [l for l in p.stdout.splitlines() if l.startswith('{')]
            r = json.loads(ln[-1]) if ln else {'q': q, 'err': True}
            r['rep'] = rep; f.write(json.dumps(r) + '\n'); f.flush()
            for k in ('cold', 'hot'):
                if isinstance(r.get(k), (int, float)):
                    mine[k][q] = min(mine[k].get(q, 1e18), r[k])
refs = {e: {json.loads(l)['q']: json.loads(l) for l in open(REF % e) if l.strip()}
        for e in ('umbra', 'clickhouse', 'duckdb')}
for kind in ('cold', 'hot'):
    eng = {'wavedb': mine[kind]}
    for e in refs:
        eng[e] = {q: refs[e][q][kind] for q in refs[e] if isinstance(refs[e][q].get(kind), (int, float))}
    Q = sorted(set.intersection(*(set(v) for v in eng.values())))
    best = {q: min(eng[e][q] for e in eng) for q in Q}
    print('%s (%d queries):' % (kind.upper(), len(Q)))
    for e in eng:
        sc = math.exp(sum(math.log((eng[e][q] + 10) / (best[q] + 10)) for q in Q) / len(Q))
        wins = sum(1 for q in Q if eng[e][q] == best[q])
        print('  %-10s score %.2f  total %6.1f s  fastest on %2d queries' % (e, sc, sum(eng[e][q] for q in Q) / 1e3, wins))
    missing = [q for q in range(len(qs)) if q not in mine[kind]]
    if missing:
        print('  wavedb had no time on', missing)
