"""THE CLUSTER A/B: the same ClickBench board on two databases loaded by the same command, differing
only in the operator's --cluster-by. Per query, interleaved A B A B (reps), each run bench/true_cold.py
(fresh process, files + kernels evicted; cold, then hot = best of two). Prints per query and the
ClickBench cold/hot scores of each against the fair-trial referees.

Usage: PYTHONPATH=src python bench/cluster_ab.py DB_A DB_B QUERIES_SQL OUT_JSONL [reps]
"""
import sys, os, json, subprocess

A, B, SQLF, OUT = sys.argv[1:5]
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 2
HERE = os.path.dirname(os.path.abspath(__file__))
REF = '/workspace/session_2026-09-23/fair_%s.jsonl'
qs = [l.strip() for l in open(SQLF) if l.strip() and not l.strip().startswith('--')]


def tc(db, q):
    p = subprocess.run([sys.executable, os.path.join(HERE, 'true_cold.py'), db, SQLF, str(q)],
                       capture_output=True, text=True, timeout=900)
    ln = [l for l in p.stdout.splitlines() if l.startswith('{')]
    return json.loads(ln[-1]) if ln else {'err': (p.stderr or '?')[-200:]}


def score(times, best):
    import math
    r = [math.log((times[q] + 10) / (best[q] + 10)) for q in times]
    return math.exp(sum(r) / len(r))


res = {'A': {}, 'B': {}}
with open(OUT, 'w') as f:
    for q in range(len(qs)):
        for rep in range(REPS):
            for tag, db in (('A', A), ('B', B)):
                r = tc(db, q); r['db'] = tag; r['q'] = q; r['rep'] = rep
                f.write(json.dumps(r) + '\n'); f.flush()
                res[tag].setdefault(q, []).append(r)
        def best_of(tag, k):
            v = [r.get(k) for r in res[tag][q] if isinstance(r.get(k), (int, float))]
            return min(v) if v else None
        print('q%02d  A cold %s hot %s | B cold %s hot %s' % (q, best_of('A', 'cold_ms') or best_of('A', 'cold'),
              best_of('A', 'hot_ms') or best_of('A', 'hot'), best_of('B', 'cold_ms') or best_of('B', 'cold'),
              best_of('B', 'hot_ms') or best_of('B', 'hot')), flush=True)

# THE SCORES: ClickBench's geometric mean of (t + 10 ms) / (best + 10 ms), best over the referees and
# both databases; per database the better of its runs.
refs = {}
for e in ('umbra', 'clickhouse', 'duckdb'):
    refs[e] = {json.loads(l)['q']: json.loads(l) for l in open(REF % e) if l.strip()}
for kind in ('cold', 'hot'):
    ok = [q for q in res['A'] if all(any(isinstance(r.get(kind), (int, float)) for r in res[t][q]) for t in ('A', 'B'))]
    mine = {t: {q: min(r[kind] for r in res[t][q] if isinstance(r.get(kind), (int, float))) for q in ok}
            for t in ('A', 'B')}
    best = {q: min([mine['A'][q], mine['B'][q]] + [refs[e][q][kind] for e in refs if q in refs[e]
                                                    and isinstance(refs[e][q].get(kind), (int, float))])
            for q in mine['A']}
    line = '%s: A score %.2f total %.1f s | B score %.2f total %.1f s' % (
        kind, score(mine['A'], best), sum(mine['A'].values()) / 1e3, score(mine['B'], best), sum(mine['B'].values()) / 1e3)
    for e in refs:
        t9 = {q: refs[e][q][kind] for q in mine['A'] if q in refs[e] and isinstance(refs[e][q].get(kind), (int, float))}
        if len(t9) == len(mine['A']):
            line += ' | %s %.2f' % (e, score(t9, best))
    print(line, flush=True)
