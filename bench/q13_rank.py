"""THE Q13 BOARD, the ranking: read OUT/results.jsonl and rank every (K, D) pair against the submitted
shape's submitted hot time. Prints the bands, the reads, the worst pairs, and per-profile rows for the K slot and the
D slot; writes OUT/grid.csv (K profile x D profile, hot seconds; T = timeout).

Usage: python bench/q13_rank.py OUT_DIR [--form filtered] [--worst 40]"""
import sys, os, json, csv, argparse, collections, statistics
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--form', default='filtered')
ap.add_argument('--worst', type=int, default=40)
A = ap.parse_args()
R = [json.loads(l) for l in open(os.path.join(A.out, 'results.jsonl')) if l.strip()]
SUB = next((r['sub_hot'] for r in R if r.get('sub_hot')), 0.498798)   # the shape's submitted hot (Q13's before shapes)
if A.form == 'filtered' and not any(r['form'] == 'filtered' for r in R):
    A.form = R[0]['form'] if R else A.form
R = [r for r in R if r['form'] == A.form]
INF = float('inf')


def t(r):
    return r['hot'] if r.get('hot') is not None else INF


def band(r):
    x = t(r) / SUB
    if x == INF: return 'TIMEOUT/error'
    if x <= 1: return 'at or under submitted'
    if x <= 2: return '1-2x'
    if x <= 10: return '2-10x'
    return 'over 10x'


print('pairs %d  (form %s)   submitted hot %.3fs' % (len(R), A.form, SUB))
print('verdicts:', dict(collections.Counter(r.get('verdict', '?') for r in R)))
print('\nbands vs submitted:')
for b in ('at or under submitted', '1-2x', '2-10x', 'over 10x', 'TIMEOUT/error'):
    n = sum(1 for r in R if band(r) == b)
    print('  %-24s %4d' % (b, n))
print('\nreads (pairs, median hot, worst hot):')
by = collections.defaultdict(list)
for r in R:
    by[r.get('read') or r.get('verdict')].append(t(r))
for rd, ts in sorted(by.items(), key=lambda kv: -len(kv[1])):
    fin = [x for x in ts if x < INF]
    print('  %-22s %4d   %8s   %8s' % (rd, len(ts), '%.3f' % statistics.median(fin) if fin else '-',
                                       '%.3f' % max(fin) if fin else '-'))
print('\nworst %d:' % A.worst)
for r in sorted(R, key=t, reverse=True)[:A.worst]:
    print('  %8s  x%-7s %-18s K=%-20s (%-22s) D=%-20s (%s)' % (
        '%.3f' % t(r) if t(r) < INF else 'TIMEOUT', '%.1f' % (t(r) / SUB) if t(r) < INF else '-',
        r.get('read', '-'), r['k'], r['kprof'], r['d'], r['dprof']))


def slot(key, pkey, title):
    print('\n%s (pairs, median, worst, over 2x, general_scan, timeouts, reads):' % title)
    g = collections.defaultdict(list)
    for r in R:
        g[(r[pkey], r[key])].append(r)
    rows = []
    for (p, c), rs in g.items():
        ts = [t(r) for r in rs]; fin = [x for x in ts if x < INF]
        rows.append((max(ts), p, c, len(rs), statistics.median(fin) if fin else INF,
                     sum(1 for x in ts if x > 2 * SUB), sum(1 for r in rs if r.get('read') == 'general_scan'),
                     sum(1 for x in ts if x == INF), collections.Counter(r.get('read', 'T') for r in rs)))
    for w, p, c, n, med, o2, gs, to, rd in sorted(rows, reverse=True):
        print('  %-24s %-20s %3d  %7.3f  %8s  %3d  %3d  %3d  %s' % (
            p, c, n, med, '%.3f' % w if w < INF else 'TIMEOUT', o2, gs, to,
            ', '.join('%s %d' % kv for kv in rd.most_common(3))))


slot('k', 'kprof', 'K slot (group key) by profile')
slot('d', 'dprof', 'D slot (distinct column) by profile')

kp = sorted({r['kprof'] for r in R}); dp = sorted({r['dprof'] for r in R})
cell = {(r['kprof'], r['dprof']): r for r in R}
with open(os.path.join(A.out, 'grid_%s.csv' % A.form), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['K \\ D'] + dp)
    for k in kp:
        row = [k]
        for d in dp:
            r = cell.get((k, d))
            row.append('' if r is None else ('T' if t(r) == INF else '%.3f' % t(r)))
        w.writerow(row)
print('\ngrid -> %s' % os.path.join(A.out, 'grid_%s.csv' % A.form))
