"""A board (true_cold.py's JSON lines) against referee boards: ClickBench's score, the geometric
mean over queries of (t + 10 ms) / (best + 10 ms), best taken over the referees and the board
itself; totals; wins per referee; the queries where any referee is faster.
    python bench/board_vs.py BOARD.jsonl REFEREE_DIR     (REFEREE_DIR holds fair_<name>.jsonl)"""
import sys, os, json, math, glob


def load(p):
    return {d['q']: d for d in (json.loads(l) for l in open(p) if l.strip().startswith('{'))}


def main():
    W = load(sys.argv[1])
    R = {os.path.basename(p)[5:-6]: load(p) for p in sorted(glob.glob(os.path.join(sys.argv[2], 'fair_*.jsonl')))}
    qs = sorted(q for q in W if all(q in r for r in R.values()))
    for kind in ('cold', 'hot'):
        def score(X, pool):
            ls = [math.log((X[q][kind] + 10) / (min([p[q][kind] for p in pool] + [X[q][kind]]) + 10)) for q in qs]
            return math.exp(sum(ls) / len(ls))
        line = '%-4s WaveDB %.3f (%.1f s)' % (kind, score(W, list(R.values())), sum(W[q][kind] for q in qs) / 1e3)
        for n, r in R.items():
            others = [x for m, x in R.items() if m != n]
            line += ' | %s %.3f (%.1f s, WaveDB wins %d)' % (n, score(r, others), sum(r[q][kind] for q in qs) / 1e3,
                                                          sum(W[q][kind] < r[q][kind] for q in qs))
        print(line)
    for kind in ('cold', 'hot'):
        lose = [q for q in qs if any(r[q][kind] < W[q][kind] for r in R.values())]
        print('%s: a referee is faster on %d of %d: %s' % (kind, len(lose), len(qs), ' '.join(
            'Q%d(%d|%s)' % (q, W[q][kind], min(R, key=lambda n: R[n][q][kind])) for q in lose)))


if __name__ == '__main__':
    main()
