"""THE REFEREE, ClickHouse: run a query file against a local ClickHouse server, three runs per query
(the ClickBench protocol: first run cold, then two warm), and SAVE the numbers so they are never
re-run -- bench/referee/<name>.json carries the machine, the version, the load time and the size.

    python3 bench/referee_clickhouse.py NAME QUERIES.sql OUT.json [--bin ./clickhouse] [--runs 3] [--timeout 300]
                                        [--load-seconds S] [--data-mb MB] [--settings "k=v,k=v"]

Queries are split on ';' (a line starting with -- is a comment). Timing is the server's own
elapsed for the statement (system.query_log is not needed: the client prints it under --time).
Errors and timeouts are recorded by name, never dropped.
"""
import sys, os, json, time, subprocess, platform, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('queries'); ap.add_argument('out')
    ap.add_argument('--bin', default='./clickhouse'); ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--timeout', type=float, default=300.0); ap.add_argument('--load-seconds', type=float)
    ap.add_argument('--data-mb', type=float); ap.add_argument('--settings', default='')
    ap.add_argument('--database', default='default')
    a = ap.parse_args()
    text = '\n'.join(l for l in open(a.queries).read().splitlines() if not l.strip().startswith('--'))
    qs = [q.strip() for q in text.split(';') if q.strip()]
    ver = subprocess.run([a.bin, 'client', '--query', 'SELECT version()'], capture_output=True, text=True).stdout.strip()
    settings = [s.strip() for s in a.settings.split(',') if s.strip()]
    rows = []
    for i, q in enumerate(qs):
        times = []; err = None
        for r in range(a.runs):
            cmd = [a.bin, 'client', '--database', a.database, '--time', '--format', 'Null', '--query', q] + sum((['--' + s.split('=')[0], s.split('=', 1)[1]] for s in settings), [])
            t0 = time.perf_counter()
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            except subprocess.TimeoutExpired:
                times.append(None); err = 'TIMEOUT>%ds' % a.timeout; break
            wall = time.perf_counter() - t0
            if p.returncode != 0:
                times.append(None); err = p.stderr.strip().splitlines()[-1][:160] if p.stderr.strip() else 'exit %d' % p.returncode; break
            try:
                srv = float(p.stderr.strip().splitlines()[-1])       # --time prints the elapsed on stderr
            except Exception:
                srv = wall
            times.append(round(srv, 4))
        ok = [t for t in times if t is not None]
        rows.append({'i': i, 'sql': q, 'times': times, 'best': min(ok) if ok else None, 'warm': min(ok[1:]) if len(ok) > 1 else (ok[0] if ok else None), 'error': err})
        print('Q%02d %s %s' % (i, ' '.join('%.3f' % t if t is not None else 'ERR' for t in times), err or ''), flush=True)
    doc = {'name': a.name, 'engine': 'clickhouse', 'version': ver, 'machine': {'node': platform.node(), 'cpus': os.cpu_count()},
           'queries_file': os.path.basename(a.queries), 'runs': a.runs, 'load_seconds': a.load_seconds, 'data_mb': a.data_mb,
           'recorded': time.strftime('%Y-%m-%d %H:%M:%S'), 'settings': settings, 'results': rows,
           'sum_best': round(sum(r['best'] for r in rows if r['best'] is not None), 3),
           'sum_warm': round(sum(r['warm'] for r in rows if r['warm'] is not None), 3),
           'errors': sum(1 for r in rows if r['error'])}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(doc, open(a.out, 'w'), indent=1)
    print('SAVED %s: %d queries, sum(best)=%.2fs sum(warm)=%.2fs errors=%d' % (a.out, len(rows), doc['sum_best'], doc['sum_warm'], doc['errors']))


if __name__ == '__main__':
    main()
