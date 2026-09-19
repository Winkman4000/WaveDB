"""THE REFEREE, PostgreSQL protocol (Umbra, and anything else that speaks it): run a query file, three
runs per query (first cold, two warm), timed at the client like ClickBench's psql \\timing, and SAVE
the numbers to bench/referee/<name>.json so they are never re-run.

    python3 bench/referee_pg.py NAME QUERIES.sql OUT.json [--dsn ...] [--runs 3] [--timeout 300]
                                [--engine umbra] [--version X] [--load-seconds S] [--data-mb MB]
"""
import sys, os, json, time, platform, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('queries'); ap.add_argument('out')
    ap.add_argument('--dsn', default='host=127.0.0.1 port=5432 user=postgres password=postgres dbname=postgres')
    ap.add_argument('--runs', type=int, default=3); ap.add_argument('--timeout', type=float, default=300.0)
    ap.add_argument('--engine', default='umbra'); ap.add_argument('--version', default='')
    ap.add_argument('--load-seconds', type=float); ap.add_argument('--data-mb', type=float)
    a = ap.parse_args()
    import psycopg2
    text = '\n'.join(l for l in open(a.queries).read().splitlines() if not l.strip().startswith('--'))
    qs = [q.strip() for q in text.split(';') if q.strip()]
    rows = []
    for i, q in enumerate(qs):
        times = []; err = None
        for r in range(a.runs):
            try:
                con = psycopg2.connect(a.dsn, options='-c statement_timeout=%d' % int(a.timeout * 1000))
                con.autocommit = True
                cur = con.cursor()
                t0 = time.perf_counter(); cur.execute(q)
                try: cur.fetchall()
                except psycopg2.ProgrammingError: pass
                times.append(round(time.perf_counter() - t0, 4)); con.close()
            except Exception as e:
                times.append(None); err = str(e).strip().splitlines()[0][:160]; break
        ok = [t for t in times if t is not None]
        rows.append({'i': i, 'sql': q, 'times': times, 'best': min(ok) if ok else None, 'warm': min(ok[1:]) if len(ok) > 1 else (ok[0] if ok else None), 'error': err})
        print('Q%02d %s %s' % (i, ' '.join('%.3f' % t if t is not None else 'ERR' for t in times), err or ''), flush=True)
    doc = {'name': a.name, 'engine': a.engine, 'version': a.version, 'machine': {'node': platform.node(), 'cpus': os.cpu_count()},
           'queries_file': os.path.basename(a.queries), 'runs': a.runs, 'load_seconds': a.load_seconds, 'data_mb': a.data_mb,
           'recorded': time.strftime('%Y-%m-%d %H:%M:%S'), 'results': rows,
           'sum_best': round(sum(r['best'] for r in rows if r['best'] is not None), 3),
           'sum_warm': round(sum(r['warm'] for r in rows if r['warm'] is not None), 3),
           'errors': sum(1 for r in rows if r['error'])}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(doc, open(a.out, 'w'), indent=1)
    print('SAVED %s: %d queries, sum(best)=%.2fs sum(warm)=%.2fs errors=%d' % (a.out, len(rows), doc['sum_best'], doc['sum_warm'], doc['errors']))


if __name__ == '__main__':
    main()
