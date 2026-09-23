"""THE FAIR TRIAL (2026-09-23): every engine gets the cold run ClickBench defines -- a fresh process
(or a restarted server) whose data files have left the OS page cache -- then two hot runs. The pod
refuses /proc/sys/vm/drop_caches, so each engine's data files are evicted with
posix_fadvise(DONTNEED) (no root), exactly as bench/true_cold.py does for WaveDB.

  umbra       fresh umbra-sql process per query (the image's own loader); time = exec + compile
  clickhouse  server stopped, data evicted, server started; time = clickhouse client --time
  duckdb      fresh python process per query on the native file; time = execute + fetch

Usage: python bench/fair_cold.py ENGINE OUT_JSONL   (one JSON line per query: q, cold, hot, all)
"""
import os, sys, re, json, time, glob, signal, subprocess

REF = '/workspace/referee'


def evict_tree(*paths):
    n = 0
    for p in paths:
        files = [p] if os.path.isfile(p) else [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs]
        for f in files:
            try:
                fd = os.open(f, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
            except OSError:
                pass
    return n


def queries(path):
    return [l.strip() for l in open(path) if l.strip() and not l.strip().startswith('--')]


# ------------------------------------------------------------------ umbra
U = REF + '/umbra/rootfs'
ULD = [U + '/lib64/ld-linux-x86-64.so.2', '--library-path',
       U + '/lib/x86_64-linux-gnu:' + U + '/usr/lib/x86_64-linux-gnu:' + U + '/usr/local/lib']
UDB = REF + '/umbra/db'


def umbra(q, sql):
    evict_tree(UDB)
    qf = '/tmp/umbra_q.sql'
    open(qf, 'w').write((sql.rstrip(';') + ';\n') * 3)
    env = dict(os.environ, ASYNCIO='0')          # as the official start script runs it
    p = subprocess.run(ULD + [U + '/usr/local/bin/umbra-sql', UDB + '/umbra.db', qf],
                       capture_output=True, text=True, env=env, cwd=UDB, timeout=600)
    t = [float(a) + float(b) for a, b in re.findall(r'exec: ([0-9.]+) s, comp: ([0-9.]+) s', p.stdout + p.stderr)]
    if len(t) != 3:
        return {'q': q, 'err': (p.stdout + p.stderr)[-300:]}
    return {'q': q, 'cold': round(t[0] * 1e3), 'hot': round(min(t[1:]) * 1e3), 'all': [round(x * 1e3) for x in t]}


# ------------------------------------------------------------------ clickhouse
CH = REF + '/clickhouse'
_srv = [None]


def ch_client(*args, timeout=600):
    return subprocess.run([CH + '/clickhouse', 'client'] + list(args), capture_output=True, text=True, timeout=timeout, cwd=CH)


def ch_start():
    _srv[0] = subprocess.Popen([CH + '/clickhouse', 'server', '--config-file=' + CH + '/config.xml'],
                               cwd=CH, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(600):
        if ch_client('--query', 'SELECT 1', timeout=10).returncode == 0:
            return
        time.sleep(0.2)
    raise RuntimeError('clickhouse did not start')


def ch_stop():
    if _srv[0] is not None:
        _srv[0].send_signal(signal.SIGTERM)
        try:
            _srv[0].wait(timeout=60)
        except subprocess.TimeoutExpired:
            _srv[0].kill(); _srv[0].wait()
        _srv[0] = None


def clickhouse(q, sql):
    ch_stop(); evict_tree(CH + '/data'); ch_start()
    t = []
    for _ in range(3):
        p = ch_client('--time', '--format=Null', '--query', sql)
        if p.returncode != 0:
            return {'q': q, 'err': p.stderr[-300:]}
        t.append(float(p.stderr.strip().splitlines()[-1]))
    return {'q': q, 'cold': round(t[0] * 1e3), 'hot': round(min(t[1:]) * 1e3), 'all': [round(x * 1e3) for x in t]}


# ------------------------------------------------------------------ duckdb
def duckdb_one(q):
    fix = set()
    for l in open('/workspace/session_2026-09-23/duckcold_fix.jsonl'):
        r = json.loads(l)
        if r.get('via') == 'parquet':
            fix.add(r['idx'])
    evict_tree('/workspace/data/duck_native.db', '/workspace/data/hits.parquet')
    args = [sys.executable, '/workspace/duckcold.py', str(q)] + (['parquet'] if q in fix else [])
    p = subprocess.run(args, capture_output=True, text=True, timeout=900)
    r = json.loads(p.stdout.strip().splitlines()[-1])
    if 'err' in r:
        return {'q': q, 'err': r['err']}
    return {'q': q, 'cold': round(r['first']), 'hot': round(r['hot']), 'via': r.get('via', 'native')}


if __name__ == '__main__':
    engine, out = sys.argv[1], sys.argv[2]
    with open(out, 'w') as f:
        if engine == 'umbra':
            for q, sql in enumerate(queries(REF + '/umbra/queries.sql')):
                f.write(json.dumps(umbra(q, sql)) + '\n'); f.flush()
        elif engine == 'clickhouse':
            try:
                for q, sql in enumerate(queries(CH + '/queries.sql')):
                    f.write(json.dumps(clickhouse(q, sql)) + '\n'); f.flush()
            finally:
                ch_stop()
        elif engine == 'duckdb':
            for q in range(43):
                f.write(json.dumps(duckdb_one(q)) + '\n'); f.flush()
