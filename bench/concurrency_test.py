"""THE CONCURRENCY HARNESS: many readers while the table is rewritten under them.

  1. a fresh database (scope realm), a server on :8767
  2. K client threads hammer the server with a rotating query set for T seconds
  3. meanwhile a writer process runs DELETE -> flush -> compact cycles on the same table
  4. two in-process engines race to birth the same census/road sidecars

Every answer must equal a VALID state: queries invariant to the delete must be exact;
queries affected by it must equal either the before or the after truth. Nothing may
crash, and the server must never restart.
usage: python3 bench/concurrency_test.py [seconds] [clients] [segments]
"""
import sys, os, time, json, shutil, threading, subprocess, urllib.request
sys.path.insert(0, 'src')

SRC = '/workspace/data/scope/x.parquet'
DB = '/tmp/concdb'; PORT = 8767; PY = sys.executable

INV = ["SELECT COUNT(*) FROM x WHERE id4 = 3", "SELECT id1, COUNT(*) AS c FROM x WHERE id4 = 3 GROUP BY id1 ORDER BY c DESC, id1 LIMIT 3",
       "SELECT SUM(v1) FROM x WHERE id4 = 3", "SELECT COUNT(DISTINCT id6) FROM x WHERE id4 = 3"]
AFF = ["SELECT COUNT(*) FROM x", "SELECT SUM(v1) FROM x", "SELECT COUNT(*) FROM x WHERE id4 = 7"]

def ask(sql):
    req = urllib.request.Request('http://127.0.0.1:%d/sql' % PORT, data=json.dumps({'sql': sql}).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())

def truth():
    import duckdb
    con = duckdb.connect()
    out = {}
    for tag, where in (('before', ''), ('after', " WHERE NOT (id4 = 7)")):
        con.execute("CREATE OR REPLACE VIEW x AS SELECT * FROM read_parquet('%s')%s" % (SRC, where))
        out[tag] = {q: con.execute(q).fetchall() for q in INV + AFF}
    return out

def norm(rows):
    def nm(v): return ('%.6g' % v) if isinstance(v, float) else str(v)
    return sorted(tuple(nm(v) for v in r) for r in rows)

def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 60; K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    NSEG = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    shutil.rmtree(DB, ignore_errors=True); os.makedirs(DB)
    import pyarrow.parquet as pq
    t = pq.read_table(SRC); step = (t.num_rows + NSEG - 1) // NSEG
    for i in range(NSEG):
        p = '/tmp/conc_slice%d.parquet' % i; pq.write_table(t.slice(i * step, step), p)
        r = subprocess.run([PY, 'bin/wdb', 'load', DB, 'x', p, '--workers', '4'], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-300:]
    print('segments: %d' % NSEG, flush=True)
    TR = truth()
    srv = subprocess.Popen([PY, 'bin/wdb', 'serve', DB, '--port', str(PORT)], stdout=open('/tmp/conc_serve.log', 'w'), stderr=subprocess.STDOUT)
    for _ in range(90):
        time.sleep(1)
        try: urllib.request.urlopen('http://127.0.0.1:%d/health' % PORT, timeout=2); break
        except Exception: pass
    stats = {'ok': 0, 'bad': 0, 'err': 0, 'examples': []}; lock = threading.Lock(); stop = [False]
    def client(i):
        qs = INV + AFF; j = i
        while not stop[0]:
            q = qs[j % len(qs)]; j += 1
            try:
                r = ask(q)
                if 'error' in r:
                    with lock: stats['err'] += 1; stats['examples'].append(('err', q, r['error'][:80]))
                    continue
                got = norm([tuple(x) for x in r['rows']])
                valid = (got == norm(TR['before'][q])) or (got == norm(TR['after'][q]))
                with lock:
                    if valid: stats['ok'] += 1
                    else: stats['bad'] += 1; stats['examples'].append(('bad', q, str(r['rows'])[:60]))
            except Exception as e:
                with lock: stats['err'] += 1; stats['examples'].append(('exc', q, str(e)[:80]))
    threads = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(K)]
    for t in threads: t.start()
    # the writer: DELETE -> flush -> compact, repeatedly, in its own process (a second engine on the same files)
    writer = ("import sys, time; sys.path.insert(0, 'src')\n"
              "from wdb_db import Database\nimport wdb_dml\n"
              "for k in range(%d):\n"
              "    db = Database.open('%s')\n"
              "    n = wdb_dml.delete(db.cat, 'DELETE FROM x WHERE id4 = 7')\n"
              "    r = db.compact('x', tier_rows=%d); print('cycle', k, 'deleted', n, 'compacted', r.get('merged'), '->', r.get('new_segment'), r.get('rows'), flush=True)\n"
              "    time.sleep(2)\n") % (max(1, T // 15), DB, 120_000 if NSEG > 1 else 25_000_000)
    wp = subprocess.Popen([PY, '-c', writer], stdout=open('/tmp/conc_writer.log', 'w'), stderr=subprocess.STDOUT)
    # the racers: two engines birthing the same sidecars at once
    racer = ("import sys; sys.path.insert(0, 'src')\nfrom wdb_db import Database\n"
             "db = Database.open('%s')\n"
             "for q in ['SELECT id4, COUNT(*) FROM x GROUP BY id4', 'SELECT COUNT(*) FROM x WHERE id6 = 42', 'SELECT id2, COUNT(DISTINCT id6) FROM x GROUP BY id2']:\n"
             "    print(len(db.run(q)[0]), flush=True)\n") % DB
    rp = [subprocess.Popen([PY, '-c', racer], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    t0 = time.time()
    while time.time() - t0 < T: time.sleep(1)
    stop[0] = True
    for t in threads: t.join(5)
    wp.wait(300); rr = [p.communicate(timeout=300)[0][-200:] for p in rp]
    srv_restarts = open('/tmp/conc_serve.log').read().count('restarting')
    print('CONCURRENCY: %ds, %d clients | answers ok=%d bad=%d err=%d | writer rc=%s | racers rc=%s | server restarts=%d' % (
        T, K, stats['ok'], stats['bad'], stats['err'], wp.returncode, [p.returncode for p in rp], srv_restarts), flush=True)
    for ex in stats['examples'][:6]: print('   ', ex)
    print('writer:', open('/tmp/conc_writer.log').read().strip().splitlines()[-2:])
    for x in rr: print('racer:', x.strip().replace('\n', ' | ')[-160:])
    srv.terminate()
    verdict = stats['bad'] == 0 and stats['err'] == 0 and wp.returncode == 0 and all(p.returncode == 0 for p in rp) and srv_restarts == 0
    print('VERDICT:', 'PROVEN' if verdict else 'FAILED', flush=True)

if __name__ == '__main__':
    main()
