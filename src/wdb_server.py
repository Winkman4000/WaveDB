"""THE WIRE, second form: a server. `wdb serve DB --port 8765` keeps one Database open -- warm
shelves, warm JIT, warm sidecars -- and answers HTTP:

    POST /sql        body: {"sql": "...", "format": "rows"|"columns", "limit": N}
                     -> {"names": [...], "rows": [[...], ...], "wall": s} or {"error": "...", "kind": ...}
    GET  /health     -> {"ok": true, "db": DB, "queries": n}
    GET  /tables     -> {"tables": [...]}
    POST /explain    body: {"sql": "..."} -> {"plan": "..."}

One query at a time (a lock): no concurrency claim is made yet -- that proof is item 3b.
Errors are named: a decline is {"kind": "unsupported"}, a bad query {"kind": "error"}.
The thin client in bin/wdb (--server URL) imports nothing heavy and talks JSON.
"""
import json, sys, os, time, threading, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _js(v):
    if isinstance(v, (bytes, bytearray)): return v.decode('utf-8', 'replace')
    if isinstance(v, (datetime.date, datetime.datetime)): return v.isoformat()
    try:
        import numpy as np
        if isinstance(v, np.generic): return v.item()
    except Exception:
        pass
    return v


class _State:
    def __init__(self, dbdir):
        from wdb_db import Database
        import wdb_kernels
        wdb_kernels.warm()
        self.db = Database.open(dbdir); self.dbdir = dbdir
        self.lock = threading.Lock(); self.queries = 0; self.started = time.time()


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body))); self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get('Content-Length') or 0)
            return json.loads(self.rfile.read(n) or b'{}')

        def do_GET(self):
            if self.path == '/health':
                import resource
                return self._send(200, {'ok': True, 'db': state.dbdir, 'queries': state.queries, 'uptime': round(time.time() - state.started, 1),
                                        'rss_gb': round(_rss_gb(), 2), 'peak_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)})
            if self.path == '/tables':
                return self._send(200, {'tables': state.db.cat.list_tables()})
            self._send(404, {'error': 'no such path', 'kind': 'error'})

        def do_POST(self):
            try:
                req = self._body()
            except Exception as e:
                return self._send(400, {'error': 'bad json: %s' % e, 'kind': 'error'})
            sql = (req.get('sql') or '').strip().rstrip(';')
            if not sql:
                return self._send(400, {'error': 'empty sql', 'kind': 'error'})
            with state.lock:
                t0 = time.perf_counter()
                if os.environ.get('WDB_SERVE_LEDGER'):
                    print('  -> rss %.2f GB  %s' % (_rss_gb(), sql[:100].replace(chr(10), ' ')), flush=True)   # BEFORE the run: a killer names itself
                try:
                    if self.path == '/explain':
                        return self._send(200, {'plan': state.db.explain(sql, run=bool(req.get('run', True)))})
                    head = sql.split(None, 1)[0].upper()
                    if head in ('INSERT', 'DELETE', 'UPDATE'):
                        import wdb_dml
                        fn = {'INSERT': wdb_dml.insert, 'DELETE': wdb_dml.delete, 'UPDATE': wdb_dml.update}[head]
                        n = fn(state.db.cat, sql); state.queries += 1
                        return self._send(200, {'names': ['rows'], 'rows': [[n]], 'wall': round(time.perf_counter() - t0, 4), 'statement': head})
                    r = state.db.run(sql); state.queries += 1
                    if os.environ.get('WDB_SERVE_LEDGER'):
                        print('  q%-5d %6.3fs  rss %.2f GB  %s' % (state.queries, time.perf_counter() - t0, _rss_gb(), sql[:80].replace(chr(10), ' ')), flush=True)
                    rows, names = (r if isinstance(r, tuple) else (r, ['col%d' % i for i in range(len(r[0]) if r else 0)]))
                    lim = req.get('limit')
                    rows = list(rows)[:lim] if lim else list(rows)
                    out = {'names': list(names), 'wall': round(time.perf_counter() - t0, 4), 'count': len(rows)}
                    if req.get('format') == 'columns':
                        out['columns'] = [[_js(v) for v in col] for col in zip(*rows)] if rows else [[] for _ in names]
                    else:
                        out['rows'] = [[_js(v) for v in r] for r in rows]
                    self._send(200, out)
                except NotImplementedError as e:
                    self._send(200, {'error': str(e).splitlines()[0][:400], 'kind': 'unsupported'})
                except (KeyError, ValueError) as e:
                    self._send(200, {'error': str(e)[:400], 'kind': 'error'})
                except Exception as e:
                    self._send(500, {'error': '%s: %s' % (type(e).__name__, str(e)[:400]), 'kind': 'failed'})
    return H


def _rss_gb():
    try:
        with open('/proc/self/statm') as f:
            return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1e9
    except Exception:
        return 0.0


def supervise(dbdir, host='127.0.0.1', port=8765):
    """THE WATCHDOG: the engine runs in a child; if the kernel kills it (OOM) or it crashes,
    the parent logs the death and starts a new one -- a client's in-flight query fails once
    (the thin client retries after a short wait); the server never stays dead."""
    import subprocess
    n = 0
    while True:
        n += 1
        t0 = time.time()
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), dbdir, str(port), host, '--child'])
        rc = p.wait()
        if rc in (0, -2, -15):
            return
        print('wdb serve: engine child exited rc=%s after %.0fs (%s) -- restarting (%d)' % (rc, time.time() - t0, 'OOM-killed' if rc == -9 else 'crashed', n), flush=True)
        time.sleep(1)


def _die_with_parent():
    """an engine child never outlives its watchdog: an orphan holding the port answered 500s
    with a stale catalog for a whole harness run (2026-09-14)"""
    try:
        import ctypes, signal
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        libc.prctl(1, signal.SIGTERM)                  # PR_SET_PDEATHSIG
    except Exception:
        pass


def serve(dbdir, host='127.0.0.1', port=8765):
    _die_with_parent()
    state = _State(dbdir)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    print('wdb serving %s on http://%s:%d  (POST /sql, /explain; GET /health, /tables)' % (dbdir, host, port), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _args = [a for a in sys.argv[1:] if not a.startswith('--')]
    _port = int(_args[1]) if len(_args) > 1 else 8765; _host = _args[2] if len(_args) > 2 else '127.0.0.1'
    if '--child' in sys.argv:
        serve(_args[0], host=_host, port=_port)
    else:
        supervise(_args[0], host=_host, port=_port)
