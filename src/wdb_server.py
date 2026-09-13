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
                return self._send(200, {'ok': True, 'db': state.dbdir, 'queries': state.queries, 'uptime': round(time.time() - state.started, 1)})
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


def serve(dbdir, host='127.0.0.1', port=8765):
    state = _State(dbdir)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    print('wdb serving %s on http://%s:%d  (POST /sql, /explain; GET /health, /tables)' % (dbdir, host, port), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    serve(sys.argv[1], port=int(sys.argv[2]) if len(sys.argv) > 2 else 8765)
