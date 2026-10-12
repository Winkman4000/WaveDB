"""The shifting board's WaveDB worker: one long-lived process, one open database, queries over
stdin/stdout as JSON lines. The parent kills and restarts it when a query overruns its timeout,
so a pathological variant costs one restart, not the run.

request:  {"sql": ..., "runs": n}
reply:    {"times": [s, ...], "hash": md5, "nrows": n, "read": winning read}  or  {"err": ...}
argv: src_dir db_dir
"""
import sys, os, json, time
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cbnorm as N
import controller
from wdb_db import Database


def rows_of(res):
    if isinstance(res, tuple) and len(res) == 2:
        data, names = res
        return list(zip(*[data[n] for n in names])) if isinstance(data, dict) else data
    return res[0] if isinstance(res, tuple) else res


db = Database.open(sys.argv[2])
print(json.dumps({'ready': True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    try:
        times = []
        rows = None
        read = None
        for _ in range(int(req.get('runs', 1))):
            controller._SERVED[0] = None
            t0 = time.perf_counter()
            res = db.run(req['sql'])
            times.append(time.perf_counter() - t0)
            read = controller._SERVED[0]
            if rows is None:
                rows = rows_of(res)
        out = {'times': times, 'hash': N.limit_hash(rows), 'nrows': len(rows), 'read': read or '(outside read order)'}
    except Exception as e:
        out = {'err': '%s: %s' % (type(e).__name__, str(e)[:160])}
    print(json.dumps(out), flush=True)
