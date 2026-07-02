"""Correctness worker: run ONE query in its own process (so a slow query can't stall the harness) and
emit the ACTUAL result rows, normalized, as JSON. Distinct from _cbq_worker.py, which emits only a hash
for the speed board. argv: src_dir db_dir query"""
import sys, time, json, os
src, dbdir, q = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, src)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cbnorm as N
from wdb_db import Database
def rows_of(res):
    if isinstance(res, tuple) and len(res) == 2:
        data, names = res
        if isinstance(data, dict):
            return list(zip(*[data[n] for n in names])) if names else []
        return data
    return res
try:
    db = Database.open(dbdir)
    t = time.perf_counter(); res = db.run(q); ms = (time.perf_counter() - t) * 1000
    rows = rows_of(res)
    norm = [[N.norm_cell_exact(v) for v in r] for r in rows]
    print(json.dumps({'ms': ms, 'nrows': len(norm), 'rows': norm}, default=str))
except Exception as e:
    print(json.dumps({'err': '%s: %s' % (type(e).__name__, str(e)[:120])}))
