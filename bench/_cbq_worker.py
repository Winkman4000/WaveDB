"""Single-query worker for the ClickBench board runner. Runs one query in its own
process so the parent can enforce a hard kill-timeout (a slow query can't stall the
board). Measures cold (this fresh process's first run) + warm (immediate re-run), and
emits a normalized result hash for correctness comparison. argv: src_dir db_dir query"""
import sys, time, json, hashlib
src, dbdir, q = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, src)
from wdb_db import Database
def nh(rows):
    h = hashlib.md5()
    for r in sorted([tuple(round(v,3) if isinstance(v,float) else v for v in row) for row in rows], key=repr):
        h.update(repr(r).encode())
    return h.hexdigest(), len(rows)
try:
    db = Database.open(dbdir)
    t = time.perf_counter(); res = db.run(q); cold = (time.perf_counter()-t)*1000
    rows = res[0] if isinstance(res, tuple) else res
    t = time.perf_counter(); db.run(q); warm = (time.perf_counter()-t)*1000
    hsh, n = nh(rows); out = {'cold_ms': cold, 'warm_ms': warm, 'nrows': n, 'hash': hsh}
except Exception as e:
    out = {'err': '%s: %s' % (type(e).__name__, str(e)[:90])}
print(json.dumps(out))
