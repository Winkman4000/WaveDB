"""Single-run correctness worker: run one query once, emit its order-insensitive normalized result hash.
Own process so the parent can enforce a hard kill-timeout. argv: src_dir db_dir query"""
import sys, json, os
src, dbdir, q = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, src)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cbnorm as N
from wdb_db import Database
def rows_of(res):
    if isinstance(res, tuple) and len(res) == 2:
        data, names = res
        return list(zip(*[data[n] for n in names])) if isinstance(data, dict) else data
    return res[0] if isinstance(res, tuple) else res
try:
    db = Database.open(dbdir)
    rows = rows_of(db.run(q))
    out = {'hash': N.limit_hash(rows), 'nrows': len(rows)}
except Exception as e:
    out = {'err': '%s: %s' % (type(e).__name__, str(e)[:120])}
print(json.dumps(out))
