"""Path run (separate from the timed board): execute each query ONCE with the controller's
_PATH_SINK live, record the winning read per query. Overhead is irrelevant here -- we never
time anything; we only capture which read fired. Speed stays in board_clickbench.py with the
sink OFF, so the two never contaminate each other.

Usage: python bench/path_run.py SRC DB_DIR QUERIES_SQL OUT_JSON
Output: {"0": "fused_agg", "17": "heavypair", ...}  (idx -> winning read; "merge" for two-tier)
"""
import sys, os, json
SRC, DBDIR, SQLF, OUT = sys.argv[1:5]
sys.path.insert(0, SRC)
import controller
from wdb_db import Database

qs = [l.strip() for l in open(SQLF) if l.strip() and not l.strip().startswith('--')]
db = Database.open(DBDIR)

result = {}
for i, q in enumerate(qs):
    captured = []
    controller._PATH_SINK = lambda ctx, name, _c=captured: _c.append(name)
    try:
        db.run(q)
        # last recorded name is the winner for the (single) segment this query routed through;
        # multiple appends => multi-segment merge touched several segments.
        read = captured[-1] if captured else "merge"   # no single-seg record => two-tier/merge path
        if len(set(captured)) > 1:
            read = "merge:" + "+".join(dict.fromkeys(captured))
    except Exception as e:
        read = f"ERR:{type(e).__name__}"
    finally:
        controller._PATH_SINK = None
    result[str(i)] = read
    print(f"Q{i:02d} -> {read}", flush=True)

json.dump(result, open(OUT, 'w'), indent=2)
print("wrote", OUT, "(", len(result), "queries )")
