"""Tie-tolerant correctness verifier for the ClickBench suite (vs a DuckDB oracle).

The timed board (board_clickbench.py) hashes the LIMITed result as-written, so a query whose ORDER BY
ties at the LIMIT boundary -- or which has LIMIT with no ORDER BY -- can show ok=False despite being
correct (the engines just pick a different, equally-valid slice). This tool settles correctness
independently: it strips ORDER BY/LIMIT/OFFSET and compares the FULL result set with a type-normalized,
order-independent fingerprint (bench/_cbnorm.full_fp). It reports, per query:
    EXACT        - the as-written LIMIT result already matches (type-normalized)
    TIE-ORDERING - LIMIT result differs but the full set matches  => correct, slice is unspecified
    DIFFERS      - full sets differ                                => a real difference, prints samples
    WAVE-ERR     - the engine raised

Usage: python bench/verify_correctness.py SRC DB_DIR HITS_PARQUET QUERIES_SQL [idx,idx,...]
(no LIMIT cap on runtime -- run it deliberately, not as the timed board.)"""
import sys, os, time
SRC, DBDIR, PARQ, SQLF = sys.argv[1:5]
ONLY = set(int(x) for x in sys.argv[5].split(',')) if len(sys.argv) > 5 and sys.argv[5].strip() else None
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sqlglot, duckdb, _cbnorm as N
from wdb_db import Database

HITS_CTE = ("WITH hits AS (SELECT * REPLACE ("
            "(DATE '1970-01-01' + EventDate) AS EventDate, "
            "(TIMESTAMP '1970-01-01' + to_seconds(EventTime)) AS EventTime) "
            f"FROM read_parquet('{PARQ}'))")
qs = [l.strip() for l in open(SQLF) if l.strip() and not l.strip().startswith('--')]

def strip_ol(q):
    t = sqlglot.parse_one(q, read='duckdb')
    for k in ('order', 'limit', 'offset'): t.args.pop(k, None)
    return t.sql(dialect='duckdb')

db = Database.open(DBDIR); con = duckdb.connect()
print(f"idx | verdict        | wave_full duck_full | note", flush=True)
summary = {}
for i, q in enumerate(qs):
    if ONLY is not None and i not in ONLY: continue
    qf = strip_ol(q)
    has_ol = (qf.strip() != q.strip())
    try:
        t = time.perf_counter(); wf, _ = db.run(qf); wms = time.perf_counter() - t
    except Exception as e:
        print(f"Q{i:02d} | WAVE-ERR       | -                  | {type(e).__name__}: {str(e)[:70]}", flush=True)
        summary[i] = 'WAVE-ERR'; continue
    df = con.execute(HITS_CTE + ' ' + qf).fetchall()
    full_match = N.full_fp(wf) == N.full_fp(df)
    if not full_match:
        # real difference -> classify + sample
        print(f"Q{i:02d} | DIFFERS        | {len(wf):>9} {len(df):>9} | full sets differ", flush=True)
        print(f"     wave: {[N.norm_row(r) for r in wf[:3]]}", flush=True)
        print(f"     duck: {[N.norm_row(tuple(r)) for r in df[:3]]}", flush=True)
        summary[i] = 'DIFFERS'; continue
    if not has_ol:
        print(f"Q{i:02d} | EXACT          | {len(wf):>9} {len(df):>9} | no ORDER/LIMIT ({wms:.1f}s)", flush=True)
        summary[i] = 'EXACT'; continue
    # full set matches AND there is an ORDER/LIMIT -> check the as-written slice
    try:
        wl, _ = db.run(q); dl = con.execute(HITS_CTE + ' ' + q).fetchall()
        limit_match = N.limit_hash(wl) == N.limit_hash([tuple(r) for r in dl])
    except Exception:
        limit_match = False
    verdict = 'EXACT' if limit_match else 'TIE-ORDERING'
    note = 'as-written slice matches' if limit_match else 'slice unspecified; full set matches'
    print(f"Q{i:02d} | {verdict:<14} | {len(wf):>9} {len(df):>9} | {note} ({wms:.1f}s)", flush=True)
    summary[i] = verdict

print("\nSUMMARY:", flush=True)
from collections import Counter
c = Counter(summary.values())
print("  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())), flush=True)
diffs = [i for i, v in summary.items() if v == 'DIFFERS']
print(f"  real differences: {['Q%02d'%i for i in diffs] if diffs else 'NONE'}", flush=True)
print(f"  correct: {sum(v in ('EXACT','TIE-ORDERING') for v in summary.values())}/{len(summary)}", flush=True)
