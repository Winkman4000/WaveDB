"""ClickBench board runner (hits dataset). Robust: each query runs in its own process
(bench/_cbq_worker.py) under a hard kill-timeout, so slow queries get TIMEOUT instead
of stalling the board. Reports cold (fresh process) + warm (re-run) WaveDB times vs
DuckDB, with hash-based correctness. Writes a JSON board.

Usage: python bench/board_clickbench.py SRC_DIR DB_DIR HITS_PARQUET QUERIES_SQL OUT_JSON [timeout_s]
(produced examples/boards/clickbench_board.json on the RunPod 100M hits volume)."""
import sys, subprocess, json, re, time, hashlib, statistics as st, os
SRC, DBDIR, PARQ, SQLF, OUT = sys.argv[1:6]
T = int(sys.argv[6]) if len(sys.argv) > 6 else 45
WORKER = os.path.join(os.path.dirname(__file__), '_cbq_worker.py')
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for _cbnorm
import duckdb, _cbnorm as N
from _cbvalidate import total_order_sql
qs = [l.strip() for l in open(SQLF) if l.strip() and not l.strip().startswith('--')]
con = duckdb.connect()
# Oracle types EventDate/EventTime as DATE/TIMESTAMP (cb25db stores them as int
# days/epoch-seconds; the canonical ClickBench SQL compares them to date literals).
# Wrapping the parquet in a CTE named `hits` lets `FROM hits` resolve to the typed view.
HITS_CTE = ("WITH hits AS (SELECT * REPLACE ("
            "(DATE '1970-01-01' + EventDate) AS EventDate, "
            "(TIMESTAMP '1970-01-01' + to_seconds(EventTime)) AS EventTime) "
            f"FROM read_parquet('{PARQ}'))")
def duck(q):
    s = HITS_CTE + ' ' + q
    t = time.perf_counter(); rows = con.execute(s).fetchall(); ms = (time.perf_counter()-t)*1000
    return N.limit_hash(rows), len(rows), ms
env = dict(os.environ); env['PYTHONPATH'] = SRC
VT = max(T, 240)   # validation-only timeout: correctness adjudication is OFF the perf-timing path, so a
                   # slow total-order variant (e.g. imposing an order on an unordered high-card GROUP BY)
                   # is allowed to run long. The reported speed still comes from the real query below.
def run_wdb(q, tmo=T):
    try:
        p = subprocess.run([sys.executable, WORKER, SRC, DBDIR, q], capture_output=True, text=True, timeout=tmo, env=env)
        ln = [l for l in p.stdout.strip().splitlines() if l.startswith('{')]
        return json.loads(ln[-1]) if ln else {'err': 'noout:' + (p.stderr.strip()[-90:] or '?')}
    except subprocess.TimeoutExpired:
        return {'err': 'TIMEOUT>%ds' % tmo}
def validated_ok(q, dh):
    # Exact hash mismatched. If DuckDB can't adjudicate (tie/unordered LIMIT), impose a total order so
    # exactly one answer is legal; if WaveDB and DuckDB agree there, WaveDB's answer was a valid one.
    q2 = total_order_sql(q)
    if not q2:
        return False
    w2 = run_wdb(q2, VT)
    if 'hash' not in w2:
        return False
    try:
        d2h, _, _ = duck(q2)
    except Exception:
        return False
    return w2['hash'] == d2h
# optional: per-query winning read from a path run (bench/path_run.py). Joined by idx if present.
_PATHS_F = os.path.join(os.path.dirname(OUT), "clickbench_paths.json")
PATHS = json.load(open(_PATHS_F)) if os.path.exists(_PATHS_F) else {}
out = []; print("idx | cold_ms warm_ms | duck_ms | ratio(warm) | read | status", flush=True)
for i, q in enumerate(qs):
    try: dh, dn, dms = duck(q)
    except Exception: dh, dn, dms = None, None, None
    w = run_wdb(q)
    if w.get('err'):
        status = 'timeout' if 'TIMEOUT' in w['err'] else 'err'
    elif ('hash' in w and dh is not None):
        if w['hash'] == dh:
            status = 'ok'
        elif validated_ok(q, dh):
            status = 'ok'          # correct, but non-deterministic: DuckDB's arbitrary tie pick differed
        else:
            status = 'false'
    else:
        status = 'n/a'
    ratio = (dms / w['warm_ms']) if (w.get('warm_ms') and dms) else None
    read = PATHS.get(str(i), "-")
    print("Q%02d | %s %s | %s | %s | %-12s | %s" % (i,
        ('%.0f' % w['cold_ms'] if w.get('cold_ms') else '----'),
        ('%.1f' % w['warm_ms'] if w.get('warm_ms') else '----'),
        ('%.1f' % dms if dms else '----'), ('%.2fx' % ratio if ratio else '-'), read, status), flush=True)
    out.append({'idx': i, 'read': read, 'status': status,
                **{k: w.get(k) for k in ('cold_ms','warm_ms','nrows','err')},
                'duck_ms': dms, 'ratio': ratio})
okc = sum(1 for r in out if r['status'] == 'ok')
falsec = sum(1 for r in out if r['status'] == 'false')
toc = sum(1 for r in out if r['status'] == 'timeout')
fast = sum(1 for r in out if r['ratio'] and r['ratio'] >= 1 and r['status'] == 'ok')
rr = [r['ratio'] for r in out if r['ratio'] and r['status'] == 'ok']
print("\nSUMMARY  ok=%d  false=%d  timeout=%d  (/43)   faster=%d   median_ratio_ok=%.2fx"
      % (okc, falsec, toc, fast, st.median(rr) if rr else 0), flush=True)
json.dump(out, open(OUT, 'w'), indent=0)
