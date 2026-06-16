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
import duckdb
qs = [l.strip() for l in open(SQLF) if l.strip() and not l.strip().startswith('--')]
con = duckdb.connect()
def duck(q):
    s = re.sub(r'\bFROM hits\b', f"FROM read_parquet('{PARQ}')", q)
    t = time.perf_counter(); rows = con.execute(s).fetchall(); ms = (time.perf_counter()-t)*1000
    h = hashlib.md5()
    for r in sorted([tuple(round(v,3) if isinstance(v,float) else v for v in row) for row in rows], key=repr):
        h.update(repr(r).encode())
    return h.hexdigest(), len(rows), ms
env = dict(os.environ); env['PYTHONPATH'] = SRC
out = []; print("idx | cold_ms warm_ms | duck_ms | ratio(warm) | ok", flush=True)
for i, q in enumerate(qs):
    try: dh, dn, dms = duck(q)
    except Exception: dh, dn, dms = None, None, None
    try:
        p = subprocess.run(['python3', WORKER, SRC, DBDIR, q], capture_output=True, text=True, timeout=T, env=env)
        ln = [l for l in p.stdout.strip().splitlines() if l.startswith('{')]
        w = json.loads(ln[-1]) if ln else {'err': 'noout:' + (p.stderr.strip()[-90:] or '?')}
    except subprocess.TimeoutExpired:
        w = {'err': 'TIMEOUT>%ds' % T}
    ok = bool(w.get('hash') == dh) if ('hash' in w and dh is not None) else '?'
    ratio = (dms / w['warm_ms']) if (w.get('warm_ms') and dms) else None
    print("Q%02d | %s %s | %s | %s | %s%s" % (i,
        ('%.0f' % w['cold_ms'] if w.get('cold_ms') else '----'),
        ('%.1f' % w['warm_ms'] if w.get('warm_ms') else '----'),
        ('%.1f' % dms if dms else '----'), ('%.2fx' % ratio if ratio else '-'), ok,
        ('  <' + w['err'] if w.get('err') else '')), flush=True)
    out.append({'idx': i, **{k: w.get(k) for k in ('cold_ms','warm_ms','nrows','err')},
                'duck_ms': dms, 'ratio': ratio, 'ok': ok})
okc = sum(1 for r in out if r['ok'] is True)
fast = sum(1 for r in out if r['ratio'] and r['ratio'] >= 1 and r['ok'] is True)
rr = [r['ratio'] for r in out if r['ratio'] and r['ok'] is True]
print("\nSUMMARY ok=%d/43 faster=%d median_ratio_ok=%.2fx" % (okc, fast, st.median(rr) if rr else 0), flush=True)
json.dump(out, open(OUT, 'w'), indent=0)
