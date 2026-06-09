#!/usr/bin/env python3
"""bench/throughput.py -- the throughput scoreboard (WaveDB vs DuckDB).

Throughput is WaveDB's optimization target: a shared analytical DB serving many
concurrent queries. This dedicates the box to one query type and runs W
single-threaded workers in parallel (one per core), measuring the REAL
concurrent aggregate queries/sec for each engine -- memory-bandwidth contention
included, not extrapolated.

    throughput = aggregate queries/sec across W workers running together
    W          = min(cores, (budget_MB) / private_per_worker_MB)

WaveDB workers run in the default NON-ESCALATED (throughput) mode, so the BSI
filter-index engages where it pays; the BSI column marks which queries use it.
Memmap'd segment files are shared across workers (paid once), so only the
private working set scales per worker.

Writes examples/throughput.md (+ stdout). Needs the bench DB at
/tmp/jbprof_sf1.0 (build: python bench/join_prof.py 1.0).

  python bench/throughput.py [budget_gb] [cores]      # defaults 32, nproc
"""
import os, sys, subprocess, datetime

# Pin math libs AND numba to one thread: the model is W single-threaded workers, one per core.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); SRC = os.path.join(ROOT, 'src')
DIR = '/tmp/jbprof_sf1.0'; WDB = os.path.join(DIR, 'wdb'); DUCK = os.path.join(DIR, 'baseline.duckdb')
MB = 1024 * 1024
FKS = [('orders', 'o_custkey', 'customer', 'c_custkey'),
       ('lineitem', 'l_orderkey', 'orders', 'o_orderkey')]


def _smaps_private_dirty():
    """Anonymous private KB->MB: the part that scales per worker (not shared file pages)."""
    try:
        for l in open('/proc/self/smaps_rollup'):
            if l.startswith('Private_Dirty:'):
                return int(l.split()[1]) // 1024
    except FileNotFoundError:
        pass
    for l in open('/proc/self/status'):
        if l.startswith('VmHWM'):
            return int(l.split()[1]) // 1024
    return -1


def _worker(qi, dur):
    """WaveDB worker (default non-escalated -> BSI engages). Prints: count priv_MB bsi_fired."""
    import time
    sys.path.insert(0, SRC); sys.path.insert(0, HERE)
    from wdb_db import Database
    import wdb_bsi_exec as BX
    import wdb_cube as CB
    from catalog import QUERIES
    import numba; numba.set_num_threads(1)              # one worker == one core
    db = Database.open(WDB)
    for a in FKS:
        try: db.create_fk_pointer(*a)
        except Exception: pass
    sql = QUERIES[qi][2]
    for _ in range(3): db.run(sql)                      # warm (JIT + caches + lazy BSI build)
    h = BX._BSI_HITS; hc = CB._CUBE_HITS; db.run(sql)
    fired = int(BX._BSI_HITS > h); cube = int(CB._CUBE_HITS > hc)
    end = time.perf_counter() + dur; c = 0
    while time.perf_counter() < end: db.run(sql); c += 1
    print(f"{c} {_smaps_private_dirty()} {fired} {cube}", flush=True)


def _dworker(qi, dur):
    """DuckDB worker, single-threaded. Prints: count."""
    import time
    sys.path.insert(0, HERE)
    import duckdb
    from catalog import QUERIES
    con = duckdb.connect(DUCK, read_only=True); con.execute("SET threads=1")
    sql = QUERIES[qi][2]
    for _ in range(3): con.execute(sql).fetchall()
    end = time.perf_counter() + dur; c = 0
    while time.perf_counter() < end: con.execute(sql).fetchall(); c += 1
    print(f"{c}", flush=True)


def _spawn(engine, qi, dur, W):
    """Run W workers of one engine concurrently; return list of their stdout token-lists."""
    me = os.path.abspath(__file__)
    ps = [subprocess.Popen([sys.executable, me, engine, str(qi), str(dur)],
                           stdout=subprocess.PIPE, text=True) for _ in range(W)]
    return [p.communicate()[0].strip().split() for p in ps]


def run(budget_gb=32, cores=None):
    if cores is None:
        cores = os.cpu_count() or 8
    if not os.path.isdir(WDB):
        print(f"bench DB absent at {DIR}\n  build: python bench/join_prof.py 1.0"); sys.exit(2)
    sys.path.insert(0, HERE)
    import report                                       # ONE baseline builder/validator, shared with the
    report.duck_baseline()                              # latency scoreboard: (re)builds an empty/partial DB
    from catalog import QUERIES
    budget = budget_gb * 1024
    SOLO, CONC = 1.5, 2.0
    print(f"Throughput vs DuckDB @ {budget_gb} GB / {cores} cores -- real concurrent aggregate q/s.")
    print(f"  {'#':>2} {'query':<26}{'BSI':>4}{'solo':>9}{'W':>4}{'wdb@W':>9}{'duck@W':>9}{'ratio':>8}")
    rows = []
    for i, (cat, name, sql, _) in enumerate(QUERIES):
        s = _spawn('w', i, SOLO, 1)[0]
        try:
            solo = int(s[0]) / SOLO; priv = int(s[1]); fired = bool(int(s[2])); cube = bool(int(s[3]))
        except (ValueError, IndexError):
            rows.append((cat, name, False, 0, 0, 0.0, 0.0, 0.0, False))
            print(f"  {i:>2} {name:<26} ERR"); continue
        W = min(cores, max(1, budget // max(priv, 1)))
        wq = sum(int(o[0]) for o in _spawn('w', i, CONC, W) if o) / CONC
        dq = sum(int(o[0]) for o in _spawn('d', i, CONC, W) if o) / CONC
        ratio = wq / dq if dq > 0 else 0.0
        rows.append((cat, name, fired, solo, W, wq, dq, ratio, cube))
        print(f"  {i:>2} {name:<26}{'BSI' if fired else '-':>4}{solo:>9.0f}{W:>4}"
              f"{wq:>9.0f}{dq:>9.0f}{ratio:>7.2f}x")
    _write_md(rows, budget_gb, cores)


def _write_md(rows, budget_gb, cores):
    git = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    ratios = sorted(r[7] for r in rows if r[7] > 0)
    med = ratios[len(ratios) // 2] if ratios else 0.0
    wins = sum(1 for r in rows if r[7] >= 1.0)
    bsi_n = sum(1 for r in rows if r[2])
    cube_n = sum(1 for r in rows if len(r) > 8 and r[8])
    bw_n = sum(1 for r in rows if not (len(r) > 8 and r[8])      # cubes are parse-bound, not bandwidth-bound
               and r[3] > 0 and r[4] > 0 and (r[5] / (r[3] * r[4])) < 0.6)
    L = ["# WaveDB throughput scoreboard\n"]
    L.append(f"_TPC-H sf=1 - vs DuckDB - commit `{git}` - {datetime.date.today().isoformat()} - "
             f"W={cores} single-thread workers per engine, {budget_gb} GB budget._\n")
    L.append("Throughput is the optimization target for a shared analytical DB. Each engine runs W "
             "single-threaded workers concurrently (one per core); the number is the **real measured** "
             "aggregate queries/sec, memory-bandwidth contention included (not extrapolated from "
             "single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the "
             "BSI filter-index engages where it pays (BSI column).\n")
    L.append("Rows marked **cube** in the bound column are answered from a materialised low-card GROUP BY "
             "aggregate (a precomputed [count, sums] per cell, built at load time for filter-free group-bys "
             "whose cell count is under the cap) -- the query reads a few hundred bytes instead of scanning "
             "the value columns, so it is parse-bound, not bandwidth-bound. This is a materialised view: a "
             "DIFFERENT class than a faster scan (DuckDB could build the same), and it applies ONLY to "
             "filter-free low-card group-bys with COUNT/SUM/AVG; everything else falls back to the scan.\n")
    L.append("| # | category | query | BSI | WaveDB 1-wkr q/s | WaveDB @W q/s | scale | bound | "
             "DuckDB @W q/s | ratio |")
    L.append("|---|---|---|:-:|--:|--:|--:|:-:|--:|--:|")
    last = None
    for i, r in enumerate(rows):
        cat, name, fired, solo, W, wq, dq, ratio = r[:8]
        is_cube = len(r) > 8 and r[8]
        if cat != last:
            L.append(f"| **{cat}** | | | | | | | | | |"); last = cat
        scale = wq / (solo * W) if solo > 0 and W > 0 else 0.0
        bound = 'cube' if is_cube else ('RAM' if W < cores else ('BW' if scale < 0.6 else 'CPU'))
        mark = " **" if ratio >= 1 else ""
        L.append(f"| {i} | {cat} | {name} | {'Y' if fired else '-'} | {solo:.0f} | {wq:.0f} | "
                 f"{scale*100:.0f}% | {bound} | {dq:.0f} | {ratio:.2f}x{mark} |")
    L.append("")
    L.append(f"**{wins}/{len(rows)} faster than DuckDB - median {med:.2f}x.** "
             f"{cube_n} filter-free low-card group-bys are answered from a materialised cube (bound=cube; "
             f"precomputed aggregate, not a scan). BSI filter-index engaged on {bsi_n} "
             f"{'query' if bsi_n == 1 else 'queries'}. Of the scan queries, {bw_n}/{len(rows)} are "
             f"memory-bandwidth-bound under concurrency (scale < 60%): the shared memory bus, not core "
             f"count or RAM capacity, is the throughput wall -- so the lever is **bytes read per query** "
             f"(compression, the BSI index, and -- where the shape allows -- not reading at all via the "
             f"cube), not single-query latency. Bold ratios are WaveDB wins.\n")
    rep = os.path.join(ROOT, 'examples', 'throughput.md')
    open(rep, 'w').write("\n".join(L) + "\n")
    print(f"\nwrote {rep}")
    print(f"  {wins}/{len(rows)} faster, median {med:.2f}x, BSI on {bsi_n}, cube on {cube_n}")
    return rep


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else None
    if a in ('_worker', 'w'):
        _worker(int(sys.argv[2]), float(sys.argv[3]))
    elif a in ('_dworker', 'd'):
        _dworker(int(sys.argv[2]), float(sys.argv[3]))
    else:
        gb = int(a) if (a and a.isdigit()) else 32
        cores = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
        run(gb, cores)
