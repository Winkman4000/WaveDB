#!/usr/bin/env python3
"""bench/throughput.py -- per-query throughput at a RAM budget.

The RAM footprint of a columnar engine only matters as a constraint on
throughput. This measures, for each catalog query, the metric that actually
decides deployment economics:

    queries/sec achievable at a fixed RAM budget (default 32 GB)

Model: dedicate the box to one query type and run W single-threaded workers in
parallel (threads pinned to 1 so W workers saturate W cores cleanly). Then

    throughput = W x (1000 / latency_ms)
    W          = min(cores, (budget_MB - shared_MB) / private_per_worker_MB)

Memmap'd segment files are shared across workers (paid once), so the number that
scales per worker is the private (anonymous) working set, not the total RSS.
A worker is RAM-bound only when RAM-per-core < its private working set; on all
normal hardware (>=1 GB/core) every query is CPU-bound. See docs/throughput.md.

  python bench/throughput.py [budget_gb] [cores]      # defaults 32, nproc

Needs the bench DB at /tmp/jbprof_sf1.0 (build: python bench/join_prof.py 1.0).
"""
import os, sys, subprocess

# Pin math libs AND numba to a single thread: the throughput model is W single-threaded
# workers, one per core. numba.prange must be pinned too, else the per-worker latency is the
# 8-thread parallel time while W still = cores -> QPS overstated by the (poor) parallel speedup.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); SRC = os.path.join(ROOT, 'src')
WDB = '/tmp/jbprof_sf1.0/wdb'
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
    # fallback: VmHWM (overcounts -- includes shared file pages)
    for l in open('/proc/self/status'):
        if l.startswith('VmHWM'):
            return int(l.split()[1]) // 1024
    return -1


def _worker(qi, dur):
    import time
    sys.path.insert(0, SRC); sys.path.insert(0, HERE)
    from wdb_db import Database
    from catalog import QUERIES
    import numba; numba.set_num_threads(1)          # one worker == one core
    db = Database.open(WDB)
    for a in FKS:
        try: db.create_fk_pointer(*a)
        except Exception: pass
    sql = QUERIES[qi][2]
    for _ in range(3): db.run(sql)                  # warm (JIT + caches)
    end = time.perf_counter() + dur; c = 0
    while time.perf_counter() < end: db.run(sql); c += 1
    print(f"{c} {_smaps_private_dirty()}", flush=True)


def run(budget_gb=32, cores=None):
    if cores is None:
        cores = os.cpu_count() or 8
    if not os.path.isdir(WDB):
        print(f"bench DB absent at {WDB}\n  build: python bench/join_prof.py 1.0"); sys.exit(2)
    sys.path.insert(0, HERE)
    from catalog import QUERIES
    budget = budget_gb * 1024
    print(f"Sustained throughput @ {budget_gb} GB / {cores} cores -- ACTUAL concurrent aggregate.\n"
          f"  W single-thread workers run AT THE SAME TIME; the @W column is real measured\n"
          f"  queries/sec, so memory-bandwidth contention is included (not extrapolated).\n")
    print(f"  {'#':>2} {'query':<26}{'1wkr q/s':>9}{'priv MB':>8}{'W':>4}{'@W q/s':>9}{'scale':>7}{'bound':>6}")
    me = os.path.abspath(__file__); SOLO, CONC = 1.5, 2.0
    for i, (cat, name, sql, _) in enumerate(QUERIES):
        r = subprocess.run([sys.executable, me, '_worker', str(i), str(SOLO)], capture_output=True, text=True)
        try:
            c, priv = r.stdout.strip().split(); solo = int(c) / SOLO; priv = int(priv)
        except ValueError:
            print(f"  {i+1:>2} {name:<26} ERR {r.stderr.strip().splitlines()[-1][:46] if r.stderr.strip() else '?'}")
            continue
        w_ram = max(1, budget // max(priv, 1))
        W = min(cores, w_ram)
        ps = [subprocess.Popen([sys.executable, me, '_worker', str(i), str(CONC)],
                               stdout=subprocess.PIPE, text=True) for _ in range(W)]
        agg = sum(int(p.communicate()[0].strip().split()[0]) for p in ps) / CONC
        scale = agg / (solo * W) if solo > 0 else 0
        bound = 'RAM' if w_ram < cores else ('BW' if scale < 0.6 else 'CPU')
        print(f"  {i+1:>2} {name:<26}{solo:>9.0f}{priv:>8}{W:>4}{agg:>9.0f}{scale*100:>6.0f}%{bound:>6}")
    print(f"\n  @W q/s = real aggregate with W workers running together (lower than W x 1wkr when")
    print(f"  scaling < 100%: the box is memory-bandwidth-bound, not CPU- or RAM-bound).")
    print(f"  bound: RAM = won't fit W=cores in budget; BW = scales < 60% (bandwidth); CPU otherwise.")


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else None
    if a == '_worker':
        _worker(int(sys.argv[2]), float(sys.argv[3]))
    else:
        gb = int(a) if (a and a.isdigit()) else 32
        cores = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
        run(gb, cores)
