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

# Pin math libs to a single thread: one worker == one core, so W workers fill W
# cores without internal oversubscription skewing the per-worker latency.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
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


def _worker(qi):
    import time
    sys.path.insert(0, SRC); sys.path.insert(0, HERE)
    from wdb_db import Database
    from catalog import QUERIES
    db = Database.open(WDB)
    for a in FKS:
        try: db.create_fk_pointer(*a)
        except Exception: pass
    sql = QUERIES[qi][2]
    for _ in range(3): db.run(sql)                  # warm
    t = time.perf_counter(); db.run(sql); est = (time.perf_counter() - t) * 1000
    K = max(5, min(300, int(1500 / max(est, 0.5))))
    t = time.perf_counter()
    for _ in range(K): db.run(sql)
    ms = (time.perf_counter() - t) * 1000 / K
    print(f"{ms:.4f} {_smaps_private_dirty()}", flush=True)


def run(budget_gb=32, cores=None):
    if cores is None:
        cores = os.cpu_count() or 8
    if not os.path.isdir(WDB):
        print(f"bench DB absent at {WDB}\n  build: python bench/join_prof.py 1.0"); sys.exit(2)
    sys.path.insert(0, HERE)
    from catalog import QUERIES
    budget = budget_gb * 1024
    print(f"Throughput @ {budget_gb} GB / {cores} cores  (single-thread workers, "
          f"private working set per worker)\n")
    print(f"  {'#':>2} {'query':<26}{'lat ms':>8}{'QPS/core':>9}{'priv MB':>8}{'W':>4}{'bound':>6}{'QPS@'+str(budget_gb)+'GB':>11}")
    tot = 0.0
    for i, (cat, name, sql, _) in enumerate(QUERIES):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '_worker', str(i)],
                           capture_output=True, text=True)
        try:
            ms, priv = r.stdout.strip().split(); ms = float(ms); priv = int(priv)
        except ValueError:
            print(f"  {i+1:>2} {name:<26} ERR {r.stderr.strip().splitlines()[-1][:48] if r.stderr.strip() else '?'}")
            continue
        qps_core = 1000 / ms if ms > 0 else 0
        w_ram = max(1, budget // max(priv, 1))
        W = min(cores, w_ram)
        bound = 'RAM' if w_ram < cores else 'CPU'
        qps = qps_core * W
        tot += qps
        print(f"  {i+1:>2} {name:<26}{ms:>8.2f}{qps_core:>9.0f}{priv:>8}{W:>4}{bound:>6}{qps:>11.0f}")
    print(f"\n  All queries CPU-bound means RAM never throttles throughput at this budget.")
    print(f"  Break-even: a worker is RAM-bound only when RAM/core < its private MB above.")


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else None
    if a == '_worker':
        _worker(int(sys.argv[2]))
    else:
        gb = int(a) if (a and a.isdigit()) else 32
        cores = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
        run(gb, cores)
