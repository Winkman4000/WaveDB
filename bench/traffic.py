#!/usr/bin/env python3
"""bench/traffic.py -- sustained-traffic RAM test (one command, both engines).

Opens each engine ONCE, runs `passes` full sweeps of the catalog
(passes x len(QUERIES) queries total), and samples peak RAM (VmHWM) after each
sweep -- showing whether memory grows with traffic or plateaus to a fixed
working set. Each engine runs in its own fresh subprocess so the two footprints
never contaminate each other.

  python bench/traffic.py [passes]      # default 6

Needs the bench DB at /tmp/jbprof_sf1.0 (build: python bench/join_prof.py 1.0).
"""
import sys, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); SRC = os.path.join(ROOT, 'src')
DIR = '/tmp/jbprof_sf1.0'; WDB = os.path.join(DIR, 'wdb')
DUCK = os.path.join(DIR, 'baseline.duckdb')


def _hwm():
    for l in open('/proc/self/status'):
        if l.startswith('VmHWM'): return int(l.split()[1]) // 1024
    return -1


def _wdb_worker(passes):
    sys.path.insert(0, SRC); sys.path.insert(0, HERE)
    from wdb_db import Database
    from catalog import QUERIES
    db = Database.open(WDB)
    for a in [('orders', 'o_custkey', 'customer', 'c_custkey'),
              ('lineitem', 'l_orderkey', 'orders', 'o_orderkey')]:
        try: db.create_fk_pointer(*a)
        except Exception: pass
    qs = [q[2] for q in QUERIES]
    print(f"0 {_hwm()}", flush=True)
    for rnd in range(passes):
        for sql in qs: db.run(sql)
        print(f"{rnd+1} {_hwm()}", flush=True)


def _duck_worker(passes):
    sys.path.insert(0, HERE)
    import duckdb
    from catalog import QUERIES
    con = duckdb.connect(DUCK, read_only=True)
    qs = [q[2] for q in QUERIES]
    print(f"0 {_hwm()}", flush=True)
    for rnd in range(passes):
        for sql in qs: con.execute(sql).fetchall()
        print(f"{rnd+1} {_hwm()}", flush=True)


def _spawn(mode, passes):
    r = subprocess.run([sys.executable, os.path.abspath(__file__), mode, str(passes)],
                       capture_output=True, text=True)
    out = {}
    for line in r.stdout.strip().splitlines():
        try: p, m = line.split(); out[int(p)] = int(m)
        except ValueError: pass
    return out


def run(passes=6):
    if not os.path.isdir(WDB):
        print(f"bench DB absent at {WDB}\n  build: python bench/join_prof.py 1.0"); sys.exit(2)
    sys.path.insert(0, HERE)
    from catalog import QUERIES
    nq = len(QUERIES)
    print(f"measuring WaveDB ({passes} passes) ..."); w = _spawn('_wdb', passes)
    print(f"measuring DuckDB ({passes} passes) ..."); d = _spawn('_duck', passes)
    print(f"\nSustained-traffic RAM -- {passes} passes x {nq} queries = "
          f"{passes*nq} queries, one process per engine\n")
    print("| after | WaveDB | DuckDB |")
    print("|---|--:|--:|")
    print(f"| open | {w.get(0,'-')} MB | {d.get(0,'-')} MB |")
    for p in range(1, passes + 1):
        print(f"| {p*nq} queries ({p}p) | {w.get(p,'-')} MB | {d.get(p,'-')} MB |")
    wf = w.get(passes, 0); df = d.get(passes, 0)
    if wf and df:
        print(f"\nsteady state: WaveDB {wf} MB vs DuckDB {df} MB = {wf/df:.2f}x heavier")
        g = w.get(passes, 0) - w.get(max(1, passes // 2), 0)
        print(f"WaveDB growth over last {passes - max(1, passes//2)} passes: {g} MB "
              f"(flat = doesn't grow with traffic)")


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else None
    ps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    if a == '_wdb': _wdb_worker(ps)
    elif a == '_duck': _duck_worker(ps)
    else: run(int(a) if (a and a.isdigit()) else 6)
