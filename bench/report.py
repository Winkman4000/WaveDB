#!/usr/bin/env python3
"""bench/report.py -- the living scoreboard (one regenerated artifact).

  (1) STORAGE   WaveDB on-disk (per table + FK pointers) vs DuckDB native -> compression total
  (2) PER QUERY correctness, speed vs DuckDB, fused path, code-bits read, peak-RAM fingerprint

Peak RAM is the true kernel high-water mark (VmHWM) measured in a FRESH subprocess per query,
so each number is that query's real footprint, not a within-process running max.

Writes examples/report.md (+ stdout summary). Needs the bench DB at /tmp/jbprof_sf1.0/wdb
(build: python bench/join_prof.py 1.0).
"""
import sys, os, time, subprocess, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC  = os.path.join(ROOT, 'src')
sys.path.insert(0, HERE); sys.path.insert(0, SRC)
import duckdb
import query_matrix as M
from catalog import QUERIES

DIR = '/tmp/jbprof_sf1.0'; WDB = os.path.join(DIR, 'wdb'); MB = 1024 * 1024


def _need_db():
    if not os.path.isdir(WDB):
        print(f"bench DB absent at {WDB}\n  build: python bench/join_prof.py 1.0")
        sys.exit(2)


def wdb_sizes():
    tables = {}; fk = 0; side = 0
    for fn in os.listdir(WDB):
        sz = os.path.getsize(os.path.join(WDB, fn))
        if '.fkptr' in fn:            fk += sz
        elif fn.endswith('.wdb'):     tables[fn[:-6].replace('_0', '')] = sz
        else:                         side += sz
    return tables, fk, side


def duck_baseline():
    """A DuckDB native file holding ONLY the three bench tables (fair size comparison)."""
    ddb = os.path.join(DIR, 'baseline.duckdb')
    if not os.path.exists(ddb):
        con = duckdb.connect()
        con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=1)")
        con.execute(f"ATTACH '{ddb}' AS f")
        for t in ('customer', 'orders', 'lineitem'):
            con.execute(f"CREATE TABLE f.{t} AS SELECT * FROM {t}")
        con.execute("CHECKPOINT f"); con.close()
    return os.path.getsize(ddb)


_MEMRUN = '/tmp/_wdb_memrun.py'
def _write_memrun():
    open(_MEMRUN, 'w').write(
        "import sys\n"
        f"sys.path.insert(0,{HERE!r}); sys.path.insert(0,{SRC!r})\n"
        "from wdb_db import Database\n"
        "from catalog import QUERIES\n"
        f"db=Database.open({WDB!r})\n"
        "if sys.argv[1]!='floor':\n"
        "    for a in [('orders','o_custkey','customer','c_custkey'),('lineitem','l_orderkey','orders','o_orderkey')]:\n"
        "        try: db.create_fk_pointer(*a)\n"
        "        except Exception: pass\n"
        "    sql=QUERIES[int(sys.argv[1])][2]\n"
        "    db.run(sql); db.run(sql)\n"
        "else:\n"
        "    db.run('SELECT COUNT(*) FROM lineitem')\n"
        "hwm=0\n"
        "for line in open('/proc/self/status'):\n"
        "    if line.startswith('VmHWM'): hwm=int(line.split()[1])\n"
        "print(hwm)\n")

def _peak_kb(arg):
    r = subprocess.run([sys.executable, _MEMRUN, str(arg)], capture_output=True, text=True)
    try: return int(r.stdout.strip().splitlines()[-1])
    except Exception: return -1


def write_md(tabs, fk, side, dsz, out, mem, floor, Nl, secs):
    wtab = sum(tabs.values()); wtot = wtab + fk + side
    git = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    L = []
    L.append("# WaveDB scoreboard\n")
    L.append(f"_TPC-H sf=1 (lineitem N={Nl:,}) - vs DuckDB - commit `{git}` - "
             f"{datetime.date.today().isoformat()} - generated in {secs:.0f}s_\n")
    L.append("## Storage  (compression total)\n")
    L.append("| store | size | vs DuckDB |")
    L.append("|---|--:|--:|")
    for t in ('lineitem', 'orders', 'customer'):
        if t in tabs:
            L.append(f"| WaveDB {t} | {tabs[t]/MB:.1f} MB | |")
    L.append(f"| WaveDB FK pointers | {fk/MB:.1f} MB | _(join index, like a sort key)_ |")
    L.append(f"| **WaveDB total** | **{wtot/MB:.1f} MB** | **{dsz/wtot:.2f}x smaller** |")
    L.append(f"| DuckDB native (3 tables) | {dsz/MB:.1f} MB | 1.00x |")
    L.append("")
    L.append(f"WaveDB stores the same data in **{dsz/wtot:.2f}x less space** than DuckDB "
             f"({wtot/MB:.1f} MB vs {dsz/MB:.1f} MB), FK-pointer join index included. "
             f"Column data alone is {wtab/MB:.1f} MB ({dsz/wtab:.2f}x).\n")
    L.append("## Per-query  (speed - memory - bits)\n")
    L.append(f"Peak RAM = VmHWM in a fresh process per query. Load floor (open + COUNT) "
             f"= {floor/1024:.0f} MB; anything above that is the query's own footprint.\n")
    L.append("| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | peak RAM | fused | ok |")
    L.append("|---|---|---|--:|--:|--:|--:|--:|--:|:-:|:-:|")
    okc = fc = 0; sps = []
    for i, ((cat, name, bits, nr, ok, fast, d, w), mkb) in enumerate(zip(out, mem), 1):
        okc += ok; fc += fast
        bs = "-" if bits <= 0 else (f"{bits/1e6:.0f} Mb" if bits < 1e9 else f"{bits/1e9:.2f} Gb")
        if w == w and w > 0:
            sp = f"{d/w:.2f}x"; sps.append(d/w); dd = f"{d:.1f} ms"; ww = f"{w:.1f} ms"
        else:
            sp = "-"; dd = ww = "ERR"
        mm = "-" if mkb < 0 else f"{mkb/1024:.0f} MB"
        L.append(f"| {i} | {cat} | {name} | {nr:,} | {bs} | {dd} | {ww} | {sp} | {mm} | "
                 f"{'Y' if fast else '-'} | {'ok' if ok else 'X'} |")
    sps.sort(); med = sps[len(sps)//2] if sps else 0
    win = sum(1 for s in sps if s >= 1.0)
    vmem = [m for m in mem if m > 0]
    pmax = max(vmem)/1024 if vmem else 0
    pmed = sorted(vmem)[len(vmem)//2]/1024 if vmem else 0
    L.append("")
    L.append(f"**{okc}/{len(out)} correct - {fc}/{len(out)} fused - {win}/{len(sps)} faster than DuckDB "
             f"- median {med:.2f}x - peak RAM median {pmed:.0f} MB / max {pmax:.0f} MB**\n")
    rep = os.path.join(ROOT, 'examples', 'report.md')
    open(rep, 'w').write("\n".join(L) + "\n")
    return rep, (okc, len(out), fc, win, med, pmed, pmax, dsz/wtot)


def main():
    _need_db(); t0 = time.time()
    tabs, fk, side = wdb_sizes()
    print("building DuckDB baseline (once) ..."); dsz = duck_baseline()
    db = M.open_db(); con = M.open_duck(); segs = M.open_segs(); COL = M.build_cols(segs)
    Nl = segs['lineitem'].N
    print("running query matrix ..."); out = M.run_matrix(db, con, COL, log=print)
    print("measuring peak RAM per query (fresh process each) ...")
    _write_memrun(); floor = _peak_kb('floor')
    mem = []
    for i in range(len(QUERIES)):
        k = _peak_kb(i); mem.append(k); print(f"  mem[{i:2d}] {k/1024:6.0f} MB  {QUERIES[i][1]}")
    rep, s = write_md(tabs, fk, side, dsz, out, mem, floor, Nl, time.time() - t0)
    print(f"\nwrote {rep}")
    print(f"  storage: WaveDB {(sum(tabs.values())+fk+side)/MB:.1f} MB vs DuckDB {dsz/MB:.1f} MB "
          f"= {s[7]:.2f}x smaller")
    print(f"  queries: {s[0]}/{s[1]} correct, {s[2]} fused, {s[3]} faster, median {s[4]:.2f}x")
    print(f"  peak RAM: median {s[5]:.0f} MB / max {s[6]:.0f} MB  (floor {floor/1024:.0f} MB)")


if __name__ == '__main__':
    main()
