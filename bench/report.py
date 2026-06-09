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
import wdb_bsi_exec as BX
from catalog import QUERIES

DIR = '/tmp/jbprof_sf1.0'; WDB = os.path.join(DIR, 'wdb'); MB = 1024 * 1024
BSIBUDGET = BX.BSI_RAM_BUDGET


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


def _baseline_has_tables(ddb):
    """True iff the file exists AND actually holds the three bench tables. A bare
    duckdb.connect(path) (e.g. a stray inspection, or an interrupted build) creates an EMPTY
    file; trusting existence alone would then make every DuckDB measurement nan."""
    if not os.path.exists(ddb):
        return False
    try:
        c = duckdb.connect(ddb, read_only=True)
        have = {r[0] for r in c.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
        c.close()
        return {'customer', 'orders', 'lineitem'}.issubset(have)
    except Exception:
        return False


def duck_baseline():
    """A DuckDB native file holding ONLY the three bench tables (fair size comparison). Rebuilds
    when missing OR present-but-empty/partial (validated by table presence, not just file
    existence), so a stray/interrupted connect can never silently zero out the comparison."""
    ddb = os.path.join(DIR, 'baseline.duckdb')
    if not _baseline_has_tables(ddb):
        if os.path.exists(ddb):
            os.remove(ddb)                                  # empty/partial/corrupt -> rebuild clean
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
        "import sys, time\n"
        f"sys.path.insert(0,{HERE!r}); sys.path.insert(0,{SRC!r})\n"
        "from wdb_db import Database\n"
        "from catalog import QUERIES\n"
        f"db=Database.open({WDB!r})\n"
        "ms=0.0\n"
        "if sys.argv[1]!='floor':\n"
        "    for a in [('orders','o_custkey','customer','c_custkey'),('lineitem','l_orderkey','orders','o_orderkey')]:\n"
        "        try: db.create_fk_pointer(*a)\n"
        "        except Exception: pass\n"
        "    sql=QUERIES[int(sys.argv[1])][2]\n"
        "    for _ in range(5): db.run(sql, escalate=True)\n"   # warm to steady state (numba thread pool)
        "    ts=[]\n"
        "    for _ in range(8):\n"
        "        t=time.perf_counter(); db.run(sql, escalate=True); ts.append(time.perf_counter()-t)\n"
        "    ms=min(ts)*1000\n"
        "else:\n"
        "    db.run('SELECT COUNT(*) FROM lineitem')\n"
        "hwm=0\n"
        "for line in open('/proc/self/status'):\n"
        "    if line.startswith('VmHWM'): hwm=int(line.split()[1])\n"
        "print(ms, hwm)\n")

def _peak_kb(arg):
    """Run one query alone in a fresh process (no other engine resident -> production-representative).
    Returns (best_of_5_ms, peak_VmHWM_kb)."""
    r = subprocess.run([sys.executable, _MEMRUN, str(arg)], capture_output=True, text=True)
    try:
        p = r.stdout.strip().splitlines()[-1].split()
        return float(p[0]), int(p[1])
    except Exception: return float('nan'), -1


_DUCKMEM = '/tmp/_duck_memrun.py'
def _write_duckmem():
    open(_DUCKMEM, 'w').write(
        "import sys, time\n"
        f"sys.path.insert(0,{HERE!r})\n"
        "import duckdb\n"
        "from catalog import QUERIES\n"
        f"con=duckdb.connect({os.path.join(DIR,'baseline.duckdb')!r}, read_only=True)\n"
        "ms=0.0\n"
        "if sys.argv[1]!='floor':\n"
        "    sql=QUERIES[int(sys.argv[1])][2]\n"
        "    for _ in range(5): con.execute(sql).fetchall()\n"   # symmetric warmup
        "    ts=[]\n"
        "    for _ in range(8):\n"
        "        t=time.perf_counter(); con.execute(sql).fetchall(); ts.append(time.perf_counter()-t)\n"
        "    ms=min(ts)*1000\n"
        "else:\n"
        "    con.execute('SELECT COUNT(*) FROM lineitem').fetchall()\n"
        "hwm=0\n"
        "for line in open('/proc/self/status'):\n"
        "    if line.startswith('VmHWM'): hwm=int(line.split()[1])\n"
        "print(ms, hwm)\n")

def _duck_peak_kb(arg):
    """DuckDB alone in a fresh process. Returns (best_of_5_ms, peak_VmHWM_kb)."""
    r = subprocess.run([sys.executable, _DUCKMEM, str(arg)], capture_output=True, text=True)
    try:
        p = r.stdout.strip().splitlines()[-1].split()
        return float(p[0]), int(p[1])
    except Exception: return float('nan'), -1


def write_caps(out, mem, dmem, dsz, wtot, git):
    """Refresh the auto-generated measured-performance section inside CAPABILITIES.md (between
    sentinel markers), so capability list and live numbers live in ONE file, updated together."""
    B = "<!-- BEGIN MEASURED (auto-generated by bench/report.py - do not edit) -->"
    E = "<!-- END MEASURED -->"
    L = [B, "", "## Measured performance vs DuckDB", ""]
    L.append(f"_TPC-H sf=1, commit `{git}`, {datetime.date.today().isoformat()}. "
             f"Speed in ms (lower is better), best of 8 after 5 warmups. Each engine measured ALONE "
             f"in a fresh process per query (production-representative -- neither contends with the "
             f"other); RAM = peak VmHWM._\n")
    L.append(f"**Storage:** WaveDB {wtot/MB:.1f} MB vs DuckDB {dsz/MB:.1f} MB "
             f"= **{dsz/wtot:.2f}x smaller** (same data).\n")
    L.append("| capability | WaveDB | DuckDB | speed | WaveDB RAM | DuckDB RAM |")
    L.append("|---|--:|--:|:--|--:|--:|")
    sps = []; okc = 0; last = None
    for (cat, name, bits, nr, ok, fast, d, w), wk, dk in zip(out, mem, dmem):
        okc += ok
        if cat != last:
            L.append(f"| **{cat}** | | | | | |"); last = cat
        if w == w and w > 0:
            verdict = f"{d/w:.2f}x {'faster' if d/w >= 1 else 'slower'}"; sps.append(d/w)
            dd = f"{d:.1f} ms"; ww = f"{w:.1f} ms"
        else:
            verdict = "-"; dd = ww = "ERR"
        wkm = "-" if wk < 0 else f"{wk/1024:.0f} MB"
        dkm = "-" if dk < 0 else f"{dk/1024:.0f} MB"
        L.append(f"| {name} | {ww} | {dd} | {verdict} | {wkm} | {dkm} |")
    sps.sort(); med = sps[len(sps)//2] if sps else 0
    win = sum(1 for s in sps if s >= 1.0)
    L.append("")
    L.append(f"**{okc}/{len(out)} correct - {win}/{len(sps)} faster than DuckDB - "
             f"median {med:.2f}x.**")
    L.append(""); L.append(E)
    block = "\n".join(L)
    cap = os.path.join(ROOT, 'CAPABILITIES.md')
    txt = open(cap).read() if os.path.exists(cap) else ""
    if B in txt and E in txt:
        pre = txt[:txt.index(B)]; post = txt[txt.index(E) + len(E):]
        txt = pre + block + post
    else:
        txt = txt.rstrip() + "\n\n" + block + "\n"
    open(cap, 'w').write(txt)
    return cap


def write_md(tabs, fk, side, dsz, out, mem, dmem, floor, Nl, secs, bsi=(0, [])):
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
    bb, bcols = bsi
    if bcols:
        L.append(f"_Throughput mode (`escalate=False`, the default) additionally builds a BSI "
                 f"filter-index: {bb/MB:.1f} MB in RAM across {len(bcols)} column(s) "
                 f"({', '.join(bcols)}), built lazily only for filtered columns, capped at "
                 f"{BSIBUDGET/MB:.0f} MB/segment. The per-query table below is the **escalated** "
                 f"(latency) path -- fully parallel fused scan, no BSI -- so it does not include this._\n")
    L.append("## Per-query  (speed - memory - bits)\n")
    L.append(f"Each engine is measured ALONE in a fresh process per query (best-of-5 latency + peak "
             f"RAM) -- the production scenario, since WaveDB and DuckDB never run together in "
             f"deployment. Load floor (open + COUNT) = {floor/1024:.0f} MB; anything above that is "
             f"the query's own footprint.\n")
    L.append("| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |")
    L.append("|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|")
    okc = fc = 0; sps = []
    for i, ((cat, name, bits, nr, ok, fast, d, w), mkb, dkb) in enumerate(zip(out, mem, dmem), 1):
        okc += ok; fc += fast
        bs = "-" if bits <= 0 else (f"{bits/1e6:.0f} Mb" if bits < 1e9 else f"{bits/1e9:.2f} Gb")
        if w == w and w > 0:
            sp = f"{d/w:.2f}x"; sps.append(d/w); dd = f"{d:.1f} ms"; ww = f"{w:.1f} ms"
        else:
            sp = "-"; dd = ww = "ERR"
        mm = "-" if mkb < 0 else f"{mkb/1024:.0f} MB"
        dm = "-" if dkb < 0 else f"{dkb/1024:.0f} MB"
        L.append(f"| {i} | {cat} | {name} | {nr:,} | {bs} | {dd} | {ww} | {sp} | {mm} | {dm} | "
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
    print("running query matrix (correctness) ..."); out = M.run_matrix(db, con, COL, log=print, escalate=True)
    db = con = segs = None        # drop both engines before the isolated timing/RAM runs
    print("measuring WaveDB alone per query (fresh process each: latency + peak RAM) ...")
    _write_memrun(); floor = _peak_kb('floor')[1]
    mem = []; wms = []
    for i in range(len(QUERIES)):
        ms, k = _peak_kb(i); mem.append(k); wms.append(ms)
        print(f"  wdb  [{i:2d}] {ms:7.2f} ms  {k/1024:6.0f} MB  {QUERIES[i][1]}")
    print("measuring DuckDB alone per query (fresh process each: latency + peak RAM) ...")
    _write_duckmem(); dmem = []; dms = []
    for i in range(len(QUERIES)):
        ms, k = _duck_peak_kb(i); dmem.append(k); dms.append(ms)
        print(f"  duck [{i:2d}] {ms:7.2f} ms  {k/1024:6.0f} MB")
    # Replace the co-resident matrix timings with the isolated, production-representative ones
    # (each engine alone in its own process -- neither contends with the other's thread pool).
    out = [(c, n, b, nr, ok, fast, dms[i], wms[i])
           for i, (c, n, b, nr, ok, fast, _d, _w) in enumerate(out)]
    wtot = sum(tabs.values()) + fk + side
    git = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    # BSI filter-index footprint: run the filter-category queries in-process so the lazy
    # index builds, then read what it cost (additive in-RAM state -- option (b), no sidecar).
    abx = M.open_db()
    for a in [('orders', 'o_custkey', 'customer', 'c_custkey'),
              ('lineitem', 'l_orderkey', 'orders', 'o_orderkey')]:
        try: abx.create_fk_pointer(*a)
        except Exception: pass
    for c, n, sql, _ in QUERIES:
        if c == 'filter':
            try: abx.run(sql)
            except Exception: pass
    try:
        bseg = abx.open_segment(abx.cat.segment_paths('lineitem')[0], 'lineitem')
        bsi_acct = BX.footprint(bseg)
    except Exception:
        bsi_acct = (0, [])
    abx = None
    rep, s = write_md(tabs, fk, side, dsz, out, mem, dmem, floor, Nl, time.time() - t0, bsi=bsi_acct)
    cap = write_caps(out, mem, dmem, dsz, wtot, git)
    print(f"\nwrote {rep}\nwrote {cap}")
    print(f"  BSI filter-index: {bsi_acct[0]/MB:.1f} MB across {len(bsi_acct[1])} cols {bsi_acct[1]}")
    print(f"  storage: WaveDB {(sum(tabs.values())+fk+side)/MB:.1f} MB vs DuckDB {dsz/MB:.1f} MB "
          f"= {s[7]:.2f}x smaller")
    print(f"  queries: {s[0]}/{s[1]} correct, {s[2]} fused, {s[3]} faster, median {s[4]:.2f}x")
    print(f"  peak RAM: median {s[5]:.0f} MB / max {s[6]:.0f} MB  (floor {floor/1024:.0f} MB)")


if __name__ == '__main__':
    main()
