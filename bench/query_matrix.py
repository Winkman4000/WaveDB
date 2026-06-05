"""Query-type matrix: for every query shape WaveDB supports, measure
   BITS READ (sum of code-stream bits of the columns it touches: N x code_width)
   and SPEED (best-of-5 ms) vs DuckDB on identical TPC-H sf=1 data.
Correctness checked against DuckDB. Output -> /tmp/qmatrix.md (+ stdout).
"""
import sys, os, time, math, datetime, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, numpy as np
from wdb_db import Database
from wdb_engine import Segment
import wdb_join

DIR = '/tmp/jbprof_sf1.0'
db = Database.open(os.path.join(DIR, 'wdb'))
# FK relationships are pre-resolved once (like building an index/sort key) so equi-joins
# become a gather instead of a runtime hash build.
for _a in [('orders','o_custkey','customer','c_custkey'),('lineitem','l_orderkey','orders','o_orderkey')]:
    try: db.create_fk_pointer(*_a)
    except Exception: pass
con = duckdb.connect(); con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=1)")

SEGS = {t: Segment(os.path.join(DIR, 'wdb', f'{t}_0.wdb')) for t in ('lineitem','orders','customer')}
COL = {}                                   # colname -> (table_rows, code_bits, on_disk_bytes)
for t, sg in SEGS.items():
    for nm, c in sg.cols.items():
        nb = (sg.N*c['bits']+7)//8 if c.get('code_enc',0)==0 else c.get('czlen', 0)
        COL[nm] = (sg.N, c['bits'], nb)

def bits_read(sql):
    refs = sorted(set(re.findall(r'\b([loc]_[a-z_]+)\b', sql)))
    refs = [r for r in refs if r in COL and r != 'l_ord_ptr']
    return sum(COL[r][0]*COL[r][1] for r in refs), refs

def _cell(c):
    from decimal import Decimal
    if isinstance(c, Decimal): c = float(c)
    if isinstance(c, datetime.datetime): return c.strftime('%Y-%m-%d %H:%M:%S').replace(' 00:00:00','')
    if isinstance(c, datetime.date): return c.strftime('%Y-%m-%d')
    return c
def _norm(rows):
    return sorted(([_cell(c) for c in r] for r in rows), key=lambda t: tuple((x is None, str(x)) for x in t))
def _eq(g, e):
    if len(g) != len(e): return False
    for gr, er in zip(g, e):
        if len(gr) != len(er): return False
        for a, b in zip(gr, er):
            if isinstance(a, float) or isinstance(b, float):
                if a is None or b is None:
                    if a is not b: return False
                elif not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-2): return False
            elif a != b: return False
    return True
def best(fn, n=5):
    fn(); ts = []
    for _ in range(n): t = time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)*1000

from catalog import QUERIES   # single source of truth (also feeds docs/queries.md + planner workload)

print(f"lineitem N={SEGS['lineitem'].N:,}  orders N={SEGS['orders'].N:,}  customer N={SEGS['customer'].N:,}\n")
out = []
for i,(cat,name,q,_ex) in enumerate(QUERIES, 1):
    h = wdb_join._FAST_HITS
    try:
        gw = db.run(q); fast = wdb_join._FAST_HITS > h
        exp = [tuple(r) for r in con.execute(q).fetchall()]
        ok = _eq(_norm(gw[0]), _norm(exp))
        w = best(lambda: db.run(q)); d = best(lambda: con.execute(q).fetchall())
        bits,_ = bits_read(q)
        out.append((cat,name,bits,len(gw[0]),ok,fast,d,w))
        print(f"{i:2d} {name:30s} {bits/1e6:8.1f}Mb  ok={ok} fast={fast}  D={d:7.1f} W={w:7.1f} {d/w:5.2f}x")
    except Exception as e:
        out.append((cat,name,-1,0,False,False,float('nan'),float('nan')))
        print(f"{i:2d} {name:30s}  ERROR: {str(e)[:70]}")

with open('/tmp/qmatrix.md','w') as f:
    f.write("| # | category | query type | rows out | code bits read | DuckDB | WaveDB | speedup | fast | ✓ |\n")
    f.write("|---|---|---|--:|--:|--:|--:|--:|:-:|:-:|\n")
    for i,(cat,name,bits,nr,ok,fast,d,w) in enumerate(out,1):
        bs = "—" if bits<=0 else (f"{bits/1e6:.1f} Mbit" if bits<1e9 else f"{bits/1e9:.2f} Gbit")
        sp = "" if (w!=w or w==0) else f"{d/w:.2f}x"
        f.write(f"| {i} | {cat} | {name} | {nr:,} | {bs} | {d:.1f} ms | {w:.1f} ms | {sp} | {'Y' if fast else '—'} | {'✓' if ok else '✗'} |\n")
    okc = sum(1 for r in out if r[4]); fc = sum(1 for r in out if r[5])
    sps = [r[6]/r[7] for r in out if r[7]==r[7] and r[7]>0]
    sps.sort(); med = sps[len(sps)//2] if sps else 0
    f.write(f"\n**{okc}/{len(out)} correct vs DuckDB · {fc}/{len(out)} on fused fast path · median speedup {med:.2f}x**\n")
print(f"\nwrote /tmp/qmatrix.md  ({okc}/{len(out)} correct, {fc} fast, median {med:.2f}x)")
