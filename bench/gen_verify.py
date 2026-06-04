"""Generalize-and-verify: many join/agg shapes through the fast path vs DuckDB at sf=1.
Checks correctness (sorted compare) AND speed, and asserts the fast path was actually taken."""
import sys, os, time, math, datetime, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb
from wdb_db import Database
import wdb_join

db = Database.open('/tmp/jbprof_sf1.0/wdb')
con = duckdb.connect(); con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=1)")

_TS = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(\.\d+)?$')
def _cell(c):
    from decimal import Decimal
    if isinstance(c, Decimal): c = float(c)
    if isinstance(c, datetime.datetime): return c.strftime('%Y-%m-%d %H:%M:%S').replace(' 00:00:00','')
    if isinstance(c, datetime.date): return c.strftime('%Y-%m-%d')
    if isinstance(c, str):
        m = _TS.match(c)
        if m: c = m.group(1)
        if c.endswith(' 00:00:00'): c = c[:10]
    return c
def _norm(rows):
    out=[tuple(_cell(c) for c in r) for r in rows]
    return sorted(out, key=lambda t: tuple((x is None,str(x)) for x in t))
def _eq(g,e):
    if len(g)!=len(e): return False
    for gr,er in zip(g,e):
        for a,b in zip(gr,er):
            if isinstance(a,float) or isinstance(b,float):
                if a is None or b is None:
                    if a is not b: return False
                elif not math.isclose(float(a),float(b),rel_tol=1e-6,abs_tol=1e-3): return False
            elif a!=b: return False
    return True
def best(fn,n=5):
    fn(); ts=[]
    for _ in range(n): t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)*1000

SHAPES = [
 ("A revenue/segment (scattered,K5)",   "SELECT c.c_mktsegment, COUNT(*), SUM(o.o_totalprice) FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment"),
 ("B qty/priority (clustered,K5)",      "SELECT o.o_orderpriority, COUNT(*), SUM(l.l_quantity) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY o.o_orderpriority"),
 ("high-K parent date (K~2400)",        "SELECT o.o_orderdate, SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY o.o_orderdate"),
 ("child group key returnflag (K3)",    "SELECT l.l_returnflag, COUNT(*), SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY l.l_returnflag"),
 ("child group shipmode (K7)",          "SELECT l.l_shipmode, AVG(l.l_quantity) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY l.l_shipmode"),
 ("WHERE mask 6M",                      "SELECT o.o_orderpriority, SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey WHERE l.l_quantity > 30 GROUP BY o.o_orderpriority"),
 ("AVG scattered",                      "SELECT c.c_mktsegment, AVG(o.o_totalprice) FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment"),
 ("mixed incl MIN/MAX (serial)",        "SELECT c.c_mktsegment, MIN(o.o_totalprice), MAX(o.o_totalprice), AVG(o.o_totalprice) FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment"),
 ("whole-table + WHERE",                "SELECT COUNT(*), SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey WHERE o.o_orderpriority='1-URGENT'"),
]
print(f"{'shape':38s} {'rows':>5s}  {'ok':>3s}  {'fast':>4s}  {'DuckDB':>8s} {'WaveDB':>8s}  {'spd':>5s}")
for name,q in SHAPES:
    h=wdb_join._FAST_HITS; gw=db.run(q); fast=wdb_join._FAST_HITS>h
    ok=_eq(_norm(gw[0]), _norm([tuple(r) for r in con.execute(q).fetchall()]))
    d=best(lambda:con.execute(q).fetchall()); w=best(lambda:db.run(q))
    print(f"{name:38s} {len(gw[0]):5d}  {('Y' if ok else 'N!'):>3s}  {('Y' if fast else 'n'):>4s}  {d:7.1f}ms {w:7.1f}ms  {d/w:4.2f}x")
