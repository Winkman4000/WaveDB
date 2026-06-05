import sys, os, time, math, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb
from wdb_db import Database
import wdb_join
DIR='/tmp/jbprof_sf1.0'
db=Database.open(os.path.join(DIR,'wdb'))
for a in [('orders','o_custkey','customer','c_custkey'),('lineitem','l_orderkey','orders','o_orderkey')]:
    try: db.create_fk_pointer(*a); print(f"fkptr {a[0]}.{a[1]} -> {a[2]}.{a[3]}  OK")
    except Exception as e: print(f"fkptr {a}: ERR {e}")
con=duckdb.connect(); con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=1)")
def _c(x):
    from decimal import Decimal
    if isinstance(x,Decimal): x=float(x)
    if isinstance(x,(datetime.datetime,datetime.date)): return str(x)[:10]
    return x
def _n(rs): return sorted(([_c(c) for c in r] for r in rs), key=lambda t: tuple((v is None,str(v)) for v in t))
def _eq(g,e):
    if len(g)!=len(e): return False
    for gr,er in zip(g,e):
        for a,b in zip(gr,er):
            if isinstance(a,float) or isinstance(b,float):
                if not math.isclose(float(a),float(b),rel_tol=1e-6,abs_tol=1e-2): return False
            elif a!=b: return False
    return True
def best(fn,n=5):
    fn(); ts=[]
    for _ in range(n): t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)*1000
JOINS=[
 ("JOIN group parent-key","SELECT c.c_mktsegment,COUNT(*),SUM(o.o_totalprice) FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment"),
 ("JOIN group child-key","SELECT l.l_returnflag,SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY l.l_returnflag"),
 ("JOIN group parent-date hiK","SELECT o.o_orderdate,SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY o.o_orderdate"),
 ("JOIN + WHERE","SELECT o.o_orderpriority,SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey WHERE l.l_quantity > 30 GROUP BY o.o_orderpriority"),
 ("3-table JOIN","SELECT c.c_mktsegment,SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment"),
]
print(f"\n{'shape':28s} {'ok':>3s} {'fast':>4s} {'DuckDB':>9s} {'WaveDB':>9s} {'speedup':>7s}")
for name,q in JOINS:
    h=wdb_join._FAST_HITS; gw=db.run(q); fast=wdb_join._FAST_HITS>h
    ok=_eq(_n(gw[0]), _n([tuple(r) for r in con.execute(q).fetchall()]))
    w=best(lambda:db.run(q)); d=best(lambda:con.execute(q).fetchall())
    print(f"{name:28s} {('Y' if ok else 'N!'):>3s} {('Y' if fast else 'n'):>4s} {d:8.1f}ms {w:8.1f}ms {d/w:6.2f}x")
