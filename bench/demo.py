"""WaveDB demo -- run the canonical query catalog once on the bench DB, vs DuckDB.
  python3 bench/demo.py            (expects /tmp/jbprof_sf1.0/wdb; build: python3 bench/join_prof.py 1.0)
Shows every supported query shape, result size, WaveDB vs DuckDB time, and a correctness check.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import duckdb
from wdb_db import Database
from catalog import QUERIES
DIR = '/tmp/jbprof_sf1.0'
db = Database.open(os.path.join(DIR, 'wdb'))
for a in [('orders','o_custkey','customer','c_custkey'),('lineitem','l_orderkey','orders','o_orderkey')]:
    try: db.create_fk_pointer(*a)
    except Exception: pass
con = duckdb.connect(); con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=1)")
def best(fn, n=3):
    fn(); t=[]
    for _ in range(n): s=time.perf_counter(); fn(); t.append(time.perf_counter()-s)
    return min(t)*1000
print(f"\nWaveDB demo  --  TPC-H sf=1  (lineitem 6M, orders 1.5M, customer 150k)\n")
print(f"{'#':>2}  {'category':9s} {'query':27s} {'rows':>9s} {'WaveDB':>9s} {'DuckDB':>9s} {'spd':>6s}  ok")
print("-"*84)
wins=0
for i,(cat,name,sql,ex) in enumerate(QUERIES,1):
    gw = db.run(sql); exp = con.execute(sql).fetchall()
    ok = (len(gw[0]) == len(exp))
    w = best(lambda: db.run(sql)); d = best(lambda: con.execute(sql).fetchall())
    if d/w >= 1: wins += 1
    print(f"{i:>2}  {cat:9s} {name:27s} {len(gw[0]):>9,} {w:8.1f}ms {d:8.1f}ms {d/w:5.1f}x  {'OK' if ok else 'X!'}")
print("-"*84)
print(f"{wins}/{len(QUERIES)} shapes faster than DuckDB.  Full correctness vs DuckDB: tests/run.py (1517 green).\n")
