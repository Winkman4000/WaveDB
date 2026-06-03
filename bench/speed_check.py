"""Does the gather fast path reach db.run()? Reuse the persistent sf=1 build, add FK pointers, and
time the actual SQL join queries vs DuckDB."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb
from wdb_db import Database
import wdb_join

DIR = '/tmp/jbprof_sf1.0'
db = Database.open(os.path.join(DIR, 'wdb'))
try: db.create_fk_pointer('orders', 'o_custkey', 'customer', 'c_custkey')
except Exception as e: print("ptrA:", e)
try: db.create_fk_pointer('lineitem', 'l_orderkey', 'orders', 'o_orderkey')
except Exception as e: print("ptrB:", e)

con = duckdb.connect(); con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=1)")

qA = ("SELECT c.c_mktsegment, COUNT(*), SUM(o.o_totalprice) FROM orders o "
      "JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment")
qB = ("SELECT o.o_orderpriority, COUNT(*), SUM(l.l_quantity) FROM lineitem l "
      "JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY o.o_orderpriority")

def best(fn, n=5):
    fn(); ts=[]
    for _ in range(n): t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)*1000

for tag, q in (('[A] orders x customer', qA), ('[B] lineitem x orders', qB)):
    h0 = wdb_join._FAST_HITS; db.run(q); fast = wdb_join._FAST_HITS > h0
    d = best(lambda: con.execute(q).fetchall())
    w = best(lambda: db.run(q))
    print(f"{tag:24s}  DuckDB {d:7.1f} ms   WaveDB db.run {w:7.1f} ms   fast_path={fast}   speedup {d/w:.2f}x")
