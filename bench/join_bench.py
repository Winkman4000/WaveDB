"""Honest head-to-head: WaveDB vs DuckDB on TPC-H joins.
Axes: correctness, end-to-end query time, storage, and the pointer-walk gather kernel (FK pre-resolved
to a parent row pointer at load -> query-time join is a pure gather, no hash build)."""
import sys, os, time, tempfile, uuid, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, numpy as np, pandas as pd, zstandard as zstd
import wdb_encode
from wdb_db import Database
from wdb_engine import Segment

SF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
con = duckdb.connect()
con.execute(f"INSTALL tpch; LOAD tpch; CALL dbgen(sf={SF})")
DIR = os.path.join(tempfile.gettempdir(), f'jb_{uuid.uuid4().hex[:6]}'); os.makedirs(DIR)
db = Database.create(os.path.join(DIR, 'wdb'))
WT = {'BIGINT':'int','INTEGER':'int','VARCHAR':'string','DATE':'datetime'}
wt = lambda t: 'float' if t.startswith('DECIMAL') else WT[t]

def load(tbl, order_by=None, extra=None):
    desc = con.execute(f"DESCRIBE {tbl}").fetchall()
    cols = [(f"CAST({c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else c[0]) for c in desc]
    sch = [[c[0], wt(c[1])] for c in desc]
    if extra: cols.append(extra[0]); sch.append(extra[1])
    ob = f" ORDER BY {order_by}" if order_by else ""
    pq = os.path.join(DIR, f'{tbl}.parquet')
    con.execute(f"COPY (SELECT {', '.join(cols)} FROM {tbl}{ob}) TO '{pq}' (FORMAT parquet)")
    db.cat.add_table(tbl, sch); seg = f'{tbl}_0.wdb'
    wdb_encode.encode(pq, os.path.join(DIR, 'wdb', seg)); db.cat.add_segment(tbl, seg)
    return os.path.getsize(os.path.join(DIR, 'wdb', seg))

# customer sorted by custkey so row position == rank(custkey); pre-resolve orders.o_custkey -> pointer
load('customer', order_by='c_custkey')
con.execute("CREATE TEMP TABLE cust_pk AS SELECT c_custkey, row_number() OVER (ORDER BY c_custkey)-1 AS pos FROM customer")
load('orders', order_by='o_orderkey', extra=("(SELECT pos FROM cust_pk WHERE c_custkey=o_custkey) AS o_cust_ptr", ['o_cust_ptr','int']))
# orders sorted by orderkey (natural) so lineitem->orders is the clustered arm
con.execute("CREATE TEMP TABLE ord_pk AS SELECT o_orderkey, row_number() OVER (ORDER BY o_orderkey)-1 AS pos FROM orders")
load('lineitem', extra=("(SELECT pos FROM ord_pk WHERE o_orderkey=l_orderkey) AS l_ord_ptr", ['l_ord_ptr','int']))

def best(fn, n=3):
    fn(); ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)

def norm(rows):
    out=[]
    for r in rows:
        row=[]
        for x in r:
            if isinstance(x,(bytes,bytearray)): x=x.decode('utf-8','replace')
            elif hasattr(x,'as_integer_ratio') and not isinstance(x,(int,float)): x=float(x)
            if isinstance(x,(int,float)) and not isinstance(x,bool): x=round(float(x),2)
            row.append(x)
        out.append(tuple(row))
    return sorted(out, key=lambda t: tuple(str(x) for x in t))

print(f"\n{'='*64}\nTPC-H sf={SF}  |  rows: customer={con.execute('SELECT count(*) FROM customer').fetchone()[0]:,}  "
      f"orders={con.execute('SELECT count(*) FROM orders').fetchone()[0]:,}  "
      f"lineitem={con.execute('SELECT count(*) FROM lineitem').fetchone()[0]:,}\n{'='*64}")

# ---------- Query A: revenue by market segment (orders x customer, scattered FK) ----------
qA = ("SELECT c.c_mktsegment, COUNT(*), SUM(o.o_totalprice) FROM orders o "
      "JOIN customer c ON o.o_custkey = c.c_custkey GROUP BY c.c_mktsegment")
duck_A = lambda: con.execute(qA).fetchall()
wdb_A  = lambda: db.run(qA)[0]
def gather_A():
    custseg = Segment(os.path.join(DIR,'wdb','customer_0.wdb'))
    ordseg  = Segment(os.path.join(DIR,'wdb','orders_0.wdb'))
    ptr = ordseg.values('o_cust_ptr').astype(np.int64)
    seg = custseg.values('c_mktsegment')[ptr]          # GATHER: array index, no hash
    price = ordseg.values('o_totalprice')
    df = pd.DataFrame({'s':seg,'p':price}); g=df.groupby('s'); return list(zip(g.size().index, g.size(), g['p'].sum()))
okA = norm(duck_A())==norm(wdb_A()) and norm(duck_A())==norm([(s,n,p) for s,n,p in gather_A()])
print(f"\n[A] revenue by segment  (orders x customer, FK scattered)   correct={okA}")
print(f"    DuckDB        : {best(duck_A)*1000:8.1f} ms")
print(f"    WaveDB SQL    : {best(wdb_A)*1000:8.1f} ms")
print(f"    WaveDB gather : {best(gather_A)*1000:8.1f} ms   (FK pre-resolved -> pure gather)")

# ---------- Query B: lineitem x orders (clustered FK arm) ----------
qB = ("SELECT o.o_orderpriority, COUNT(*), SUM(l.l_quantity) FROM lineitem l "
      "JOIN orders o ON l.l_orderkey = o.o_orderkey GROUP BY o.o_orderpriority")
duck_B = lambda: con.execute(qB).fetchall()
wdb_B  = lambda: db.run(qB)[0]
def gather_B():
    li = Segment(os.path.join(DIR,'wdb','lineitem_0.wdb')); oo=Segment(os.path.join(DIR,'wdb','orders_0.wdb'))
    ptr = li.values('l_ord_ptr').astype(np.int64)
    pri = oo.values('o_orderpriority')[ptr]
    q = li.values('l_quantity')
    df=pd.DataFrame({'pr':pri,'q':q}); g=df.groupby('pr'); return list(zip(g.size().index,g.size(),g['q'].sum()))
okB = norm(duck_B())==norm(wdb_B()) and norm(duck_B())==norm(list(gather_B()))
print(f"\n[B] qty by priority     (lineitem x orders, FK clustered)   correct={okB}")
print(f"    DuckDB        : {best(duck_B)*1000:8.1f} ms")
print(f"    WaveDB SQL    : {best(wdb_B)*1000:8.1f} ms")
print(f"    WaveDB gather : {best(gather_B)*1000:8.1f} ms")

# ---------- Storage ----------
def fsize(p): return os.path.getsize(p)
def zsz(arr):
    z=zstd.ZstdCompressor(level=9)
    a=np.asarray(arr); 
    return len(z.compress(a.tobytes()))
zc=zstd.ZstdCompressor(level=9)
wdb_total = sum(fsize(os.path.join(DIR,'wdb',f)) for f in os.listdir(os.path.join(DIR,'wdb')) if f.endswith('.wdb'))
pq_total  = sum(fsize(os.path.join(DIR,f'{t}.parquet')) for t in ('customer','orders','lineitem'))
dpath=os.path.join(DIR,'tables.duckdb'); dcon=duckdb.connect(dpath)
for t in ('customer','orders','lineitem'):
    dcon.execute(f"CREATE TABLE {t} AS SELECT * FROM con_{t}") if False else None
dcon.execute("ATTACH ':memory:' AS m"); 
for t in ('customer','orders','lineitem'):
    df=con.execute(f"SELECT * FROM {t}").df(); dcon.register('tmp_'+t, df); dcon.execute(f"CREATE TABLE {t} AS SELECT * FROM tmp_"+t)
dcon.close(); duck_native=os.path.getsize(dpath)
# FK pointer column: clustered (lineitem->orders) vs scattered (orders->customer)
li=Segment(os.path.join(DIR,'wdb','lineitem_0.wdb')); oo=Segment(os.path.join(DIR,'wdb','orders_0.wdb'))
lptr=li.values('l_ord_ptr').astype(np.int32); optr=oo.values('o_cust_ptr').astype(np.int32)
def delta_z(a): 
    d=np.diff(a, prepend=a[0]).astype(np.int32); return len(zc.compress(d.tobytes()))
print(f"\n[STORAGE]")
print(f"    WaveDB segments (3 tables): {wdb_total/1e6:8.2f} MB")
print(f"    DuckDB native   (3 tables): {duck_native/1e6:8.2f} MB")
print(f"    Parquet zstd    (3 tables): {pq_total/1e6:8.2f} MB")
print(f"    FK ptr lineitem->orders : raw {lptr.nbytes/1e6:.2f}MB  delta+zstd {delta_z(lptr)/1e6:.2f}MB  (clustered arm)")
print(f"    FK ptr orders->customer : raw {optr.nbytes/1e6:.2f}MB  delta+zstd {delta_z(optr)/1e6:.2f}MB  (scattered arm)")
shutil.rmtree(DIR)
