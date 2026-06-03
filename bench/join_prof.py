"""Profile the gather-join hotspot: where does the time actually go?
Breaks the lineitem x orders aggregate into decode / gather(walk) / group-by, and contrasts pandas
object-key group-by against a numpy group-by on WaveDB's integer codes."""
import sys, os, time, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, numpy as np, pandas as pd
import wdb_encode
from wdb_db import Database
from wdb_engine import Segment

SF = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
DIR = f'/tmp/jbprof_sf{SF}'
WT = {'BIGINT':'int','INTEGER':'int','VARCHAR':'string','DATE':'datetime'}
wt = lambda t: 'float' if t.startswith('DECIMAL') else WT[t]

if not os.path.exists(DIR):
    os.makedirs(DIR); con = duckdb.connect(); con.execute(f"INSTALL tpch; LOAD tpch; CALL dbgen(sf={SF})")
    db = Database.create(os.path.join(DIR,'wdb'))
    def load(tbl, order_by=None, extra=None):
        desc = con.execute(f"DESCRIBE {tbl}").fetchall()
        cols=[(f"CAST({c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else c[0]) for c in desc]
        sch=[[c[0],wt(c[1])] for c in desc]
        if extra: cols.append(extra[0]); sch.append(extra[1])
        ob=f" ORDER BY {order_by}" if order_by else ""
        pq=os.path.join(DIR,f'{tbl}.parquet'); con.execute(f"COPY (SELECT {', '.join(cols)} FROM {tbl}{ob}) TO '{pq}' (FORMAT parquet)")
        db.cat.add_table(tbl,sch); seg=f'{tbl}_0.wdb'; wdb_encode.encode(pq,os.path.join(DIR,'wdb',seg)); db.cat.add_segment(tbl,seg)
    load('customer', order_by='c_custkey')
    con.execute("CREATE TEMP TABLE ord_pk AS SELECT o_orderkey, row_number() OVER (ORDER BY o_orderkey)-1 AS pos FROM orders")
    load('orders', order_by='o_orderkey')
    load('lineitem', extra=("(SELECT pos FROM ord_pk WHERE o_orderkey=l_orderkey) AS l_ord_ptr",['l_ord_ptr','int']))
    print("built", DIR)

li = Segment(os.path.join(DIR,'wdb','lineitem_0.wdb'))
oo = Segment(os.path.join(DIR,'wdb','orders_0.wdb'))
N = li.nrows if hasattr(li,'nrows') else len(li.values('l_ord_ptr'))

def tm(fn, n=5):
    fn(); ts=[]
    for _ in range(n):
        t=time.perf_counter(); r=fn(); ts.append(time.perf_counter()-t)
    return min(ts)*1000, r

print(f"\nlineitem rows = {N:,}\n--- gather-join [B] sub-steps (best of 5, ms) ---")

t,ptr = tm(lambda: li.values('l_ord_ptr').astype(np.int64));            print(f"  decode l_ord_ptr (6M)        : {t:8.1f}")
t,pri_str = tm(lambda: oo.values('o_orderpriority')[ptr]);             print(f"  WALK gather priority[ptr]    : {t:8.1f}   (the 'linear walk')")
t,q = tm(lambda: li.values('l_quantity'));                             print(f"  decode l_quantity (6M)       : {t:8.1f}")
t,_ = tm(lambda: pd.DataFrame({'pr':pri_str,'q':q}).groupby('pr').agg(n=('q','size'), s=('q','sum')));
print(f"  group-by pandas (bytes keys) : {t:8.1f}   <-- suspected hotspot")

# faster: gather the integer CODE, group with numpy bincount (no object keys, no pandas)
pri_codes_full = oo.codes('o_orderpriority')   # int code per orders row
def numpy_groupby():
    c = pri_codes_full[ptr]                      # gather codes (ints) instead of strings
    K = int(c.max())+1
    cnt = np.bincount(c, minlength=K)
    ssum = np.bincount(c, weights=q, minlength=K)
    return cnt, ssum
t,_ = tm(numpy_groupby);                                               print(f"  group-by numpy (int codes)   : {t:8.1f}   <-- codes instead of strings")

# full gather paths end-to-end
def full_pandas():
    p=li.values('l_ord_ptr').astype(np.int64); pr=oo.values('o_orderpriority')[p]; qq=li.values('l_quantity')
    return pd.DataFrame({'pr':pr,'q':qq}).groupby('pr').agg(n=('q','size'),s=('q','sum'))
def full_numpy():
    p=li.values('l_ord_ptr').astype(np.int64); c=oo.codes('o_orderpriority')[p]; qq=li.values('l_quantity')
    K=int(c.max())+1; return np.bincount(c,minlength=K), np.bincount(c,weights=qq,minlength=K)
t,_=tm(full_pandas); print(f"\n  FULL gather + pandas agg     : {t:8.1f}")
t,_=tm(full_numpy);  print(f"  FULL gather + numpy-codes agg: {t:8.1f}")
