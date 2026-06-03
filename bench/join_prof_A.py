"""Apply the code-based aggregation to query [A] (orders x customer, SCATTERED FK).
Reuses the persistent sf=1 build; pre-resolves o_custkey -> customer row pointer once (untimed, as it
would be stored), then times the gather + aggregate, pandas-on-strings vs numpy-on-codes."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, numpy as np, pandas as pd
from wdb_engine import Segment

DIR = '/tmp/jbprof_sf1.0'
cust = Segment(os.path.join(DIR,'wdb','customer_0.wdb'))
ordr = Segment(os.path.join(DIR,'wdb','orders_0.wdb'))

ckey = cust.values('c_custkey')                  # sorted ascending (loaded ORDER BY c_custkey)
okey = ordr.values('o_custkey')
ptr  = np.searchsorted(ckey, okey).astype(np.int64)   # pre-resolved pointer (one-time, untimed)
assert np.all(ckey[ptr] == okey), "pointer resolve mismatch"

seg_str   = cust.values('c_mktsegment')          # string per customer row
seg_codes = cust.codes('c_mktsegment')           # integer code per customer row (value-identity)
price     = ordr.values('o_totalprice')

# correctness vs DuckDB
con = duckdb.connect(); con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=1)")
exp = {r[0]: (r[1], round(float(r[2]),2)) for r in con.execute(
    "SELECT c.c_mktsegment, COUNT(*), SUM(o.o_totalprice) FROM orders o "
    "JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment").fetchall()}

def numpy_codes():
    c = seg_codes[ptr]
    K = int(c.max())+1
    return np.bincount(c, minlength=K), np.bincount(c, weights=price, minlength=K)
cnt, ssum = numpy_codes()
# map codes back to segment strings to verify
code_to_str = {}
for i in range(len(seg_codes)):
    code_to_str[seg_codes[i]] = seg_str[i]
    if len(code_to_str) == int(seg_codes.max())+1: break
got = {code_to_str[k].decode() if isinstance(code_to_str[k],(bytes,bytearray)) else code_to_str[k]:
       (int(cnt[k]), round(float(ssum[k]),2)) for k in range(len(cnt))}
ok = got == exp
print(f"[A] code-based correct vs DuckDB = {ok}")
if not ok: print(" got",got,"\n exp",exp)

def tm(fn,n=5):
    fn(); ts=[t for _ in range(n) for t in [(-time.perf_counter())] ]  # placeholder
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)*1000

print(f"\norders rows = {len(okey):,}  (gather is RANDOM scatter into {len(ckey):,} customers)")
print("--- [A] gather sub-steps (best of 5, ms) ---")
print(f"  WALK gather seg_str[ptr]     : {tm(lambda: seg_str[ptr]):8.1f}   (string gather)")
print(f"  WALK gather seg_codes[ptr]   : {tm(lambda: seg_codes[ptr]):8.1f}   (code gather)")
print(f"  group-by pandas (str keys)   : {tm(lambda: pd.DataFrame({'s':seg_str[ptr],'p':price}).groupby('s').agg(n=('p','size'),x=('p','sum'))):8.1f}")
print(f"  group-by numpy (int codes)   : {tm(numpy_codes):8.1f}")
def full_pandas():
    s=seg_str[ptr]; return pd.DataFrame({'s':s,'p':price}).groupby('s').agg(n=('p','size'),x=('p','sum'))
print(f"\n  FULL gather + pandas agg     : {tm(full_pandas):8.1f}")
print(f"  FULL gather + numpy-codes agg: {tm(numpy_codes):8.1f}")
print(f"\n  (DuckDB whole-query [A] was ~9.7 ms in the head-to-head)")
