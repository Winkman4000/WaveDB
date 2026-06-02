"""v2: corrected int case (pool holds int64, dtype-preserving scatter) + dtype-preserving
string case (no object promotion when not needed). Isolates the TRUE cost of override
resolution from the dtype-promotion artifact found in v1."""
import sys, time
sys.path.insert(0, '/home/jack/WaveDB/src')
import numpy as np, pandas as pd
import wdb_encode
from wdb_engine import Segment

N = 1_000_000
rng = np.random.default_rng(0)

def build(kind):
    if kind == 'str':
        vocab = np.array([f'value_{i:05d}' for i in range(2000)], dtype=object)
        col = vocab[rng.integers(0, 2000, N)]
    else:
        col = rng.integers(0, 5000, N).astype(np.int64)
    pd.DataFrame({'c': col}).to_parquet(f'/tmp/pb_{kind}.parquet', index=False)
    sp = f'/tmp/pb_{kind}.wdb'; wdb_encode.encode(f'/tmp/pb_{kind}.parquet', sp)
    return sp

def best(fn, k=5):
    ts=[]
    for _ in range(k):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)*1000

def pool(N, f, kind, n_pool=500):
    n=int(N*f); idx=rng.choice(N,size=n,replace=False); idx.sort()
    if kind=='int':
        vals=(np.arange(n_pool)+100000).astype(np.int64)   # NEW ints, same dtype
    else:
        vals=np.array([f'POOL_{i:05d}' for i in range(n_pool)],dtype=object)
    code=rng.integers(0,n_pool,n)
    return idx, code, vals

def base(seg): return seg.values('c')

def branchless_preserve(seg, idx, code, vals):
    out = seg.values('c').copy()        # keep native dtype (int64 or object)
    out[idx] = vals[code]               # dtype-preserving scatter
    return out

print(f"N={N:,}  min-of-5 (ms) -- dtype-preserving branchless\n")
for kind in ('str','int'):
    sp=build(kind); b=best(lambda: base(Segment(sp)))
    print(f"=== kind={kind} ===   baseline (no pool): {b:6.1f} ms")
    print(f"  {'f':>6} {'branchless':>12} {'vs_base':>9}")
    for f in (0.0,0.01,0.05,0.10,0.25,0.50,1.0):
        idx,code,vals=pool(N,f,kind)
        ms=best(lambda: branchless_preserve(Segment(sp), idx, code, vals))
        print(f"  {f*100:5.0f}% {ms:10.1f}ms {ms/b:8.2f}x")
    print()
