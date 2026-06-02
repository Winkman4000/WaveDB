"""Measure how read latency degrades as a fraction of rows are overridden into a pool,
under different resolution strategies. Answers: (1) where does the pool grow 'too big',
(2) how much does branchless vectorized resolution beat naive per-row branching.
Pure measurement, not a test. Prints a table; speculation-free."""
import sys, os, time
sys.path.insert(0, '/home/jack/WaveDB/src')
import numpy as np, pandas as pd
import wdb_encode
from wdb_engine import Segment

N = 1_000_000
rng = np.random.default_rng(0)

def build_segment(kind):
    if kind == 'str':
        vocab = np.array([f'value_{i:05d}' for i in range(2000)], dtype=object)
        col = vocab[rng.integers(0, 2000, N)]
    else:
        col = rng.integers(0, 5000, N).astype(np.int64)
    df = pd.DataFrame({'c': col})
    pq = f'/tmp/pool_bench_{kind}.parquet'; df.to_parquet(pq, index=False)
    sp = f'/tmp/pool_bench_{kind}.wdb'; wdb_encode.encode(pq, sp)
    return sp

def best(fn, k=5):
    ts = []
    for _ in range(k):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts) * 1000

def make_pool(N, f, n_pool_vals=500):
    n_over = int(N * f)
    over_idx = rng.choice(N, size=n_over, replace=False); over_idx.sort()
    pool_vals = np.array([f'POOL_NEW_{i:05d}' for i in range(n_pool_vals)], dtype=object)
    over_pool_code = rng.integers(0, n_pool_vals, n_over)
    mask = np.zeros(N, dtype=bool); mask[over_idx] = True
    return mask, over_idx, over_pool_code, pool_vals

def baseline(seg):
    return seg.values('c')

def branchless(seg, over_idx, over_pool_code, pool_vals):
    out = np.asarray(seg.values('c'), dtype=object)
    out[over_idx] = pool_vals[over_pool_code]
    return out

def branchy(seg, mask, over_idx_pos, over_pool_code, pool_vals):
    codes = seg.codes('c'); dv = seg._typed_dict('c')
    over_lookup = dict(zip(over_idx_pos.tolist(), over_pool_code.tolist()))
    out = [None] * len(codes)
    for i in range(len(codes)):
        if mask[i]: out[i] = pool_vals[over_lookup[i]]
        else: out[i] = dv[codes[i]]
    return out

print(f"N={N:,}  min-of-5 (ms)\n")
for kind in ('str', 'int'):
    sp = build_segment(kind)
    base_ms = best(lambda: baseline(Segment(sp)))
    print(f"=== kind={kind} ===")
    print(f"  baseline (no pool): {base_ms:7.1f} ms")
    print(f"  {'f':>6} {'branchless':>12} {'vs_base':>9} {'branchy':>12} {'speedup':>9}")
    for f in (0.0, 0.01, 0.05, 0.10, 0.25, 0.50):
        mask, over_idx, over_pool_code, pool_vals = make_pool(N, f)
        bl = best(lambda: branchless(Segment(sp), over_idx, over_pool_code, pool_vals))
        if f <= 0.05:
            by = best(lambda: branchy(Segment(sp), mask, over_idx, over_pool_code, pool_vals), k=1)
            sp_ratio = f"{by/bl:7.1f}x"; by_str = f"{by:10.1f}ms"
        else:
            by_str = "         -"; sp_ratio = "        -"
        print(f"  {f*100:5.0f}% {bl:10.1f}ms {bl/base_ms:8.2f}x {by_str} {sp_ratio}")
    print()
