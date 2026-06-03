#!/usr/bin/env python3
"""Big-picture storage on a REALISTIC CLUSTERED table (rows sorted by user; each user's region/
device cluster into runs; sequential id/ts; skewed amount; high-entropy high-card session). Clean
ablation of all THREE storage features, each toggled independently so its standalone contribution
is explicit:
  mode-4 (affine/sequence)   -> wdb_encode._try_seq
  code-stream compression    -> wdb_encode._code_section
  mode-5 (inline strings)    -> wdb_encode._INLINE_ENABLED
vs DuckDB native and parquet(zstd). Report -> /tmp/clus.out."""
import sys, os, time
sys.path.insert(0, '/home/jack/WaveDB/src')
import numpy as np, pandas as pd, duckdb
import wdb_encode
OUT=open('/tmp/clus.out','w')
def say(*a):
    s=' '.join(str(x) for x in a); print(s); OUT.write(s+'\n'); OUT.flush()

N=2_000_000; U=150_000; rng=np.random.default_rng(7); t0=time.time()
home_region=rng.integers(0,8,U); home_device=rng.integers(0,5,U)
user=np.minimum(rng.zipf(1.15,N),U-1).astype(np.int64); user.sort()
REG=np.array(['us-east','us-west','eu-west','eu-cent','ap-south','ap-ne','sa-east','af-s'])
DEV=np.array(['ios','android','web','desktop','tv'])
base_ts=np.datetime64('2024-01-01T00:00:00')
hx=np.array([f"{v:08x}" for v in rng.integers(0,2**32,N,dtype=np.uint64)])
prices=np.array([9.99,19.99,4.99,49.99,1.0,2.5,14.99,99.0,0.99,29.99])
df=pd.DataFrame({
    'id':1_000_000+np.arange(N,dtype=np.int64),
    'ts':base_ts+np.arange(N,dtype='timedelta64[s]'),
    'user_id':user,
    'region':REG[home_region[user]], 'device':DEV[home_device[user]],
    'status':rng.choice(['ok','retry','fail','queued','cancel','timeout'],N,p=[.7,.1,.05,.08,.04,.03]),
    'amount':np.round(rng.choice(prices,N,p=np.array([5,4,4,2,3,3,2,1,5,2])/31.0),2),
    'quantity':rng.integers(1,12,N).astype(np.int64),
    'session':np.char.add(np.char.add(hx,'-'),np.array([f"{v:04x}" for v in rng.integers(0,2**16,N)])),
})
say(f"Realistic clustered table: {N:,} rows x {df.shape[1]} cols (built {time.time()-t0:.1f}s)")
pq='/tmp/clus.parquet'; df.to_parquet(pq,index=False,compression='zstd'); pq_sz=os.path.getsize(pq)

_seq=wdb_encode._try_seq; _cs=wdb_encode._code_section
def _cs_raw(codes,bits): return bytes([0])+wdb_encode._pack_codes(codes,bits)
def measure(seq, cs, inline):
    wdb_encode._try_seq = _seq if seq else (lambda *a,**k: None)
    wdb_encode._code_section = _cs if cs else _cs_raw
    wdb_encode._INLINE_ENABLED = inline
    w='/tmp/clus.wdb'
    for f in (w,w+'.tmp'):
        if os.path.exists(f): os.remove(f)
    r=wdb_encode.encode(pq,w)
    wdb_encode._try_seq=_seq; wdb_encode._code_section=_cs; wdb_encode._INLINE_ENABLED=True
    return os.path.getsize(w), r['sizes']
base,_   = measure(0,0,0)
m4,_     = measure(1,0,0)
cs,_     = measure(0,1,0)
m5,_     = measure(0,0,1)
both,szs = measure(1,1,1)

ddb='/tmp/clus.duckdb'
if os.path.exists(ddb): os.remove(ddb)
con=duckdb.connect(ddb); con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{pq}')")
con.execute("CHECKPOINT"); con.close(); ddb_sz=os.path.getsize(ddb)

MB=1048576
say("="*68)
say(f"{'Configuration':<32}{'Size':>11}{'vs DuckDB':>14}")
say("-"*68)
say(f"{'Parquet (zstd)':<32}{pq_sz/MB:>8.2f} MB{pq_sz/ddb_sz:>11.2f}x")
say(f"{'DuckDB (native)':<32}{ddb_sz/MB:>8.2f} MB{1.0:>11.2f}x")
say(f"{'WaveDB: none':<32}{base/MB:>8.2f} MB{ddb_sz/base:>10.2f}x")
say(f"{'WaveDB: +mode-4 only':<32}{m4/MB:>8.2f} MB{ddb_sz/m4:>10.2f}x")
say(f"{'WaveDB: +code-stream only':<32}{cs/MB:>8.2f} MB{ddb_sz/cs:>10.2f}x")
say(f"{'WaveDB: +mode-5 only':<32}{m5/MB:>8.2f} MB{ddb_sz/m5:>10.2f}x")
say(f"{'WaveDB: ALL (main)':<32}{both/MB:>8.2f} MB{ddb_sz/both:>10.2f}x")
say("="*68)
say(f"standalone contributions vs 'none' ({base/MB:.2f} MB):")
say(f"  mode-4      {base/m4:.2f}x   code-stream {base/cs:.2f}x   mode-5 {base/m5:.2f}x")
say(f"  ALL three together: {base/both:.2f}x   ({base/MB:.2f} -> {both/MB:.2f} MB)")
say(f"WaveDB(all) vs DuckDB: {ddb_sz/both:.2f}x smaller   vs Parquet: {pq_sz/both:.2f}x")
say("")
say(f"Per-column (ALL on):  {'col':<9}{'mode':>5}{'bytes':>13}{'%tbl':>7}")
for c in df.columns:
    b=szs[c][0]
    say(f"{'':22}{c:<9}{szs[c][4]:>5}{b:>13,}{100*b/both:>6.1f}%")
say(f"\n[done {time.time()-t0:.1f}s]"); OUT.close()
