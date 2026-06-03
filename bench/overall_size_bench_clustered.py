#!/usr/bin/env python3
"""Big-picture storage on a REALISTIC CLUSTERED table (rows sorted by user; each user's region
/device cluster into runs; sequential id/ts; skewed amount; random high-card session). Ablates
the two new features so each one's contribution is explicit:
  base   = mode-4 OFF, code-stream OFF   (the pre-both baseline)
  +seq   = mode-4 ON,  code-stream OFF
  +cs    = mode-4 OFF, code-stream ON
  both   = mode-4 ON,  code-stream ON    (current main)
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
user=np.minimum(rng.zipf(1.15,N),U-1).astype(np.int64); user.sort()        # CLUSTER by user
REG=np.array(['us-east','us-west','eu-west','eu-cent','ap-south','ap-ne','sa-east','af-s'])
DEV=np.array(['ios','android','web','desktop','tv'])
base_ts=np.datetime64('2024-01-01T00:00:00')
hx=np.array([f"{v:08x}" for v in rng.integers(0,2**32,N,dtype=np.uint64)])
prices=np.array([9.99,19.99,4.99,49.99,1.0,2.5,14.99,99.0,0.99,29.99])
df=pd.DataFrame({
    'id':       1_000_000+np.arange(N,dtype=np.int64),                     # sequential -> mode 4
    'ts':       base_ts+np.arange(N,dtype='timedelta64[s]'),               # sequential -> mode 4
    'user_id':  user,                                                      # clustered int (sort key)
    'region':   REG[home_region[user]],                                    # clustered string -> code-stream
    'device':   DEV[home_device[user]],                                    # clustered string -> code-stream
    'status':   rng.choice(['ok','retry','fail','queued','cancel','timeout'],N,p=[.7,.1,.05,.08,.04,.03]),
    'amount':   np.round(rng.choice(prices,N,p=np.array([5,4,4,2,3,3,2,1,5,2])/31.0),2),
    'quantity': rng.integers(1,12,N).astype(np.int64),
    'session':  np.char.add(np.char.add(hx,'-'),np.array([f"{v:04x}" for v in rng.integers(0,2**16,N)])),
})
say(f"Realistic clustered table: {N:,} rows x {df.shape[1]} cols (built {time.time()-t0:.1f}s)")
pq='/tmp/clus.parquet'; df.to_parquet(pq,index=False,compression='zstd'); pq_sz=os.path.getsize(pq)

_seq=wdb_encode._try_seq
_cs=wdb_encode._code_section
def _cs_raw(codes,bits): return bytes([0])+wdb_encode._pack_codes(codes,bits)
def measure(seq_on, cs_on):
    wdb_encode._try_seq = _seq if seq_on else (lambda *a,**k: None)
    wdb_encode._code_section = _cs if cs_on else _cs_raw
    w='/tmp/clus.wdb'
    for f in (w,w+'.tmp'):
        if os.path.exists(f): os.remove(f)
    r=wdb_encode.encode(pq,w)
    wdb_encode._try_seq=_seq; wdb_encode._code_section=_cs
    return os.path.getsize(w), r['sizes']
base_sz,_   = measure(False,False)
seq_sz,_    = measure(True, False)
cs_sz,_     = measure(False,True)
both_sz,szs = measure(True, True)

ddb='/tmp/clus.duckdb'
if os.path.exists(ddb): os.remove(ddb)
con=duckdb.connect(ddb); con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{pq}')")
con.execute("CHECKPOINT"); con.close(); ddb_sz=os.path.getsize(ddb)

MB=1048576
say("="*70)
say(f"{'Store':<34}{'Size':>12}{'vs DuckDB':>14}")
say("-"*70)
say(f"{'Parquet (zstd)':<34}{pq_sz/MB:>9.2f} MB{pq_sz/ddb_sz:>12.2f}x")
say(f"{'DuckDB (native)':<34}{ddb_sz/MB:>9.2f} MB{1.0:>12.2f}x")
say(f"{'WaveDB base (no seq, no cs)':<34}{base_sz/MB:>9.2f} MB{ddb_sz/base_sz:>11.2f}x")
say(f"{'WaveDB +seq only':<34}{seq_sz/MB:>9.2f} MB{ddb_sz/seq_sz:>11.2f}x")
say(f"{'WaveDB +codestream only':<34}{cs_sz/MB:>9.2f} MB{ddb_sz/cs_sz:>11.2f}x")
say(f"{'WaveDB both (current main)':<34}{both_sz/MB:>9.2f} MB{ddb_sz/both_sz:>11.2f}x")
say("="*70)
say(f"mode-4 contribution:      {base_sz/MB:.2f} -> {seq_sz/MB:.2f} MB  ({base_sz/seq_sz:.2f}x)")
say(f"code-stream contribution: {base_sz/MB:.2f} -> {cs_sz/MB:.2f} MB  ({base_sz/cs_sz:.2f}x)")
say(f"both together:            {base_sz/MB:.2f} -> {both_sz/MB:.2f} MB  ({base_sz/both_sz:.2f}x)")
say(f"WaveDB(both) vs DuckDB:   {ddb_sz/both_sz:.2f}x smaller   vs Parquet: {pq_sz/both_sz:.2f}x")
say("")
say(f"Per-column (current main):  {'col':<9}{'mode':>5}{'cs':>4}{'bytes':>13}{'%tbl':>7}")
for c in df.columns:
    b=szs[c][0]; ce='-'
    say(f"{'':27}{c:<9}{szs[c][4]:>5}{'':>4}{b:>13,}{100*b/both_sz:>6.1f}%")
say(f"\n[done {time.time()-t0:.1f}s]"); OUT.close()
