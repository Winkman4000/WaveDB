#!/usr/bin/env python3
"""overall_size_bench -- big-picture storage: WaveDB (mode-4 ON vs OFF) vs DuckDB vs parquet
on a REPRESENTATIVE wide analytical table (sequential PK + monotonic ts + the usual mix of
categoricals / measures / high-card strings). Isolates what the mode-4 addition bought at the
whole-table level, not just on an isolated key column. Writes a report to /tmp/sizebench.out."""
import sys, os, time
sys.path.insert(0, '/home/jack/WaveDB/src')
import numpy as np, pandas as pd, duckdb
import wdb_encode
from wdb_engine import Segment

N = 2_000_000
rng = np.random.default_rng(7)
OUT = open('/tmp/sizebench.out', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n'); OUT.flush()

t0 = time.time()
base_ts = np.datetime64('2024-01-01T00:00:00')
hex32 = np.array([f"{v:08x}" for v in rng.integers(0, 2**32, N, dtype=np.uint64)])
df = pd.DataFrame({
    'id':       1_000_000 + np.arange(N, dtype=np.int64),                  # sequential PK -> mode 4
    'ts':       base_ts + np.arange(N, dtype='timedelta64[s]'),            # monotonic ts -> mode 4
    'user_id':  rng.integers(1, 60_000, N).astype(np.int64),              # medium-card int
    'region':   rng.choice(['us-east','us-west','eu-west','eu-cent','ap-south','ap-ne','sa-east','af-s'], N),
    'device':   rng.choice(['ios','android','web','desktop','tv'], N),     # low-card
    'status':   rng.choice(['ok','retry','fail','queued','cancel','timeout'], N),
    'amount':   np.round(rng.gamma(2.0, 25.0, N), 2),                      # float measure
    'quantity': rng.integers(1, 12, N).astype(np.int64),                  # small int
    'session':  np.char.add(np.char.add(hex32, '-'),
                            np.array([f"{v:04x}" for v in rng.integers(0, 2**16, N)])),  # high-card string
})
say(f"Dataset: {N:,} rows x {df.shape[1]} cols  (built in {time.time()-t0:.1f}s)")
say("="*64)

pq_path = '/tmp/sb.parquet'
df.to_parquet(pq_path, index=False, compression='zstd')
pq_sz = os.path.getsize(pq_path)

# raw CSV reference
csv_path = '/tmp/sb.csv'; df.to_csv(csv_path, index=False); csv_sz = os.path.getsize(csv_path)

def encode_measure(tag):
    w = f'/tmp/sb_{tag}.wdb'
    for f in (w, w+'.tmp'):
        if os.path.exists(f): os.remove(f)
    r = wdb_encode.encode(pq_path, w)
    return os.path.getsize(w), r['sizes']

# WaveDB mode-4 ON (current main)
on_sz, on_sizes = encode_measure('on')

# WaveDB mode-4 OFF (monkeypatch the detector to always decline) -> the pre-mode-4 baseline
_orig = wdb_encode._try_seq
wdb_encode._try_seq = lambda *a, **k: None
off_sz, off_sizes = encode_measure('off')
wdb_encode._try_seq = _orig

# DuckDB native storage (import the same parquet, checkpoint to flush)
ddb = '/tmp/sb.duckdb'
if os.path.exists(ddb): os.remove(ddb)
con = duckdb.connect(ddb)
con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{pq_path}')")
con.execute("CHECKPOINT"); con.close()
ddb_sz = os.path.getsize(ddb)

MB = 1024*1024
say(f"{'Store':<26}{'Size':>14}{'vs DuckDB':>12}")
say("-"*64)
say(f"{'Raw CSV':<26}{csv_sz/MB:>11.2f} MB{csv_sz/ddb_sz:>11.2f}x")
say(f"{'Parquet (zstd)':<26}{pq_sz/MB:>11.2f} MB{pq_sz/ddb_sz:>11.2f}x")
say(f"{'DuckDB (native)':<26}{ddb_sz/MB:>11.2f} MB{1.0:>11.2f}x")
say(f"{'WaveDB (mode-4 OFF)':<26}{off_sz/MB:>11.2f} MB{ddb_sz/off_sz:>11.2f}x smaller")
say(f"{'WaveDB (mode-4 ON)':<26}{on_sz/MB:>11.2f} MB{ddb_sz/on_sz:>11.2f}x smaller")
say("="*64)
say(f"Mode-4 shrank the whole table:  {off_sz/MB:.2f} MB -> {on_sz/MB:.2f} MB"
    f"  ({off_sz/on_sz:.2f}x, saved {(off_sz-on_sz)/MB:.2f} MB)")
say(f"WaveDB(on) vs DuckDB:           {ddb_sz/on_sz:.1f}x smaller")
say(f"WaveDB(on) vs Parquet:          {pq_sz/on_sz:.1f}x smaller")
say("")
say("Per-column (the two mode-4 columns vs their pre-mode-4 cost):")
for c in ('id', 'ts'):
    on_b = on_sizes[c][0]; off_b = off_sizes[c][0]
    say(f"  {c:<10} mode {off_sizes[c][4]}->{on_sizes[c][4]}   {off_b:>12,} B -> {on_b:>6,} B"
        f"   ({off_b/max(on_b,1):,.0f}x)")
say("")
say("Full per-column breakdown (mode-4 ON):")
say(f"  {'col':<10}{'mode':>5}{'bytes':>14}{'% of table':>12}")
for c in df.columns:
    b = on_sizes[c][0]
    say(f"  {c:<10}{on_sizes[c][4]:>5}{b:>14,}{100*b/on_sz:>11.1f}%")

for f in (csv_path,):  # keep parquet/wdb/duckdb for any follow-up; drop the big csv
    if os.path.exists(f): os.remove(f)
say(f"\n[done in {time.time()-t0:.1f}s]")
OUT.close()
