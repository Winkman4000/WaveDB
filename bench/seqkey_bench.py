#!/usr/bin/env python3
"""seqkey_bench -- measure the position-derivable (formula) codec hypothesis on sequential
key columns vs (A) the real WaveDB encoder and (B) zstd-19 on raw int64. Honest exceptions:
real byte arrays, zstd-compressed. Reports bits/row (scale-free) and the detector verdict.

Hypothesis: a sequential key looks like max entropy (N distinct -> ~N*log2(N) bits naive)
but true entropy is O(log N); current modes pay the per-row code array, the formula mode
sheds it. Decline cases (shuffled, random) must show the codec correctly losing/declining."""
import sys, os, time, struct
sys.path.insert(0, '/home/jack/WaveDB/src')
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, zstandard as zstd
import wdb_encode

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
BASE = 1_000_000
rng = np.random.default_rng(0)
ZC = zstd.ZstdCompressor(level=19)

def scenarios():
    i = np.arange(N, dtype=np.int64)
    clean   = BASE + i
    strided = BASE + i * 7
    # gaps: sequential ids with ~1% deleted, stored in remaining order
    keep = rng.random(N + N//50) > 0.01
    gaps = (BASE + np.flatnonzero(keep)[:N]).astype(np.int64)
    # shuffled: sequential VALUES but rows stored in random order
    shuffled = clean.copy(); rng.shuffle(shuffled)
    random64 = rng.integers(0, 2**62, size=N, dtype=np.int64)
    return {'clean':clean, 'strided':strided, 'gaps':gaps,
            'shuffled':shuffled, 'random64':random64}

def wavedb_bytes(col):
    p = f'/tmp/_sk.parquet'; w = f'/tmp/_sk.wdb'
    pq.write_table(pa.table({'id': col}), p)
    for f in (w, w+'.tmp'):
        if os.path.exists(f): os.remove(f)
    wdb_encode.encode(p, w)
    return os.path.getsize(w)

def zstd_raw(col):
    return len(ZC.compress(col.tobytes()))

def formula_codec(col):
    """base + dominant-stride + sparse exception deltas. Reconstruct = cumsum of deltas.
    Returns (bytes, conforming_fraction, n_exceptions). Exceptions stored as real arrays
    (position int64 + correction int64), concatenated and zstd-19'd -- honest sizing."""
    d = np.diff(col)                              # deltas, len N-1
    vals, counts = np.unique(d, return_counts=True)
    stride = int(vals[np.argmax(counts)])         # dominant delta
    exc = np.flatnonzero(d != stride)             # exception positions (in delta space)
    conforming = 1.0 - len(exc) / max(len(d), 1)
    header = 24                                   # base(8)+stride(8)+N(8)
    if len(exc):
        payload = exc.astype(np.int64).tobytes() + d[exc].astype(np.int64).tobytes()
        ebytes = len(ZC.compress(payload))
    else:
        ebytes = len(ZC.compress(b''))
    return header + ebytes, conforming, len(exc)

def reconstruct_time(col):
    base = int(col[0]); d = np.diff(col)
    t0 = time.perf_counter()
    for _ in range(5):
        out = np.empty(len(col), dtype=np.int64); out[0] = base
        np.cumsum(d, out=out[1:]); out[1:] += base
    return (time.perf_counter() - t0) / 5 * 1000   # ms per full reconstruct

def bpr(b): return b * 8 / N
print(f"N = {N:,}   (naive 'max entropy' floor = {np.log2(N):.1f} bits/row for N distinct)")
print(f"{'scenario':<10} {'WaveDB':>12} {'zstd-19':>12} {'formula':>12}  "
      f"{'bpr:wdb':>8} {'zstd':>7} {'form':>7}  {'conform%':>8} {'recon_ms':>8}  verdict")
print("-"*108)
for name, col in scenarios().items():
    wb = wavedb_bytes(col)
    zb = zstd_raw(col)
    fb, conf, nexc = formula_codec(col)
    rt = reconstruct_time(col)
    best_other = min(wb, zb)
    verdict = f"FIRE {best_other/fb:.0f}x vs best" if fb < best_other else f"DECLINE ({fb/best_other:.1f}x worse)"
    print(f"{name:<10} {wb:>12,} {zb:>12,} {fb:>12,}  "
          f"{bpr(wb):>8.2f} {bpr(zb):>7.2f} {bpr(fb):>7.3f}  {conf*100:>7.2f}% {rt:>7.2f}  {verdict}")
