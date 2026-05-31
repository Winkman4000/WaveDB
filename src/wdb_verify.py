#!/usr/bin/env python3
"""Verify a WVDB2 segment is byte-exact lossless vs its source file. Generic."""
import duckdb, numpy as np, sys
sys.path.insert(0, '/home/jack/WaveDB/src')
from wdb_engine import Segment

def verify(segment_path, source_path):
    seg = Segment(segment_path); con = duckdb.connect(); con.execute("PRAGMA threads=8")
    ok = bad = 0; bad_cols = []
    for nm in seg.order:
        orig = con.execute(f'SELECT "{nm}" FROM \'{source_path}\'').fetchnumpy()[nm]
        c = seg.cols[nm]
        if c['dt'] == 0:
            iv = np.array([int(v) for v in c['vals']], dtype=np.int64)
            recon = iv[seg.codes(nm)]; match = np.array_equal(recon, orig.astype(np.int64))
        else:
            recon = seg.values(nm)
            ob = np.array([x if isinstance(x,(bytes,bytearray)) else (b'' if x is None else str(x).encode('utf-8','surrogatepass')) for x in orig], dtype=object)
            match = np.array_equal(recon, ob)
        if match: ok += 1
        else: bad += 1; bad_cols.append(nm)
    return ok, bad, bad_cols

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: wdb_verify.py <segment.wdb> <source.parquet>"); sys.exit(1)
    ok, bad, bad_cols = verify(sys.argv[1], sys.argv[2])
    print(f"LOSSLESS: {ok}/{ok+bad} columns byte-perfect" + ("" if bad==0 else f"  MISMATCH: {bad_cols}"))
    sys.exit(0 if bad==0 else 1)
