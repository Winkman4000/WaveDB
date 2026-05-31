#!/usr/bin/env python3
"""
WaveDB encoder — schema-driven, works on ANY columnar file.
Reads column names + types directly from the input (parquet/csv); no hardcoded schema.

Segment format "WVDB3":
  magic "WVDB3" | u16 n_cols | u32 n_rows
  per column: u16 name_len | name | u32 V | u8 bits | u8 dtype(0=int,1=bytes) | u8 mode
    mode 0 (plain):       dict = V * (u32 len | bytes)
    mode 1 (front-coded): u16 R | u32 n_restart | restart_offsets(u32*) | u32 fclen | u32 zlen | zstd(front-coded dict)
  then: packed codes (n_rows * bits, MSB-first)

High-cardinality string columns (V > threshold) are front-coded + zstd: store each
distinct value as (shared-prefix-len, suffix) against the previous sorted value, with
restart points every R for random access. Lossless; codes already index sorted order.
"""
import duckdb, numpy as np, zstandard as zstd, struct, time, sys

FC_THRESHOLD = 50000   # front-code string columns with more distinct values than this
R = 128                # restart point interval (random-access granularity)
ZSTD_LEVEL = 9         # build-time/size knob; 9 is the sweet spot (19 costs ~15x time for ~10% size)

def encode(input_path, out_path, columns=None):
    con = duckdb.connect(); con.execute("PRAGMA threads=8")
    src = f"'{input_path}'"
    schema = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
    cols = columns if columns else [r[0] for r in schema]
    N = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    zc = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    t0 = time.time()
    out = bytearray(b'WVDB3'); out += struct.pack('<H', len(cols)); out += struct.pack('<I', N)
    sizes = {}
    for nm in cols:
        col = con.execute(f'SELECT "{nm}" FROM {src}').fetchnumpy()[nm]
        if col.dtype.kind in 'iuf':
            uniq, codes = np.unique(col, return_inverse=True)
            valb = [str(int(v)).encode() if col.dtype.kind in 'iu' else repr(float(v)).encode() for v in uniq]
            dtype = 0
        else:
            asb = np.array([x if isinstance(x,(bytes,bytearray)) else (b'' if x is None else str(x).encode('utf-8','surrogatepass')) for x in col], dtype=object)
            uniq, codes = np.unique(asb, return_inverse=True); valb = [bytes(u) for u in uniq]; dtype = 1
        V = len(uniq); bits = max(1, int(np.ceil(np.log2(max(V,2))))); codes = codes.astype(np.uint64)
        mode = 1 if (dtype == 1 and V > FC_THRESHOLD) else 0
        start = len(out)
        hb = nm.encode()
        out += struct.pack('<H', len(hb)) + hb + struct.pack('<I', V) + struct.pack('<B', bits) + struct.pack('<B', dtype) + struct.pack('<B', mode)
        if mode == 0:
            for u in valb: out += struct.pack('<I', len(u)) + u
        else:
            fc = bytearray(); restarts = []; prev = b''
            for i, s in enumerate(valb):
                if i % R == 0: prev = b''; restarts.append(len(fc))
                cp = 0; m = min(len(prev), len(s))
                while cp < m and prev[cp] == s[cp]: cp += 1
                suf = s[cp:]; fc += struct.pack('<HH', cp, len(suf)) + suf; prev = s
            z = zc.compress(bytes(fc))
            out += struct.pack('<H', R) + struct.pack('<I', len(restarts)) + np.array(restarts, dtype=np.uint32).tobytes()
            out += struct.pack('<I', len(fc)) + struct.pack('<I', len(z)) + z
        bitsarr = ((codes[:,None] >> np.arange(bits-1,-1,-1,dtype=np.uint64)) & 1).astype(np.uint8).reshape(-1)
        out += np.packbits(bitsarr).tobytes()
        sizes[nm] = (len(out)-start, V, bits, dtype, mode)
    open(out_path,'wb').write(out)
    return dict(n_rows=N, n_cols=len(cols), bytes=len(out), seconds=time.time()-t0, sizes=sizes)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: wdb_encode.py <input.parquet|csv> <out.wdb> [col1,col2,...]"); sys.exit(1)
    cols = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    r = encode(sys.argv[1], sys.argv[2], cols)
    fc = sum(1 for v in r['sizes'].values() if v[4]==1)
    print(f"Encoded {r['n_cols']} cols x {r['n_rows']:,} rows -> {r['bytes']/1e6:.1f} MB in {r['seconds']:.0f}s ({fc} front-coded)")
