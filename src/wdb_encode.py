#!/usr/bin/env python3
"""
WaveDB encoder — schema-driven, works on ANY columnar file.
Reads column names + types directly from the input (parquet/csv); no hardcoded schema.

Segment format "WVDB2":
  magic "WVDB2" | u16 n_cols | u32 n_rows
  per column: u16 name_len | name | u32 V | u8 bits | u8 dtype(0=int,1=bytes) | dict(V * (u32 len|bytes)) | packed codes
Lossless: every column is dictionary-encoded; reconstruction is byte-exact.
"""
import duckdb, numpy as np, struct, time, sys, os

def encode(input_path, out_path, columns=None):
    con = duckdb.connect(); con.execute("PRAGMA threads=8")
    src = f"'{input_path}'"
    schema = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
    allcols = [r[0] for r in schema]
    cols = columns if columns else allcols
    N = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    t0 = time.time()
    out = bytearray(b'WVDB2'); out += struct.pack('<H', len(cols)); out += struct.pack('<I', N)
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
        V = len(uniq); bits = max(1, int(np.ceil(np.log2(max(V,2)))))
        start = len(out)
        hb = nm.encode()
        out += struct.pack('<H', len(hb)) + hb + struct.pack('<I', V) + struct.pack('<B', bits) + struct.pack('<B', dtype)
        for u in valb: out += struct.pack('<I', len(u)) + u
        codes = codes.astype(np.uint64)
        bitsarr = ((codes[:,None] >> np.arange(bits-1,-1,-1,dtype=np.uint64)) & 1).astype(np.uint8).reshape(-1)
        out += np.packbits(bitsarr).tobytes()
        sizes[nm] = (len(out)-start, V, bits, dtype)
    open(out_path,'wb').write(out)
    return dict(n_rows=N, n_cols=len(cols), bytes=len(out), seconds=time.time()-t0, sizes=sizes)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: wdb_encode.py <input.parquet|csv> <out.wdb> [col1,col2,...]"); sys.exit(1)
    cols = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    r = encode(sys.argv[1], sys.argv[2], cols)
    print(f"Encoded {r['n_cols']} cols x {r['n_rows']:,} rows -> {r['bytes']/1e6:.1f} MB in {r['seconds']:.0f}s")
