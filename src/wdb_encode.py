#!/usr/bin/env python3
"""
WaveDB encoder — schema-driven, works on ANY columnar file.

Segment format "WVDB3":
  magic "WVDB3" | u16 n_cols | u32 n_rows
  per column: u16 name_len | name | u32 V | u8 bits | u8 dtype | u8 mode | u8 has_null
    dtype: 0=int, 1=bytes, 2=float(8-byte IEEE double)
    mode:  0=plain dict, 1=front-coded dict (dtype 1 only)
    has_null: 1 if column has NULLs. NULL is the reserved highest code (V-1);
              the dict stores V-has_null real values (codes 0..V-1-has_null).
    plain dict (mode 0):       (V-has_null) * (u32 len | bytes)   [float: bytes = 8-byte double]
    front-coded (mode 1):      u16 R | u32 n_restart | restart_offsets(u32*) | u32 fclen | u32 zlen | zstd(dict)
    delta-int  (mode 2):       u32 zlen | zstd(int64 deltas of sorted dict)   [dtype 0/3, non-null, high-card]
  then: packed codes (n_rows * bits, MSB-first)
"""
import numpy as np, numpy.ma as ma, pandas as pd, zstandard as zstd, struct, time, sys
import wdb_read

_DT_UNITS = ['us','ns','ms','s','D','h','m','M','Y','W']   # code = index; aux byte stores it
def _unit_code(u): return _DT_UNITS.index(u) if u in _DT_UNITS else 0

FC_THRESHOLD = 50000
NUM_THRESHOLD = 50000   # delta-code numeric dictionaries above this cardinality (mode 2)
R = 128
ZSTD_LEVEL = 9

def _encode_column(col):
    """Return (dtype, has_null, V, uniq_value_bytes_list, codes:int64[N], mode_is_string)."""
    if isinstance(col, ma.MaskedArray):
        null_mask = ma.getmaskarray(col); data = np.asarray(col.data)
    else:
        null_mask = None; data = np.asarray(col)
    has_null = 1 if (null_mask is not None and null_mask.any()) else 0
    k = data.dtype.kind
    dtype = 0 if k in 'iu' else (2 if k == 'f' else (3 if k == 'M' else 1))
    N = len(data)
    codes = np.empty(N, dtype=np.int64); aux = 0
    if dtype == 3:
        aux = _unit_code(np.datetime_data(data.dtype)[0])   # remember the time unit
        iv = data.view('int64')                              # time IS an int64 count
        if has_null:
            nn = iv[~null_mask]; uniq, inv = np.unique(nn, return_inverse=True)
            codes[~null_mask] = inv; codes[null_mask] = len(uniq)
        else:
            uniq, inv = np.unique(iv, return_inverse=True); codes[:] = inv
        valb = [struct.pack('<q', int(v)) for v in uniq]
    elif dtype in (0, 2):
        if has_null:
            nn = data[~null_mask]
            uniq, inv = np.unique(nn, return_inverse=True)
            codes[~null_mask] = inv; codes[null_mask] = len(uniq)
        else:
            uniq, inv = np.unique(data, return_inverse=True); codes[:] = inv
        if dtype == 0: valb = [str(int(v)).encode() for v in uniq]
        else:          valb = [struct.pack('<d', float(v)) for v in uniq]
    else:
        def to_b(x):
            if isinstance(x,(bytes,bytearray)): return bytes(x)
            if isinstance(x,str): return x.encode('utf-8','surrogatepass')
            return str(x).encode('utf-8','surrogatepass')  # datetime64, etc.
        # factorize (hash-based, sorted) is 30-50x faster than per-value to_b + np.unique:
        # it encodes only the unique values to bytes, not every row.
        if has_null:
            nn_idx = np.nonzero(~null_mask)[0]
            inv, uniq = pd.factorize(pd.Series(data[nn_idx]), sort=True, use_na_sentinel=False)
            codes[nn_idx] = inv; codes[null_mask] = len(uniq)
        else:
            inv, uniq = pd.factorize(pd.Series(data), sort=True, use_na_sentinel=False)
            codes[:] = inv
        valb = [to_b(u) for u in uniq]
    V = len(valb) + has_null
    return dtype, has_null, V, valb, codes, aux, uniq

def _encode_column_blob(nm, col):
    """Encode ONE column into a self-contained byte blob (header + dict + packed codes).
    Pure/independent so it can run in a worker thread; heavy ops (factorize, unique,
    packbits, zstd) release the GIL. Concatenating blobs in column order is byte-identical
    to the serial encoder. Returns (blob_bytes, sizes_tuple)."""
    zc = zstd.ZstdCompressor(level=ZSTD_LEVEL)   # thread-local: zstd compressors aren't shareable
    dtype, has_null, V, valb, codes, aux, uniq = _encode_column(col)
    bits = max(1, int(np.ceil(np.log2(max(V,2)))))
    codes = codes.astype(np.uint64)
    if dtype == 1 and (V - has_null) > FC_THRESHOLD:
        mode = 1
    elif dtype in (0, 3) and has_null == 0 and (V - has_null) > NUM_THRESHOLD:
        mode = 2
    else:
        mode = 0
    out = bytearray()
    hb = nm.encode()
    out += struct.pack('<H', len(hb)) + hb + struct.pack('<I', V)
    out += struct.pack('<B', bits) + struct.pack('<B', dtype) + struct.pack('<B', mode) + struct.pack('<B', has_null) + struct.pack('<B', aux)
    if mode == 0:
        for u in valb: out += struct.pack('<I', len(u)) + u
    elif mode == 2:
        uniq_i = uniq.astype(np.int64)
        deltas = np.diff(uniq_i, prepend=np.int64(0)).astype(np.int64)
        z = zc.compress(deltas.tobytes())
        out += struct.pack('<I', len(z)) + z
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
    return bytes(out), (len(out), V, bits, dtype, mode, has_null, aux)

def encode(input_path, out_path, columns=None, workers=None, reader='auto'):
    import os, concurrent.futures as cf
    t0 = time.time()
    # read columns via the reader module (arrow for parquet, duckdb otherwise); encode in parallel
    coldata, N, cols = wdb_read.read_columns(input_path, columns, reader=reader)
    if workers is None:
        workers = min(len(cols), (os.cpu_count() or 4))
    blobs = {}; sizes = {}
    if workers > 1 and len(cols) > 1:
        # per-column work is independent; heavy ops (factorize/unique/packbits/zstd) release the GIL
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_encode_column_blob, nm, coldata[nm]): nm for nm in cols}
            for fut in cf.as_completed(futs):
                nm = futs[fut]; blobs[nm], sizes[nm] = fut.result()
    else:
        for nm in cols:
            blobs[nm], sizes[nm] = _encode_column_blob(nm, coldata[nm])
    # assemble in column order -> byte-identical to the serial encoder
    out = bytearray(b'WVDB3'); out += struct.pack('<H', len(cols)); out += struct.pack('<I', N)
    for nm in cols: out += blobs[nm]
    open(out_path,'wb').write(out)
    return dict(n_rows=N, n_cols=len(cols), bytes=len(out), seconds=time.time()-t0, sizes=sizes)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: wdb_encode.py <input.parquet|csv> <out.wdb> [col1,col2,...]"); sys.exit(1)
    cols = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    r = encode(sys.argv[1], sys.argv[2], cols)
    fc=sum(1 for v in r['sizes'].values() if v[4]==1); fl=sum(1 for v in r['sizes'].values() if v[3]==2); dt=sum(1 for v in r['sizes'].values() if v[3]==3); nu=sum(1 for v in r['sizes'].values() if v[5]==1)
    print(f"Encoded {r['n_cols']} cols x {r['n_rows']:,} rows -> {r['bytes']/1e6:.1f} MB in {r['seconds']:.0f}s ({fc} front-coded, {fl} float, {dt} datetime, {nu} nullable)")
