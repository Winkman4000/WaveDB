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
CODE_ZSTD_LEVEL = 19    # code-stream compression: clustered/skewed code arrays compress hugely
_INLINE_ENABLED = True  # mode-5 inline strings (toggleable for ablation/debug)

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

def _pack_codes(codes, bits):
    codes = np.asarray(codes, dtype=np.uint64)
    bitsarr = ((codes[:,None] >> np.arange(bits-1,-1,-1,dtype=np.uint64)) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bitsarr).tobytes()

def _try_seq(nm, col, allow_seq=True):
    """Mode-4 (affine/sequence) detection for non-null int/datetime columns. Returns a mode-4
    prep dict (carrying the WSQ1 blob) when the column is a clear sequential win, else None
    (caller falls through to the dict-based modes). Mandatory lossless self-check on the EXACT
    blob that will be stored -- mode 4 is never emitted unless it round-trips."""
    if not allow_seq:
        return None
    import wdb_seqcodec
    if isinstance(col, ma.MaskedArray):
        if ma.getmaskarray(col).any():
            return None                               # nulls break the affine progression
        data = np.asarray(col.data)
    else:
        data = np.asarray(col)
    k = data.dtype.kind
    if k in 'iu':
        dtype = 0; aux = 0; iv = data.astype(np.int64, copy=False)
    elif k == 'M':
        dtype = 3; aux = _unit_code(np.datetime_data(data.dtype)[0]); iv = data.view('int64')
    else:
        return None                                   # floats / strings: not eligible
    blob = wdb_seqcodec.encode(iv, max_exc_frac=0.2)  # fire only on clear wins (>=80% conform)
    if blob is None:
        return None
    if not np.array_equal(wdb_seqcodec.decode(blob), iv):
        return None                                   # safety: never emit a lossy mode-4
    N = len(iv); V = N; bits = max(1, int(np.ceil(np.log2(max(V, 2)))))
    return dict(nm=nm, dtype=dtype, has_null=0, V=V, bits=bits, aux=aux, mode=4, seqblob=blob)

def _prep_column(nm, col, allow_seq=True):
    """Heavy, independent per-column work (parallel-safe): dict + codes + mode choice."""
    seq = _try_seq(nm, col, allow_seq)
    if seq is not None:
        return seq
    dtype, has_null, V, valb, codes, aux, uniq = _encode_column(col)
    bits = max(1, int(np.ceil(np.log2(max(V,2)))))
    if dtype == 1 and (V - has_null) > FC_THRESHOLD:
        mode = 1
    elif dtype in (0, 3) and has_null == 0 and (V - has_null) > NUM_THRESHOLD:
        mode = 2
    else:
        mode = 0
    return dict(nm=nm, dtype=dtype, has_null=has_null, V=V, valb=valb,
                codes=codes.astype(np.uint64), aux=aux, uniq=uniq, bits=bits, mode=mode)

def _header(nm, V, bits, dtype, mode, has_null, aux):
    hb = nm.encode()
    return (struct.pack('<H', len(hb)) + hb + struct.pack('<I', V)
            + struct.pack('<B', bits) + struct.pack('<B', dtype) + struct.pack('<B', mode)
            + struct.pack('<B', has_null) + struct.pack('<B', aux))

def _dict_bytes_plain(valb):
    out = bytearray()
    for u in valb: out += struct.pack('<I', len(u)) + u
    return out

def _dict_bytes(p, zc):
    if p['mode'] == 0:
        return _dict_bytes_plain(p['valb'])
    out = bytearray()
    if p['mode'] == 2:
        uniq_i = p['uniq'].astype(np.int64)
        deltas = np.diff(uniq_i, prepend=np.int64(0)).astype(np.int64)
        z = zc.compress(deltas.tobytes())
        out += struct.pack('<I', len(z)) + z
    else:
        fc = bytearray(); restarts = []; prev = b''
        for i, sv in enumerate(p['valb']):
            if i % R == 0: prev = b''; restarts.append(len(fc))
            cp = 0; m = min(len(prev), len(sv))
            while cp < m and prev[cp] == sv[cp]: cp += 1
            suf = sv[cp:]; fc += struct.pack('<HH', cp, len(suf)) + suf; prev = sv
        z = zc.compress(bytes(fc))
        out += struct.pack('<H', R) + struct.pack('<I', len(restarts)) + np.array(restarts, dtype=np.uint32).tobytes()
        out += struct.pack('<I', len(fc)) + struct.pack('<I', len(z)) + z
    return out

def _code_section(codes, bits):
    """Per-row code array (mode 0/1/2): 1 tag byte + payload. tag 0 = raw bit-packed (current);
    tag 1 = zstd of byte-aligned codes. Picks the smaller (gated) -- clustered/skewed code
    arrays compress hugely (measured 34x on a sorted key), incompressible ones stay raw, paying
    only the 1-byte tag. Byte-aligned (not bit-packed) before zstd: lets its matching work."""
    packed = _pack_codes(codes, bits)
    width = 1 if bits <= 8 else (2 if bits <= 16 else 4)
    wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[width]
    z = zstd.ZstdCompressor(level=CODE_ZSTD_LEVEL).compress(np.asarray(codes, dtype=wdt).tobytes())
    if len(z) + 6 < len(packed):                 # tag(1)+width(1)+zlen(4) overhead
        return bytes([1, width]) + struct.pack('<I', len(z)) + z
    return bytes([0]) + packed

def _serialize_column(p, zc):
    """Normal blob (mode 0/1/2), or mode-4 affine blob (header + WSQ1 seqcodec blob)."""
    if p['mode'] == 4:
        out = bytearray()
        out += _header(p['nm'], p['V'], p['bits'], p['dtype'], 4, 0, p['aux'])
        out += p['seqblob']
        return bytes(out), (len(out), p['V'], p['bits'], p['dtype'], 4, 0, p['aux'])
    out = bytearray()
    out += _header(p['nm'], p['V'], p['bits'], p['dtype'], p['mode'], p['has_null'], p['aux'])
    out += _dict_bytes(p, zc)
    out += _code_section(p['codes'], p['bits'])
    normal = bytes(out), (len(out), p['V'], p['bits'], p['dtype'], p['mode'], p['has_null'], p['aux'])
    # mode-5 inline candidate: high-cardinality non-null string -> storing rows inline often beats
    # dict+codes (pointers are dead weight when values rarely repeat). Compute both, keep smaller.
    if _INLINE_ENABLED and p['dtype'] == 1 and p['has_null'] == 0:
        N = len(p['codes'])
        if N and (p['V'] / N) >= 0.5:
            inline = _serialize_inline(p)
            if len(inline[0]) < len(normal[0]):
                return inline
    return normal

def _serialize_inline(p):
    """Mode-5 inline string column: rows stored directly (no dict, no per-row codes). Wins when
    values rarely repeat -- the dictionary pointers become pure overhead. Reconstructs row-order
    bytes from the prepped dict (valb[codes]); payload = zstd(lengths u32) + zstd(concat bytes)."""
    valb = np.array(p['valb'] + [b''], dtype=object)[:-1]   # object array of distinct byte values
    rows = valb[np.asarray(p['codes'])]                     # row-order bytes (has_null==0 by gate)
    lengths = np.fromiter((len(x) for x in rows), dtype=np.uint32, count=len(rows))
    concat = b''.join(rows.tolist())
    zc = zstd.ZstdCompressor(level=CODE_ZSTD_LEVEL)
    zl = zc.compress(lengths.tobytes()); zv = zc.compress(concat)
    out = bytearray()
    out += _header(p['nm'], p['V'], p['bits'], 1, 5, 0, p['aux'])
    out += struct.pack('<I', len(zl)) + zl
    out += struct.pack('<I', len(zv)) + zv
    return bytes(out), (len(out), p['V'], p['bits'], 1, 5, 0, p['aux'])

def _serialize_fd(p, det_idx, det_codes):
    """Mode-3 blob: dependent column Y stored as y_by_xcode (Vx entries of Y-codes)
    referencing column det_idx. No per-row codes. Y dict stored plain. Lossless iff X->Y
    is an exact FD (the compactor only passes verified FDs)."""
    import wdb_fdcodec
    det_codes = np.asarray(det_codes)
    Vx = int(det_codes.max()) + 1 if det_codes.size else 0
    ymap = wdb_fdcodec.fd_encode(det_codes, p['codes'])   # Vx array of Y-codes
    out = bytearray()
    out += _header(p['nm'], p['V'], p['bits'], p['dtype'], 3, p['has_null'], p['aux'])
    out += struct.pack('<H', det_idx) + struct.pack('<I', Vx)
    out += _dict_bytes_plain(p['valb'])
    out += _pack_codes(ymap, p['bits'])
    return bytes(out), (len(out), p['V'], p['bits'], p['dtype'], 3, p['has_null'], p['aux'])

def _cluster_order(kc, N):
    """Stable row permutation sorting by the cluster key (nulls last) + the slice-boundary
    index (sorted unique key values -> first-row offsets) used by the executor's searchsorted."""
    if isinstance(kc, ma.MaskedArray):
        mask = ma.getmaskarray(kc); base = np.asarray(kc.data)
    else:
        mask = None; base = np.asarray(kc)
    k = base.dtype.kind
    str_uniq = None
    if k == 'M':
        sortkey = base.view('int64'); aux = _unit_code(np.datetime_data(base.dtype)[0]); dt = 3
    elif k in 'iu':
        sortkey = base.astype(np.int64, copy=False); aux = 0; dt = 0
    elif k == 'f':
        sortkey = base.astype(np.float64, copy=False); aux = 0; dt = 2
    elif k in 'SUO':
        # string/bytes key: factorize to value-sorted integer codes (fast int sort), and keep the
        # sorted unique strings so the grouped/range reader can emit the group value directly. dt=1.
        import pandas as pd
        codes, uniq = pd.factorize(base, sort=True)
        sortkey = codes.astype(np.int64); aux = 0; dt = 1; str_uniq = np.asarray(uniq)
    else:
        raise TypeError(f"cluster key must be int/float/datetime/string, got {base.dtype}")
    if mask is not None and mask.any():
        order = np.lexsort((sortkey, mask)); nn = int((~mask).sum())
    else:
        order = np.argsort(sortkey, kind='stable'); nn = N
    ks = sortkey[np.asarray(order)][:nn]
    vals, idx = np.unique(ks, return_index=True)
    offsets = np.append(idx.astype(np.int64), np.int64(nn))
    if str_uniq is not None:
        vals = str_uniq[vals]                       # map present value-sorted codes -> their strings
    return np.asarray(order), dict(dtype=dt, aux=aux, n=int(N), nn=int(nn),
                                   values=vals, offsets=offsets)


def encode(input_path, out_path, columns=None, workers=None, reader='auto', fd_specs=None, cluster_by=None, cubes=None):
    """fd_specs: optional {dependent_col: determinant_col} — store the dependent column as
    a mode-3 FD-reference into the determinant (lossless iff the FD is exact; callers pass
    only verified FDs). Determinant must be a normal (non-FD) column in the same segment."""
    import os, concurrent.futures as cf
    fd_specs = fd_specs or {}
    t0 = time.time()
    coldata, N, cols = wdb_read.read_columns(input_path, columns, reader=reader)
    cluster_meta = None
    if cluster_by is not None:
        if cluster_by not in coldata:
            raise KeyError(f"cluster_by {cluster_by!r} not among columns {list(coldata)}")
        _order, cluster_meta = _cluster_order(coldata[cluster_by], N)
        for _nm in cols:
            coldata[_nm] = coldata[_nm][_order]
        cluster_meta['key'] = cluster_by
    if workers is None:
        workers = min(len(cols), (os.cpu_count() or 4))
    blobs = {}; sizes = {}
    if not fd_specs:
        # fast path (no FDs): fused prep+serialize in one parallel pass — byte-identical to
        # the original encoder, no two-phase overhead.
        def _blob(nm):
            return nm, _serialize_column(_prep_column(nm, coldata[nm]),
                                         zstd.ZstdCompressor(level=ZSTD_LEVEL))
        if workers > 1 and len(cols) > 1:
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                for nm, res in ex.map(_blob, cols):
                    blobs[nm], sizes[nm] = res
        else:
            for nm in cols:
                _, res = _blob(nm); blobs[nm], sizes[nm] = res
    else:
        # FD path: prep all columns first (dependents need their determinant's codes),
        # then serialize normal columns in parallel and mode-3 dependents serially.
        preps = {}
        fd_involved = set(fd_specs) | set(fd_specs.values())  # only these must avoid mode 4
        if workers > 1 and len(cols) > 1:
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_prep_column, nm, coldata[nm], nm not in fd_involved): nm for nm in cols}
                for fut in cf.as_completed(futs):
                    nm = futs[fut]; preps[nm] = fut.result()
        else:
            for nm in cols: preps[nm] = _prep_column(nm, coldata[nm], nm not in fd_involved)
        col_idx = {nm: i for i, nm in enumerate(cols)}
        normal = [nm for nm in cols if nm not in fd_specs]
        def _ser_normal(nm):
            return nm, _serialize_column(preps[nm], zstd.ZstdCompressor(level=ZSTD_LEVEL))
        if workers > 1 and len(normal) > 1:
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                for nm, res in ex.map(_ser_normal, normal):
                    blobs[nm], sizes[nm] = res
        else:
            for nm in normal:
                _, res = _ser_normal(nm); blobs[nm], sizes[nm] = res
        for nm in fd_specs:
            det = fd_specs[nm]
            blobs[nm], sizes[nm] = _serialize_fd(preps[nm], col_idx[det], preps[det]['codes'])
    # assemble in column order
    out = bytearray(b'WVDB4'); out += struct.pack('<H', len(cols)); out += struct.pack('<I', N)
    for nm in cols: out += blobs[nm]
    open(out_path,'wb').write(out)
    try:                                          # advisory: write per-column stats sidecar (planner reads it)
        from wdb_engine import Segment
        import wdb_profile
        wdb_profile.profile_and_save(Segment(out_path), out_path)
    except Exception:
        pass
    if cluster_meta is not None:
        import pickle
        with open(out_path + '.cluster', 'wb') as _cf:
            pickle.dump(cluster_meta, _cf, protocol=4)
    if cubes:                                     # materialise GROUP BY cubes (cap in wdb_cube declines
        try:                                      # any grouping above CUBE_MAX_CELLS)
            from wdb_engine import Segment
            import wdb_cube
            _seg = Segment(out_path)
            if cubes == 'auto':                   # exhaustive: every column-subset whose card product fits
                cards = wdb_cube.segment_cardinalities(_seg)
                specs = wdb_cube.enumerate_cube_specs(cards)
            else:
                specs = cubes
            wdb_cube.build_and_write(_seg, specs, workers=workers)
        except Exception:
            pass
    return dict(n_rows=N, n_cols=len(cols), bytes=len(out), seconds=time.time()-t0,
                sizes=sizes, cluster=cluster_by)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: wdb_encode.py <input.parquet|csv> <out.wdb> [col1,col2,...]"); sys.exit(1)
    cols = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    r = encode(sys.argv[1], sys.argv[2], cols)
    fc=sum(1 for v in r['sizes'].values() if v[4]==1); fl=sum(1 for v in r['sizes'].values() if v[3]==2); dt=sum(1 for v in r['sizes'].values() if v[3]==3); nu=sum(1 for v in r['sizes'].values() if v[5]==1)
    print(f"Encoded {r['n_cols']} cols x {r['n_rows']:,} rows -> {r['bytes']/1e6:.1f} MB in {r['seconds']:.0f}s ({fc} front-coded, {fl} float, {dt} datetime, {nu} nullable)")
