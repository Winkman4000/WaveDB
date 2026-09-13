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
import os
import wdb_read

_DT_UNITS = ['us','ns','ms','s','D','h','m','M','Y','W']   # code = index; aux byte stores it
def _unit_code(u): return _DT_UNITS.index(u) if u in _DT_UNITS else 0

FC_THRESHOLD = 50000
NUM_THRESHOLD = 50000   # delta-code numeric dictionaries above this cardinality (mode 2)
R = 128
ZSTD_LEVEL = 9
BLOCK_ROWS = int(os.environ.get('WDB_BLOCK_ROWS', 524288))   # rows per enc=3 frame
# code-stream compression level. Measured on 10.7M-row ClickBench columns: level 19 -> 9 is
# 7-13x faster (URL 14.5s -> 1.9s, SearchPhrase 18.1s -> 1.4s) for 6-8% more bytes (EventTime
# 22%). Ingest speed wins by default; WDB_CODE_ZSTD=19 for archival encodes.
CODE_ZSTD_LEVEL = int(os.environ.get('WDB_CODE_ZSTD', '9'))
INLINE_ZSTD_LEVEL = 19  # mode-5 inline value blobs: the mode decision is a size race at the archival level
CHUNK_DICT = bool(int(os.environ.get('WDB_CHUNK_DICT', '1')))   # block-segment front-coded dicts (default on; WDB_CHUNK_DICT=0 to opt out)
CHUNK_DICT_VALS = 16384                                          # values per independent zstd frame (mult of R)
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
    if k == 'b' or (k == 'O' and N and all(isinstance(v, (bool, np.bool_)) for v in data[:64] if v is not None)):
        aux = 9                                              # THE BOOL MARKER: dt1 strings 'False'/'True' decode to Python bools
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
    """bit-pack in CHUNKS: the N x bits intermediate was 16 GB for 100M x 20-bit codes
    (measured: the true peak of a string column's encode, not the strings)"""
    codes = np.asarray(codes, dtype=np.uint64)
    n = codes.size
    if n == 0: return b''
    step = max(1, (1 << 23) // max(1, bits) * 8)           # ~8M bit-cells per chunk
    step -= step % 8                                        # chunk rows x bits must be a multiple of 8 bits
    if step <= 0: step = 8
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    out = bytearray()
    for lo in range(0, n, step):
        part = codes[lo:lo + step]
        bitsarr = ((part[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
        out += np.packbits(bitsarr).tobytes()
    return bytes(out)

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
        # THE CLOCK LAW: a clock that REPEATS (EventTime: ~70 rows per second) is a key -- a
        # DICTIONARY whose staircase is the sequence, which the window and range doors read
        # (enc=2); as a mode-4 sequence it lost them (w-runmin 1.9s -> 23s). A clock that
        # never repeats (a reading per row, an affine series) is a genuine sequence.
        _iv9 = data.view('int64'); _sm9 = _iv9[:: max(1, _iv9.size // 2_000_000)]
        if _sm9.size >= 1024 and np.unique(_sm9).size < 0.9 * _sm9.size:
            return None
        dtype = 3; aux = _unit_code(np.datetime_data(data.dtype)[0]); iv = _iv9
    else:
        return None                                   # floats / strings: not eligible
    # cardinality guard: a narrow column (flags, enums, small ids) is a DICTIONARY
    # column no matter how affine its runs look -- dict codes give free per-row
    # identity (no cumsum reconstruction ever) and pack tighter. Found 2026-07 when
    # six flag/enum columns misfired into mode 4 and every read paid delta-decode.
    import os as _os
    if iv.size and _os.environ.get('WDB_SEQ_NARROW_OK') != '1':
        lo = int(iv.min()); hi = int(iv.max())
        if hi - lo < (1 << 16):
            tv = int(np.count_nonzero(np.bincount((iv - lo).astype(np.int64))))
            if tv <= 65536:
                return None                           # narrow: the dict modes win
        else:
            # THE CARDINALITY LAW: judge by DISTINCT COUNT, not span -- CounterID (6,506 values
            # spanning millions) became a 100M-value sequence under the span-only guard (2026-09-14)
            samp = iv[:: max(1, iv.size // 4_000_000)]
            if np.unique(samp).size <= 65536 and samp.size >= 65536:
                return None
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
    if mode == 1 and CHUNK_DICT:
        aux |= 0x40                                     # bit6 = write the dict as chunked zstd frames
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

def _front_code_py(valb):
    """the reference front-coder (Python): restart every R values; per value <HH cp suflen> + suffix"""
    fc = bytearray(); restarts = []; prev = b''
    for i, sv in enumerate(valb):
        if i % R == 0: prev = b''; restarts.append(len(fc))
        cp = 0; m = min(len(prev), len(sv))
        while cp < m and prev[cp] == sv[cp]: cp += 1
        suf = sv[cp:]; fc += struct.pack('<HH', cp, len(suf)) + suf; prev = sv
    return bytes(fc), np.array(restarts, dtype=np.uint32)


def _front_code(valb):
    """THE FRONT-CODER IN NUMBA: the dictionary as one byte buffer + offsets; a compiled loop
    computes each value's common prefix with its predecessor and emits <HH cp suflen>+suffix
    with a restart every R values -- byte-identical to _front_code_py (15s of Python per 2.7M
    URLs -> ~0.2s). Falls back to Python without numba."""
    try:
        import numba
    except Exception:
        return _front_code_py(valb)
    n = len(valb)
    if n == 0:
        return b'', np.zeros(0, dtype=np.uint32)
    lens = np.fromiter((len(v) for v in valb), dtype=np.int64, count=n)
    offs = np.zeros(n + 1, dtype=np.int64); np.cumsum(lens, out=offs[1:])
    buf = np.frombuffer(b''.join(valb), dtype=np.uint8)
    out, rst, total = _front_code_jit(buf, offs, n, R)
    return out[:total].tobytes(), rst


def _front_code_jit_py(buf, offs, n, R):
    # sizing: every value costs 4 header bytes + at most its full length
    out = np.empty(int(offs[-1]) + 4 * n, dtype=np.uint8)
    rst = np.empty((n + R - 1) // R, dtype=np.uint32)
    pos = 0; pstart = 0; plen = 0; nr = 0
    for i in range(n):
        s0 = offs[i]; l0 = offs[i + 1] - s0
        if i % R == 0:
            plen = 0; rst[nr] = pos; nr += 1
        m = plen if plen < l0 else l0
        cp = 0
        while cp < m and buf[pstart + cp] == buf[s0 + cp]:
            cp += 1
        suf = l0 - cp
        out[pos] = cp & 0xFF; out[pos + 1] = (cp >> 8) & 0xFF
        out[pos + 2] = suf & 0xFF; out[pos + 3] = (suf >> 8) & 0xFF
        pos += 4
        for k in range(suf):
            out[pos + k] = buf[s0 + cp + k]
        pos += suf
        pstart = s0; plen = l0
    return out, rst, pos


try:
    import numba as _nb9
    _front_code_jit = _nb9.njit(cache=True)(_front_code_jit_py)
except Exception:
    _front_code_jit = _front_code_jit_py


def _dict_bytes(p, zc):
    if p['mode'] == 0:
        return _dict_bytes_plain(p['valb'])
    out = bytearray()
    if p['mode'] == 2:
        uniq_i = p['uniq'].astype(np.int64)
        I2CH = int(os.environ.get('WDB_I2CHUNK', str(1 << 19)))      # values per chunk (4MB raw)
        I2MIN = int(os.environ.get('WDB_I2CHUNK_MIN', str(1 << 20)))  # chunk only big dicts
        if uniq_i.size > I2MIN:
            # CHUNKED SPINE (the hits_6 dress): each chunk's deltas prepend 0, so every
            # chunk cumsums to absolutes independently -- point reads pop one ~3MB chunk
            # instead of inflating a 102MB monolith. Sentinel 0xFFFFFFFF versions the header.
            zs = []
            for a in range(0, uniq_i.size, I2CH):
                ck = uniq_i[a:a + I2CH]
                zs.append(zc.compress(np.diff(ck, prepend=np.int64(0)).astype(np.int64).tobytes()))
            out += struct.pack('<I', 0xFFFFFFFF) + struct.pack('<II', I2CH, len(zs))
            for z in zs:
                out += struct.pack('<I', len(z))
            for z in zs:
                out += z
        else:
            deltas = np.diff(uniq_i, prepend=np.int64(0)).astype(np.int64)
            z = zc.compress(deltas.tobytes())
            out += struct.pack('<I', len(z)) + z
    else:
        fc, rst = _front_code(p['valb'])
        nb = len(rst)
        if p['aux'] & 0x40:                       # chunked: one independent zstd frame per CHUNK_DICT_VALS
            V = len(p['valb']); CH = CHUNK_DICT_VALS; BPC = CH // R
            n_chunks = (V + CH - 1) // CH
            ustart = []; czl = []; frames = []
            for j in range(n_chunks):
                b0 = int(rst[j*BPC])
                b1 = int(rst[(j+1)*BPC]) if (j+1)*BPC < nb else len(fc)
                fr = zc.compress(fc[b0:b1]); frames.append(fr); ustart.append(b0); czl.append(len(fr))
            out += struct.pack('<H', R) + struct.pack('<I', CH) + struct.pack('<I', n_chunks)
            out += struct.pack('<I', nb) + rst.tobytes() + struct.pack('<I', len(fc))
            out += np.array(ustart, dtype=np.uint32).tobytes()
            out += np.array(czl, dtype=np.uint32).tobytes()
            for fr in frames: out += fr
        else:
            z = zc.compress(fc)
            out += struct.pack('<H', R) + struct.pack('<I', nb) + rst.tobytes()
            out += struct.pack('<I', len(fc)) + struct.pack('<I', len(z)) + z
    return out

def _code_section(codes, bits, enc5_ok=False, nm=None, date_vals=None):
    _fovr = dict((kv.split(':')[0], int(kv.split(':')[1]))
                 for kv in os.environ.get('WDB_FRAME_OVERRIDES', '').split(',') if ':' in kv)
    BLOCK_ROWS = _fovr[nm] if (nm is not None and nm in _fovr)         else globals()['BLOCK_ROWS']             # the passport elects the frame
    """Per-row code array (mode 0/1/2): 1 tag byte + payload. tag 0 = raw bit-packed; tag 1 = zstd
    of byte-aligned codes; tag 2 = STAIRCASE (codes non-decreasing in row order, e.g. time-ordered
    ingest): store only the gap-packed rows where the code ticks +1 -- the norm is 'same as the row
    above', the steps are the exceptions. EventTime measured: 3.13 MB zstd -> 1.37 MB steps, and
    the steps serve point reads/GROUP BY with NO decode. Smallest candidate wins; incompressible
    arrays stay raw, paying only the tag byte."""
    arr = np.asarray(codes, dtype=np.int64)
    stair = None
    if arr.size:
        d = np.diff(arr)
        if int(arr[0]) == 0 and (d.size == 0 or (int(d.min()) >= 0 and int(d.max()) <= 1)):
            steps = (np.nonzero(d)[0] + 1).astype(np.int64)      # rows where the code ticks +1
            gaps = np.diff(np.concatenate(([0], steps)))
            gbits = max(1, int(gaps.max()).bit_length()) if gaps.size else 1
            pay = _pack_codes(gaps, gbits) if gaps.size else b''
            stair = bytes([2, gbits]) + struct.pack('<I', steps.size) + pay
    packed = bytes([0]) + _pack_codes(codes, bits)
    width = 1 if bits <= 8 else (2 if bits <= 16 else 4)
    wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[width]
    # THE LAYOUT LAW (Jackson): a low-V column is a bit-pack laid down as fast as possible --
    # zstd only earns its time on wide codes. Narrow codes (<= 4 bits) skip the zstd candidate
    # when the pack is already small; wider ones compress the BYTE-ALIGNED codes as before.
    # (IsRefresh, V=2: 1.2s of zstd per 2M rows to lose to a 250KB pack.)
    if bits <= 4 and len(packed) <= (64 << 20):
        # narrow codes: a CHEAP zstd (level 3, 0.3s per 100M rows) still wins the runs a
        # time-clustered table produces (skipping it cost 1.8 GB on the clustered encode)
        z = zstd.ZstdCompressor(level=3, threads=4).compress(np.asarray(codes, dtype=wdt).tobytes())
        zsec = bytes([1, width]) + struct.pack('<I', len(z)) + z
    else:
        z = zstd.ZstdCompressor(level=CODE_ZSTD_LEVEL, threads=4).compress(np.asarray(codes, dtype=wdt).tobytes())   # zstd's own threads: measured 2x on a 43 MB code stream
        zsec = bytes([1, width]) + struct.pack('<I', len(z)) + z
    # tag 8 = SPARSE-DEFAULT (Jackson's dress): store nothing for the dominant value.
    # presence bitmap + rank checkpoints + bitpacked literals. Beat zstd outright on
    # SearchPhrase (50.4 vs 52.6 MB) with zero decoders; adopted on strict size
    # dominance only -- no knobs, smaller or nothing.
    sparse = None
    cn8 = np.bincount(np.asarray(codes, dtype=np.int64)) if codes.size else np.zeros(0)
    dflt = int(cn8.argmax()) if cn8.size else 0
    if codes.size and cn8.size and cn8[dflt] * 2 > codes.size:               # majority default: the only shape it fits
        pres = (np.asarray(codes, dtype=np.int64) != dflt)
        lits = np.asarray(codes)[pres]
        pb = np.packbits(pres)
        CK = 65536
        nck = (codes.size + CK - 1) // CK
        per = np.add.reduceat(pres.astype(np.int64),
                              np.arange(0, codes.size, CK))
        ck = np.zeros(nck, dtype=np.uint64)
        if nck > 1:
            ck[1:] = np.cumsum(per[:-1]).astype(np.uint64)
        litp = _pack_codes(lits, bits) if lits.size else b''
        sparse = (bytes([8, bits]) + struct.pack('<IQQ', dflt, lits.size, codes.size)
                  + pb.tobytes() + ck.tobytes() + litp)
    # tag 9 = TIERED dress (rule eleven, Jackson's design): the dominant value
    # is NOTHING (absence bitmap), then within the typed remainder the most
    # common code is ONE BIT, tiering down; the small tail rides u8. Elected
    # for low-V columns with concentrated histograms; zero-pop serving.
    tiered = None
    if codes.size and cn8.size and cn8[dflt] * 2 > codes.size and bits <= 8 \
            and cn8.size <= 256:
        arr9 = np.asarray(codes, dtype=np.int64)
        pres9 = arr9 != dflt
        pb9 = np.packbits(pres9)
        CK = 65536
        nck9 = (codes.size + CK - 1) // CK
        per9 = np.add.reduceat(pres9.astype(np.int64), np.arange(0, codes.size, CK))
        ck9 = np.zeros(nck9, dtype=np.uint64)
        if nck9 > 1:
            ck9[1:] = np.cumsum(per9[:-1]).astype(np.uint64)
        rem = arr9[pres9]                    # typed codes, row order
        planes = b''
        tcodes = []
        for _ in range(3):                   # up to three one-bit tiers
            if rem.size < 65536:
                break
            cnr = np.bincount(rem)
            dom = int(cnr.argmax())
            if int(cnr[dom]) * 4 < rem.size:
                break                        # no concentration left: tail it
            bit9 = rem == dom
            planes += struct.pack('<IQ', dom, rem.size) + np.packbits(bit9).tobytes()
            tcodes.append(dom)
            rem = rem[~bit9]
        tiered = (bytes([9, bits]) + struct.pack('<IQQ', dflt, int(pres9.sum()), codes.size)
                  + pb9.tobytes() + ck9.tobytes()
                  + bytes([len(tcodes)]) + planes
                  + struct.pack('<Q', rem.size) + rem.astype(np.uint8).tobytes())
    # tag 10 = SEGMENTED BITPACK-PLUS (Jackson's dress): 4096-row blocks,
    # each electing bitpack or run-tokens by the profit formula -- runs
    # compress only where run_len*bits beats the token, so the whole
    # column is structurally never worse than bitpack (+1B/block).
    bplus = None
    if codes.size and bits <= 16:
        arrA = np.asarray(codes, dtype=np.int64)
        B10 = 4096
        nblkA = (arrA.size + B10 - 1) // B10
        bndA = np.flatnonzero(np.diff(arrA) != 0)
        stA = np.concatenate([[0], bndA + 1]).astype(np.int64)
        blk_firstA = np.searchsorted(stA, np.arange(0, arrA.size, B10), side='right') - 1
        blk_lastA = np.searchsorted(stA, np.minimum(
            np.arange(B10, arrA.size + B10, B10), arrA.size), side='left')
        nruns_bA = np.maximum(1, blk_lastA - blk_firstA)
        rows_bA = np.minimum(np.arange(B10, arrA.size + B10, B10), arrA.size) \
            - np.arange(0, arrA.size, B10)
        run_modeA = nruns_bA * 32 < rows_bA * bits    # u16 count + u16 value
        if run_modeA.any():                           # only dress when runs pay
            payloadA = bytearray()
            dirA = np.zeros(nblkA, np.int64)
            for bA in range(nblkA):
                loA = bA * B10
                hiA = min(arrA.size, loA + B10)
                dirA[bA] = len(payloadA) << 1
                blkA = arrA[loA:hiA]
                if run_modeA[bA]:
                    dirA[bA] |= 1
                    bnd_b = np.flatnonzero(np.diff(blkA) != 0)
                    st_b = np.concatenate([[0], bnd_b + 1])
                    en_b = np.concatenate([bnd_b + 1, [blkA.size]])
                    payloadA += struct.pack('<H', st_b.size)
                    pairs = np.empty(st_b.size * 2, np.uint16)
                    pairs[0::2] = (en_b - st_b).astype(np.uint16)
                    pairs[1::2] = blkA[st_b].astype(np.uint16)
                    payloadA += pairs.tobytes()
                else:
                    nbyA = (blkA.size * bits + 7) // 8
                    accA = np.zeros(nbyA * 8, np.uint8)
                    for bitA in range(bits):
                        accA[bitA::bits][:blkA.size] = (blkA >> (bits - 1 - bitA)) & 1
                    payloadA += np.packbits(accA[:nbyA * 8]).tobytes()
            bplus = (bytes([10, bits]) + struct.pack('<QI', codes.size, nblkA)
                     + dirA.tobytes() + bytes(payloadA))
    cands8 = [s for s in (stair, zsec, packed, sparse) if s is not None]
    best = min(cands8, key=len)
    if sparse is not None and os.environ.get('WDB_E8_FORCE'):
        best = sparse                            # rehearsal-only: exercise the readers
    if bplus is not None and os.environ.get('WDB_PLUS_FORCE'):
        best = bplus                             # rehearsal-only: enc-10's readers
    _serve = os.environ.get('WDB_SERVE_COLS', '')
    if nm is not None and _serve and nm in _serve.split(','):
        # THE SERVING-DRESS ELECTION (the clustering era's first law):
        # crumb-read columns trade zstd's entropy edge for RANDOM ACCESS --
        # bit arithmetic at any row, no frame ever decompresses to serve a
        # point read. Candidates: plain bitpack or bitpack-plus (whichever
        # is smaller); staircase columns already serve randomly and stay.
        if best is not stair:
            # THE VERTICAL DRESS (enc-12): planes instead of rows. Same
            # bytes as bitpack, resliced -- one u64 = one bit of 64 rows.
            # Scans word-parallel + Jackson's snowball; windows via the
            # 64x64 transpose tapes; scattered gathers measured FASTER
            # than horizontal at DRAM scale. Storage stays flat.
            n9 = len(codes)
            nw9 = (n9 + 63) // 64
            pad9 = (-n9) % 64
            pl9 = np.zeros(bits * nw9, np.uint64)
            arr9 = np.asarray(codes, np.int64)
            for p9 in range(bits):
                bc9 = ((arr9 >> (bits - 1 - p9)) & 1).astype(np.uint8)
                if pad9:
                    bc9 = np.concatenate([bc9, np.zeros(pad9, np.uint8)])
                pl9[p9 * nw9:(p9 + 1) * nw9] = np.frombuffer(
                    np.packbits(bc9, bitorder='little').tobytes(), np.uint64)
            best = bytes([12]) + struct.pack('<I', nw9) + pl9.tobytes()
    elif bplus is not None and len(bplus) <= 4 * len(best):
        # JACKSON'S ELECTION: real locality (the run census already proved
        # profitable blocks exist) within the 4x serving seal -- random
        # access and R-sized censuses outvote bounded disk, the tag-9
        # precedent. Columns without locality never built a bplus with
        # run-blocks cheaper than bitpack, so this only fires where the
        # formula found profit.
        _lensA = np.diff(np.concatenate([stA, [arrA.size]]))
        if float(_lensA.mean()) >= 5.0:          # mean run >= 5: locality is real
            best = bplus
    elif tiered is not None and os.environ.get('WDB_TIER_FORCE'):
        best = tiered                            # rehearsal-only: rule eleven's readers
    elif tiered is not None and cn8.size >= 10 \
            and cn8[dflt] * 10 >= codes.size * 9 \
            and (codes.size - int(cn8[dflt])) >= 65536 \
            and len(tiered) <= 4 * len(best):
        # RULE ELEVEN'S ELECTION (Jackson): low V + a histogram concentrated
        # to a >=90% default elects the tiered dress on SERVING dominance --
        # zero-pop reads, the census IS the planes -- accepting bounded disk
        # (<=4x, in practice pennies against the whole segment). The same
        # within-tolerance precedent that seated enc-3's frames.
        best = tiered

    # tag 13 = BYTE-PLANES (Jackson's dress, v2): per frame, codes split
    # into byte lanes (top byte first), EACH plane its own zstd frame --
    # a plane you don't need is a plane never inflated. Descent prunes
    # 255/256 per level and the top plane's census fits a cache line
    # neighborhood. Operator-tagged columns only (semantic knowledge is
    # the operator's), passing the data gates (multi-byte, growth <=25%).
    _e13_tags = set(x for x in os.environ.get('WDB_E13_TAGS', '').split(',') if x)
    _e13_want = os.environ.get('WDB_E13_FORCE') or (nm is not None and nm in _e13_tags)
    if _e13_want and codes.size and int(np.max(codes)) >= 256:
        a13 = np.asarray(codes, dtype=np.int64)
        nby = (max(1, int(np.max(a13)).bit_length()) + 7) // 8
        if nby > 1:
            cxb13 = zstd.ZstdCompressor(level=3)
            fr13 = []
            for i in range(0, a13.size, BLOCK_ROWS):
                ch = a13[i:i + BLOCK_ROWS]
                if ch.size < BLOCK_ROWS:
                    ch = np.concatenate([ch, np.zeros(BLOCK_ROWS - ch.size,
                                                      np.int64)])
                for b in range(nby):
                    pl = ((ch >> (8 * (nby - 1 - b))) & 0xFF).astype(np.uint8)
                    fr13.append(cxb13.compress(pl.tobytes()))
            v13 = (bytes([13, nby]) + struct.pack('<II', BLOCK_ROWS, len(fr13))
                   + b''.join(struct.pack('<Q', x) for x in
                              np.cumsum([0] + [len(f) for f in fr13]).tolist())
                   + b''.join(fr13))
            if os.environ.get('WDB_E13_FORCE'):
                best = v13                        # rehearsal: wear it regardless
            elif len(v13) <= 1.25 * len(best):
                best = v13                        # the tagged column's gate
    if best is zsec:

        # tag 3 = BLOCKED frames: independent zstd frame per BLOCK_ROWS rows + a frame offset
        # index. Buys pop/scan/prune access (touched frames only, ~0.6 ms/frame) for a measured
        # +1-9% per column at 512K rows (knee sweep) -- adopted when within 10% of the seal.
        # Bitpack (already point-readable) and staircase still win outright when smaller.
        a = np.asarray(codes, dtype=wdt)
        cxb = zstd.ZstdCompressor(level=CODE_ZSTD_LEVEL)
        blocked = None
        for BR9 in ((65536, BLOCK_ROWS) if (BLOCK_ROWS > 65536
                    and codes.size and int(np.max(codes)) > 1)
                    else (BLOCK_ROWS,)):
            frames = [cxb.compress(a[i:i + BR9].tobytes())
                      for i in range(0, a.size, BR9)]
            offs = np.zeros(len(frames) + 1, dtype=np.uint32)
            np.cumsum([len(f) for f in frames], out=offs[1:])
            cand9 = (bytes([3, width]) + struct.pack('<II', BR9, len(frames))
                     + offs.tobytes() + b''.join(frames))
            if len(cand9) <= (len(zsec) if zsec is not None else len(packed)) * 1.10 \
                    and (blocked is None or len(cand9) <= 1.10 * len(blocked)):
                blocked = cand9                  # Jackson's rule: fine frames
                break                            # up to +10%; else the coarse
        if blocked is not None:
            best = blocked
    # tag 14 = FIELD PLANES (Jackson's dress): dates decompose to y/m/d u8
    # planes, each its own zstd stream -- the calendar's internal correlation
    # compresses BELOW naive entropy (82.7 vs 94.1MB measured on l_shipdate),
    # and band predicates later read only the fields they constrain. Types
    # NOMINATE (the caller passes date_vals only for dt==3), measurements
    # SIZE (year width from the real range), the size election alone ELECTS.
    if date_vals is not None and codes.size:
        try:
            dv = np.asarray(date_vals, dtype=np.int64)[np.asarray(codes, dtype=np.int64)]
            dt64 = dv.astype('timedelta64[D]') + np.datetime64('1970-01-01')
            Y14 = dt64.astype('datetime64[Y]').astype(np.int64) + 1970
            ybase = int(Y14.min())
            if int(Y14.max()) - ybase <= 255:
                M14 = (dt64.astype('datetime64[M]').astype(np.int64) % 12).astype(np.uint8)
                D14 = (dv - dt64.astype('datetime64[M]').astype('datetime64[D]').astype(np.int64)).astype(np.uint8)
                zc14 = zstd.ZstdCompressor(level=CODE_ZSTD_LEVEL)
                FR14 = 1 << 23                       # 8M rows/frame: the planes are LANEABLE
                nfr14 = (codes.size + FR14 - 1) // FR14
                secs14 = []
                offs14 = []
                for pl in ((Y14 - ybase).astype(np.uint8), M14, D14):
                    frs = [zc14.compress(pl[i:i + FR14].tobytes())
                           for i in range(0, pl.size, FR14)]
                    o9 = np.zeros(nfr14 + 1, dtype=np.uint32)
                    np.cumsum([len(f) for f in frs], out=o9[1:])
                    offs14.append(o9); secs14.append(b''.join(frs))
                cand14 = (bytes([14, 1]) + struct.pack('<HII', ybase, FR14, nfr14)
                          + b''.join(o.tobytes() for o in offs14)
                          + b''.join(secs14))
                if len(cand14) < len(best):
                    best = cand14
        except Exception:
            pass
    # tag 5 = PATCHED BUCKETS (Jackson's format): 4-bit pointers into a 15-entry hot
    # table + escape patches (u16) + per-32K escape offsets. Skewed low-V numeric
    # streams only; adopted when within 25% of the zstd seal. Buys O(1) point reads
    # (no frame ever inflates), an escape-array hunt, and a stream decode that beats
    # parallel zstd -- measured 0.1ms/84ms point, 54ms/70ms stream on ResolutionWidth.
    if enc5_ok and best is not stair:
        arr16 = np.asarray(codes, dtype=np.int64)
        V5 = int(arr16.max()) + 1 if arr16.size else 0
        if 15 < V5 <= 65535:
            cn5 = np.bincount(arr16, minlength=V5)
            hot = np.argsort(cn5)[::-1][:15].astype(np.uint16)
            lut = np.full(V5, 15, dtype=np.uint8)
            lut[hot] = np.arange(15, dtype=np.uint8)
            nib = lut[arr16]
            em = nib == 15
            patches = arr16[em].astype(np.uint16)
            Nr = arr16.size
            nibp = np.zeros(Nr + (Nr & 1), dtype=np.uint8)
            nibp[:Nr] = nib
            pk = (nibp[0::2] | (nibp[1::2] << 4)).astype(np.uint8)
            BR5 = 32768
            nb5 = (Nr + BR5 - 1) // BR5
            eb = np.add.reduceat(em.astype(np.int64), np.arange(0, Nr, BR5)) if Nr else np.zeros(0, dtype=np.int64)
            eo = np.concatenate([[0], np.cumsum(eb)]).astype(np.uint32)
            e5 = (bytes([5, 2]) + struct.pack('<IIQH', BR5, nb5, int(patches.size), 15)
                  + hot.tobytes() + eo.tobytes() + patches.tobytes() + pk.tobytes())
            if len(e5) <= (len(zsec) if zsec is not None else len(packed)) * 1.25 and (best is not packed or len(e5) < len(best)):
                best = e5
            elif V5 > 270:
                # tag 6 = WARM BUCKETS: hot nibble -> warm byte (255 seats) -> u16 cold.
                # The two-tier sibling for columns whose top-15 is thin but top-270 is
                # fat (RegionID 49%->94%). Same 1.25 bar, same block independence.
                hot255 = np.argsort(cn5)[::-1][:270].astype(np.uint16)
                warm = hot255[15:270]
                lutw = np.full(V5, 255, dtype=np.uint8)
                lutw[warm] = np.arange(255, dtype=np.uint8)
                em1 = nib == 15                          # nibble escapes (reuse enc-5's hot-15)
                wb = lutw[arr16[em1]]                    # warm byte per nibble-escape
                em2pos = wb == 255
                patches6 = arr16[em1][em2pos].astype(np.uint16)
                BR5b = 32768
                nb6 = (Nr + BR5b - 1) // BR5b
                if Nr:
                    e1b = np.add.reduceat(em1.astype(np.int64), np.arange(0, Nr, BR5b))
                    full_e2 = np.zeros(Nr, dtype=np.int64)
                    idx1 = np.flatnonzero(em1)
                    full_e2[idx1[em2pos]] = 1
                    e2b = np.add.reduceat(full_e2, np.arange(0, Nr, BR5b))
                else:
                    e1b = np.zeros(0, dtype=np.int64); e2b = np.zeros(0, dtype=np.int64)
                e1o = np.concatenate([[0], np.cumsum(e1b)]).astype(np.uint32)
                e2o = np.concatenate([[0], np.cumsum(e2b)]).astype(np.uint32)
                e6 = (bytes([6, 2]) + struct.pack('<IIQQH', BR5b, nb6, int(wb.size),
                                                  int(patches6.size), 15)
                      + hot.tobytes() + warm.tobytes()
                      + e1o.tobytes() + e2o.tobytes()
                      + wb.tobytes() + patches6.tobytes() + pk.tobytes())
                if len(e6) < len(best):          # post-hits_4 doctrine: the warm tier
                    best = e6                    # adopts on STRICT dominance only --
                                                 # its 1.25x access bar died with the
                                                 # heir whose tolls it never repaid
    # tag 17 = RAW PACKED CODES (Jackson's deal law). Nomination: the zstd
    # frame stream won so far, V >= 3 (binaries are their own terminal dress),
    # and the column is big enough that the toll is real (small fixtures stay
    # deterministic). Election: S > T on MEASURED terms -- S the symmetric
    # shrink ratio zstd buys vs packed, T the symmetric slowdown its
    # decompression charges. zstd keeps the column only when it shrinks it
    # more than it slows it.
    if best and best[0] == 3 and bits >= 2 and bits <= 16 and codes.size >= (1 << 22):
        arr17 = np.asarray(codes, dtype=np.int64)
        tb17 = np.zeros(arr17.size * bits, dtype=np.uint8)
        for k17 in range(bits):
            tb17[k17::bits] = (arr17 >> k17) & 1
        pk17 = np.packbits(tb17, bitorder='little')
        cand17 = (bytes([17, bits]) + struct.pack('<I', arr17.size)
                  + pk17.tobytes() + b'\x00\x00')
        S17 = (len(cand17) - len(best)) / max(1, len(best) + len(cand17))   # zstd SHRINK vs packed
        T17 = 0.17
        # T is the ENGINE's measured slowdown ratio for zstd on real ops
        # (scan+group geo across the deals table: 0.15-0.20, near-constant
        # per column) -- a calibrated constant, deterministic at encode.
        # Per-column micro-timing tried twice and lied both ways: a bare
        # sum overweighted decompression, a 1T bincount underweighted it.
        # Recalibrate by re-running bench deals when the engine changes.
        if not (S17 > T17):
            best = cand17
    return best

def _pair15_candidate(pa, pb):
    """Size the CLOCK dress for a date pair: anchor planes + u8 delta +
    orientation bit, framed. Returns (blob_bytes, size) or None."""
    ua, ub = np.asarray(pa['uniq'], np.int64), np.asarray(pb['uniq'], np.int64)
    da, db = ua[np.asarray(pa['codes'])], ub[np.asarray(pb['codes'])]
    dl = np.abs(da - db)
    if int(dl.max(initial=0)) > 255:
        return None
    bit = (da <= db)                              # 1: column A is the anchor(min)
    mn = np.minimum(da, db)
    dt64 = mn.astype('datetime64[D]')
    Y = (dt64.astype('datetime64[Y]').astype(np.int64) + 1970)
    ybase = int(Y.min())
    if int(Y.max()) - ybase > 255:
        return None
    Yp = (Y - ybase).astype(np.uint8)
    Mp = (dt64.astype('datetime64[M]').astype(np.int64) % 12).astype(np.uint8)
    Dp = (mn - dt64.astype('datetime64[M]').astype('datetime64[D]').astype(np.int64)).astype(np.uint8)
    DL = dl.astype(np.uint8)
    BB = np.packbits(bit)
    zc15 = zstd.ZstdCompressor(level=CODE_ZSTD_LEVEL)
    FR = 1 << 23
    nfr = (mn.size + FR - 1) // FR
    offs, secs = [], []
    for pl, ib in ((Yp, 1), (Mp, 1), (Dp, 1), (DL, 1), (BB, 0)):
        if ib:
            frs = [zc15.compress(pl[i:i + FR].tobytes()) for i in range(0, pl.size, FR)]
        else:                                     # bit plane: FR/8 bytes per frame
            F8 = FR >> 3
            frs = [zc15.compress(pl[i:i + F8].tobytes()) for i in range(0, pl.size, F8)]
            while len(frs) < nfr: frs.append(zc15.compress(b''))
        o9 = np.zeros(nfr + 1, dtype=np.uint32)
        np.cumsum([len(f) for f in frs], out=o9[1:])
        offs.append(o9); secs.append(b''.join(frs))
    pn = pb['nm'].encode()
    blob = (bytes([15, 1]) + struct.pack('<HII', ybase, FR, nfr)
            + b''.join(o.tobytes() for o in offs)
            + struct.pack('<H', len(pn)) + pn
            + b''.join(secs))
    return blob, len(blob)


def _apply_pairs15(preps, cols, date_pairs):
    """OPERATOR-DECLARED clock pairs (Jackson's ruling): two dates that are
    sides of one event pair ONLY when the operator says so -- the bit's
    meaning (outstanding vs closed) is workload semantics the data cannot
    reveal, so the engine never guesses. Declared pairs must satisfy the
    physical property (both plane-eligible, measured delta <= 255) or the
    encode FAILS LOUD."""
    if not date_pairs:
        return
    for na, nb in date_pairs:
        if na not in preps or nb not in preps:
            raise ValueError("date_pairs: unknown column in (%s, %s)" % (na, nb))
        pa, pb = preps[na], preps[nb]
        for p9 in (pa, pb):
            if p9.get('has_null') or p9.get('mode') not in (0, 2):
                raise ValueError("date_pairs: %s not clock-eligible (nulls or mode)" % p9['nm'])
        cand = _pair15_candidate(pa, pb)
        if cand is None:
            raise ValueError("date_pairs: (%s, %s) delta exceeds u8 or year span too wide"
                             % (na, nb))
        if os.environ.get('WDB_ENCODE_VERBOSE'):
            sa = len(_code_section(pa['codes'], pa['bits'], nm=na, date_vals=pa['uniq']))
            sb = len(_code_section(pb['codes'], pb['bits'], nm=nb, date_vals=pb['uniq']))
            print('PAIR15 declared %s+%s: %d -> %d bytes (%+d)'
                  % (na, nb, sa + sb, cand[1], cand[1] - (sa + sb)), flush=True)
        pa['force15'] = cand[0]
        pb['force16'] = na.encode()


def _elect_pair15_retired(preps, cols):
    """Nominate date pairs by PROPERTY (both plane-eligible, no nulls,
    measured bounded delta); size the clock against the two standalone
    code sections; elect at most one pair per table, best savings."""
    def eligible(p):
        if p.get('has_null') or p.get('mode') not in (0, 2):
            return False
        u = np.asarray(p['uniq'])
        if u.size == 0:
            return False
        if p.get('dtype') == 3:
            return True
        return (p.get('dtype') == 0 and 366 <= int(u[0]) and int(u[-1]) <= 65700
                and np.asarray(u).size <= 20000)   # keys are dense-from-1; dates are neither
    dcols = [nm for nm in cols if eligible(preps[nm])]
    if os.environ.get('WDB_ENCODE_VERBOSE'):
        print('PAIR15: dcols=%r' % dcols, flush=True)
    best = None
    for i in range(len(dcols)):
        for j in range(i + 1, len(dcols)):
            pa, pb = preps[dcols[i]], preps[dcols[j]]
            cand = _pair15_candidate(pa, pb)
            if cand is None:
                continue
            sa = len(_code_section(pa['codes'], pa['bits'], nm=pa['nm'], date_vals=pa['uniq']))
            sb = len(_code_section(pb['codes'], pb['bits'], nm=pb['nm'], date_vals=pb['uniq']))
            save = (sa + sb) - cand[1]
            if os.environ.get('WDB_ENCODE_VERBOSE'):
                print('PAIR15: %s+%s sa=%d sb=%d cand=%d save=%d' % (dcols[i], dcols[j], sa, sb, cand[1], save), flush=True)
            if save > 0 and (best is None or save > best[0]):
                best = (save, dcols[i], dcols[j], cand[0])
    if best is not None:
        _sv, na, nb, blob = best
        preps[na]['force15'] = blob
        preps[nb]['force16'] = na.encode()


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
    if p.get('force15') is not None:
        out += p['force15']
        return bytes(out), (len(out), p['V'], p['bits'], p['dtype'], p['mode'], p['has_null'], p['aux'])
    if p.get('force16') is not None:
        pn16 = p['force16']
        out += bytes([16]) + struct.pack('<H', len(pn16)) + pn16
        return bytes(out), (len(out), p['V'], p['bits'], p['dtype'], p['mode'], p['has_null'], p['aux'])
    out += _code_section(p['codes'], p['bits'],
                         enc5_ok=(p.get('dtype') == 0 and p['mode'] in (0, 1, 2)),
                         nm=p['nm'],
                         date_vals=(p['uniq'] if ((p.get('dtype') == 3
                                                   or (p.get('dtype') == 0 and p['mode'] in (0, 2)
                                                       and 0 < np.asarray(p['uniq']).size <= 20000
                                                       and 366 <= int(np.asarray(p['uniq'])[0])
                                                       and int(np.asarray(p['uniq'])[-1]) <= 65700))
                                                  and not p['has_null']
                                                  and p['mode'] in (0, 2)) else None))
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
    # the inline VALUE blob keeps the archival level: the inline-vs-dictionary decision is a
    # size race and must not ride the code-stream speed knob (a near-unique column flipped
    # modes when CODE_ZSTD_LEVEL went 19 -> 9)
    zc = zstd.ZstdCompressor(level=max(CODE_ZSTD_LEVEL, INLINE_ZSTD_LEVEL))
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


def _arrow_string_prep(nm, chunked):
    """THE ARROW LAW: a string column is dictionary-encoded IN ARROW MEMORY (no 100M Python
    objects -- measured 45 GB and 387s for one 100M-row column the pandas way), its dictionary
    sorted with arrow, codes remapped with numpy. Returns a prep dict like _prep_column's
    (mode 0/1 strings) or None when the column is not a string column."""
    import pyarrow as pa, pyarrow.compute as pc
    t = chunked.type
    if not (pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_binary(t) or pa.types.is_large_binary(t)):
        return None
    if isinstance(chunked, pa.ChunkedArray):
        chunked = pa.concat_arrays(chunked.chunks) if chunked.num_chunks > 1 else chunked.chunk(0)
    de = pc.dictionary_encode(chunked)                      # indices (int32) + dictionary (unique, first-seen order)
    dct = de.dictionary; idx = de.indices
    null_mask = None; has_null = 0
    if chunked.null_count:
        null_mask = np.asarray(pc.is_null(chunked).to_numpy(zero_copy_only=False), dtype=bool); has_null = 1
    order = pc.sort_indices(dct).to_numpy()                 # dictionary in sorted order
    rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
    codes = np.zeros(len(chunked), np.int64)
    raw = idx.to_numpy(zero_copy_only=False)
    if null_mask is not None:
        nn = ~null_mask
        codes[nn] = rank[np.asarray(raw[nn], dtype=np.int64)]; codes[null_mask] = len(order)
    else:
        codes[:] = rank[np.asarray(raw, dtype=np.int64)]
    sorted_dict = pc.take(dct, pa.array(order))
    valb = [(v if isinstance(v, bytes) else v.encode('utf-8', 'surrogatepass')) for v in sorted_dict.to_pylist()]
    del de, dct, idx, raw, sorted_dict
    V = len(valb) + has_null
    bits = max(1, int(np.ceil(np.log2(max(V, 2)))))
    mode = 1 if (V - has_null) > FC_THRESHOLD else 0
    uniq = None
    aux = 0
    if mode == 1 and CHUNK_DICT:
        aux |= 0x40      # THE CHUNK LAW: a big front-coded dictionary is written as independent zstd frames
                         # (CHUNK_DICT_VALS values each) so a lookup decompresses one frame, not 6M values
                         # (SearchPhrase <> '' on gov7: 0.34s = one 6M-value frame per probe; the reference 0.05s)
    return dict(nm=nm, dtype=1, has_null=has_null, V=V, valb=valb, codes=codes, aux=aux, uniq=uniq, bits=bits, mode=mode)


_CASTS = {
    # DECLARED CASTS at ingest (the operator's ruling, never a guess): a uint16 day count is a
    # DATE, an int64 epoch-second count is a TIMESTAMP
    'date_days':      lambda a: np.asarray(a).astype(np.int64).astype('datetime64[D]'),
    'timestamp_s':    lambda a: np.asarray(a).astype(np.int64).astype('datetime64[s]'),
    'timestamp_ms':   lambda a: np.asarray(a).astype(np.int64).astype('datetime64[ms]'),
    'timestamp_us':   lambda a: np.asarray(a).astype(np.int64).astype('datetime64[us]'),
}


def _column_job(input_path, nm, reader, cast=None, perm_path=None):
    """one column, start to blob, in a worker process (reads its own column: no table in RAM).
    perm_path: THE CLUSTER ORDER -- a row permutation every column gathers through, so the
    segment is time-ordered and the clock columns become STAIRCASES (free ordering for
    windows and ranges: the reference encode had it; the raw file order does not)."""
    prep = None
    perm = np.load(perm_path, mmap_mode='r') if perm_path else None
    if cast is None and str(input_path).lower().endswith('.parquet'):
        try:
            import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
            col = pq.read_table(input_path, columns=[nm]).column(0)
            if perm is not None:
                col = pc.take(col, pa.array(np.asarray(perm)))
            prep = _arrow_string_prep(nm, col)
            del col
        except Exception:
            prep = None
    if prep is None:
        arr = wdb_read.read_one_column(input_path, nm, reader=reader)
        if perm is not None:
            arr = arr[np.asarray(perm)]
        if cast is not None:
            arr = _CASTS[cast](arr)
        prep = _prep_column(nm, arr)
        del arr
    blob, size = _serialize_column(prep, zstd.ZstdCompressor(level=ZSTD_LEVEL))
    import resource as _rs
    peak = _rs.getrusage(_rs.RUSAGE_SELF).ru_maxrss * 1024      # this worker's peak so far: THE MEASURED TRUTH
    return nm, bytes(blob), size, peak


def _column_cost(input_path, cols, reader, N):
    """THE ORDER OF OPERATIONS (Jackson): columns ranked by expected encode cost so the heavy
    ones (wide dictionaries, strings: front-coding + zstd) start first and never become the
    tail, while the layout-only ones (flags, sequences, narrow codes) fill the gaps. Cost is
    estimated from the schema and a sample, never from a full read."""
    est = {}
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(input_path)
        samp = pf.read_row_group(0, columns=cols)
        n0 = max(1, samp.num_rows)
        for nm in cols:
            c = samp.column(nm)
            t = str(c.type)
            try:
                nuniq = len(c.unique())
            except Exception:
                nuniq = n0
            frac = nuniq / n0
            if 'string' in t or 'binary' in t or 'large' in t:
                if nuniq <= 4096:
                    est[nm] = 0.5                         # a low-cardinality string is an enum: a narrow column
                else:
                    est[nm] = 8.0 * (1 + 20 * frac)      # strings: dictionary + front-coding + zstd
            elif frac > 0.5:
                est[nm] = 2.0                             # near-unique numerics: sequence / delta
            elif nuniq <= 16:
                est[nm] = 0.3                             # flags and small enums: a bit-pack
            else:
                est[nm] = 1.0 + 4 * frac
    except Exception:
        est = {nm: 1.0 for nm in cols}
    return est


def _encode_streaming(input_path, out_path, columns, reader, cubes, workers, t0, casts=None, cluster_by=None):
    """THE ENCODER UNDER THE GOVERNOR: every column is an independent job in a worker
    process (it reads its own column: the table is never in RAM); jobs start heavy-first
    (_column_cost) so the long ones never become the tail and lighter ones FILL THE GAPS;
    the number in flight is capped by workers AND by a byte budget (WDB_ENCODE_MB, default
    half of RAM / cgroup) on measured working-set classes, with at most three string
    columns cooking at once; each finished blob is written to the file the moment it lands
    and freed. SELF-HEALING: a pool killed by the OOM killer halves its concurrency and
    re-queues the columns it was cooking -- the governor's promise is never to crash. Blob
    order in the file is completion order: the reader keys columns by name and the catalog
    holds the logical order."""
    import concurrent.futures as cf, os as _os
    cols, N = wdb_read.column_schema(input_path, columns, reader=reader)
    est = _column_cost(input_path, cols, reader, N)
    order = sorted(cols, key=lambda c: -est.get(c, 1.0))
    nworkers = max(1, min(workers or (_os.cpu_count() or 4), _os.cpu_count() or 4))
    try:
        budget = int(float(_os.environ.get('WDB_ENCODE_MB', '0'))) << 20
    except Exception:
        budget = 0
    if not budget:
        try:
            phys = _os.sysconf('SC_PAGE_SIZE') * _os.sysconf('SC_PHYS_PAGES')
            with open('/sys/fs/cgroup/memory.max') as f:
                v = f.read().strip()
                if v.isdigit(): phys = min(phys, int(v))
            budget = phys // 2
        except Exception:
            budget = 32 << 30
    def cls(nm):
        e = est.get(nm, 1.0)
        return 'string' if e >= 8 else ('wide' if e >= 2 else 'narrow')
    measured = {'string': 250, 'wide': 100, 'narrow': 40}     # bytes per row: conservative starts (URL/Referer/Title peak ~25 GB at 100M rows), raised as workers report
    def working_set(nm):
        return int(N * measured[cls(nm)]) + (400 << 20)
    def learn(nm, peak):
        # THE ENCODER MEASURES ITSELF: a worker's peak RSS updates its class's bytes-per-row
        # (max seen, plus 25% headroom) so the budget stops guessing after the first column.
        # One observation can never teach more than "two of this class fit the budget": an
        # outlier (73 GB on URL) had starved the string class to one column at a time.
        c = cls(nm); seen = (peak - (400 << 20)) / max(1, N) * 1.25
        cap = (budget / 2 - (400 << 20)) / max(1, N)
        seen = min(seen, cap)
        if seen > measured[c]:
            measured[c] = seen
    sizes = {}
    perm_path = None
    if cluster_by:
        # THE CLUSTER ORDER: argsort the clustering column(s) once (stable), share the
        # permutation as an mmap'd file; every worker gathers its column through it
        keys9 = [cluster_by] if isinstance(cluster_by, str) else list(cluster_by)
        arrs9 = [np.asarray(wdb_read.read_one_column(input_path, k, reader=reader)) for k in keys9]
        perm = np.lexsort(tuple(reversed(arrs9))).astype(np.int64 if N >= (1 << 31) else np.int32)
        del arrs9
        perm_path = out_path + '.perm.tmp.npy'
        np.save(perm_path, perm); del perm
        if _os.environ.get('WDB_ENCODE_VERBOSE'):
            print('  cluster order by %s computed (%d rows)' % (', '.join(keys9), N), flush=True)
    fh = open(out_path, 'wb')
    fh.write(b'WVDB4' + struct.pack('<H', len(cols)) + struct.pack('<I', N))
    pending = list(order); verbose = bool(_os.environ.get('WDB_ENCODE_VERBOSE'))
    while pending:
        inflight = {}; used = 0
        try:
            # a FRESH PROCESS PER COLUMN: ru_maxrss is a process-lifetime max, so a reused worker
            # reported its heaviest column's peak for every later one (a flag column at 658 B/row)
            try:
                ex_ctx = cf.ProcessPoolExecutor(max_workers=nworkers, max_tasks_per_child=1)
            except TypeError:
                ex_ctx = cf.ProcessPoolExecutor(max_workers=nworkers)
            with ex_ctx as ex:
                while pending or inflight:
                    while pending and len(inflight) < nworkers:
                        heavy_in = sum(1 for v in inflight.values() if est.get(v, 1.0) >= 8)
                        pick = None
                        for i9, cand in enumerate(pending):
                            if est.get(cand, 1.0) >= 8 and heavy_in >= 4: continue        # strings measured 9-25 GB: four fit
                            if not inflight or used + working_set(cand) <= budget:
                                pick = i9; break
                        if pick is None: break
                        nm = pending.pop(pick)
                        inflight[ex.submit(_column_job, input_path, nm, reader, (casts or {}).get(nm), perm_path)] = nm
                        used += working_set(nm)
                    if not inflight: break
                    fut = next(iter(cf.as_completed(list(inflight.keys()))))
                    res = fut.result()                 # BEFORE the pop: a result that raises leaves the column in inflight for the re-queue
                    nm = inflight.pop(fut); used -= working_set(nm)
                    cname, blob, size = res[0], res[1], res[2]
                    if len(res) > 3: learn(cname, int(res[3]))
                    fh.write(blob); fh.flush(); sizes[cname] = size
                    del blob
                    if verbose:
                        print('  encoded %-24s (%d/%d, %.0fs, %d in flight, class %s @ %.0f B/row)' % (cname, len(sizes), len(cols), time.time() - t0, len(inflight), cls(cname), measured[cls(cname)]), flush=True)
        except cf.process.BrokenProcessPool:
            lost = [v for v in inflight.values() if v not in sizes]
            pending = lost + [p for p in pending if p not in sizes]
            if nworkers <= 1:
                fh.close(); raise
            nworkers = max(1, nworkers // 2)
            print('  pool killed (memory): retreating to %d workers, re-queueing %d columns (%s)' % (nworkers, len(lost), ', '.join(lost[:5])), flush=True)
        # THE COMPLETENESS LAW: the header promised len(cols) blobs; whatever the retreats lost
        # is encoded again, in this process, before the file closes (measured: URL and Referer
        # vanished across two retreats and the encode reported success)
        missing = [c for c in cols if c not in sizes]
        if missing and not pending:
            print('  completeness: %d columns missing after the pool (%s): encoding in-process' % (len(missing), ', '.join(missing[:5])), flush=True)
            for nm in missing:
                res = _column_job(input_path, nm, reader, (casts or {}).get(nm), perm_path)
                cname, blob, size = res[0], res[1], res[2]
                fh.write(blob); fh.flush(); sizes[cname] = size; del blob
    fh.close()
    if perm_path:
        try: _os.remove(perm_path)
        except OSError: pass
    out = None
    if cubes:
        try:
            from wdb_engine import Segment
            import wdb_cube
            _seg = Segment(out_path)
            if cubes == 'auto':
                cards = wdb_cube.segment_cardinalities(_seg)
                specs = wdb_cube.maximal_cube_specs(cards)
            else:
                specs = cubes
            wdb_cube.build_and_write(_seg, specs, workers=workers or 4)
        except Exception:
            pass
    # THE DIFFERENTIATOR LAW (Jackson): a column whose role is differentiation
    # (V near N) gets its exception shelf born AT ENCODE TIME -- row position
    # takes over identity; values demote to decode-only. Qualifies only while
    # the exceptions stay small: repeated rows < 0.75% of the distinct count.
    try:
        _birth_differentiator_shelves(out_path)
    except Exception:
        pass
    return dict(n_rows=N, n_cols=len(cols), bytes=(len(out) if out is not None else __import__('os').path.getsize(out_path)), seconds=time.time() - t0,
                sizes=sizes, cluster=None)


def encode(input_path, out_path, columns=None, workers=None, reader='auto', fd_specs=None, cluster_by=None, cubes=None, stream=False, date_pairs=None, casts=None):
    """fd_specs: optional {dependent_col: determinant_col} — store the dependent column as
    a mode-3 FD-reference into the determinant (lossless iff the FD is exact; callers pass
    only verified FDs). Determinant must be a normal (non-FD) column in the same segment.
    stream=True: read+encode one column at a time (peak memory ~= one column) for the
    no-cluster, no-FD case; output is byte-identical to the default path."""
    import os, concurrent.futures as cf
    fd_specs = fd_specs or {}
    t0 = time.time()
    if stream and not fd_specs:                      # the streaming encoder carries cluster_by (a shared permutation)
        return _encode_streaming(input_path, out_path, columns, reader, cubes, workers, t0, casts=casts, cluster_by=cluster_by)
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
        preps15 = {}
        if workers > 1 and len(cols) > 1:
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                for nm, p9 in zip(cols, ex.map(lambda n: _prep_column(n, coldata[n]), cols)):
                    preps15[nm] = p9
        else:
            for nm in cols: preps15[nm] = _prep_column(nm, coldata[nm])
        _apply_pairs15(preps15, cols, date_pairs)
        def _blob(nm):
            return nm, _serialize_column(preps15[nm],
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
        _apply_pairs15(preps, cols, date_pairs)
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
    if cluster_meta is not None:
        import pickle
        with open(out_path + '.cluster', 'wb') as _cf:
            pickle.dump(cluster_meta, _cf, protocol=4)
    if cubes:                                     # materialise GROUP BY cubes (cap in wdb_cube declines
        try:                                      # any grouping above CUBE_MAX_CELLS)
            from wdb_engine import Segment
            import wdb_cube
            _seg = Segment(out_path)
            if cubes == 'auto':                   # exhaustive but non-redundant: only the maximal cubes
                cards = wdb_cube.segment_cardinalities(_seg)   # are persisted; sub-cubes derive at query time
                specs = wdb_cube.maximal_cube_specs(cards)
            else:
                specs = cubes
            wdb_cube.build_and_write(_seg, specs, workers=workers)
        except Exception:
            if os.environ.get('WDB_ENCODE_VERBOSE'):
                import traceback, sys as _sy
                print('CUBE BUILD DIED:', file=_sy.stderr); traceback.print_exc()
    try:
        _birth_differentiator_shelves(out_path)
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


def _birth_differentiator_shelves(out_path):
    import importlib, pickle
    import numpy as _np
    eng = importlib.import_module('wdb_engine')
    seg = eng.Segment(out_path)
    N = int(seg.N)
    for cn, c in list(seg.cols.items()):
        try:
            V = int(c.get('V') or 0)
            if V * 2 < N or V < 1024:
                continue                          # role isn't differentiation
            codes = _np.asarray(seg._raw_codes(cn))
            cnt = _np.bincount(codes, minlength=V)
            rep = _np.flatnonzero(cnt[codes] >= 2)
            if rep.size >= 0.0075 * V:
                continue                          # too many exceptions: disqualified
            pickle.dump({'n': N, 'rows': rep.astype(_np.uint32)},
                        open(out_path + '.%s.ptrep' % cn, 'wb'), protocol=4)
            _expand_pair_shelves(seg, out_path, cn, rep)
        except Exception:
            continue


def _expand_pair_shelves(seg, out_path, a, rep):
    """Jackson's eager expansion: the differentiator pairs with every
    qualifying companion. The law recurses verbatim -- repeated-PAIR rows
    must stay under 0.75% of the companion's distinct count; low-V columns
    self-disqualify (their repeats swamp the threshold). Kilobytes total."""
    import pickle
    import numpy as _np
    rep = _np.sort(_np.asarray(rep, _np.int64))
    if rep.size == 0:
        return
    ac = _np.asarray(seg._raw_codes(a), _np.int64)[rep]
    for y, cy in list(seg.cols.items()):
        try:
            if y == a:
                continue
            Vy = int(cy.get('V') or 0)
            if Vy < 1024:
                continue                          # low V: repeats swamp 0.75%
            bc = _np.asarray(seg._raw_codes(y), _np.int64)[rep]
            key = (ac << 32) | bc
            order = _np.argsort(key, kind='stable')
            key2 = key[order]
            sidx = rep[order]
            brk = _np.empty(key2.size, bool)
            brk[0] = True
            _np.not_equal(key2[1:], key2[:-1], out=brk[1:])
            st = _np.flatnonzero(brk)
            gcnt = _np.diff(_np.append(st, key2.size))
            rp = int(gcnt[gcnt >= 2].sum())
            if rp >= 0.0075 * Vy:
                continue                          # relationship disqualified
            gid = _np.zeros(key2.size, _np.int64)
            repg = _np.flatnonzero(gcnt >= 2)
            ga = []
            gb = []
            for j, gi in enumerate(repg.tolist()):
                gid[st[gi]:st[gi] + int(gcnt[gi])] = j + 1
                ga.append(int(key2[st[gi]] >> 32))
                gb.append(int(key2[st[gi]] & 0xFFFFFFFF))
            pickle.dump({'n': int(seg.N), 'rows': sidx.astype(_np.uint32),
                         'gid': gid.astype(_np.uint16),
                         'ga': _np.asarray(ga, _np.int64),
                         'gb': _np.asarray(gb, _np.int64),
                         'ra': key2 >> 32, 'rb': key2 & 0xFFFFFFFF},
                        open(out_path + '.%s__%s.pt2' % (a, y), 'wb'),
                        protocol=4)
        except Exception:
            continue
