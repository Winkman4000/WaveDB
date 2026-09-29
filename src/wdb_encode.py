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
# THE THREE STREAMS (Jackson, 2026-09-23): a chunked front-coded dictionary is written as headers,
# a start-of-character mask, and text -- each chunk's three frames, each kind contiguous. Length is
# read from headers + mask without a text byte; the interleaved form is rejoined byte-for-byte.
# Measured on cbdb (all six chunked dictionaries): 1820 -> 1857 MB, +0.42% of the file.
FC3_DICT = bool(int(os.environ.get('WDB_FC3', '1')))
_INLINE_ENABLED = True  # mode-5 inline strings (toggleable for ablation/debug)

def _int_dictionary(data):
    """THE DICTIONARY WITHOUT THE SORT (Jackson, 2026-09-18): np.unique(return_inverse=True) on an
    integer column is a full O(N log N) sort of 100M values to discover a distinct set -- 20-40s
    per column, the largest single cost of an integer column's encode (RegionID 24.6s for 9,040
    values). The route is ASSIGNED from what is known before any pass:
      bounded range (<= 4N) -> bincount: one linear pass, sorted by construction   (RegionID 24.6s -> 0.4s)
      otherwise -> hash factorize UNSORTED, then sort the DICTIONARY and remap: V log V, not N log N
                   (ClientIP 15.7s -> 4.2s; WatchID, V = N, breaks even -- there is no worse case)
      near-unique (>= 95% distinct in a 1M sample) -> np.unique: the hash table of N keys loses to
                   the sort (WatchID 68s -> 80s without this gate; a 50% gate mis-sent ClientIP,
                   whose 10%-distinct column reads 63% distinct in a sample -- samples lie about V/N
                   except at the top). Same contract everywhere: a SORTED dictionary, identical codes."""
    data = np.asarray(data)
    N = data.size
    if N == 0:
        return np.unique(data, return_inverse=True)
    lo = int(data.min()); hi = int(data.max()); rng = hi - lo + 1
    if rng <= (8 << 20):
        # THE RANGE IS ABSOLUTE, NOT RELATIVE TO N: 'rng <= 4N' let a column with a 300M-value range
        # allocate a 300M-entry count array and a 300M-entry LUT (5 GB of transient for a 'narrow'
        # column); the encoder's learner read that peak, re-classed every int column, and ran the
        # whole realm 2-3 wide instead of 12 (405s -> 920s). bincount only when the count array is
        # small in absolute terms (8M entries = 64 MB); everything wider takes the hash route.
        # offsets and codes at 4 bytes (2026-09-26): the range is under 8M, so both fit, and the prep
        # narrows the codes to u32 anyway -- the int64 offsets and int64 codes were 1.6 GB per 100M rows.
        # (A u32 column's offsets stay int64: its minimum need not fit an int32.)
        off = data.astype(np.int32 if (data.dtype.itemsize <= 2 or data.dtype == np.int32) else np.int64)
        off -= lo                                  # in place: 'astype - lo' held two arrays
        cnt = np.bincount(off, minlength=rng)
        present = np.flatnonzero(cnt)
        uniq = (present + lo).astype(data.dtype)
        lut = np.empty(rng, np.uint32); lut[present] = np.arange(present.size, dtype=np.uint32)
        return uniq, lut[off]
    if N > 2_000_000:
        # THE NEAR-UNIQUE GATE, by the number the sample CAN tell: a 1M sample of a 10%-distinct
        # column reads 63% distinct (it lies about V/N), but a near-unique column reads ~100% --
        # and only that case matters, because there the hash table of N keys loses to the sort
        # (WatchID: 68s -> 80s without the gate). >= 95% distinct in the sample -> np.unique.
        smp = data[np.random.default_rng(0).integers(0, N, 1_000_000)]
        if pd.Series(smp).nunique() >= 950_000:
            return np.unique(data, return_inverse=True)
    inv, uniq = pd.factorize(pd.Series(data), sort=False)
    uniq = np.asarray(uniq, dtype=data.dtype)
    order = np.argsort(uniq, kind='stable')
    rank = np.empty(order.size, np.int64); rank[order] = np.arange(order.size, dtype=np.int64)
    return uniq[order], rank[inv]


def _encode_column(col):
    """Return (dtype, has_null, V, uniq_value_bytes_list, codes[N], mode_is_string). codes: int64, or
    u32 when the integer dictionary's bincount route made them (the prep narrows to u32 either way)."""
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
            uniq, inv = np.unique(iv, return_inverse=True); codes = np.asarray(inv, dtype=np.int64)   # no null: the inverse IS the codes (a copy was 800 MB)
        valb = [struct.pack('<q', int(v)) for v in uniq]
    elif dtype in (0, 2):
        if has_null:
            nn = data[~null_mask]
            uniq, inv = _int_dictionary(nn) if dtype == 0 else np.unique(nn, return_inverse=True)
            codes[~null_mask] = inv; codes[null_mask] = len(uniq)
        else:
            uniq, inv = _int_dictionary(data) if dtype == 0 else np.unique(data, return_inverse=True)
            codes = inv if inv.dtype.kind in 'iu' else np.asarray(inv, dtype=np.int64)   # u32 from the bincount route: kept
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
            codes = np.asarray(inv, dtype=np.int64)
        valb = [to_b(u) for u in uniq]
    V = len(valb) + has_null
    return dtype, has_null, V, valb, codes, aux, uniq

def _pack_codes(codes, bits):
    """bit-pack in CHUNKS: the N x bits intermediate was 16 GB for 100M x 20-bit codes
    (measured: the true peak of a string column's encode, not the strings)"""
    # the u64 widening is done per chunk: a whole-column np.asarray(codes, uint64) was 800 MB for a
    # 100M-row flag column whose codes arrive as u32 (2026-09-26, the flag columns' 4.8 GB peak)
    codes = np.asarray(codes)
    n = codes.size
    if n == 0: return b''
    # ~8M bit-cells per chunk, as the comment always said: '(1 << 23) // bits * 8' was 64M cells -- a
    # 1-bit column widened 64M rows at a time into three u64 matrices (1.5 GB). Any multiple of 8 rows
    # packs to whole bytes, so the chunk size never changes the output.
    step = max(8, (1 << 23) // max(1, bits))
    step -= step % 8                                        # chunk rows x bits must be a multiple of 8 bits
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    out = bytearray()
    for lo in range(0, n, step):
        part = codes[lo:lo + step].astype(np.uint64)
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
        # A COLUMN IS JUDGED IN ITS OWN WIDTH (2026-09-26): a 2-byte flag column was widened to int64
        # (800 MB) and again for a bincount (800 MB) only to be declined as narrow -- the prep's peak.
        # Up to 4 bytes the values are the same numbers in their own width; the int64 copy is made
        # only for a column that reaches the codec. (8-byte columns widen as before: a uint64 wraps.)
        dtype = 0; aux = 0
        iv = data if data.dtype.itemsize <= 4 else data.astype(np.int64, copy=False)
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
            # narrow: the dict modes win. (This counted the distinct values with a bincount and
            # declined at <= 65536 -- a span under 65536 cannot hold more, so the span decides.)
            return None
        else:
            # THE CARDINALITY LAW: judge by DISTINCT COUNT, not span -- CounterID (6,506 values
            # spanning millions) became a 100M-value sequence under the span-only guard (2026-09-14)
            samp = iv[:: max(1, iv.size // 4_000_000)]
            if np.unique(samp).size <= 65536 and samp.size >= 65536:
                return None
    iv = iv.astype(np.int64, copy=False)              # the codec's width (no copy for int64 / clocks)
    blob = wdb_seqcodec.encode(iv, max_exc_frac=0.2)  # fire only on clear wins (>=80% conform)
    if blob is None:
        return None
    if not np.array_equal(wdb_seqcodec.decode(blob), iv):
        return None                                   # safety: never emit a lossy mode-4
    # THE IDENTITY LAW (2026-09-29): a mode-4 column's codes ARE row positions (V = N). The general
    # GROUP BY learned to group mode 4 by value, but reads keep arriving that take equal codes for
    # equal values (affinegroup, pairtop, cdgroup: counts of 1 where DuckDB said 2) -- so a sequence
    # is emitted only when no value repeats, and codes always identify values. Strictly monotone is
    # distinct for free; otherwise one exact check, paid only by a column that would become a
    # sequence. WDB_SEQ_REPEATS_OK=1 (the suite's machinery toys) keeps the old admission.
    if iv.size > 1 and _os.environ.get('WDB_SEQ_REPEATS_OK') != '1':
        d9 = np.diff(iv)
        if not ((d9 > 0).all() or (d9 < 0).all()):
            s9 = np.sort(iv)
            if (s9[1:] == s9[:-1]).any():
                return None
            del s9
        del d9
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
        if FC3_DICT:
            aux |= 0x80                                 # bit7 = as THE THREE STREAMS
    return dict(nm=nm, dtype=dtype, has_null=has_null, V=V, valb=valb,
                codes=codes.astype(np.uint32 if V < (1 << 32) else np.uint64, copy=False), aux=aux, uniq=uniq, bits=bits, mode=mode)   # THE NARROW-CODES LAW: codes travel at the narrowest width (no copy when they already do)

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


class _ByteVals:
    """THE DICTIONARY STAYS ONE BUFFER (2026-09-27): the sorted text dictionary in Arrow's own layout --
    one byte buffer + offsets -- standing in for the list of Python bytes objects the text prep used
    to build (URL: 18.3M objects, 4+ GB and ~15 s) only for the front-coder to join them back into
    one buffer. It reads like a sequence of bytes (len, index, iterate, compare) for the small
    consumers; _front_code takes .buf and .offs directly -- the very arrays it built before."""
    def __init__(self, arr):
        import pyarrow as pa
        self.arr = arr                                      # keeps Arrow's buffers alive
        n = len(arr)
        bufs = arr.buffers()
        odt = np.int64 if (pa.types.is_large_binary(arr.type) or pa.types.is_large_string(arr.type)) else np.int32
        o = np.frombuffer(bufs[1], dtype=odt)[arr.offset:arr.offset + n + 1] if n else np.zeros(1, odt)
        o0 = int(o[0]); o1 = int(o[-1])
        self.offs = o.astype(np.int64) - o0
        self.buf = (np.frombuffer(bufs[2], dtype=np.uint8)[o0:o1] if (bufs[2] is not None and o1 > o0)
                    else np.zeros(0, np.uint8))

    def __len__(self):
        return len(self.offs) - 1

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        return self.buf[self.offs[i]:self.offs[i + 1]].tobytes()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other):
        if isinstance(other, _ByteVals):
            return np.array_equal(self.offs, other.offs) and np.array_equal(self.buf, other.buf)
        try:
            return len(self) == len(other) and all(a == b for a, b in zip(self, other))
        except TypeError:
            return NotImplemented

    __hash__ = None


def _front_code(valb):
    """THE FRONT-CODER IN NUMBA: the dictionary as one byte buffer + offsets; a compiled loop
    computes each value's common prefix with its predecessor and emits <HH cp suflen>+suffix
    with a restart every R values -- byte-identical to _front_code_py (15s of Python per 2.7M
    URLs -> ~0.2s). Falls back to Python without numba."""
    try:
        import numba
    except Exception:
        return _front_code_py(list(valb))
    if isinstance(valb, _ByteVals):
        # the buffer and offsets ARE what the join below would build: no 18M-object pass, no copy
        n = len(valb)
        if n == 0:
            return b'', np.zeros(0, dtype=np.uint32)
        out, rst, total = _front_code_jit(valb.buf, valb.offs, n, R)
        return out[:total].tobytes(), rst
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
        I2CH = int(os.environ.get('WDB_I2CHUNK', str(1 << 13)))      # values per chunk (64 KB raw: the toll floor)
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
            bounds = []
            for j in range(n_chunks):
                b0 = int(rst[j*BPC])
                b1 = int(rst[(j+1)*BPC]) if (j+1)*BPC < nb else len(fc)
                bounds.append((b0, b1)); ustart.append(b0)
            # THE PARALLEL FRAMES: the dictionary chunks are independent zstd frames and zstd
            # releases the GIL -- 575 frames of Title took 23s on one core (the string tail)
            from concurrent.futures import ThreadPoolExecutor
            _lvl9 = getattr(zc, '_wdb_level', ZSTD_LEVEL)
            def _cz9(b):
                return zstd.ZstdCompressor(level=_lvl9).compress(fc[b[0]:b[1]])
            if p['aux'] & 0x80:
                # THE THREE STREAMS: split each chunk into headers, mask and text, one frame each
                import wdb_kernels as _WK3
                def _cz3(jb):
                    j, (b0, b1) = jb
                    n = min(CH, V - j * CH)
                    a = np.frombuffer(fc, dtype=np.uint8, count=b1 - b0, offset=b0)
                    hdr = np.empty(4 * n, np.uint8); text = np.empty(b1 - b0 - 4 * n, np.uint8)
                    mask = np.zeros(((text.size + 63) // 64) * 8, np.uint8)
                    tl = int(_WK3.fc3_split(a, np.int64(n), hdr, text, mask))
                    assert tl == text.size, ('fc3 split', p['nm'], j, tl, text.size)
                    cz = zstd.ZstdCompressor(level=_lvl9)
                    return cz.compress(hdr.tobytes()), cz.compress(mask.tobytes()), cz.compress(text.tobytes())
                with ThreadPoolExecutor(max_workers=min(16, max(1, os.cpu_count() or 4))) as _ex9:
                    trip = list(_ex9.map(_cz3, list(enumerate(bounds))))
                out += struct.pack('<H', R) + struct.pack('<I', CH) + struct.pack('<I', n_chunks)
                out += struct.pack('<I', nb) + rst.tobytes() + struct.pack('<I', len(fc))
                out += np.array(ustart, dtype=np.uint32).tobytes()
                for k in range(3):
                    out += np.array([len(t3[k]) for t3 in trip], dtype=np.uint32).tobytes()
                for k in range(3):
                    for t3 in trip: out += t3[k]
                return out
            with ThreadPoolExecutor(max_workers=min(16, max(1, os.cpu_count() or 4))) as _ex9:
                frames = list(_ex9.map(_cz9, bounds))
            czl = [len(fr) for fr in frames]
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

def _hash_cols():
    """THE OPERATOR'S HASH DECLARATION (bin/wdb load --hash C,..): columns stored as tag 20."""
    return set(c for c in os.environ.get('WDB_HASH_COLS', '').split(',') if c)


def _e20_section(arr, bits):
    """THE BACK-REFERENCE (tag 20, Jackson 2026-09-24), for operator-declared hash columns.
    Layout: [20][bits u8][BR u32][nb u32][P u64] + boff[nb+1] i64 (byte starts of the blocks in the
    payload) + payload[P] + 8 zero bytes (an 8-byte load never leaves the section). See
    wdb_kernels.e20_write for the row format."""
    import wdb_kernels as _WK20
    BR = int(os.environ.get('WDB_E20_BR', '65536'))
    a = np.ascontiguousarray(arr, dtype=np.int64)
    N = a.size
    o = np.argsort(a, kind='stable')
    s = a[o]
    same = np.zeros(N, np.bool_)
    same[1:] = s[1:] == s[:-1]
    del s
    idx = np.flatnonzero(same)
    prev = np.full(N, -1, np.int64)
    prev[o[idx]] = o[idx - 1]
    del o, idx, same
    r = np.arange(N, dtype=np.int64)
    gap = np.where((prev >= 0) & (prev // BR == r // BR), r - prev, 0)
    del prev
    k = np.zeros(N, np.int64)
    gg = gap >> 1
    while True:
        m = gg > 0
        if not m.any():
            break
        k[m] += 1
        gg >>= 1
    rowbits = np.where(gap > 0, 5 + k, 1 + bits)
    del k, gg, r
    nb = (N + BR - 1) // BR
    bbits = np.add.reduceat(rowbits, np.arange(0, N, BR)) if N else np.zeros(0, np.int64)
    boff = np.zeros(nb + 1, np.int64)
    np.cumsum((bbits + 7) // 8, out=boff[1:])
    P = int(boff[-1])
    head = bytes([20, bits]) + struct.pack('<IIQ', BR, nb, P) + boff.tobytes()
    buf = np.zeros(len(head) + P + 8, np.uint8)
    buf[:len(head)] = np.frombuffer(head, np.uint8)
    _WK20.e20_write(a, gap, np.int64(BR), np.int64(bits), boff, np.int64(len(head)), buf)
    return buf.tobytes()


E19_SLACK = float(os.environ.get('WDB_E19_SLACK', '0.05'))   # enc 19 may cost this many more bytes
                                                              # than the inflating dress it replaces


def _e19_candidate(arr, bits, seal):
    """THE BLOCK DICTIONARIES (tag 19), or None when no block size fits under `seal` bytes.
    Layout: [19][gbits u8][BR u32][nb u32][P u64][D u64] + lb[nb] u8 + gw[nb] u8 + dcnt[nb] u32
    + poff[nb+1] i64 + doff[nb+1] i64 (u64-word offsets) + pointer words[P+1] + dict words[D+1]
    (one trailing zero word each: a straddling read never leaves the section)."""
    import wdb_kernels as _WK19
    a = np.ascontiguousarray(arr, dtype=np.int64)
    best = None
    for BR in (16384, 65536):                        # the census: near-unique columns take 65536,
        nb = (a.size + BR - 1) // BR                 # the mid-V ones 16384; the plan sizes both
        lb = np.empty(nb, np.uint8); gw = np.empty(nb, np.uint8); dc = np.empty(nb, np.uint32)
        pwn = np.empty(nb, np.int64); dwn = np.empty(nb, np.int64)
        _WK19.e19_plan(a, np.int64(BR), np.int64(bits), lb, gw, dc, pwn, dwn)
        poff = np.zeros(nb + 1, np.int64); np.cumsum(pwn, out=poff[1:])
        doff = np.zeros(nb + 1, np.int64); np.cumsum(dwn, out=doff[1:])
        size = 26 + nb * 6 + (nb + 1) * 16 + (int(poff[-1]) + 1) * 8 + (int(doff[-1]) + 1) * 8
        if size <= seal and (best is None or size < best[0]):
            best = (size, BR, nb, lb, gw, dc, poff, doff)
    if best is None:
        return None
    size, BR, nb, lb, gw, dc, poff, doff = best
    R = _e19_shelf_count(int(doff[-1]) * 8)
    if R >= 2:
        sec = _e19_shelved(a, bits, BR, nb, lb, gw, dc, poff, R)
        if sec is not None and len(sec) <= seal:
            return sec
    pw = np.zeros(int(poff[-1]) + 1, np.uint64); dw = np.zeros(int(doff[-1]) + 1, np.uint64)
    _WK19.e19_write(a, np.int64(BR), np.int64(bits), lb, gw, poff, doff, pw, dw)
    sec = (bytes([19, bits]) + struct.pack('<IIQQ', BR, nb, int(poff[-1]), int(doff[-1]))
           + lb.tobytes() + gw.tobytes() + dc.tobytes() + poff.tobytes() + doff.tobytes()
           + pw.tobytes() + dw.tobytes())
    assert len(sec) == size, (len(sec), size)
    return sec


E19_SHELF_BYTES = 256 << 10     # THE SHELVES: one code range's pieces of every label, ~256 KB a shelf


def _e19_shelf_count(label_bytes):
    """how many shelves the labels are cut into: WDB_E19_SHELVES=0 keeps the labels by block, a number
    forces that many (tests), auto sizes shelves at ~256 KB; under 2 shelves the labels stay by block"""
    s = os.environ.get('WDB_E19_SHELVES', 'auto')
    if s == '0':
        return 0
    if s not in ('', 'auto'):
        return max(2, int(s))
    R = label_bytes // E19_SHELF_BYTES
    return int(min(R, 4096)) if R >= 2 else 0


def _e19_shelved(a, bits, BR, nb, lb, gw, dc, poff, R):
    """THE SHELVES (Jackson, 2026-09-27): tag 19 with its block labels laid out by code range.
    Layout: [19][gbits | 0x80][BR u32][nb u32][P u64][D u64] + lb[nb] u8 + gw[nb] u8 + dcnt[nb] u32
    + poff[nb+1] i64 + [R u32][wb u32][W u64] + SW[R+1] i64 + pre[(R+1) x nb] u32 + soff[R x (nb+1)] u32
    + pointer words[P+1] + shelf words[D+1]. Shelf r holds every block's entries with code in
    [r*W, (r+1)*W): first as (code - r*W) in wb bits, then gaps at gw[b] (wdb_kernels.e19s_*).
    None when a table would not fit its u32."""
    import wdb_kernels as _WK19
    LS = np.zeros(nb + 1, np.int64); np.cumsum(dc, out=LS[1:])
    L = np.empty(int(LS[-1]), np.uint32)
    pw = np.zeros(int(poff[-1]) + 1, np.uint64)
    _WK19.e19s_write(a, np.int64(BR), lb, poff, pw, LS, L)
    W = -(-(int(L.max()) + 1) // R) if L.size else 1
    wb = int(W - 1).bit_length()
    cnt = np.zeros((R + 1, nb), np.int64)
    _WK19.e19s_counts(L, LS, np.int64(W), cnt)
    pre = np.cumsum(cnt, axis=0)
    cc = pre[1:] - pre[:-1]
    nbits = np.where(cc > 0, wb + (cc - 1) * gw.astype(np.int64)[None, :], 0)
    soff = np.zeros((R, nb + 1), np.int64); np.cumsum(nbits, axis=1, out=soff[:, 1:])
    if soff.max() >= (1 << 32) or pre.max() >= (1 << 32):
        return None
    SW = np.zeros(R + 1, np.int64); np.cumsum((soff[:, nb] + 63) // 64 + 1, out=SW[1:])   # +1: a straddle stays inside
    pre = pre.astype(np.uint32); soff = soff.astype(np.uint32)
    D = int(SW[-1])
    dw = np.zeros(D + 1, np.uint64)
    _WK19.e19s_shelve(L, LS, np.int64(W), np.int64(wb), gw, pre, soff, SW, dw)
    del L
    return (bytes([19, bits | 0x80]) + struct.pack('<IIQQ', BR, nb, int(poff[-1]), D)
            + lb.tobytes() + gw.tobytes() + dc.tobytes() + poff.tobytes()
            + struct.pack('<IIQ', R, wb, W) + SW.tobytes() + pre.tobytes() + soff.tobytes()
            + pw.tobytes() + dw.tobytes())


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
    # ONE SIGNED view of the stream, shared by every candidate (six copies of 800 MB were alive at once),
    # at the narrowest signed width that holds the largest code (2026-09-26): int64 was 800 MB of a
    # 100M-row flag column whose codes are 0 and 1. Signed, so a neighbour difference is exact. Wide
    # codes stay int64: tags 18/19/20 need int64 and would copy a narrower view whole.
    _mx = int(np.max(codes)) if np.size(codes) else 0
    _sw = np.int8 if _mx < (1 << 7) else (np.int16 if _mx < (1 << 15) else np.int64)
    arr = np.asarray(codes, dtype=_sw)
    # THE FLAG SKIPS THE CONTEST (Jackson, 2026-09-26): a column the operator declared a hash is
    # stored as tag 20 whatever the size contest picks -- the only exception is a staircase, which
    # can only win when the codes climb from 0 by steps of 0 or 1. When they do not, every other
    # dress (zstd, sparse, tiered, bitpack-plus, frames, packed frames, block dictionaries...) was
    # built only to be thrown away: URLHash spent 96 s and RefererHash 98 s serializing.
    if nm is not None and codes.size and 1 <= bits <= 32 and nm in _hash_cols():
        d20 = np.diff(arr)
        climbs = int(arr[0]) == 0 and (d20.size == 0 or (int(d20.min()) >= 0 and int(d20.max()) <= 1))
        del d20
        if not climbs:
            return _e20_section(arr, bits)
    stair = None
    if arr.size:
        d = np.diff(arr)
        if int(arr[0]) == 0 and (d.size == 0 or (int(d.min()) >= 0 and int(d.max()) <= 1)):
            steps = (np.nonzero(d)[0] + 1).astype(np.int64)      # rows where the code ticks +1
            gaps = np.diff(np.concatenate(([0], steps)))
            gbits = max(1, int(gaps.max()).bit_length()) if gaps.size else 1
            pay = _pack_codes(gaps, gbits) if gaps.size else b''
            stair = bytes([2, gbits]) + struct.pack('<I', steps.size) + pay
            del steps, gaps, pay
    d = None                                          # the diff is dead past the staircase test
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
    cn8 = np.bincount(arr) if codes.size else np.zeros(0)
    dflt = int(cn8.argmax()) if cn8.size else 0
    if codes.size and cn8.size and cn8[dflt] * 2 > codes.size:               # majority default: the only shape it fits
        pres = (arr != dflt)
        lits = np.asarray(codes)[pres]
        pb = np.packbits(pres)
        CK = 65536
        nck = (codes.size + CK - 1) // CK
        per = np.add.reduceat(pres.astype(np.uint8),
                              np.arange(0, codes.size, CK))
        ck = np.zeros(nck, dtype=np.uint64)
        if nck > 1:
            ck[1:] = np.cumsum(per[:-1]).astype(np.uint64)
        litp = _pack_codes(lits, bits) if lits.size else b''
        sparse = (bytes([8, bits]) + struct.pack('<IQQ', dflt, lits.size, codes.size)
                  + pb.tobytes() + ck.tobytes() + litp)
        pres = lits = pb = per = ck = litp = None      # sized: its working arrays go
    # tag 9 = TIERED dress (rule eleven, Jackson's design): the dominant value
    # is NOTHING (absence bitmap), then within the typed remainder the most
    # common code is ONE BIT, tiering down; the small tail rides u8. Elected
    # for low-V columns with concentrated histograms; zero-pop serving.
    tiered = None
    if codes.size and cn8.size and cn8[dflt] * 2 > codes.size and bits <= 8 \
            and cn8.size <= 256:
        arr9 = arr
        pres9 = arr9 != dflt
        pb9 = np.packbits(pres9)
        CK = 65536
        nck9 = (codes.size + CK - 1) // CK
        per9 = np.add.reduceat(pres9.astype(np.uint8), np.arange(0, codes.size, CK)).astype(np.int64)
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
        pres9 = pb9 = per9 = ck9 = rem = planes = bit9 = cnr = None   # sized: its working arrays go
    # tag 10 = SEGMENTED BITPACK-PLUS (Jackson's dress): 4096-row blocks,
    # each electing bitpack or run-tokens by the profit formula -- runs
    # compress only where run_len*bits beats the token, so the whole
    # column is structurally never worse than bitpack (+1B/block).
    bplus = None
    if codes.size and bits <= 16:
        arrA = arr
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
            payloadA = dirA = None
        # sized: its working arrays go. The election below needs only the MEAN run length, which
        # is rows / runs exactly (the run lengths sum to the rows), so the run starts go too.
        nrunsA = stA.size
        stA = bndA = blk_firstA = blk_lastA = nruns_bA = rows_bA = run_modeA = None
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
            arr9 = arr
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
        if arrA.size / nrunsA >= 5.0:            # mean run >= 5: locality is real
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
        a13 = arr
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
    blocked = None
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
        a = frames = None                        # the narrow copy and the frame list go
        if blocked is not None:
            best = blocked
    # tag 18 = PACKED FRAMES (Jackson's question: "bitpack, then zstd the 1s and 0s").
    # Per 65536-row frame: LE bit-pack the codes (no byte padding per row), then one
    # zstd frame. Measured on cbdb: Title 224 MB vs 226 (u32 frames) -- the packing
    # removes the zero bytes zstd was paying tokens to skip. Wide codes only (>= 17
    # bits: below that the u16 rows are already tight). Elected on size alone against
    # the plain dresses (zstd / blocked / bitpack); a point read inflates a frame
    # PREFIX (the stream stops at its highest row).
    if 17 <= bits <= 32 and codes.size and (best is zsec or best is packed
                                            or best is blocked):
        from wdb_kernels import pk32_pack as _pk32
        BR18 = 65536
        a18 = np.ascontiguousarray(arr, dtype=np.int64)
        cx18 = zstd.ZstdCompressor(level=CODE_ZSTD_LEVEL)
        fr18 = []
        for i in range(0, a18.size, BR18):
            ch = a18[i:i + BR18]
            out = np.zeros((ch.size * bits + 7) // 8 + 8, np.uint8)
            _pk32(ch, bits, out)
            fr18.append(cx18.compress(out.tobytes()))
        o18 = np.zeros(len(fr18) + 1, dtype=np.uint32)
        np.cumsum([len(f) for f in fr18], out=o18[1:])
        cand18 = (bytes([18, bits]) + struct.pack('<II', BR18, len(fr18))
                  + o18.tobytes() + b''.join(fr18))
        # THE BITPACK GUARD (Jackson): over the plain bitpack -- whose point reads are bit
        # arithmetic on the mmap, no inflate ever -- the packed frames must win by 10%, not by
        # a hair (HID: 337.5 -> 337.0 MB, 0.15%, was not worth a frame inflate per point read).
        # Over zstd / blocked frames, which already pay the inflate, strictly smaller elects.
        seal18 = 0.90 * len(best) if best is packed else len(best)
        if len(cand18) < seal18 or os.environ.get('WDB_E18_FORCE'):
            best = cand18
    # tag 19 = THE BLOCK DICTIONARIES (Jackson, 2026-09-23: pointer compression one level down --
    # per block, the distinct codes present, once, sorted and gap-coded; per row, a pointer into
    # its block's list at the block's own width). Decode is one jump per row, no inflate.
    # THE ELECTION (Jackson's general rule): it replaces an INFLATING dress (zstd 1 / blocked 3 /
    # packed frames 18) when its bytes are within E19_SLACK (5%) of that dress's. Measured on
    # cbdb: UserID 246.8 vs 246.0 MB at 32 vs 122 ms per full decode; RegionID 1.19x -> stays.
    _f19 = bool(os.environ.get('WDB_E19_FORCE'))
    if 1 <= bits <= 32 and codes.size and (_f19 or (bits >= 9 and codes.size >= (1 << 20)
                                                    and best[0] in (1, 3, 18))):
        cand19 = _e19_candidate(arr, bits, float('inf') if _f19 else len(best) * (1.0 + E19_SLACK))
        if cand19 is not None:
            best = cand19
    # tag 14 = FIELD PLANES (Jackson's dress): dates decompose to y/m/d u8
    # planes, each its own zstd stream -- the calendar's internal correlation
    # compresses BELOW naive entropy (82.7 vs 94.1MB measured on l_shipdate),
    # and band predicates later read only the fields they constrain. Types
    # NOMINATE (the caller passes date_vals only for dt==3), measurements
    # SIZE (year width from the real range), the size election alone ELECTS.
    if date_vals is not None and codes.size:
        try:
            dv = np.asarray(date_vals, dtype=np.int64)[arr]
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
        arr16 = arr
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
            # counted in int64 by the ufunc itself: em.astype(int64) was an 800 MB copy of a bool mask
            eb = np.add.reduceat(em, np.arange(0, Nr, BR5), dtype=np.int64) if Nr else np.zeros(0, dtype=np.int64)
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
                    # the 0/1 marks are bytes, the per-block counts int64 (the ufunc counts in the
                    # asked dtype): the int64 marks were 800 MB for 100M rows
                    e1b = np.add.reduceat(em1, np.arange(0, Nr, BR5b), dtype=np.int64)
                    full_e2 = np.zeros(Nr, dtype=np.uint8)
                    idx1 = np.flatnonzero(em1)
                    full_e2[idx1[em2pos]] = 1
                    e2b = np.add.reduceat(full_e2, np.arange(0, Nr, BR5b), dtype=np.int64)
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
                full_e2 = idx1 = em1 = em2pos = wb = patches6 = lutw = e1b = e2b = e6 = None
            # the buckets' working arrays die with their election (full_e2 alone was 800 MB held
            # to the end of the contest)
            nib = em = patches = nibp = pk = lut = cn5 = eb = e5 = None
    # tag 17 = RAW PACKED CODES (Jackson's deal law). Nomination: the zstd
    # frame stream won so far, V >= 3 (binaries are their own terminal dress),
    # and the column is big enough that the toll is real (small fixtures stay
    # deterministic). Election: S > T on MEASURED terms -- S the symmetric
    # shrink ratio zstd buys vs packed, T the symmetric slowdown its
    # decompression charges. zstd keeps the column only when it shrinks it
    # more than it slows it.
    if best and best[0] == 3 and bits >= 2 and bits <= 16 and codes.size >= (1 << 22):
        arr17 = arr
        # IN CHUNKS OF 1M ROWS (2026-09-26): the one-byte-per-bit array was rows x bits bytes (1.3 GB
        # for CounterID) plus an 800 MB int64 temporary per bit -- the job's peak. A chunk of a
        # multiple of 8 rows packs to whole bytes, so the joined chunks are the same bytes.
        pk17p = []
        for lo17 in range(0, arr17.size, 1 << 20):
            ch17 = arr17[lo17:lo17 + (1 << 20)]
            tb17 = np.zeros(ch17.size * bits, dtype=np.uint8)
            for k17 in range(bits):
                tb17[k17::bits] = (ch17 >> k17) & 1
            pk17p.append(np.packbits(tb17, bitorder='little').tobytes())
        del tb17, ch17
        cand17 = (bytes([17, bits]) + struct.pack('<I', arr17.size)
                  + b''.join(pk17p) + b'\x00\x00')
        del pk17p
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
    # tag 20 = THE BACK-REFERENCE: the OPERATOR's ruling, not an election -- a column declared a
    # hash (bin/wdb load --hash) is stored this way whatever the size contest would pick.
    if nm is not None and nm in _hash_cols() and codes.size and 1 <= bits <= 32 and best is not stair:
        best = _e20_section(arr, bits)
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
    valb = np.array(list(p['valb']) + [b''], dtype=object)[:-1]   # object array of distinct byte values
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


def _arrow_string_prep(nm, chunked, perm=None):
    """THE ARROW LAW: a string column is dictionary-encoded IN ARROW MEMORY (no 100M Python
    objects -- measured 45 GB and 387s for one 100M-row column the pandas way), its dictionary
    sorted with arrow, codes remapped with numpy. Returns a prep dict like _prep_column's
    (mode 0/1 strings) or None when the column is not a string column.
    chunked may be a one-element list (the caller hands over ownership: the text is freed as soon
    as it is dictionary-encoded). perm: the cluster order, applied to the CODES."""
    import pyarrow as pa, pyarrow.compute as pc
    if isinstance(chunked, list):
        chunked = chunked.pop()
    t = chunked.type
    if not (pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_binary(t) or pa.types.is_large_binary(t)):
        return None
    if isinstance(chunked, pa.ChunkedArray):
        if pa.types.is_string(chunked.type):
            # THE LARGE-STRING CAST: 100M titles exceed arrow's 2 GB 'string' offset space; the concat
            # raised 'offset overflow', the exception was swallowed, and every big string column fell
            # to the pandas/Python-object path -- 170s and 57 GB per column (the string tail, 2026-09-14).
            # (Per-chunk dictionary_encode + unify_dictionaries was measured: 5+ minutes on one core.)
            chunked = pc.cast(chunked, pa.large_string())
    # THE TEXT FLOWS THROUGH IN ITS ROW GROUPS (Jackson, 2026-09-27: hold less in RAM at once). The row
    # groups were glued into one array first -- a second full copy of the text beside the first (URL:
    # 8.8 GB) only to be hashed once. Arrow's dictionary_encode on the chunked column keeps ONE memo
    # across the chunks: every chunk carries the same final dictionary, in the same first-seen order
    # the glued array gave (measured on SearchPhrase and URL: identical indices and dictionary). If a
    # build ever hands back per-chunk dictionaries, the glued road runs as before.
    idx = dct = None
    if isinstance(chunked, pa.ChunkedArray) and chunked.num_chunks > 1:
        de = pc.dictionary_encode(chunked)
        dct = de.chunk(de.num_chunks - 1).dictionary
        _db9 = dct.buffers()
        _one9 = all(len(c.dictionary) == len(dct) and c.dictionary.buffers()[-1] is not None
                    and _db9[-1] is not None and c.dictionary.buffers()[-1].address == _db9[-1].address
                    for c in de.chunks)
        if _one9:
            idx = pa.concat_arrays([c.indices for c in de.chunks])
        else:
            dct = None
        de = None                                           # the per-chunk indices go; dct holds the dictionary
    if idx is None:
        if isinstance(chunked, pa.ChunkedArray):
            chunked = pa.concat_arrays(chunked.chunks) if chunked.num_chunks > 1 else (
                chunked.chunk(0) if chunked.num_chunks else pa.array([], type=chunked.type))
        de = pc.dictionary_encode(chunked)                  # indices (int32) + dictionary (unique, first-seen order)
        dct = de.dictionary; idx = de.indices
    null_mask = None; has_null = 0
    if chunked.null_count:
        null_mask = np.asarray(pc.is_null(chunked).to_numpy(zero_copy_only=False), dtype=bool); has_null = 1
    nrows = len(chunked)
    del chunked                                             # the row text is dead: only the dictionary lives on
    try:
        pa.default_memory_pool().release_unused()           # hand the text's pages back (the pool had kept them)
    except Exception:
        pass
    order = pc.sort_indices(dct).to_numpy()                 # dictionary in sorted order
    rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
    codes = np.zeros(nrows, np.uint32 if len(order) < (1 << 31) else np.int64)
    raw = idx.to_numpy(zero_copy_only=False)
    if null_mask is not None:
        nn = ~null_mask
        codes[nn] = rank[np.asarray(raw[nn], dtype=np.int64)]; codes[null_mask] = len(order)
    else:
        codes[:] = rank[np.asarray(raw, dtype=np.int64)]
    de = idx = raw = rank = None
    dbox = [dct]; dct = None
    return _string_prep_tail(nm, dbox, order, codes, has_null, perm)


def _string_prep_tail(nm, dbox, order, codes, has_null, perm):
    """the common end of both text preps: the codes into the cluster order, the dictionary in sorted
    order as bytes, the mode. dbox: a one-element list holding the (unsorted) arrow dictionary -- the
    caller hands over ownership so it is freed here once the sorted copy exists."""
    import pyarrow as pa, pyarrow.compute as pc
    dct = dbox.pop()
    if perm is not None:
        codes = codes[np.asarray(perm)]                     # the cluster order, on 4-byte codes
    sorted_dict = pc.take(dct, pa.array(order))
    # THE DICTIONARY COMES OUT AS BYTES (2026-09-27): read as text it became 18M Python str objects
    # (URL: +6 GB, 11 s) and then 18M bytes objects beside them (+4 GB, 4 s) -- the job's peak. Viewed
    # as binary (the same buffers, no copy) it comes out as the bytes directly: the very bytes the
    # utf-8 encode gave back for valid text, and the raw bytes for any that is not.
    if pa.types.is_large_string(sorted_dict.type):
        sorted_dict = sorted_dict.cast(pa.large_binary())
    elif pa.types.is_string(sorted_dict.type):
        sorted_dict = sorted_dict.cast(pa.binary())
    dct = None
    # ... and stays ONE buffer (2026-09-27): _ByteVals over the sorted binary array, no Python objects
    if os.environ.get('WDB_TEXT_BYTEVALS', '1') != '0':
        valb = _ByteVals(sorted_dict)
    else:
        valb = [(v if isinstance(v, bytes) else v.encode('utf-8', 'surrogatepass')) for v in sorted_dict.to_pylist()]
    sorted_dict = None
    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    V = len(valb) + has_null
    bits = max(1, int(np.ceil(np.log2(max(V, 2)))))
    mode = 1 if (V - has_null) > FC_THRESHOLD else 0
    uniq = None
    aux = 0
    if mode == 1 and CHUNK_DICT:
        aux |= 0x40      # THE CHUNK LAW: a big front-coded dictionary is written as independent zstd frames
                         # (CHUNK_DICT_VALS values each) so a lookup decompresses one frame, not 6M values
                         # (SearchPhrase <> '' on gov7: 0.34s = one 6M-value frame per probe; the reference 0.05s)
        if FC3_DICT:
            aux |= 0x80  # as THE THREE STREAMS
    return dict(nm=nm, dtype=1, has_null=has_null, V=V, valb=valb, codes=codes, aux=aux, uniq=uniq, bits=bits, mode=mode)


TEXT_SLICES = int(os.environ.get('WDB_TEXT_SLICES', '16'))      # 0 = the whole-column road
TEXT_SLICE_THREADS = int(os.environ.get('WDB_TEXT_SLICE_THREADS', '4'))


def _arrow_string_prep_sliced(nm, input_path, perm=None, slices=None, threads=None):
    """THE TEXT FLOWS THROUGH RAM IN SLICES (Jackson, 2026-09-27: work in RAM, never wait on the source,
    write the finished product). The column is never whole in memory: its row groups are cut into
    `slices` runs; `threads` threads each read one run and dictionary-encode it (a local dictionary +
    local indices), and the run's text is freed as soon as it is encoded -- reading one run overlaps
    hashing another. The local dictionaries are then merged by one more dictionary_encode, the merged
    dictionary is sorted, and every row's code is its value's rank in that sorted dictionary. The
    codes and the sorted dictionary depend only on WHICH values exist and WHICH row holds which, so
    they are the whole-column road's exactly (checked: URL, Title). Prototype on URL: 35.5 s / 13.4 GB
    against 37.6 s / 24.1 GB. Returns None (the caller takes the whole-column road) for a non-text
    column, fewer than 2 row groups, an all-null column, or dictionaries that are not one per chunk."""
    import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    from concurrent.futures import ThreadPoolExecutor
    pf = pq.ParquetFile(input_path)
    G = pf.num_row_groups
    t = pf.schema_arrow.field(nm).type
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        big = pa.large_string()
    elif pa.types.is_binary(t) or pa.types.is_large_binary(t):
        big = pa.large_binary()
    else:
        return None
    K = min(G, int(slices or TEXT_SLICES)); W = max(1, int(threads or TEXT_SLICE_THREADS))
    if K < 2:
        return None
    bounds = np.linspace(0, G, K + 1).astype(int)

    def _one_dict(de):
        """the one dictionary every chunk of an encoded chunked array carries, or None"""
        if de.num_chunks == 0:
            return None
        d = de.chunk(de.num_chunks - 1).dictionary
        if len(d) == 0:
            return d
        a = d.buffers()[-1]
        for c in de.chunks:
            b = c.dictionary.buffers()[-1]
            if len(c.dictionary) != len(d) or a is None or b is None or b.address != a.address:
                return None
        return d

    def one(s):
        tb = pq.ParquetFile(input_path).read_row_groups(list(range(bounds[s], bounds[s + 1])), columns=[nm], use_threads=False)
        col = tb.column(0); tb = None
        if col.type != big:
            col = pc.cast(col, big)
        n = len(col)
        de = pc.dictionary_encode(col); col = None
        d = _one_dict(de)
        if d is None:
            raise ValueError('per-chunk dictionaries')
        idx = pa.concat_arrays([c.indices for c in de.chunks]) if de.num_chunks else pa.array([], pa.int32())
        de = None
        nul = None
        if idx.null_count:
            nul = np.asarray(idx.is_null().to_numpy(zero_copy_only=False), dtype=bool)
            idx = pc.fill_null(idx, 0)
        li = np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int32)
        assert li.size == n
        return d, li, nul

    try:
        with ThreadPoolExecutor(max_workers=W) as ex:
            parts = list(ex.map(one, range(K)))
    except Exception:
        return None
    try:
        pa.default_memory_pool().release_unused()           # every run's text is gone
    except Exception:
        pass
    ldicts = [p[0] for p in parts]; lidx = [p[1] for p in parts]; lnul = [p[2] for p in parts]
    parts = None
    if sum(len(d) for d in ldicts) == 0:
        return None
    has_null = 1 if any(x is not None and x.any() for x in lnul) else 0
    g = pc.dictionary_encode(pa.chunked_array(ldicts, type=big))    # the merge: one dictionary for all runs
    gd = _one_dict(g)
    if gd is None:
        return None
    # the merged indices split by each run's dictionary LENGTH, not by the output's chunks: an empty run
    # (a row group of only nulls) leaves no chunk of its own, and the maps would slide one run over
    allm = np.concatenate([np.asarray(c.indices.to_numpy(zero_copy_only=False), dtype=np.int64) for c in g.chunks]) \
        if g.num_chunks else np.zeros(0, np.int64)
    lens = [len(d) for d in ldicts]
    assert allm.size == sum(lens), ('merge', nm, allm.size, sum(lens))
    cuts = np.concatenate(([0], np.cumsum(lens)))
    maps = [allm[cuts[i]:cuts[i + 1]] for i in range(len(lens))]
    g = None; ldicts = None; allm = None
    order = pc.sort_indices(gd).to_numpy()                 # the merged dictionary in sorted order
    V0 = len(order)
    rank = np.empty(V0, np.int64); rank[order] = np.arange(V0)
    nrows = sum(li.size for li in lidx)
    codes = np.zeros(nrows, np.uint32 if V0 < (1 << 31) else np.int64)
    o = 0
    for m, li, nul in zip(maps, lidx, lnul):
        n = li.size
        if m.size:
            codes[o:o + n] = rank[m[li]]
        if nul is not None:
            codes[o:o + n][nul] = V0                        # the null bin, as the whole-column road
        o += n
    maps = lidx = lnul = rank = None
    dbox = [gd]; gd = None
    return _string_prep_tail(nm, dbox, order, codes, has_null, perm)


_CASTS = {
    # DECLARED CASTS at ingest (the operator's ruling, never a guess): a uint16 day count is a
    # DATE, an int64 epoch-second count is a TIMESTAMP
    'date_days':      lambda a: np.asarray(a).astype(np.int64).astype('datetime64[D]'),
    'timestamp_s':    lambda a: np.asarray(a).astype(np.int64).astype('datetime64[s]'),
    'timestamp_ms':   lambda a: np.asarray(a).astype(np.int64).astype('datetime64[ms]'),
    'timestamp_us':   lambda a: np.asarray(a).astype(np.int64).astype('datetime64[us]'),
}


def _crash_point(name):
    """THE CRASH HARNESS: WDB_CRASH_AT=<name> kills the process here (SIGKILL: no cleanup,
    no finally) so recovery can be tested at every step of every write path."""
    if os.environ.get('WDB_CRASH_AT') == name:
        import signal
        print('CRASH POINT %s: dying' % name, flush=True)
        os.kill(os.getpid(), signal.SIGKILL)


def _own_hwm():
    """this process's own resident high-water mark in bytes (/proc/self/status VmHWM), or 0"""
    try:
        with open('/proc/self/status') as f:
            for ln in f:
                if ln.startswith('VmHWM:'):
                    return int(ln.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _column_job(input_path, nm, reader, cast=None, perm_path=None, N=None):
    """one column, start to blob, in a worker process (reads its own column: no table in RAM).
    perm_path: THE CLUSTER ORDER -- a row permutation every column gathers through, so the
    segment is time-ordered and the clock columns become STAIRCASES (free ordering for
    windows and ranges: the reference encode had it; the raw file order does not)."""
    prep = None
    # THE ENCODE CLOCK (2026-09-26): where one column's seconds go -- read, gather (the cluster
    # order), prep (dictionary / front-coding / layout), serialize (zstd) -- returned to the parent
    _tm = {}; _tk = [time.time()]
    def _mark(k):
        now = time.time(); _tm[k] = _tm.get(k, 0.0) + (now - _tk[0]); _tk[0] = now
    # THE JOB'S NOTE TO THE GOVERNOR: which process is cooking this column, and (text columns) when
    # the raw text is gone -- the governor sizes the room it keeps for this job from it
    _jd = os.environ.get('WDB_ENC_JOBDIR')
    def _note(kind):
        if _jd:
            try:
                with open(os.path.join(_jd, '%s.%s' % (nm, kind)), 'w') as f9:
                    f9.write(str(os.getpid()))
            except Exception:
                pass
    _note('pid')
    perm = np.load(perm_path, mmap_mode='r') if perm_path else None
    if cast is None and str(input_path).lower().endswith('.parquet'):
        try:
            import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
            ft = pq.read_schema(input_path).field(nm).type
            # ONE READ PER COLUMN (2026-09-26): the arrow road serves text only; a number column was
            # read and gathered here, declined by the prep, then read and gathered AGAIN below
            if (pa.types.is_string(ft) or pa.types.is_large_string(ft)
                    or pa.types.is_binary(ft) or pa.types.is_large_binary(ft)) and TEXT_SLICES >= 2:
                try:
                    prep = _arrow_string_prep_sliced(nm, input_path, perm=perm)   # read + prep: the column in slices
                except Exception as _e7:
                    print('  sliced text prep declined %s (%s: %s) -- the whole-column road' % (nm, type(_e7).__name__, str(_e7)[:80]), flush=True)
                    prep = None
                if prep is not None:
                    _mark('prep')
                    _tm['peak_after_prep_gb'] = _own_hwm() / 2**30
                    _note('prepped')
            if prep is None and (pa.types.is_string(ft) or pa.types.is_large_string(ft)
                    or pa.types.is_binary(ft) or pa.types.is_large_binary(ft)):
                col = pq.read_table(input_path, columns=[nm]).column(0)
                try:
                    pa.default_memory_pool().release_unused()   # the read's freed page buffers: 2.9 GB kept on URL
                except Exception:
                    pass
                _mark('read')
                if pa.types.is_string(col.type):
                    col = pc.cast(col, pa.large_string())    # 100M values overflow 'string' offsets in the concat
                box = [col]; del col                          # the prep owns the text: freed once it is encoded
                # THE CODES ARE GATHERED, NOT THE TEXT: the dictionary is sorted by value, so a code
                # is the same number in any row order -- gathering 400 MB of codes through the cluster
                # order replaces gathering 8 GB of URL text (same bytes out: the codes, the sorted
                # dictionary and the null bin do not depend on the order the rows arrive in)
                prep = _arrow_string_prep(nm, box, perm=perm)
                _mark('prep')
                import resource as _rs0
                _tm['peak_after_prep_gb'] = (_own_hwm() or _rs0.getrusage(_rs0.RUSAGE_SELF).ru_maxrss * 1024) / 2**30   # where the peak falls
                # measured on URL: the job's whole-life peak (22.0 GB) is reached by the end of the
                # prep; the serialize that follows works on the dictionary and the codes only
                _note('prepped')
        except Exception as _e9:
            # A FALLBACK THAT IS SILENT IS A FAST PATH THAT ISN'T THERE: the arrow path had failed
            # on every big string column for weeks ('offset overflow') and nobody knew
            print('  arrow prep declined %s (%s: %s) -- falling back to the object path' % (nm, type(_e9).__name__, str(_e9)[:80]), flush=True)
            prep = None
    if prep is None:
        _mark('declined')
        arr = wdb_read.read_one_column(input_path, nm, reader=reader)
        _mark('read')
        if perm is not None:
            arr = arr[np.asarray(perm)]
        _mark('gather')
        if cast is not None:
            arr = _CASTS[cast](arr)
        prep = _prep_column(nm, arr)
        _mark('prep')
        del arr
    blob, size = _serialize_column(prep, zstd.ZstdCompressor(level=ZSTD_LEVEL))
    _mark('serialize')
    del prep
    extras = None
    if N is not None:
        try:
            extras = _column_extras(nm, blob, N)
        except Exception as _e8:
            extras = None
            print('  in-job statistics declined %s (%s: %s) -- the after-step will compute them' % (nm, type(_e8).__name__, str(_e8)[:80]), flush=True)
        _mark('extras')
    import resource as _rs
    _ru = _rs.getrusage(_rs.RUSAGE_SELF)
    # THE JOB'S OWN PEAK (2026-09-26): ru_maxrss survives the exec that starts a spawned worker, so
    # every job reported at least the PARENT's resident size at launch -- the late flag columns all
    # read 4.2 GB against a measured 2.0-2.5 alone, and the budget priced and reserved them at that.
    # VmHWM is this process's own high-water mark; ru_maxrss stays the fallback.
    peak = _own_hwm() or _ru.ru_maxrss * 1024
    _tm['cpu'] = _ru.ru_utime + _ru.ru_stime          # core-seconds this column burned (all its threads)
    return nm, bytes(blob), size, peak, _tm, extras


def _column_extras(nm, blob, N):
    """THE AFTER-STEPS, DONE WHILE THE COLUMN IS STILL IN HAND (Jackson, 2026-09-26: fill the idle
    cores with work we must do anyway). The load statistics and the string lengths of this column
    are computed in its own job -- by the very functions the after-steps call (wdb_blockstats,
    wdb_lens), on a one-column segment made of this column's finished blob -- instead of re-reading
    every column from the sealed file at the end. The parent only writes them. Returns
    {'stats': {key: array}, 'dict': body or None, 'row': body or None}."""
    import tempfile
    d = '/dev/shm' if (os.path.isdir('/dev/shm') and os.access('/dev/shm', os.W_OK)) else tempfile.gettempdir()
    p = os.path.join(d, 'wdbcol_%d_%s.wdb' % (os.getpid(), nm))
    try:
        with open(p, 'wb') as f:
            f.write(b'WVDB4' + struct.pack('<H', 1) + struct.pack('<I', N))
            f.write(blob)
        from wdb_engine import Segment
        seg = Segment(p)
        st = {}
        if os.environ.get('WDB_LOAD_STATS', '1') != '0':
            import wdb_blockstats as _B
            rep = _B.differentiator_rows(seg, nm)
            if rep is not None:
                st[nm + '.rep'] = rep
            vc = _B.value_counts(seg, nm)
            if vc is not None:
                st[nm + '.vcnt'] = vc
            spp = _B.e19_signposts(seg, nm)            # THE SIGNPOSTS (block-dictionary columns)
            if spp is not None:
                st[nm + '.sp19'], st[nm + '.sp19o'] = spp
                st[nm + '.sp19s'] = np.int64(_B.SIGNPOST_EVERY)
            seg._codes.pop(nm, None)
            if _B.eligible(seg, nm):
                try:
                    s9 = _B.compute(seg, nm)
                    for kk in ('cnt', 'nn', 'sum', 'cmin', 'cmax'):
                        st[nm + '.' + kk] = s9[kk]
                    st[nm + '.mode4'] = np.bool_(s9['mode4']); st[nm + '.maxabs'] = np.float64(s9['maxabs'])
                    st[nm + '.dt'] = np.int64(s9['dt'])
                except Exception:
                    pass
        dict9 = row9 = None
        if os.environ.get('WDB_LOAD_LENGTHS', '1') != '0':
            import wdb_lens
            dict9 = wdb_lens.dict_body(seg, nm)
            if nm in [c for c in os.environ.get('WDB_ROWLEN_COLS', '').split(',') if c]:
                row9 = wdb_lens.row_body(seg, nm)
        return {'stats': st, 'dict': dict9, 'row': row9}
    finally:
        try:
            os.remove(p)
        except OSError:
            pass


_TEXT_BYTES = {}                                  # text column -> estimated in-memory bytes (from _column_cost's sample)


def _cg_mem():
    """(limit, live) bytes of this container's memory, or None. live = usage minus the inactive
    page cache -- the part the kernel cannot hand back without refusing someone."""
    try:
        if os.path.exists('/sys/fs/cgroup/memory.max'):
            lim = open('/sys/fs/cgroup/memory.max').read().strip()
            use = int(open('/sys/fs/cgroup/memory.current').read())
            st = dict(l.split()[:2] for l in open('/sys/fs/cgroup/memory.stat'))
            inact = int(st.get('inactive_file', 0))
        else:
            b = '/sys/fs/cgroup/memory/'
            lim = open(b + 'memory.limit_in_bytes').read().strip()
            use = int(open(b + 'memory.usage_in_bytes').read())
            st = dict(l.split()[:2] for l in open(b + 'memory.stat'))
            inact = int(st.get('total_inactive_file', st.get('inactive_file', 0)))
        phys = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        lim = min(int(lim), phys) if lim.isdigit() else phys
        return lim, max(0, use - inact)
    except Exception:
        return None


def _anon_rss(pids):
    """resident bytes of the given processes that are their own (resident minus file-backed shared:
    the mmap'd cluster order and the libraries every worker shares are not counted 16 times)"""
    pg = os.sysconf('SC_PAGE_SIZE'); tot = 0
    for p in pids:
        try:
            f = open('/proc/%d/statm' % p).read().split()
            tot += max(0, int(f[1]) - int(f[2])) * pg
        except Exception:
            pass
    return tot


def _column_cost(input_path, cols, reader, N):
    """THE ORDER OF OPERATIONS (Jackson): columns ranked by expected encode cost so the heavy
    ones (wide dictionaries, strings: front-coding + zstd) start first and never become the
    tail, while the layout-only ones (flags, sequences, narrow codes) fill the gaps. Cost is
    estimated from the schema and a sample, never from a full read."""
    est = {}
    _TEXT_BYTES.clear()                          # this load's text sizes only (never a previous file's)
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
                try:                                      # the column's text: parquet's own uncompressed size, every row group
                    # (row group 0 alone guessed Title at 21 GB of text; the file's metadata says 7.6)
                    md9 = pf.metadata; j9 = [md9.schema.column(k).name for k in range(md9.num_columns)].index(nm)
                    _TEXT_BYTES[nm] = int(sum(md9.row_group(i).column(j9).total_uncompressed_size for i in range(md9.num_row_groups)))
                except Exception:
                    pass
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
    # THE CORES WE MAY RUN ON (2026-09-26): os.cpu_count() is the HOST's 64; the container is
    # pinned to 16 (sched_getaffinity), and the pool had been sized for 64
    try:
        ncpu = len(_os.sched_getaffinity(0)) or (_os.cpu_count() or 4)
    except Exception:
        ncpu = _os.cpu_count() or 4
    nworkers = max(1, min(workers or ncpu, ncpu))
    try:
        budget = int(float(_os.environ.get('WDB_ENCODE_MB', '0'))) << 20
    except Exception:
        budget = 0
    if not budget:
        try:
            phys = _os.sysconf('SC_PAGE_SIZE') * _os.sysconf('SC_PHYS_PAGES')
            # BOTH CGROUP LAYOUTS: v2 (memory.max) and v1 (memory/memory.limit_in_bytes). A pod
            # exposing only v1 fell to the 32 GB fallback, ran the string columns one at a time
            # and the ints 2-3 wide: 920s for a realm the same code encodes in 405s at 96 GB.
            for p in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
                try:
                    with open(p) as f:
                        v = f.read().strip()
                    if v.isdigit() and int(v) < (1 << 60): phys = min(phys, int(v))
                    break
                except FileNotFoundError:
                    continue
            budget = phys * 3 // 4                 # THREE-QUARTERS: children peaked at 55 GB of 128 at half; the page cache yields
        except Exception:
            budget = 32 << 30
        if _os.environ.get('WDB_ENCODE_VERBOSE'):
            print('  encode budget: %.0f GB (physical %.0f GB)' % (budget / 2**30, phys / 2**30), flush=True)
    def cls(nm):
        # FIVE CLASSES, not three: one heavy 'narrow' member (a 40%-distinct hash) had set the
        # price for every flag column -- the class was budgeted at 12.5 GB/column when a flag
        # peaks near 3 GB and a small dictionary near 7 GB (measured 2026-09-14)
        e = est.get(nm, 1.0)
        return 'string' if e >= 8 else ('wide' if e >= 2.0 else ('mid' if e >= 1.5 else ('narrow' if e >= 0.5 else 'tiny')))
    # starting rates raised to the peaks measured 2026-09-26 (a flag column 3.9-4.6 GB at 100M rows,
    # a narrow one 4.5-9, mid 6-13, wide 10-15): the live rule below reserves by these, so they must
    # not start under the truth
    # 2026-09-27: tiny and narrow restarted at their TRUE p80 peaks (3.1 / 4.5 GB, VmHWM) -- the 50 / 75
    # were set from ru_maxrss, which carried the parent's size into every spawned job
    measured = {'string': 310, 'wide': 130, 'mid': 100, 'narrow': 55, 'tiny': 36}   # strings: the MEASURED peak on the arrow path (Title 29 GB at 100M rows; it was 57 GB on the object path)     # bytes per row: conservative starts (URL/Referer/Title peak ~25 GB at 100M rows), raised as workers report
    # A TEXT COLUMN IS PRICED BY ITS TEXT (2026-09-26): one class rate for every string charged
    # SearchPhrase (8 GB measured) like URL (38 GB), so SearchPhrase and OriginalURL could not fit
    # while small columns ran, waited to the end and ran alone for 84 s. The charge is a floor (a
    # worker's own weight, 4.6 GB measured on a flag column) plus a learned multiple of the text.
    # measured with the codes gathered, against parquet's uncompressed text: URL 21.5 GB on 7.9 (2.08x
    # over the floor), Title 22.5 on 7.6 (2.30x), SearchPhrase 6.6 on 0.77 (2.1x)
    # REPRICED 2026-09-27 for the text in slices: measured peaks URL 16.1 GB on 7.9 of text, OriginalURL
    # 10.9 on 5.0, Referer 10.9 on 6.1, Title 6.6 on 7.6, SearchPhrase 3.9 on 0.77, a small text column
    # 1.8-1.9 -- (peak - 2.5 GB) / text is at most ~1.8. The rate still only learns upward.
    sbase = int(2.5 * (1 << 30)); sratio = [1.9]
    def working_set(nm):
        tb = _TEXT_BYTES.get(nm)
        if cls(nm) == 'string' and tb:
            return int(min(budget / 2, sbase + sratio[0] * tb))
        return int(N * measured[cls(nm)]) + (400 << 20)
    def learn(nm, peak):
        tb = _TEXT_BYTES.get(nm)
        if cls(nm) == 'string' and tb:
            r = max(0.0, peak - sbase) / tb * 1.05      # the budget already sits 30 GB under the cgroup
            if r > sratio[0]:
                sratio[0] = r
            return
        # THE ENCODER MEASURES ITSELF: a worker's peak RSS updates its class's bytes-per-row
        # (max seen, plus 25% headroom) so the budget stops guessing after the first column.
        # One observation can never teach more than "two of this class fit the budget": an
        # outlier (73 GB on URL) had starved the string class to one column at a time.
        c = cls(nm); seen = (peak - (400 << 20)) / max(1, N) * 1.25
        cap = (budget / 2 - (400 << 20)) / max(1, N)
        seen = min(seen, cap)
        # A CLASS IS PRICED BY ITS TYPICAL MEMBER (2026-09-26): the max rule let one member set the price
        # of all -- CounterID (8.6 GB) charged every flag column ~10 GB against a real 4. From three
        # observations on, the rate is the 80th-percentile peak plus 20%; the limit's margin carries
        # the rare column above it.
        pk = _pk9.setdefault(c, []); pk.append(peak)
        if len(pk) >= 3:
            q = sorted(pk)[int(0.8 * (len(pk) - 1))]
            measured[c] = min(cap, max(36.0, (q - (400 << 20)) / max(1, N) * 1.2))
        elif seen > measured[c]:
            measured[c] = seen
    sizes = {}; _sub9 = {}; _chg9 = {}; _pk9 = {}; _xt9 = {}; _ord9 = []
    injob = _os.environ.get('WDB_LOAD_INJOB', '1') != '0'
    perm_path = None
    if _os.environ.get('WDB_ENCODE_VERBOSE'):
        print('  schema + cost estimate: %.1fs' % (time.time() - t0), flush=True)
        for nm9 in order:
            if cls(nm9) == 'string':
                print('  text %-14s %5.1f GB of text, charged %5.1f GB' % (nm9, _TEXT_BYTES.get(nm9, 0) / 2**30, working_set(nm9) / 2**30), flush=True)
    if cluster_by:
        # THE CLUSTER ORDER: argsort the clustering column(s) once (stable), share the
        # permutation as an mmap'd file; every worker gathers its column through it
        keys9 = [cluster_by] if isinstance(cluster_by, str) else list(cluster_by)
        _c0 = time.time()
        arrs9 = [np.asarray(wdb_read.read_one_column(input_path, k, reader=reader)) for k in keys9]
        _c1 = time.time()
        # ONE INTEGER KEY IS ORDERED BY COUNTING (2026-09-26): the same stable permutation lexsort
        # gives (equal keys keep file order), 0.75 s against 8.4 s for EventTime -- time every other
        # core spent waiting, since no column can start before the order exists
        perm = None
        if len(arrs9) == 1:
            try:
                import wdb_kernels as _WKo
                perm = _WKo.counting_order(arrs9[0], np.int64 if N >= (1 << 31) else np.int32)
            except Exception:
                perm = None
        if perm is None:
            perm = np.lexsort(tuple(reversed(arrs9))).astype(np.int64 if N >= (1 << 31) else np.int32)
        del arrs9
        _c2 = time.time()
        # IN RAM WHEN THERE IS ONE (2026-09-26): the order was saved next to the output, on the network
        # volume (0.9 s, before any column can start), and every job mapped it from there. /dev/shm
        # holds it in memory -- the page cache the jobs shared anyway. Removed at the close as before.
        perm_path = out_path + '.perm.tmp.npy'
        try:
            _sd9 = '/dev/shm'
            _need9 = perm.nbytes + (1 << 26)
            if _os.path.isdir(_sd9) and _os.access(_sd9, _os.W_OK):
                _st9 = _os.statvfs(_sd9)
                if _st9.f_bavail * _st9.f_frsize > 2 * _need9:
                    perm_path = _os.path.join(_sd9, 'wdbperm_%d_%s.npy' % (_os.getpid(), _os.path.basename(out_path)))
        except Exception:
            pass
        np.save(perm_path, perm); del perm
        if _os.environ.get('WDB_ENCODE_VERBOSE'):
            print('  cluster order by %s computed (%d rows) at %.1fs (read %.1f s, sort %.1f s, save %.1f s)' % (
                ', '.join(keys9), N, time.time() - t0, _c1 - _c0, _c2 - _c1, time.time() - _c2), flush=True)
    _final9 = out_path; out_path = out_path + '.partial'        # THE RENAME LAW: a partial file never wears the final name
    fh = open(out_path, 'wb')
    fh.write(b'WVDB4' + struct.pack('<H', len(cols)) + struct.pack('<I', N))
    pending = list(order); verbose = bool(_os.environ.get('WDB_ENCODE_VERBOSE'))
    _wr9 = [0.0, 0, 0.0, 0, 0]                     # the write clock: seconds, bytes, longest write, queued bytes, most queued
    # THE WRITER THREAD (Jackson, 2026-09-26: write as fast as the lane allows). The volume under
    # /workspace is a network filesystem: 591 MB/s measured, and the parent spent 14.1 s of a load
    # writing blobs INLINE -- no admission, no collection while a write crossed the wire (longest
    # 1.5 s). The blobs now go to a queue drained by one thread in arrival order: the same bytes in
    # the same order; the scheduling loop never waits on the disk. A write error is re-raised by
    # the loop at its next blob and at the close.
    import threading as _th9, queue as _qu9
    _wq9 = _qu9.Queue(); _werr9 = []
    def _writer9():
        while True:
            b9 = _wq9.get()
            if b9 is None:
                return
            if not _werr9:
                try:
                    _w0 = time.time()
                    fh.write(b9)
                    _wr9[0] += time.time() - _w0; _wr9[1] += len(b9); _wr9[2] = max(_wr9[2], time.time() - _w0)
                except BaseException as e9:
                    _werr9.append(e9)
            _wr9[3] -= len(b9)
            del b9
    _wt9 = _th9.Thread(target=_writer9, name='wdb-blob-writer', daemon=True); _wt9.start()
    def _put9(b9):
        if _werr9:
            raise _werr9[0]
        _wr9[3] += len(b9); _wr9[4] = max(_wr9[4], _wr9[3])
        _wq9.put(b9)
    def _drain9():
        if _wt9.is_alive():
            _wq9.put(None); _wt9.join()
        if _werr9:
            raise _werr9[0]
    live_on = _os.environ.get('WDB_ENCODE_LIVE', '1') != '0' and _cg_mem() is not None
    jobdir = out_path + '.jobs'
    try:
        _os.makedirs(jobdir, exist_ok=True); _os.environ['WDB_ENC_JOBDIR'] = jobdir   # workers fork after this
    except Exception:
        live_on = False

    def live_fits(w, running, memo):
        """THE LIVE RULE (2026-09-26, Jackson: fill the idle cores): the paper budget counts every
        running job at its PEAK, and the peaks do not coincide -- 15-150 s of a load ran 5-9 of 16
        cores with the paper full and 40+ GB really free. A job may also start when what the
        container really holds, plus every running job's remaining growth to its charge, plus the
        newcomer's charge, stays a margin under the cgroup limit. memo: a one-slot list, filled once
        per admission round."""
        if memo[0] is None:
            cg = _cg_mem()
            if cg is None:
                return False
            # each running job keeps the room to grow to its charge: charge minus its own resident
            # bytes; a text job past its prep (the raw text freed, its peak behind it -- measured: URL
            # 21.8 GB, Title 22.4, Referer 16.5 at the end of the prep and never above) keeps a
            # quarter of its charge for the serialize
            grow = 0
            for c9 in running.values():
                ch9 = _chg9.get(c9, working_set(c9))
                try:
                    with open(_os.path.join(jobdir, c9 + '.pid')) as fp9:
                        pid9 = int(fp9.read())
                except Exception:
                    grow += ch9
                    continue
                if cls(c9) == 'string' and _os.path.exists(_os.path.join(jobdir, c9 + '.prepped')):
                    grow += ch9 // 4
                else:
                    grow += max(0, ch9 - _anon_rss([pid9]))
            memo[0] = (cg[0], cg[1], grow)
        lim, live, growth = memo[0]
        return live + growth + w <= lim - max(8 << 30, lim // 12)
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
                        # THE NEXT TEXT COLUMN KEEPS ITS SEAT: a small column may take free budget only
                        # if the first waiting text column would still fit beside it -- the long jobs
                        # start as early as memory allows instead of being overtaken to the end
                        heavy_wait = next((p for p in pending if est.get(p, 1.0) >= 8), None)
                        live9 = [None]
                        for i9, cand in enumerate(pending):
                            if est.get(cand, 1.0) >= 8 and heavy_in >= 6: continue
                            w9 = working_set(cand)
                            paper = (not inflight) or used + w9 <= budget
                            if not paper and (est.get(cand, 1.0) >= 8 or not live_on
                                              or not live_fits(w9, inflight, live9)): continue
                            if (paper and heavy_wait is not None and cand != heavy_wait and est.get(cand, 1.0) < 8
                                    and inflight and used + w9 + working_set(heavy_wait) > budget):
                                continue
                            pick = i9; break
                        if pick is None: break
                        nm = pending.pop(pick)
                        inflight[ex.submit(_column_job, input_path, nm, reader, (casts or {}).get(nm), perm_path,
                                           N if injob else None)] = nm
                        # THE CHARGE IS REMEMBERED (2026-09-26): the release used to re-price the column at
                        # the class's LEARNED rate, larger than the rate it was admitted at, so the books
                        # drifted to -123 GB by the end of a load and the budget stopped meaning anything
                        _chg9[nm] = working_set(nm); used += _chg9[nm]; _sub9[nm] = time.time() - t0
                    if not inflight: break
                    # a running job passing its peak frees real memory without finishing: look again
                    # every few seconds instead of only when a column lands
                    done9, _ = cf.wait(list(inflight.keys()), timeout=(3.0 if (live_on and pending) else None),
                                       return_when=cf.FIRST_COMPLETED)
                    if not done9:
                        continue
                    fut = next(iter(done9))
                    res = fut.result()                 # BEFORE the pop: a result that raises leaves the column in inflight for the re-queue
                    nm = inflight.pop(fut); used -= _chg9.pop(nm, working_set(nm))
                    cname, blob, size = res[0], res[1], res[2]
                    if len(res) > 3: learn(cname, int(res[3]))
                    if len(res) > 5 and res[5] is not None:
                        _xt9[cname] = res[5]
                    _put9(blob); sizes[cname] = size; _ord9.append(cname)
                    del blob
                    if verbose:
                        print('  encoded %-24s (%d/%d, %.0fs, %d in flight, class %s @ %.0f B/row)' % (cname, len(sizes), len(cols), time.time() - t0, len(inflight), cls(cname), measured[cls(cname)]), flush=True)
                        if len(res) > 4 and isinstance(res[4], dict):
                            _cg9 = _cg_mem()
                            print('    clock %-22s est %5.1f  started %5.0fs  peak %5.1f GB  budget-in-use %5.1f GB  live %5.1f GB  ' % (
                                      cname, est.get(cname, 1.0), _sub9.get(cname, 0.0), int(res[3]) / 2**30, used / 2**30,
                                      (_cg9[1] / 2**30) if _cg9 else -1)
                                  + '  '.join('%s %.1f' % kv for kv in res[4].items()), flush=True)
        except cf.process.BrokenProcessPool:
            lost = [v for v in inflight.values() if v not in sizes]
            pending = lost + [p for p in pending if p not in sizes]
            if nworkers <= 1:
                _drain9(); fh.close(); raise
            nworkers = max(1, nworkers // 2)
            live_on = False                        # after a kill, only the paper budget admits
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
                _put9(blob); sizes[cname] = size; _ord9.append(cname); del blob
    _w0 = time.time()
    _drain9()
    _w1 = time.time()
    fh.flush(); _os.fsync(fh.fileno()); fh.close()
    if verbose:
        # THE WRITE CLOCK (2026-09-26): what the writer thread spent, and what the close waited for
        print('  writer thread wrote %.2f GB in %.1f s (%.0f MB/s, longest single write %.2f s, at most %.2f GB queued); '
              'close waited %.2f s for the queue + %.2f s flush/fsync' % (
            _wr9[1] / 2**30, _wr9[0], _wr9[1] / 2**20 / max(1e-9, _wr9[0]), _wr9[2], _wr9[4] / 2**30,
            _w1 - _w0, time.time() - _w1), flush=True)
    _os.replace(out_path, _final9); out_path = _final9
    try:
        import shutil as _sh9
        _sh9.rmtree(jobdir, ignore_errors=True); _os.environ.pop('WDB_ENC_JOBDIR', None)
    except Exception:
        pass
    _crash_point('encode:renamed')
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
    _tp9 = time.time()
    if verbose:
        print('  file closed at %.1fs' % (_tp9 - t0), flush=True)
    # THE THREE AFTER-STEPS SIDE BY SIDE (2026-09-26): the differentiator shelves, the load
    # statistics and the string lengths each open the closed segment and write their own files;
    # none reads another's output. They ran one after another (17 + 24 + 9 s); each now runs in its
    # own process (numba's parallel kernels are not shared across threads) and the load waits for all.
    _post9 = [('differentiator shelves', _post_shelves), ('load stats', _write_load_stats), ('lengths', _write_lengths)]
    if injob and all(c in _xt9 for c in cols):
        # every column brought its statistics and lengths from its own job: write them, and leave
        # only the shelves (they pair columns with each other) to the after-step
        _write_extras(out_path, N, _ord9, _xt9, verbose)
        _post9 = []
        if verbose:
            print('  statistics + lengths written from the jobs at +%.1fs' % (time.time() - _tp9), flush=True)
        try:
            import wdb_sidecar as _SC9
            # THE SWITCH governs the load's shelves too (2026-09-29): the pair tables are derived
            # group facts, and with sidecars off they were born anyway -- measured, they only slowed
            # the pair boards they serve (Q31 cold 265 -> 192 ms without them, Q32 194-216 -> 110)
            if _os.environ.get('WDB_LOAD_SHELVES', '1') != '0' and _SC9.births_on(_os.path.dirname(out_path)):
                _birth_differentiator_shelves(out_path, known={c: (_xt9[c].get('stats') or {}).get(c + '.rep') for c in cols})
        except Exception:
            pass
        if verbose:
            print('  differentiator shelves done at +%.1fs' % (time.time() - _tp9), flush=True)
    try:
        if _post9:
            with cf.ProcessPoolExecutor(max_workers=len(_post9)) as _px9:
                _fu9 = [(lab, _px9.submit(fn, out_path)) for lab, fn in _post9]
                for lab, fu in _fu9:
                    fu.result()
                    if verbose:
                        print('  %s done at +%.1fs' % (lab, time.time() - _tp9), flush=True)
    except cf.process.BrokenProcessPool:
        for lab, fn in _post9:                     # the promise is the files, not the parallelism
            fn(out_path)
    return dict(n_rows=N, n_cols=len(cols), bytes=(len(out) if out is not None else __import__('os').path.getsize(out_path)), seconds=time.time() - t0,
                sizes=sizes, cluster=None)


def input_column_order(input_path, names):
    """the input file's own column order for the columns a load stored (a segment keeps them in the
    order their jobs finished): the table's declared order, which SELECT * answers in. The segment's
    order when the input's cannot be read or does not name exactly these columns."""
    names = list(names)
    try:
        import pyarrow.parquet as _pq
        ins = list(_pq.ParquetFile(input_path).schema_arrow.names)
    except Exception:
        try:
            import pyarrow.csv as _pc
            ins = list(_pc.open_csv(input_path).schema.names)
        except Exception:
            return names
    return ins if sorted(ins) == sorted(names) else names


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
        try:
            return _encode_streaming(input_path, out_path, columns, reader, cubes, workers, t0, casts=casts, cluster_by=cluster_by)
        finally:
            _TEXT_BYTES.clear()                      # load planning state dies with the load (the qmem law's witness)
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
        if os.environ.get('WDB_LOAD_SHELVES', '1') != '0':
            _birth_differentiator_shelves(out_path)
    except Exception:
        pass
    _write_load_stats(out_path)
    _write_lengths(out_path)
    return dict(n_rows=N, n_cols=len(cols), bytes=len(out), seconds=time.time()-t0,
                sizes=sizes, cluster=cluster_by)


def _write_extras(out_path, N, order, extras, verbose=False):
    """The load statistics file and the string length files, from what each column's job computed
    (the same keys, in the file's column order, as wdb_blockstats.write_for_segment; the same
    headers and payloads as wdb_lens.write_for_segment)."""
    import wdb_blockstats as _B, wdb_lens
    if os.environ.get('WDB_LOAD_STATS', '1') != '0':
        out = {'N': np.int64(N)}
        n = 0
        for col in order:
            st = extras[col].get('stats') or {}
            if (col + '.rep') in st and verbose:
                print('  stats: %s is a differentiator, %d exception rows' % (col, st[col + '.rep'].size), flush=True)
            out.update(st)
            n += 1 if (col + '.cnt') in st else 0
        p = _B.stats_path(out_path)
        tmp = p + '.partial.npz'
        np.savez(tmp, **out)
        os.replace(tmp, p)
        _B._LOADED.pop(p, None)
        if verbose:
            print('  stats: %d columns, %.1f KB -> %s' % (n, os.path.getsize(p) / 1024, os.path.basename(p)), flush=True)
    if os.environ.get('WDB_LOAD_LENGTHS', '1') != '0':
        size = os.path.getsize(out_path)
        for col in order:
            if extras[col].get('dict') is not None:
                wdb_lens.write_dict(out_path, col, extras[col]['dict'], size, verbose)
        for col in [c for c in os.environ.get('WDB_ROWLEN_COLS', '').split(',') if c]:
            if col in extras and extras[col].get('row') is not None:
                wdb_lens.write_row(out_path, col, extras[col]['row'], size, verbose)


def _post_shelves(out_path):
    try:
        import wdb_sidecar
        if os.environ.get('WDB_LOAD_SHELVES', '1') != '0' \
                and wdb_sidecar.births_on(os.path.dirname(out_path)):     # THE SWITCH (see the in-job twin)
            _birth_differentiator_shelves(out_path)
    except Exception:
        pass
    return 0


def _write_lengths(out_path):
    """STRING LENGTHS AS LOAD DATA (wdb_lens): every front-coded text column's dictionary character
    lengths, and the row-order lengths of the columns the operator named (bin/wdb load --row-lengths
    -> WDB_ROWLEN_COLS). WDB_LOAD_LENGTHS=0 skips."""
    import os as _os
    if _os.environ.get('WDB_LOAD_LENGTHS', '1') == '0':
        return 0
    try:
        import wdb_lens
        rc = [c for c in _os.environ.get('WDB_ROWLEN_COLS', '').split(',') if c]
        return wdb_lens.write_for_segment(out_path, rc, verbose=bool(_os.environ.get('WDB_ENCODE_VERBOSE')))
    except Exception:
        if _os.environ.get('WDB_ENCODE_VERBOSE'):
            import traceback, sys as _sy
            print('LOAD LENGTHS DIED:', file=_sy.stderr); traceback.print_exc()
        return 0


def _write_load_stats(out_path):
    """THE STATISTICS OF THE LOAD (Jackson, 2026-09-20): block statistics -- count, non-null count,
    sum, min and max code per 32K-row block, a few bytes per block -- are METADATA OF THE LOAD,
    written by the encoder and counted in its time, not a sidecar a query births. The block-stats
    read answers SUM/AVG/MIN/MAX/COUNT from them under any switch setting. WDB_LOAD_STATS=0 skips."""
    import os as _os
    if _os.environ.get('WDB_LOAD_STATS', '1') == '0':
        return 0
    try:
        import wdb_blockstats
        return wdb_blockstats.write_for_segment(out_path, verbose=bool(_os.environ.get('WDB_ENCODE_VERBOSE')))
    except Exception:
        if _os.environ.get('WDB_ENCODE_VERBOSE'):
            import traceback, sys as _sy
            print('LOAD STATS DIED:', file=_sy.stderr); traceback.print_exc()
        return 0

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: wdb_encode.py <input.parquet|csv> <out.wdb> [col1,col2,...]"); sys.exit(1)
    cols = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    r = encode(sys.argv[1], sys.argv[2], cols)
    fc=sum(1 for v in r['sizes'].values() if v[4]==1); fl=sum(1 for v in r['sizes'].values() if v[3]==2); dt=sum(1 for v in r['sizes'].values() if v[3]==3); nu=sum(1 for v in r['sizes'].values() if v[5]==1)
    print(f"Encoded {r['n_cols']} cols x {r['n_rows']:,} rows -> {r['bytes']/1e6:.1f} MB in {r['seconds']:.0f}s ({fc} front-coded, {fl} float, {dt} datetime, {nu} nullable)")


def _birth_differentiator_shelves(out_path, known=None):
    """THE SHELVES WITHOUT A FULL DECODE (2026-09-26, measured: 12.7 s of the load's tail was HID and
    WatchID decoded and counted, then 42 companion columns decoded whole to read 8 rows each).
    - known: {column: exception rows} its own load job already found by the same law
      (wdb_blockstats.differentiator_rows); used as given instead of counting again.
    - A column of V distinct codes over N rows has at least N - V rows in repeated groups: when that
      floor alone breaks 0.75% of V the column cannot qualify, and nothing is decoded to learn it.
    - The pairs read the companions only at the exception rows (Segment.codes_at). Same files."""
    import importlib, pickle
    import numpy as _np
    eng = importlib.import_module('wdb_engine')
    seg = eng.Segment(out_path)
    N = int(seg.N)
    known = known or {}
    for cn, c in list(seg.cols.items()):
        try:
            V = int(c.get('V') or 0)
            if V * 2 < N or V < 1024:
                continue                          # role isn't differentiation
            if N - V - 1 >= 0.0075 * V:
                continue                          # the repeat floor alone disqualifies (one spare code for a null)
            if known.get(cn) is not None:
                rep = _np.asarray(known[cn], _np.int64)
            else:
                codes = _np.asarray(seg._raw_codes(cn))
                cnt = _np.bincount(codes, minlength=V)
                rep = _np.flatnonzero(cnt[codes] >= 2)
                del codes, cnt
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
    self-disqualify (their repeats swamp the threshold). Kilobytes total.
    Both sides are read only at the exception rows (Segment.codes_at)."""
    import numpy as _np
    rep = _np.sort(_np.asarray(rep, _np.int64))
    if rep.size == 0:
        return
    ac = _np.asarray(seg.codes_at(a, rep), _np.int64)
    ys = [y for y, cy in seg.cols.items() if y != a and int(cy.get('V') or 0) >= 1024]   # low V: repeats swamp 0.75%
    for y in ys:
        _pair_shelf_one(seg, out_path, a, rep, ac, y)


def _pair_shelf_one(seg, out_path, a, rep, ac, y):
    """one differentiator-companion pair shelf (the body of Jackson's eager expansion)"""
    import pickle
    import numpy as _np
    for _once in (0,):
        try:
            cy = seg.cols[y]
            Vy = int(cy.get('V') or 0)
            bc = _np.asarray(seg.codes_at(y, rep), _np.int64)
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
                return                            # relationship disqualified
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
