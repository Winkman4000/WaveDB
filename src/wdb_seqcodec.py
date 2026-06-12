"""WaveDB sequence (affine) codec -- mode 4 building blocks, standalone and engine-free.

A sequential key column (auto-increment PK, monotonic timestamp) looks like maximum entropy
(N distinct -> ~N*log2(N) bits) but its TRUE entropy is O(log N): value[i] = base + i*stride,
plus a sparse list of exceptions where the progression breaks (gaps from deletes, manual ids).
This codec stores only (base, stride, n, exceptions) -- NO per-row codes, NO dictionary -- so
a clean key column collapses from megabytes to ~32 bytes.

Losslessness rests on int64 two's-complement wraparound: corrections are stored as
(true_delta - stride) and the column is rebuilt as base + cumsum(deltas), all in int64. Even
when an individual delta overflows int64 (e.g. a giant gap), stride + correction recovers the
true delta mod 2^64, and cumsum recovers the originals exactly (pinned by
tests/test_edges.test_wraparound_cancellation_direct). Therefore the MECHANISM is lossless on
ANY int64 input; the gate in encode() only decides when it's BENEFICIAL.

Format (little-endian):
  magic  'WSQ1'                          4 bytes
  base   int64                           8
  stride int64                           8
  n      uint64   (row count)            8
  n_exc  uint32   (exception count)      4
  [if n_exc > 0]
    zlen uint32                          4
    zstd( exc_pos[int64] ++ corr[int64] )   exc_pos are delta-indices in [0, n-2]
"""
import struct
import numpy as np
import zstandard as zstd

MAGIC = b'WSQ1'
_HDR = struct.Struct('<qqQI')          # base, stride, n, n_exc  (after magic)
_ZLEN = struct.Struct('<I')
ZSTD_LEVEL = 9
MIN_ROWS = 3                            # below this, the formula header costs more than the data

def params(col):
    """Detector. col: 1-D int64 array. Returns dict(base, stride, n, exc, corr, n_exc,
    conform) or None if too small to consider. Pure measurement -- no gating."""
    col = np.asarray(col)
    n = int(col.shape[0])
    if n < 2:
        return None
    d = np.diff(col.astype(np.int64, copy=False))           # int64 deltas (may wrap; that's fine)
    vals, counts = np.unique(d, return_counts=True)
    stride = np.int64(vals[np.argmax(counts)])              # dominant delta
    exc = np.flatnonzero(d != stride).astype(np.int64)      # positions where the stride breaks
    corr = (d[exc] - stride).astype(np.int64)               # correction = true_delta - stride
    conform = 1.0 - (len(exc) / len(d)) if len(d) else 1.0
    return dict(base=np.int64(col[0]), stride=stride, n=n,
                exc=exc, corr=corr, n_exc=int(len(exc)), conform=conform)

def build(p):
    """Serialize detector params -> blob bytes (the MECHANISM; no gating)."""
    blob = MAGIC + _HDR.pack(int(p['base']), int(p['stride']), int(p['n']), p['n_exc'])
    if p['n_exc']:
        payload = p['exc'].astype('<i8').tobytes() + p['corr'].astype('<i8').tobytes()
        z = zstd.ZstdCompressor(level=ZSTD_LEVEL).compress(payload)
        blob += _ZLEN.pack(len(z)) + z
    return blob

def decode(blob):
    """blob -> int64 column (exact inverse of build)."""
    assert blob[:4] == MAGIC, "bad WSQ1 magic"
    base, stride, n, n_exc = _HDR.unpack_from(blob, 4)
    base = np.int64(base); stride = np.int64(stride); n = int(n)
    out = np.empty(n, dtype=np.int64)
    if n == 0:
        return out
    out[0] = base
    if n == 1:
        return out
    d = np.full(n - 1, stride, dtype=np.int64)
    if n_exc:
        off = 4 + _HDR.size
        (zlen,) = _ZLEN.unpack_from(blob, off); off += _ZLEN.size
        payload = zstd.ZstdDecompressor().decompress(blob[off:off + zlen])
        half = n_exc * 8
        exc = np.frombuffer(payload[:half], dtype='<i8')
        corr = np.frombuffer(payload[half:2 * half], dtype='<i8')
        d[exc] = stride + corr                              # recover true delta (int64 wrap)
    np.cumsum(d, out=out[1:])
    out[1:] += base                                         # base + cumsum(deltas)
    return out

def header(blob):
    """Cheap inspection of a WSQ1 blob without decoding: (base, stride, n, n_exc).
    Lets callers detect a clean-affine column (n_exc == 0) for O(1) predicate fast-paths."""
    assert blob[:4] == MAGIC, "bad WSQ1 magic"
    base, stride, n, n_exc = _HDR.unpack_from(blob, 4)
    return int(base), int(stride), int(n), int(n_exc)

def exceptions(blob):
    """Parse a WSQ1 blob into its raw pieces WITHOUT reconstructing the column:
    (base, stride, n, exc, corr). exc are the delta-indices in [0, n-2] where the true delta
    departs from stride (sorted ascending, as params() produced them); corr[k] = true_delta - stride
    at exc[k]. The column is base + cumsum(d) where d == stride except d[exc] == stride + corr.
    This lets a predicate be resolved against the O(n_exc) change-points instead of the O(n) column."""
    assert blob[:4] == MAGIC, "bad WSQ1 magic"
    base, stride, n, n_exc = _HDR.unpack_from(blob, 4)
    if n_exc == 0:
        z = np.empty(0, dtype=np.int64)
        return int(base), int(stride), int(n), z, z
    off = 4 + _HDR.size
    (zlen,) = _ZLEN.unpack_from(blob, off); off += _ZLEN.size
    payload = zstd.ZstdDecompressor().decompress(blob[off:off + zlen])
    half = n_exc * 8
    exc = np.frombuffer(payload[:half], dtype='<i8')
    corr = np.frombuffer(payload[half:2 * half], dtype='<i8')
    return int(base), int(stride), int(n), exc, corr

def encode(col, max_exc_frac=0.5):
    """Gated encode for the column. Returns a blob if mode 4 is BENEFICIAL for this column,
    else None (decline -> caller uses another mode). Gate: enough rows, the stride actually
    dominates, and the blob beats raw int64 storage."""
    p = params(col)
    if p is None or p['n'] < MIN_ROWS:
        return None
    if p['conform'] < (1.0 - max_exc_frac):                 # stride not dominant -> decline
        return None
    blob = build(p)
    if len(blob) >= p['n'] * 8:                             # not beating raw int64 -> decline
        return None
    return blob

def is_lossless(col):
    """Mechanism self-check: build then decode reproduces col exactly (regardless of gate).
    Used as the encode-time safety gate in integration."""
    col = np.asarray(col).astype(np.int64, copy=False)
    p = params(col)
    if p is None:
        return True
    return np.array_equal(decode(build(p)), col)
