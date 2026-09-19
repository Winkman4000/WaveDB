"""Foreign-key pointer sidecar.

A pre-resolved join: for each row of a child table, the row position of its matching parent row
(parent stored sorted by its key). With this stored, a join becomes a gather -- parent_col[ptr] --
instead of a hash match. Kept as a sidecar next to the child segment (like the presence/override
sidecars) so the immutable .wdb bytes are never touched. A MISSING sidecar just means "no pointer".

Stored delta-encoded + zstd: on a clustered arm (child sorted so the pointer is monotonic) this
compresses ~95% (measured on TPC-H lineitem->orders); reconstructed with one cumsum on load.
"""
import os, struct, numpy as np, zstandard as zstd

_MAGIC = b'WFKP1'


def path_for(seg_path, fk_col):
    return f"{seg_path}.fkptr.{fk_col}"


def save(seg_path, fk_col, ptr):
    ptr = np.asarray(ptr, dtype=np.int64)
    deltas = np.diff(ptr, prepend=np.int64(0))          # delta[0] = ptr[0]
    comp = zstd.ZstdCompressor(level=9).compress(deltas.tobytes())
    body = _MAGIC + struct.pack('<Q', len(ptr)) + comp
    p = path_for(seg_path, fk_col); tmp = p + '.tmp'
    import wdb_sidecar
    if not wdb_sidecar.births_on(os.path.dirname(p)): return             # THE SWITCH
    with open(tmp, 'wb') as f: f.write(body)
    os.replace(tmp, p)


def load(seg_path, fk_col):
    """Return the int64 pointer array (absolute parent row positions), or None if no sidecar."""
    p = path_for(seg_path, fk_col)
    if not os.path.exists(p): return None
    buf = open(p, 'rb').read()
    assert buf[:5] == _MAGIC, f"not an fk-pointer sidecar: {p}"
    n = struct.unpack_from('<Q', buf, 5)[0]
    raw = zstd.ZstdDecompressor().decompress(buf[13:])
    deltas = np.frombuffer(raw, dtype=np.int64)
    assert len(deltas) == n, f"fk-pointer length {len(deltas)} != stored {n}"
    return np.cumsum(deltas).astype(np.int64)
