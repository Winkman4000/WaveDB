"""Presence sidecar: per-segment mutable bitset marking which rows are live.

A cold .wdb segment is immutable. To DELETE without rewriting it, we keep a tiny sidecar
file next to it ('<segment>.presence') holding one bit per row: 1 = live, 0 = tombstoned.
Reads mask out tombstoned rows; compaction physically drops them when it rewrites.

A MISSING sidecar means "all rows live" — so every existing segment and every table that
never deletes needs no sidecar and no migration. The immutable .wdb bytes are never touched.

Format: magic b'WPRS1' + <I row_count + packbits(presence_bool[, MSB-first]).
"""
import os, struct, numpy as np

_MAGIC = b'WPRS1'

def path_for(seg_path):
    return seg_path + '.presence'

def load(seg_path, N):
    """Return a bool array (len N, True=live), or None if no sidecar exists (all live)."""
    p = path_for(seg_path)
    if not os.path.exists(p):
        return None
    buf = open(p, 'rb').read()
    assert buf[:5] == _MAGIC, f"not a presence sidecar: {p}"
    stored_N = struct.unpack_from('<I', buf, 5)[0]
    assert stored_N == N, f"presence sidecar row count {stored_N} != segment N {N}"
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8, offset=9))
    return bits[:N].astype(bool)

def save(seg_path, presence):
    """Write the sidecar atomically. presence: bool array, True=live."""
    presence = np.asarray(presence, dtype=bool)
    N = len(presence)
    body = _MAGIC + struct.pack('<I', N) + np.packbits(presence.astype(np.uint8)).tobytes()
    p = path_for(seg_path)
    tmp = p + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(body)
    os.replace(tmp, p)   # atomic

def mark_deleted(seg_path, N, row_idx):
    """Tombstone the given row indices in the sidecar (creating it if absent).
    Returns the number of rows newly tombstoned."""
    presence = load(seg_path, N)
    if presence is None:
        presence = np.ones(N, dtype=bool)
    row_idx = np.asarray(row_idx, dtype=np.int64)
    newly = int(presence[row_idx].sum())   # how many were still live
    presence[row_idx] = False
    save(seg_path, presence)
    return newly

def live_count(seg_path, N):
    presence = load(seg_path, N)
    return N if presence is None else int(presence.sum())
