"""Presence (tombstone) sidecar: which rows of a segment are still live.

THE TABLE-LEVEL PRESENCE FILE (B, step 5): one file per TABLE, <dbdir>/<table>.presence,
holding every segment's mask, written with a single atomic rename. A DELETE over a
five-segment table used to write five per-segment files one after another; a reader
between two of them saw a PARTIAL delete (the concurrency harness on the multi-segment
realm: COUNT(*) = 199,586 for 200,000 before / 197,967 after). A statement's effect now
lands on every segment at once. The per-segment API (load / save / mark_deleted by
segment path) is unchanged for its callers; mark_deleted_many is the batch that DELETE
uses. A missing entry means all live. Legacy per-segment files (<seg>.presence, WPRS1)
are still read.

file: magic WPRS2, then entries of <H name_len><name><I N><I nbytes><packed bits>.
"""
import os, struct
import numpy as np


def _touch_stamp(dbdir):
    """A DML WRITE MOVES THE CATALOG STAMP: every per-process verdict about a segment's cleanness
    (hot buffer, overrides, tombstones) is memoised under that stamp, so it must move when the
    truth does; the catalog's contents are unchanged, its mtime is the signal"""
    import os as _o
    try: _o.utime(_o.path.join(dbdir, 'catalog.json'), None)
    except Exception: pass
    try:
        import wdb_sidecar; wdb_sidecar.stamp_moved(dbdir)
    except Exception: pass

_MAGIC = b'WPRS2'
_OLD_MAGIC = b'WPRS1'


def table_of(seg_path):
    """x_3.wdb -> x ; the segment's table name by the naming law <table>_<n>.wdb"""
    base = os.path.basename(seg_path)
    stem = base[:-4] if base.endswith('.wdb') else base
    return stem.rsplit('_', 1)[0] if '_' in stem else stem


def path_for(seg_path):
    """the TABLE-level presence file this segment's mask lives in"""
    return os.path.join(os.path.dirname(seg_path), table_of(seg_path) + '.presence')


def _read_table(p):
    """{segment basename: bool array} or {} when the file is absent"""
    if not os.path.exists(p):
        return {}
    buf = open(p, 'rb').read()
    if buf[:5] == _OLD_MAGIC:                       # a legacy per-segment file under the table name: ignore
        return {}
    assert buf[:5] == _MAGIC, 'not a presence file: %s' % p
    out = {}; off = 5
    while off < len(buf):
        (nl,) = struct.unpack_from('<H', buf, off); off += 2
        name = buf[off:off + nl].decode(); off += nl
        N, nb = struct.unpack_from('<II', buf, off); off += 8
        bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8, offset=off, count=nb)); off += nb
        out[name] = bits[:N].astype(bool)
    return out


def _write_table(p, entries):
    """atomically: the rename law"""
    if not entries:
        if os.path.exists(p):
            os.remove(p)
        return
    parts = [_MAGIC]
    for name, pres in entries.items():
        pres = np.asarray(pres, dtype=bool); packed = np.packbits(pres.astype(np.uint8)).tobytes()
        nb = name.encode()
        parts.append(struct.pack('<H', len(nb)) + nb + struct.pack('<II', len(pres), len(packed)) + packed)
    tmp = p + '.partial'
    with open(tmp, 'wb') as f:
        f.write(b''.join(parts)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    _touch_stamp(os.path.dirname(p))


def _legacy(seg_path, N):
    """a pre-B per-segment file (<seg>.presence) still counts, read-only"""
    p = seg_path + '.presence'
    if not os.path.exists(p): return None
    buf = open(p, 'rb').read()
    if buf[:5] != _OLD_MAGIC: return None
    stored_N = struct.unpack_from('<I', buf, 5)[0]
    assert stored_N == N, 'presence sidecar row count %d != segment N %d' % (stored_N, N)
    return np.unpackbits(np.frombuffer(buf, dtype=np.uint8, offset=9))[:N].astype(bool)


def has(seg_path):
    """does this segment carry tombstones?"""
    if os.path.basename(seg_path) in _read_table(path_for(seg_path)): return True
    return os.path.exists(seg_path + '.presence')


def load(seg_path, N):
    """Return a bool array (len N, True=live), or None if no tombstones exist (all live)."""
    pres = _read_table(path_for(seg_path)).get(os.path.basename(seg_path))
    if pres is None:
        return _legacy(seg_path, N)
    assert len(pres) == N, 'presence row count %d != segment N %d' % (len(pres), N)
    return pres


def save(seg_path, presence):
    """Write this segment's mask into the table file atomically."""
    p = path_for(seg_path); entries = _read_table(p)
    entries[os.path.basename(seg_path)] = np.asarray(presence, dtype=bool)
    _write_table(p, entries)


def forget(seg_path):
    """drop this segment's entry (compaction replaced it)"""
    p = path_for(seg_path); entries = _read_table(p)
    if entries.pop(os.path.basename(seg_path), None) is not None:
        _write_table(p, entries)
    lp = seg_path + '.presence'
    if os.path.exists(lp):
        os.remove(lp)


def mark_deleted(seg_path, N, row_idx):
    """Tombstone the given row indices (creating the entry if absent). Returns rows newly tombstoned."""
    return mark_deleted_many({seg_path: (N, row_idx)})


def mark_deleted_many(updates):
    """THE ATOMIC DELETE: {seg_path: (N, row_idx)} for every segment of ONE table, tombstoned in a
    single write. Returns the total newly tombstoned."""
    if not updates: return 0
    p = path_for(next(iter(updates))); entries = _read_table(p); newly = 0
    for seg_path, (N, row_idx) in updates.items():
        name = os.path.basename(seg_path)
        pres = entries.get(name)
        if pres is None:
            pres = _legacy(seg_path, N)
            pres = np.ones(N, dtype=bool) if pres is None else pres.copy()
        row_idx = np.asarray(row_idx, dtype=np.int64)
        newly += int(pres[row_idx].sum())
        pres[row_idx] = False
        entries[name] = pres
    _write_table(p, entries)
    for seg_path in updates:                       # a legacy per-segment file is superseded
        lp = seg_path + '.presence'
        if os.path.exists(lp): os.remove(lp)
    return newly


def live_count(seg_path, N):
    presence = load(seg_path, N)
    return N if presence is None else int(presence.sum())
