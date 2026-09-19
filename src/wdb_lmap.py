"""wdb_lmap: the lower-collation map -- old_code -> lower_gid, as a file.

LOWER(col) GROUP BY doesn't want the column's values; it wants to know which
codes COLLIDE under lowering. That is a derived fact about the dictionary,
computed once and memmapped forever (nline doctrine: the file is the memory).
24MB for SearchPhrase's 6M codes instead of a 417MB value store.

Format (.lmap.<col>, little-endian):
  magic 'WLM1' | uint32 V | uint32 G | int32 lmap[V] | int32 rep[G]
lmap[code] = gid of the code's lowered value; rep[gid] = one original code of
the group (any member: its lowered bytes ARE the group's emitted key).
Lowering matches the engine's scalar path exactly (wdb_sql: utf-8 'replace'
decode -> str.lower()) so board answers stay bit-identical.
"""
import os
import numpy as np

_MAGIC = b'WLM1'


def _path(seg, col):
    return seg.path + '.lmap.' + col


def _lower(v):
    if isinstance(v, (bytes, bytearray)):
        return v.decode('utf-8', 'replace').lower()
    return str(v).lower()


def load(seg, col):
    """Memmapped (lmap, rep) or None. Validates V against the live dict."""
    p = _path(seg, col)
    try:
        if not os.path.exists(p):
            return None
        with open(p, 'rb') as f:
            head = f.read(12)
        if len(head) < 12 or head[:4] != _MAGIC:
            return None
        V = int(np.frombuffer(head, np.uint32, 1, 4)[0])
        G = int(np.frombuffer(head, np.uint32, 1, 8)[0])
        if V != int(seg.cols[col]['V']):
            return None                            # stale sidecar: dict changed
        mm = np.memmap(p, dtype=np.int32, mode='r', offset=12)
        if mm.size < V + G:
            return None
        return mm[:V], mm[V:V + G]
    except Exception:
        return None


def build(seg, col):
    """One decode-all (paid once, EVER), lower, group; atomic write; memmap back."""
    c = seg.cols[col]
    V = int(c['V'])
    dv = seg._typed_dict(col)
    lmap = np.empty(V, np.int32)
    groups = {}
    rep = []
    for i, v in enumerate(dv):
        lv = _lower(v)
        g = groups.get(lv)
        if g is None:
            g = len(groups)
            groups[lv] = g
            rep.append(i)
        lmap[i] = g
    p = _path(seg, col)
    import wdb_sidecar
    if not wdb_sidecar.births_on(os.path.dirname(p)):
        return lmap, np.asarray(rep, np.int32)          # THE SWITCH: the same pair, in RAM
    tmp = p + '.tmp.%d' % os.getpid()
    with open(tmp, 'wb') as f:
        f.write(_MAGIC)
        f.write(np.uint32(V).tobytes())
        f.write(np.uint32(len(rep)).tobytes())
        f.write(lmap.tobytes())
        f.write(np.asarray(rep, np.int32).tobytes())
    os.replace(tmp, p)
    return load(seg, col)


def load_or_build(seg, col):
    got = load(seg, col)
    return got if got is not None else build(seg, col)
