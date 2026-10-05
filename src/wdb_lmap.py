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


def _build_stream(seg, col):
    """THE LOWERING COMPILED (2026-10-05): the dictionary's byte stream lowered in one parallel pass (ASCII and
    Cyrillic in the kernel; any other value -- other scripts, malformed bytes -- lowered by Python's str.lower,
    the reference, and spliced back), then one compiled dedupe gives each lowered value its group in first-seen
    order. The Python road decoded and lowered all 6M SearchPhrases every query: 10.4 s of GROUP BY LOWER."""
    import wdb_scalar, wdb_kernels as K
    st = wdb_scalar._stream(seg, col)
    if st is None:
        return None
    blob, off = st
    n = off.size - 1
    V = int(seg.cols[col]['V'])
    if n > V:
        return None
    low = np.empty(max(blob.size, 1), np.uint8)
    flag = np.empty(n, np.bool_)
    K.plower_ru(blob, off, low, flag)
    fi = np.flatnonzero(flag)
    if fi.size:
        repl = [_lower(blob[int(off[i]):int(off[i + 1])].tobytes()).encode('utf-8') for i in fi]
        lens = np.diff(off)
        lens[fi] = [len(r) for r in repl]
        off2 = np.zeros(n + 1, np.int64); np.cumsum(lens, out=off2[1:])
        blob2 = np.empty(max(int(off2[-1]), 1), np.uint8)
        K.pstream_splice(low, off, ~flag, off2, blob2)
        for i, r in zip(fi.tolist(), repl):
            blob2[int(off2[i]):int(off2[i]) + len(r)] = np.frombuffer(r, np.uint8)
    else:
        blob2, off2 = low, off
    # THE UNCHANGED ARE ALREADY DISTINCT: a value lowering leaves alone keeps its own group (the dictionary holds
    # each value once), so only the values lowering CHANGED need a group found -- by bisection in the sorted
    # dictionary (a lowered value equal to an untouched one joins it), the rest deduped among themselves. The
    # whole-dictionary hash (1.25 s, one thread, over 6M phrases) runs only when the dictionary is not in order.
    chg = np.empty(n, np.bool_)
    K.pdiff_vals(blob, low, off, chg)
    if fi.size:
        chg[fi] = [r != blob[int(off[i]):int(off[i + 1])].tobytes() for i, r in zip(fi.tolist(), repl)]
    srt = np.empty(n, np.uint8)
    if n:
        K.pslice_change(blob, np.ascontiguousarray(off[:-1]), np.ascontiguousarray(off[1:]), srt)
    lmap = np.empty(V, np.int32)
    if n and not (srt == 2).any():
        keep = ~chg
        rank = np.cumsum(keep) - 1
        U = int(keep.sum())
        reps = np.flatnonzero(keep).tolist()
        ci = np.flatnonzero(chg)
        g = rank.astype(np.int32)
        if ci.size:
            ql = off2[ci + 1] - off2[ci]
            qo = np.zeros(ci.size + 1, np.int64); np.cumsum(ql, out=qo[1:])
            qb = np.zeros(max(int(qo[-1]), 1), np.uint8)
            K.pgather_vals(blob2, off2, ci, qo, qb)
            hit = np.empty(ci.size, np.int64)
            K.pstream_find(blob, off, np.ascontiguousarray(qb), qo, hit)
            ok = (hit >= 0)
            ok[ok] = keep[hit[ok]]                 # joined only to a value lowering left alone
            g[ci[ok]] = rank[hit[ok]]
            rest = np.flatnonzero(~ok)
            if rest.size:
                ro = np.zeros(rest.size + 1, np.int64); np.cumsum(ql[rest], out=ro[1:])
                rb = np.zeros(max(int(ro[-1]), 1), np.uint8)
                K.pgather_vals(qb, qo, rest, ro, rb)
                lg = np.empty(rest.size, np.int64); lr = np.empty(rest.size, np.int64)
                LG = int(K.str_dedupe(np.ascontiguousarray(rb), ro, lg, lr))
                g[ci[rest]] = U + lg
                reps.extend(ci[rest[lr[:LG]]].tolist())
        lmap[:n] = g
    else:
        gid = np.empty(n, np.int64); rep = np.empty(max(n, 1), np.int64)
        G = int(K.str_dedupe(blob2, off2, gid, rep))
        lmap[:n] = gid
        reps = rep[:G].tolist()
    for k in range(n, V):                      # a code past the dictionary (the NULL code): a group of its own
        lmap[k] = len(reps); reps.append(k)
    return lmap, np.asarray(reps, np.int32)


def build(seg, col):
    """One decode-all (paid once, EVER), lower, group; atomic write; memmap back."""
    c = seg.cols[col]
    V = int(c['V'])
    got = _build_stream(seg, col)
    if got is not None:
        lmap, rep = got
    else:
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
