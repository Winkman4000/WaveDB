"""
wdb_gdsidecar -- materialized group-wise COUNT(DISTINCT) answer (the "sidecar").

The default shape comes free from the dictionary: encode already enumerated every group, so the
group cardinality V (a ~1us header read) sets the sidecar to cover all groups, one count each. Build
runs the exact single-pass walk ONCE (reusing wdb_groupdistinct), and stores the per-group distinct
counts indexed by group code. Serving a `GROUP BY g, COUNT(DISTINCT t)` query then READS the stored
counts instead of re-walking N rows -- exact, and orders of magnitude faster on repeat.

Group labels are NOT stored: they come back free from the segment's own dictionary at serve time
(codes 0..V-1 -> values). Per-group trim (which groups the sidecar covers) lives in the catalog and
is applied at serve time; trimmed-out groups fall through to the live walk.

Storage = V counts (+ the present-index when the target is nullable). Keyed by (group_col, target_col),
persisted next to the segment as <segment>.gd-<group>-<target>.npz.
"""
import os, json
import numpy as np
import wdb_groupdistinct as gd
import wdb_sql

E = wdb_sql.E


def sidecar_path(segment_path, group_col, target_col):
    return f"{segment_path}.gd-{group_col}-{target_col}.npz"


def build(seg, group_col, target_col):
    """Run the exact walk once over this segment; return {counts, present, meta} or None.

    Declines (returns None) on exactly the shapes the live walk declines: non-value-identity
    columns (mode-4 positional), nullable group key (v1), or a pack that would overflow int64.
    counts[code] = exact distinct(target) for the group with that code; present = the group codes
    that actually occur (for a nullable target a group can occur with count 0)."""
    if group_col not in seg.cols or target_col not in seg.cols:
        return None
    if seg.cols[group_col].get('has_null'):              # v1: NULL-as-a-group deferred (matches operator)
        return None
    kinfo = gd._ids(seg, group_col)
    tinfo = gd._ids(seg, target_col)
    if kinfo is None or tinfo is None:                   # not value-identity -> walk can't serve it
        return None
    grp, _knull, _kdec = kinfo
    tgt, tnull, _tdec = tinfo
    if grp.shape[0] != tgt.shape[0]:
        return None
    N = int(grp.shape[0])
    gmax = (int(grp.max()) + 1) if N else 0
    k = (int(tgt.max()) + 1) if N else 1
    if N and gmax * k >= (1 << 62):                      # pair-id would overflow int64 -> decline
        return None

    if N == 0:
        counts = np.zeros(gmax, np.int64)
    elif gd._HAVE_NUMBA:
        capbits = max(20, min(28, int(np.ceil(np.log2(max(N, 2)))) + 1))
        counts = gd._walk(grp, tgt, np.int64(k), gmax, np.int64(tnull), capbits)
    else:                                                # numba absent: exact NumPy pack+unique
        keep = (tgt != tnull) if tnull >= 0 else slice(None)
        key = grp[keep].astype(np.int64) * k + tgt[keep].astype(np.int64)
        uq = np.unique(key)
        counts = np.bincount((uq // k).astype(np.int64), minlength=gmax)
    if tnull >= 0:                                        # nullable target: occurs != count>0
        present = np.nonzero(np.bincount(grp, minlength=gmax))[0].astype(np.int64)
    else:
        present = np.nonzero(counts)[0].astype(np.int64)
    meta = {'group_col': group_col, 'target_col': target_col,
            'N': N, 'V': int(gmax), 'target_nullable': bool(tnull >= 0)}
    return {'counts': counts.astype(np.int64, copy=False), 'present': present, 'meta': meta}


def save(segment_path, sc):
    p = sidecar_path(segment_path, sc['meta']['group_col'], sc['meta']['target_col'])
    tmp = p + '.tmp'
    np.savez(tmp, counts=sc['counts'], present=sc['present'],
             meta=np.frombuffer(json.dumps(sc['meta']).encode(), dtype=np.uint8))
    os.replace(tmp + '.npz' if os.path.exists(tmp + '.npz') else tmp, p)  # np.savez appends .npz
    return p


def load(segment_path, group_col, target_col):
    p = sidecar_path(segment_path, group_col, target_col)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    meta = json.loads(bytes(z['meta']).decode())
    return {'counts': z['counts'], 'present': z['present'], 'meta': meta}


def rows_from_sidecar(sc, seg, proj, tree, ci, ki, excluded_codes=None):
    """Assemble result rows from the stored counts -- the read that replaces the walk. `excluded_codes`
    is the per-pair trim (group codes the user left to the walk); they are dropped from the candidate
    set here. Row assembly / ORDER BY / LIMIT reuse the exact same helpers as try_groupdistinct, so the
    served rows are identical to the live operator's."""
    counts = sc['counts']
    present = sc['present']
    if excluded_codes is not None and len(excluded_codes):
        ex = np.asarray(sorted(excluded_codes), dtype=np.int64)
        present = present[~np.isin(present, ex)]
    kdecode = gd._ids(seg, sc['meta']['group_col'])[2]   # labels free from the live dict (None => code==value)
    sel = present[np.argsort(-counts[present], kind='stable')]
    lim = wdb_sql._limit(tree)
    if lim is not None:
        sel = sel[:lim]
    names = [wdb_sql._alias(p) for p in proj]
    rows = []
    for gid in sel.tolist():
        row = [None, None]
        row[ki] = wdb_sql._pyval(kdecode[gid] if kdecode is not None else np.int64(gid))
        row[ci] = int(counts[gid])
        rows.append(tuple(row))
    rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))
    if lim is not None:
        rows = rows[:lim]
    return rows, names


_SERVE_HITS = 0   # telemetry: queries answered from the materialized sidecar (not the walk)


def try_serve(db, table, seg, segment_path, tree, col_map):
    """Serve a group-distinct query from the materialized sidecar, or return None to fall through to
    the live walk. Fires only when: (a) the query is the value-identity GROUP BY..COUNT(DISTINCT) shape
    (shared `detect` with the walk), (b) the pair is registered materialized in the catalog, (c) the
    sidecar file exists for this segment, and (d) the needed groups are fully covered. In v1 there is no
    WHERE, so a query needs EVERY group -- any trim therefore falls through to the walk for exact parity
    with the live operator. (Subset/filtered serving over the kept groups is v2.)"""
    global _SERVE_HITS
    det = gd.detect(seg, tree, col_map)
    if det is None:
        return None
    kcol, tcol, ci, ki, proj = det
    entry = db.cat.gd_entry(table, kcol, tcol)
    if entry is None:                                    # pair not materialized -> walk
        return None
    s = load(segment_path, kcol, tcol)
    if s is None:                                        # registered but no data on disk -> walk
        return None
    if entry.get('excluded'):                            # v1: trimmed groups still needed here -> walk
        return None
    rows, names = rows_from_sidecar(s, seg, proj, tree, ci, ki, excluded_codes=None)
    _SERVE_HITS += 1
    return rows, names


def codes_for_values(seg, group_col, values):
    """Map group VALUES (what a human sees in the view) to internal group codes, for trimming. Raw-int
    mode-0 columns: code == value. Dict columns: look the value up in the dict. Unknown values are
    skipped (can't trim a group that isn't there)."""
    decode = gd._ids(seg, group_col)[2]
    if decode is None:                                   # raw int: code is the value
        return sorted(int(v) for v in values)
    inv = {}
    for code, val in enumerate(decode):
        key = val.decode() if isinstance(val, (bytes, bytearray)) else val
        inv[key] = code
    out = []
    for v in values:
        key = v.decode() if isinstance(v, (bytes, bytearray)) else v
        if key in inv:
            out.append(int(inv[key]))
    return sorted(out)


def inspect(seg, segment_path, group_col, target_col, excluded=None):
    """The materialized view a human eyeballs: (group_value, distinct_count) rows, descending by count.
    Reads the stored sidecar -- no walk. `excluded` codes are flagged so you can see what's trimmed."""
    s = load(segment_path, group_col, target_col)
    if s is None:
        return None
    counts = s['counts']; present = s['present']
    decode = gd._ids(seg, group_col)[2]
    ex = set(excluded or [])
    order = present[np.argsort(-counts[present], kind='stable')]
    rows = []
    for code in order.tolist():
        val = decode[code] if decode is not None else code
        val = val.decode() if isinstance(val, (bytes, bytearray)) else val
        rows.append((val, int(counts[code]), code in ex))
    return rows
