"""
wdb_groupdistinct — group-wise exact COUNT(DISTINCT) via single-pass code hashing.

  SELECT key, COUNT(DISTINCT target) FROM t GROUP BY key [ORDER BY <count> DESC] [LIMIT N]

This is the non-foldable aggregate — you can't merge partial counts, you must remember every distinct
thing seen. WaveDB's edge: dictionary codes are distinctness-preserving, so distinct(target) ==
distinct(target_codes) with NO value decode. We pack (group_code, target_code) into one int64 and count
distinct pairs per group in a single pass using an open-addressing hash set: one probe per row, and a
re-seen pair is a no-op. Flips the group-wise-distinct class (the ClickBench Q08/Q09/Q13 shape) from a
~150s scalar fallback to sub-second, single-threaded and exact.

Same contract as the other operators: try_groupdistinct(seg, tree, col_map) -> (rows, colnames) | None
(None => caller falls through to the scan paths). Fail-closed on anything outside the exact shape.

v1 scope: single segment, no WHERE/HAVING/JOIN, projection exactly {bare key, COUNT(DISTINCT target)},
both key and target value-identity (mode 0/1/2, never mode-4 positional codes), group key not nullable,
ORDER BY the distinct-count DESC (or absent), optional LIMIT. Filters and co-aggregates are v2.
"""
import numpy as np
import wdb_sql
import wdb_gbcount
import wdb_policies as P
E = wdb_sql.E

_HITS = 0   # telemetry: queries answered by the group-distinct kernel

try:
    from numba import njit

    @njit(cache=True)
    def _walk(grp, tgt, k, gmax, null_tgt, capbits):
        cap = 1 << capbits
        mask = cap - 1
        table = np.full(cap, -1, np.int64)
        counts = np.zeros(gmax, np.int64)
        for i in range(grp.shape[0]):
            tv = tgt[i]
            if tv == null_tgt:                       # COUNT(DISTINCT) ignores NULL
                continue
            key = grp[i] * k + tv                    # collision-free pair id (grp dense, tgt < k)
            h = (key * np.int64(0x9E3779B1)) & mask
            while True:
                slot = table[h]
                if slot == -1:                       # new pair -> mark + count it for its group
                    table[h] = key
                    counts[grp[i]] += 1
                    break
                if slot == key:                      # already seen ("standing twice") -> no-op
                    break
                h = (h + 1) & mask
        return counts
    _HAVE_NUMBA = True
except Exception:                                    # numba absent: NumPy pack+unique fallback (still exact)
    _HAVE_NUMBA = False


def _ids(seg, col):
    """Per-row value-identity integer ids for `col`, plus (null_code or -1) and an optional by-code
    decode array. Returns (ids:int64, null_code:int, decode_or_None) or None if not value-identity."""
    c = seg.cols[col]
    if c['mode'] == 4:
        return None
    V = wdb_gbcount._code_values(seg, col)               # dict-backed (string dict, or high-card int)
    if V is not None:
        ids = np.asarray(seg.codes(col)).astype(np.int64, copy=False)
        null_code = (c['V'] - 1) if c.get('has_null') else -1
        return ids, null_code, V
    if c.get('dt') == 0 and c['mode'] in (0, 1):         # raw small-range int (e.g. RegionID): id == value
        ids = np.asarray(seg.values(col)).astype(np.int64, copy=False)
        null_code = -1                                   # raw ints carry no dict null sentinel here
        return ids, null_code, None
    return None


def _distinct_index(proj):
    """(index, colname) of the single COUNT(DISTINCT col) projection, or None if not exactly one."""
    found = None
    for i, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Count) and isinstance(inner.this, E.Distinct):
            dx = inner.this.expressions
            if len(dx) != 1 or not isinstance(dx[0], E.Column):
                return None
            if found is not None:
                return None
            found = (i, dx[0].name)
    return found


def _order_is_distinct_desc(tree, proj, ci):
    """True iff ORDER BY is absent, or its primary key is the distinct-count projection, descending."""
    order = tree.args.get('order')
    if order is None or not order.expressions:
        return True                                      # no ORDER BY: any order is acceptable pre-LIMIT
    first = order.expressions[0]
    if not isinstance(first, E.Ordered) or not first.args.get('desc'):
        return False
    tgt = first.this
    alias = wdb_sql._alias(proj[ci])
    if isinstance(tgt, E.Column) and tgt.name == alias:
        return True
    inner = tgt.this if isinstance(tgt, E.Alias) else tgt
    return isinstance(inner, E.Count) and isinstance(inner.this, E.Distinct)


def detect(seg, tree, col_map, _allow_group_filter=False):
    """Shape gate shared by the live walk and the materialized sidecar. Returns
    (kcol, tcol, ci, ki, proj) for a `GROUP BY key, COUNT(DISTINCT target)` query inside v1 scope,
    else None. kcol/tcol are col_map-resolved physical names; ci/ki index the distinct-count and the
    bare-key projections. Both serve paths must agree on eligibility, so neither duplicates this."""
    # --- shared shape guards (lifted to wdb_policies; the sidecar relaxes no_where to group-key-only) ---
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    if not (P.no_where(tree) or _allow_group_filter): return None
    if not P.single_group_key(tree):   return None
    group = tree.args.get('group')
    # --- distinct-family shape match (also EXTRACTS the column/projection indices, so it stays here) ---
    proj = tree.expressions
    if len(proj) != 2:
        return None
    di = _distinct_index(proj)
    if di is None:
        return None
    ci, tname = di                                       # ci = distinct-count projection index
    ki = 1 - ci
    kp = proj[ki]
    if wdb_sql._agg_kind(kp) is not None:                # the other projection must be the bare key
        return None
    knm = wdb_sql._proj_colname(kp)
    gnm = wdb_sql._colname(group.expressions[0])
    if knm is None or gnm is None or knm != gnm:
        return None
    if not _order_is_distinct_desc(tree, proj, ci):
        return None
    # --- resolve physical names, then shared segment/column guards ---
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    kcol = sc(knm); tcol = sc(tname)
    if not P.columns_exist(seg, kcol, tcol): return None
    if not P.no_deleted_rows(seg):           return None
    if not P.key_not_nullable(seg, kcol):    return None
    return kcol, tcol, ci, ki, proj


def try_groupdistinct(seg, tree, col_map):
    global _HITS
    det = detect(seg, tree, col_map)
    if det is None:
        return None
    kcol, tcol, ci, ki, proj = det

    kinfo = _ids(seg, kcol)
    tinfo = _ids(seg, tcol)
    if kinfo is None or tinfo is None:
        return None
    grp, _knull, kdecode = kinfo
    tgt, tnull, _tdecode = tinfo
    if grp.shape[0] != tgt.shape[0]:
        return None
    N = grp.shape[0]
    if N == 0:
        _HITS += 1
        return [], [wdb_sql._alias(p) for p in proj]

    k = int(tgt.max()) + 1
    gmax = int(grp.max()) + 1
    if gmax * k >= (1 << 62):                            # pack would overflow int64 -> decline
        return None

    if _HAVE_NUMBA:
        capbits = max(20, min(28, int(np.ceil(np.log2(max(N, 2)))) + 1))
        counts = _walk(grp, tgt, np.int64(k), gmax, np.int64(tnull), capbits)
    else:
        keep = (tgt != tnull) if tnull >= 0 else slice(None)
        key = grp[keep].astype(np.int64) * k + tgt[keep].astype(np.int64)
        uq = np.unique(key)
        counts = np.bincount((uq // k).astype(np.int64), minlength=gmax)

    lim = wdb_sql._limit(tree)
    if tnull >= 0:                                       # nullable target: a group whose targets are all
        present = np.nonzero(np.bincount(grp, minlength=gmax))[0]   # NULL has count 0 but still appears
    else:
        present = np.nonzero(counts)[0]                  # no nulls: count>0 exactly when the group occurs
    sel = present[np.argsort(-counts[present], kind='stable')]
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
    _HITS += 1
    return rows, names
