"""
wdb_groupmix -- single-pass GROUP BY with foldable co-aggregates + one COUNT(DISTINCT).

  SELECT key, <foldable aggs...>, COUNT(DISTINCT target) FROM t GROUP BY key [ORDER BY ...] [LIMIT n]

The ClickBench Q09 shape: one group key carrying SUM / COUNT(*) / AVG alongside a single non-foldable
COUNT(DISTINCT). The foldables are per-group reductions (np.bincount over the group codes); the distinct
rides the exact same code-hashing walk as wdb_groupdistinct. One pass over the segment, exact -- flips the
Q09 class off the >150s scalar fallback.

try_groupmix(seg, tree, col_map) -> (rows, colnames) | None (None => caller falls through).
v1 scope: single segment, no WHERE/HAVING/JOIN; one value-identity group key (not nullable); projection =
{bare key} + >=1 foldable in {COUNT(*), SUM(numeric), AVG(numeric)} + exactly one COUNT(DISTINCT target)
(value-identity). SUM/AVG columns must be non-null (so AVG's denominator is COUNT(*)). ORDER BY any projected
aggregate/key, optional LIMIT. MIN/MAX/COUNT(col) decline (fall through) in v1.
"""
import numpy as np
import wdb_sql
import wdb_groupdistinct as gd

E = wdb_sql.E
_HITS = 0   # telemetry: queries answered by the multi-aggregate group-distinct kernel


def _detect(seg, tree, col_map):
    """Returns (kcol, tcol, ci, ki, folds, proj) or None. folds = list of (proj_index, kind, phys_col, dt);
    ci/ki index the COUNT(DISTINCT) and the bare key; kcol/tcol are col_map-resolved physical names."""
    if tree.args.get('joins') or tree.args.get('distinct') is not None:
        return None
    if tree.args.get('where') is not None or tree.args.get('having') is not None:
        return None
    group = tree.args.get('group')
    if group is None or len(group.expressions) != 1:
        return None
    proj = tree.expressions
    if len(proj) < 3:                                    # need key + >=1 foldable + 1 distinct
        return None
    di = gd._distinct_index(proj)                        # the single COUNT(DISTINCT col), or None
    if di is None:
        return None
    ci, tname = di
    gnm = wdb_sql._colname(group.expressions[0])
    if gnm is None:
        return None
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    key_index = None
    folds = []                                           # (proj_index, kind, phys_col_or_None, dt_or_None)
    for i, p in enumerate(proj):
        if i == ci:
            continue
        kind = wdb_sql._agg_kind(p)
        if kind is None:                                 # must be exactly the bare group key
            knm = wdb_sql._colname(p.this if isinstance(p, E.Alias) else p)
            if knm is None or knm != gnm or key_index is not None:
                return None
            key_index = i
        elif kind[0] == 'COUNT_STAR':
            folds.append((i, kind, None, None))
        elif kind[0] in ('SUM', 'AVG'):
            if kind[1] is None:
                return None
            pc = sc(kind[1])
            c = seg.cols.get(pc)
            if c is None or c.get('dt') not in (0, 2) or c.get('has_null'):   # numeric, non-null only (v1)
                return None
            folds.append((i, kind, pc, c['dt']))
        else:                                            # COUNT(col) / MIN / MAX -> v1 decline
            return None
    if key_index is None or not folds:
        return None
    kcol = sc(gnm); tcol = sc(tname)
    if kcol not in seg.cols or tcol not in seg.cols:
        return None
    if seg.presence_mask() is not None:
        return None
    if seg.cols[kcol].get('has_null'):
        return None
    return kcol, tcol, ci, key_index, folds, proj


def try_groupmix(seg, tree, col_map):
    global _HITS
    det = _detect(seg, tree, col_map)
    if det is None:
        return None
    kcol, tcol, ci, ki, folds, proj = det

    kinfo = gd._ids(seg, kcol)
    tinfo = gd._ids(seg, tcol)
    if kinfo is None or tinfo is None:
        return None
    grp, _knull, kdecode = kinfo
    tgt, tnull, _td = tinfo
    if grp.shape[0] != tgt.shape[0]:
        return None
    N = grp.shape[0]
    names = [wdb_sql._alias(p) for p in proj]
    if N == 0:
        _HITS += 1
        return [], names
    gmax = int(grp.max()) + 1
    k = int(tgt.max()) + 1
    if gmax * k >= (1 << 62):                            # pair-id would overflow int64 -> decline
        return None

    # ---- foldables: per-group reductions (vectorized) ----
    count = np.bincount(grp, minlength=gmax)
    sums = {}                                            # proj_index -> per-group sum (float64)
    for (i, kind, pc, dt) in folds:
        if kind[0] in ('SUM', 'AVG'):
            vals = np.asarray(seg.values(pc)).astype(np.float64, copy=False)
            sums[i] = np.bincount(grp, weights=vals, minlength=gmax)

    # ---- non-foldable distinct: the single-pass code-hashing walk (reused) ----
    if gd._HAVE_NUMBA:
        capbits = max(20, min(28, int(np.ceil(np.log2(max(N, 2)))) + 1))
        distinct = gd._walk(grp, tgt, np.int64(k), gmax, np.int64(tnull), capbits)
    else:
        keep = (tgt != tnull) if tnull >= 0 else slice(None)
        key = grp[keep].astype(np.int64) * k + tgt[keep].astype(np.int64)
        uq = np.unique(key)
        distinct = np.bincount((uq // k).astype(np.int64), minlength=gmax)

    present = np.nonzero(count)[0]                        # a group occurs iff it has >=1 row

    # ---- assemble in projection order ----
    rows = []
    for g in present.tolist():
        row = [None] * len(proj)
        row[ki] = wdb_sql._pyval(kdecode[g] if kdecode is not None else np.int64(g))
        row[ci] = int(distinct[g])
        cg = int(count[g])
        for (i, kind, pc, dt) in folds:
            if kind[0] == 'COUNT_STAR':
                row[i] = cg
            elif kind[0] == 'SUM':
                s = sums[i][g]
                row[i] = int(round(s)) if dt == 0 else float(s)   # SUM(int)->int, SUM(float)->float
            elif kind[0] == 'AVG':
                row[i] = float(sums[i][g] / cg) if cg else None   # AVG always float (non-null col)
        rows.append(tuple(row))

    rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))
    lim = wdb_sql._limit(tree)
    if lim is not None:
        rows = rows[:lim]
    _HITS += 1
    return rows, names
