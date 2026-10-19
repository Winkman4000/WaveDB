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
import workers
import wdb_groupdistinct as gd
import wdb_policies as P

E = wdb_sql.E
_HITS = 0   # telemetry: queries answered by the multi-aggregate group-distinct kernel
_SIDECAR_HITS = 0   # telemetry: distinct component served from a materialized gd sidecar (no walk)


def _sidecar_counts(db, table, segment_path, kcol, tcol, gmax):
    """Per-group distinct counts from a fully-materialized gd sidecar (indexed by group code), or None.
    Lets the multi-aggregate path (Q09) skip the distinct walk when the (kcol,tcol) pair is materialized.
    Declines on a trimmed sidecar -- groupmix has no WHERE, so it needs every group's count."""
    global _SIDECAR_HITS
    if db is None or table is None or segment_path is None:
        return None
    try:
        import wdb_gdsidecar  # noqa: F401  (ensures module loaded; sidecar lives on db)
        entry = db.cat.gd_entry(table, kcol, tcol)
        if entry is None or entry.get('excluded'):
            return None
        s = db.gd_sidecar(segment_path, kcol, tcol)
        if s is None:
            return None
        cnts = s['counts']
        if cnts is None or len(cnts) < gmax:
            return None
        _SIDECAR_HITS += 1
        return cnts
    except Exception:
        return None


def _has_sidecar(db, table, kcol, tcol):
    """a materialized (key, target) distinct shelf is registered: the stored answer path is asked first"""
    try:
        e = db.cat.gd_entry(table, kcol, tcol) if db is not None and table is not None else None
        return e is not None and not e.get('excluded')
    except Exception:
        return False


def detect(seg, tree, col_map):
    """Returns (kcol, tcol, ci, ki, folds, proj) or None. folds = list of (proj_index, kind, phys_col, dt);
    ci/ki index the COUNT(DISTINCT) and the bare key; kcol/tcol are col_map-resolved physical names."""
    # --- shared shape guards (wdb_policies) ---
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_where(tree):           return None
    if not P.no_having(tree):          return None
    if not P.single_group_key(tree):   return None
    group = tree.args.get('group')
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
            knm = wdb_sql._proj_colname(p)
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
    # --- shared segment/column guards (wdb_policies) ---
    if not P.columns_exist(seg, kcol, tcol): return None
    if not P.no_deleted_rows(seg):           return None
    if not P.key_not_nullable(seg, kcol):    return None
    return kcol, tcol, ci, key_index, folds, proj


def execute(seg, det, tree, db=None, table=None, segment_path=None):
    """THE READ: group + foldable aggs + one distinct in a single pass. Prefers a materialized
    sidecar for the distinct, else the code-hashing walk. May decline (None) on measured
    conditions (missing ids, pack overflow)."""
    global _HITS
    kcol, tcol, ci, ki, folds, proj = det

    import wdb_kernels as _WK
    import wdb_window as WN
    kc = seg.cols[kcol]
    if kc.get('mode') in (0, 1, 2) and not kc.get('has_null'):
        # THE KEY IS ITS CODE (Q09's line items): the group id is the dictionary code as stored --
        # no value array (a 100M fancy-index), no astype (630 ms of the 1,781). Decoded at emission.
        grp = np.ascontiguousarray(np.asarray(seg._raw_codes(kcol)))
        gmax = int(kc['V'])
        import wdb_gbcount as _GB
        kdecode = _GB._code_values(seg, kcol)
        if kdecode is None:
            kdecode = np.asarray(WN._int_table(seg, kcol), dtype=np.int64)
    else:
        kinfo = gd._ids(seg, kcol)
        if kinfo is None:
            return None
        grp, _knull, kdecode = kinfo
        gmax = int(grp.max()) + 1 if grp.shape[0] else 0
    N = grp.shape[0]
    names = [wdb_sql._alias(p) for p in proj]
    if N == 0:
        _HITS += 1
        return [], names

    # ---- foldables: per-group reductions on the CODE STREAMS (THE FOLD ON CODES) ----
    count = _WK.bincount_par(grp, gmax)
    if kc.get('mode') in (0, 1, 2) and not kc.get('has_null') and LATE and not _has_sidecar(db, table, kcol, tcol):
        late = _late_rows(seg, tree, proj, folds, ci, ki, tcol, grp, gmax, count, kdecode)
        if late is not None:
            _HITS += 1
            return late, names
    sums = {}                                            # proj_index -> per-group sum (float64)
    for (i, kind, pc, dt) in folds:
        if kind[0] in ('SUM', 'AVG'):
            fc = seg.cols[pc]
            if fc.get('mode') in (0, 1, 2) and dt == 0 and gmax <= (1 << 20):
                dv = np.ascontiguousarray(np.asarray(WN._int_table(seg, pc), dtype=np.float64))
                if dv.size < int(fc['V']):
                    dv = np.concatenate([dv, np.zeros(int(fc['V']) - dv.size, np.float64)])
                out = np.empty(gmax, np.float64)
                _WK.group_fold_dict(grp, np.ascontiguousarray(np.asarray(seg._raw_codes(pc))), dv,
                                    np.int64(gmax), out)
                sums[i] = out
            else:
                vals = np.asarray(seg.values(pc)).astype(np.float64, copy=False)
                sums[i] = np.bincount(grp, weights=vals, minlength=gmax)

    # ---- non-foldable distinct: prefer a materialized sidecar, else the code-hashing walk ----
    key_on_codes = kc.get('mode') in (0, 1, 2) and not kc.get('has_null')
    import wdb_gbcount as _GB2
    raw_space_shelf = key_on_codes and _GB2._code_values(seg, kcol) is None
    distinct = _sidecar_counts(db, table, segment_path, kcol, tcol, 0 if raw_space_shelf else gmax)
    if distinct is not None and raw_space_shelf:
        # the shelf was born in the RAW-VALUE id space (gd._ids for a small-range int key); our
        # groups are codes now -- translate through the int table, exactly
        tab = np.asarray(WN._int_table(seg, kcol), dtype=np.int64)
        dcnt = np.asarray(distinct, dtype=np.int64)
        ok = (tab >= 0) & (tab < dcnt.size)
        distinct = np.where(ok, dcnt[np.clip(tab, 0, max(0, dcnt.size - 1))], 0)
    if distinct is None:                                 # not materialized -> decode target and walk
        tc = seg.cols[tcol]
        t_codes = False
        if tc.get('mode') in (0, 1, 2) and not tc.get('has_null'):
            tgt = np.ascontiguousarray(np.asarray(seg._raw_codes(tcol))); tnull = -1   # codes as stored, no astype
            t_codes = True
        else:
            tinfo = gd._ids(seg, tcol)
            if tinfo is None:
                return None
            tgt, tnull, _td = tinfo
            t_codes = _td is not None
        if grp.shape[0] != tgt.shape[0]:
            return None
        if t_codes and tc.get('V'):
            k = int(tc['V'])                             # codes: [0, V), no min / max passes
        else:
            if tgt.dtype.kind == 'i' and tgt.size and int(tgt.min()) < 0:
                lo9 = int(tgt.min())                     # raw-value targets below zero: shift onto >= 0
                tgt = tgt.astype(np.int64) - lo9
                tnull = tnull - lo9 if tnull >= 0 else -1
            k = int(tgt.max()) + 1
        if gmax * k >= (1 << 62):                        # pair-id would overflow int64 -> decline
            return None
        # THE Q13 BOARD'S LANES, shared (2026-10-19, the Q9 board): dense / scatter / keymark / keysort with
        # the heavy groups counted apart -- the serial code-hashing walk served every key that was not
        # small x wide (ClientEventTime x RefererHash 21 s; OriginalURL keys past the board's 60 s)
        _lane, runs = gd.lane_runs(grp, tgt, tnull, int(gmax), k)
        if runs is None:
            return None
        distinct = np.maximum(runs, 0)

    present = np.nonzero(count)[0]                        # a group occurs iff it has >=1 row
    present = _order_cut(present, proj, folds, tree, ci, ki, count, sums, distinct)

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

    rows = workers.finalize(rows, proj, tree.args.get('order'), wdb_sql._limit(tree))
    _HITS += 1
    return rows, names


LATE = True                # the late fold first (tests turn it off to compare)
LATE_MAX_GROUPS = 4096     # past this many selected groups (a multi-term ORDER BY tying a plateau): the full fold
LATE_HITS = 0
LATE_SHARE = 8             # the late fold reads at most 1/8 of the rows


def _order_term(o0, proj):
    """(projection index, descending) of an ORDER BY term naming a projection by alias or by expression"""
    if not isinstance(o0, E.Ordered):
        return None, False
    tgt = o0.this
    tsql = tgt.sql() if hasattr(tgt, 'sql') else None
    for i, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        if (isinstance(tgt, E.Column) and tgt.name == wdb_sql._alias(p)) or (tsql is not None and inner.sql() == tsql):
            return i, bool(o0.args.get('desc'))
    return None, False


def _late_rows(seg, tree, proj, folds, ci, ki, tcol, grp, gmax, count, kdecode):
    """THE LATE FOLD (2026-10-19, the Q9 board): when the ORDER BY needs only the row counts or the key, the
    LIMIT's groups are chosen first, and every other aggregate -- the sums, the averages, the distinct
    count -- is computed only over those groups' rows. Q9 orders by COUNT(*): a 100M-group key (WatchID)
    folded two sums over 100M groups and walked 100M distinct runs to return 10 rows; a small key reads its
    ten biggest groups' rows instead of all. One ORDER BY term: exactly `lim` groups by (value, code) --
    at a tie the lowest codes, a legal answer; more terms: the whole plateau tied with the lim-th value.
    None (the full fold answers) on anything else."""
    global LATE_HITS
    import wdb_kernels as _WK
    import wdb_window as WN
    lim = wdb_sql._limit(tree)
    order = tree.args.get('order')
    if lim is None or order is None or not order.expressions:
        return None
    pi, desc = _order_term(order.expressions[0], proj)
    if pi is None:
        return None
    kind = 'KEY' if pi == ki else next((f[1][0] for f in folds if f[0] == pi), None)
    if kind not in ('KEY', 'COUNT_STAR'):
        return None
    need = int(lim) + int(wdb_sql._offset(tree) or 0)
    if need <= 0:
        return None
    single = len(order.expressions) == 1
    count = np.asarray(count)
    if kind == 'COUNT_STAR' and desc and single:
        sel = np.asarray(_WK.gd_topk(count, -1, need), np.int64)      # count desc, code asc
        sel = sel[count[sel] > 0]
    else:
        present = np.flatnonzero(count)
        v = count[present].astype(np.float64) if kind == 'COUNT_STAR' else present.astype(np.float64)
        if desc:
            v = -v
        if present.size > need:
            kth = np.partition(v, need - 1)[need - 1]
            above = np.flatnonzero(v < kth)
            on = np.flatnonzero(v == kth)
            if single:
                on = on[:max(0, need - above.size)]
            sel = present[np.concatenate([above, on])]
        else:
            sel = present
    if sel.size > LATE_MAX_GROUPS:
        return None
    sel = np.sort(sel).astype(np.int64)
    G = int(sel.size)
    N = grp.shape[0]
    if G == 0:
        LATE_HITS += 1
        return []
    if int(count[sel].sum()) * LATE_SHARE > N:
        return None                                      # the chosen groups hold much of the table (Q9's ten
                                                         # biggest regions): gathering their rows from every
                                                         # column costs more than the full fold (0.62 -> 1.05 s)
    rows, sub = _WK.rows_of_groups(grp, sel, int(gmax))
    small = rows.size * 64 <= N                          # few rows: read the other columns at them only

    def codes_at(col):
        return np.asarray(seg.codes_at(col, rows)) if small else np.asarray(seg._raw_codes(col))[rows]

    sums = {}
    for (i, kd, pc, dt) in folds:
        if kd[0] in ('SUM', 'AVG'):
            fc = seg.cols[pc]
            if fc.get('mode') in (0, 1, 2) and dt == 0:
                dv = np.asarray(WN._int_table(seg, pc), dtype=np.float64)
                if dv.size < int(fc['V']):
                    dv = np.concatenate([dv, np.zeros(int(fc['V']) - dv.size, np.float64)])
                vals = dv[codes_at(pc).astype(np.int64)]
            else:
                vals = np.asarray(seg.values(pc)).astype(np.float64, copy=False)[rows]
            sums[i] = np.bincount(sub, weights=vals, minlength=G)
    tc = seg.cols[tcol]
    if tc.get('mode') in (0, 1, 2):
        tgt = np.ascontiguousarray(codes_at(tcol))
        tnull = (int(tc['V']) - 1) if tc.get('has_null') else -1
        k = int(tc['V'])
    else:
        tinfo = gd._ids(seg, tcol)
        if tinfo is None:
            return None
        tgt = np.ascontiguousarray(np.asarray(tinfo[0])[rows]); tnull = tinfo[1]
        if tgt.dtype.kind == 'i' and tgt.size and int(tgt.min()) < 0:
            lo9 = int(tgt.min()); tgt = tgt.astype(np.int64) - lo9
            tnull = tnull - lo9 if tnull >= 0 else -1
        k = int(tgt.max()) + 1 if tgt.size else 1
    _lane, runs = gd.lane_runs(np.ascontiguousarray(sub), tgt, tnull, G, k)
    if runs is None:
        return None
    distinct = np.maximum(runs, 0)
    out = []
    for j, g in enumerate(sel.tolist()):
        row = [None] * len(proj)
        row[ki] = wdb_sql._pyval(kdecode[g] if kdecode is not None else np.int64(g))
        row[ci] = int(distinct[j])
        cg = int(count[g])
        for (i, kd, pc, dt) in folds:
            if kd[0] == 'COUNT_STAR':
                row[i] = cg
            elif kd[0] == 'SUM':
                s = sums[i][j]
                row[i] = int(round(s)) if dt == 0 else float(s)
            elif kd[0] == 'AVG':
                row[i] = float(sums[i][j] / cg) if cg else None
        out.append(tuple(row))
    LATE_HITS += 1
    return workers.finalize(out, proj, tree.args.get('order'), wdb_sql._limit(tree))


def _order_cut(present, proj, folds, tree, ci, ki, count, sums, distinct):
    """THE ORDER BEFORE THE ROWS (2026-10-19): with a LIMIT, only the groups that can be returned become
    Python rows (a 100M-group key built 100M of them). The first ORDER BY term is evaluated per group as an
    array; with one term the cut is exact -- the first `lim` by (value, code), so a tie at the LIMIT keeps the
    lowest codes, a legal answer -- with more terms every group tied with the lim-th value is kept for
    finalize to order. Anything else (an expression, no LIMIT): every present group, as before."""
    lim = wdb_sql._limit(tree)
    order = tree.args.get('order')
    if lim is None or order is None or not order.expressions or present.size <= 8 * max(int(lim), 1):
        return present
    lim = int(lim) + int(wdb_sql._offset(tree) or 0)
    o0 = order.expressions[0]
    if not isinstance(o0, E.Ordered):
        return present
    tgt = o0.this
    tsql = tgt.sql() if hasattr(tgt, 'sql') else None
    pi = None
    for i, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        if (isinstance(tgt, E.Column) and tgt.name == wdb_sql._alias(p)) or (tsql is not None and inner.sql() == tsql):
            pi = i
            break
    if pi is None:
        return present
    if pi == ci:
        v = np.asarray(distinct, np.float64)[present]
    elif pi == ki:
        v = present.astype(np.float64)                   # value-sorted codes: code order is value order
    else:
        f = [x for x in folds if x[0] == pi]
        if not f:
            return present
        kind = f[0][1][0]
        cg = np.asarray(count)[present].astype(np.float64)
        if kind == 'COUNT_STAR':
            v = cg
        elif kind == 'SUM':
            v = np.asarray(sums[pi])[present]
        else:
            v = np.asarray(sums[pi])[present] / np.maximum(cg, 1)
    if o0.args.get('desc'):
        v = -v
    if np.isnan(v).any():
        return present
    kth = np.partition(v, lim - 1)[lim - 1]
    above = np.flatnonzero(v < kth)
    on = np.flatnonzero(v == kth)
    if len(order.expressions) == 1:
        on = on[:max(0, lim - above.size)]               # present is ascending: the lowest codes first
    return present[np.sort(np.concatenate([above, on]))]


def try_groupmix(seg, tree, col_map, db=None, table=None, segment_path=None):
    """Detect + execute, kept as the backward-compatible single-call entry."""
    det = detect(seg, tree, col_map)
    if det is None:
        return None
    return execute(seg, det, tree, db=db, table=table, segment_path=segment_path)
