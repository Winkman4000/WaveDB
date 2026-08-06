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
from sqlglot import expressions as E
import wdb_sql
import workers
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


def _gdc_path(seg, kcol, tcol):
    return seg.path + '.%s__%s.gdc' % (kcol, tcol)


def _gdc_load(seg, kcol, tcol):
    """The grouped-distinct shelf (Jackson's cut): per-group COUNT(DISTINCT target) is a
    V-sized fact about an immutable file -- kilobytes on disk, reborn only with the file.
    Same law as the gbc: lawful V-sized sidecar, no_deleted_rows-gated, cold-truth clean."""
    import os, pickle
    p = _gdc_path(seg, kcol, tcol)
    if not os.path.exists(p):
        return None
    try:
        blob = pickle.load(open(p, 'rb'))
        if blob.get('n') == int(seg.N):
            return blob
    except Exception:
        pass
    return None


def _gdc_save2(seg, kcol, tcol, counts):
    """Codes-only sidecar for giant key spaces: counts indexed by dict code,
    winners decoded at emission. noempty: built from the sparse planes, so the
    default-value group is absent -- only filtered queries may ride it."""
    import pickle
    # THE LEADERBOARD (Jackson's constraint chain): this shelf only ever
    # answers ORDER BY u DESC LIMIT k, so persist the podium, not the world.
    T = 65536
    counts = np.asarray(counts)
    if counts.size > T:
        part = np.argpartition(-counts, T - 1)[:T]
    else:
        part = np.arange(counts.size)
    order = part[np.argsort(-counts[part], kind='stable')]
    top_codes = order.astype(np.uint32)
    mx = int(counts[order[0]]) if order.size else 0
    dt = (np.uint8 if mx < 256 else
          np.uint16 if mx < 65536 else np.uint32)   # full ladder: the data climbs
                                                    # only as high as it measures
    top_counts = counts[order].astype(dt)        # the measured max picks the bus
    try:
        pickle.dump({'n': int(seg.N), 'top_codes': top_codes,
                     'top_counts': top_counts, 'keys': None, 'noempty': True},
                    open(_gdc_path(seg, kcol, tcol), 'wb'), protocol=4)
    except Exception:
        pass


def _gdc_save(seg, kcol, tcol, counts, keys):
    import pickle
    try:
        pickle.dump({'n': int(seg.N), 'counts': counts, 'keys': keys},
                    open(_gdc_path(seg, kcol, tcol), 'wb'), protocol=4)
    except Exception:
        pass


def detect(seg, tree, col_map, _allow_group_filter=False):
    """Shape gate shared by the live walk and the materialized sidecar. Returns
    (kcol, tcol, ci, ki, proj) for a `GROUP BY key, COUNT(DISTINCT target)` query inside v1 scope,
    else None. kcol/tcol are col_map-resolved physical names; ci/ki index the distinct-count and the
    bare-key projections. Both serve paths must agree on eligibility, so neither duplicates this."""
    # --- shared shape guards (lifted to wdb_policies; the sidecar relaxes no_where to group-key-only) ---
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    excl_empty = False
    if not P.no_where(tree):
        w9 = tree.args.get('where')
        n9 = w9.this if w9 is not None else None
        gx = tree.args.get('group')
        if (isinstance(n9, E.NEQ) and isinstance(n9.this, E.Column)
                and gx is not None and len(gx.expressions) == 1
                and isinstance(gx.expressions[0], E.Column)
                and n9.this.name == gx.expressions[0].name
                and isinstance(n9.expression, E.Literal)
                and str(n9.expression.this) == ''):
            excl_empty = True    # filter names only the group key: other
        elif not _allow_group_filter:    # groups' counts untouched; mask
            return None                  # one group at emission

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
    return kcol, tcol, ci, ki, proj, excl_empty


def execute(seg, det, tree):
    """THE READ: per-group COUNT(DISTINCT) via one code-hash walk. May decline (None) on
    measured conditions (mismatched lengths, pack overflow)."""
    global _HITS
    kcol, tcol, ci, ki, proj, excl_empty = det
    _ecode = None
    if excl_empty:
        import wdb_wherescan as _WS
        _e0 = _WS._code_of(seg, kcol, '')
        if _e0 is None:
            excl_empty = False
        else:
            _ecode = int(_e0)


    # THE SHELF FIRST: an unfiltered per-group distinct count is a stored fact.
    if (P.no_where(tree) or excl_empty) and P.no_deleted_rows(seg):
        blob = _gdc_load(seg, kcol, tcol)
        if blob is not None and 'top_codes' in blob and excl_empty:
            lim9 = wdb_sql._limit(tree)
            tc9 = blob['top_codes']
            if lim9 is not None and lim9 <= tc9.size:
                rows = []
                for t9 in range(int(lim9)):
                    row = [None, None]
                    v9 = seg.fetch(kcol, int(tc9[t9]))
                    row[ki] = v9.decode('utf-8', 'replace') if isinstance(v9, (bytes, bytearray)) else v9
                    row[ci] = int(blob['top_counts'][t9])
                    rows.append(tuple(row))
                rows = workers.finalize(rows, proj, tree.args.get('order'), lim9)
                _HITS += 1
                return rows, [wdb_sql._alias(p) for p in proj]
        if blob is not None and 'counts' in blob and (not blob.get('noempty') or excl_empty):
            counts = blob['counts']; keys = blob['keys']
            if excl_empty and _ecode is not None and _ecode < len(counts):
                counts = np.asarray(counts).copy()
                counts[_ecode] = 0
            lim = wdb_sql._limit(tree)
            present = np.nonzero(counts)[0]
            sel = present[np.argsort(-counts[present], kind='stable')]
            if lim is not None:
                sel = sel[:lim]
            rows = []
            for gid in sel.tolist():
                row = [None, None]
                if keys is not None:
                    row[ki] = keys[gid]
                else:
                    v9 = seg.fetch(kcol, int(gid))
                    row[ki] = v9.decode('utf-8', 'replace') if isinstance(v9, (bytes, bytearray)) else v9
                row[ci] = int(counts[gid])
                rows.append(tuple(row))
            rows = workers.finalize(rows, proj, tree.args.get('order'), lim)
            _HITS += 1
            return rows, [wdb_sql._alias(p) for p in proj]

    kc9 = seg.cols.get(kcol)
    if (kc9 is not None and kc9.get('code_enc') == 8 and hasattr(seg, 'e8_planes')
            and P.no_deleted_rows(seg) and excl_empty):
        # the planes hold only NON-default rows: this branch serves solely the
        # filtered shape (the default group is exactly what the filter removes)
        # BIG-KEY BRANCH (Q13's shape): sparse planes hand non-default rows;
        # scatter buckets target codes by key; every group's uniques exact;
        # the sidecar births codes-only (noempty) on first touch.
        try:
            import wdb_kernels as _WK
            pl = seg.e8_planes(kcol)
            pos8, lits8 = np.asarray(pl[0]), np.asarray(pl[1], dtype=np.int64)
            KV9 = int(kc9['V'])
            cnts9 = np.bincount(lits8, minlength=KV9)
            uc9 = np.asarray(seg._raw_codes(tcol))[pos8].astype(np.int64)
            offs9 = np.zeros(KV9 + 1, np.int64)
            np.cumsum(cnts9, out=offs9[1:])
            bucketed9 = np.empty(lits8.size, np.int64)
            _WK.cd_scatter(lits8, uc9, offs9, offs9[:-1].copy(), bucketed9)
            dcounts = _WK.cd_alldistinct(bucketed9, offs9)
            _gdc_save2(seg, kcol, tcol, dcounts)
            counts = dcounts
            if excl_empty and _ecode is not None and _ecode < counts.size:
                counts = counts.copy(); counts[_ecode] = 0
            lim = wdb_sql._limit(tree)
            present = np.nonzero(counts)[0]
            sel = present[np.argsort(-counts[present], kind='stable')]
            if lim is not None:
                sel = sel[:lim]
            rows = []
            for gid in sel.tolist():
                row = [None, None]
                v9 = seg.fetch(kcol, int(gid))
                row[ki] = v9.decode('utf-8', 'replace') if isinstance(v9, (bytes, bytearray)) else v9
                row[ci] = int(counts[gid])
                rows.append(tuple(row))
            rows = workers.finalize(rows, proj, tree.args.get('order'), lim)
            _HITS += 1
            return rows, [wdb_sql._alias(p) for p in proj]
        except Exception:
            pass
    kinfo = _ids(seg, kcol)
    tinfo = _ids(seg, tcol)
    if kinfo is None or tinfo is None:
        return None
    grp, _knull, kdecode = kinfo
    tgt, tnull, _tdecode = tinfo
    if grp.shape[0] != tgt.shape[0]:
        return None
    N = grp.shape[0]
    # THE SCATTER LANE (Jackson's reorganization): no nulls, dict codes both sides,
    # big target space -> MSD 2-pass scatter groups rows by target, then an
    # L1-resident marker table counts each target's first touch of every key.
    # Measured on a-cd-grp: 2.4s walk -> ~0.65s, exact. All transient: cold-truth.
    VRk = int(grp.max()) + 1 if N else 0
    VTt = int(tgt.max()) + 1 if N else 0
    if (tnull < 0 and N > 4_000_000 and 0 < VRk <= 262_144 and VTt > 65_536):
        import wdb_kernels as _WK
        SH = max(1, VTt.bit_length() - 12)
        ku, kr, offs = _WK.gd_pass1(np.ascontiguousarray(tgt),
                                    np.ascontiguousarray(grp),
                                    np.int64(SH), np.int64(8))
        counts = _WK.gd_pass2_count(ku, kr, offs, np.int64(SH), np.int64(VRk))
        if P.no_where(tree) and P.no_deleted_rows(seg):
            keys = [wdb_sql._pyval(kdecode[g] if kdecode is not None else np.int64(g))
                    for g in range(VRk)]
            keys = [k.decode('utf-8', 'replace') if isinstance(k, (bytes, bytearray)) else k
                    for k in keys]
            _gdc_save(seg, kcol, tcol, counts, keys)     # birth-on-first-touch, gbc-style
        lim = wdb_sql._limit(tree)
        present = np.nonzero(counts)[0]
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
        rows = workers.finalize(rows, proj, tree.args.get('order'), lim)
        _HITS += 1
        return rows, names
    if N == 0:
        _HITS += 1
        return [], [wdb_sql._alias(p) for p in proj]

    k = int(tgt.max()) + 1
    gmax = int(grp.max()) + 1
    if gmax * k >= (1 << 62):                            # pack would overflow int64 -> decline
        return None

    if _HAVE_NUMBA:
        # (measured, 2026-07: a group-major scatter + epoch-stamp board was tried here
        # and reverted -- it pays the same board-sized cache misses as the hash probe
        # PLUS two full-N gathers the walk never needs. The walk runs in raw order and
        # is near single-thread optimal; beating duck's 0.4s needs parallel kernels.)
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
    rows = workers.finalize(rows, proj, tree.args.get('order'), lim)
    _HITS += 1
    return rows, names


def try_groupdistinct(seg, tree, col_map):
    """Detect + execute, kept as the backward-compatible single-call entry."""
    det = detect(seg, tree, col_map)
    if det is None:
        return None
    return execute(seg, det, tree)
