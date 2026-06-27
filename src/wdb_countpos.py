"""
wdb_countpos — per-row count-class column + presence-scan for 2-key GROUP BY COUNT(*) top-K.

The lighter, per-ROW alternative to wdb_heavypair's per-PAIR sidecar. Instead of storing the
occurring pairs (identity + count) sorted by count -- which grows with DISTINCT PAIRS -- we store
ONE value per row: the count of that row's pair (its "count-class"). The pair identities are NOT
stored; they already live in the row's own key columns. Reads are a PRESENCE-SCAN: to get the top-K
pairs we threshold the count column at the K-th largest distinct count, take the (few) rows at or
above it, read their pairs from the segment, dedupe in count order, and stop at K. Row order is
irrelevant -- it is a masked scan, not a sorted read -- which is exactly why maintenance needs no
resort: a new occurrence is a value-overwrite of that pair's rows (+1) in place.

Tradeoff vs heavypair (measured, 100M cb25db, UserID x SearchPhrase): footprint scales per-ROW
(~2 B/row uint16) instead of per-PAIR; read top-10 ~47 ms presence-scan vs heavypair's ~microsecond
pre-sorted front; maintenance a ~37 ms find+rewrite (no resort) vs a full rebuild. ~17x faster than
DuckDB on the read, with a per-row footprint that scales with the data, not the pair cardinality.

Same detect/execute contract as wdb_heavypair. Scope (v1): exactly two value-identity (non-mode-4)
keys, projections {bare k1, bare k2, COUNT(*)}, LIMIT N, NO WHERE/HAVING/DISTINCT/JOIN, ORDER BY
COUNT(*) DESC or no ORDER BY. Filtered shapes are declined (heavypair's kept-axis walk serves those).
Gated behind _ENABLED (default False) so the shipped heavypair path is untouched until countpos is
promoted; when disabled detect returns None and the query routes to heavypair exactly as before.
"""
import numpy as np
import wdb_sql
import workers
import wdb_policies as P
E = wdb_sql.E

_FMT = 1
_ENABLED = False          # opt-in; default off keeps the shipped heavypair path byte-identical
_HITS = 0
_CACHE = {}               # (seg.path,(a,b),N) -> (countcol, dcs); RAM-resident, never persisted
_VCACHE = {}              # (seg.path,col) -> by-code value dict (decode)


def enable():
    global _ENABLED
    _ENABLED = True


def disable():
    global _ENABLED
    _ENABLED = False


def is_enabled():
    return _ENABLED


def _vals(seg, col):
    """By-code value dictionary (sorted, indexable by code), or None for non-value-identity
    (mode-4) encodings we can't decode by rank. Cached per (segment, column)."""
    if seg.cols[col]['mode'] == 4:
        return None
    ck = (seg.path, col)
    hit = _VCACHE.get(ck)
    if hit is not None:
        return hit
    try:
        v = np.asarray(seg._typed_dict(col))
    except Exception:
        return None
    _VCACHE[ck] = v
    return v


def _build(seg, cols):
    """Per-row count-class column + sorted distinct-count list. Stores NO pair identity -- the
    identities stay in the segment's own key columns. Returns (countcol, dcs, N) or None.
    countcol[i] = count of the pair occurring in row i. dtype is the smallest that holds max count."""
    a, b = sorted(cols)
    ca = seg._raw_codes(a); cb = seg._raw_codes(b)
    if ca.size == 0:
        return None
    Vb = int(cb.max()) + 1
    key = ca.astype(np.int64) * Vb + cb.astype(np.int64)
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    mx = int(cnt.max())
    dt = np.uint16 if mx < 65536 else np.uint32
    countcol = np.ascontiguousarray(cnt[inv], dtype=dt)   # per-row count-class column
    dcs = np.ascontiguousarray(np.unique(cnt), dtype=np.int64)   # distinct counts ascending (tiny)
    return countcol, dcs, int(seg.N)


def _load(seg, cols):
    """RAM-resident only: return the cached (countcol, dcs) for this pair, building it into the heap
    on first touch. Never reads or writes disk -- the column lives in process memory for the lifetime
    of the database handle (disk stays compressed; the per-row footprint is small enough to hold)."""
    a, b = sorted(cols)
    ck = (seg.path, (a, b), int(seg.N))
    hit = _CACHE.get(ck)
    if hit is not None:
        return hit
    built = _build(seg, cols)
    if built is None:
        return None
    cc, dcs, n = built
    _CACHE[ck] = (cc, dcs)
    return _CACHE[ck]


def build(seg, cols):
    """Eagerly materialize a pair's count-class column into the RAM cache (the launch-time/eager
    path). Returns True if built/available. Use to pre-warm pairs at startup instead of on first
    query -- with live maintenance the one-time build cost is paid once and never blocks a read."""
    return _load(seg, cols) is not None


def _count_index(proj):
    ci = None
    for i, p in enumerate(proj):
        ak = wdb_sql._agg_kind(p)
        if ak is not None and ak[0] == 'COUNT_STAR':
            if ci is not None:
                return None
            ci = i
    return ci


def _order_is_count_desc(tree, proj, ci):
    order = tree.args.get('order')
    if order is None or not order.expressions:
        return False
    first = order.expressions[0]
    if not isinstance(first, E.Ordered) or not first.args.get('desc'):
        return False
    tgt = first.this
    alias = wdb_sql._alias(proj[ci])
    if isinstance(tgt, E.Column) and tgt.name == alias:
        return True
    ak = wdb_sql._agg_kind(tgt)
    return ak is not None and ak[0] == 'COUNT_STAR'


def detect(seg, tree, col_map):
    if not _ENABLED:                   return None    # opt-in; default off -> route to heavypair
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    if not P.has_limit(tree):          return None
    if P.has_where(tree):              return None    # filtered shapes -> heavypair's kept-axis walk
    group = tree.args.get('group')
    if group is None or len(group.expressions) != 2:
        return None
    proj = tree.expressions
    if len(proj) != 3:
        return None
    ci = _count_index(proj)
    if ci is None:
        return None
    key_proj = [p for i, p in enumerate(proj) if i != ci]
    if any(wdb_sql._agg_kind(p) is not None for p in key_proj):
        return None
    knames = [wdb_sql._proj_colname(p) for p in key_proj]
    gnames = [wdb_sql._colname(g) for g in group.expressions]
    if any(k is None for k in knames) or any(g is None for g in gnames):
        return None
    if set(knames) != set(gnames):
        return None
    cols = [col_map.get(k, k) if col_map else k for k in knames]
    for col in cols:
        if not P.columns_exist(seg, col):  return None
        if not P.not_positional(seg, col): return None
    if not P.no_deleted_rows(seg):         return None
    order = tree.args.get('order')
    if order is None or not order.expressions:
        unordered = True
    elif _order_is_count_desc(tree, proj, ci):
        unordered = False
    else:
        return None
    Vs = {col: _vals(seg, col) for col in cols}
    if any(v is None for v in Vs.values()):
        return None
    return {'cols': cols, 'ci': ci, 'lim': wdb_sql._limit(tree), 'proj': proj,
            'knames': knames, 'V': Vs, 'order': tree.args.get('order'), 'unordered': unordered}


def execute(seg, spec):
    """Presence-scan top-K. Threshold the count column at the (lim+1)-th largest distinct count,
    take the rows at/above it, read their pairs from the segment's own key columns, dedupe in
    count-descending order, stop at lim. No pair-identity store is touched -- the identities come
    from the rows. Declines (returns None) if the heavy front can't satisfy lim, so heavypair/scan
    backs it up."""
    global _HITS
    cols = spec['cols']; ci = spec['ci']; lim = spec['lim']; proj = spec['proj']
    knames = spec['knames']; V = spec['V']
    loaded = _load(seg, cols)
    if loaded is None:
        return None
    countcol, dcs = loaded
    if lim <= 0 or dcs.size == 0:
        return None

    # threshold = the (lim+1)-th largest distinct count (buffer of 1 to test the tie boundary).
    # lim distinct count VALUES guarantee >= lim distinct PAIRS (a pair has exactly one count), so
    # this front always holds enough -- unless total distinct pairs < lim, caught below.
    need = min(lim + 1, dcs.size)
    thr = int(dcs[-need])
    rows = np.flatnonzero(countcol >= thr)
    if rows.size == 0:
        return None

    a, b = sorted(cols)
    ca = seg._raw_codes(a); cb = seg._raw_codes(b)
    Vb = int(cb.max()) + 1
    cand_keys = ca[rows].astype(np.int64) * Vb + cb[rows].astype(np.int64)
    cand_cnt = countcol[rows].astype(np.int64)

    # dedupe candidates to distinct pairs in count-descending order (stable): first occurrence in
    # the count-desc ordering is each pair's (single) count.
    order = np.argsort(cand_cnt, kind='stable')[::-1]
    ok = cand_keys[order]
    uvals, first_idx = np.unique(ok, return_index=True)   # first occ of each key, in array order
    so = np.argsort(first_idx)                             # back to count-desc appearance order
    uniq_keys = uvals[so]
    uniq_cnt = cand_cnt[order][first_idx[so]]
    if uniq_keys.size < lim:
        return None

    # tie straddling the LIMIT matters only for ORDERED top-N (which of the equally-counted pairs
    # are "top N" is ambiguous) -- defer to the scan, matching heavypair. Unordered: any N is legal.
    if not spec.get('unordered') and uniq_cnt.size > lim and int(uniq_cnt[lim - 1]) == int(uniq_cnt[lim]):
        return None

    sel_keys = uniq_keys[:lim]; sel_cnt = uniq_cnt[:lim]
    rows_out = []
    for k, cnt_val in zip(sel_keys, sel_cnt):
        k = int(k)
        by_col = {a: k // Vb, b: k % Vb}
        row = [None] * len(proj)
        row[ci] = int(cnt_val)
        for pi, p in enumerate(proj):
            if pi == ci:
                continue
            knm = wdb_sql._proj_colname(p)
            col = cols[knames.index(knm)]
            row[pi] = wdb_sql._pyval(V[col][int(by_col[col])])
        rows_out.append(tuple(row))
    rows_out = workers.finalize(rows_out, proj, spec['order'], lim)
    _HITS += 1
    return rows_out, [wdb_sql._alias(p) for p in proj]


def try_countpos(seg, tree, col_map):
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)


# --- maintenance: promote a pair by one occurrence. No resort -- the read is a presence-scan, so
# the column is never order-bearing; a bump is a pure value-overwrite of that pair's rows in place.
# (Provided for the live-update path; the static board does not exercise it.) ---

def promote(seg, cols, code_a, code_b, delta=1):
    """Bump the count-class of every row carrying (code_a, code_b) by delta, in place. Returns the
    number of rows touched. O(one scan to locate + the touched rows); never a resort or rebuild."""
    loaded = _load(seg, cols)
    if loaded is None:
        return 0
    countcol, dcs = loaded
    a, b = sorted(cols)
    ca = seg._raw_codes(a); cb = seg._raw_codes(b)
    r = np.flatnonzero((ca == code_a) & (cb == code_b))
    if r.size == 0:
        return 0
    new_val = int(countcol[r[0]]) + delta
    if new_val >= np.iinfo(countcol.dtype).max:           # widen if a bump overflows the dtype
        countcol = countcol.astype(np.uint32)
        _CACHE[(seg.path, (a, b), int(seg.N))] = (countcol, dcs)
    countcol[r] = new_val
    if new_val not in dcs:                                 # a new distinct count -> extend the tiny list
        dcs2 = np.unique(np.append(dcs, new_val))
        _CACHE[(seg.path, (a, b), int(seg.N))] = (countcol, dcs2)
    return int(r.size)
