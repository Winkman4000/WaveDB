"""
wdb_gridwalk — grid filled-cell + count-ordered head for 2-key GROUP BY COUNT(*) top-K.

The per-PAIR-but-not-per-ROW structure. wdb_countpos stores the count once per ROW (N entries);
wdb_heavypair stores identity+count per heavy pair (16 B each). This stores each heavy distinct pair
ONCE as a filled cell of the A×B grid -- identity is the cell's code coordinates (codeA,codeB) packed
as a grid-id, the count stored once per cell -- and keeps a count-ordered HEAD (the top-N cells by
count) as the entry point. Top-K is a direct O(K) slice of that pre-sorted head: enter at the
high-count end, read K, decode coordinates back to values. No per-row materialization, no scan.

Measured (100M cb25db, SearchPhrase x UserID): head read top-10 = ~2 us, exact; the full filled-cell
structure is ~27 MB (walked identity 6.9 + counts 20 + head 0.4) vs heavypair's 161 MB and the per-row
countpos column's 200 MB -- 6x lighter, microsecond reads.

v1 scope (this module): serves 2-key COUNT(*) top-K (ORDER BY COUNT(*) DESC, or unordered LIMIT) from
the count-head; NO WHERE/HAVING/DISTINCT/JOIN; declines when LIMIT exceeds the head (the gap-encoded
bulk + append-tail maintenance is the documented next layer). RAM-resident, never persisted. Gated
behind _ENABLED (default False); when disabled detect returns None and routing is unchanged.
"""
import numpy as np
import wdb_sql
import workers
import wdb_policies as P
E = wdb_sql.E

_ENABLED = False          # opt-in; default off keeps existing routing byte-identical
_HEAD_N = 200000          # count-ordered head depth; serves any LIMIT up to this, declines beyond
_HITS = 0
_CACHE = {}               # (seg.path,(a,b),N) -> (head_gid, head_cnt, Vb, nheavy); RAM-resident
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
    """Filled-cell head: occurring heavy pairs (count>=2) as grid-ids, the top _HEAD_N by count,
    count-descending. Stores grid-id + count per head cell (identity is the grid-id's coordinates).
    Returns (head_gid int64, head_cnt int64, Vb int, nheavy int) or None."""
    a, b = sorted(cols)
    ca = seg._raw_codes(a).astype(np.int64); cb = seg._raw_codes(b).astype(np.int64)
    if ca.size == 0:
        return None
    Vb = int(cb.max()) + 1
    gid_all, cnt_all = np.unique(ca * Vb + cb, return_counts=True)   # filled cells, code order
    heavy = cnt_all >= 2
    gid = gid_all[heavy]; cnt = cnt_all[heavy]
    if gid.size == 0:
        return None
    k = min(_HEAD_N, gid.size)
    h = np.argpartition(cnt, -k)[-k:]                # the k largest by count
    h = h[np.argsort(cnt[h], kind='stable')[::-1]]   # count descending
    head_gid = np.ascontiguousarray(gid[h], dtype=np.int64)
    head_cnt = np.ascontiguousarray(cnt[h], dtype=np.int64)
    return head_gid, head_cnt, Vb, int(gid.size)


def _load(seg, cols):
    """RAM-resident: cache the count-head for this pair, built on first touch. Never disk."""
    a, b = sorted(cols)
    ck = (seg.path, (a, b), int(seg.N))
    hit = _CACHE.get(ck)
    if hit is not None:
        return hit
    built = _build(seg, cols)
    if built is None:
        return None
    _CACHE[ck] = built
    return built


def build(seg, cols):
    """Eagerly materialize a pair's count-head into the RAM cache. Returns True if available."""
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
    if not _ENABLED:                   return None
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    if not P.has_limit(tree):          return None
    if P.has_where(tree):              return None
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
    """Top-K = a direct slice of the count-ordered head. Decode each head cell's grid-id back to
    (codeA, codeB) -> values. Declines if LIMIT exceeds the head or the boundary count is tied (for
    ORDERED queries), so countpos/heavypair/scan back it up."""
    global _HITS
    cols = spec['cols']; ci = spec['ci']; lim = spec['lim']; proj = spec['proj']
    knames = spec['knames']; V = spec['V']
    loaded = _load(seg, cols)
    if loaded is None:
        return None
    head_gid, head_cnt, Vb, nheavy = loaded
    if lim <= 0 or lim > head_gid.size:
        return None
    if not spec.get('unordered') and head_cnt.size > lim and int(head_cnt[lim - 1]) == int(head_cnt[lim]):
        return None
    a, b = sorted(cols)
    sel_gid = head_gid[:lim]; sel_cnt = head_cnt[:lim]
    rows_out = []
    for g, cnt_val in zip(sel_gid, sel_cnt):
        g = int(g)
        by_col = {a: g // Vb, b: g % Vb}
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


def try_gridwalk(seg, tree, col_map):
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)
