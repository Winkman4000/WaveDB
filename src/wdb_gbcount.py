"""
wdb_gbcount — pre-aggregated count projection for high-cardinality single-key GROUP BY COUNT(*).

A filter-free `SELECT key, COUNT(*) FROM t GROUP BY key ORDER BY COUNT(*) DESC LIMIT N` never needs a
scan: the per-group counts are fixed between writes. We persist them once (sorted by count, heavy
hitters only — singletons are an implicit count of 1), so the query becomes a top-N read instead of a
full-table scan + high-cardinality accumulator scatter.

This is the cube idea extended to high-card keys. try_cube runs first and answers the low-card cases
from a small dense cube; try_gbcount runs right after and answers the high-card single-COUNT(*) case
the cube declines. Same contract as wdb_cube: try_gbcount(seg, tree, col_map) -> (rows, colnames) or
None (caller falls through to the scan paths). Fail-closed on anything outside its exact shape.

Scope (v1): single segment (inherited from the caller's len(segs)==1 gate), one non-mode-4 key whose
codes are value-identity, projections exactly {bare key, COUNT(*)}, no WHERE/HAVING/DISTINCT/JOIN,
ORDER BY the COUNT descending + LIMIT N within the stored heavy-hitter set, no deleted rows. The
sidecar is built lazily on first eligible query and persisted next to the segment as <seg>.<col>.gbc,
keyed by column — general across any table/column. Staleness-guarded by seg.N.
"""
import os, pickle, numpy as np
import wdb_sql
import wdb_policies as P
E = wdb_sql.E

_HITS = 0   # telemetry: queries answered from a count projection
_CACHE = {}  # (seg.path, col, N) -> (codes, counts), so a repeated query never re-reads the sidecar


def _path(seg, col):
    return f"{seg.path}.{col}.gbc"


def _code_values(seg, col):
    """The column's by-code value dictionary (engine-cached, indexable by code), or None if the
    column isn't a value-identity dictionary we can decode (mode-4 affine, etc.). Returns the RAW
    dict (no per-value conversion) so the caller decodes only the few codes it actually emits."""
    c = seg.cols[col]
    if c['mode'] == 4:
        return None
    if c['dt'] == 1:                                    # string dict (modes 0/1): list of bytes
        try:
            return seg.dict_vals(col)
        except Exception:
            return None
    if c['dt'] == 0 and c['mode'] == 2:                 # high-card int: sorted int64 dictionary
        try:
            return seg._dict_ints(c)
        except Exception:
            return None
    return None                                         # other encodings: let the scan path handle it (v1)


def _build(seg, col):
    """Per-group counts, sorted by count descending, heavy hitters (count>=2) only. Returns
    (codes uint32, counts int64, K, N) or None. The count is already computed when the dictionary is
    built at encode time; here we just (re)materialize and persist it."""
    codes = seg._raw_codes(col)
    if codes.size == 0:
        return None
    K = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=K)
    order = np.argsort(counts, kind='stable')[::-1]      # count descending
    keep = counts[order] >= 2                            # singletons are implicit (count 1), don't store
    hc = np.ascontiguousarray(order[keep], dtype=np.uint32)
    hn = np.ascontiguousarray(counts[order][keep], dtype=np.int64)
    return hc, hn, K, int(seg.N)


def _load(seg, col):
    """Load the persisted sidecar (rebuilding if absent or stale vs seg.N, caching in memory).
    Returns (codes, counts) or None if the column can't be projected."""
    ck = (seg.path, col, int(seg.N))
    hit = _CACHE.get(ck)
    if hit is not None:
        return hit
    p = _path(seg, col)
    if os.path.exists(p):
        try:
            hc, hn, K, n = pickle.load(open(p, 'rb'))
            if n == int(seg.N):
                _CACHE[ck] = (hc, hn)
                return hc, hn
        except Exception:
            pass
    built = _build(seg, col)
    if built is None:
        return None
    hc, hn, K, n = built
    try:
        pickle.dump((hc, hn, K, n), open(p, 'wb'), protocol=4)
    except Exception:
        pass
    _CACHE[ck] = (hc, hn)
    return hc, hn


def _count_index(proj):
    """Index of the single COUNT(*) projection, or None if not exactly one."""
    ci = None
    for i, p in enumerate(proj):
        ak = wdb_sql._agg_kind(p)
        if ak is not None and ak[0] == 'COUNT_STAR':
            if ci is not None:
                return None
            ci = i
    return ci


def _order_is_count_desc(tree, proj, ci):
    """True iff the primary ORDER BY is the COUNT(*) projection, descending — which makes the
    count-descending sidecar prefix contain the answer."""
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


def try_gbcount(seg, tree, col_map):
    """Answer a filter-free high-card `key, COUNT(*) GROUP BY key ORDER BY COUNT(*) DESC LIMIT N`
    from the persisted count projection, or return None to fall through to the scan paths."""
    global _HITS
    # --- shared shape guards (wdb_policies); filter-free COUNT(*) top-N ---
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_where(tree):           return None
    if not P.no_having(tree):          return None
    if not P.single_group_key(tree):   return None
    if not P.has_limit(tree):          return None      # only the bounded top-N shape
    group = tree.args.get('group')
    lim = wdb_sql._limit(tree)
    proj = tree.expressions
    if len(proj) != 2:
        return None
    ci = _count_index(proj)
    if ci is None:
        return None
    ki = 1 - ci
    kp = proj[ki]
    if wdb_sql._agg_kind(kp) is not None:               # the other projection must be the bare key
        return None
    knm = wdb_sql._colname(kp.this if isinstance(kp, E.Alias) else kp)
    gnm = wdb_sql._colname(group.expressions[0])
    if knm is None or gnm is None or knm != gnm:
        return None
    col = col_map.get(knm, knm) if col_map else knm
    # --- shared segment/column guards (wdb_policies) ---
    if not P.columns_exist(seg, col):  return None
    if not P.not_positional(seg, col): return None
    if not P.no_deleted_rows(seg):     return None      # deleted rows make stored counts stale
    if not _order_is_count_desc(tree, proj, ci):
        return None
    V = _code_values(seg, col)
    if V is None:
        return None
    loaded = _load(seg, col)
    if loaded is None:
        return None
    hc, hn = loaded
    if lim > hn.size:                                   # would need singletons (count 1): fall through
        return None
    if lim < hn.size and int(hn[lim - 1]) == int(hn[lim]):
        return None                                     # a tie straddles the LIMIT boundary: the top-N
                                                        # SET is ambiguous -> defer to the scan path so
                                                        # tie-breaking stays consistent with the engine
    rows = []
    for code, n in zip(hc[:lim].tolist(), hn[:lim].tolist()):
        row = [None, None]
        row[ki] = wdb_sql._pyval(V[code])               # decode only the N emitted keys
        row[ci] = int(n)
        rows.append(tuple(row))
    rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))[:lim]
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in proj]
