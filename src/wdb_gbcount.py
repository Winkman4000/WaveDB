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
import wdb_qmem
import wdb_sql
import workers
import wdb_policies as P
E = wdb_sql.E

_HITS = 0   # telemetry: queries answered from a count projection
_CACHE = wdb_qmem.register({})  # (seg.path, col, N) -> (codes, counts), so a repeated query never re-reads the sidecar


def _path(seg, col):
    return f"{seg.path}.{col}.gbc"


def _fetchable(seg, col):
    """Can this column's values be point-fetched by code (seg.fetch)? A pure capability
    check -- touches NO data. The old version materialized the ENTIRE dictionary here
    (dict_vals: 18.3M string reconstructions for URL) so the emit could index ten codes.
    The rule: only decode when needed, never more than needed -- the emit point-fetches."""
    c = seg.cols[col]
    if c['mode'] == 4:
        return False
    if c['dt'] == 1 and c['mode'] in (0, 1):            # string dict: restart-walk fetch
        return True
    if c['dt'] == 0 and c['mode'] == 2:                 # high-card int: nline/dict fetch
        return True
    if c['dt'] == 0 and c['mode'] in (0, 1):            # int-valued byte dict: plain point fetch
        return True                                     # (the j-dim-grp inner: RegionID dt0 mode0)
    return False


class _FetchDecoder:
    """Lazy by-code decoder: indexable like a materialized dictionary, but every [code]
    is a point-fetch (restart-walk for strings, nline/dict for ints). The rule: only
    decode when needed, never more than needed. len() is n_dict -- metadata, no data."""
    __slots__ = ('_seg', '_col', '_n')

    def __init__(self, seg, col):
        self._seg = seg; self._col = col
        c = seg.cols[col]
        self._n = int(c.get('n_dict') or c.get('V') or 0)

    def __len__(self):
        return self._n

    def __getitem__(self, code):
        return self._seg.fetch(self._col, int(code))


def _code_values(seg, col):
    """Shared by gdsidecar/groupdistinct/survgroup: a LAZY by-code decoder (or None).
    Materializes nothing -- the old version decoded ENTIRE dictionaries here."""
    return _FetchDecoder(seg, col) if _fetchable(seg, col) else None


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


def detect(seg, tree, col_map):
    """ACTIVATION for the count-projection read. A pure decision over query shape +
    segment metadata -- touches no row data. Returns a spec dict the read needs, or
    None to decline. Self-validating, so it's robust called on its own (the controller
    calls this to route; try_gbcount calls it too)."""
    # --- shared shape guards (wdb_policies); filter-free COUNT(*) top-N ---
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    excl_lit = None
    w = tree.args.get('where')
    if w is not None:
        # admit exactly: key <> literal (one conjunct) -- served by cell exclusion
        import sqlglot.expressions as E
        cj = w.this
        if not isinstance(cj, E.NEQ) or not isinstance(cj.expression, E.Literal):
            return None
        if wdb_sql._colname(cj.this) is None:
            return None
        excl_lit = cj.expression.this
        excl_key = wdb_sql._colname(cj.this)
    having_min = None
    h = tree.args.get('having')
    if h is not None:
        # admit exactly: COUNT(*) > lit (lit >= 2: the heavy list is complete there)
        import sqlglot.expressions as E
        hc_ = h.this
        if not isinstance(hc_, E.GT) or wdb_sql._agg_kind(hc_.this) is None:
            return None
        if wdb_sql._agg_kind(hc_.this)[0] != 'COUNT_STAR':
            return None
        if not isinstance(hc_.expression, E.Literal):
            return None
        having_min = float(hc_.expression.this)
        if having_min < 2:
            return None
    if not P.single_group_key(tree):   return None
    # bounded top-N is the home shape; the UNBOUNDED full-counts shape is also
    # servable when the dictionary is small (heavy list + implicit singletons =
    # the complete answer) -- born of join orientation's inner rewrite, which
    # asks for full single-key counts with no limit
    unbounded = not P.has_limit(tree)
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
    knm = wdb_sql._proj_colname(kp)
    gnm = wdb_sql._colname(group.expressions[0])
    if knm is None or gnm is None or knm != gnm:
        return None
    if excl_lit is not None and excl_key != knm:
        return None                                     # exclusion must be on the key itself
    col = col_map.get(knm, knm) if col_map else knm
    # --- shared segment/column guards (wdb_policies) ---
    if not P.columns_exist(seg, col):  return None
    if not P.not_positional(seg, col): return None
    if not P.no_deleted_rows(seg):     return None      # deleted rows make stored counts stale
    if unbounded:
        if tree.args.get('order') is not None:
            return None                                 # unbounded serve emits any order
        col_ = col_map.get(knm, knm) if col_map else knm
        c_ = seg.cols.get(col_)
        if c_ is None or int(c_.get('V') or 1 << 30) > 65536:
            return None                                 # big dicts: unbounded stays scan-side
    elif not _order_is_count_desc(tree, proj, ci):
        return None
    if not _fetchable(seg, col):                        # capability only: NO data touched
        return None
    return {'col': col, 'ci': ci, 'ki': ki, 'lim': lim, 'proj': proj,
            'order': tree.args.get('order'), 'excl_lit': excl_lit,
            'having_min': having_min, 'unbounded': unbounded}


def execute(seg, spec):
    """THE READ: pull the persisted count projection and emit the top-N rows. May still
    decline (return None) on measured boundary conditions that need the loaded sidecar --
    a LIMIT past the stored heavy hitters, or a tie straddling the LIMIT boundary."""
    global _HITS
    col = spec['col']; ci = spec['ci']; ki = spec['ki']; lim = spec['lim']
    proj = spec['proj']
    loaded = _load(seg, col)
    if loaded is None:
        return None
    hc, hn = loaded
    if spec.get('excl_lit') is not None:
        import wdb_wherescan as WS
        kc = WS._code_of(seg, col, spec['excl_lit'])
        if kc is not None:
            keep = hc != int(kc)                        # one cell out; count-desc order kept
            hc, hn = hc[keep], hn[keep]
    if spec.get('unbounded'):
        V = int(seg.cols[col]['V'])
        cnt = np.ones(V, np.int64)                      # dict codes appear >= 1;
        cnt[hc] = hn                                    # absent from heavy == exactly 1
        hm2 = spec.get('having_min')
        rows = []
        for code in range(V):
            n = int(cnt[code])
            if hm2 is not None and not n > hm2:
                continue
            row = [None, None]
            row[ki] = wdb_sql._pyval(seg.fetch(col, code))
            row[ci] = n
            rows.append(tuple(row))
        _HITS += 1
        return rows, [wdb_sql._alias(p) for p in proj]
    hm = spec.get('having_min')
    if hm is not None:
        qual = int(np.count_nonzero(hn > hm))           # hn is count-desc: a clean prefix
        hc, hn = hc[:qual], hn[:qual]
        lim = min(lim, qual)                            # fewer qualifiers than LIMIT is a
        if lim == 0:                                    # legitimate short answer, not a decline
            _HITS += 1
            return [], [wdb_sql._alias(p) for p in proj]
    if hm is None and lim > hn.size:                    # would need singletons: fall through
        return None
    if lim < hn.size and int(hn[lim - 1]) == int(hn[lim]):
        return None                                     # a tie straddles the LIMIT boundary: the top-N
                                                        # SET is ambiguous -> defer to the scan path so
                                                        # tie-breaking stays consistent with the engine
    rows = []
    for code, n in zip(hc[:lim].tolist(), hn[:lim].tolist()):
        row = [None, None]
        row[ki] = wdb_sql._pyval(seg.fetch(col, int(code)))   # decode ONLY the N emitted keys:
        row[ci] = int(n)                                      # point-fetch, never the dictionary
        rows.append(tuple(row))
    rows = workers.finalize(rows, proj, spec['order'], lim)
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in proj]


def try_gbcount(seg, tree, col_map):
    """Detect + execute, kept as the backward-compatible single-call entry (read_methods
    and the tests call this). The controller will eventually call detect()/execute() directly."""
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)
