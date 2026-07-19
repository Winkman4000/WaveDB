"""
wdb_heavypair — pre-aggregated count projection for high-cardinality TWO-key GROUP BY COUNT(*).

The 2-key generalization of wdb_gbcount. A filter-free
  SELECT k1, k2, COUNT(*) FROM t GROUP BY k1, k2 ORDER BY COUNT(*) DESC LIMIT N
never needs a scan: the per-pair counts are fixed between writes. We persist the occurring pairs
once (sorted by count descending, heavy hitters count>=2 only — singleton pairs are an implicit
count of 1), so the query becomes a top-N slice instead of a full-table scan + giant composite-key
accumulator. On 100M hits the sorted pair node answered the 2-key top-K in microseconds vs an ~12s
scan.

LANDMARKS: alongside the count-sorted block we store ratio-spaced landmarks — (position, count) at
each point where the count has halved from the previous landmark. A handful of integers (~log2 of
the top count). They turn "count >= T" / "rank of T" into a jump-to-segment + short scan instead of
a walk from the front, using only the landmark counts (no data touched to locate the segment). For
the top-K query itself the answer is already the front of the block, so landmarks are free insurance
for the threshold/rank shapes.

Same contract as wdb_gbcount / wdb_cube: detect(seg, tree, col_map) -> spec | None;
execute(seg, spec) -> (rows, colnames) | None; try_heavypair = detect+execute. Fail-closed outside
its exact shape. Sidecar persisted next to the segment as <seg>.<a>__<b>.gbp (columns name-sorted, so
GROUP BY k1,k2 and k2,k1 share one node). Staleness-guarded by seg.N. Format version guarded so an
older sidecar without landmarks is rebuilt rather than mis-read.

Scope (v1): single segment, exactly two value-identity (non-mode-4) keys, projections exactly
{bare k1, bare k2, COUNT(*)}, no WHERE/HAVING/DISTINCT/JOIN (a single eq/neq on a pair axis is
allowed), LIMIT N, no deleted rows. Two ORDER forms: ORDER BY COUNT(*) DESC (classic top-N), or NO
ORDER BY at all (unordered "any N" -- the count-descending heavy front is a legal, cheapest answer;
this is the Q17 shape). If the heavy block holds < N pairs, decline to the scan (singleton top-up
is a future v2). The loaded block is heap-resident in _CACHE across queries for the process lifetime.
"""
import os, pickle, numpy as np
import wdb_qmem
import wdb_sql
import workers
import wdb_policies as P
E = wdb_sql.E

_FMT = 2        # sidecar format version (1 = no landmarks; 2 = with landmarks)

_VCACHE = wdb_qmem.register({})    # (seg.path, col) -> by-code value dict (built once; high-card dicts are big)

def _vals(seg, col):
    """By-code value dictionary (sorted, indexable by code), or None for non-value-identity
    encodings (mode-4 affine) we can't decode by rank. Cached per (segment, column): the dict is
    deterministic and, for high-card keys, large enough that rebuilding it per query dominates."""
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

_HITS = 0
_CACHE = wdb_qmem.register({})     # (seg.path, (a,b), N) -> (codesA, codesB, counts, landmarks); heap-resident across queries

# Per-pair on/off. A disabled (name-sorted) pair is blocked from getting a sidecar: detect declines,
# so the query falls through to the scan and no RAM is spent on that pair. The operator's RAM-budget
# control -- enable only the pairs worth keeping resident. Empty set = all eligible pairs allowed.
_DISABLED = set()


def disable_pair(a, b):
    _DISABLED.add(tuple(sorted((a, b))))


def enable_pair(a, b):
    _DISABLED.discard(tuple(sorted((a, b))))


def is_disabled(cols):
    return tuple(sorted(cols)) in _DISABLED


def _path(seg, cols):
    a, b = sorted(cols)
    return f"{seg.path}.{a}__{b}.gbp"


def _landmarks(cn):
    """Ratio-spaced landmarks over the count-descending block: (position, count) each time the count
    has at least halved since the last landmark. ~log2(top_count) entries. Placed where a steep
    count curve actually bends, so threshold/rank jumps land in the right segment immediately."""
    if cn.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    lm = []
    nxt = int(cn[0])
    for i in range(cn.size):
        c = int(cn[i])
        if c <= nxt:
            lm.append((i, c))
            nxt = c // 2
            if nxt < 1:
                break
    return np.asarray(lm, dtype=np.int64)


def _build(seg, cols):
    """Occurring pairs with count>=2, sorted by count descending, plus ratio-spaced landmarks.
    Returns (a, b, codesA uint32, codesB uint32, counts int64, landmarks int64[k,2], n) or None."""
    a, b = sorted(cols)
    ca = seg._raw_codes(a); cb = seg._raw_codes(b)
    if ca.size == 0:
        return None
    Vb = int(cb.max()) + 1
    key = ca.astype(np.int64) * Vb + cb.astype(np.int64)
    uk, cnt = np.unique(key, return_counts=True)
    order = np.argsort(cnt, kind='stable')[::-1]         # count descending
    keep = cnt[order] >= 2                               # singleton pairs implicit (count 1)
    uk = uk[order][keep]; cn = cnt[order][keep]
    codesA = np.ascontiguousarray(uk // Vb, dtype=np.uint32)
    codesB = np.ascontiguousarray(uk % Vb, dtype=np.uint32)
    cn = np.ascontiguousarray(cn, dtype=np.int64)
    return a, b, codesA, codesB, cn, _landmarks(cn), int(seg.N)


def _load(seg, cols):
    a, b = sorted(cols)
    ck = (seg.path, (a, b), int(seg.N))
    hit = _CACHE.get(ck)
    if hit is not None:
        return hit
    p = _path(seg, cols)
    if os.path.exists(p):
        try:
            blob = pickle.load(open(p, 'rb'))
            if blob.get('fmt') == _FMT and blob['n'] == int(seg.N) and (blob['a'], blob['b']) == (a, b):
                _CACHE[ck] = (blob['cA'], blob['cB'], blob['cn'], blob['lm'])
                return _CACHE[ck]
        except Exception:
            pass
    built = _build(seg, cols)
    if built is None:
        return None
    aa, bb, cA, cB, cn, lm, n = built
    try:
        pickle.dump({'fmt': _FMT, 'a': aa, 'b': bb, 'cA': cA, 'cB': cB, 'cn': cn, 'lm': lm, 'n': n},
                    open(p, 'wb'), protocol=4)
    except Exception:
        pass
    _CACHE[ck] = (cA, cB, cn, lm)
    return _CACHE[ck]


def threshold_floor(lm, cn, T):
    """Smallest block position to start scanning for "count >= T", using ONLY the landmark counts
    (no pair data touched). Returns a position p such that all rows < p are guaranteed >= T's segment
    head; the caller scans forward from p. O(#landmarks)."""
    start = 0
    for pos, c in lm:
        if c >= T:
            start = int(pos)
        else:
            break
    return start


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


def _axis_filter(seg, where_node, cols, col_map):
    """Parse a WHERE that is a single equality/inequality on ONE of the two pair axes into a walk
    filter: (col, neg, codeset). The pre-sorted block already separates pairs by both axes, so such
    a filter is answered by walking the count-descending block and skipping rows whose axis code is
    (for '=') not in / (for '<>') in the codeset -- no recount, no scan. Self-referential: it only
    works because the filtered column is one of the sidecar's KEPT axes.

    Returns (col, neg, codes ndarray) or None to decline (filter not on a pair axis, or not a simple
    eq/neq against a literal). codes is the set of axis codes equal to the literal; empty means the
    literal is absent (eq -> matches nothing, neq -> matches everything)."""
    n = where_node.this if isinstance(where_node, E.Where) else where_node
    if not isinstance(n, (E.EQ, E.NEQ)):
        return None
    col = wdb_sql._colname(n.this)
    lit = n.expression
    if col is None or not isinstance(lit, E.Literal):
        return None
    phys = col_map.get(col, col) if col_map else col
    if phys not in cols:                     # filter must be on one of the two pair axes
        return None
    vals = _vals(seg, phys)
    if vals is None:
        return None
    c = seg.cols[phys]
    kind = 'i' if c['dt'] == 0 else ('f' if c['dt'] == 2 else ('i' if c['dt'] == 3 else 'S'))
    v = wdb_sql._lit_for_col(seg, phys, lit, kind)
    if c['dt'] not in (0, 2, 3) and isinstance(v, int):
        v = str(v).encode()
    try:
        codes = np.nonzero(np.asarray(vals) == v)[0].astype(np.int64)
    except Exception:
        return None
    return (phys, isinstance(n, E.NEQ), codes)


def detect(seg, tree, col_map):
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    if not P.has_limit(tree):          return None
    group = tree.args.get('group')
    if group is None or len(group.expressions) != 2:     # exactly two group keys
        return None
    proj = tree.expressions
    if len(proj) != 3:                                   # k1, k2, COUNT(*)
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
    if set(knames) != set(gnames):                       # the two bare keys are exactly the group keys
        return None
    cols = [col_map.get(k, k) if col_map else k for k in knames]
    if is_disabled(cols):                  # operator turned this pair off -> no sidecar, fall to scan
        return None
    for col in cols:
        if not P.columns_exist(seg, col):  return None
        if not P.not_positional(seg, col): return None
    if not P.no_deleted_rows(seg):         return None
    # ORDER shape: two legal forms.
    #   (a) ORDER BY COUNT(*) DESC + LIMIT N  -> classic top-N by count.
    #   (b) LIMIT N with NO ORDER BY          -> "any N pairs" (result order unspecified). The
    #       count-descending heavy front is a LEGAL answer to an unordered limit, and it is the
    #       cheapest N to produce (front of the already-sorted block). This is the Q17 shape.
    # Anything else (ordered by a key, ASC, etc.) the count-sorted block can't serve -> decline.
    order = tree.args.get('order')
    if order is None or not order.expressions:
        unordered = True
    elif _order_is_count_desc(tree, proj, ci):
        unordered = False
    else:
        return None
    # WHERE: allowed only if it is a single eq/neq on one of the two pair axes (a kept-axis filter,
    # answered by a walk-and-skip over the sorted block). Any other WHERE -> decline to the scan.
    afilter = None
    if P.has_where(tree):
        afilter = _axis_filter(seg, tree.args.get('where'), cols, col_map)
        if afilter is None:
            return None
    Vs = {col: _vals(seg, col) for col in cols}
    if any(v is None for v in Vs.values()):              # both keys must be value-identity decodable
        return None
    return {'cols': cols, 'ci': ci, 'lim': wdb_sql._limit(tree), 'proj': proj,
            'knames': knames, 'V': Vs, 'order': tree.args.get('order'), 'afilter': afilter,
            'unordered': unordered}


def execute(seg, spec):
    global _HITS
    cols = spec['cols']; ci = spec['ci']; lim = spec['lim']; proj = spec['proj']
    knames = spec['knames']; V = spec['V']
    loaded = _load(seg, cols)
    if loaded is None:
        return None
    cA, cB, cn, lm = loaded
    a, b = sorted(cols)
    by_col = {a: cA, b: cB}                               # code array per (name-sorted) column

    # --- kept-axis filter: keep only block rows whose filtered-axis code matches the predicate.
    # The block is already count-descending, so the filtered top-K is the first `lim` kept rows.
    # Pure selection over stored cells -- no recount, no scan (works because the filter is on a
    # KEPT axis). positions[] indexes the block in count order. ---
    afilter = spec.get('afilter')
    if afilter is not None:
        fcol, neg, fcodes = afilter
        axis = by_col[fcol]
        inset = np.isin(axis, fcodes) if fcodes.size else np.zeros(axis.size, dtype=bool)
        keep = ~inset if neg else inset
        positions = np.nonzero(keep)[0]                  # block indices passing the filter, count order
    else:
        positions = np.arange(cn.size)

    if lim > positions.size:                             # not enough heavy pairs -> singletons needed -> scan
        return None
    if not spec.get('unordered') and lim < positions.size:   # tie straddling LIMIT (matters only when ORDERED:
        if int(cn[positions[lim - 1]]) == int(cn[positions[lim]]):  # which of the tied pairs are "top N"?)
            return None                                  # ambiguous boundary -> defer to scan
    # unordered LIMIT: any N pairs is a legal answer, so a tie across the boundary is fine -- no decline.

    rows = []
    for r in positions[:lim]:
        r = int(r)
        row = [None] * len(proj)
        row[ci] = int(cn[r])
        for pi, p in enumerate(proj):
            if pi == ci:
                continue
            knm = wdb_sql._proj_colname(p)
            col = cols[knames.index(knm)]
            code = int(by_col[col][r])
            row[pi] = wdb_sql._pyval(V[col][code])
        rows.append(tuple(row))
    rows = workers.finalize(rows, proj, spec['order'], lim)
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in proj]


def try_heavypair(seg, tree, col_map):
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)
