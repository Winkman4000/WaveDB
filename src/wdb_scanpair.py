"""
wdb_scanpair -- NON-KEY filter regime for a 2-key COUNT(*) top-K with a WHERE on a separate column.

The companion to wdb_heavypair. When the filter column C is one of the two group keys, the sorted
pair sidecar answers it by walk-and-skip (heavypair). When C is a SEPARATE column, the sidecar
summed C away -- its cells can't be sliced by C -- so we SCAN: jump to the rows where C = v
(a code-space equality scan: one O(N) boolean pass, flat ~25ms on 100M regardless of how many rows
match), gather the two group-key codes there, fuse + bincount + argpartition for top-K, decode only
the K winners. The scan IS the answer; nothing persisted.

This handles C at ANY cardinality. We deliberately do NOT build a per-C sidecar: storing each C
value's per-(A,B) contribution is the same data volume as a 3-key (A,B,C) sidecar -- it explodes --
so there is no cardinality threshold here. High-card C -> tiny matched set, scan dominates (~25ms);
low-card C -> larger matched set, the gather+count grows but still beats a full group-by. One scan,
every C, no extra segments to manage.

Same contract as the other reads: detect(seg, tree, col_map) -> spec | None;
execute(seg, spec) -> (rows, colnames) | None. Fail-closed outside its exact shape.

Scope (v1): single segment, exactly two value-identity group keys, projections {k1, k2, COUNT(*)},
ORDER BY COUNT(*) DESC + LIMIT, no HAVING/DISTINCT/JOIN/OFFSET, WHERE = a single `col = literal`
(equality only) on a dict-coded column that is NOT a group key.
"""
import numpy as np
import wdb_sql
import wdb_kernel
import wdb_policies as P
import workers
E = wdb_sql.E

_HITS = 0


def _eq_on_nonkey(seg, where_node, gnames, col_map):
    """Parse WHERE as a single `col = literal` on a dict-coded NON-group-key column whose
    cardinality is high. Returns (phys_col, codes) -- the axis codes equal to the literal -- or None
    to decline. codes empty -> literal absent (result is empty)."""
    n = where_node.this if isinstance(where_node, E.Where) else where_node
    if not isinstance(n, E.EQ):                     # equality only (the high-card-rare case)
        return None
    col = wdb_sql._colname(n.this); lit = n.expression
    if col is None or not isinstance(lit, E.Literal):
        return None
    if col in gnames:                              # filter on a group key -> heavypair's job, not ours
        return None
    phys = col_map.get(col, col) if col_map else col
    if phys not in seg.cols:
        return None
    c = seg.cols[phys]
    if c['mode'] == 4 or seg._overrides(phys) is not None:
        return None
    # No cardinality gate: the scan handles a non-key equality filter at ANY cardinality of C. For
    # high-card C the matched set is tiny (flat ~25ms scan dominates); for low-card C the matched set
    # is larger so the gather+count grows, but it still beats a full group-by -- and crucially it
    # needs no persisted per-C structure (a C-keyed sidecar storing per-(A,B) adjustments is the same
    # data volume as a 3-key sidecar -> it explodes, so we don't build one). One scan, every C.
    try:
        codes = seg.codes(phys)
    except Exception:
        return None
    if codes is None or codes.dtype.kind not in 'iu':
        return None
    td = np.asarray(seg._typed_dict(phys))
    kind = 'i' if c['dt'] == 0 else ('f' if c['dt'] == 2 else ('i' if c['dt'] == 3 else 'S'))
    v = wdb_sql._lit_for_col(seg, phys, lit, kind)
    if c['dt'] not in (0, 2, 3) and isinstance(v, int):
        v = str(v).encode()
    try:
        mcodes = np.nonzero(td == v)[0].astype(np.int64)
    except Exception:
        return None
    if c['has_null']:
        mcodes = mcodes[mcodes != (c['V'] - 1)]
    return (phys, mcodes)


def detect(seg, tree, col_map):
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    if not P.has_limit(tree):          return None
    if not P.no_deleted_rows(seg):     return None
    if wdb_sql._offset(tree):          return None
    group = tree.args.get('group')
    if group is None or len(group.expressions) != 2:
        return None
    proj = tree.expressions
    if len(proj) != 3:
        return None
    ci = None
    for i, p in enumerate(proj):
        ak = wdb_sql._agg_kind(p)
        if ak is not None and ak[0] == 'COUNT_STAR':
            if ci is not None: return None
            ci = i
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
    if not _order_is_count_desc(tree, proj, ci):
        return None
    if not P.has_where(tree):
        return None
    f = _eq_on_nonkey(seg, tree.args.get('where'), gnames, col_map)
    if f is None:
        return None
    cols = [col_map.get(k, k) if col_map else k for k in knames]
    for col in cols:
        if col not in seg.cols: return None
        if seg.cols[col]['mode'] == 4: return None     # need value-identity codes for the keys
    return {'cols': cols, 'ci': ci, 'lim': wdb_sql._limit(tree), 'proj': proj,
            'knames': knames, 'filter': f, 'order': tree.args.get('order')}


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


def execute(seg, spec):
    global _HITS
    cols = spec['cols']; ci = spec['ci']; lim = spec['lim']; proj = spec['proj']
    knames = spec['knames']; fcol, fcodes = spec['filter']
    # 1) code-space scan: rows where C = v  (high-card v -> rare -> small mask)
    cC = seg.codes(fcol)
    if fcodes.size == 0:
        return [], [wdb_sql._alias(p) for p in proj]      # literal absent -> empty result
    a, b = cols
    Va = int(seg.cols[a]['V']); Vb = int(seg.cols[b]['V'])
    prod = Va * Vb
    # FAST PATH (single C code, fused key fits uint32): the AVX-512 kernel scans C alone for
    # survivor positions, gathers A/B only there (native width, no upcast), and HASH-tallies the
    # pairs (counts, never sorts; memory = pairs that occur, not the a*b key space). Returns the
    # decoded-code top-K, or None (kernel unavailable / out of shape) to fall back to numpy below.
    winners = None
    if fcodes.size == 1 and prod < (1 << 32):
        kr = wdb_kernel.scanpair_topk(seg.codes(fcol), seg.codes(a), seg.codes(b),
                                      int(fcodes[0]), Vb, lim)
        if kr is not None:
            ka, kb, kc = kr
            if ka.size == 0:
                return [], [wdb_sql._alias(p) for p in proj]
            winners = [(int(ka[i]), int(kb[i]), int(kc[i])) for i in range(ka.size)]

    if winners is None:
        # numpy fallback: scan -> gather both columns -> narrow-fuse -> unique-sort -> top-K
        mask = (cC == fcodes[0]) if fcodes.size == 1 else np.isin(cC, fcodes)
        pos = np.nonzero(mask)[0]
        if pos.size == 0:
            return [], [wdb_sql._alias(p) for p in proj]
        if prod < (1 << 32):
            key = (seg.codes(a)[pos].astype(np.uint32) * np.uint32(Vb) + seg.codes(b)[pos].astype(np.uint32))
        else:
            key = seg.codes(a)[pos].astype(np.int64) * Vb + seg.codes(b)[pos].astype(np.int64)
        uk, cnt = np.unique(key, return_counts=True)
        k = lim
        if k < cnt.size:
            part = np.argpartition(cnt, -k)[-k:]
            order = part[np.argsort(cnt[part])[::-1]]
        else:
            order = np.argsort(cnt)[::-1]
        winners = [(int(uk[idx]) // Vb, int(uk[idx]) % Vb, int(cnt[idx])) for idx in order[:k]]

    # decode ONLY the K winner codes via random-access fetch (never the whole dictionary)
    rows = []
    for acode, bcode, ccount in winners:
        codeof = {a: acode, b: bcode}
        row = [None] * len(proj)
        row[ci] = ccount
        for pi, pp in enumerate(proj):
            if pi == ci: continue
            knm = wdb_sql._proj_colname(pp)
            col = cols[knames.index(knm)]
            row[pi] = wdb_sql._pyval(seg.fetch(col, codeof[col]))
        rows.append(tuple(row))
    rows = workers.finalize(rows, proj, spec['order'], lim)
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in proj]


def try_scanpair(seg, tree, col_map):
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)
