"""
wdb_clustertopk -- cluster-ordered projection top-K.

Answers   SELECT cols FROM t [WHERE c <> ''] ORDER BY K [ASC|DESC][, sec...] LIMIT k
when the segment is CLUSTERED by K (the primary ORDER BY key). Clustering physically
sorts rows by K over [0, nn) (see wdb_encode._cluster_order), so the k smallest- (ASC)
or largest- (DESC) by-K surviving rows are a short prefix/suffix of the row space. We
walk from the clustered end, apply the WHERE mask per row-range via _raw_codes_range,
collect survivors until LIMIT (draining the boundary tie-group when a secondary key is
present), then read ONLY those rows' projected values via the chunk-aware dict fetch
(NOT values_range, which would materialise the whole string dictionary). Touches ~k
rows, not N.

The gate is pure system state: detect derives the required cluster key from the query
shape (the primary ORDER BY column) and matches it against seg.cluster_meta()['key'],
which is resident (the .cluster sidecar is loaded + cached at segment open). Match ->
this path; no match -> None and the caller scans. Nothing is probed at query time.

detect (fail-closed): clustered segment; no deleted rows; single table; no GROUP BY /
HAVING / DISTINCT / JOIN; LIMIT present; projection is 1+ bare non-agg dict columns
(modes 0-3); the PRIMARY ORDER BY key is a bare Column == the cluster key; any
secondary ORDER BY keys are themselves projected columns; WHERE is absent or exactly
`C <> ''` on a string-dict column (dt==1, mode 0/1). Anything else -> None.
"""
import numpy as np
import wdb_sql
import wdb_policies as P
E = wdb_sql.E

_HITS = 0


def _order_keys(tree):
    """[(colname, desc_bool), ...] for a pure-column ORDER BY, else None."""
    order = tree.args.get('order')
    if order is None or not order.expressions:
        return None
    keys = []
    for o in order.expressions:
        if not isinstance(o, E.Ordered):
            return None
        tgt = o.this
        if not isinstance(tgt, E.Column):
            return None
        keys.append((tgt.name, bool(o.args.get('desc'))))
    return keys


def _where_neq_empty_col(tree):
    """col name for a `col <> ''` WHERE; '' when there is no WHERE; None for anything else."""
    w = tree.args.get('where')
    if w is None:
        return ''
    pred = w.this
    if not isinstance(pred, E.NEQ):
        return None
    lhs, rhs = pred.this, pred.expression
    if not (isinstance(lhs, E.Column) and isinstance(rhs, E.Literal)
            and rhs.is_string and rhs.this == ''):
        return None
    return lhs.name


def detect(seg, tree, col_map):
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    if not P.has_limit(tree):          return None
    if tree.args.get('group') is not None: return None
    if not P.no_deleted_rows(seg):     return None
    cm = seg.cluster_meta()
    if cm is None:                     return None     # <<< not clustered: not our shape

    cmap = col_map or {}
    resolve = lambda n: cmap.get(n, n)

    proj = tree.expressions
    if not proj:                       return None
    pcols = []
    for p in proj:
        if wdb_sql._agg_kind(p) is not None: return None
        nm = wdb_sql._proj_colname(p)
        if nm is None:                 return None
        pcols.append(nm)

    keys = _order_keys(tree)
    if not keys:                       return None
    prim_name, prim_desc = keys[0]
    if resolve(prim_name) != cm['key']: return None    # <<< THE STATE GATE
    for (kn, _kd) in keys[1:]:
        if kn not in pcols:            return None      # secondary must be projected

    wname = _where_neq_empty_col(tree)
    if wname is None:                  return None

    rcols = [resolve(c) for c in pcols]
    if not all(P.columns_exist(seg, c) for c in rcols): return None
    for c in rcols:
        if seg.cols[c]['mode'] not in (0, 1, 2, 3):     return None   # fetch-able dict cols
    wcol = resolve(wname) if wname else None
    if wcol is not None:
        if not P.columns_exist(seg, wcol):              return None
        wc = seg.cols[wcol]
        if wc['dt'] != 1 or wc['mode'] not in (0, 1):   return None   # `<>''` -> code 0
    kcol = resolve(prim_name)
    if not P.columns_exist(seg, kcol):                  return None

    return {'pcols': rcols, 'proj': proj, 'kcol': kcol, 'kdesc': prim_desc,
            'sec': [(resolve(kn), kd) for (kn, kd) in keys[1:]],
            'wcol': wcol, 'nn': int(cm['nn']), 'lim': wdb_sql._limit(tree)}


def _gather(seg, col, rows, maxrow):
    """Values for a small set of rows (all <= maxrow) via prefix codes + chunk-aware fetch."""
    rc = seg._raw_codes_range(col, 0, maxrow + 1)
    return [seg.fetch(col, int(rc[r])) for r in rows]


def execute(seg, spec):
    global _HITS
    nn = spec['nn']; lim = spec['lim']; desc = spec['kdesc']
    kcol = spec['kcol']; wcol = spec['wcol']; sec = spec['sec']
    aliases = [wdb_sql._alias(p) for p in spec['proj']]
    if nn <= 0 or lim <= 0:
        return [], aliases

    off = np.asarray(seg.cluster_meta()['offsets'])   # first-row of each distinct key value
    BATCH = 8192
    cand = []      # global row indices, in cluster order
    cand_k = []    # cluster-key RANK (slice id) per cand -- equal key VALUES share a rank,
                   # so a secondary ORDER BY key can break ties correctly (encoding-independent)
    boundary = None

    def survivors(lo, hi):
        if wcol is not None:
            m = seg._raw_codes_range(wcol, lo, hi) != 0    # '' == code 0
            return np.nonzero(m)[0] + lo
        return np.arange(lo, hi)

    if not desc:
        i = 0
        while i < nn:
            j = min(i + BATCH, nn)
            rows = survivors(i, j)
            ranks = np.searchsorted(off, rows, 'right') - 1
            for r, rk in zip(rows.tolist(), ranks.tolist()):
                cand.append(r); cand_k.append(rk)
                if boundary is None and len(cand) >= lim:
                    boundary = cand_k[lim - 1]
            i = j
            if boundary is not None and cand_k and cand_k[-1] > boundary:
                break
    else:
        i = nn
        while i > 0:
            j = max(i - BATCH, 0)
            rows = survivors(j, i)
            ranks = np.searchsorted(off, rows, 'right') - 1
            for r, rk in zip(rows.tolist()[::-1], ranks.tolist()[::-1]):
                cand.append(r); cand_k.append(rk)
                if boundary is None and len(cand) >= lim:
                    boundary = cand_k[lim - 1]
            i = j
            if boundary is not None and cand_k and cand_k[-1] < boundary:
                break

    if not cand:
        return [], aliases
    maxrow = max(cand)

    # composite stable sort over the tiny candidate set (least-significant key first).
    # _gather returns values aligned with cand positions; reverse per key honours direction.
    order = list(range(len(cand)))
    for (scol, sdesc) in reversed(sec):
        vals = _gather(seg, scol, cand, maxrow)
        order.sort(key=lambda p, v=vals: (v[p] is None, v[p]), reverse=sdesc)
    order.sort(key=lambda p: cand_k[p], reverse=desc)      # primary, most-significant
    order = order[:lim]

    final_rows = [cand[p] for p in order]
    gathered = {c: _gather(seg, c, final_rows, maxrow) for c in spec['pcols']}
    out = [tuple(wdb_sql._pyval(gathered[c][fi]) for c in spec['pcols'])
           for fi in range(len(final_rows))]
    _HITS += 1
    return out, aliases
