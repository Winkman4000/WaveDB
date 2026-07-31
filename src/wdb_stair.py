"""stair: single-key GROUP BY reads served from a STAIRCASE column's step positions -- no code
decode, no dict materialization, no scan. A staircase column (codes non-decreasing in row order,
i.e. data ingested sorted by it: EventTime, EventDate) is fully described by the rows where its
code ticks +1: per-value COUNT(*) is literally np.diff over the step array, so the whole group-by
is O(V) arithmetic on a structure that is ~KB (code_enc=2) instead of a 100M-row scan. Values
decode winner-only via seg.fetch (never the full dict). The exception law as a read: the norm is
'same as the row above', the steps are the exceptions, and the exceptions ARE the answer.

v1 scope: GROUP BY col with projections {bare col, COUNT(*)}, no WHERE/JOIN/HAVING/DISTINCT/
OFFSET, no overrides/deleted rows; ORDER BY COUNT(*) DESC (canonical count-desc, code-asc
tiebreak), ORDER BY col ASC, or unordered; LIMIT optional when the value space is small."""
import numpy as np
import sqlglot.expressions as E
import wdb_sql
import wdb_pairagg
import workers
import wdb_policies as P
E = wdb_sql.E

_ENABLED = True
_HITS = 0
_VMAX_UNBOUNDED = 65536      # LIMIT-less group-bys only when the whole answer is this small


def enable():
    global _ENABLED; _ENABLED = True


def disable():
    global _ENABLED; _ENABLED = False


def _count_index(proj):
    ci = None
    for i, p in enumerate(proj):
        ak = wdb_sql._agg_kind(p)
        if ak is not None:
            if ak[0] != 'COUNT_STAR' or ci is not None:
                return None
            ci = i
    return ci


def _order_kind(tree, proj, ci, col, col_map):
    """'none' | 'count_desc' | 'key_asc' | None (decline). count_desc accepts one optional
    secondary key ascending == this read's canonical tiebreak (code asc = value asc)."""
    order = tree.args.get('order')
    if order is None or not order.expressions:
        return 'none'
    o0 = order.expressions[0]
    nm = wdb_sql._colname(o0.this) if isinstance(o0.this, E.Column) else None
    if o0.args.get('desc'):
        target = wdb_sql._alias(proj[ci])
        if (nm is not None and (nm == target or nm == 'count')
                or wdb_pairagg._order_targets_count(o0, proj, ci)):
            extra = order.expressions[1:]
            if not extra:
                return 'count_desc'
            if len(extra) == 1 and not extra[0].args.get('desc'):
                en = wdb_sql._colname(extra[0].this)
                if en is not None and (col_map.get(en, en) if col_map else en) == col:
                    return 'count_desc'
            return None
        return None
    if (nm is not None and len(order.expressions) == 1
            and (col_map.get(nm, nm) if col_map else nm) == col):
        return 'key_asc'
    return None


def detect(seg, tree, col_map):
    if not _ENABLED:                   return None
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_where(tree):           return None
    if not P.no_having(tree):          return None
    if not P.single_group_key(tree):   return None
    if wdb_sql._offset(tree):          return None
    proj = tree.expressions
    if len(proj) != 2:
        return None
    ci = _count_index(proj)
    if ci is None:
        return None
    kp = proj[1 - ci]
    if wdb_sql._agg_kind(kp) is not None:
        return None
    # DERIVED-UNIT GROUPS ON A STAIRCASE (the stair-walk): extract(hour|minute) of a
    # stair column groups in V-space -- step k's value is the dict's k-th entry, so the
    # whole query is a unit map over dict values + one weighted bincount of step counts.
    # No row is ever read.
    inner = kp.this if isinstance(kp, E.Alias) else kp
    g0 = tree.args.get('group').expressions[0]
    if isinstance(inner, E.Extract) and isinstance(inner.expression, E.Column):
        unit = (inner.this.name if hasattr(inner.this, 'name') else str(inner.this)).lower()
        alias_nm = kp.alias if isinstance(kp, E.Alias) else None
        g_matches = (isinstance(g0, E.Extract)
                     or (isinstance(g0, E.Column) and alias_nm and g0.name == alias_nm))
        if unit in ('hour', 'minute') and g_matches:
            dcol = col_map.get(inner.expression.name, inner.expression.name) if col_map \
                else inner.expression.name
            if (P.columns_exist(seg, dcol) and P.not_positional(seg, dcol)
                    and P.no_deleted_rows(seg) and seg._effective(dcol) is None
                    and seg.cols[dcol].get('dt') in (0, 3)
                    and not seg.cols[dcol].get('has_null')):
                order = tree.args.get('order')
                ord_ok = order is None
                if order is not None and len(order.expressions) == 1:
                    onm = wdb_sql._colname(order.expressions[0].this)
                    ord_ok = bool(order.expressions[0].args.get('desc')) \
                        and onm == wdb_sql._alias(proj[ci])
                if ord_ok:
                    return {'col': dcol, 'ci': ci, 'unit': unit, 'proj': proj,
                            'lim': wdb_sql._limit(tree), 'order': order}
    knm = wdb_sql._proj_colname(kp)
    group = tree.args.get('group')
    gnm = wdb_sql._colname(group.expressions[0])
    if knm is None or gnm is None or knm != gnm:
        return None
    col = col_map.get(knm, knm) if col_map else knm
    if not P.columns_exist(seg, col):  return None
    if not P.not_positional(seg, col): return None
    if not P.no_deleted_rows(seg):     return None
    if seg._effective(col) is not None:            # overrides falsify the stored steps
        return None
    kind = _order_kind(tree, proj, ci, col, col_map)
    if kind is None:
        return None
    lim = wdb_sql._limit(tree)
    if lim is None and int(seg.cols[col]['V']) > _VMAX_UNBOUNDED:
        return None
    return {'col': col, 'ci': ci, 'lim': lim, 'proj': proj, 'kind': kind,
            'order': tree.args.get('order')}


def execute(seg, spec):
    global _HITS
    col = spec['col']
    if spec.get('unit'):
        return _unit_execute(seg, spec)
    steps = seg.stairs(col)
    if steps is None:
        return None
    counts = np.diff(np.concatenate(([0], steps, [seg.N]))).astype(np.int64)
    nc = counts.size
    lim = spec['lim'] if spec['lim'] is not None else nc
    k = min(lim, nc)
    if spec['kind'] == 'count_desc':
        if k < nc:
            part = np.argpartition(counts, nc - k)[nc - k:]
        else:
            part = np.arange(nc)
        o = part[np.lexsort((part, -counts[part]))][:k]    # canonical: count DESC, code ASC
    else:                                                  # 'none' / 'key_asc': code order = value order
        o = np.arange(k)
    ci = spec['ci']
    rows = []
    for i in range(o.size):
        v = wdb_sql._pyval(seg.fetch(col, int(o[i])))      # winner-only decode, O(1) per value
        row = [None, None]
        row[ci] = int(counts[o[i]])
        row[1 - ci] = v
        rows.append(tuple(row))
    rows = workers.finalize(rows, spec['proj'], spec['order'], lim)
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in spec['proj']]


def _unit_execute(seg, spec):
    """The stair-walk: per-code counts from step offsets, unit map over dict values,
    weighted bincount. Emits unit buckets ordered by count."""
    global _HITS
    import wdb_window as WN
    col = spec['col']
    steps = seg.stairs(col)
    if steps is None:
        return None
    counts = np.diff(np.concatenate(([0], steps, [seg.N]))).astype(np.int64)
    tv = np.asarray(WN._int_table(seg, col), dtype=np.int64)
    if tv.size != counts.size:
        return None
    if spec['unit'] == 'hour':
        key = (tv % 86400) // 3600
        span = 24
    else:
        key = (tv % 3600) // 60
        span = 60
    agg = np.bincount(key, weights=counts, minlength=span).astype(np.int64)
    present = np.flatnonzero(agg)
    sel = present[np.argsort(-agg[present], kind='stable')] if spec.get('order') is not None \
        else present
    lim = spec['lim']
    if lim is not None:
        if spec.get('order') is not None and lim < sel.size \
                and int(agg[sel[lim - 1]]) == int(agg[sel[lim]]):
            return None                          # tie straddles LIMIT: defer
        sel = sel[:lim]
    ci = spec['ci']; ki = 1 - ci
    rows = []
    for u in sel.tolist():
        row = [None, None]
        row[ki] = int(u)
        row[ci] = int(agg[u])
        rows.append(tuple(row))
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in spec['proj']]
