"""
wdb_sumtopk -- single high-card key, SUM top-K: the counting board's SUM sibling.

GROUP BY <big dict key> ORDER BY SUM(<numeric dict col>) DESC LIMIT k paid ~0.9s of
generic gid machinery (factorize-or-dense, casts, radix, prefilter) above a two-stream
floor. Here: both code streams read once, one fused kernel accumulates value-table
gathers into per-thread boards (no casts, no factorize), argpartition finds the k
winners in O(V), and only their labels decode. Small dictionaries stay with fused_agg,
which already wins there.
"""
import numpy as np
import wdb_sql
import wdb_policies as P
import wdb_kernels as WK
import sqlglot.expressions as E

_HITS = 0
_MIN_V = 1_000_000


def detect(seg, tree, col_map):
    if not P.no_joins(tree) or not P.no_having(tree) or not P.no_select_distinct(tree):
        return None
    if tree.args.get('where') is not None or tree.args.get('qualify') is not None:
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 1 or not isinstance(g.expressions[0], E.Column):
        return None
    lim = wdb_sql._limit(tree)
    if lim is None or lim > 10000:
        return None
    proj = tree.expressions
    if len(proj) != 2:
        return None
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    key = None; sumcol = None; ki = None; si = None
    for pi, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        nm = wdb_sql._colname(inner)
        kd = wdb_sql._agg_kind(p)
        if nm is not None and kd is None:
            key, ki = sc(nm), pi
        elif kd is not None and kd[0] == 'SUM':
            ac = wdb_sql._colname(inner.this)
            if ac is None:
                return None
            sumcol, si = sc(ac), pi
        else:
            return None
    if key is None or sumcol is None:
        return None
    gnm = sc(g.expressions[0].name)
    if gnm != key:
        return None
    order = tree.args.get('order')
    if order is None or len(order.expressions) != 1:
        return None
    o0 = order.expressions[0]
    if not bool(o0.args.get('desc')):
        return None
    onm = wdb_sql._colname(o0.this)
    if onm != wdb_sql._alias(proj[si]) and onm != sumcol:
        return None
    ck, cv = seg.cols.get(key), seg.cols.get(sumcol)
    if ck is None or cv is None:
        return None
    if ck.get('mode') not in (0, 1, 2) or ck.get('has_null'):
        return None
    if cv.get('mode') not in (0, 1, 2) or cv.get('has_null') or cv.get('dt') != 0:
        return None
    if int(ck.get('V') or 0) < _MIN_V:
        return None                                     # small boards: fused_agg already wins
    if not P.no_deleted_rows(seg):
        return None
    return {'key': key, 'val': sumcol, 'lim': lim, 'ki': ki, 'si': si,
            'proj': proj, 'off': wdb_sql._offset(tree) or 0}


def execute(seg, spec):
    global _HITS
    import wdb_window as W
    key, val, lim, off = spec['key'], spec['val'], spec['lim'], spec['off']
    kc = np.asarray(seg._raw_codes(key))
    vc = np.asarray(seg._raw_codes(val))
    vt = np.asarray(W._int_table(seg, val), dtype=np.int64)
    K = int(seg.cols[key]['V'])
    sums = WK.grouped_sum_codes(kc, vc, vt, K)
    need = lim + off
    if need >= K:
        top = np.argsort(sums)[::-1][:need]
    else:
        part = np.argpartition(sums, K - need)[K - need:]
        top = part[np.argsort(sums[part])[::-1]]
    # ties at the boundary: SQL any-order among equals is fine for M-kind, but be
    # deterministic: sums desc, then code asc
    top = top[np.lexsort((top, -sums[top]))][off:off + lim]
    rows = []
    ki, si = spec['ki'], spec['si']
    for code in top.tolist():
        v = wdb_sql._pyval(seg.fetch(key, int(code)))
        if isinstance(v, (bytes, bytearray)):
            v = v.decode('utf-8', 'replace')
        row = [None, None]
        row[ki] = v
        row[si] = int(sums[code])
        rows.append(tuple(row))
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in spec['proj']]
