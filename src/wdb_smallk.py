"""
wdb_smallk -- the small-K weapon: GROUP BY 2-3 narrow keys + COUNT(*), answered by
one fused pass over raw code streams onto a composite bean board. The router's twin
gates: diskpair owns wide dictionaries; this owns products up to _CAP cells. Built
the day the ten-worst-ratios list showed pair/triple counting at the pandas desk.
Also serves ROLLUP/CUBE for free: groupsets' finest sub-query re-enters db.run and
lands here.
"""
import numpy as np
import wdb_sql
import wdb_policies as P
import wdb_kernels as WK
import sqlglot.expressions as E

_HITS = 0

_CAP = 1 << 26          # 67M cells: the small-K frontier (board = 512MB transient max)


def detect(seg, tree, col_map):
    if not P.no_joins(tree) or not P.no_where(tree) or not P.no_having(tree):
        return None
    if not P.no_select_distinct(tree):
        return None
    group = tree.args.get('group')
    if group is None or len(group.expressions) not in (2, 3):
        return None
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    keys = []
    for g in group.expressions:
        nm = wdb_sql._colname(g)
        if nm is None:
            return None
        keys.append(sc(nm))
    proj = tree.expressions
    if len(proj) != len(keys) + 1:
        return None
    order_slots, cnt_i, seen = [], None, set()
    for i, p in enumerate(proj):
        kind = wdb_sql._agg_kind(p)
        if kind is None:
            nm = sc(wdb_sql._proj_colname(p) or '')
            if nm not in keys or nm in seen:
                return None
            seen.add(nm)
            order_slots.append(('k', keys.index(nm)))
        elif kind[0] == 'COUNT_STAR' and cnt_i is None:
            cnt_i = i
            order_slots.append(('c', None))
        else:
            return None
    if cnt_i is None:
        return None
    spans = []
    for nm in keys:
        c = seg.cols.get(nm)
        if c is None or c.get('mode') not in (0, 1, 2) or c.get('has_null'):
            return None
        spans.append(int(c['V']))
    total = 1
    for v in spans:
        total *= v
    if total > _CAP:
        return None                                  # wide territory: diskpair's desk
    od = tree.args.get('order')
    desc = None
    if od is not None:
        if len(od.expressions) != 1:
            return None
        oe = od.expressions[0]
        onm = wdb_sql._colname(oe.this) or wdb_sql._alias(proj[cnt_i])
        if onm != wdb_sql._alias(proj[cnt_i]):
            return None
        desc = bool(oe.args.get('desc'))
    if not P.no_deleted_rows(seg):
        return None
    return keys, spans, order_slots, desc, wdb_sql._limit(tree), (wdb_sql._offset(tree) or 0), proj


def execute(seg, spec):
    global _HITS
    _HITS += 1
    keys, spans, slots, desc, lim, off, proj = spec
    codes = [np.asarray(seg._raw_codes(nm)) for nm in keys]     # native dtypes: no astype
    board = WK.grid_count(codes, spans)
    gid = np.flatnonzero(board)
    cnt = board[gid]
    if desc is not None:
        o = np.lexsort((gid, -cnt)) if desc else np.lexsort((gid, cnt))
        gid, cnt = gid[o], cnt[o]
    if lim is not None or off:
        gid = gid[off: None if lim is None else off + lim]
        cnt = cnt[off: None if lim is None else off + lim]
    parts = []
    rem = gid.astype(np.int64)
    for i in range(len(keys) - 1, 0, -1):
        parts.append(rem % spans[i])
        rem = rem // spans[i]
    parts.append(rem)
    parts = parts[::-1]                              # per-key code arrays, cells only
    tabs = []
    for i, nm in enumerate(keys):
        vals = {}
        for code in np.unique(parts[i]):
            v = wdb_sql._pyval(seg.fetch(nm, int(code)))
            if isinstance(v, (bytes, bytearray)):
                v = v.decode('utf-8', 'replace')
            vals[int(code)] = v
        tabs.append(vals)
    rows = []
    for r in range(gid.size):
        row = []
        for kind, ki in slots:
            if kind == 'k':
                row.append(tabs[ki][int(parts[ki][r])])
            else:
                row.append(int(cnt[r]))
        rows.append(tuple(row))
    return rows, [wdb_sql._alias(p) for p in proj]
