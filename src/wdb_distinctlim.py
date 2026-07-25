"""
wdb_distinctlim -- DISTINCT (extract(hour FROM t), key) LIMIT n: the early exit.

The full-scan path decodes 100M timestamps to find fifty pairs that live in the
first block. Here: hour-of-code table from the dictionary once (V ints), per-row
order codes ride the stairs (searchsorted per block -- the timestamp stream is
never decompressed), the key column reads block by block, and the walk stops the
moment LIMIT distinct pairs exist. Values are for winners only: <= n key fetches.
"""
import numpy as np
import wdb_sql
import wdb_policies as P
import sqlglot.expressions as E

_HITS = 0


def detect(seg, tree, col_map):
    if not tree.args.get('distinct'):
        return None
    if not P.no_joins(tree) or not P.no_where(tree) or not P.no_having(tree):
        return None
    if tree.args.get('group') is not None or tree.args.get('order') is not None:
        return None
    lim = wdb_sql._limit(tree)
    if lim is None or lim > 100000:
        return None
    proj = tree.expressions
    if len(proj) != 2:
        return None
    ex = proj[0].this if isinstance(proj[0], E.Alias) else proj[0]
    if not isinstance(ex, E.Extract):
        return None
    unit = str(ex.this.name if hasattr(ex.this, 'name') else ex.this).lower()
    if unit != 'hour':
        return None
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    tcol = wdb_sql._colname(ex.expression)
    kcol = wdb_sql._proj_colname(proj[1])
    if tcol is None or kcol is None:
        return None
    tcol, kcol = sc(tcol), sc(kcol)
    ct, ck = seg.cols.get(tcol), seg.cols.get(kcol)
    if ct is None or ck is None:
        return None
    if ct.get('dt') != 3 or ct.get('mode') != 2 or seg.stairs(tcol) is None:
        return None
    if ck.get('mode') not in (0, 1, 2) or ck.get('has_null'):
        return None
    if not P.no_deleted_rows(seg):
        return None
    return {'tcol': tcol, 'kcol': kcol, 'lim': lim, 'off': wdb_sql._offset(tree) or 0,
            'proj': proj}


def execute(seg, spec):
    global _HITS
    tcol, kcol, lim, off = spec['tcol'], spec['kcol'], spec['lim'], spec['off']
    import wdb_window as W
    tvals = W._int_table(seg, tcol)                       # dict seconds, V-sized, once
    hour_tab = ((tvals % 86400) // 3600).astype(np.int64)
    stairs = np.asarray(seg.stairs(tcol))
    Vc = int(seg.cols[kcol]['V'])
    BR = int(seg.cols[kcol].get('BR') or 524288)
    need = lim + off
    seen = set()
    ordered = []
    for a in range(0, seg.N, BR):
        b = min(seg.N, a + BR)
        rows = np.arange(a, b)
        hcodes = np.searchsorted(stairs, rows, side='right')
        hh = hour_tab[hcodes]
        cc = np.asarray(seg.codes_at(kcol, rows)).astype(np.int64)
        pair = hh * Vc + cc
        for u in np.unique(pair).tolist():
            if u not in seen:
                seen.add(u)
                ordered.append(u)
        if len(ordered) >= need:
            break
    pick = ordered[off: off + lim]
    rows_out = []
    for u in pick:
        h, c = divmod(u, Vc)
        v = wdb_sql._pyval(seg.fetch(kcol, int(c)))
        if isinstance(v, (bytes, bytearray)):
            v = v.decode('utf-8', 'replace')
        rows_out.append((int(h), v))
    _HITS += 1
    return rows_out, [wdb_sql._alias(p) for p in spec['proj']]
