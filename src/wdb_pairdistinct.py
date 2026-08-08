"""wdb_pairdistinct -- Q11's guillotine hunt (Jackson's order of operations).

SELECT a, b, COUNT(DISTINCT u) FROM t WHERE b <> '' GROUP BY a, b
ORDER BY cnt DESC LIMIT k   (a, b low-V; u the big differentiator)

Reduce rows by the pair row-census first (the tiny columns cost nothing);
then correct counts rows -> distinct users, biggest pair first, reading
UserID only at each candidate pair's rows; STOP when the k-th distinct
beats every remaining pair's ROW count -- distinct <= rows, so nothing
below the guillotine can climb.
"""
import numpy as np
from sqlglot import expressions as E

import wdb_sql
import wdb_wherescan as WS

_HITS = 0


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') \
            or tree.args.get('having') or tree.args.get('distinct'):
        return None
    w9 = tree.args.get('where')
    if w9 is None:
        return None
    n9 = w9.this
    if not (isinstance(n9, E.NEQ) and isinstance(n9.this, E.Column)
            and isinstance(n9.expression, E.Literal)
            and str(n9.expression.this) == ''):
        return None
    cm = col_map or {}
    fcol = cm.get(n9.this.name, n9.this.name)
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 2:
        return None
    gcols = []
    for ge in g.expressions:
        if not isinstance(ge, E.Column):
            return None
        gcols.append(cm.get(ge.name, ge.name))
    if fcol not in gcols:
        return None
    ucol = None
    calias = None
    aggs = []
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            cn = cm.get(inner.name, inner.name)
            if cn not in gcols:
                return None
            aggs.append(('K', cn)); continue
        if isinstance(inner, E.Count) and inner.args.get('distinct') is None:
            di = inner.this
            if isinstance(di, E.Distinct) and len(di.expressions) == 1 \
                    and isinstance(di.expressions[0], E.Column):
                ucol = cm.get(di.expressions[0].name, di.expressions[0].name)
                aggs.append(('D',))
                if isinstance(p, E.Alias):
                    calias = p.alias
                continue
        if isinstance(inner, E.Count) and isinstance(inner.this, E.Distinct):
            di = inner.this
            if len(di.expressions) == 1 and isinstance(di.expressions[0], E.Column):
                ucol = cm.get(di.expressions[0].name, di.expressions[0].name)
                aggs.append(('D',))
                if isinstance(p, E.Alias):
                    calias = p.alias
                continue
        return None
    if ucol is None:
        return None
    for cn in gcols:
        c = seg.cols.get(cn)
        if c is None or int(c.get('V') or 1 << 30) > 4096:
            return None                          # the census must be near-free
    if seg.cols.get(ucol) is None:
        return None
    ox = tree.args.get('order'); lx = tree.args.get('limit')
    if ox is None or lx is None or len(ox.expressions) != 1:
        return None
    o = ox.expressions[0]
    if not o.args.get('desc'):
        return None
    io = o.this
    if isinstance(io, E.Column):
        if calias is None or io.name != calias:
            return None
    try:
        lim = int(lx.expression.this)
    except Exception:
        return None
    return {'a': gcols[0] if gcols[0] != fcol else gcols[1],
            'b': fcol, 'u': ucol, 'lim': lim, 'aggs': aggs,
            'proj': tree.expressions}


def execute(seg, spec):
    global _HITS
    a, b, u, k = spec['a'], spec['b'], spec['u'], spec['lim']
    ac = np.asarray(seg._raw_codes(a))
    bc = np.asarray(seg._raw_codes(b))
    e0 = int(WS._code_of(seg, b, ''))
    typed = np.flatnonzero(bc != e0)
    key = (ac[typed].astype(np.int64) << 16) | bc[typed]
    uk, inv, ukc = np.unique(key, return_inverse=True, return_counts=True)
    order = np.argsort(-ukc, kind='stable')
    uu_all = np.asarray(seg._raw_codes(u))[typed]   # ONE read; slices after
    gsort = np.argsort(inv, kind='stable')       # pair-grouped order, ONCE --
    gends = np.cumsum(ukc)                       # no 5.6M scan per pair
    board = []                                   # (distinct, acode, bcode)
    kth = 0
    for oi in order.tolist():
        if len(board) >= k and kth >= int(ukc[oi]):
            break                                # the guillotine: distinct<=rows
        lo9 = int(gends[oi - 1]) if oi else 0
        uu = uu_all[gsort[lo9:int(gends[oi])]]
        d9 = int(np.unique(uu).size)
        board.append((d9, int(uk[oi] >> 16), int(uk[oi] & 0xFFFF)))
        board.sort(reverse=True)
        board = board[:max(k, 12)]
        kth = board[k - 1][0] if len(board) >= k else 0
    out = []
    for d9, acode, bcode in board[:k]:
        av = seg.fetch(a, acode)
        bv = seg.fetch(b, bcode)
        if isinstance(av, (bytes, bytearray)):
            av = av.decode('utf-8', 'replace')
        if isinstance(bv, (bytes, bytearray)):
            bv = bv.decode('utf-8', 'replace')
        row = []
        for t in spec['aggs']:
            if t[0] == 'K':
                row.append(av if t[1] == a else bv)
            else:
                row.append(d9)
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
