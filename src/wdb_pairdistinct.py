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
    ac = np.asarray(seg._raw_codes(a)).astype(np.int64)
    bc = np.asarray(seg._raw_codes(b)).astype(np.int64)
    e0 = int(WS._code_of(seg, b, ''))
    Vb = int(seg.cols[b]['V'])
    typed = np.flatnonzero(bc != e0)
    key = ac[typed] * Vb + bc[typed]         # <= Va*Vb cells: bincount land
    ukc = np.bincount(key, minlength=int(seg.cols[a]['V']) * Vb)
    # Jackson's guillotine, sort-free: candidates are pairs whose ROW count
    # could still beat the k-th DISTINCT count; distinct <= rows prunes the
    # rest before UserID is ever touched. One packed sort dedups them all.
    live = np.flatnonzero(ukc)
    live = live[np.argsort(-ukc[live], kind='stable')]
    ubits = max(1, int(seg.cols[u].get('bits') or 25))
    uu_all = np.asarray(seg._raw_codes(u))[typed]
    board = []                               # (distinct, pairkey)
    kth = 0
    idx = 0
    while idx < live.size:
        take = [];
        while idx < live.size and (len(board) < k or int(ukc[live[idx]]) > kth):
            take.append(int(live[idx])); idx += 1
            if len(take) >= max(k, 12) and len(board) >= k:
                break
        if not take:
            break
        tk = np.asarray(take, np.int64)
        m9 = np.isin(key, tk)
        pk = (key[m9] << ubits) | uu_all[m9]
        pk.sort()                            # ONE flat sort dedups every
        brk = np.empty(pk.size, bool)        # candidate pair at once
        if pk.size:
            brk[0] = True
            np.not_equal(pk[1:], pk[:-1], out=brk[1:])
        dk = pk[brk] >> ubits
        du, dc = np.unique(dk, return_counts=True)
        for j9 in range(du.size):
            board.append((int(dc[j9]), int(du[j9])))
        board.sort(reverse=True)
        board = board[:max(k, 12)]
        kth = board[k - 1][0] if len(board) >= k else 0
        if idx < live.size and kth >= int(ukc[live[idx]]):
            break                            # nothing left can climb
    board = [(d9, pk9 // Vb, pk9 % Vb) for d9, pk9 in board[:k]]
    out = []
    for d9, acode, bcode in board:
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
