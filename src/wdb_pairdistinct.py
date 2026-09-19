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


def _tier_shelf(seg, a, b, u):
    """The dress's serving layer (rule eleven), a birth-on-touch sidecar:
    typed row positions, both tiny columns' compacted typed codes, AND the
    differentiator's typed codes -- 'read only what you need' as an mmap
    slice. Born once; no zstd frame pops again for this query family."""
    import os
    p9 = os.path.join(os.path.dirname(seg.path),
                      'tier2__%s__%s__%s.bin' % (a, b, u))
    if not os.path.exists(p9):
        bc0 = np.asarray(seg._raw_codes(b))
        ac0 = np.asarray(seg._raw_codes(a))
        uc0 = np.asarray(seg._raw_codes(u))
        e0 = int(WS._code_of(seg, b, ''))
        typed0 = np.flatnonzero(bc0 != e0).astype(np.uint32)
        import wdb_sidecar
        if not wdb_sidecar.births_on(os.path.dirname(seg.path)):          # THE SWITCH: the same four, in RAM
            return typed0, ac0[typed0].astype(np.uint8), bc0[typed0].astype(np.uint8), uc0[typed0].astype(np.uint32)
        with open(p9 + '.tmp', 'wb') as f:
            f.write(np.asarray([typed0.size], np.int64).tobytes())
            f.write(typed0.tobytes())
            f.write(ac0[typed0].astype(np.uint8).tobytes())
            f.write(bc0[typed0].astype(np.uint8).tobytes())
            f.write(uc0[typed0].astype(np.uint32).tobytes())
        os.replace(p9 + '.tmp', p9)
        import wdb_shelves
        wdb_shelves.record(seg, 'tier2', a=a, b=b, u=u)
    mm = np.memmap(p9, dtype=np.uint8, mode='r')
    n9 = int(np.frombuffer(mm[:8], np.int64)[0])
    typed = np.frombuffer(mm[8:8 + 4 * n9], np.uint32)
    at = np.frombuffer(mm[8 + 4 * n9:8 + 5 * n9], np.uint8)
    bt = np.frombuffer(mm[8 + 5 * n9:8 + 6 * n9], np.uint8)
    ut = np.frombuffer(mm[8 + 6 * n9:8 + 10 * n9], np.uint32)
    return typed, at, bt, ut


def execute(seg, spec):
    global _HITS
    a, b, u, k = spec['a'], spec['b'], spec['u'], spec['lim']
    Vb = int(seg.cols[b]['V'])
    typed, at9, bt9, ut9 = _tier_shelf(seg, a, b, u)
    key = at9.astype(np.int64) * Vb + bt9
    ukc = np.bincount(key, minlength=int(seg.cols[a]['V']) * Vb)
    # Jackson's guillotine, sort-free: candidates are pairs whose ROW count
    # could still beat the k-th DISTINCT count; distinct <= rows prunes the
    # rest before UserID is ever touched. One packed sort dedups them all.
    live = np.flatnonzero(ukc)
    live = live[np.argsort(-ukc[live], kind='stable')]
    board = []                               # (distinct, pairkey)
    kth = 0
    idx = 0
    while idx < live.size:
        take = [];
        while idx < live.size and (len(board) < k or int(ukc[live[idx]]) > kth):
            take.append(int(live[idx])); idx += 1
            if len(take) >= 16:
                break                            # the u16 jar holds 16 bits
        if not take:
            break
        tks = np.sort(np.asarray(take, np.int64))
        lut = np.full(int(seg.cols[a]['V']) * Vb, -1, np.int16)
        lut[tks] = np.arange(tks.size, dtype=np.int16)
        import wdb_kernels as _WK
        dc = _WK.pd_hunt(key, ut9, lut,
                         int(seg.cols[u]['V']), tks.size)
        for j9 in range(tks.size):           # the stamp jar: one pass,
            board.append((int(dc[j9]), int(tks[j9])))   # no sort at all
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
