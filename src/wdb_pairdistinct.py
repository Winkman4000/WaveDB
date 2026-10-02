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
        e0 = int(WS._code_of(seg, b, ''))
        cb = seg.cols.get(b) or {}
        pl = seg.e8_planes(b) if cb.get('code_enc') in (8, 9) and hasattr(seg, 'e8_planes') else None
        if pl is not None and int(pl[2]) == e0:
            # THE PLANES ARE THE FILTER (Q11's line items): a tiered/sparse column's planes hold
            # exactly the non-default rows and their codes -- no full decode, no 100M compare
            typed0 = np.ascontiguousarray(np.asarray(pl[0]), dtype=np.uint32)
            bt0 = np.asarray(pl[1]).astype(np.uint8)
        else:
            bc0 = np.asarray(seg._raw_codes(b))
            typed0 = np.flatnonzero(bc0 != e0).astype(np.uint32)
            bt0 = bc0[typed0].astype(np.uint8)
        ac0 = np.asarray(seg._raw_codes(a))
        uc0 = np.asarray(seg._raw_codes(u))
        import wdb_sidecar
        if not wdb_sidecar.births_on(os.path.dirname(seg.path)):          # THE SWITCH: the same four, in RAM
            return typed0, ac0[typed0].astype(np.uint8), bt0, uc0[typed0].astype(np.uint32)
        with open(p9 + '.tmp', 'wb') as f:
            f.write(np.asarray([typed0.size], np.int64).tobytes())
            f.write(typed0.tobytes())
            f.write(ac0[typed0].astype(np.uint8).tobytes())
            f.write(bt0.tobytes())
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


_NARROW = [__import__('os').environ.get('WDB_PD_NARROW', '1') != '0']


def _narrow(seg, a, b, u):
    """THE NARROW READ (2026-10-02, Q11's line items on the c6a): only the rows that pass b <> '' are ever
    read. b's planes ARE the filter (its non-default rows, ascending, and their codes); a's codes at those
    rows by THE ZIPPER from a's own planes (5.6 ms -- the old way expanded a to all 100M rows, ~31 ms, and
    gathered); u's codes at those rows only (codes_at, 35 ms -- the old way decoded all 100M, 47 ms, and
    gathered, ~24 ms). None when b is not a sparse/tiered column whose default is ''."""
    import wdb_kernels as _WK
    cb = seg.cols.get(b) or {}
    if cb.get('code_enc') not in (8, 9) or not hasattr(seg, 'e8_planes'):
        return None
    e0 = WS._code_of(seg, b, '')
    plb = seg.e8_planes(b)
    if plb is None or e0 is None or int(plb[2]) != int(e0):
        return None
    typed = np.ascontiguousarray(np.asarray(plb[0]), dtype=np.int64)
    bt = np.ascontiguousarray(np.asarray(plb[1]), dtype=np.uint16)
    ca = seg.cols.get(a) or {}
    pla = seg.e8_planes(a) if ca.get('code_enc') in (8, 9) else None
    if pla is not None:
        at = np.empty(typed.size, np.uint16)
        _WK.pd_at_planes(typed, np.ascontiguousarray(np.asarray(pla[0]), dtype=np.int64),
                         np.ascontiguousarray(np.asarray(pla[1]), dtype=np.uint16),
                         np.uint16(int(pla[2])), at, np.int64(16))
    else:
        at = np.ascontiguousarray(np.asarray(seg.codes_at(a, typed)), dtype=np.uint16)
    ut = np.ascontiguousarray(np.asarray(seg.codes_at(u, typed)), dtype=np.uint32)
    Va, Vb = int(seg.cols[a]['V']), int(seg.cols[b]['V'])
    key, ukc = _WK.pd_pair_count(at, bt, np.int64(Vb), np.int64(Va * Vb), np.int64(8))
    return key, ukc, ut


def _hunt_lanes(seg, u, key, ukc, ut, K, k):
    """Jackson's guillotine with the lane hunt: 32 pairs a round (one round on ClickBench's Q11)."""
    import wdb_kernels as _WK
    Vu = max(2, int(seg.cols[u]['V']))
    SH = max(1, int(Vu - 1).bit_length() - 6)    # <= 64 user lanes, each table 2^SH u32
    NL = ((Vu - 1) >> SH) + 1
    live = np.flatnonzero(ukc)
    live = live[np.argsort(-ukc[live], kind='stable')]
    board = []; kth = 0; idx = 0
    while idx < live.size:
        take = []
        while idx < live.size and (len(board) < k or int(ukc[live[idx]]) > kth):
            take.append(int(live[idx])); idx += 1
            if len(take) >= 32:
                break                            # the u32 marks hold 32 slots
        if not take:
            break
        tks = np.sort(np.asarray(take, np.int64))
        lut = np.full(K, -1, np.int8)
        lut[tks] = np.arange(tks.size, dtype=np.int8)
        dc = _WK.pd_hunt_lanes(key, ut, lut, np.int64(tks.size), np.int64(SH), np.int64(NL), np.int64(8))
        for j9 in range(tks.size):
            board.append((int(dc[j9]), int(tks[j9])))
        board.sort(reverse=True)
        board = board[:max(k, 12)]
        kth = board[k - 1][0] if len(board) >= k else 0
        if idx < live.size and kth >= int(ukc[live[idx]]):
            break
    return board


def execute(seg, spec):
    global _HITS
    a, b, u, k = spec['a'], spec['b'], spec['u'], spec['lim']
    Vb = int(seg.cols[b]['V'])
    nr = _narrow(seg, a, b, u) if _NARROW[0] else None
    if nr is not None:
        key, ukc, ut = nr
        board = _hunt_lanes(seg, u, key, ukc, ut, int(seg.cols[a]['V']) * Vb, k)
        return _emit(seg, spec, a, b, Vb, k, board)
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
    return _emit(seg, spec, a, b, Vb, k, board)


def _emit(seg, spec, a, b, Vb, k, board):
    global _HITS
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
