"""wdb_tripletop -- Jackson's double-bound hunt for a top-k COUNT(*) over a triple key.

SELECT kcol, extract(<part> FROM dt) AS m, tcol, COUNT(*) ... GROUP BY the three
ORDER BY COUNT(*) DESC LIMIT k -- no WHERE. <part> is a clock or calendar field with a small fixed
range (_PARTS); the two-key form drops it.

The law: a triple's count can never exceed min(user's total rows, phrase's
total rows), and BOTH censuses are one bincount away (the phrase side rides
the sparse planes). Threshold theta keeps only rows whose two bounds clear
it; survivors pack one int64 key (kcol | part | tcol, each field as wide as its codes need: bits(V_kcol),
bits(part's top value), bits(V_tcol), at most 63 in all) and one sort crowns the podium. Exactness proof: accept only when the k-th
count >= theta, so every excluded row's triple is strictly below the board.
Otherwise theta drops and the hunt re-runs -- each pass is seconds.

Lore: built for ClickBench Q18's triple-key giant.
# lore: 2026-11-07, the widening: the key had fixed fields (kcol 25b | minute 6b | tcol 23b) and only the minute.
"""
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0


# the EXTRACT fields served: field -> its top value (every value lies in 0..top), and each of wdb_sql.extract_field's
# spellings -> its field. The rest (YEAR, EPOCH, WEEK, ...) have no small fixed range: declined.
_TOPS = {'SECOND': 59, 'MINUTE': 59, 'HOUR': 23, 'DAY': 31, 'MONTH': 12, 'QUARTER': 4, 'DOW': 6, 'ISODOW': 7,
         'DOY': 366}
_PARTS = {f: n for n, fs in (('SECOND', ()), ('MINUTE', ()), ('HOUR', ()),
                                         ('DAY', ('DD', 'DAYOFMONTH')), ('MONTH', ('MON', 'MM')), ('QUARTER', ()),
                                         ('DOW', ('DAYOFWEEK', 'WEEKDAY')), ('ISODOW', ()), ('DOY', ('DAYOFYEAR',)))
          for f in (n,) + fs}


def _extract_part(p):
    """EXTRACT(field FROM col) (or its named form, MINUTE(col)) with a field in _PARTS -> (col, alias, field)"""
    inner = p.this if isinstance(p, E.Alias) else p
    fld = wdb_sql.extract_unit(inner)
    if fld in _PARTS and isinstance(wdb_sql.extract_arg(inner), E.Column):
        return wdb_sql.extract_arg(inner).name, (p.alias if isinstance(p, E.Alias) else None), _PARTS[fld]
    return None


def _bits(n):
    """the bits a value 0..n needs"""
    return int(max(0, n)).bit_length()


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') or tree.args.get('where') \
            or tree.args.get('having') or tree.args.get('distinct'):
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) not in (2, 3):
        return None
    cm = col_map or {}
    proj = tree.expressions
    dtcol = malias = part = None
    plain = []
    aggs = []
    for pi, p in enumerate(proj):
        em = _extract_part(p)
        if em is not None:
            if dtcol is not None:
                return None                      # one part only
            dtcol = cm.get(em[0], em[0]); malias = em[1]; part = em[2]
            aggs.append(('M',)); continue
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            plain.append(cm.get(inner.name, inner.name))
            aggs.append(('K', cm.get(inner.name, inner.name))); continue
        ak = wdb_sql._agg_kind(inner) if not isinstance(inner, E.Alias) else wdb_sql._agg_kind(p.this)
        if ak is not None and ak[0] == 'COUNT_STAR':
            aggs.append(('C',)); continue
        return None
    if len(plain) != 2:
        return None
    if dtcol is None and len(g.expressions) != 2:
        return None                          # 3 group keys need the part
    # group keys must be exactly the two plain columns + the part (by alias or the same expr)
    gnames = set()
    for ge in g.expressions:
        if isinstance(ge, E.Column):
            gnames.add(cm.get(ge.name, ge.name) if cm.get(ge.name, ge.name) in seg.cols
                       else ge.name)
        elif _extract_part(ge) is not None:
            eg = _extract_part(ge)
            if dtcol is None or cm.get(eg[0], eg[0]) != dtcol or eg[2] != part:
                return None                      # a group key that is another part, or of another column
            gnames.add('__m__')
        else:
            return None
    if malias and malias in gnames:
        gnames.discard(malias); gnames.add('__m__')
    want9 = set(plain) | ({'__m__'} if dtcol is not None else set())
    if gnames != want9:
        return None
    # identify tcol (the sparse text column, enc 8/9) and kcol (the dictionary key)
    tcol = kcol = None
    for cn in plain:
        c = seg.cols.get(cn)
        if c is None:
            return None
        if c.get('code_enc') in (8, 9):
            tcol = cn
        else:
            kcol = cn
    if tcol is None or kcol is None:
        return None
    if any(seg.cols[c9].get('mode') == 4 or seg.cols[c9].get('has_null') for c9 in (tcol, kcol)):
        return None                              # a sequence's codes are row positions; NULL is no group here
    mtop = 0
    if dtcol is not None:
        dc = seg.cols.get(dtcol)
        if dc is None or dc.get('dt') != 3 or dc.get('has_null'):
            return None                          # a time column; a NULL time has no part to pack
        mtop = _TOPS[part]
    # THE KEY'S FIELDS: kcol | part | tcol, each as wide as its codes need
    tbits = _bits(int(seg.cols[tcol]['V']) - 1)
    mbits = _bits(mtop) if dtcol is not None else 0
    kbits = _bits(int(seg.cols[kcol]['V']) - 1)
    if kbits + mbits + tbits > 63:
        return None
    lay = {'msh': tbits, 'ksh': tbits + mbits, 'mmask': (1 << mbits) - 1, 'tmask': (1 << tbits) - 1,
           'nm': mtop + 1, 'part': part}
    ox = tree.args.get('order'); lx = tree.args.get('limit')
    if ox is None:
        # THE FREEDOM SHAPE: ORDER-less LIMIT -- any k groups, exact counts.
        if (lx is None or tree.args.get('where') or dtcol is not None
                or kcol is None or tcol is None):
            return None
        try:
            lim9 = int(lx.expression.this)
        except Exception:
            return None
        return {'kcol': kcol, 'tcol': tcol, 'dt': None, 'aggs': aggs,
                'lim': lim9, 'free': True, 'proj': tree.expressions, **lay}
    if ox is None or lx is None or len(ox.expressions) != 1:
        return None
    o = ox.expressions[0]
    if not o.args.get('desc'):
        return None
    inner_o = o.this
    if not (isinstance(inner_o, E.Count) or isinstance(inner_o, E.Column)):
        return None
    try:
        lim = int(lx.expression.this)
    except Exception:
        return None
    return {'kcol': kcol, 'tcol': tcol, 'dt': dtcol, 'aggs': aggs, 'lim': lim,
            'proj': proj, **lay}


def _part_table(seg, dtcol, part):
    """code -> the part of that code's time: wdb_sql.extract_field (the row evaluator's own definition) over the
    V distinct times, never the N rows"""
    c = seg.cols[dtcol]
    V = int(c['V'])
    vals = np.asarray(seg._dict_ints_at(c, np.arange(V, dtype=np.int64)), np.int64)
    return np.asarray(wdb_sql.extract_field(part, vals.view('datetime64[%s]' % seg.unit(dtcol))), np.int64)


_TTBITS = [__import__('os').environ.get('WDB_TT_CHECKLISTS', '1') != '0']
TALLY_MAX_BYTES = 512 << 20     # the checklists' (heavy person x minute) tally boards, all lanes


def _emit_rows(seg, spec, kcol, tcol, picks):
    """picks: (packed key kcol<<ksh | part<<msh | tcol, count), best first -> the result rows"""
    global _HITS
    rows = []
    ksh, msh, mmask, tmask = spec['ksh'], spec['msh'], spec['mmask'], spec['tmask']
    for kk, c9 in picks:
        ucode = kk >> ksh; mv = (kk >> msh) & mmask; scode = kk & tmask
        uv = seg.fetch(kcol, ucode)
        sv = seg.fetch(tcol, scode)
        if isinstance(uv, (bytes, bytearray)):
            uv = uv.decode('utf-8', 'replace')
        if isinstance(sv, (bytes, bytearray)):
            sv = sv.decode('utf-8', 'replace')
        uv = wdb_sql._pyval(uv); sv = wdb_sql._pyval(sv)   # a time key emits as a time, not numpy datetime64
        row = []
        for a in spec['aggs']:
            row.append(uv if a[0] == 'K' and a[1] == kcol else
                       sv if a[0] == 'K' else
                       int(mv) if a[0] == 'M' else c9)
        rows.append(tuple(row))
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in spec['proj']]


def _walk_checklists(seg, spec, kc, kcnt, pos8, lits8, tcnt, e0, emptyc, dtc, mt):
    """THE CHECKLIST WALK (WDB_TT_CHECKLISTS=0 restores tt_survivors).
    The same double bound (a triple's count <= its user's rows and <= its phrase's rows), asked of
    1-bit checklists instead of the 141 MB census; the part from the time column's staircase steps (no 100M
    decode); empty-phrase survivors counted on a (heavy user x the part's NM values) tally, only real-phrase
    survivors packed and sorted. Exact under the same law: accept when the k-th count >= theta. None
    when the time column is not a clean staircase (one step per code, no nulls) or the tally is too big.

    Lore: 2026-10-02, from ClickBench Q18's line items; the staircase column there was EventTime."""
    import wdb_kernels as WK
    kcol, tcol, k = spec['kcol'], spec['tcol'], spec['lim']
    ct = seg.cols[dtc]
    Vt = int(ct['V'])
    steps = seg.stairs(dtc)
    if steps is None or ct.get('has_null') or int(np.asarray(steps).size) != Vt - 1 or mt.size != Vt:
        return None
    steps = np.ascontiguousarray(steps, np.int64)
    N = int(seg.N)
    L = 8
    bounds = (np.arange(L + 1, dtype=np.int64) * N) // L
    koff = np.ascontiguousarray(np.searchsorted(pos8, bounds)[:-1], np.int64)
    kout = np.empty(max(1, pos8.size), np.int64)          # lane l's real-phrase keys <= its plane rows
    NM, ksh, msh = int(spec['nm']), int(spec['ksh']), int(spec['msh'])
    theta = 256
    while True:
        hv = kcnt >= theta
        heavy = np.flatnonzero(hv)
        H = int(heavy.size)
        empty_ok = bool(emptyc >= theta)
        if empty_ok and L * max(1, H) * NM * 4 > TALLY_MAX_BYTES:
            return 'TALLY_TOO_BIG'                        # the tally would not be small: the old walk (or a decline)
        ubit = np.packbits(hv, bitorder='little')
        pbit = np.packbits(tcnt >= theta, bitorder='little')
        uidx = np.full(kcnt.size, -1, np.int32)
        uidx[heavy] = np.arange(H, dtype=np.int32)
        board = np.zeros((L, max(1, H) * NM if empty_ok else 1), np.int32)
        klen = np.zeros(L, np.int64)
        WK.tt_walk(kc, ubit, uidx, pos8, lits8, pbit, steps, mt, np.int64(e0), empty_ok,
                   np.int64(NM), board, kout, koff, klen, np.int64(ksh), np.int64(msh))
        keys, cnts = [], []
        if empty_ok:
            tally = board.sum(axis=0, dtype=np.int64)
            nz = np.flatnonzero(tally)
            keys.append((heavy[nz // NM].astype(np.int64) << ksh) | ((nz % NM).astype(np.int64) << msh) | np.int64(e0))
            cnts.append(tally[nz])
        kk = np.concatenate([kout[int(koff[l]):int(koff[l]) + int(klen[l])] for l in range(L)])
        if kk.size:
            kk.sort()
            brk = np.empty(kk.size, bool); brk[0] = True
            np.not_equal(kk[1:], kk[:-1], out=brk[1:])
            st = np.flatnonzero(brk)
            keys.append(kk[st]); cnts.append(np.diff(np.append(st, kk.size)))
        allk = np.concatenate(keys) if keys else np.empty(0, np.int64)
        allc = np.concatenate(cnts) if cnts else np.empty(0, np.int64)
        if allk.size >= k or theta <= 1:
            kk9 = min(k, int(allk.size))
            if kk9:
                topi = np.argpartition(-allc, kk9 - 1)[:kk9]
                order = topi[np.argsort(-allc[topi], kind='stable')]
                kth = int(allc[order[-1]])
            else:
                order = np.empty(0, np.int64); kth = 0
            if (allk.size >= k and kth >= theta) or theta <= 1:
                return _emit_rows(seg, spec, kcol, tcol, [(int(allk[g]), int(allc[g])) for g in order.tolist()])
        theta //= 4


def execute(seg, spec):
    global _HITS
    import wdb_wherescan as WS
    kcol, tcol, dtc, k = spec['kcol'], spec['tcol'], spec['dt'], spec['lim']
    N = int(seg.N)
    import wdb_kernels as WK
    kc = np.ascontiguousarray(seg._raw_codes(kcol))
    kcnt = None
    try:
        import wdb_gbshelf
        sh = wdb_gbshelf.open_shelf(seg, kcol)
        if sh is not None:
            # lore: the Q19 shelf pays Q18's rent
            kcnt = wdb_gbshelf.bulk(sh)          # an existing count shelf on the person column gives its census
    except Exception:
        kcnt = None
    if kcnt is None:
        kcnt = seg.raw_census(kcol, int(seg.cols[kcol]['V']))       # THE PERSON COUNT FROM THE SHELVES (shelved enc 19)
                                                                    # else decode + parallel boards, as before
    kcnt = np.ascontiguousarray(kcnt, np.int64)
    pl = seg.e8_planes(tcol)
    if pl is None:
        return None
    pos8 = np.ascontiguousarray(pl[0], np.int64)
    lits8 = np.ascontiguousarray(pl[1], np.int64)
    tcnt = np.ascontiguousarray(WK.bincount_par(lits8, int(seg.cols[tcol]['V'])), np.int64)   # was one thread, 52 ms
    e0 = int(pl[2])                              # the rows outside the planes hold the stored default (c129: it was
                                                 # taken to be '', which labelled them wrongly on any other column)
    emptyc = N - int(pos8.size)
    if dtc is not None:
        mt = np.ascontiguousarray(_part_table(seg, dtc, spec['part']))
        if _TTBITS[0] and not spec.get('free'):
            got = _walk_checklists(seg, spec, kc, kcnt, pos8, lits8, tcnt, e0, emptyc, dtc, mt)
            if got == 'TALLY_TOO_BIG':
                # THE FLAT-PERSON GATE: hundreds of thousands of "person" values clear the bar at once (a
                # time-like person: ~40 rows a value, none heavy) -- the checklists cannot tally them
                # and the old walk ladders the bar down over every row. Hand the query on -- unless the person IS
                # the minute's own time column (t x minute of t): its groups never split by minute, the old walk
                # is quick.
                # lore: 2026-10-14, the Q18 board: ClientEventTime, LocalEventTime as persons, 3.4-3.8 s (old walk).
                # lore: the bound hunt's flat scan answers them in ~1.0 s; EventTime x minute of EventTime: old walk
                # lore: 0.7 s against the hunt's 1.3 s. Measured on all 124 tripletop pairs.
                if kcol != dtc:
                    return None
                got = None
            if got is not None:
                return got
        ec = np.ascontiguousarray(seg._raw_codes(dtc))
    else:
        ec = np.zeros(1, np.int64)           # never read: has_m gates the branch
        mt = np.zeros(1, np.int64)
    if spec.get('free'):
        return _free_serve(seg, spec)
    if dtc is None:
        # JACKSON'S HEAD-FIRST HUNT: '' owns ~87% of rows, so (user, '')
        # dominates the board -- and its counts are pure census arithmetic:
        # E(u) = total(u) - plane(u). Real-phrase pairs are bounded by the
        # user's PLANE rows, so only users with plane(u) > kth can compete;
        # usually that set is empty and no row is ever walked.
        V9u = int(seg.cols[kcol]['V'])
        ucp = kc[pos8]                       # plane rows' users (13.2M)
        pcnt = WK.bincount_par(ucp, V9u)
        # Jackson's cut: top users come off the gbc2 shelf's u16 tail --
        # outside the >=4 tier, total<=3 so E<=3 and nobody boards. The
        # M-window widens until kth >= the M-th total (exactness guard).
        tot9 = None
        try:
            import wdb_gbshelf
            sh0 = wdb_gbshelf.open_shelf(seg, kcol)
            if sh0 is not None and sh0['V'] == V9u:
                t4pos = np.flatnonzero(np.unpackbits(sh0['bm4'])[:V9u])
                t4cnt = np.asarray(sh0['tail'], np.int64)
                board = None
                M9 = max(64, 4 * k)
                while True:
                    M9 = min(M9, t4cnt.size)
                    part = np.argpartition(-t4cnt, M9 - 1)[:M9] if M9 else np.empty(0, np.int64)
                    uu = t4pos[part]
                    EE = t4cnt[part] - pcnt[uu]
                    kk9 = min(k, int((EE > 0).sum()) + 1)
                    b2 = sorted(((int(EE[j]), int(uu[j]), int(e0))
                                 for j in range(uu.size)), reverse=True)[:k]
                    kth0 = b2[k - 1][0] if len(b2) >= k else 0
                    Mth = int(np.min(t4cnt[part])) if part.size else 0
                    if kth0 >= Mth or M9 >= t4cnt.size:
                        board = b2
                        break
                    M9 *= 4                  # a bigger E may hide below: widen
                if board is not None:
                    tot9 = True
        except Exception:
            tot9 = None
        if tot9 is None:
            if kcnt is None:
                kcnt = np.bincount(kc, minlength=V9u)
            E = kcnt - pcnt
            kk9 = min(k, int((E > 0).sum()))
            topi = WK.topk_bar(E, kk9) if kk9 else np.empty(0, np.int64)   # THE BAR, not a 17.6M partition
            board = [(int(E[u]), int(u), int(e0)) for u in topi.tolist()]
            board.sort(reverse=True)
        kth9 = board[k - 1][0] if len(board) >= k else 0
        if kth9 <= 0:
            # THE EMPTY BAR: fewer than k persons own an empty-phrase group (a yes/no or tiny person), so the bar
            # is 0 and the walk would pack and SORT every non-empty phrase row on one core. Hand the query on: the
            # pair count composed from the atoms (gridwalk: a parallel tally of the cells) answers it.
            # lore: 2026-10-15, the Q16 board: GoodEvent, IsOldCounter as persons; 13.2M rows, 4.9 s against DuckDB's
            # lore: 0.55. There gridwalk tallies the 2 x 6M cells.
            return None
        cand = np.flatnonzero(pcnt > kth9)
        if cand.size:
            cm = np.zeros(kcnt.size, bool)
            cm[cand] = True
            spm9 = tcnt > kth9                # the phrase-cut: a pair can't beat
            sel = np.flatnonzero(cm[ucp] & spm9[lits8])   # kth on a rarer phrase
            # the person's codes widen BEFORE the shift (2026-10-15: a narrow person -- uint8 codes, a yes/no
            # column -- shifted 23 places inside its own dtype came back 0, and every person's phrase rows merged)
            pk = (ucp[sel].astype(np.int64) << spec['ksh']) | lits8[sel]
            pk.sort(kind='stable')
            b9 = np.empty(pk.size, bool)
            if pk.size:
                b9[0] = True
                np.not_equal(pk[1:], pk[:-1], out=b9[1:])
                s9 = np.flatnonzero(b9)
                c9a = np.diff(np.append(s9, pk.size))
                for j9 in range(s9.size):
                    board.append((int(c9a[j9]), int(pk[s9[j9]] >> spec['ksh']),
                                  int(pk[s9[j9]] & spec['tmask'])))
            board.sort(reverse=True)
        rows = []
        for c9, u9, s9c in board[:k]:
            uv = seg.fetch(kcol, u9)
            sv = seg.fetch(tcol, s9c)
            if isinstance(uv, (bytes, bytearray)):
                uv = uv.decode('utf-8', 'replace')
            if isinstance(sv, (bytes, bytearray)):
                sv = sv.decode('utf-8', 'replace')
            uv = wdb_sql._pyval(uv); sv = wdb_sql._pyval(sv)   # a time key emits as a time, not numpy datetime64
            row = []
            for a9 in spec['aggs']:
                row.append(uv if a9[0] == 'K' and a9[1] == kcol else
                           sv if a9[0] == 'K' else c9)
            rows.append(tuple(row))
        _HITS += 1
        return rows, [wdb_sql._alias(p) for p in spec['proj']]
    T9 = 32
    theta = 256
    while True:
        cap = N // T9 + 65536
        outs = np.empty((T9, cap), np.int64)
        lens = np.zeros(T9, np.int64)
        WK.tt_survivors(kc, kcnt, pos8, lits8, tcnt, ec, mt,
                        np.int64(e0), np.int64(emptyc), np.int64(theta),
                        outs, lens, dtc is not None, np.int64(spec['ksh']), np.int64(spec['msh']))
        key = np.concatenate([outs[t, :int(lens[t])] for t in range(T9)])             if int(lens.sum()) else np.empty(0, np.int64)
        idx = key                                 # naming kept for flow below
        if key.size:
            SHk = max(0, int(key.max()).bit_length() - 12)
            key = WK.sort_keys_par(key, np.int64(SHk))   # THE BUCKETED SORT (np.sort: 217 ms, one thread)
            brk9 = np.empty(key.size, bool)
            brk9[0] = True
            np.not_equal(key[1:], key[:-1], out=brk9[1:])
            st9 = np.flatnonzero(brk9)
            uq = key[st9]
            cn = np.diff(np.append(st9, key.size))
            if uq.size >= k:
                topi = np.argpartition(-cn, k - 1)[:k]
                order = topi[np.argsort(-cn[topi], kind='stable')]
                kth = int(cn[order[-1]])
                if kth >= theta or theta <= 1:
                    return _emit_rows(seg, spec, kcol, tcol, [(int(uq[g]), int(cn[g])) for g in order.tolist()])
        if theta <= 1:
            return None                      # exhausted: yield to the general road
        theta //= 4


def _free_serve(seg, spec):
    """Jackson's freedom serve for ORDER-less LIMIT: pick k random LOW-COUNT
    phrase codes -- codes differentiate without decoding -- gather each code's
    complete instance list from the plane (its whole truth), count per user
    inside it (exact by construction), decode only the winners."""
    global _HITS
    import wdb_sql
    kcol, tcol, k = spec['kcol'], spec['tcol'], spec['lim']
    pl = seg.e8_planes(tcol)
    pos8 = np.asarray(pl[0], np.int64)
    lits8 = np.asarray(pl[1], np.int64)
    tcnt = np.bincount(lits8, minlength=int(seg.cols[tcol]['V']))
    low = np.flatnonzero((tcnt >= 1) & (tcnt <= 4))
    rng = np.random.default_rng()
    picks = rng.choice(low, size=min(k, low.size), replace=False)         if low.size else np.empty(0, np.int64)
    sel = np.flatnonzero(np.isin(lits8, picks))
    rows9 = pos8[sel]
    codes9 = lits8[sel]
    import wdb_pairtop
    uu = wdb_pairtop._pt_codes(seg, kcol, rows9)
    groups = {}
    for j in range(rows9.size):
        groups.setdefault((int(uu[j]), int(codes9[j])), 0)
        groups[(int(uu[j]), int(codes9[j]))] += 1
    out = []
    for (uc9, sc9), c9 in list(groups.items())[:k]:
        uv = seg.fetch(kcol, uc9)
        sv = seg.fetch(tcol, sc9)
        if isinstance(uv, (bytes, bytearray)):
            uv = uv.decode('utf-8', 'replace')
        if isinstance(sv, (bytes, bytearray)):
            sv = sv.decode('utf-8', 'replace')
        uv = wdb_sql._pyval(uv); sv = wdb_sql._pyval(sv)   # a time key emits as a time, not numpy datetime64
        row = []
        for a9 in spec['aggs']:
            row.append(uv if a9[0] == 'K' and a9[1] == kcol else
                       sv if a9[0] == 'K' else c9)
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
