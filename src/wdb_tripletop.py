"""wdb_tripletop -- Jackson's double-bound hunt for Q18's triple-key giant.

SELECT uid, extract(minute FROM dt) AS m, sp, COUNT(*) ... GROUP BY the three
ORDER BY COUNT(*) DESC LIMIT k -- no WHERE.

The law: a triple's count can never exceed min(user's total rows, phrase's
total rows), and BOTH censuses are one bincount away (the phrase side rides
the sparse planes). Threshold theta keeps only rows whose two bounds clear
it; survivors pack a 54-bit key (uid 25b | minute 6b | sp 23b) and one
np.unique crowns the podium. Exactness proof: accept only when the k-th
count >= theta, so every excluded row's triple is strictly below the board.
Otherwise theta drops and the hunt re-runs -- each pass is seconds.
"""
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0


def _extract_minute(p):
    inner = p.this if isinstance(p, E.Alias) else p
    if isinstance(inner, E.Extract) and isinstance(inner.expression, E.Column):
        if str(inner.this.name if hasattr(inner.this, 'name') else inner.this).lower() == 'minute':
            return inner.expression.name, (p.alias if isinstance(p, E.Alias) else None)
    return None


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') or tree.args.get('where') \
            or tree.args.get('having') or tree.args.get('distinct'):
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) not in (2, 3):
        return None
    cm = col_map or {}
    proj = tree.expressions
    dtcol = malias = None
    plain = []
    aggs = []
    for pi, p in enumerate(proj):
        em = _extract_minute(p)
        if em is not None:
            dtcol = cm.get(em[0], em[0]); malias = em[1]
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
        return None                          # 3 group keys need the minute
    # group keys must be exactly the two plain columns + the minute (by alias or expr)
    gnames = set()
    for ge in g.expressions:
        if isinstance(ge, E.Column):
            gnames.add(cm.get(ge.name, ge.name) if cm.get(ge.name, ge.name) in seg.cols
                       else ge.name)
        elif _extract_minute(ge) is not None:
            gnames.add('__m__')
        else:
            return None
    if malias and malias in gnames:
        gnames.discard(malias); gnames.add('__m__')
    want9 = set(plain) | ({'__m__'} if dtcol is not None else set())
    if gnames != want9:
        return None
    # identify sp (enc-8 str) and uid (dict col)
    sp = uid = None
    for cn in plain:
        c = seg.cols.get(cn)
        if c is None:
            return None
        if c.get('code_enc') == 8:
            sp = cn
        else:
            uid = cn
    if sp is None or uid is None:
        return None
    if dtcol is not None:
        dc = seg.cols.get(dtcol)
        if dc is None or dc.get('dt') is None:
            return None
    if int(seg.cols[uid]['V']) >= (1 << 25) or int(seg.cols[sp]['V']) >= (1 << 23):
        return None
    ox = tree.args.get('order'); lx = tree.args.get('limit')
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
    return {'uid': uid, 'sp': sp, 'dt': dtcol, 'aggs': aggs, 'lim': lim,
            'proj': proj}


def _minute_table(seg, dtcol):
    c = seg.cols[dtcol]
    V = int(c['V'])
    vals = np.asarray(seg._dict_ints_at(c, np.arange(V, dtype=np.int64)), np.int64)
    mx = int(np.abs(vals).max()) if vals.size else 0
    div = (60_000_000_000 if mx > int(1e17) else    # ns
           60_000_000 if mx > int(1e14) else        # us
           60_000 if mx > int(1e11) else 60)        # ms | s
    return ((vals // div) % 60).astype(np.int64)


def execute(seg, spec):
    global _HITS
    import wdb_wherescan as WS
    uid, sp, dtc, k = spec['uid'], spec['sp'], spec['dt'], spec['lim']
    N = int(seg.N)
    import wdb_kernels as WK
    uc = np.ascontiguousarray(seg._raw_codes(uid))
    ucnt = None
    try:
        import wdb_gbshelf
        sh = wdb_gbshelf.open_shelf(seg, uid)
        if sh is not None:
            ucnt = wdb_gbshelf.bulk(sh)          # the Q19 shelf pays Q18's rent
    except Exception:
        ucnt = None
    if ucnt is None:
        ucnt = np.bincount(uc, minlength=int(seg.cols[uid]['V']))
    ucnt = np.ascontiguousarray(ucnt, np.int64)
    pl = seg.e8_planes(sp)
    if pl is None:
        return None
    pos8 = np.ascontiguousarray(pl[0], np.int64)
    lits8 = np.ascontiguousarray(pl[1], np.int64)
    spc = np.ascontiguousarray(np.bincount(lits8, minlength=int(seg.cols[sp]['V'])), np.int64)
    e0 = WS._code_of(seg, sp, '')
    if e0 is None:
        return None
    e0 = int(e0)
    emptyc = N - int(pos8.size)
    if dtc is not None:
        ec = np.ascontiguousarray(seg._raw_codes(dtc))
        mt = np.ascontiguousarray(_minute_table(seg, dtc))
    else:
        ec = np.zeros(1, np.int64)           # never read: has_m gates the branch
        mt = np.zeros(1, np.int64)
    if dtc is None:
        # JACKSON'S HEAD-FIRST HUNT: '' owns ~87% of rows, so (user, '')
        # dominates the board -- and its counts are pure census arithmetic:
        # E(u) = total(u) - plane(u). Real-phrase pairs are bounded by the
        # user's PLANE rows, so only users with plane(u) > kth can compete;
        # usually that set is empty and no row is ever walked.
        V9u = int(seg.cols[uid]['V'])
        ucp = uc[pos8]                       # plane rows' users (13.2M)
        pcnt = np.bincount(ucp, minlength=V9u)
        # Jackson's cut: top users come off the gbc2 shelf's u16 tail --
        # outside the >=4 tier, total<=3 so E<=3 and nobody boards. The
        # M-window widens until kth >= the M-th total (exactness guard).
        tot9 = None
        try:
            import wdb_gbshelf
            sh0 = wdb_gbshelf.open_shelf(seg, uid)
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
            if ucnt is None:
                ucnt = np.bincount(uc, minlength=V9u)
            E = ucnt - pcnt
            kk9 = min(k, int((E > 0).sum()))
            topi = np.argpartition(-E, kk9 - 1)[:kk9] if kk9 else np.empty(0, np.int64)
            board = [(int(E[u]), int(u), int(e0)) for u in topi.tolist()]
            board.sort(reverse=True)
        kth9 = board[k - 1][0] if len(board) >= k else 0
        cand = np.flatnonzero(pcnt > kth9)
        if cand.size:
            cm = np.zeros(ucnt.size, bool)
            cm[cand] = True
            spm9 = spc > kth9                # the phrase-cut: a pair can't beat
            sel = np.flatnonzero(cm[ucp] & spm9[lits8])   # kth on a rarer phrase
            pk = (ucp[sel] << 23) | lits8[sel]
            pk.sort(kind='stable')
            b9 = np.empty(pk.size, bool)
            if pk.size:
                b9[0] = True
                np.not_equal(pk[1:], pk[:-1], out=b9[1:])
                s9 = np.flatnonzero(b9)
                c9a = np.diff(np.append(s9, pk.size))
                for j9 in range(s9.size):
                    board.append((int(c9a[j9]), int(pk[s9[j9]] >> 23),
                                  int(pk[s9[j9]] & ((1 << 23) - 1))))
            board.sort(reverse=True)
        rows = []
        for c9, u9, s9c in board[:k]:
            uv = seg.fetch(uid, u9)
            sv = seg.fetch(sp, s9c)
            if isinstance(uv, (bytes, bytearray)):
                uv = uv.decode('utf-8', 'replace')
            if isinstance(sv, (bytes, bytearray)):
                sv = sv.decode('utf-8', 'replace')
            row = []
            for a9 in spec['aggs']:
                row.append(uv if a9[0] == 'K' and a9[1] == uid else
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
        WK.tt_survivors(uc, ucnt, pos8, lits8, spc, ec, mt,
                        np.int64(e0), np.int64(emptyc), np.int64(theta),
                        outs, lens, dtc is not None)
        key = np.concatenate([outs[t, :int(lens[t])] for t in range(T9)])             if int(lens.sum()) else np.empty(0, np.int64)
        idx = key                                 # naming kept for flow below
        if key.size:
            key.sort(kind='stable')
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
                    rows = []
                    for gi in order.tolist():
                        kk = int(uq[gi]); c9 = int(cn[gi])
                        ucode = kk >> 29; mv = (kk >> 23) & 63; scode = kk & ((1 << 23) - 1)
                        # dtc None: mv is provably 0 and never emitted
                        uv = seg.fetch(uid, ucode)
                        sv = seg.fetch(sp, scode)
                        if isinstance(uv, (bytes, bytearray)):
                            uv = uv.decode('utf-8', 'replace')
                        if isinstance(sv, (bytes, bytearray)):
                            sv = sv.decode('utf-8', 'replace')
                        row = []
                        for a in spec['aggs']:
                            row.append(uv if a[0] == 'K' and a[1] == uid else
                                       sv if a[0] == 'K' else
                                       int(mv) if a[0] == 'M' else c9)
                        rows.append(tuple(row))
                    _HITS += 1
                    return rows, [wdb_sql._alias(p) for p in spec['proj']]
        if theta <= 1:
            return None                      # exhausted: yield to the general road
        theta //= 4
