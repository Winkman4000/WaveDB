"""wdb_pairfold -- Q30's one-walk pair board (the eight taxes become one).

SELECT a, b, COUNT(*) c, SUM(x), AVG(y) FROM hits
WHERE f <> '' GROUP BY a, b ORDER BY c DESC LIMIT n

The filter is free (the planes hand the typed positions, memoized). One
fused kernel walks those positions, reads all four columns at the row,
radix-buckets by b, then per-bucket sorts and folds count + both
dictionary-valued sums. Ten pairs decode at the pluck.
"""
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') \
            or tree.args.get('having') or tree.args.get('distinct'):
        return None
    w = tree.args.get('where')
    if w is None or not isinstance(w.this, E.NEQ):
        return None
    wl, wr = w.this.this, w.this.expression
    if not (isinstance(wl, E.Column) and isinstance(wr, E.Literal)
            and wr.is_string and str(wr.this) == ''):
        return None
    cm = col_map or {}
    fcol = cm.get(wl.name, wl.name)
    fc = seg.cols.get(fcol)
    if fc is None or fc.get('code_enc') not in (8, 9) or fc.get('has_null'):
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 2 \
            or not all(isinstance(x, E.Column) for x in g.expressions):
        return None
    acol = cm.get(g.expressions[0].name, g.expressions[0].name)
    bcol = cm.get(g.expressions[1].name, g.expressions[1].name)
    ac, bc = seg.cols.get(acol), seg.cols.get(bcol)
    if ac is None or bc is None or ac.get('has_null') or bc.get('has_null'):
        return None
    if int(ac.get('V') or 1 << 30) > 256:
        acol, bcol = bcol, acol                  # small key rides the low byte
        ac, bc = bc, ac
    if int(ac.get('V') or 1 << 30) > 256 or int(bc.get('V') or 1 << 30) >= 1 << 40:
        return None
    proj = []                                    # ('A',)('B',)('C',)('S',col)('V',col)
    calias = None
    scols = []
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            cn = cm.get(inner.name, inner.name)
            if cn == acol:
                proj.append(('A',)); continue
            if cn == bcol:
                proj.append(('B',)); continue
            return None
        ak = wdb_sql._agg_kind(inner)
        if ak is None:
            return None
        if ak[0] == 'COUNT_STAR':
            proj.append(('C',))
            if isinstance(p, E.Alias):
                calias = p.alias
            continue
        if ak[0] in ('SUM', 'AVG'):
            xc = cm.get(ak[1], ak[1])
            c9 = seg.cols.get(xc)
            if c9 is None or c9.get('dt') != 0 or c9.get('has_null') \
                    or int(c9.get('V') or 1 << 30) > 65536:
                return None
            if xc not in scols:
                scols.append(xc)
            proj.append(('S' if ak[0] == 'SUM' else 'V', xc)); continue
        return None
    if len(scols) > 2 or not any(k[0] == 'C' for k in proj):
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
    elif not isinstance(io, E.Count) or isinstance(io.this, E.Distinct):
        return None
    try:
        lim = int(lx.expression.this)
    except Exception:
        return None
    import wdb_policies as P
    if not P.no_deleted_rows(seg):
        return None
    return {'f': fcol, 'a': acol, 'b': bcol, 'scols': scols, 'lim': lim,
            'projkinds': proj, 'proj': tree.expressions}


def _dv(seg, col):
    V = int(seg.cols[col]['V'])
    try:
        return np.asarray(seg._dict_ints_at(seg.cols[col],
                                            np.arange(V, dtype=np.int64)),
                          np.float64)
    except Exception:
        return np.asarray([float(seg.fetch(col, v9)) for v9 in range(V)],
                          np.float64)


def execute(seg, spec):
    global _HITS
    import wdb_kernels as _WK
    pl = seg.e8_planes(spec['f'])
    pos = np.ascontiguousarray(np.asarray(pl[0], dtype=np.int64))
    bc9 = np.asarray(seg._raw_codes(spec['b']))
    BV = int(seg.cols[spec['b']]['V'])
    memo = seg.__dict__.setdefault('_censusmemo', {})
    ckey = ('pf', spec['f'], spec['b'])
    cnt_b = memo.get(ckey)
    if cnt_b is None:                        # the bound: a pair can never
        cnt_b = np.bincount(bc9[pos], minlength=BV)   # outscore its b's total
        memo[ckey] = cnt_b
    k = spec['lim']
    ac9 = np.asarray(seg._raw_codes(spec['a']))
    scols = spec['scols']
    x1 = scols[0] if scols else spec['a']
    x2 = scols[1] if len(scols) > 1 else x1
    xc9 = np.asarray(seg._raw_codes(x1))
    yc9 = np.asarray(seg._raw_codes(x2))
    SH = max(0, BV.bit_length() - 12)
    try:
        import numba
        T9 = max(1, numba.get_num_threads())
    except Exception:
        T9 = 1
    M = 4096
    nz = int((cnt_b > 0).sum())
    while True:
        M9 = min(M, nz)
        if M9 >= nz:                         # no prune possible: whole board
            spos = pos
            max_excl = 0
        else:
            hkey = ('pfh', spec['f'], spec['b'], M9)
            hit = memo.get(hkey)                   # the heavy set is static
            if hit is None:                        # per (filter, column, M):
                thr_idx = np.argpartition(-cnt_b, M9 - 1)[:M9]   # select once,
                heavy = np.zeros(BV, np.bool_)                   # remember
                heavy[thr_idx] = True
                max_excl = int(cnt_b[~heavy].max()) if M9 < BV else 0
                spos9 = _WK.pf_prune(pos, bc9, heavy)
                memo[hkey] = (max_excl, spos9)
                hit = memo[hkey]
            max_excl, spos = hit
        key, pay, offs = _WK.pr_scatter(spos, ac9, bc9, xc9, yc9, SH, T9)
        n = key.size
        ucnt = np.zeros(n, np.int64)
        us1 = np.zeros(n, np.float64)
        us2 = np.zeros(n, np.float64)
        ukey = np.zeros(n, np.int64)
        nruns = np.zeros(1 << 12, np.int64)
        _WK.pr_fold(key, pay, offs, _dv(seg, x1), _dv(seg, x2),
                    ucnt, us1, us2, ukey, nruns)
        live = np.zeros(n, bool)
        for b in range(nruns.size):
            if nruns[b]:
                live[offs[b]:offs[b] + nruns[b]] = True
        ucnt = ucnt[live]; us1 = us1[live]; us2 = us2[live]; ukey = ukey[live]
        k9 = min(k, ucnt.size)
        if k9 == 0 and max_excl > 0:
            M *= 4; continue
        order = np.argpartition(-ucnt, max(0, k9 - 1))[:k9]
        order = order[np.argsort(-ucnt[order], kind='stable')]
        p10 = int(ucnt[order[-1]]) if k9 else 0
        if max_excl <= p10 or M9 >= nz:      # THE CERTIFICATE: every excluded
            break                            # b PROVABLY hosts no better pair
        M *= 4                               # widen once, re-check
    n = key.size
    ucnt = np.zeros(n, np.int64)
    us1 = np.zeros(n, np.float64)
    us2 = np.zeros(n, np.float64)
    ukey = np.zeros(n, np.int64)
    nruns = np.zeros(1 << 12, np.int64)
    _WK.pr_fold(key, pay, offs, _dv(seg, x1), _dv(seg, x2),
                ucnt, us1, us2, ukey, nruns)
    live = np.zeros(n, bool)
    for b in range(nruns.size):
        if nruns[b]:
            live[offs[b]:offs[b] + nruns[b]] = True
    ucnt = ucnt[live]; us1 = us1[live]; us2 = us2[live]; ukey = ukey[live]
    sums = {x1: us1, x2: us2}
    out = []
    for j in order.tolist():
        kv = int(ukey[j])
        bcode, acode = kv >> 8, kv & 0xFF
        va = seg.fetch(spec['a'], acode)         # THE pluck
        vb = seg.fetch(spec['b'], bcode)
        if isinstance(va, (bytes, bytearray)):
            va = va.decode('utf-8', 'replace')
        if isinstance(vb, (bytes, bytearray)):
            vb = vb.decode('utf-8', 'replace')
        row = []
        for kind in spec['projkinds']:
            if kind[0] == 'A':
                row.append(va)
            elif kind[0] == 'B':
                row.append(vb)
            elif kind[0] == 'C':
                row.append(int(ucnt[j]))
            elif kind[0] == 'S':
                row.append(int(round(float(sums[kind[1]][j]))))
            else:
                row.append(float(sums[kind[1]][j]) / max(1, int(ucnt[j])))
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
