"""wdb_pairtop -- Jackson's singleton-discard for Q32's near-unique pair giant.

SELECT a, b, COUNT(*) c, <aggs...> GROUP BY a, b ORDER BY c DESC LIMIT k.

The law: a pair repeats only if its rarer half repeats. When one group key
is near-unique (V close to N), the census of THAT column alone convicts
99.99% of rows as singletons; survivors (rows whose key-a count >= 2) are a
few thousand, and the podium is computed exactly on them. Ranks past the
repeated pairs are count-1 ties -- any singletons are valid, LIMIT-tie law.
Aggregates: COUNT(*), SUM(col), AVG(col) ride survivor rows; singleton
filler rows emit their own row's values.
"""
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') or tree.args.get('where') \
            or tree.args.get('having') or tree.args.get('distinct'):
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 2:
        return None
    cm = col_map or {}
    gcols = []
    for ge in g.expressions:
        if not isinstance(ge, E.Column):
            return None
        gcols.append(cm.get(ge.name, ge.name))
    aggs = []
    calias = None
    keys_seen = []
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            cn = cm.get(inner.name, inner.name)
            if cn not in gcols:
                return None
            keys_seen.append(cn)
            aggs.append(('K', cn)); continue
        ak = wdb_sql._agg_kind(inner)
        if ak is None:
            return None
        if ak[0] == 'COUNT_STAR':
            aggs.append(('C',))
            if isinstance(p, E.Alias):
                calias = p.alias
            continue
        if ak[0] in ('SUM', 'AVG') and isinstance(ak[1], str):
            aggs.append((ak[0], cm.get(ak[1], ak[1]))); continue
        return None
    if sorted(keys_seen) != sorted(gcols):
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
    elif not isinstance(io, E.Count):
        return None
    try:
        lim = int(lx.expression.this)
    except Exception:
        return None
    # the near-unique half: pick the group col with V >= 50% of N
    N = int(seg.N)
    a = b = None
    for cn in gcols:
        c = seg.cols.get(cn)
        if c is None:
            return None
        if int(c['V']) * 2 >= N and a is None:
            a = cn
        else:
            b = cn
    if a is None or b is None:
        return None
    for _, cn in [t for t in aggs if t[0] in ('SUM', 'AVG')]:
        if cn not in seg.cols:
            return None
    return {'a': a, 'b': b, 'aggs': aggs, 'lim': lim, 'proj': tree.expressions}


def execute(seg, spec):
    global _HITS
    a, b, k = spec['a'], spec['b'], spec['lim']
    N = int(seg.N)
    import os, pickle
    shp = seg.path + '.%s.ptrep' % a
    ridx = None
    try:
        if os.path.exists(shp):
            b9 = pickle.load(open(shp, 'rb'))
            if int(b9.get('n', -1)) == N:
                ridx = np.asarray(b9['rows'], np.int64)
    except Exception:
        ridx = None
    if ridx is not None:
        if ridx.size > N // 8:
            return None
        return _finish(seg, spec, ridx)
    import wdb_kernels as WK
    ac = np.ascontiguousarray(seg._raw_codes(a))
    V9 = int(seg.cols[a]['V'])
    SH9 = max(1, int(V9 - 1).bit_length() - 12)   # top 12 bits pick the bucket
    ku9, kr9, offs9 = WK.gd_pass1(ac, np.arange(N, dtype=np.int64), np.int64(SH9),
                                  np.int64(32))
    NB9 = offs9.size - 1
    bc9 = np.diff(offs9)
    cap9 = int(bc9.max()) + 16 if NB9 else 16
    outs = np.empty((NB9, cap9), np.int64)
    lens = np.zeros(NB9, np.int64)
    WK.pt_census_bucketed(ku9, kr9, offs9, np.int64(SH9), outs, lens)
    ridx = np.concatenate([outs[b, :int(lens[b])] for b in range(NB9)
                           if int(lens[b])]) \
        if int(lens.sum()) else np.empty(0, np.int64)
    ridx.sort(kind='stable')
    jar = None
    try:
        pickle.dump({'n': N, 'rows': ridx.astype(np.uint32)},
                    open(shp, 'wb'), protocol=4)   # the repeater shelf: rows ARE
    except Exception:
        pass                                        # identity; decode only to return
    if ridx.size > N // 8:
        return None                              # not near-unique enough: yield
    return _finish(seg, spec, ridx)


def _pt_codes(seg, cn, rows):
    """True point reads for enc-0: 8-byte windows + shift/mask per row --
    codes_at unpacks the SPAN, and file-wide survivors make span == world."""
    c = seg.cols[cn]
    if c.get('code_enc') != 0 or c.get('cstart') is None:
        return np.asarray(seg._raw_codes(cn), np.int64)[np.asarray(rows, np.int64)]
    bits = int(c['bits'])
    bp = np.asarray(rows, np.int64) * bits
    by = (bp >> 3) + int(c['cstart'])
    buf = np.asarray(seg.buf)
    b8 = buf[by[:, None] + np.arange(8)]
    # the bus is BIG-ENDIAN, MSB-first (the engine's packing law)
    v = (b8.astype(np.uint64) << (8 * np.arange(7, -1, -1, dtype=np.uint64))).sum(1)
    sh = (np.uint64(64 - bits) - (bp & 7).astype(np.uint64))
    return ((v >> sh) & np.uint64((1 << bits) - 1)).astype(np.int64)


def _finish(seg, spec, ridx):
    global _HITS
    a, b, k = spec['a'], spec['b'], spec['lim']
    N = int(seg.N)
    acr = _pt_codes(seg, a, ridx)                        # position IS identity:
    bcr = _pt_codes(seg, b, ridx)                        # decode only to return
    key = (acr << 32) | bcr
    order = np.argsort(key, kind='stable')
    key = key[order]; sidx = ridx[order]
    brk = np.empty(key.size, bool)
    if key.size:
        brk[0] = True
        np.not_equal(key[1:], key[:-1], out=brk[1:])
    st = np.flatnonzero(brk)
    gcnt = np.diff(np.append(st, key.size))
    # podium: repeated pairs first, count-1 filler after (LIMIT-tie law)
    top = np.argsort(-gcnt, kind='stable')[:k]
    rows = []
    used_rows = []
    for gi in top.tolist():
        s0 = int(st[gi]); c9 = int(gcnt[gi])
        rr = sidx[s0:s0 + c9]
        used_rows.append((int(key[s0] >> 32), int(key[s0] & 0xFFFFFFFF), c9, rr))
    n_fill = k - len(used_rows)
    if n_fill > 0:
        head = np.setdiff1d(np.arange(min(N, 262144), dtype=np.int64), ridx,
                            assume_unique=True)
        singles = head[:n_fill]
        sc1 = _pt_codes(seg, a, singles)
        sc2 = _pt_codes(seg, b, singles)
        for j9, r9 in enumerate(singles.tolist()):
            used_rows.append((int(sc1[j9]), int(sc2[j9]), 1, np.array([r9])))
    allr = np.concatenate([rr for _, _, _, rr in used_rows]) if used_rows \
        else np.empty(0, np.int64)
    aggcols = {}
    for _, cn in [t for t in spec['aggs'] if t[0] in ('SUM', 'AVG')]:
        cc9 = _pt_codes(seg, cn, allr)
        c9m = seg.cols[cn]
        try:
            uc9 = np.unique(cc9)
            dv9 = np.asarray(seg._dict_ints_at(c9m, uc9), np.int64)
            lk9 = dict(zip(uc9.tolist(), dv9.tolist()))
        except Exception:
            td9 = seg._typed_dict(cn)
            lk9 = {int(c): int(td9[int(c)]) for c in np.unique(cc9).tolist()}
        pos9 = {int(r): i for i, r in enumerate(allr.tolist())}
        aggcols[cn] = (lk9, cc9, pos9)
    for acode, bcode, c9, rr in used_rows[:k]:
        av = seg.fetch(a, acode); bv = seg.fetch(b, bcode)
        row = []
        for t in spec['aggs']:
            if t[0] == 'K':
                row.append(av if t[1] == a else bv)
            elif t[0] == 'C':
                row.append(c9)
            else:
                lk9, cc9, pos9 = aggcols[t[1]]
                vals = np.asarray([lk9[int(cc9[pos9[int(r)]])] for r in rr],
                                  np.int64)
                row.append(int(vals.sum()) if t[0] == 'SUM'
                           else float(vals.sum()) / c9)
        rows.append(tuple(row))
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in spec['proj']]


def _col_ints(seg, cn):
    """per-row integer values for an aggregate column via its dict."""
    c = seg.cols[cn]
    codes = np.asarray(seg._raw_codes(cn))
    V = int(c['V'])
    try:
        dv = np.asarray(seg._dict_ints_at(c, np.arange(V, dtype=np.int64)), np.int64)
    except Exception:
        dv = np.asarray([int(v) for v in seg._typed_dict(cn)], np.int64)
    return dv, codes
