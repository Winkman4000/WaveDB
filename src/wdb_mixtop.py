"""wdb_mixtop -- Q9's shelf-fed mixed board (Jackson's design).

SELECT k, SUM(a), COUNT(*) c, AVG(b), COUNT(DISTINCT t) FROM hits
GROUP BY k ORDER BY c DESC LIMIT n

Nothing decodes until the pluck: region codes bincount for c (top-n falls
out), SUM/AVG ride 2D code-space bincounts against V-sized dictionaries,
and COUNT(DISTINCT t) is LOOKED UP off the gdc shelf the realm already
birthed -- the 2-second walk dies without successor. Only the n winning
key values are ever fetched.
"""
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') \
            or tree.args.get('having') or tree.args.get('distinct') \
            or tree.args.get('where'):
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 1 or not isinstance(g.expressions[0], E.Column):
        return None
    cm = col_map or {}
    kcol = cm.get(g.expressions[0].name, g.expressions[0].name)
    kc = seg.cols.get(kcol)
    if kc is None or int(kc.get('V') or 1 << 30) > 200000 or kc.get('has_null'):
        return None
    aggs = []                                    # ('K',) ('C',) ('S',col) ('A',col) ('D',t)
    calias = None
    tcol = None
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            if cm.get(inner.name, inner.name) != kcol:
                return None
            aggs.append(('K',)); continue
        if isinstance(inner, E.Count) and isinstance(inner.this, E.Distinct):
            di = inner.this
            if len(di.expressions) != 1 or not isinstance(di.expressions[0], E.Column):
                return None
            tcol = cm.get(di.expressions[0].name, di.expressions[0].name)
            aggs.append(('D',)); continue
        ak = wdb_sql._agg_kind(inner)
        if ak is None:
            return None
        if ak[0] == 'COUNT_STAR':
            aggs.append(('C',))
            if isinstance(p, E.Alias):
                calias = p.alias
            continue
        if ak[0] in ('SUM', 'AVG'):
            acol = cm.get(ak[1], ak[1])
            c9 = seg.cols.get(acol)
            if c9 is None or c9.get('dt') != 0 or c9.get('has_null') \
                    or int(c9.get('V') or 1 << 30) > 65536:
                return None
            aggs.append(('S' if ak[0] == 'SUM' else 'A', acol)); continue
        return None
    if tcol is None or seg.cols.get(tcol) is None:
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
    import wdb_groupdistinct as GD
    if GD._gdc_load(seg, kcol, tcol) is None:
        return None                              # the shelf feeds the board
    return {'k': kcol, 't': tcol, 'lim': lim, 'aggs': aggs, 'proj': tree.expressions}


def execute(seg, spec):
    global _HITS
    kcol, tcol, k = spec['k'], spec['t'], spec['lim']
    kc9 = np.asarray(seg._raw_codes(kcol))
    KV = int(seg.cols[kcol]['V'])
    scols = []
    for kind, *rest in spec['aggs']:
        if kind in ('S', 'A') and rest[0] not in scols:
            scols.append(rest[0])
    def _dv(acol):
        AV = int(seg.cols[acol]['V'])
        try:
            return np.asarray(seg._dict_ints_at(seg.cols[acol],
                                                np.arange(AV, dtype=np.int64)),
                              np.float64)
        except Exception:
            return np.asarray([float(seg.fetch(acol, v9)) for v9 in range(AV)],
                              np.float64)
    per = {}
    import wdb_kernels as _WK
    if len(scols) == 2:
        a1, a2 = scols
        cnt, f1, f2 = _WK.mx_fold2(kc9,
                                   np.asarray(seg._raw_codes(a1)), _dv(a1),
                                   np.asarray(seg._raw_codes(a2)), _dv(a2), KV)
        per[a1], per[a2] = f1, f2
    else:
        cnt = np.bincount(kc9, minlength=KV)
        for acol in scols:
            per[acol] = _WK.mx_fold(kc9, np.asarray(seg._raw_codes(acol)),
                                    _dv(acol), KV)
    k9 = min(k, KV - 1)
    order = np.argpartition(-cnt, k9)[:k]
    order = order[np.argsort(-cnt[order], kind='stable')]
    import wdb_groupdistinct as GD
    blob = GD._gdc_load(seg, kcol, tcol)
    dc_by_code = blob.get('counts') if blob else None
    keys9 = blob.get('keys') if blob else None
    kmap = None
    if keys9 is not None:
        kmap = {kk: int(cc) for kk, cc in zip(keys9, np.asarray(dc_by_code).tolist())}
    out = []
    for j in range(order.size):
        code = int(order[j])
        v9 = seg.fetch(kcol, code)               # THE pluck: n values, no more
        if isinstance(v9, (bytes, bytearray)):
            v9 = v9.decode('utf-8', 'replace')
        row = []
        for kind, *rest in spec['aggs']:
            if kind == 'K':
                row.append(v9)
            elif kind == 'C':
                row.append(int(cnt[code]))
            elif kind == 'S':
                row.append(int(round(float(per[rest[0]][code]))))
            elif kind == 'A':
                row.append(float(per[rest[0]][code]) / max(1, int(cnt[code])))
            elif kind == 'D':
                if kmap is not None:
                    row.append(kmap.get(v9, 0))
                else:
                    row.append(int(np.asarray(dc_by_code)[code]))
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
