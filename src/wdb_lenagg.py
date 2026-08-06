"""wdb_lenagg -- grouped string-measure aggregates from the charlens table.

Q27's lane: AVG/SUM(LENGTH(c)) [+ COUNT(*)] GROUP BY key [WHERE c <> ''] [HAVING
COUNT(*) > lit] [ORDER BY agg] [LIMIT n]  ==  two weighted bincounts over the code
streams. The lengths live in the V-sized charlens table; no string is ever born.
Fail-closed: any unrecognized limb -> None, the general scan serves."""
import numpy as np
from sqlglot import expressions as E
import wdb_sql

_HITS = 0


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') or tree.args.get('distinct'):
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 1 or not isinstance(g.expressions[0], E.Column):
        return None
    key = (col_map or {}).get(g.expressions[0].name, g.expressions[0].name)
    aggs = []
    lcol = None
    lkind = None
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            if (col_map or {}).get(inner.name, inner.name) != key:
                return None
            aggs.append(('KEY', None))
            continue
        kd = wdb_sql._agg_kind(inner)
        if kd is not None and kd[0] == 'COUNT_STAR':
            aggs.append(('CNT', None))
            continue
        if isinstance(inner, (E.Avg, E.Sum)):
            fn = inner.this
            kf = None
            cn2 = None
            if isinstance(fn, E.Length) and isinstance(fn.this, E.Column):
                kf, cn2 = 'chars', fn.this.name
            elif (isinstance(fn, E.Anonymous) and str(fn.this).upper() == 'STRLEN'
                  and fn.expressions and isinstance(fn.expressions[0], E.Column)):
                kf, cn2 = 'bytes', fn.expressions[0].name
            if kf is not None:
                c2 = (col_map or {}).get(cn2, cn2)
                if lcol is not None and (c2 != lcol or kf != lkind):
                    return None
                lcol, lkind = c2, kf
                aggs.append(('AVGL' if isinstance(inner, E.Avg) else 'SUML', None))
                continue
        return None
    if lcol is None:
        return None
    w = tree.args.get('where')
    excl_empty = False
    if w is not None:
        n2 = w.this
        if (isinstance(n2, E.NEQ) and isinstance(n2.this, E.Column)
                and (col_map or {}).get(n2.this.name, n2.this.name) == lcol
                and isinstance(n2.expression, E.Literal) and str(n2.expression.this) == ''):
            excl_empty = True
        else:
            return None
    hv = tree.args.get('having')
    hmin = None
    if hv is not None:
        h = hv.this
        if not isinstance(h, E.GT):
            return None
        hk = wdb_sql._agg_kind(h.this)
        if hk is None or hk[0] != 'COUNT_STAR' or not isinstance(h.expression, E.Literal):
            return None
        hmin = int(str(h.expression.this))
    for c in (key, lcol):
        if c not in seg.cols or seg.cols[c].get('code_enc') not in (0, 3, 5, 8):
            return None
    lens = (seg.dict_bytelens(lcol) if lkind == 'bytes' else seg.dict_charlens(lcol)) \
        if hasattr(seg, 'dict_charlens') else None
    if lens is None:
        return None
    lim = None
    lx = tree.args.get('limit')
    if lx is not None:
        try:
            lim = int(lx.expression.this)
        except Exception:
            return None
    oi = None
    ox = tree.args.get('order')
    if ox is not None:
        if len(ox.expressions) != 1 or not ox.expressions[0].args.get('desc'):
            return None
        onm = ox.expressions[0].this
        if not isinstance(onm, E.Column):
            return None
        alias_names = [ (p.alias if isinstance(p, E.Alias) else None) for p in tree.expressions ]
        if onm.name not in alias_names:
            return None
        oi = alias_names.index(onm.name)
    return {'key': key, 'lcol': lcol, 'lkind': lkind, 'aggs': aggs, 'excl_empty': excl_empty,
            'hmin': hmin, 'lim': lim, 'oi': oi, 'proj': tree.expressions}


def execute(seg, spec):
    global _HITS
    key, lcol = spec['key'], spec['lcol']
    lens = (seg.dict_bytelens(lcol) if spec.get('lkind') == 'bytes'
            else seg.dict_charlens(lcol))
    if lens is None:
        return None
    KV = int(seg.cols[key]['V'])
    ec = -1
    if spec['excl_empty']:
        import wdb_wherescan as WS
        ec0 = WS._code_of(seg, lcol, '')
        ec = int(ec0) if ec0 is not None else -1
    N = int(seg.N)
    if int(lens.max() if lens.size else 0) < 65536:
        lens16 = lens.astype(np.uint16)          # the tiny alphabet rides a u16 bus
    else:
        lens16 = lens.astype(np.int64)
    import wdb_kernels as WK
    BR = 524288
    nfr = (N + BR - 1) // BR
    from concurrent.futures import ThreadPoolExecutor
    def _pour(f):
        lo, hi = f * BR, min((f + 1) * BR, N)
        kcf = np.asarray(seg._raw_codes_range(key, lo, hi))
        ucf = np.asarray(seg._raw_codes_range(lcol, lo, hi))
        j = np.zeros(KV, np.int64)
        c = np.zeros(KV, np.int64)
        WK.lenagg_pour(kcf, ucf, lens16, j, c, np.int64(ec))
        return j, c
    sums = np.zeros(KV, np.float64)
    cnt = np.zeros(KV, np.float64)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for j, c in ex.map(_pour, range(nfr)):   # frame-fused: pop the aligned pair,
            sums += j                            # pour while cache-hot, discard
            cnt += c
    keep = cnt > (spec['hmin'] if spec['hmin'] is not None else 0)
    gs = np.flatnonzero(keep)
    rows = []
    vals = seg.values_at(key, gs) if gs.size else []
    for i2, g0 in enumerate(gs.tolist()):
        kv = vals[i2]
        if isinstance(kv, (bytes, bytearray)):
            kv = kv.decode('utf-8', 'replace')
        row = []
        for kind, _ in spec['aggs']:
            if kind == 'KEY':
                row.append(kv)
            elif kind == 'CNT':
                row.append(int(cnt[g0]))
            elif kind == 'SUML':
                row.append(float(sums[g0]))
            else:
                row.append(float(sums[g0] / cnt[g0]) if cnt[g0] else None)
        rows.append(tuple(row))
    if spec['oi'] is not None:
        rows.sort(key=lambda r: (r[spec['oi']] is None, r[spec['oi']]), reverse=True)
    if spec['lim'] is not None:
        rows = rows[:spec['lim']]
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in spec['proj']]
