"""wdb_funnel -- the selective funnel (Jackson's law: own the best "no").

SELECT g..., COUNT(*) c FROM hits
WHERE sel = lit [AND stair-range] [AND flag (=|<>) lit ...]
GROUP BY g... ORDER BY c DESC LIMIT k [OFFSET o]

Start FROM the selector: the plist shelf (one sort of all positions by
selector code, mmap sidecar, ledger species) hands the counter's rows
pre-sorted in row order -- so a staircase window is one searchsorted
pair, hygiene flags are point reads at the crumb, and the group runs
in code space. Only the emitted rows decode.
"""
import os
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0
_VHITS = 0                                       # fused consumption fires


def _days_of(v):
    """Any date-ish value -> days since 1970-01-01; ints pass through."""
    if hasattr(v, 'toordinal'):
        return v.toordinal() - 719163
    tn = type(v).__name__
    if tn == 'datetime64':
        return int(np.datetime64(v, 'D').astype('int64'))
    return int(v)


def _plist_path(seg, col):
    return seg.path + '.%s.plist' % col


def _plist(seg, col):
    """Position lists per code for a selector column; birth-on-touch, ledgered."""
    memo = seg.__dict__.setdefault('_plistmemo', {})
    hit = memo.get(col)
    if hit is not None:
        return hit
    p = _plist_path(seg, col)
    V = int(seg.cols[col]['V'])
    if not os.path.exists(p):
        cc = np.asarray(seg._raw_codes(col)).astype(np.int64)
        order = np.argsort(cc, kind='stable')    # row order preserved per code
        cnts = np.bincount(cc, minlength=V)
        offs = np.zeros(V + 1, np.int64)
        np.cumsum(cnts, out=offs[1:])
        tmp = p + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(np.array([V], np.int64).tobytes())
            f.write(offs.tobytes())
            f.write(order.astype(np.uint32).tobytes())
        os.replace(tmp, p)
        import wdb_shelves
        wdb_shelves.record(seg, 'plist', col=col)
    mm = np.memmap(p, dtype=np.uint8, mode='r')
    V9 = int(np.frombuffer(mm[:8], np.int64)[0])
    offs = np.frombuffer(mm[8:8 + 8 * (V9 + 1)], np.int64)
    pos = np.frombuffer(mm[8 + 8 * (V9 + 1):], np.uint32)
    memo[col] = (offs, pos)
    return memo[col]


def _lit(node):
    if isinstance(node, E.Literal) and not node.is_string:
        try:
            return int(str(node.this))
        except Exception:
            return None
    if isinstance(node, E.Literal) and node.is_string:
        return str(node.this)
    if isinstance(node, E.Neg):
        v = _lit(node.this)
        return -v if isinstance(v, int) else None
    return None


def _flatten_and(node, out):
    if isinstance(node, E.And):
        _flatten_and(node.this, out)
        _flatten_and(node.expression, out)
    else:
        out.append(node)


def _compile_key(seg, expr, cm):
    """THE KEY COMPILER: an expression over dict-coded columns is itself a
    code-space citizen. Compiles a group-key expression to a plan that
    evaluates to codes on the crumb and decodes only emitted winners.
    v1 atoms: Column | CASE WHEN <conj of col (=|<>) intlit> THEN
    <Column|strlit> ELSE <Column|strlit> END."""
    if isinstance(expr, E.Column):
        cn = cm.get(expr.name, expr.name)
        c9 = seg.cols.get(cn)
        if c9 is None or c9.get('has_null'):
            return None
        return {'kind': 'col', 'col': cn, 'W': int(c9['V'])}
    if isinstance(expr, E.Case):
        ifs = expr.args.get('ifs') or []
        els = expr.args.get('default')
        if len(ifs) != 1:
            return None
        cond = ifs[0].this
        then = ifs[0].args.get('true')
        conds = []
        stack = [cond]
        while stack:
            nd = stack.pop()
            if isinstance(nd, E.Paren):
                stack.append(nd.this); continue
            if isinstance(nd, E.And):
                stack.append(nd.this); stack.append(nd.expression); continue
            if not isinstance(nd, (E.EQ, E.NEQ)):
                return None
            l9, r9 = nd.this, nd.expression
            if not isinstance(l9, E.Column):
                return None
            v9 = _lit(r9)
            if not isinstance(v9, int):
                return None
            cn9 = cm.get(l9.name, l9.name)
            if seg.cols.get(cn9) is None or seg.cols[cn9].get('has_null'):
                return None
            conds.append((cn9, v9, isinstance(nd, E.EQ)))
        def _branch(b9):
            if isinstance(b9, E.Column):
                cn9 = cm.get(b9.name, b9.name)
                c99 = seg.cols.get(cn9)
                if c99 is None or c99.get('has_null'):
                    return None
                return ('col', cn9)
            if isinstance(b9, E.Literal) and b9.is_string:
                return ('lit', str(b9.this))
            return None
        tb = _branch(then)
        eb = _branch(els) if els is not None else ('lit', None)
        if tb is None or eb is None or not conds:
            return None
        wcol = tb[1] if tb[0] == 'col' else (eb[1] if eb[0] == 'col' else None)
        if wcol is None:
            return None
        return {'kind': 'case', 'conds': conds, 'then': tb, 'els': eb,
                'col': wcol, 'W': int(seg.cols[wcol]['V']) + 2}
    return None


def _key_eval(seg, plan, crumb):
    """Codes for a compiled key at the crumb -- pure numpy, no values."""
    if plan['kind'] == 'col':
        return np.asarray(seg.codes_at(plan['col'], crumb)).astype(np.int64)
    mask = np.ones(crumb.size, bool)
    for cn9, v9, eq9 in plan['conds']:
        c9 = _code_of(seg, cn9, v9)
        fc = np.asarray(seg.codes_at(cn9, crumb))
        if c9 is None:
            m9 = np.zeros(crumb.size, bool) if eq9 else np.ones(crumb.size, bool)
        else:
            m9 = (fc == c9) if eq9 else (fc != c9)
        mask &= m9
    V9 = int(seg.cols[plan['col']]['V'])
    def _side(b9, tok):
        if b9[0] == 'col':
            return np.asarray(seg.codes_at(b9[1], crumb)).astype(np.int64)
        return np.full(crumb.size, V9 + tok, np.int64)
    return np.where(mask, _side(plan['then'], 0), _side(plan['els'], 1))


def _key_decode(seg, plan, code):
    """Value for one emitted key code -- the only place values exist."""
    if plan['kind'] == 'col':
        v9 = seg.fetch(plan['col'], int(code))
    else:
        V9 = int(seg.cols[plan['col']]['V'])
        if code == V9:
            v9 = plan['then'][1] if plan['then'][0] == 'lit' else None
        elif code == V9 + 1:
            v9 = plan['els'][1] if plan['els'][0] == 'lit' else None
        else:
            v9 = seg.fetch(plan['col'], int(code))
    if isinstance(v9, (bytes, bytearray)):
        v9 = v9.decode('utf-8', 'replace')
    return v9


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') \
            or tree.args.get('having') or tree.args.get('distinct'):
        return None
    w = tree.args.get('where')
    g = tree.args.get('group')
    if w is None or g is None or not g.expressions:
        return None
    cm = col_map or {}
    preds = []
    _flatten_and(w.this, preds)
    sel = None                                   # (col, value) -- the plist start
    rng = {}                                     # staircase ranges: col -> [lo_v, hi_v]
    flags = []                                   # (col, value, keep_eq)
    strneq = []                                  # (col,) p <> '' via planes
    for p9 in preds:
        if isinstance(p9, (E.EQ, E.NEQ)):
            l9, r9 = p9.this, p9.expression
            if not isinstance(l9, E.Column):
                return None
            cn = cm.get(l9.name, l9.name)
            v9 = _lit(r9)
            if v9 is None and not (isinstance(r9, E.Literal) and r9.is_string):
                return None
            if isinstance(p9, E.NEQ) and v9 == '':
                if seg.cols.get(cn) is None:
                    return None
                strneq.append(cn); continue
            if not isinstance(v9, int):
                return None
            c9 = seg.cols.get(cn)
            if c9 is None or c9.get('dt') != 0 or c9.get('has_null'):
                return None
            if isinstance(p9, E.EQ) and sel is None \
                    and int(c9.get('V') or 0) >= 64:
                sel = (cn, v9)                   # first wide-eq is the selector
            else:
                flags.append((cn, v9, isinstance(p9, E.EQ)))
        elif isinstance(p9, (E.GTE, E.LTE, E.GT, E.LT)):
            l9, r9 = p9.this, p9.expression
            if not isinstance(l9, E.Column) or not isinstance(r9, E.Literal):
                return None
            cn = cm.get(l9.name, l9.name)
            if seg.stairs(cn) is None:
                return None                      # ranges only on the staircase
            rng.setdefault(cn, [None, None])
            v9 = str(r9.this)
            if isinstance(p9, (E.GTE, E.GT)):
                rng[cn][0] = (v9, isinstance(p9, E.GT))
            else:
                rng[cn][1] = (v9, isinstance(p9, E.LT))
        elif isinstance(p9, E.In):
            l9 = p9.this
            if not isinstance(l9, E.Column):
                return None
            vs = [_lit(x) for x in p9.expressions]
            if any(not isinstance(v, int) for v in vs):
                return None
            flags.append((cm.get(l9.name, l9.name), tuple(vs), 'in'))
        else:
            return None
    if sel is None:
        return None
    if int(seg.cols[sel[0]].get('V') or 1 << 40) > 1 << 22:
        return None                              # plist stays a bounded species
    trunc = None
    if len(g.expressions) == 1:
        ge0 = g.expressions[0]
        tn = ge0.this if isinstance(ge0, E.Alias) else ge0
        if isinstance(tn, (E.DateTrunc, E.TimestampTrunc)) or (
                isinstance(tn, E.Anonymous)
                and str(tn.this).upper() == 'DATE_TRUNC'):
            argsT = list(tn.args.get('expressions') or [])
            unitT = tn.args.get('unit')
            colT = tn.this if isinstance(tn.this, E.Column) else None
            for aT in argsT:
                if isinstance(aT, E.Literal):
                    unitT = aT
                if isinstance(aT, E.Column):
                    colT = aT
            if unitT is not None and str(getattr(unitT, 'this', unitT)).lower() == 'minute' \
                    and colT is not None:
                cnT = cm.get(colT.name, colT.name)
                if seg.stairs(cnT) is not None:
                    trunc = (cnT, 60)
    if trunc is not None:
        ox = tree.args.get('order'); lx = tree.args.get('limit')
        if ox is None or lx is None or len(ox.expressions) != 1 \
                or ox.expressions[0].args.get('desc'):
            return None                      # the file order IS minute order
        proj9 = []
        for p9 in tree.expressions:
            inner = p9.this if isinstance(p9, E.Alias) else p9
            if isinstance(inner, (E.DateTrunc, E.TimestampTrunc, E.Anonymous)):
                proj9.append(('M',))
            else:
                ak = wdb_sql._agg_kind(inner)
                if ak is None or ak[0] != 'COUNT_STAR':
                    return None
                proj9.append(('C',))
        try:
            lim = int(lx.expression.this)
        except Exception:
            return None
        off = 0
        offx = tree.args.get('offset')
        if offx is not None:
            try:
                off = int(offx.expression.this)
            except Exception:
                return None
        import wdb_policies as P
        if not P.no_deleted_rows(seg):
            return None
        return {'sel': sel, 'rng': rng, 'flags': flags, 'strneq': strneq,
                'g': [], 'trunc': trunc, 'k': lim, 'off': off,
                'projkinds': proj9, 'proj': tree.expressions}
    aliasmap = {}
    for p9 in tree.expressions:
        if isinstance(p9, E.Alias):
            aliasmap[p9.alias] = p9.this
    plans = []
    plankeys = []
    for ge in g.expressions:
        src = ge
        if isinstance(ge, E.Literal) and not ge.is_string:
            idx = int(str(ge.this)) - 1
            if idx < 0 or idx >= len(tree.expressions):
                return None
            item = tree.expressions[idx]
            src = item.this if isinstance(item, E.Alias) else item
        if isinstance(src, E.Column) and src.name in aliasmap:
            src = aliasmap[src.name]             # GROUP BY alias -> its expr
        if isinstance(src, E.Literal):
            continue                             # constant: grouping no-op
        k9 = src.sql()
        if k9 in plankeys:
            continue
        pl9 = _compile_key(seg, src, cm)
        if pl9 is None:
            return None
        plankeys.append(k9)
        plans.append(pl9)
    if not plans or len(plans) > 6:
        return None
    gcols = [pl['col'] for pl in plans]
    proj = []
    calias = None
    for p9 in tree.expressions:
        inner = p9.this if isinstance(p9, E.Alias) else p9
        k9 = inner.sql()
        if k9 in plankeys:
            proj.append(('G', plankeys.index(k9))); continue
        ak = wdb_sql._agg_kind(inner)
        if ak is None or ak[0] != 'COUNT_STAR':
            return None
        proj.append(('C',))
        if isinstance(p9, E.Alias):
            calias = p9.alias
    if not any(x[0] == 'C' for x in proj):
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
    off = 0
    offx = tree.args.get('offset')
    if offx is not None:
        try:
            off = int(offx.expression.this)
        except Exception:
            return None
    import wdb_policies as P
    if not P.no_deleted_rows(seg):
        return None
    return {'sel': sel, 'rng': rng, 'flags': flags, 'strneq': strneq,
            'g': gcols, 'plans': plans, 'k': lim, 'off': off, 'projkinds': proj,
            'proj': tree.expressions}


def _code_of(seg, col, value):
    c = seg.cols[col]
    V = int(c['V'])
    lo, hi = 0, V - 1                            # sorted dict: binary search
    while lo <= hi:
        mid = (lo + hi) // 2
        v = seg.fetch(col, mid)
        try:
            v = _days_of(v)
        except (TypeError, ValueError):
            pass                             # string dicts compare raw
        if v == value:
            return mid
        if v < value:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


def execute(seg, spec):
    global _HITS
    if spec.get('trunc'):
        return _execute_trunc(seg, spec)
    import time as _tm
    import wdb_ledger as _LG
    _t9 = _tm.perf_counter()
    scol, sval = spec['sel']
    code = _code_of(seg, scol, sval)
    if code is None:
        crumb = np.empty(0, np.int64)
    else:
        offs, plist = _plist(seg, scol)
        crumb = plist[int(offs[code]):int(offs[code + 1])].astype(np.int64)
    # staircase windows: the crumb is row-ordered, so a range is a slice
    for rcol, (lo9, hi9) in spec['rng'].items():
        steps = np.asarray(seg.stairs(rcol), dtype=np.int64)
        full = np.concatenate([[0], steps, [seg.N]])
        import datetime as _dt
        def _tov(sv):
            try:
                return int(sv)
            except ValueError:
                y, m, d = [int(x) for x in sv.split('-')]
                return (_dt.date(y, m, d) - _dt.date(1970, 1, 1)).days
        V9 = int(seg.cols[rcol]['V'])
        vals = [seg.fetch(rcol, c9) for c9 in range(V9)]
        vals = [_days_of(v) for v in vals]
        rlo, rhi = 0, seg.N
        if lo9 is not None:
            import bisect
            c9 = bisect.bisect_left(vals, _tov(lo9[0]) + (1 if lo9[1] else 0))
            rlo = int(full[c9])
        if hi9 is not None:
            import bisect
            c9 = bisect.bisect_right(vals, _tov(hi9[0]) - (1 if hi9[1] else 0))
            rhi = int(full[c9])
        a = np.searchsorted(crumb, rlo, side='left')
        b = np.searchsorted(crumb, rhi, side='left')
        crumb = crumb[a:b]
    _LG.stage('crumb', (_tm.perf_counter() - _t9) * 1000); _t9 = _tm.perf_counter()
    # hygiene at the crumb: point reads, never the column. When the first
    # two flags are enc-10 scalar tests, ONE fused walk serves both with
    # per-row short-circuit (Jackson's crossing scheme, row granularity).
    # THE MASKED VERTICAL CONSUMPTION PATH: single enc-12 key, <=2 enc-10
    # scalar flags, optional key<>'' -- hygiene, extraction, and bincount
    # fuse into ONE kernel launch guided by the crumb bitmap. No stage
    # boundary ever materializes.
    fusedV = None
    plans0 = spec['plans']
    if crumb.size >= (1 << 15) and len(plans0) == 1 and plans0[0]['kind'] == 'col':
        kcol = plans0[0]['col']
        kc0 = seg.cols.get(kcol, {})
        VK = int(kc0.get('V', 0))
        fl0 = list(spec['flags'])
        okf = (len(fl0) <= 2 and all(f[2] != 'in' for f in fl0)
               and all(seg.cols.get(f[0], {}).get('code_enc') == 10 for f in fl0))
        oks = all(pc == kcol for pc in spec['strneq'])
        if kc0.get('code_enc') == 12 and okf and oks and 0 < VK <= (1 << 25):
            codesF = [_code_of(seg, f[0], f[1]) for f in fl0]
            if all(c9 is not None for c9 in codesF) or not fl0:
                import wdb_kernels as _WK
                bufF = np.frombuffer(seg.buf, np.uint8)
                nwK = int(kc0['nwords'])
                maskF = np.zeros(nwK, np.uint64)
                _WK.vbits_set(np.ascontiguousarray(crumb), maskF)
                skip9 = np.int64(-1)
                if spec['strneq']:
                    z9 = seg.fetch(kcol, 0)
                    if isinstance(z9, (bytes, bytearray)):
                        z9 = z9.decode('utf-8', 'replace')
                    if z9 == '':
                        skip9 = np.int64(0)
                def _fd(f9):
                    ca = seg.cols[f9[0]]
                    return (np.ascontiguousarray(np.frombuffer(seg.buf, np.int64,
                            int(ca['pXnblk']), ca['pXdir'])),
                            int(ca['pXpay']), int(ca['pXbits']))
                if fl0:
                    da, pa, ba = _fd(fl0[0])
                else:
                    da, pa, ba = np.zeros(1, np.int64), 0, 1
                if len(fl0) >= 2:
                    db, pb, bb = _fd(fl0[1])
                else:
                    db, pb, bb = np.zeros(1, np.int64), 0, 1
                CHW = 1 << 14
                nchw = (nwK + CHW - 1) // CHW
                cw9 = np.zeros(nchw, np.int64)
                _WK.vfused_count(maskF, seg.vplanes(kcol), nwK, int(kc0['bits']), bufF,
                                 da, pa, ba, int(codesF[0]) if len(fl0) >= 1 else 0,
                                 bool(fl0[0][2]) if len(fl0) >= 1 else True,
                                 db, pb, bb, int(codesF[1]) if len(fl0) >= 2 else 0,
                                 bool(fl0[1][2]) if len(fl0) >= 2 else True,
                                 len(fl0), skip9, cw9, CHW)
                offs9 = np.zeros(nchw + 1, np.int64)
                np.cumsum(cw9, out=offs9[1:])
                codes9 = np.zeros(max(1, int(offs9[-1])), np.int32)
                _WK.vfused_fill(maskF, seg.vplanes(kcol), nwK, int(kc0['bits']), bufF,
                                da, pa, ba, int(codesF[0]) if len(fl0) >= 1 else 0,
                                bool(fl0[0][2]) if len(fl0) >= 1 else True,
                                db, pb, bb, int(codesF[1]) if len(fl0) >= 2 else 0,
                                bool(fl0[1][2]) if len(fl0) >= 2 else True,
                                len(fl0), skip9, offs9, codes9, CHW)
                cntF = np.bincount(codes9[:int(offs9[-1])], minlength=VK)
                uk9 = np.flatnonzero(cntF)
                global _VHITS
                _VHITS += 1
                fusedV = (uk9.astype(np.int64), cntF[uk9].astype(np.int64))
    flags9 = list(spec['flags']) if fusedV is None else []
    if crumb.size and len(flags9) >= 2:
        f1, f2 = flags9[0], flags9[1]
        c1a = seg.cols.get(f1[0], {})
        c2a = seg.cols.get(f2[0], {})
        if f1[2] != 'in' and f2[2] != 'in' \
                and c1a.get('code_enc') == 10 and c2a.get('code_enc') == 10:
            k1 = _code_of(seg, f1[0], f1[1])
            k2 = _code_of(seg, f2[0], f2[1])
            if k1 is not None and k2 is not None:
                import wdb_kernels as _WK
                bufH = np.frombuffer(seg.buf, np.uint8)
                d1 = np.ascontiguousarray(np.frombuffer(seg.buf, np.int64,
                     int(c1a['pXnblk']), c1a['pXdir']))
                d2 = np.ascontiguousarray(np.frombuffer(seg.buf, np.int64,
                     int(c2a['pXnblk']), c2a['pXdir']))
                keep = np.zeros(crumb.size, np.bool_)
                _WK.bp10_hygiene2(bufH, d1, int(c1a['pXpay']), int(c1a['pXbits']),
                                  int(k1), bool(f1[2]), d2, int(c2a['pXpay']),
                                  int(c2a['pXbits']), int(k2), bool(f2[2]),
                                  np.ascontiguousarray(crumb), keep)
                crumb = crumb[keep]
                flags9 = flags9[2:]
    for fcol, fval, kind in flags9:
        if crumb.size == 0:
            break
        fc = np.asarray(seg.codes_at(fcol, crumb))
        if kind == 'in':
            want = set()
            for v9 in fval:
                c9 = _code_of(seg, fcol, v9)
                if c9 is not None:
                    want.add(c9)
            m9 = np.isin(fc, list(want)) if want else np.zeros(crumb.size, bool)
        else:
            c9 = _code_of(seg, fcol, fval)
            if c9 is None:
                m9 = np.zeros(crumb.size, bool) if kind else np.ones(crumb.size, bool)
            else:
                m9 = (fc == c9) if kind else (fc != c9)
        crumb = crumb[m9]
    for pcol in (spec['strneq'] if fusedV is None else []):
        if crumb.size == 0:
            break
        if seg.cols[pcol].get('code_enc') not in (8, 9):
            # any dress: '' owns code 0 in a sorted dict when present
            z9 = seg.fetch(pcol, 0)
            if isinstance(z9, (bytes, bytearray)):
                z9 = z9.decode('utf-8', 'replace')
            if z9 == '':
                fc = np.asarray(seg.codes_at(pcol, crumb))
                crumb = crumb[fc != 0]
            continue
        pl = seg.e8_planes(pcol)
        pos8 = np.asarray(pl[0])
        j = np.searchsorted(pos8, crumb)
        j = np.minimum(j, max(0, pos8.size - 1))
        m9 = (pos8[j] == crumb) if pos8.size else np.zeros(crumb.size, bool)
        crumb = crumb[m9]
    _LG.stage('hygiene', (_tm.perf_counter() - _t9) * 1000); _t9 = _tm.perf_counter()
    # group in code space via the KEY COMPILER: each plan evaluates to
    # codes on the crumb (pure numpy), the composite packs into one int64
    plans = spec['plans']
    if fusedV is not None:
        ucodes = fusedV[0].reshape(1, -1)
        ucnt = fusedV[1]
    widths = [max(1, int(pl['W']).bit_length()) for pl in plans]
    packed = sum(widths) <= 62                   # one int64 when it fits,
    if fusedV is not None:
        pass                                     # counts came from the kernel
    elif crumb.size == 0:                        # lexsort when it doesn't
        ucodes = np.empty((len(plans), 0), np.int64); ucnt = np.empty(0, np.int64)
    elif packed:
        shifts = np.cumsum([0] + widths[::-1])[:-1][::-1]
        key = np.zeros(crumb.size, np.int64)
        karrs = []
        for pl, sh in zip(plans, shifts):
            ka = _key_eval(seg, pl, crumb)
            karrs.append(ka)
            key |= ka << int(sh)
        KVtot = 1 << int(sum(widths))
        if KVtot <= (1 << 24):
            # Jackson's reduction, restored: lengths ARE the counts --
            # one bincount over the packed space, no sort at all
            cnts9 = np.bincount(key, minlength=KVtot)
            ukv = np.flatnonzero(cnts9)
            ucnt = cnts9[ukv].astype(np.int64)
        else:
            ks = np.sort(key, kind='stable')
            bnd = np.flatnonzero(np.diff(ks) != 0)
            starts = np.concatenate([[0], bnd + 1])
            ends = np.concatenate([bnd + 1, [ks.size]])
            ukv = ks[starts]
            ucnt = (ends - starts).astype(np.int64)
        ucodes = np.empty((len(plans), ukv.size), np.int64)
        for i9, (pl, sh) in enumerate(zip(plans, shifts)):
            ucodes[i9] = (ukv >> int(sh)) & ((1 << widths[i9]) - 1)
    else:
        karrs = [_key_eval(seg, pl, crumb) for pl in plans]
        order = np.lexsort(tuple(reversed(karrs)))
        srt = [ka[order] for ka in karrs]
        difs = np.zeros(crumb.size - 1, bool)
        for sa in srt:
            difs |= np.diff(sa) != 0
        bnd = np.flatnonzero(difs)
        starts = np.concatenate([[0], bnd + 1])
        ends = np.concatenate([bnd + 1, [crumb.size]])
        ucnt = (ends - starts).astype(np.int64)
        ucodes = np.empty((len(plans), starts.size), np.int64)
        for i9, sa in enumerate(srt):
            ucodes[i9] = sa[starts]
    k, off = spec['k'], spec['off']
    need = off + k
    n9 = min(need, ucnt.size)
    if n9 == 0:
        picks = np.empty(0, np.int64)
    else:
        order = np.argpartition(-ucnt, n9 - 1)[:n9] if ucnt.size > n9 \
            else np.arange(ucnt.size)
        order = order[np.argsort(-ucnt[order], kind='stable')]
        picks = order[off:off + k]
    _LG.stage('group', (_tm.perf_counter() - _t9) * 1000); _t9 = _tm.perf_counter()
    # batch the pluck: one values_at per column-plan for ALL picks
    batch9 = {}
    for i9, pl in enumerate(plans):
        if pl['kind'] == 'col' and picks.size and hasattr(seg, 'values_at'):
            cds = np.unique(ucodes[i9, picks])
            try:
                vs = seg.values_at(pl['col'], cds.astype(np.int64))
                mp9 = {}
                for cx, vx in zip(cds.tolist(), list(vs)):
                    if isinstance(vx, (bytes, bytearray)):
                        vx = vx.decode('utf-8', 'replace')
                    mp9[cx] = vx
                batch9[i9] = mp9
            except Exception:
                pass
    out = []
    for j in picks.tolist():
        vals = []
        for i9, pl in enumerate(plans):
            cx = int(ucodes[i9, j])
            if i9 in batch9 and cx in batch9[i9]:
                vals.append(batch9[i9][cx])
            else:
                vals.append(_key_decode(seg, pl, cx))   # THE pluck
        row = []
        for kind in spec['projkinds']:
            if kind[0] == 'G':
                row.append(vals[kind[1]])
            else:
                row.append(int(ucnt[j]))
        out.append(tuple(row))
    _LG.stage('emit', (_tm.perf_counter() - _t9) * 1000)
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]


def _sec_at(seg, ecol, rows):
    """Epoch seconds at ROWS for a staircase datetime column, all in code
    space: codes via one searchsorted on the steps (free), values via a
    memoized int64 view of the dictionary (V-sized, census-memo law).
    No row-range value decode anywhere."""
    memo = seg.__dict__.setdefault('_dv64memo', {})
    hit9 = memo.get(ecol)
    if hit9 is None:
        c9 = seg.cols[ecol]
        dv = np.asarray(seg._typed_dict(ecol), dtype=np.int64)
        unit = {0: 1, 1: 1, 2: 1}.get(0)         # placeholder; refined below
        u9 = c9.get('aux')
        # normalize to SECONDS whatever the stored unit
        div = {'s': 1, 'ms': 1000, 'us': 1000000, 'ns': 1000000000}
        import wdb_engine as _E
        un = _E._DT_UNITS[u9] if c9.get('dt') == 3 else 's'
        dv = dv // div.get(un, 1)
        steps9 = np.asarray(seg.stairs(ecol), dtype=np.int64)
        memo[ecol] = (steps9, dv)
        hit9 = memo[ecol]
    steps9, dv = hit9
    etc = np.searchsorted(steps9, np.asarray(rows, np.int64), side='right')
    return dv[etc]


def _execute_trunc(seg, spec):
    """Q42's shape (Jackson's cut): the window is a staircase span, so we
    walk CHUNKS from its LEFT EDGE -- popping only the selector frames the
    answer needs -- filter to the counter at each chunk, read minute values
    for survivors only, and run-walk minutes in file order (which IS minute
    order). The walk STOPS the moment run off+k closes. OFFSET deep pops
    almost nothing past its own answer."""
    global _HITS
    import datetime as _dtm
    scol, sval = spec['sel']
    code = _code_of(seg, scol, sval)
    ecol = spec['trunc'][0]
    rlo, rhi = 0, seg.N
    for rcol, (lo9, hi9) in spec['rng'].items():
        steps = np.asarray(seg.stairs(rcol), dtype=np.int64)
        full = np.concatenate([[0], steps, [seg.N]])
        def _tov(sv):
            try:
                return int(sv)
            except ValueError:
                y, m, d = [int(x) for x in sv.split('-')]
                return (_dtm.date(y, m, d) - _dtm.date(1970, 1, 1)).days
        V9 = int(seg.cols[rcol]['V'])
        vals = [_days_of(seg.fetch(rcol, c9)) for c9 in range(V9)]
        import bisect
        if lo9 is not None:
            c9 = bisect.bisect_left(vals, _tov(lo9[0]) + (1 if lo9[1] else 0))
            rlo = max(rlo, int(full[c9]))
        if hi9 is not None:
            c9 = bisect.bisect_right(vals, _tov(hi9[0]) - (1 if hi9[1] else 0))
            rhi = min(rhi, int(full[c9]))
    need = spec['off'] + spec['k']
    runs_m = []
    runs_c = []
    cur_m = -1
    cur_c = 0
    if code is not None:
        # THE PLIST START (the counter's own law finishing the job): the
        # crumb's positions are already on the shelf -- no selector frames
        # pop at all. Window = two searchsorteds on the row-ordered list.
        offs, plist = _plist(seg, scol)
        crumb = plist[int(offs[code]):int(offs[code + 1])]
        a9 = np.searchsorted(crumb, rlo, side='left')
        b9 = np.searchsorted(crumb, rhi, side='left')
        crumb = crumb[a9:b9].astype(np.int64)
    else:
        crumb = np.empty(0, np.int64)
    CH = 1 << 17                                 # crumb-prefix chunks: early stop
    a = 0
    while a < crumb.size:
        hit = crumb[a:a + CH]
        for fcol, fval, kind in spec['flags']:
            if hit.size == 0:
                break
            fc = np.asarray(seg.codes_at(fcol, hit))
            c9 = _code_of(seg, fcol, fval) if kind != 'in' else None
            if kind == 'in':
                want = [w9 for w9 in (_code_of(seg, fcol, v9) for v9 in fval)
                        if w9 is not None]
                hit = hit[np.isin(fc, want)] if want else hit[:0]
            elif c9 is None:
                hit = hit[:0] if kind else hit
            else:
                hit = hit[fc == c9] if kind else hit[fc != c9]
        if hit.size:
            mins = _sec_at(seg, ecol, hit) // 60
            bnd = np.flatnonzero(np.diff(mins) != 0)
            st9 = np.concatenate([[0], bnd + 1])
            en9 = np.concatenate([bnd + 1, [mins.size]])
            for i9 in range(st9.size):
                m9 = int(mins[st9[i9]])
                c99 = int(en9[i9] - st9[i9])
                if m9 == cur_m:
                    cur_c += c99
                else:
                    if cur_m >= 0:
                        runs_m.append(cur_m)
                        runs_c.append(cur_c)
                    cur_m = m9
                    cur_c = c99
        if len(runs_m) > need:                   # the needed runs are all CLOSED
            break
        a += CH
    if cur_m >= 0 and len(runs_m) <= need:
        runs_m.append(cur_m); runs_c.append(cur_c)
    out = []
    for i9 in range(spec['off'], min(spec['off'] + spec['k'], len(runs_m))):
        mv = _dtm.datetime(1970, 1, 1) + _dtm.timedelta(seconds=runs_m[i9] * 60)
        row = []
        for kind in spec['projkinds']:
            row.append(mv if kind[0] == 'M' else runs_c[i9])
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
