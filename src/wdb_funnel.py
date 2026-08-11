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
    gcols = []
    for ge in g.expressions:
        src = ge
        if isinstance(ge, E.Literal) and not ge.is_string:
            idx = int(str(ge.this)) - 1
            if idx < 0 or idx >= len(tree.expressions):
                return None
            item = tree.expressions[idx]
            src = item.this if isinstance(item, E.Alias) else item
        if not isinstance(src, E.Column):
            return None
        gcols.append(cm.get(src.name, src.name))
    gcols = list(dict.fromkeys(gcols))
    if len(gcols) > 2:
        return None
    for gc in gcols:
        if seg.cols.get(gc) is None or seg.cols[gc].get('has_null'):
            return None
    proj = []
    calias = None
    for p9 in tree.expressions:
        inner = p9.this if isinstance(p9, E.Alias) else p9
        if isinstance(inner, E.Column):
            cn = cm.get(inner.name, inner.name)
            if cn not in gcols:
                return None
            proj.append(('G', gcols.index(cn))); continue
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
            'g': gcols, 'k': lim, 'off': off, 'projkinds': proj,
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
    # hygiene at the crumb: point reads, never the column
    for fcol, fval, kind in spec['flags']:
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
    for pcol in spec['strneq']:
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
    # group in code space: only the differentiating bytes, no more
    gcols = spec['g']
    if crumb.size == 0:
        ukey = np.empty(0, np.int64); ucnt = np.empty(0, np.int64)
    else:
        if len(gcols) == 1:
            key = np.asarray(seg.codes_at(gcols[0], crumb)).astype(np.int64)
            SH = 0
        else:
            k1 = np.asarray(seg.codes_at(gcols[0], crumb)).astype(np.int64)
            k2 = np.asarray(seg.codes_at(gcols[1], crumb)).astype(np.int64)
            SH = max(1, int(seg.cols[gcols[1]]['V']).bit_length())
            key = (k1 << SH) | k2
        ks = np.sort(key, kind='stable')
        bnd = np.flatnonzero(np.diff(ks) != 0)
        starts = np.concatenate([[0], bnd + 1])
        ends = np.concatenate([bnd + 1, [ks.size]])
        ukey = ks[starts]
        ucnt = (ends - starts).astype(np.int64)
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
    out = []
    SH = 0 if len(gcols) == 1 else max(1, int(seg.cols[gcols[1]]['V']).bit_length())
    for j in picks.tolist():
        kv = int(ukey[j])
        codes = [kv] if len(gcols) == 1 else [kv >> SH, kv & ((1 << SH) - 1)]
        vals = []
        for gi, gc in enumerate(gcols):
            v9 = seg.fetch(gc, int(codes[gi]))   # THE pluck
            if isinstance(v9, (bytes, bytearray)):
                v9 = v9.decode('utf-8', 'replace')
            vals.append(v9)
        row = []
        for kind in spec['projkinds']:
            if kind[0] == 'G':
                row.append(vals[kind[1]])
            else:
                row.append(int(ucnt[j]))
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]


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
            sec = np.asarray(seg.values_range(ecol, int(hit[0]), int(hit[-1]) + 1))
            if sec.dtype.kind == 'M':
                sec = sec.astype('datetime64[s]').astype(np.int64)
            mins = (sec[hit - int(hit[0])].astype(np.int64) // 60)
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
