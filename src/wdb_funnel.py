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
_PHITS = 0                                       # prefix-group fires


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


def plist_ready(seg, col):
    """The position lists may serve: already loaded, on disk, or the switch allows their birth."""
    if col in seg.__dict__.get('_plistmemo', {}):
        return True
    p = _plist_path(seg, col)
    import wdb_sidecar
    return os.path.exists(p) or wdb_sidecar.may_build(p)


def positions(seg, col, code, lo=0, hi=None, blocks=None):
    """Row positions in [lo, hi) where col's code == `code`, ascending. From the position lists
    when they may serve; otherwise (THE VANILLA LAW) a scan of only the blocks whose load-time
    min/max can hold the code, then the frames covering them -- nothing built, nothing kept.
    `blocks` (a bool per load-statistics block) narrows the scan further: the blocks the query's
    OTHER equalities allow (THE REGION: search the selective column only where the rest can live)."""
    hi = int(seg.N) if hi is None else int(hi)
    if plist_ready(seg, col):
        offs, plist = _plist(seg, col)
        # (no warm_plist here: measured 2026-09-29, CounterID 62's 3 MB list is one stream either
        # way -- 20-25 ms with the parallel read first vs 17-19 by faults and the kernel's readahead)
        crumb = plist[int(offs[code]):int(offs[code + 1])]
        a = np.searchsorted(crumb, lo, side='left'); b = np.searchsorted(crumb, hi, side='left')
        return crumb[a:b].astype(np.int64)
    import wdb_blockstats, wdb_wherescan
    # the load statistics' min/max are CODES only for dictionary columns (modes 0/2); a mode-4
    # sequence keeps VALUES there while its codes are positions -- no pruning on it
    st = wdb_blockstats._from_load(seg, col) if seg.cols[col].get('mode') in (0, 2) else None
    if st is None and blocks is None:
        return np.asarray(wdb_wherescan._scan_eq(seg, col, int(code), lo, hi), dtype=np.int64)
    BR = wdb_blockstats._BR
    if st is not None:
        cand = (st['cmin'] <= code) & (st['cmax'] >= code)
        if blocks is not None and blocks.size == cand.size:
            cand &= blocks
    else:
        cand = blocks
    hit = np.flatnonzero(cand)
    hit = hit[(hit * BR < hi) & ((hit + 1) * BR > lo)]
    if hit.size == 0:
        return np.empty(0, np.int64)
    brk = np.flatnonzero(np.diff(hit) != 1) + 1               # contiguous runs of candidate blocks
    out = []
    for r in np.split(hit, brk):
        a = max(lo, int(r[0]) * BR); b = min(hi, (int(r[-1]) + 1) * BR)
        out.append(np.asarray(wdb_wherescan._scan_eq(seg, col, int(code), a, b), dtype=np.int64))
    return np.concatenate(out)


_PICK = [os.environ.get('WDB_PICKSEL', '1') != '0']


def _est_rows(seg, col, value):
    """Rows expected for col = value: the load's census when it holds the column, else the even
    share N / V (a column of many distinct values -- a hash -- is expected to be rare)."""
    import wdb_blockstats
    vc = wdb_blockstats.vcnt_from_load(seg, col)
    if vc is not None:
        c9 = _code_of(seg, col, value)
        return 0 if c9 is None else int(vc[c9])
    return int(seg.N) / max(1, int(seg.cols[col].get('V') or 1))


def _pick(seg, weq):
    """THE PICK: the selector is the wide equality expected to keep the fewest rows (the first in
    query order when the switch is off). A column whose position lists would be born over 4M codes
    stays out (the plist bound); in the vanilla scan that bound does not apply."""
    ok = [(c9, v9) for c9, v9 in weq
          if int(seg.cols[c9].get('V') or 1 << 40) <= 1 << 22 or not plist_ready(seg, c9)]
    if not ok:
        return None
    if not _PICK[0]:
        return ok[0] if ok[0] == weq[0] else None
    return min(ok, key=lambda cv: _est_rows(seg, cv[0], cv[1]))


def _eq_blocks(seg, spec):
    """THE REGION: the load-statistics blocks every equality of the query can live in (dictionary
    columns, modes 0/2, whose min/max are codes). None when no equality narrows anything."""
    if not _PICK[0]:
        return None
    import wdb_blockstats
    m = None
    for c9, v9, kind in [(spec['sel'][0], spec['sel'][1], True)] + list(spec['flags']):
        if kind is not True or seg.cols.get(c9, {}).get('mode') not in (0, 2):
            continue
        st = wdb_blockstats._from_load(seg, c9)
        if st is None:
            continue
        k9 = _code_of(seg, c9, v9)
        if k9 is None:
            continue
        ok = (st['cmin'] <= k9) & (st['cmax'] >= k9)
        m = ok if m is None else (m & ok) if m.size == ok.size else m
    return m


def _plist(seg, col):
    """Position lists per code for a selector column; birth-on-touch, ledgered."""
    memo = seg.__dict__.setdefault('_plistmemo', {})
    hit = memo.get(col)
    if hit is not None:
        return hit
    p = _plist_path(seg, col)
    V = int(seg.cols[col]['V'])
    if not os.path.exists(p):
        import wdb_sidecar
        # THE VANILLA LAW (FAIL-LOUD): with the switch off no read builds the position lists --
        # not on disk, not "RAM only". Callers ask plist_ready() and take positions()' scan.
        assert wdb_sidecar.may_build(p), ('THE VANILLA LAW: plist build refused', col)
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


def warm_plist(seg, col, codes=None, offsets=False):
    """THE COLD READ for the position lists (a memory-mapped sidecar): before a read touches them,
    the offsets (`offsets`, the whole census) and/or the lists of `codes` are brought in by parallel
    large reads instead of page faults (Q14 cold: the 64 top phrases' lists 78 -> ~13 ms). Resident
    spans read nothing. Returns the bytes read."""
    import wdb_engine
    offs, pos = _plist(seg, col)
    V9 = offs.size - 1
    P0 = 8 + 8 * (V9 + 1)                     # the file: V, offsets[V + 1], positions (uint32)
    spans = [(0, P0)] if offsets else []
    if codes is not None:
        for c9 in np.asarray(codes, dtype=np.int64).tolist():
            spans.append((P0 + 4 * int(offs[c9]), P0 + 4 * int(offs[c9 + 1])))
    if not spans:
        return 0
    fd = os.open(_plist_path(seg, col), os.O_RDONLY)
    try:
        return wdb_engine.warm_mapped(offs.ctypes.data - 8, fd, spans)
    finally:
        os.close(fd)


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
    weq = []                                     # every wide equality, in query order
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
            if isinstance(p9, E.EQ) and int(c9.get('V') or 0) >= 64:
                weq.append((cn, v9))             # wide equalities: THE PICK chooses the selector
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
    if weq:
        sel = _pick(seg, weq)
        if sel is None:
            return None                          # plist stays a bounded species
        flags.extend((c9, v9, True) for c9, v9 in weq if (c9, v9) != sel)
    if sel is None:
        return None
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
    global _PHITS
    global _HITS
    if spec.get('trunc'):
        return _execute_trunc(seg, spec)
    import time as _tm
    import wdb_ledger as _LG
    _t9 = _tm.perf_counter()
    scol, sval = spec['sel']
    code = _code_of(seg, scol, sval)
    # staircase windows first: each is a row range, their intersection is the only span
    # the selector's positions are asked for (vanilla scans just that span)
    wlo, whi = 0, int(seg.N)
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
        wlo = max(wlo, rlo); whi = min(whi, rhi)
    if code is None or wlo >= whi:
        crumb = np.empty(0, np.int64)
    else:
        crumb = positions(seg, scol, code, wlo, whi, blocks=_eq_blocks(seg, spec))
    _LG.stage('crumb', (_tm.perf_counter() - _t9) * 1000); _t9 = _tm.perf_counter()
    # hygiene at the crumb: point reads, never the column. When the first
    # two flags are enc-10 scalar tests, ONE fused walk serves both with
    # per-row short-circuit (Jackson's crossing scheme, row granularity).
    # JACKSON'S MORSEL DOCTRINE: partition the crumb once, sixteen full
    # pipelines, one combine. Covers <=3 enc-12 keys, <=2 scalar enc-10
    # flags, plus one enc-12 equality flag via a pre-built snowball mask.
    _mo_done = False
    _mof = [f for f in spec['flags'] if f[2] != 'in'
            and seg.cols.get(f[0], {}).get('code_enc') == 10]
    _moe = [f for f in spec['flags'] if f[2] != 'in'
            and seg.cols.get(f[0], {}).get('code_enc') == 12 and f[2]]
    _mop = spec['plans']
    if (crumb.size >= (1 << 15) and not spec['strneq']
            and len(_mof) + len(_moe) == len(spec['flags'])
            and len(_mof) <= 2 and len(_moe) <= 1
            and 1 <= len(_mop) <= 3 and all(p['kind'] == 'col' for p in _mop)
            and all(seg.cols.get(p['col'], {}).get('code_enc') == 12
                    and not seg.cols[p['col']].get('has_null') for p in _mop)
            and sum(max(1, int(seg.cols[p['col']].get('V', 2)).bit_length()) for p in _mop) <= 62
            and spec['off'] + spec['k'] <= 100000):
        import wdb_kernels as _WK
        codesF = [_code_of(seg, f[0], f[1]) for f in _mof]
        emc = _code_of(seg, _moe[0][0], _moe[0][1]) if _moe else None
        if all(c9 is not None for c9 in codesF):
            bufF = np.frombuffer(seg.buf, np.uint8)
            cr9 = np.ascontiguousarray(crumb)
            if _moe and emc is None:
                cr9 = cr9[:0]                    # eq on absent value: empty
            has_em = bool(_moe) and emc is not None
            if has_em:
                ec9 = seg.cols[_moe[0][0]]
                blo9 = int(cr9[0]); bhi9 = int(cr9[-1]) + 1
                w08 = blo9 // 64; w18 = (bhi9 + 63) // 64
                em9 = np.zeros(w18 - w08, np.uint64)
                _WK.vp_scan_eq_mask(seg.vplanes(_moe[0][0]), int(ec9['nwords']),
                                    int(ec9['bits']), int(emc), w08, w18, em9)
                ebase9 = w08 * 64
            else:
                em9 = np.zeros(1, np.uint64)
                ebase9 = 0
            def _fd9(f9):
                ca = seg.cols[f9[0]]
                return (np.ascontiguousarray(np.frombuffer(seg.buf, np.int64,
                        int(ca['pXnblk']), ca['pXdir'])),
                        int(ca['pXpay']), int(ca['pXbits']))
            if len(_mof) >= 1:
                da9, pa9, ba9 = _fd9(_mof[0])
            else:
                da9, pa9, ba9 = np.zeros(1, np.int64), 0, 1
            if len(_mof) >= 2:
                db9, pb9, bb9 = _fd9(_mof[1])
            else:
                db9, pb9, bb9 = np.zeros(1, np.int64), 0, 1
            all12 = all(seg.cols[p9['col']].get('code_enc') == 12 for p9 in _mop)
            kp9 = []
            wl9 = []
            for p9 in _mop:
                kc9 = seg.cols[p9['col']]
                wl9.append(max(1, int(kc9.get('V', 2)).bit_length()))
                if all12:
                    kp9.append((seg.vplanes(p9['col']), int(kc9['nwords']),
                                int(kc9['bits'])))
            if all12:
                wl9 = [int(seg.cols[p9['col']]['bits']) for p9 in _mop]
            dummy9 = np.zeros(2, np.uint64)
            while len(kp9) < 3:
                kp9.append(kp9[0] if kp9 else (dummy9, 1, 1))
            kbits9 = sum(wl9)
            nkeys9 = len(_mop) if all12 else 0
            NT9 = 16
            n9 = cr9.size
            ok9 = np.empty(n9, np.int64)
            tm9 = np.empty(n9, np.int64)
            uk9a = np.empty(n9, np.int64)
            uc9a = np.empty(n9, np.int64)
            gc9 = np.zeros(NT9, np.int64)
            if n9:
                _WK.morsel_group(cr9, bufF,
                                 da9, pa9, ba9,
                                 int(codesF[0]) if len(_mof) >= 1 else 0,
                                 bool(_mof[0][2]) if len(_mof) >= 1 else True,
                                 db9, pb9, bb9,
                                 int(codesF[1]) if len(_mof) >= 2 else 0,
                                 bool(_mof[1][2]) if len(_mof) >= 2 else True,
                                 len(_mof), em9, ebase9, has_em,
                                 kp9[0][0], kp9[0][1], kp9[0][2],
                                 kp9[1][0], kp9[1][1], kp9[1][2],
                                 kp9[2][0], kp9[2][1], kp9[2][2], nkeys9,
                                 kbits9, NT9, ok9, tm9, uk9a, uc9a, gc9)
            per9 = (n9 + NT9 - 1) // NT9 if n9 else 0
            if nkeys9 == 0 and n9:
                # rows mode: the morsel filtered in parallel; key + count the
                # small survivor set through the engine's any-encoding reads
                parts_r = [ok9[t * per9:t * per9 + int(gc9[t])] for t in range(NT9)]
                rows9 = np.concatenate(parts_r) if parts_r else np.empty(0, np.int64)
                allk9 = np.zeros(rows9.size, np.int64)
                for p9, b9 in zip(_mop, wl9):
                    f9 = np.asarray(seg.codes_at(p9['col'], rows9)).astype(np.int64)
                    np.left_shift(allk9, b9, out=allk9)
                    np.bitwise_or(allk9, f9, out=allk9)
                if allk9.size:
                    allk9 = _WK.radix_sortN(allk9, kbits9)
                    bd9 = np.flatnonzero(np.diff(allk9) != 0)
                    st9 = np.concatenate([[0], bd9 + 1])
                    uk9 = allk9[st9]
                    uc9 = np.diff(np.concatenate([st9, [allk9.size]])).astype(np.int64)
                else:
                    uk9 = np.empty(0, np.int64)
                    uc9 = np.empty(0, np.int64)
                allk9 = np.empty(0, np.int64)
                allc9 = np.empty(0, np.int64)
            else:
                parts_k = [uk9a[t * per9:t * per9 + int(gc9[t])] for t in range(NT9)] if n9 else []
                parts_c = [uc9a[t * per9:t * per9 + int(gc9[t])] for t in range(NT9)] if n9 else []
                allk9 = np.concatenate(parts_k) if parts_k else np.empty(0, np.int64)
                allc9 = np.concatenate(parts_c) if parts_c else np.empty(0, np.int64)
            if nkeys9 == 0 and n9:
                pass                             # uk9/uc9 already stand
            elif allk9.size:
                o9 = np.argsort(allk9, kind='stable')
                sk9 = allk9[o9]
                sc9 = allc9[o9]
                bd9 = np.flatnonzero(np.diff(sk9) != 0)
                st9 = np.concatenate([[0], bd9 + 1])
                uk9 = sk9[st9]
                uc9 = np.add.reduceat(sc9, st9)
            else:
                uk9 = np.empty(0, np.int64)
                uc9 = np.empty(0, np.int64)
            fields9 = np.empty((len(_mop), uk9.size), np.int64)
            acc9 = kbits9
            for i9, b9 in enumerate(wl9):
                acc9 -= b9
                fields9[i9] = (uk9 >> acc9) & ((1 << b9) - 1)
            _PHITS += 1
            ucodes = fields9
            ucnt = uc9.astype(np.int64)
            _mo_done = True
            crumb = crumb[:0]
            spec = dict(spec)
            spec['flags'] = []
    # THE FUSED BAND GROUP: Jackson's integration decree. When the shape
    # is (<=3 enc-12 keys, <=2 scalar enc-10 flags, no strneq, dense band),
    # every stage runs numba-to-numba in one entry -- hygiene as mask
    # surgery, lockstep decode, radix, walk -- zero interpreter between.
    # THE MARGINAL-BOUND LANE (Q14's family): GROUP BY (sparse-default
    # BigKey, small partner) ORDER BY COUNT DESC -- the marginal total of
    # BigKey ceilings every pair it contains, so candidates resolve in
    # descending-marginal order and the k-th resolved pair's count
    # guillotines the tail. Exact by the bound; zipf only sets how early.
    _mb_done = False
    _mbp = spec['plans']
    _q14_ok = False
    if (crumb.size == 0 and not spec['flags'] and len(_mbp) == 2
            and all(p['kind'] == 'col' for p in _mbp)
            and spec['off'] + spec['k'] <= 1000):
        e8i = [i for i, p in enumerate(_mbp)
               if seg.cols.get(p['col'], {}).get('code_enc') == 8]
        smi = [i for i, p in enumerate(_mbp)
               if seg.cols.get(p['col'], {}).get('code_enc') == 3
               and int(seg.cols[p['col']].get('V', 1 << 30)) <= 4096]
        if len(e8i) == 1 and len(smi) == 1 \
                and len(spec['strneq']) == 1 \
                and spec['strneq'][0][0] == _mbp[e8i[0]]['col'] \
                and spec['strneq'][0][1] == '':
            _q14_ok = True
    if _q14_ok:
        import wdb_kernels as _WK
        spB = seg.cols[_mbp[e8i[0]]['col']]
        spS = seg.cols[_mbp[smi[0]]['col']]
        bufQ = np.frombuffer(seg.buf, np.uint8)
        e8n = int(spB['e8n']); b8 = int(spB['e8bits']); V8 = int(spB['V'])
        lb9 = np.ascontiguousarray(
            bufQ[spB['cstart']:spB['cstart'] + (e8n * b8 + 7) // 8 + 8])
        codes9 = np.ascontiguousarray(_WK.unpack_any(lb9, e8n, b8).astype(np.int64))
        tot9 = np.bincount(codes9, minlength=V8)
        eng9 = np.ascontiguousarray(
            seg.codes_band(_mbp[smi[0]]['col'], 0, seg.N).astype(np.uint8))
        pres9 = np.ascontiguousarray(
            bufQ[spB['e8pres']:spB['e8pres'] + (seg.N + 7) // 8])
        ck9 = np.frombuffer(seg.buf, np.uint64,
                            (seg.N + 65535) // 65536, spB['e8ck']).astype(np.int64)
        ne9 = int(spS['V'])
        need9 = spec['off'] + spec['k']
        order9 = np.argsort(tot9, kind='stable')[::-1]
        K9 = max(64, 2 * need9)
        while True:
            K9 = min(K9, V8)
            candc9 = order9[:K9]
            cmap9 = np.full(V8, 255, np.uint8)
            cmap9[candc9] = np.minimum(np.arange(K9), 254)
            if K9 > 254:
                cmap9[candc9[:254]] = np.arange(254)
                cmap9[candc9[254:]] = 255
                K9 = 254
                candc9 = candc9[:254]
            bowls9 = np.zeros((16, K9 * ne9), np.int64)
            _WK.pres_pair_count(pres9, ck9, codes9, eng9, cmap9, K9, ne9,
                                bowls9)
            pc9 = bowls9.sum(0)
            srt9 = np.sort(pc9)[::-1]
            bar9 = int(srt9[need9 - 1]) if pc9.size >= need9 else 0
            maxun9 = int(tot9[order9[K9]]) if K9 < V8 else -1
            if maxun9 < bar9 or K9 >= min(V8, 254):
                break
            K9 = K9 * 4
        nz9 = np.flatnonzero(pc9)
        ci9 = nz9 // ne9
        ei9 = nz9 % ne9
        fields9 = np.empty((2, nz9.size), np.int64)
        fields9[e8i[0]] = candc9[ci9]
        fields9[smi[0]] = ei9
        _PHITS += 1
        ucodes = fields9
        ucnt = pc9[nz9].astype(np.int64)
        _mb_done = True
        spec = dict(spec)
        spec['strneq'] = []
    # THE COMPOSED LANE (Q40's shape): stair keys and enc-3 flags join the
    # stream -- every op a measured kernel, the glue microseconds. Fires
    # only when a staircase key or an enc-3 flag is present; pure shapes
    # keep the fused kernel below.
    _fs_done = False
    _fsp = spec['plans']
    _fs10 = [f for f in spec['flags'] if f[2] != 'in'
             and seg.cols.get(f[0], {}).get('code_enc') == 10]
    _fs12 = [f for f in spec['flags'] if f[2] != 'in' and f[2]
             and seg.cols.get(f[0], {}).get('code_enc') == 12]
    _fs3 = [f for f in spec['flags']
            if seg.cols.get(f[0], {}).get('code_enc') == 3]
    _hasst = any(p['kind'] == 'col'
                 and seg.cols.get(p['col'], {}).get('code_enc') == 2
                 and seg.cols.get(p['col'], {}).get('mode') == 0
                 for p in _fsp)
    import os as _os
    if (not _mo_done and not _os.environ.get('WDB_FS_OFF') 
            and crumb.size >= (1 << 15) and not spec['strneq']
            and (_hasst or _fs3)
            and (not _fs3 or (int(crumb[-1]) + 1 - int(crumb[0])) <= (1 << 22))
            and len(_fs10) + len(_fs12) + len(_fs3) == len(spec['flags'])
            and len(_fs10) <= 2 and len(_fs12) <= 1 and len(_fs3) <= 1
            and 1 <= len(_fsp) <= 3 and all(p['kind'] == 'col' for p in _fsp)
            and all((seg.cols.get(p['col'], {}).get('code_enc') == 12
                     or (seg.cols.get(p['col'], {}).get('code_enc') == 2
                         and seg.cols.get(p['col'], {}).get('mode') == 0))
                    and not seg.cols[p['col']].get('has_null') for p in _fsp)
            and sum(max(1, int(seg.cols[p['col']].get('V', 2)).bit_length())
                    for p in _fsp) <= 62
            and spec['off'] + spec['k'] <= 100000):
        blo9 = int(crumb[0]); bhi9 = int(crumb[-1]) + 1
        band9 = bhi9 - blo9
        codesA = [_code_of(seg, f[0], f[1]) for f in _fs10]
        emcA = _code_of(seg, _fs12[0][0], _fs12[0][1]) if _fs12 else None
        if band9 > 0 and crumb.size / band9 >= 0.02 \
                and all(c9 is not None for c9 in codesA):
            import wdb_kernels as _WK
            bufF = np.frombuffer(seg.buf, np.uint8)
            cr9 = np.ascontiguousarray(crumb)
            if _fs12 and emcA is None:
                cr9 = cr9[:0]
            uk9 = np.empty(0, np.int64)
            uc9 = np.empty(0, np.int64)
            wl9 = [max(1, int(seg.cols[p['col']].get('V', 2)).bit_length())
                   for p in _fsp]
            if cr9.size:
                w08 = blo9 // 64
                w18 = (bhi9 + 63) // 64
                mask9 = np.zeros(w18 - w08, np.uint64)
                _WK.vbits_set(cr9 - np.int64(w08 * 64), mask9)
                for f9, c9 in zip(_fs10, codesA):
                    ca = seg.cols[f9[0]]
                    d9 = np.ascontiguousarray(np.frombuffer(seg.buf, np.int64,
                            int(ca['pXnblk']), ca['pXdir']))
                    fm9 = np.zeros(w18 - w08, np.uint64)
                    if int(ca['pXbits']) == 1 and c9 in (0, 1):
                        wone9 = (c9 == 1) if f9[2] else (c9 == 0)
                        _WK.flag1_pass(bufF, d9, int(ca['pXpay']), w08 * 64,
                                       bhi9, wone9, _WK._REV8, fm9.view(np.uint8))
                    else:
                        _WK._flag_pass(bufF, d9, int(ca['pXpay']),
                                       int(ca['pXbits']), int(c9),
                                       bool(f9[2]), w08 * 64, bhi9, fm9)
                    np.bitwise_and(mask9, fm9, out=mask9)
                if _fs12 and emcA is not None:
                    ec9 = seg.cols[_fs12[0][0]]
                    em9 = np.zeros(w18 - w08, np.uint64)
                    _WK.vp_scan_eq_mask(seg.vplanes(_fs12[0][0]),
                                        int(ec9['nwords']), int(ec9['bits']),
                                        int(emcA), w08, w18, em9)
                    np.bitwise_and(mask9, em9, out=mask9)
                for f9 in _fs3:
                    v9 = np.ascontiguousarray(
                        seg.codes_band(f9[0], w08 * 64, bhi9).astype(np.int64))
                    fm9 = np.zeros(w18 - w08, np.uint64)
                    if f9[2] == 'in':
                        cs3 = [_code_of(seg, f9[0], lit9) for lit9 in f9[1]]
                        cs3 = [c for c in cs3 if c is not None]
                        if len(cs3) == 0:
                            mask9[:] = np.uint64(0)
                            continue
                        while len(cs3) < 2:
                            cs3.append(cs3[0])
                        if len(cs3) <= 2:
                            _WK.codes_test_mask(v9, int(cs3[0]), int(cs3[1]),
                                                0, fm9)
                        else:
                            p9m = np.zeros(v9.size, bool)
                            for c3 in cs3:
                                p9m |= (v9 == c3)
                            pb9 = np.packbits(p9m, bitorder='little')
                            fm9.view(np.uint8)[:pb9.size] = pb9
                    elif f9[2]:
                        c3 = _code_of(seg, f9[0], f9[1])
                        if c3 is None:
                            mask9[:] = np.uint64(0)
                            continue
                        _WK.codes_test_mask(v9, int(c3), int(c3), 0, fm9)
                    else:
                        c3 = _code_of(seg, f9[0], f9[1])
                        if c3 is None:
                            continue
                        _WK.codes_test_mask(v9, int(c3), int(c3), 1, fm9)
                    np.bitwise_and(mask9, fm9, out=mask9)
                CHW9 = 1024
                nch9 = (mask9.size + CHW9 - 1) // CHW9
                ob9 = np.zeros(nch9, np.int64)
                tot9 = _WK._offsets_from_mask(mask9, CHW9, ob9)
                if tot9:
                    rows9 = np.empty(tot9, np.int64)
                    _WK.mask_to_rows(mask9, w08, ob9, rows9, CHW9)
                    comp9 = np.zeros(tot9, np.int64)
                    tmp9 = np.empty(tot9, np.int64)
                    accb9 = 0
                    for p9, b9 in zip(_fsp, wl9):
                        kc9 = seg.cols[p9['col']]
                        if kc9.get('code_enc') == 12:
                            _WK.vp_gather_band(seg.vplanes(p9['col']),
                                               int(kc9['nwords']),
                                               int(kc9['bits']), mask9,
                                               w08, w18, ob9, tmp9)
                        else:
                            st9 = seg.stairs(p9['col'])
                            tmp9[:] = np.searchsorted(st9, rows9,
                                                      side='right')
                        np.left_shift(comp9, b9, out=comp9)
                        np.bitwise_or(comp9, tmp9, out=comp9)
                        accb9 += b9
                    ks9s = _WK.radix_sortN(comp9, accb9)
                    bd9 = np.flatnonzero(np.diff(ks9s) != 0)
                    st9x = np.concatenate([[0], bd9 + 1])
                    uk9 = ks9s[st9x]
                    uc9 = np.diff(np.concatenate([st9x, [ks9s.size]])).astype(np.int64)
            fields9 = np.empty((len(_fsp), uk9.size), np.int64)
            acc9 = sum(wl9)
            for i9, b9 in enumerate(wl9):
                acc9 -= b9
                fields9[i9] = (uk9 >> acc9) & ((1 << b9) - 1)
            _PHITS += 1
            ucodes = fields9
            ucnt = uc9
            _fs_done = True
            crumb = crumb[:0]
            spec = dict(spec)
            spec['flags'] = []
    _fb_done = False
    _fbf = [f for f in spec['flags'] if f[2] != 'in'
            and seg.cols.get(f[0], {}).get('code_enc') == 10]
    _fbe = [f for f in spec['flags'] if f[2] != 'in' and f[2]
            and seg.cols.get(f[0], {}).get('code_enc') == 12]
    _fbp = spec['plans']
    if (not _mo_done and crumb.size >= (1 << 15) and not spec['strneq']
            and len(_fbf) + len(_fbe) == len(spec['flags'])
            and len(_fbf) <= 2 and len(_fbe) <= 1
            and 1 <= len(_fbp) <= 3 and all(p['kind'] == 'col' for p in _fbp)
            and all(seg.cols.get(p['col'], {}).get('code_enc') in (5, 6, 12)
                    and not seg.cols[p['col']].get('has_null') for p in _fbp)
            and sum(max(1, int(seg.cols[p['col']].get('V', 2)).bit_length())
                    for p in _fbp) <= 62
            and spec['off'] + spec['k'] <= 100000):
        blo9 = int(crumb[0]); bhi9 = int(crumb[-1]) + 1
        band9 = bhi9 - blo9
        codesF = [_code_of(seg, f[0], f[1]) for f in _fbf]
        emc9 = _code_of(seg, _fbe[0][0], _fbe[0][1]) if _fbe else None
        if band9 > 0 and crumb.size / band9 >= 0.02 \
                and all(c9 is not None for c9 in codesF):
            import wdb_kernels as _WK
            bufF = np.frombuffer(seg.buf, np.uint8)
            cr9 = np.ascontiguousarray(crumb)
            if _fbe and emc9 is None:
                cr9 = cr9[:0]
            fl9 = []
            for f9, c9 in zip(_fbf, codesF):
                ca = seg.cols[f9[0]]
                fl9.append((np.ascontiguousarray(np.frombuffer(seg.buf, np.int64,
                            int(ca['pXnblk']), ca['pXdir'])),
                            int(ca['pXpay']), int(ca['pXbits']),
                            int(c9), bool(f9[2])))
            em9 = None
            if _fbe and emc9 is not None and cr9.size:
                ec9 = seg.cols[_fbe[0][0]]
                w08 = (int(cr9[0]) // 64)
                w18 = ((int(cr9[-1]) + 1 + 63) // 64)
                em9 = np.zeros(w18 - w08, np.uint64)
                _WK.vp_scan_eq_mask(seg.vplanes(_fbe[0][0]), int(ec9['nwords']),
                                    int(ec9['bits']), int(emc9), w08, w18, em9)
            ks9 = []
            wl9 = []
            for p9 in _fbp:
                kc9 = seg.cols[p9['col']]
                if kc9.get('code_enc') == 12:
                    b9 = int(kc9['bits'])
                    ks9.append(('p', seg.vplanes(p9['col']),
                                int(kc9['nwords']), b9))
                elif kc9.get('code_enc') == 6:
                    b9 = max(1, int(kc9['V']).bit_length())
                    pk9 = np.frombuffer(seg.buf, np.uint8,
                                        int(kc9['czlen']), int(kc9['cstart']))
                    ks9.append(('e6', pk9, np.asarray(kc9['e5hot']),
                                np.asarray(kc9['e6warm']),
                                np.asarray(kc9['e6wb']),
                                np.asarray(kc9['e5patch']),
                                np.asarray(kc9['e6o1']).astype(np.int64),
                                np.asarray(kc9['e6o2']).astype(np.int64),
                                np.int64(kc9['BR']), b9))
                else:
                    b9 = max(1, int(kc9['V']).bit_length())
                    pk9 = np.frombuffer(seg.buf, np.uint8,
                                        int(kc9['czlen']), int(kc9['cstart']))
                    ks9.append(('e5', pk9, np.asarray(kc9['e5hot']),
                                np.asarray(kc9['e5patch']),
                                np.asarray(kc9['e5off']).astype(np.int64),
                                np.int64(kc9['BR']), b9))
                wl9.append(b9)
            if cr9.size:
                uk9, uc9, _ = _WK.fused_band_group(bufF, fl9, ks9, cr9,
                                                   int(cr9[0]),
                                                   int(cr9[-1]) + 1, em9)
            else:
                uk9 = np.empty(0, np.int64)
                uc9 = np.empty(0, np.int64)
            fields9 = np.empty((len(_fbp), uk9.size), np.int64)
            acc9 = sum(wl9)
            for i9, b9 in enumerate(wl9):
                acc9 -= b9
                fields9[i9] = (uk9 >> acc9) & ((1 << b9) - 1)
            _PHITS += 1
            ucodes = fields9
            ucnt = uc9
            _fb_done = True
            crumb = crumb[:0]
            spec = dict(spec)
            spec['flags'] = []
    # THE SELECTIVE-FIRST CUT: an enc-12 scalar flag is an equality the
    # SNOWBALL tests word-parallel over the band -- maximally selective
    # predicates run first, and everything downstream touches only their
    # survivors. (Q41's URLHash = const was buried in per-row hygiene.)
    _sf = [] if (_fs_done or _mo_done or _fb_done) else [f for f in spec['flags']
           if f[2] != 'in' and seg.cols.get(f[0], {}).get('code_enc') == 12]
    if _sf and crumb.size >= (1 << 15):
        import wdb_kernels as _WK
        blo8 = int(crumb[0]); bhi8 = int(crumb[-1]) + 1
        w08 = blo8 // 64; w18 = (bhi8 + 63) // 64
        keepF = None
        rel8 = crumb - np.int64(w08 * 64)
        done8 = []
        for f8 in _sf:
            c8 = _code_of(seg, f8[0], f8[1])
            kc8 = seg.cols[f8[0]]
            if f8[2] and c8 is None:
                crumb = crumb[:0]
                done8.append(f8)
                continue
            if c8 is None:
                done8.append(f8)                 # neq missing value: all pass
                continue
            m8 = np.zeros(w18 - w08, np.uint64)
            _WK.vp_scan_eq_mask(seg.vplanes(f8[0]), int(kc8['nwords']),
                                int(kc8['bits']), int(c8), w08, w18, m8)
            hit8 = (m8[rel8 >> 6] >> (rel8 & 63).astype(np.uint64)) \
                & np.uint64(1)
            k8 = hit8.astype(bool) if f8[2] else ~hit8.astype(bool)
            keepF = k8 if keepF is None else (keepF & k8)
            done8.append(f8)
        if keepF is not None:
            crumb = crumb[keepF]
        if done8:
            spec = dict(spec)
            spec['flags'] = [f for f in spec['flags'] if f not in done8]
    flags9 = list(spec['flags'])
    # THE BAND-DECODE LAW (hygiene half): dense crumbs stream their band's
    # flag columns once instead of probing per row.
    if crumb.size >= (1 << 15) and 1 <= len(flags9) <= 2 \
            and not spec['strneq'] \
            and all(f[2] != 'in' for f in flags9) \
            and all(seg.cols.get(f[0], {}).get('code_enc') == 10 for f in flags9):
        blo9 = int(crumb[0]); bhi9 = int(crumb[-1]) + 1
        band9 = bhi9 - blo9
        if band9 > 0 and crumb.size / band9 >= 0.02 and band9 <= (1 << 26):
            import wdb_kernels as _WK
            bufB = np.frombuffer(seg.buf, np.uint8)
            keepB = np.ones(crumb.size, bool)
            rel9 = crumb - blo9
            okB = True
            for f9 in flags9:
                c9 = _code_of(seg, f9[0], f9[1])
                if c9 is None:
                    okB = False
                    break
                ca = seg.cols[f9[0]]
                d9 = np.ascontiguousarray(np.frombuffer(seg.buf, np.int64,
                        int(ca['pXnblk']), ca['pXdir']))
                v9 = np.empty(band9, np.int16)
                _WK.bp10_range(bufB, d9, int(ca['pXpay']), int(ca['pXbits']),
                               blo9, bhi9, v9)
                fv9 = v9[rel9]
                keepB &= (fv9 == c9) if f9[2] else (fv9 != c9)
            if okB:
                crumb = crumb[keepB]
                flags9 = []
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
    _pg_plans = None
    _pg_part = False
    if crumb.size >= (1 << 15):
        _cand = spec['plans']
        def _pgb(p9):
            if p9['kind'] == 'col':
                c9 = seg.cols.get(p9['col'], {})
                if not c9 or c9.get('has_null'):
                    return None
                return max(1, int(c9['V']).bit_length())
            return max(1, int(p9['W']).bit_length())
        if _cand and any(p9['kind'] == 'col' for p9 in _cand):
            _bl = [_pgb(p9) for p9 in _cand]
            if all(b9 is not None and b9 >= 1 for b9 in _bl):
                if sum(_bl) <= 63:
                    _pg_plans = list(zip(_cand, _bl))
                elif len(_bl) >= 2 and _bl[0] <= 12 and sum(_bl[1:]) <= 62:
                    _pg_plans = list(zip(_cand, _bl))
                    _pg_part = True              # Jackson's partition-first
    _pg_cols = set(p9['col'] for p9, _ in (_pg_plans or [])
                   if p9['kind'] == 'col')
    for pcol in spec['strneq']:
        if pcol in _pg_cols:
            continue                             # folded into the radix walk
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
    widths = [max(1, int(pl['W']).bit_length()) for pl in plans]
    # JACKSON'S RADIX GROUP: enc-12 keys span-gather, other encodings ride
    # codes_at, compiled CASE keys evaluate in code space. <=63 bits pack
    # one word for the radix atom; wider composites run PARTITION-FIRST --
    # the top field (<=12 bits) routes rows into buckets in one stable
    # pass, the <=62-bit remainder rides as payload, and each bucket runs
    # the single-word engine. Empty-string hygiene folds into dropped
    # field-0 groups.
    prefixed = False
    if _mb_done or _fs_done or _mo_done or _fb_done:
        _pg_plans = None
    if _pg_plans is not None:
        import wdb_kernels as _WK
        cr9 = np.ascontiguousarray(crumb)

        def _field(p9, b9):
            if p9['kind'] != 'col':
                return _key_eval(seg, p9, cr9)
            kc9 = seg.cols[p9['col']]
            if kc9.get('code_enc') != 12:
                return np.asarray(seg.codes_at(p9['col'], cr9)).astype(np.int64)
            nw9 = int(kc9['nwords'])
            pl9 = seg.vplanes(p9['col'])
            blo9 = int(cr9[0]); bhi9 = int(cr9[-1]) + 1
            band9 = bhi9 - blo9
            if band9 > 0 and cr9.size / band9 >= 0.02:
                # JACKSON'S LOCKSTEP GATHER: planes stream shoulder-to-
                # shoulder across the band, output compacts by the mask
                w0 = blo9 // 64
                w1 = (bhi9 + 63) // 64
                maskB = np.zeros(w1 - w0, np.uint64)
                _WK.vbits_set(cr9 - np.int64(w0 * 64), maskB)
                pcs = np.zeros(maskB.size, np.int64)
                _WK.vbits_pop(maskB, pcs)
                CHW = 1024
                nch = (w1 - w0 + CHW - 1) // CHW
                ob = np.zeros(nch, np.int64)
                cs = np.add.reduceat(pcs, np.arange(0, pcs.size, CHW))
                np.cumsum(cs[:-1], out=ob[1:])
                outC = np.empty(cr9.size, np.int64)
                _WK.vp_gather_band(pl9, nw9, b9, maskB, w0, w1, ob, outC)
                return outC
            if b9 > 16:
                P9 = b9 - 12
                a9 = np.zeros(cr9.size, np.uint16)
                _WK.vp_gather_span(pl9, nw9, 0, P9, cr9, a9)
                z9 = np.zeros(cr9.size, np.uint16)
                _WK.vp_gather_span(pl9, nw9, P9, b9, cr9, z9)
                return (a9.astype(np.int64) << (b9 - P9)) | z9.astype(np.int64)
            w9 = np.zeros(cr9.size, np.uint16)
            _WK.vp_gather_span(pl9, nw9, 0, b9, cr9, w9)
            return w9.astype(np.int64)

        comp9 = np.zeros(cr9.size, np.int64)
        top9 = None
        lowb9 = 0
        for oix, (p9, b9) in enumerate(_pg_plans):
            f9 = _field(p9, b9)
            if _pg_part and oix == 0:
                top9 = f9
                continue
            comp9 = (comp9 << b9) | f9
            lowb9 += b9
        if _pg_part:
            # the top digit NEVER re-enters the key (top << 61 would
            # overflow int64) -- it rides as its own per-bucket field
            _, sp9, bounds9 = _WK.radix_partition(top9, comp9)
            uks = []
            ucs = []
            tvs = []
            for dval9, a9x, b9x in bounds9:
                kb9 = _WK.radix_sortN(sp9[a9x:b9x], lowb9)
                bb9 = np.flatnonzero(np.diff(kb9) != 0)
                sb9 = np.concatenate([[0], bb9 + 1])
                uks.append(kb9[sb9])
                ucs.append(np.diff(np.concatenate([sb9, [kb9.size]])))
                tvs.append(np.full(sb9.size, dval9, np.int64))
            uk9 = np.concatenate(uks) if uks else np.empty(0, np.int64)
            uc9 = np.concatenate(ucs).astype(np.int64) if ucs else np.empty(0, np.int64)
            topv9 = np.concatenate(tvs) if tvs else np.empty(0, np.int64)
            fullb9 = lowb9
        else:
            topv9 = None
            ks9 = _WK.radix_sortN(comp9, lowb9)
            bnd9 = np.flatnonzero(np.diff(ks9) != 0)
            st9 = np.concatenate([[0], bnd9 + 1])
            uk9 = ks9[st9]
            uc9 = np.diff(np.concatenate([st9, [ks9.size]])).astype(np.int64)
            fullb9 = lowb9
        shifts9 = []
        acc0 = fullb9
        for oix, (p9, b9) in enumerate(_pg_plans):
            if _pg_part and oix == 0:
                shifts9.append(0)                # top rides apart
                continue
            acc0 -= b9
            shifts9.append(acc0)
        fields9 = np.empty((len(_pg_plans), uk9.size), np.int64)
        for oix, (p9, b9) in enumerate(_pg_plans):
            if _pg_part and oix == 0:
                fields9[0] = topv9
            else:
                fields9[oix] = (uk9 >> shifts9[oix]) & ((1 << b9) - 1)
        drop9 = np.zeros(uk9.size, bool)
        for oix, (p9, b9) in enumerate(_pg_plans):
            if p9['kind'] == 'col' and any(pc == p9['col'] for pc in spec['strneq']):
                z9 = seg.fetch(p9['col'], 0)
                if isinstance(z9, (bytes, bytearray)):
                    z9 = z9.decode('utf-8', 'replace')
                if z9 == '':
                    drop9 |= fields9[oix] == 0
        if drop9.any():
            keep9 = ~drop9
            fields9 = fields9[:, keep9]
            uc9 = uc9[keep9]
        _PHITS += 1
        ucodes = fields9
        ucnt = uc9
        prefixed = True
    packed = sum(widths) <= 62                   # one int64 when it fits,
    if _mb_done or _fs_done or _mo_done or _fb_done or prefixed:
        pass                                     # counts already stand
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
        if KVtot <= (1 << 24) and KVtot <= 4 * key.size:
            # the bowl must not dwarf the rows: zeroing a V-sized
            # bincount for V >> rows costs more than sorting the rows
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
        crumb = positions(seg, scol, code, rlo, rhi, blocks=_eq_blocks(seg, spec))   # the window first: vanilla scans only it
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
