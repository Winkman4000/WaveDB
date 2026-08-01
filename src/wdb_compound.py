"""
wdb_compound -- compound-AND filtered GROUP BY COUNT(*) via survivor-set resolution.

For  SELECT k1[,k2,...], COUNT(*) FROM t WHERE <conjunction of simple predicates>
     GROUP BY k1[,k2,...] ORDER BY COUNT(*) DESC LIMIT N [OFFSET M]
the filter is resolved to a small survivor set WITHOUT decoding any filter column over all N rows:
the selective mode-4 conjuncts (few exception segments) become survivor ROW-RANGES that are
intersected; every other conjunct is applied as a mask on just those survivors. The survivors are
then grouped on integer identities (codes for dict keys, values for mode-4 keys), and only the
emitted group representatives are decoded.

Sibling of wdb_survgroup (single predicate, single key); this owns compound AND + multi-key.
Fail-closed on any shape it does not own (returns None -> caller falls through).
"""
import re
import numpy as np
import wdb_sql
import wdb_gbcount
import wdb_seqpred as SP
import wdb_policies as P
import wdb_measure_runtime as RT
E = wdb_sql.E

_HITS = 0
# range-path eligibility (filter-column exception cap) lives in wdb_measure_runtime: RT.compound_range_worth_it(nexc).
_OPSTR = {E.EQ: '=', E.NEQ: '<>', E.LT: '<', E.LTE: '<=', E.GT: '>', E.GTE: '>='}
_OP2SP = {'=': SP.EQ, '<>': SP.NEQ, '!=': SP.NEQ, '<': SP.LT, '<=': SP.LTE, '>': SP.GT, '>=': SP.GTE}
_FLIP = {'<': '>', '>': '<', '<=': '>=', '>=': '<=', '=': '=', '<>': '<>', '!=': '!='}
_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}')


def _intersect(la, ha, lb, hb):
    nA, nB = len(la), len(lb)
    olo = np.empty(nA + nB, np.int64); ohi = np.empty(nA + nB, np.int64); i = j = k = 0
    while i < nA and j < nB:
        lo = la[i] if la[i] > lb[j] else lb[j]; hi = ha[i] if ha[i] < hb[j] else hb[j]
        if lo < hi: olo[k] = lo; ohi[k] = hi; k += 1
        if ha[i] < hb[j]: i += 1
        else: j += 1
    return olo[:k].copy(), ohi[:k].copy()


def _m4_range(seg, col, lo, hi):
    sl, sh, sv = SP._struct(seg, col)
    k0 = int(np.searchsorted(sl, lo, 'right')) - 1; k1 = int(np.searchsorted(sl, hi, 'left'))
    ks = np.arange(k0, k1); a = np.maximum(sl[ks], lo); b = np.minimum(sh[ks], hi)
    return np.repeat(sv[ks], b - a)


def _m4_in_ranges(seg, col, los, his):
    return np.concatenate([_m4_range(seg, col, int(a), int(b)) for a, b in zip(los, his)])


def _codes_in_ranges(seg, col, los, his):
    return np.concatenate([np.asarray(seg._raw_codes_range(col, int(a), int(b))) for a, b in zip(los, his)])


def _APPLY(v, op, c):
    return {SP.EQ: v == c, SP.NEQ: v != c, SP.LT: v < c, SP.LTE: v <= c, SP.GT: v > c, SP.GTE: v >= c}[op]


def _coerce(seg, col, lit):
    """Literal node -> value in column `col`'s domain. Adds date-string -> day-number coercion for
    integer columns that store days-since-epoch (e.g. ClickBench EventDate)."""
    if isinstance(lit, E.Neg):
        return -_coerce(seg, col, lit.this)
    c = seg.cols[col]
    if c['dt'] == 0 and isinstance(lit, E.Literal) and lit.args.get('is_string'):
        s = str(lit.this)
        if _DATE_RE.match(s):
            return int(np.datetime64(s.replace(' ', 'T')).astype('datetime64[D]').view('int64'))
    return wdb_sql._lit_for_col(seg, col, lit, 'i')


def _code_of(seg, col, value):
    """Code for an exact value in a dict column (mode 0/1/2); -1 if absent; None if unsupported."""
    c = seg.cols[col]
    if c['mode'] == 2:
        d = seg._dict_ints(c); i = int(np.searchsorted(d, value))
        return i if (i < len(d) and d[i] == value) else -1
    if value in (b'', ''):                       # empty string sorts first in a sorted dict -> code 0
        return 0 if seg.fetch(col, 0) in (b'', '') else -1
    return None




def _conjuncts(seg, where_node, sc):
    """Flatten a top-level AND into simple conjunct descriptors, or None if any leaf is unsupported.
    Each: {'col','op','const'} | {'col','op':'in','const':[...]} | {'col','op','strempty':True}."""
    out = []
    for n in wdb_sql._flatten_and(where_node):
        if isinstance(n, (E.EQ, E.NEQ, E.LT, E.LTE, E.GT, E.GTE)):
            op = _OPSTR[type(n)]; a = n.this; b = n.args.get('expression')
            if b is None: return None
            if isinstance(a, E.Column) and not isinstance(b, E.Column):
                col = sc(a.name); lit = b
            elif isinstance(b, E.Column) and not isinstance(a, E.Column):
                col = sc(b.name); lit = a; op = _FLIP[op]
            else:
                return None
            if col not in seg.cols: return None
            if isinstance(lit, E.Literal) and lit.args.get('is_string') and str(lit.this) == '':
                out.append({'col': col, 'op': op, 'strempty': True}); continue
            try: const = _coerce(seg, col, lit)
            except Exception: return None
            out.append({'col': col, 'op': op, 'const': const})
        elif isinstance(n, E.In):
            cn = n.this
            if not isinstance(cn, E.Column): return None
            col = sc(cn.name)
            if col not in seg.cols: return None
            exprs = n.args.get('expressions') or []
            if not exprs: return None
            try: consts = [_coerce(seg, col, e) for e in exprs]
            except Exception: return None
            out.append({'col': col, 'op': 'in', 'const': consts})
        elif isinstance(n, E.Between):
            cn = n.this
            if not isinstance(cn, E.Column): return None
            col = sc(cn.name)
            if col not in seg.cols: return None
            try:
                lo = _coerce(seg, col, n.args['low']); hi = _coerce(seg, col, n.args['high'])
            except Exception: return None
            out.append({'col': col, 'op': '>=', 'const': lo})
            out.append({'col': col, 'op': '<=', 'const': hi})
        else:
            return None
    return out


def _resolve(seg, conjuncts):
    """(los,his), keep-mask over ranges_to_ids order; or None to decline (no cheap anchor)."""
    ranges = []; residual = []
    for cj in conjuncts:
        col = cj['col']; op = cj['op']
        if op == 'in' or cj.get('strempty'): residual.append(cj); continue
        ne = SP.n_exceptions(seg, col)
        if not RT.compound_range_worth_it(ne): residual.append(cj); continue
        r = SP.survivor_ranges(seg, col, _OP2SP[op], int(cj['const']))
        if r is None: residual.append(cj); continue
        if SP.survivor_count(*r) == seg.N: continue          # covers all rows -> no-op
        ranges.append(r)
    if not ranges: return None
    cur = ranges[0]
    for r in ranges[1:]:
        cur = _intersect(cur[0], cur[1], r[0], r[1])
        if len(cur[0]) == 0:
            return (np.empty(0, np.int64), np.empty(0, np.int64)), np.zeros(0, bool)
    los, his = cur; n = int((his - los).sum()); keep = np.ones(n, bool)
    for cj in residual:
        col = cj['col']; op = cj['op']; c = seg.cols[col]; ne = SP.n_exceptions(seg, col)
        if cj.get('strempty'):
            ec = _code_of(seg, col, b''); cc = _codes_in_ranges(seg, col, los, his)
            m = (cc != ec) if op in ('<>', '!=') else (cc == ec)
        elif ne is not None:
            vals = _m4_in_ranges(seg, col, los, his)
            m = np.isin(vals, np.array(cj['const'])) if op == 'in' else _APPLY(vals, _OP2SP[op], cj['const'])
        elif c['mode'] == 2:
            cc = _codes_in_ranges(seg, col, los, his)
            if op == 'in':
                cs = [x for x in (_code_of(seg, col, v) for v in cj['const']) if x is not None and x >= 0]
                if cs:
                    fl = np.zeros(int(c['V']), bool)      # membership in code space is a
                    fl[np.asarray(cs, dtype=np.int64)] = True   # V-sized flag + one gather,
                    m = fl[cc.astype(np.int64, copy=False)]     # not a sort (np.isin was
                else:                                     # 332ms of sq-nested's 753)
                    m = np.zeros(n, bool)
            else:
                cd = _code_of(seg, col, int(cj['const']))
                if cd is None: return None
                m = (cc == cd) if _OP2SP[op] == SP.EQ else (cc != cd)
        else:
            vals = np.concatenate([np.asarray(seg.values_range(col, int(a), int(b))) for a, b in zip(los, his)])
            m = np.isin(vals, np.array(cj['const'])) if op == 'in' else _APPLY(vals, _OP2SP[op], cj['const'])
        keep &= m
    return (los, his), keep


def _offset(tree):
    o = tree.args.get('offset')
    if o is None: return 0
    try: return int(o.expression.this)
    except Exception: return 0


def detect(seg, tree, col_map):
    """ACTIVATION: static shape + segment-metadata guards for the compound-AND read.
    Touches no row data. Returns a spec or None. Self-validating; fail-closed on every
    shape it does not own."""
    # --- shared shape guards (wdb_policies); FILTERED + MULTI-KEY, so has_where + has_group_key (>=1) ---
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    if not P.has_where(tree):          return None
    if not P.has_group_key(tree):      return None
    where = tree.args.get('where')
    group = tree.args.get('group')
    proj = tree.expressions
    ci = wdb_gbcount._count_index(proj)
    if ci is None: return None
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)

    keycols = []                                          # logical GROUP BY column names (plain cols only)
    for ge in group.expressions:
        nm = wdb_sql._colname(ge)
        if nm is None: return None
        keycols.append(nm)
    proj_keys = []                                        # non-count projections must be bare key cols
    for idx, p in enumerate(proj):
        if idx == ci: continue
        if wdb_sql._agg_kind(p) is not None: return None
        nm = wdb_sql._proj_colname(p)
        if nm is None: return None
        proj_keys.append(nm)
    if set(proj_keys) != set(keycols): return None
    order = tree.args.get('order')
    if order is not None and not wdb_gbcount._order_is_count_desc(tree, proj, ci): return None

    pc = [sc(k) for k in keycols]
    # --- shared segment/column guards (wdb_policies) ---
    if not P.columns_exist(seg, *pc): return None
    if not P.no_deleted_rows(seg):    return None         # deleted rows: ranges don't model presence
    return {'ci': ci, 'sc': sc, 'keycols': keycols, 'pc': pc, 'proj': proj,
            'where_node': where.this, 'tree': tree}


def execute(seg, spec):
    """THE READ: count the survivors of the compound-AND filter, grouped. Single- or multi-key.
    May decline (None) on the measured packing-overflow guard."""
    global _HITS
    ci = spec['ci']; sc = spec['sc']; keycols = spec['keycols']; pc = spec['pc']
    proj = spec['proj']; where_node = spec['where_node']; tree = spec['tree']

    conj = _conjuncts(seg, where_node, sc)
    if conj is None: return None
    res = _resolve(seg, conj)
    if res is None: return None
    (los, his), keep = res

    names = [wdb_sql._alias(p) for p in proj]
    lim = wdb_sql._limit(tree); off = _offset(tree)

    if len(pc) == 1:                                      # single-key fast path: one np.unique, no
        k = pc[0]                                         # factorize/pack/unpack (those only help multi-key)
        if SP.n_exceptions(seg, k) is not None:
            arr = _m4_in_ranges(seg, k, los, his)[keep]; kind = 'val'
        else:
            arr = _codes_in_ranges(seg, k, los, his)[keep]; kind = 'code'
        if len(arr) == 0:
            _HITS += 1; return [], names
        u, counts = np.unique(arr.astype(np.int64), return_counts=True)
        sel = np.argsort(-counts, kind='stable')
        sel = sel[off: off + lim] if lim is not None else sel[off:]
        identvals = u[sel]
        if kind == 'val':
            keyvals = [wdb_sql._pyval(np.int64(x)) for x in identvals]
        else:
            keyvals = [wdb_sql._pyval(seg.fetch(k, int(x))) for x in identvals]
        cnts = counts[sel]
        rows = []
        for r in range(len(sel)):
            row = [None] * len(proj); row[ci] = int(cnts[r])
            for j in range(len(proj)):
                if j != ci: row[j] = keyvals[r]
            rows.append(tuple(row))
        _HITS += 1
        return rows, names

    idents = []; reps = []; cards = []; kinds = []
    for k in pc:
        if SP.n_exceptions(seg, k) is not None:
            arr = _m4_in_ranges(seg, k, los, his)[keep]; kind = 'val'
        else:
            arr = _codes_in_ranges(seg, k, los, his)[keep]; kind = 'code'
        u, inv = np.unique(arr.astype(np.int64), return_inverse=True)
        idents.append(inv); reps.append(u); cards.append(len(u)); kinds.append(kind)
    if not idents or len(idents[0]) == 0:
        _HITS += 1; return [], names
    mult = 1
    for cd in cards: mult *= cd
    if mult >= (1 << 62): return None                     # packing overflow risk -> decline
    gid = np.zeros(len(idents[0]), np.int64)
    for inv, cd in zip(idents, cards): gid = gid * cd + inv
    ug, counts = np.unique(gid, return_counts=True)

    order_idx = np.argsort(-counts, kind='stable')
    lim = wdb_sql._limit(tree); off = _offset(tree)
    sel = order_idx[off: off + lim] if lim is not None else order_idx[off:]

    rem = ug[sel].copy(); dense = [None] * len(pc); tmp = rem
    for i in range(len(pc) - 1, -1, -1):
        dense[i] = tmp % cards[i]; tmp = tmp // cards[i]
    name2vals = {}
    for i, physcol in enumerate(pc):
        identvals = reps[i][dense[i]]
        if kinds[i] == 'val':
            name2vals[keycols[i]] = [wdb_sql._pyval(np.int64(x)) for x in identvals]
        else:
            name2vals[keycols[i]] = [wdb_sql._pyval(seg.fetch(physcol, int(x))) for x in identvals]
    cnts = counts[sel]
    rows = []
    for r in range(len(sel)):
        row = [None] * len(proj)
        row[ci] = int(cnts[r])
        for j, p in enumerate(proj):
            if j == ci: continue
            nm = wdb_sql._proj_colname(p)
            row[j] = name2vals[nm][r]
        rows.append(tuple(row))
    _HITS += 1
    return rows, names


def try_compound(seg, tree, col_map):
    """Detect + execute, kept as the backward-compatible single-call entry."""
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)
