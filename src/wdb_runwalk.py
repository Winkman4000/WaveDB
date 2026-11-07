"""wdb_runwalk: THE RUN WALK -- a correlated [NOT] EXISTS answered by walking the child's run per parent.

    SELECT ... FROM parent WHERE <parent's own filters> AND [NOT] EXISTS
        (SELECT * FROM child WHERE child.fk = parent.key AND <child's own filters>)

When the child is stored in order of its key to the parent (the road child -> parent never decreases),
each parent's children are ONE contiguous run. Then:
  1. the shape (_spec): one outer table, the correlation is one equality child.fk = parent.key, every other
     inner conjunct is the child's own (column vs column, column vs literal), numeric dictionary columns;
  2. the parent survivors (_parent_keep): the outer's own one-column literal filters judged over each
     column's distinct values, then one keep per parent row through its codes;
  3. the walk (_walk): for each surviving parent row, binary-search its first child in the rising road, walk
     its run, stop at the first child passing the child filters (EXISTS needs one); NOT EXISTS keeps the
     parents where none passes;
  4. the verdict: one bool per parent row (survivor AND [not] found) registered as a WDB_ROWMASK placeholder
     (wdb_groupexists' query-scoped registry), so the existing aggregation reads it as any other filter;
     the outer filters the walk spent leave the WHERE.
Nothing is kept between queries: no run starts, no masks (the registry dies with the query).
A shape outside this leaves the tree untouched. Switch WDB_RUN_WALK (default on).

Lore: Jackson, 2026-11-04, for TPC-H Q4 (lineitem stored by l_orderkey)."""
import os
import numpy as np
from numba import njit, prange
from sqlglot import expressions as E

import wdb_groupexists as GE

_RUN_WALK = [os.environ.get('WDB_RUN_WALK', '1') == '1']
_HITS = [0]
_MAX_TERMS = 3
# the op convention of wdb_kernels.pcmp_dict2: 0 '=', 1 '<>', 2 '>', 3 '<', 4 '>=', 5 '<='
_OPS = {E.EQ: 0, E.NEQ: 1, E.GT: 2, E.LT: 3, E.GTE: 4, E.LTE: 5}
_FLIP = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4}          # literal OP column -> column OP' literal
_NP = {0: np.equal, 1: np.not_equal, 2: np.greater, 3: np.less, 4: np.greater_equal, 5: np.less_equal}
_SEQ = [0]


@njit(inline='always')
def _cmp(x, y, op):
    if op == 0: return x == y
    if op == 1: return x != y
    if op == 2: return x > y
    if op == 3: return x < y
    if op == 4: return x >= y
    return x <= y


@njit(inline='always')
def _term(i, ac, av, bc, bv, lit, isl, op):
    a = av[ac[i]]
    b = lit if isl else bv[bc[i]]
    return _cmp(a, b, op)


@njit(parallel=True, cache=True, nogil=True)
def _runs(keep, road, NT):
    """Step 3a. The surviving parents (in row order) and their runs of children: binary search in the rising
    road for each one's first child, then the run's length. Returns (pids, off, rows): parent pids[k]'s children
    are rows[off[k]:off[k+1]] (child row numbers, ascending overall)."""
    npar = keep.shape[0]; n = road.shape[0]
    chunk = (npar + NT - 1) // NT
    cnt = np.zeros(NT, np.int64)
    for t in prange(NT):                                   # survivors per chunk
        lo = t * chunk; hi = min(lo + chunk, npar)
        c = 0
        for r in range(lo, hi):
            if keep[r]: c += 1
        cnt[t] = c
    base = np.zeros(NT + 1, np.int64)
    for t in range(NT): base[t + 1] = base[t] + cnt[t]
    K = base[NT]
    pids = np.empty(K, np.int64); first = np.empty(K, np.int64); ln = np.empty(K, np.int64)
    for t in prange(NT):
        lo = t * chunk; hi = min(lo + chunk, npar)
        k = base[t]
        for r in range(lo, hi):
            if not keep[r]: continue
            a = 0; b = n
            while a < b:
                m = (a + b) >> 1
                if road[m] < r: a = m + 1
                else: b = m
            e = a
            while e < n and road[e] == r: e += 1
            pids[k] = r; first[k] = a; ln[k] = e - a
            k += 1
    return pids, first, ln


@njit(parallel=True, cache=True, nogil=True)
def _rows_of(first, ln, NT):
    """Step 3a'. The runs' child rows, concatenated (ascending): parent k's children are rows[off[k]:off[k]+ln[k]]"""
    K = first.shape[0]
    off = np.zeros(K + 1, np.int64)
    for k in range(K): off[k + 1] = off[k] + ln[k]
    rows = np.empty(off[K], np.int64)
    kch = (K + NT - 1) // NT
    for t in prange(NT):
        for k in range(t * kch, min((t + 1) * kch, K)):
            f = first[k]; o = off[k]
            for j in range(ln[k]): rows[o + j] = f + j
    return off[:K].copy(), rows


@njit(parallel=True, cache=True, nogil=True)
def _walk(pids, start, ln, npar, nt,
          a0c, a0v, b0c, b0v, l0, s0, o0,
          a1c, a1v, b1c, b1v, l1, s1, o1,
          a2c, a2v, b2c, b2v, l2, s2, o2,
          neg, NT):
    """Step 3b. out[pids[k]] = (some child in its slice passes all nt terms) -- or, when neg, none does.
    Parent k's children sit at term index start[k] .. start[k] + ln[k] - 1: the run rows themselves when the
    term codes are the whole columns, or the slice of the gathered codes when only the run rows were decoded.
    Each parent stops at its first passing child."""
    out = np.zeros(npar, np.bool_)
    K = pids.shape[0]
    kch = (K + NT - 1) // NT
    for t in prange(NT):
        for k in range(t * kch, min((t + 1) * kch, K)):
            found = False
            for j in range(start[k], start[k] + ln[k]):
                ok = True
                if nt > 0 and not _term(j, a0c, a0v, b0c, b0v, l0, s0, o0): ok = False
                if ok and nt > 1 and not _term(j, a1c, a1v, b1c, b1v, l1, s1, o1): ok = False
                if ok and nt > 2 and not _term(j, a2c, a2v, b2c, b2v, l2, s2, o2): ok = False
                if ok:
                    found = True
                    break
            out[pids[k]] = (not found) if neg else found
    return out


def _flat(x):
    if isinstance(x, E.Paren): return _flat(x.this)
    if isinstance(x, E.And): return _flat(x.this) + _flat(x.expression)
    return [x]


def _numcol(seg, pc):
    """a column the walk may read: dictionary-coded number, no NULLs, no overrides"""
    c = seg.cols.get(pc)
    if c is None or c.get('has_null') or c.get('dt') not in (0, 2, 3) or c.get('mode') not in (0, 1, 2):
        return None
    try:
        if seg._overrides(pc) is not None: return None
    except Exception:
        return None
    td = np.asarray(seg._typed_dict(pc))
    if td.dtype.kind not in 'iuf': return None
    return np.ascontiguousarray(td, dtype=np.float64)


def _lit_num(node):
    if isinstance(node, E.Neg) and isinstance(node.this, E.Literal) and not node.this.is_string:
        v = _lit_num(node.this); return None if v is None else -v
    if isinstance(node, E.Literal) and not node.is_string:
        try: return float(node.this)
        except Exception: return None
    return None


def _spec(db, tree, ex, neg, ptab, palias, pcols):
    """Step 1. The EXISTS node's shape -> (child table, fk, parent key, child terms) or None."""
    sub = ex.this
    if isinstance(sub, E.Subquery): sub = sub.this
    if not isinstance(sub, E.Select): return None
    for k9 in ('joins', 'group', 'having', 'limit', 'offset', 'with', 'laterals'):
        if sub.args.get(k9): return None
    frm = sub.args.get('from') or sub.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table): return None
    ww = sub.args.get('where')
    if ww is None: return None
    ct, ca = frm.this.name, (frm.this.alias or frm.this.name)
    if ct == ptab: return None                        # a self-reference is wdb_groupexists' shape
    ccols = set(db.cat.column_names(ct))

    def side(c):                                       # SQL scoping: the inner table answers first
        if c.table:
            if c.table == ca: return ('c', c.name) if c.name in ccols else None
            if c.table == palias: return ('p', c.name) if c.name in pcols else None
            return None
        if c.name in ccols: return ('c', c.name)
        if c.name in pcols: return ('p', c.name)
        return None

    corr, terms = None, []
    for cj in _flat(ww.this):
        cols = list(cj.find_all(E.Column))
        sides = [side(c) for c in cols]
        if not cols or any(s is None for s in sides): return None
        if type(cj) not in _OPS: return None
        if any(s[0] == 'p' for s in sides):
            if not (isinstance(cj, E.EQ) and isinstance(cj.this, E.Column) and isinstance(cj.expression, E.Column)):
                return None                              # any other use of an outer column: not this shape
            a, b = side(cj.this), side(cj.expression)
            if a[0] == b[0] or corr is not None: return None
            corr = (a[1], b[1]) if a[0] == 'c' else (b[1], a[1])   # (child fk, parent key)
            continue
        op = _OPS[type(cj)]
        if isinstance(cj.this, E.Column) and isinstance(cj.expression, E.Column):
            terms.append(('cc', cj.this.name, cj.expression.name, op))
        elif isinstance(cj.this, E.Column) and _lit_num(cj.expression) is not None:
            terms.append(('cl', cj.this.name, _lit_num(cj.expression), op))
        elif isinstance(cj.expression, E.Column) and _lit_num(cj.this) is not None:
            terms.append(('cl', cj.expression.name, _lit_num(cj.this), _FLIP[op]))
        else:
            return None
    if corr is None or len(terms) > _MAX_TERMS: return None
    return ct, corr[0], corr[1], terms


_STR_V_MAX = 1 << 16


def _string_keep(seg, pm, cj):
    """(physical column, keep per code) for `col = 'x'`, `col <> 'x'`, `col IN (...)` (strings or numbers) over a
    dictionary column with at most _STR_V_MAX distinct values and no NULLs; None otherwise"""
    neg = False
    if isinstance(cj, E.Not) and isinstance(cj.this, E.In):
        cj, neg = cj.this, True
    if isinstance(cj, (E.EQ, E.NEQ)):
        col, lits = (cj.this, [cj.expression]) if isinstance(cj.this, E.Column) else (cj.expression, [cj.this])
        neg = neg or isinstance(cj, E.NEQ)
    elif isinstance(cj, E.In) and not cj.args.get('query') and cj.expressions:
        col, lits = cj.this, list(cj.expressions)
    else:
        return None
    if not isinstance(col, E.Column) or not all(isinstance(l, E.Literal) for l in lits): return None
    pc = pm.get(col.name, col.name)
    c = seg.cols.get(pc)
    if c is None or c.get('has_null') or c.get('mode') not in (0, 1, 2) or int(c.get('V') or 0) > _STR_V_MAX:
        return None
    try:
        if seg._overrides(pc) is not None: return None
    except Exception:
        return None
    td = seg._typed_dict(pc)
    if c.get('dt') == 1:
        if not all(l.is_string for l in lits): return None
        want = {str(l.this) for l in lits}
        k = np.array([(v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else str(v)) in want
                      for v in td], dtype=np.bool_)
    elif isinstance(cj, E.In):
        nums = [_lit_num(l) for l in lits]
        if any(v is None for v in nums): return None
        k = np.isin(np.asarray(td, dtype=np.float64), np.asarray(nums, dtype=np.float64))
    else:
        return None                                       # numeric = / <> go through the numeric judge
    return pc, (~k if neg else k)


def _parent_keep(db, segp, ptab, cjs):
    """Step 2. The outer's own one-column literal filters, judged over each column's distinct values; one keep
    per parent row through its codes. Returns (keep or None when nothing was spent, spent conjuncts)."""
    import wdb_kernels as WK
    pm = db.cat.phys_map(ptab)
    lut_of, spent = {}, []
    for cj in cjs:
        k = _string_keep(segp, pm, cj)
        if k is not None:                                  # a string = / <> / IN, judged over its distinct values
            pc, kk = k
            lut_of[pc] = kk if pc not in lut_of else (lut_of[pc] & kk)
            spent.append(cj)
            continue
        if type(cj) not in _OPS: continue
        if isinstance(cj.this, E.Column) and _lit_num(cj.expression) is not None:
            col, lit, op = cj.this, _lit_num(cj.expression), _OPS[type(cj)]
        elif isinstance(cj.expression, E.Column) and _lit_num(cj.this) is not None:
            col, lit, op = cj.expression, _lit_num(cj.this), _FLIP[_OPS[type(cj)]]
        else:
            continue
        pc = pm.get(col.name, col.name)
        td = _numcol(segp, pc)
        if td is None: continue
        k = _NP[op](td, lit)
        lut_of[pc] = k if pc not in lut_of else (lut_of[pc] & k)   # same column: AND at dictionary scale
        spent.append(cj)
    if not lut_of: return None, []
    keep = None
    for pc, lut in lut_of.items():
        codes = np.ascontiguousarray(segp.codes(pc))
        out = np.empty(int(segp.N), np.bool_)
        WK.plut_u8(codes, np.ascontiguousarray(lut, dtype=np.bool_), out)
        keep = out if keep is None else (keep & out)
    return keep, spent


def _term_args(segc, pm, term, rows, memo):
    """one child term as kernel arguments: each side's codes at the run rows only (rows None: the whole column,
    when the runs cover most of the table) and its values; the literal, is-literal, op. memo: a column read twice
    is decoded once."""
    dummy_c = np.zeros(1, np.uint8); dummy_v = np.zeros(1, np.float64)
    def side(nm):
        pc = pm.get(nm, nm)
        if pc not in memo:
            v = _numcol(segc, pc)
            if v is None:
                memo[pc] = None
            else:
                cd = segc.codes(pc) if rows is None else segc.codes_at(pc, rows)
                memo[pc] = (np.ascontiguousarray(cd), v)
        return memo[pc]
    kind, a, b, op = term
    sa = side(a)
    if sa is None: return None
    if kind == 'cl':
        return (sa[0], sa[1], dummy_c, dummy_v, float(b), True, op)
    sb = side(b)
    if sb is None: return None
    return (sa[0], sa[1], sb[0], sb[1], 0.0, False, op)


def rewrite(db, tree):
    """Steps 1-4 over the outer WHERE's top-level [NOT] EXISTS conjuncts; returns how many were replaced."""
    if not _RUN_WALK[0]: return 0
    w = tree.args.get('where')
    if w is None or tree.find(E.Exists) is None: return 0
    if tree.args.get('joins'): return 0                  # step 1: the outer query is one table
    frm = tree.args.get('from') or tree.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table): return 0
    ptab, palias = frm.this.name, (frm.this.alias or frm.this.name)
    try:
        pcols = set(db.cat.column_names(ptab))
        import wdb_join as WJ, wdb_kernels as WK
        segp, _ = WJ._solo_segment(db, ptab)
    except Exception:
        return 0
    cjs = _flat(w.this)
    exs = []
    for cj in cjs:
        neg = isinstance(cj, E.Not) and isinstance(cj.this, E.Exists)
        ex = cj.this if neg else cj
        if isinstance(ex, E.Exists):
            try:
                sp = _spec(db, tree, ex, neg, ptab, palias, pcols)
            except Exception:
                sp = None
            if sp is not None: exs.append((cj, neg, sp))
    if not exs: return 0
    try:
        own = [cj for cj in cjs if not any(cj is e[0] for e in exs) and cj.find(E.Exists) is None
               and cj.find(E.Select) is None]
        keep, spent = _parent_keep(db, segp, ptab, own)
        if keep is None: keep = np.ones(int(segp.N), np.bool_)
        reps = {}
        for cj, neg, (ct, fk, pkey, terms) in exs:
            segc, _ = WJ._solo_segment(db, ct)
            pmc = db.cat.phys_map(ct)
            road = WJ._hash_pointer(db, ct, fk, segc, ptab, pkey, segp)
            if road is None: continue
            road = np.ascontiguousarray(road)
            if road.dtype.kind not in 'iu' or road.shape[0] != int(segc.N) or not WK._road_rises_nb(road, 256):
                continue                                  # the child is not stored by its key: not this road
            if any(_numcol(segc, pmc.get(nm, nm)) is None for tm in terms
                   for nm in ((tm[1], tm[2]) if tm[0] == 'cc' else (tm[1],))):
                continue                                  # decide before paying: every term column readable
            pids, first, ln = _runs(keep, road, 16)
            if 2 * int(ln.sum()) >= int(segc.N):
                rows, start = None, first                 # most of the child: decode the columns whole
            else:
                start, rows = _rows_of(first, ln, 16)     # only the runs' rows are decoded
            args, memo = [], {}
            for tm in terms:
                ta = _term_args(segc, pmc, tm, rows, memo)
                if ta is None: break
                args.append(ta)
            if len(args) != len(terms): continue
            dummy = (np.zeros(1, np.uint8), np.zeros(1, np.float64), np.zeros(1, np.uint8), np.zeros(1, np.float64), 0.0, True, 0)
            while len(args) < _MAX_TERMS: args.append(dummy)
            flat = [x for ta in args for x in ta]
            verdict = _walk(pids, start, ln, int(segp.N), len(terms), *flat, bool(neg), 16)
            _SEQ[0] += 1
            key = 'w%d' % _SEQ[0]
            GE._MASKS[key] = (verdict, float(np.count_nonzero(verdict)) / verdict.size if verdict.size else 0.0)
            reps[id(cj)] = E.Anonymous(this=GE.MARK, expressions=[
                E.Column(this=E.Identifier(this=pkey, quoted=False), table=E.Identifier(this=palias, quoted=False)),
                E.Literal.string(key)])
            _HITS[0] += 1
        if not reps: return 0
        gone = {id(s) for s in spent}                     # the verdicts already hold the spent filters
        out = [reps.get(id(cj), cj) for cj in cjs if id(cj) not in gone]
        w.set('this', GE._and(out))
        return len(reps)
    except Exception:
        if os.environ.get('WDB_RUN_WALK_DEBUG'):
            import traceback; traceback.print_exc()
        return 0
