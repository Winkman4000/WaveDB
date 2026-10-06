"""wdb_subquery: uncorrelated subqueries in WHERE, by tree rewrite.

The composition pattern from set operations pointed inward: evaluate the inner SELECT through
the full pipeline, substitute its result into the outer tree -- a scalar comparison gains a
literal, IN (SELECT ...) becomes a literal IN-list -- then the OUTER query routes normally,
so the existing fast reads serve it (an IN-subquery on a dict column lands in wherescan's
code-set machinery untouched).

Semantics kept honest: a scalar subquery yielding no row compares as NULL (the predicate is
rewritten FALSE -- correct in any boolean context); more than one row raises, per the
standard. IN ignores NULLs in the inner result; NOT IN with any inner NULL yields no rows
(three-valued logic's famous trap). Correlated subqueries decline loudly for now -- the inner
run's unknown-column error is itself the detector.
"""
import numpy as np
import sqlglot.expressions as E

_MAX_IN = 2_000_000


def _samecol_codes(db, tree, node, sub):
    """C IN (SELECT C FROM sametable WHERE <simple conjuncts>): the inner's answer is a
    SET OF CODES -- scan the inner WHERE to positions, gather C's codes there, unique.
    No strings exist anywhere; the outer receives codes on the node ('_codes') and the
    query arg stays attached so every codes-unaware consumer declines exactly as before
    (fail closed, loud not wrong). Returns (codes, inner_has_null) or None."""
    import numpy as np
    import wdb_policies as P
    import wdb_wherescan as WS
    if not isinstance(node.this, E.Column):
        return None
    C = node.this.name
    inner = sub.this
    if not isinstance(inner, E.Select):
        return None
    exprs = inner.expressions
    if len(exprs) != 1 or not isinstance(exprs[0], E.Column) or exprs[0].name != C:
        return None
    if inner.args.get('group') or inner.args.get('limit') or inner.args.get('order'):
        return None
    itn, alias = _inner_tables(inner)
    f = tree.args.get('from_') or tree.args.get('from')
    if itn is None or f is None or not isinstance(f.this, E.Table) or f.this.name != itn:
        return None
    iw = inner.args.get('where')
    if iw is not None:
        quals = ({alias} if alias != itn else {itn}) if alias else {itn}
        for col in iw.find_all(E.Column):
            if col.table and col.table not in quals:
                return None                       # correlated: not our shape
    paths = db.cat.segment_paths(itn)
    if len(paths) != 1:
        return None
    seg = db.open_segment(paths[0], itn)
    if not P.no_deleted_rows(seg):
        return None
    cC = seg.cols.get(C)
    if cC is None or cC.get('mode') not in (0, 1, 2) or seg._effective(C) is not None:
        return None
    pos = None
    if iw is not None:
        conjs = []
        for cj in WS._conjuncts(iw.this):
            cl = WS._col_lit(cj)
            if cl is None:
                return None
            col2, val, op = cl[0], cl[1], cl[2]
            if op not in ('=', '<>'):
                return None
            c2 = seg.cols.get(col2)
            if c2 is None or c2.get('mode') not in (0, 1, 2) or seg._effective(col2) is not None:
                return None
            V2 = int(c2['V'])
            kc = WS._code_of(seg, col2, val)
            fl = np.zeros(V2, dtype=bool)
            if op == '=':
                if kc is None:
                    return np.empty(0, np.int64), False    # absent literal: empty inner set
                fl[kc] = True
            else:
                fl[:] = True
                if kc is not None:
                    fl[kc] = False
                if c2.get('has_null'):
                    fl[V2 - 1] = False                     # NULL <> lit is not TRUE
            conjs.append((0 if op == '=' else 1, col2, fl))
        # equality first (selective, cheap positions); every later conjunct filters
        # those positions by a code gather -- ascending stays ascending, and the
        # intersect1d whose hidden sort cost 0.87s at 100M never runs at all
        conjs.sort(key=lambda t: t[0])
        for _sel, col2, fl in conjs:
            if pos is None:
                pos = WS._scan_flag(seg, col2, fl, 0, seg.N)
            else:
                got = np.asarray(seg.codes_at(col2, pos)).astype(np.int64)
                pos = pos[fl[got]]
            if pos.size == 0:
                break
    if pos is None:
        cn = np.bincount(np.asarray(seg._raw_codes(C)), minlength=int(cC['V']))
        codes = np.flatnonzero(cn > 0).astype(np.int64)
    else:
        # the switchboard: flip a light per seen code, read the lit ones off in order.
        # flatnonzero of a presence board IS the sorted unique set -- no sort runs
        got = np.asarray(seg.codes_at(C, pos)).astype(np.int64)
        pres = np.zeros(int(cC['V']), dtype=bool)
        pres[got] = True
        codes = np.flatnonzero(pres).astype(np.int64)
    has_null = False
    if cC.get('has_null'):
        nullc = int(cC['V']) - 1
        if codes.size and int(codes[-1]) == nullc:
            has_null = True
            codes = codes[:-1]
    return codes, has_null


def rewrite_any_all(tree):
    """x > ALL (sub) -> x > (SELECT MAX(col) ...); x > ANY -> MIN; < ALL -> MIN;
    < ANY -> MAX; = ANY -> IN; <> ALL -> NOT IN. In-tree; returns the tree."""
    for nd in list(tree.find_all(E.All)) + list(tree.find_all(E.Any)):
        sub = nd.this
        if isinstance(sub, E.Select): sub = E.Subquery(this=sub)      # duckdb parse: ALL (SELECT ...) carries the Select bare
        if not isinstance(sub, E.Subquery) or not isinstance(sub.this, E.Select) or len(sub.this.expressions) != 1: continue
        cmp_ = nd.parent
        if not isinstance(cmp_, (E.GT, E.GTE, E.LT, E.LTE, E.EQ, E.NEQ)): continue
        is_all = isinstance(nd, E.All)
        t = type(cmp_).__name__
        if t in ('EQ', 'NEQ'):
            if (t == 'EQ' and not is_all) or (t == 'NEQ' and is_all):
                inn = E.In(this=cmp_.this.copy(), query=sub.copy())
                cmp_.replace(E.Not(this=inn) if t == 'NEQ' else inn)
            continue
        want = 'MAX' if ((t in ('GT', 'GTE')) == is_all) else 'MIN'
        inner = sub.this.copy()
        p0 = inner.expressions[0]
        p0i = p0.this if isinstance(p0, E.Alias) else p0
        agg = (E.Max if want == 'MAX' else E.Min)(this=p0i.copy())
        inner.set('expressions', [E.Alias(this=agg, alias=E.Identifier(this='__aa', quoted=False))])
        nd.replace(E.Subquery(this=inner))
    return tree


def substitute_select_scalars(db, tree):
    """Uncorrelated scalar subqueries in the SELECT list become literals."""
    itn = None
    for p in list(tree.expressions):
        for sub in list(p.find_all(E.Subquery)):
            inner = sub.this
            if not isinstance(inner, E.Select): continue
            inner_names = set()
            f = inner.args.get('from') or inner.args.get('from_')
            if f is None or not isinstance(f.this, E.Table): continue
            outer_alias = None
            of = tree.args.get('from') or tree.args.get('from_')
            if of is not None and isinstance(of.this, E.Table):
                outer_alias = of.this.alias or of.this.name
            corr = any(c.table and c.table not in (f.this.alias or f.this.name,) for c in inner.find_all(E.Column))
            if corr: continue
            rows = _run_inner(db, sub)
            if len(rows) != 1 or len(rows[0]) != 1:
                raise ValueError("scalar subquery returned %d rows" % len(rows))
            sub.replace(_lit(rows[0][0]))
    return tree


def has_subquery(tree):
    w = tree.args.get('where')
    if w is None:
        return False
    return (any(True for _ in w.find_all(E.Subquery))
            or any(True for _ in w.find_all(E.Exists)))


def _self_exists_codes(db, tree, inner, icol, ocol, rest):
    """THE SELF-EXISTS CODE SET: EXISTS (SELECT .. FROM T h2 WHERE h2.c = h1.c
    AND P(h2)) on the SAME table and column needs no road -- the set of
    c-codes among rows satisfying P IS the answer, already in the outer
    column's own code space (megaboard cq-notexists, 2026-09-04)."""
    try:
        tc, _al = _inner_tables(inner)
        if isinstance(tc, str):
            it = tc
        else:
            if len(tc) != 1: return None
            it = list(tc)[0]
        outer_tables = [t.name for t in tree.find_all(E.Table) if t.find_ancestor(E.Select) is tree]
        if len(set(outer_tables)) != 1 or list(set(outer_tables))[0] != it: return None
        if icol != ocol: return None
        import wdb_join as _J, wdb_sql as WS9
        seg, _sp = _J._solo_segment(db, it)
        pcol = db.cat.phys_map(it).get(icol, icol)
        if seg.cols.get(pcol, {}).get('mode') not in (0, 1, 2, 3, 4, 5): return None
        if rest:
            cond = rest[0]
            for r in rest[1:]:
                cond = E.And(this=cond, expression=r)
            cond = cond.copy()
            for c9 in list(cond.find_all(E.Column)):
                c9.set('table', None)                # inner alias -> bare column on the segment
            _pm9 = db.cat.phys_map(it)
            m = WS9._eval_pred(seg, cond, lambda nm: _pm9.get(nm, nm))
            if m is None: return None
            m = np.asarray(m, dtype=bool)
        else:
            m = None
        codes = np.asarray(seg.codes(pcol))
        sel = codes[m] if m is not None else codes
        return np.unique(sel).astype(np.int64)
    except Exception:
        return None


def _exists_road(db, tree, inner, icol, ocol, rest, outer_where=None, ex_node=None):
    """EXISTS over an FK road, executed as arrays: evaluate the residual
    child predicate with numpy (Column-vs-Column and Column-vs-literal
    conjuncts over typed dict values), scatter through the road sidecar,
    return unique qualifying OUTER codes. None on any shape doubt."""
    import time as _tt9, os as _os9
    _b9 = (lambda t0, tag: print('EXISTS-ROAD: %-18s %6.1fms' % (tag, (_tt9.perf_counter() - t0) * 1000), flush=True)) if _os9.environ.get('WDB_JOIN_BILL') else (lambda t0, tag: None)
    try:
        _t9 = _tt9.perf_counter()
        tc, _al = _inner_tables(inner)
        if tc is None:
            return None
        segp = db.cat.segment_paths(tc)[0]
        segc = db.open_segment(segp, tc)
        ot = None
        for t9 in tree.find_all(E.Table):
            if t9.name != tc and ocol in db.cat.column_names(t9.name):
                ot = t9.name
                break
        if ot is None:
            return None
        sego = db.open_segment(db.cat.segment_paths(ot)[0], ot)
        import wdb_join as _wj9                    # lazy: avoids import cycles
        ptr = _wj9._hash_pointer(db, tc, icol, segc, ot, ocol, sego)
        if ptr is None:
            try:
                db.create_fk_pointer(tc, icol, ot, ocol)
                ptr = db.fk_pointer(segp, icol)
            except Exception:
                ptr = None
            if ptr is None:
                return None
        _b9(_t9, 'setup+ptr'); _t9 = _tt9.perf_counter()
        def vals_of(colname):
            c9 = segc.cols.get(colname)
            if c9 is None or c9.get('has_null') or c9.get('dt') not in (0, 2, 3):
                return None
            td9 = np.asarray(segc._typed_dict(colname))
            if td9.dtype.kind not in 'if':
                return None
            if rows9s is not None:                 # only the lines of parents the window kept
                return td9[np.asarray(segc.codes_at(colname, rows9s))]
            return td9[np.asarray(segc.codes(colname))]
        rows9s = None
        m9 = None
        import wdb_sql as _ws9
        # QUARTER-FIRST (Jackson's original order): evaluate the outer's own
        # single-column literal conjuncts, gather through the road, and let
        # only lines of surviving parents into the scatter.
        okeep9 = None
        if outer_where is not None:
            import wdb_wherescan as _WS9
            for oc9 in _WS9._conjuncts(outer_where.this):
                if ex_node is not None and (oc9 is ex_node or any(True for _ in oc9.find_all(E.Exists)) or any(True for _ in oc9.find_all(E.In))):
                    continue
                t9o = type(oc9).__name__
                if t9o not in ('GT', 'GTE', 'LT', 'LTE', 'EQ'):
                    continue
                ocols9 = list(oc9.find_all(E.Column))
                if len(ocols9) != 1 or ocols9[0].name not in db.cat.column_names(ot):
                    continue
                co9v = sego.cols.get(ocols9[0].name)
                if co9v is None or co9v.get('has_null') or co9v.get('dt') not in (0, 2, 3):
                    continue
                tdo9 = np.asarray(sego._typed_dict(ocols9[0].name))
                if tdo9.dtype.kind not in 'if':
                    continue
                lito9 = oc9.expression if isinstance(oc9.this, E.Column) else oc9.this
                k9o = 'f' if co9v['dt'] == 2 else 'i'
                try:
                    v9o = _ws9._lit_for_col(sego, ocols9[0].name, lito9, k9o)
                except Exception:
                    continue
                if not isinstance(v9o, (int, float, np.integer, np.floating)):
                    continue
                if not isinstance(oc9.this, E.Column):
                    t9o = {'GT': 'LT', 'LT': 'GT', 'GTE': 'LTE', 'LTE': 'GTE', 'EQ': 'EQ'}[t9o]
                op9o = {'EQ': np.equal, 'GT': np.greater, 'GTE': np.greater_equal,
                        'LT': np.less, 'LTE': np.less_equal}[t9o]
                _vvc9 = locals().get('_vv_cache9')
                if _vvc9 is None:
                    _vv_cache9 = _vvc9 = {}
                vv9 = _vvc9.get(ocols9[0].name)
                if vv9 is None:
                    vv9 = _vvc9[ocols9[0].name] = tdo9[np.asarray(sego.codes(ocols9[0].name))]
                mo9 = op9o(vv9, v9o)
                okeep9 = mo9 if okeep9 is None else (okeep9 & mo9)
                _b9(_t9, 'okeep-conjunct'); _t9 = _tt9.perf_counter()
        if okeep9 is not None and rest:
            # THE FILTER FIRST (Jackson, 2026-10-05, Q4): the window already knows which parents count, so only
            # their lines are judged -- every other line's dates are never decoded. Q4 judged lateness for all
            # 60M lines (510 ms: two full code rebuilds, two full code->date gathers) and the window threw ~96%
            # of the verdicts away in the scatter.
            import wdb_kernels as _wk9s, wdb_engine as _we9s
            lk9s = np.empty(int(segc.N), dtype=np.bool_)
            _wk9s.plut_u8(np.ascontiguousarray(ptr), np.ascontiguousarray(okeep9, dtype=np.bool_), lk9s)
            rows9s = _we9s.Segment.mask_rows(lk9s)
            _b9(_t9, 'window-lines'); _t9 = _tt9.perf_counter()
        for cn in rest:
            cols = list(cn.find_all(E.Column))
            t9n = type(cn).__name__
            if t9n not in ('EQ', 'NEQ', 'GT', 'GTE', 'LT', 'LTE'):
                return None
            if len(cols) == 2 and isinstance(cn.this, E.Column) and isinstance(cn.expression, E.Column):
                nl9, nr9 = cn.this.name, cn.expression.name
                cl9, cr9 = segc.cols.get(nl9), segc.cols.get(nr9)
                if (cl9 is not None and cr9 is not None
                        and {cl9.get('code_enc'), cr9.get('code_enc')} == {15, 16}
                        and (cl9.get('e16_partner') in (nr9, None))
                        and (cr9.get('e16_partner') in (nl9, None))):
                    # THE BIT ANSWERS (the declared pair's whole purpose):
                    # bit means anchor <= partner; delta==0 means equal.
                    bit9, dl9 = segc.pair_bits(nl9)
                    if rows9s is not None:
                        bit9 = np.asarray(bit9)[rows9s]; dl9 = np.asarray(dl9)[rows9s]
                    _b9(_t9, 'pair_bits'); _t9 = _tt9.perf_counter()
                    lf9 = (cl9.get('code_enc') == 15)   # left is the anchor?
                    t9x = t9n if lf9 else {'GT': 'LT', 'LT': 'GT', 'GTE': 'LTE',
                                           'LTE': 'GTE', 'EQ': 'EQ', 'NEQ': 'NEQ'}[t9n]
                    if t9x == 'LTE':   c9m = bit9
                    elif t9x == 'GT':  c9m = ~bit9
                    elif t9x == 'LT':  c9m = bit9 & (dl9 > 0)
                    elif t9x == 'GTE': c9m = ~(bit9 & (dl9 > 0))
                    elif t9x == 'EQ':  c9m = (dl9 == 0)
                    else:              c9m = (dl9 > 0)
                    m9 = c9m if m9 is None else (m9 & c9m)
                    continue
                a9, b9 = vals_of(nl9), vals_of(nr9)
                if a9 is None or b9 is None:
                    return None
            elif len(cols) == 1:
                a9 = vals_of(cols[0].name)
                if a9 is None:
                    return None
                lit9 = cn.expression if isinstance(cn.this, E.Column) else cn.this
                kind9 = 'f' if segc.cols[cols[0].name]['dt'] == 2 else 'i'
                b9 = _ws9._lit_for_col(segc, cols[0].name, lit9, kind9)
                if not isinstance(b9, (int, float, np.integer, np.floating)):
                    return None
                if not isinstance(cn.this, E.Column):
                    t9n = {'GT': 'LT', 'LT': 'GT', 'GTE': 'LTE', 'LTE': 'GTE',
                           'EQ': 'EQ', 'NEQ': 'NEQ'}[t9n]
            else:
                return None
            op9 = {'EQ': np.equal, 'NEQ': np.not_equal, 'GT': np.greater,
                   'GTE': np.greater_equal, 'LT': np.less, 'LTE': np.less_equal}[t9n]
            c9m = op9(a9, b9)
            m9 = c9m if m9 is None else (m9 & c9m)
        if m9 is None:
            m9 = np.ones(int(segc.N), dtype=bool)
        if m9 is None:
            m9 = np.ones(int(segc.N), dtype=bool)
        _b9(_t9, 'child-mask-AND'); _t9 = _tt9.perf_counter()
        yes9 = np.zeros(int(sego.N), dtype=bool)
        if rows9s is not None:                     # the window is already in the rows
            yes9[np.asarray(ptr)[rows9s[m9]]] = True
        elif okeep9 is not None:
            import wdb_kernels as _wk9
            _wk9.exists_scatter(m9, np.asarray(ptr), okeep9, yes9)   # one fused pass
        else:
            yes9[np.asarray(ptr)[m9]] = True
        _b9(_t9, 'fused-scatter'); _t9 = _tt9.perf_counter()
        co9 = sego.cols.get(ocol)
        if co9 is None:
            return None
        if co9.get('mode') == 4:
            r9f = np.flatnonzero(yes9).astype(np.int64)
            _b9(_t9, 'codes-out')
            return r9f
        oc9 = np.asarray(sego.codes(ocol))
        kx9 = np.zeros(int(co9['V']) + 1, dtype=bool)
        kx9[oc9[yes9]] = True
        return np.flatnonzero(kx9).astype(np.int64)
    except Exception:
        if __import__('os').environ.get('WDB_JOIN_BILL'):
            import traceback
            traceback.print_exc()
        return None


def _road_codes(db, tree, node, rows):
    """Different-column IN in code space: inner VALUES -> outer dict codes.
    Numeric sorted outer dicts only; any None value falls back to the
    legacy path. Returns unique code array or None."""
    try:
        oc = node.this
        if not isinstance(oc, E.Column):
            return None
        ot = None
        for t9 in tree.find_all(E.Table):
            if oc.name in db.cat.column_names(t9.name):
                ot = t9.name
                break
        if ot is None:
            return None
        seg9 = db.open_segment(db.cat.segment_paths(ot)[0], ot)
        c9 = seg9.cols.get(oc.name)
        if c9 is None or c9.get('dt') not in (0, 2, 3) or c9.get('has_null'):
            return None
        td9 = np.asarray(seg9._typed_dict(oc.name))
        if td9.dtype.kind not in 'if' or not rows:
            return None
        if td9.size > 1 and not bool(np.all(td9[1:] >= td9[:-1])):
            return None                          # property, not mode: sorted dicts only
        vals9 = np.asarray(rows, dtype=td9.dtype).ravel()
        idx9 = np.searchsorted(td9, vals9)
        ok9 = (idx9 < td9.size)
        ok9 &= (td9[np.minimum(idx9, td9.size - 1)] == vals9)
        return np.unique(idx9[ok9]).astype(np.int64)
    except (TypeError, ValueError) as e9:
        if __import__('os').environ.get('WDB_JOIN_BILL'):
            print('ROAD-CODES declined (val):', str(e9)[:60], flush=True)
        return None
    except Exception as e9:
        if __import__('os').environ.get('WDB_JOIN_BILL'):
            import traceback; traceback.print_exc()
        return None


def _run_inner(db, sub):
    inner = sub.this
    tn, alias = _inner_tables(inner)
    if alias is not None:
        quals = {alias} if alias != tn else {tn}
        for col in inner.find_all(E.Column):
            if col.table and col.table not in quals:
                raise NotImplementedError(
                    "correlated subquery shape not supported (outer ref %s.%s)"
                    % (col.table, col.name))
    try:
        out = db.run(inner.sql())
    except (NotImplementedError, KeyError) as e:
        raise NotImplementedError(
            "correlated or unsupported subquery (inner query failed: %s)" % str(e)[:80])
    rows = out[0] if isinstance(out, tuple) else out
    return rows


def _lit(v):
    if v is None:
        return E.Null()
    if isinstance(v, bool):
        return E.Boolean(this=v)
    if isinstance(v, (int, float)):
        return E.Literal.number(v)
    return E.Literal.string(str(v))


def _inner_tables(sel):
    f = sel.args.get('from_') or sel.args.get('from')
    if f is None or not isinstance(f.this, E.Table):
        return None, None
    return f.this.name, (f.this.alias or f.this.name)


def _inner_colset(db, sel):
    try:
        _tn, _al = _inner_tables(sel)
        return set(db.cat.column_names(_tn)) if _tn else None
    except Exception:
        return None


def _corr_eq(sel, outer_names, inner_cols=None):
    """Split the inner WHERE into (the single outer-eq correlation, remaining conjuncts).
    Returns (inner_col, outer_col, rest) or None."""
    import wdb_wherescan as WS
    w = sel.args.get('where')
    if w is None:
        return None
    _tn, alias = _inner_tables(sel)
    if alias is None:
        return None
    # SQL scoping: an alias HIDES the table name inside the subquery, so with FROM hits h2,
    # a 'hits.'-qualified column refers to the OUTER query's hits
    inner_quals = {alias} if alias != _tn else {_tn}
    def is_outer(x):
        # Qualified: outer iff the qualifier isn't the inner table/alias.
        # Unqualified (the TPC-H idiom): SQL binds innermost-first, so a bare
        # name is outer exactly when the INNER table doesn't own it.
        if x.table:
            return x.table not in inner_quals
        return inner_cols is not None and x.name not in inner_cols
    corr, rest = None, []
    for c in WS._conjuncts(w.this):
        cols = list(c.find_all(E.Column))
        if not any(is_outer(x) for x in cols):
            rest.append(c)
            continue
        if (type(c).__name__ != 'EQ' or len(cols) != 2 or corr is not None
                or not isinstance(c.this, E.Column) or not isinstance(c.expression, E.Column)):
            return None
        a, b = c.this, c.expression
        if not is_outer(a) and is_outer(b):
            corr = (a.name, b.name)
        elif not is_outer(b) and is_outer(a):
            corr = (b.name, a.name)
        else:
            return None
    return None if corr is None else (corr[0], corr[1], rest)


def _try_window_decorrelate(db, tree):
    """x <cmp> (SELECT AGG(y) FROM <same table> t2 WHERE t2.g = outer.g) as the WHOLE WHERE
    -> the window read: QUALIFY x <cmp> AGG(y) OVER (PARTITION BY g). Self-joins become one
    placement."""
    w = tree.args.get('where')
    if w is None or tree.args.get('group') is not None:
        return None
    cmp_node = w.this
    if type(cmp_node).__name__ not in ('GT', 'GTE', 'LT', 'LTE'):
        return None
    a, b = cmp_node.this, cmp_node.expression
    flip = False
    if isinstance(a, E.Subquery):
        a, b, flip = b, a, True
    if not (isinstance(a, E.Column) and isinstance(b, E.Subquery)):
        return None
    inner = b.this
    f = tree.args.get('from_') or tree.args.get('from')
    itn, _al = _inner_tables(inner)
    if f is None or itn is None or f.this.name != itn:
        return None                              # v1: self-table only
    if len(inner.expressions) != 1:
        return None
    import wdb_sql
    _p0 = inner.expressions[0]
    _p0i = _p0.this if isinstance(_p0, E.Alias) else _p0
    _aggs9 = list(_p0i.find_all(E.AggFunc))
    if len(_aggs9) != 1 or _p0i.find(E.Subquery) is not None:
        return None
    _agg_node9 = _aggs9[0]
    ak = wdb_sql._agg_kind(_agg_node9)
    if ak is None or ak[0] not in ('SUM', 'AVG', 'MIN', 'MAX', 'COUNT_STAR'):
        return None
    ce = _corr_eq(inner, None, _inner_colset(db, inner))
    if ce is None or ce[2]:
        return None                              # v1: pure eq-correlation, no extra conds
    icol, ocol = ce[0], ce[1]
    if icol != ocol:
        return None                              # partition key must be the same column
    fname = {'SUM': 'SUM', 'AVG': 'AVG', 'MIN': 'MIN', 'MAX': 'MAX', 'COUNT_STAR': 'COUNT'}[ak[0]]
    # COUNT(*) over the correlated rows is COUNT(key) over the key's partition: the same count wherever the key is
    # set, and 0 in the NULL partition -- a NULL key matches no inner row, so its COUNT(*) is 0 (2026-10-06; it
    # was handed the NULL partition's size)
    arg = icol if ak[0] == 'COUNT_STAR' else ak[1]
    if _agg_node9 is _p0i:
        win_sql = f"{fname}({arg}) OVER (PARTITION BY {icol}) AS __corr0"
    else:                                        # arithmetic around the aggregate: MAX(v3) - 0.001
        _expr9 = _p0i.copy()
        import sqlglot as _sg
        _wn = _sg.parse_one(f"SELECT {fname}({arg}) OVER (PARTITION BY {icol}) FROM x").expressions[0]
        for _an in list(_expr9.find_all(E.AggFunc)):
            _an.replace(_wn.copy())
        win_sql = _expr9.sql() + " AS __corr0"
    new = tree.copy()
    # a row whose key is NULL matches no inner row (= never holds), so its subquery is NULL and the
    # comparison never true -- the window would have handed it the NULL partition's total (2026-10-05)
    # (COUNT(*) is 0 there, not NULL: that shape keeps every row and reads its 0 from COUNT(key) above)
    new.set('where', None if fname == 'COUNT' else E.Where(this=E.Not(this=E.Is(this=E.column(ocol), expression=E.Null()))))
    proj = list(new.expressions)
    import sqlglot
    proj.append(sqlglot.parse_one(f"SELECT {win_sql} FROM x").expressions[0])
    new.set('expressions', proj)
    op = {'GT': '>', 'GTE': '>=', 'LT': '<', 'LTE': '<='}[type(cmp_node).__name__]
    if flip:
        op = {'>': '<', '>=': '<=', '<': '>', '<=': '>='}[op]
    qual = sqlglot.parse_one(f"SELECT 1 FROM x QUALIFY {a.name} {op} __corr0")
    new.set('qualify', qual.args['qualify'])
    return new, len(proj) - 1                    # tree + index of the helper column to strip


def rewrite(db, tree):
    """Substitute every uncorrelated WHERE subquery; returns the rewritten tree."""
    w = tree.args.get('where')
    if w is None:
        return tree
    # EXISTS with an equality correlation -> semi-join as IN; NOT EXISTS -> null-safe NOT IN
    for ex in list(w.find_all(E.Exists)):
        inner = ex.this
        ce = _corr_eq(inner, None, _inner_colset(db, inner))
        if ce is None:
            raise NotImplementedError("EXISTS without a single eq-correlation")
        icol, ocol, rest = ce
        sub = inner.copy()
        sub.set('expressions', [E.column(icol)])
        if rest:
            r = rest[0].copy()
            for x in rest[1:]:
                r = E.And(this=r, expression=x.copy())
            sub.set('where', E.Where(this=r))
        else:
            sub.set('where', None)
        in_node = E.In(this=E.column(ocol), query=E.Subquery(this=sub))
        rd9 = _exists_road(db, tree, inner, icol, ocol, rest,
                           outer_where=tree.args.get('where'), ex_node=ex)
        if rd9 is None:
            rd9 = _self_exists_codes(db, tree, inner, icol, ocol, rest)
        if rd9 is not None:
            # THE SCATTER FORM (Jackson's Q4 walk): child residual mask in
            # arrays -> road sidecar -> unique PARENT ROWS -> the _codes
            # sentinel directly. No SQL execution of the inner, no values,
            # no rows -- and a mode-4 outer key means parent rows ARE codes.
            in_node.set('_codes', rd9)
            in_node.set('expressions', [E.Subquery(this=sub)])
        in_node.set('_exists_rewrite', True)   # NOT EXISTS drops inner NULLs; it does NOT
                                               # inherit NOT IN's null-poisoning rule
        negated = isinstance(ex.parent, E.Not)
        if negated:
            isnull = E.Is(this=E.column(ocol), expression=E.Null())
            ex.parent.replace(E.Paren(this=E.Or(this=E.Not(this=in_node), expression=isnull)))
        else:
            ex.replace(in_node)
    # IN (SELECT ...) -- handle In nodes carrying a query arg (NOT IN arrives as Not(In))
    for node in list(w.find_all(E.In)):
        sub = node.args.get('query')
        if sub is None:
            continue
        if node.args.get('_codes') is not None:
            continue                              # the scatter form already answered
        inner = sub.this
        if (isinstance(inner, E.Select) and not inner.args.get('distinct')
                and not inner.args.get('group') and not inner.args.get('limit')
                and len(inner.expressions) == 1
                and isinstance(inner.expressions[0], E.Column)):
            inner.set('distinct', E.Distinct())   # IN cares about the SET: dedup in the
                                                  # engine's code space, not python-side
        sc = _samecol_codes(db, tree, node, sub)
        if sc is not None:
            codes, in_null = sc
            negated = isinstance(node.parent, E.Not)
            if negated and in_null and not node.args.get('_exists_rewrite'):
                node.parent.replace(E.false())    # NOT IN with an inner NULL: no row qualifies
                continue
            if codes.size == 0:
                (node.parent if negated else node).replace(
                    E.true() if negated else E.false())
                continue
            node.set('_codes', codes)             # query arg stays AND the subquery node
            node.set('expressions', [sub])        # sits in expressions as a sentinel: any
            continue                              # literal-parser chokes and declines --
                                                  # empty expressions would silently match
                                                  # nothing (fail closed, loud not wrong)
        rows = _run_inner(db, sub)
        if rows and len(rows[0]) != 1:
            raise ValueError("IN subquery must return one column")
        # THE ROAD FORM (Jackson's Q4 walk, array space end to end): a
        # different-column IN resolves to the OUTER column's dict CODES --
        # searchsorted into the outer dict, unique, then the same _codes
        # sentinel the same-column path uses. No value lists, no cap: the
        # downstream LUT is V+1 bools however many million values matched.
        rc9 = _road_codes(db, tree, node, rows)
        if rc9 is not None:
            codes9 = rc9
            negated = isinstance(node.parent, E.Not)
            if codes9.size == 0:
                (node.parent if negated else node).replace(
                    E.true() if negated else E.false())
                continue
            node.set('_codes', codes9)
            node.set('expressions', [sub])
            continue
        vals = [r[0] for r in rows]
        has_null = any(v is None for v in vals)
        vals = list(dict.fromkeys(v for v in vals if v is not None))
        if len(vals) > _MAX_IN:
            raise NotImplementedError("IN subquery result too large (%d values)" % len(vals))
        negated = isinstance(node.parent, E.Not)
        if negated and has_null and not node.args.get('_exists_rewrite'):
            target = node.parent
            target.replace(E.false())        # NOT IN with an inner NULL: no row qualifies
            continue
        if not vals:
            (node.parent if negated else node).replace(
                E.true() if negated else E.false())   # empty set: IN -> false, NOT IN -> true
            continue
        node.set('query', None)
        node.set('expressions', [_lit(v) for v in vals])
    # scalar subqueries in comparisons
    for sub in list(w.find_all(E.Subquery)):
        parent = sub.parent
        if parent is None or isinstance(parent, E.In):
            continue
        rows = _run_inner(db, sub)
        if len(rows) > 1:
            raise ValueError("scalar subquery returned %d rows" % len(rows))
        if rows and len(rows[0]) != 1:
            raise ValueError("scalar subquery must return one column")
        v = rows[0][0] if rows else None
        if v is None:
            parent.replace(E.false())        # NULL comparison: unknown -> row filtered
        else:
            sub.replace(_lit(v))
    return tree
