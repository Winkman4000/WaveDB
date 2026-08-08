"""wdb_setops: UNION / UNION ALL / INTERSECT / EXCEPT as controller-level composition.

Each side runs through the full existing pipeline (any read, any table -- sides recurse, so
chains compose), then rows combine by SQL bag/set semantics: UNION ALL concatenates, UNION
dedupes (first-seen order), INTERSECT and EXCEPT operate on distinct rows per the standard.
Outer ORDER BY / LIMIT / OFFSET apply to the combined result. Headers come from the left side;
arity mismatches raise loudly (never silently truncate).
"""
import sqlglot.expressions as E

_SETOPS = (E.Union, E.Intersect, E.Except)
_HITS = 0


def is_setop(tree):
    return isinstance(tree, _SETOPS)


def _norm_row(r):
    out = []
    for v in r:
        if hasattr(v, 'item'):
            v = v.item()
        if isinstance(v, (bytes, bytearray)):
            v = v.decode('utf-8', 'replace')
        out.append(v)
    return tuple(out)


def _codes_side(db, node):
    """One union side as lights on a presence board: SELECT C FROM t WHERE <simple
    =/<> conjuncts> -> bool[V] of C-codes present. None = not our shape (fall back)."""
    import numpy as np
    import wdb_policies as P
    import wdb_wherescan as WS
    if not isinstance(node, E.Select):
        return None
    if (node.args.get('group') or node.args.get('limit') or node.args.get('order')
            or node.args.get('joins') or node.args.get('having')):
        return None
    exprs = node.expressions
    if len(exprs) != 1 or not isinstance(exprs[0], E.Column):
        return None
    C = exprs[0].name
    f = node.args.get('from_') or node.args.get('from')
    if f is None or not isinstance(f.this, E.Table):
        return None
    tn = f.this.name
    try:
        paths = db.cat.segment_paths(tn)
    except Exception:
        return None
    if len(paths) != 1:
        return None
    seg = db.open_segment(paths[0], tn)
    if not P.no_deleted_rows(seg):
        return None
    cC = seg.cols.get(C)
    if (cC is None or cC.get('mode') not in (0, 1, 2) or cC.get('dt') == 3
            or seg._effective(C) is not None):
        return None
    V = int(cC['V'])
    w = node.args.get('where')
    pos = None
    if w is not None:
        conjs = []
        for cj in WS._conjuncts(w.this):
            cl = WS._col_lit(cj)
            if cl is None:
                return None
            col2, val, op = cl[0], cl[1], cl[2]
            if op not in ('=', '<>'):
                return None
            c2 = seg.cols.get(col2)
            if (c2 is None or c2.get('mode') not in (0, 1, 2)
                    or seg._effective(col2) is not None):
                return None
            V2 = int(c2['V'])
            kc = WS._code_of(seg, col2, val)
            fl = np.zeros(V2, dtype=bool)
            if op == '=':
                if kc is None:
                    return seg, C, np.zeros(V, dtype=bool)   # absent literal: empty side
                fl[kc] = True
            else:
                fl[:] = True
                if kc is not None:
                    fl[kc] = False
                if c2.get('has_null'):
                    fl[V2 - 1] = False                       # NULL <> lit is not TRUE
            conjs.append((0 if op == '=' else 1, col2, fl))
        conjs.sort(key=lambda t: t[0])           # equality first; later conjuncts filter
        for _sel, col2, fl in conjs:             # positions by code gather -- no intersect
            if pos is None:
                pos = WS._scan_flag(seg, col2, fl, 0, seg.N)
            else:
                got = np.asarray(seg.codes_at(col2, pos)).astype(np.int64)
                pos = pos[fl[got]]
            if pos.size == 0:
                return seg, C, np.zeros(V, dtype=bool)
    pres = np.zeros(V, dtype=bool)
    if pos is None:
        pres = np.bincount(np.asarray(seg._raw_codes(C)), minlength=V) > 0
    else:
        got = np.asarray(seg.codes_at(C, pos)).astype(np.int64)
        pres[got] = True
    return seg, C, pres


def _union_codes(db, node):
    """SELECT C FROM t WHERE ... UNION SELECT C FROM t WHERE ...: the switchboard.
    Each side flips lights on a V-sized presence board (positions -> codes; no python
    row ever exists), union is OR, and only the lit codes decode -- once each. This
    replaces two general-path executions that boxed 738K python values to dedup them."""
    import numpy as np
    import wdb_sql
    a = _codes_side(db, node.this)
    if a is None:
        return None
    b = _codes_side(db, node.expression)
    if b is None:
        return None
    sa, ca, pa = a
    sb, cb, pb = b
    if sa is not sb or ca != cb:
        return None                              # different code spaces: not our shape
    seg, C = sa, ca
    pres = pa | pb
    cC = seg.cols[C]
    has_null = bool(cC.get('has_null')) and bool(pres[int(cC['V']) - 1])
    if cC.get('has_null'):
        pres[int(cC['V']) - 1] = False
    rows = []
    for code in np.flatnonzero(pres):
        v = wdb_sql._pyval(seg.fetch(C, int(code)))
        if isinstance(v, (bytes, bytearray)):
            v = v.decode('utf-8', 'replace')
        rows.append((v,))
    if has_null:
        rows.append((None,))
    return rows, [C]


def _try_fused_union(db, node):
    """One motion (Jackson's order): UNION ALL branches that differ only by one
    eq-literal on a shared column walk the books ONCE. Shared reads, shared residual
    mask, one bincount per branch, per-branch tally sheets kept separate as UNION ALL
    demands. v1 gate: single table, single group key, COUNT(*) only, same residuals."""
    import numpy as np
    import wdb_sql, sqlglot.expressions as E2
    import wdb_wherescan as WS
    import wdb_policies as P
    leaves = []
    def walk(n):
        if isinstance(n, E2.Union):
            if n.args.get('distinct'):
                return False
            return walk(n.this) and walk(n.expression)
        if isinstance(n, E2.Select):
            leaves.append(n)
            return True
        return False
    if not walk(node) or len(leaves) < 2:
        return None
    sig = None; lits = []
    for t in leaves:
        if t.args.get('joins') or t.args.get('having') or t.args.get('qualify') \
                or t.args.get('order') or wdb_sql._limit(t) is not None:
            return None
        g = t.args.get('group')
        if g is None or len(g.expressions) != 1 or not isinstance(g.expressions[0], E2.Column):
            return None
        gcol = g.expressions[0].name
        proj = t.expressions
        if len(proj) != 2:
            return None
        ks = [wdb_sql._agg_kind(p) for p in proj]
        if sorted(x[0] if x else 'COL' for x in (ks[0], ks[1]) if True) != ['COL', 'COUNT_STAR']:
            if not ((ks[0] is None and ks[1] and ks[1][0] == 'COUNT_STAR')
                    or (ks[1] is None and ks[0] and ks[0][0] == 'COUNT_STAR')):
                return None
        ci = 0 if (ks[0] and ks[0][0] == 'COUNT_STAR') else 1
        pk = proj[1 - ci]
        if wdb_sql._proj_colname(pk) != gcol:
            return None
        tbl = t.args.get('from_') or t.args.get('from')
        tn = tbl.this.name if tbl is not None and isinstance(tbl.this, E2.Table) else None
        if tn is None:
            return None
        w = t.args.get('where')
        if w is None:
            return None
        conjs = list(w.this.flatten()) if isinstance(w.this, E2.And) else [w.this]
        eqs = [c for c in conjs
               if isinstance(c, E2.EQ) and isinstance(c.this, E2.Column)
               and c.expression.is_string or
               (isinstance(c, E2.EQ) and isinstance(c.this, E2.Column)
                and isinstance(c.expression, E2.Literal))]
        eqs = [c for c in conjs if isinstance(c, E2.EQ) and isinstance(c.this, E2.Column)
               and isinstance(c.expression, E2.Literal)]
        if len(eqs) != 1:
            return None
        kcol = eqs[0].this.name
        lit = eqs[0].expression.this
        resid = sorted(c.sql() for c in conjs if c is not eqs[0])
        this_sig = (tn, gcol, kcol, tuple(resid), ci,
                    tuple(wdb_sql._alias(p) for p in proj))
        if sig is None:
            sig = this_sig
        elif this_sig != sig:
            return None
        lits.append(lit)
    tn, gcol, kcol, resid_sql, ci, hdr = sig
    paths = db.cat.segment_paths(tn)
    if len(paths) != 1:
        return None
    seg = db.open_segment(paths[0], tn)
    for nm in (gcol, kcol):
        c = seg.cols.get(nm)
        if c is None or c.get('mode') not in (0, 1, 2) or c.get('has_null'):
            return None
    if not P.no_deleted_rows(seg):
        return None
    kcodes = [WS._code_of(seg, kcol, v) for v in lits]
    gc = None if seg.cols[gcol].get('code_enc') in (8, 9) else \
        np.asarray(seg._raw_codes(gcol))     # e8 counts in literal space: no dense gc
    kc = np.asarray(seg._raw_codes(kcol))    # native widths (the int64 casts were 153ms)
    resid_mask = None
    zero_gcodes = []                             # NEQ literals ON THE GROUP COL: no row
    eq_only_gcode = None                         # mask at all -- bincount everything and
    resid_left = []                              # surgically zero (or isolate) the slot
    import sqlglot
    for rs in resid_sql:
        pred = sqlglot.parse_one(rs, read='duckdb')
        if isinstance(pred, (E2.EQ, E2.NEQ)) and isinstance(pred.this, E2.Column) \
                and pred.this.name == gcol and isinstance(pred.expression, E2.Literal):
            code2 = WS._code_of(seg, gcol, pred.expression.this)
            if isinstance(pred, E2.NEQ):
                if code2 is not None:
                    zero_gcodes.append(int(code2))
                continue
            eq_only_gcode = -1 if code2 is None else int(code2)
            continue
        resid_left.append(rs)
    resid_sql = tuple(resid_left)
    if resid_sql:
        code_arrs = {kcol: kc}
        if gc is not None:
            code_arrs[gcol] = gc
        for rs in resid_sql:
            pred = sqlglot.parse_one(rs, read='duckdb')
            m = None
            if isinstance(pred, (E2.EQ, E2.NEQ)) and isinstance(pred.this, E2.Column) \
                    and isinstance(pred.expression, E2.Literal):
                rc = pred.this.name
                cc2 = seg.cols.get(rc)
                if cc2 is not None and cc2.get('mode') in (0, 1, 2) \
                        and not cc2.get('has_null'):
                    lv = pred.expression.this
                    code2 = WS._code_of(seg, rc, lv)
                    arr = code_arrs.get(rc)
                    if arr is None:
                        arr = np.asarray(seg._raw_codes(rc)).astype(np.int64)
                        code_arrs[rc] = arr
                    if code2 is None:             # literal absent from the dictionary:
                        m = (np.zeros(arr.size, bool) if isinstance(pred, E2.EQ)
                             else np.ones(arr.size, bool))
                    else:                         # CODE SPACE, not value space -- the
                        m = (arr == code2) if isinstance(pred, E2.EQ) else (arr != code2)
            if m is None:                         # value-space _eval_pred was 2.5s here
                m = wdb_sql._eval_pred(seg, pred, lambda nm2: nm2)
            if m is None:
                return None
            resid_mask = m if resid_mask is None else (resid_mask & m)
    V = int(seg.cols[gcol]['V'])
    rows = []
    # JACKSON'S STOP LINE: with ORDER BY count DESC LIMIT k on the union, only names
    # at-or-above the k-th best count can appear -- keep that superset (ties included:
    # exactness untouchable), decode ONLY those few names. 8,000 serial page-walks
    # become ~10-40 pooled lookups.
    trunc_k = None
    ordx = node.args.get('order')
    limx = node.args.get('limit')
    if ordx is not None and limx is not None and len(ordx.expressions) == 1:
        oe = ordx.expressions[0]
        if oe.args.get('desc') and isinstance(oe.this, E2.Column):
            cnt_alias = hdr[ci]
            if oe.this.name == cnt_alias:
                try:
                    trunc_k = int(limx.expression.this)
                except Exception:
                    trunc_k = None
    boards = []
    full_dict = None                              # decoded ONCE, shared by every branch
    cache = {}
    for code in kcodes:
        if code is None:
            continue                              # literal absent: branch yields nothing
        m = kc == code
        if resid_mask is not None:
            m = m & resid_mask
        planes = seg.e8_planes(gcol) if hasattr(seg, 'e8_planes') else None
        if planes is not None:
            # LITERAL-SPACE COUNT (the differential read): bincount the literals
            # under the branch mask + one subtraction for the defaults. The 400MB
            # dense column is never written.
            pos8, lits8, dflt8 = planes
            sel8 = m[pos8]
            cnts = np.bincount(lits8[sel8], minlength=V)
            cnts[dflt8] += int(m.sum()) - int(sel8.sum())
        else:
            cnts = np.bincount(gc[m], minlength=V)
        for zc in zero_gcodes:
            cnts[zc] = 0                         # the residual, applied to the board
        if eq_only_gcode is not None:
            keepv = cnts[eq_only_gcode] if eq_only_gcode >= 0 else 0
            cnts = np.zeros(V, cnts.dtype)
            if eq_only_gcode >= 0:
                cnts[eq_only_gcode] = keepv
        pres = np.nonzero(cnts)[0]
        if trunc_k is not None:
            boards.append((pres, cnts[pres]))    # decode NOTHING yet: names wait for
            continue                             # the stop line
        if pres.size > 5000 and full_dict is None:
            dv = seg._typed_dict(gcol)            # wherescan's bulk strategy: one full
            full_dict = np.array([wdb_sql._pyval(x) for x in dv], dtype=object)
        if full_dict is not None:
            vals = full_dict[pres]
            vals = [v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray))
                    else v for v in vals.tolist()]
        else:
            vals = []
            for gcd in pres.tolist():
                v = cache.get(gcd)
                if v is None:
                    v = wdb_sql._pyval(seg.fetch(gcol, gcd))
                    if isinstance(v, (bytes, bytearray)):
                        v = v.decode('utf-8', 'replace')
                    cache[gcd] = v
                vals.append(v)
        cl = cnts[pres].tolist()
        for i2, v in enumerate(vals):
            row = [None, None]
            row[1 - ci] = v
            row[ci] = int(cl[i2])
            rows.append(tuple(row))
    if trunc_k is not None and boards:
        allc = np.concatenate([cn for _p, cn in boards])
        if allc.size > trunc_k:
            thresh = np.partition(allc, allc.size - trunc_k)[allc.size - trunc_k]
        else:
            thresh = 0
        for pres_b, cn_b in boards:
            keepm = cn_b >= thresh
            codes_k = pres_b[keepm]
            cl_k = cn_b[keepm].tolist()
            vals_k = seg.values_at(gcol, codes_k) if codes_k.size else []
            for i2, v in enumerate(vals_k):
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode('utf-8', 'replace')
                row = [None, None]
                row[1 - ci] = v
                row[ci] = int(cl_k[i2])
                rows.append(tuple(row))
    global _HITS
    _HITS += 1
    return rows, list(hdr)


def _eval(db, node, esc):
    if isinstance(node, _SETOPS):
        if isinstance(node, E.Union) and node.args.get('distinct'):
            fast = _union_codes(db, node)
            if fast is not None:
                return fast
        if isinstance(node, E.Union) and not node.args.get('distinct'):
            fused = _try_fused_union(db, node)
            if fused is not None:
                return fused
        lrows, lhdr = _eval(db, node.this, esc)
        rrows, rhdr = _eval(db, node.expression, esc)
        if lrows and rrows and len(lrows[0]) != len(rrows[0]):
            raise ValueError("set operation arity mismatch: %d vs %d columns"
                             % (len(lrows[0]), len(rrows[0])))
        if isinstance(node, E.Union):
            if node.args.get('distinct'):
                rows = list(dict.fromkeys(lrows + rrows))
            else:
                rows = lrows + rrows
        elif isinstance(node, E.Intersect):
            rs = set(rrows)
            rows = [r for r in dict.fromkeys(lrows) if r in rs]
        else:                                        # EXCEPT
            rs = set(rrows)
            rows = [r for r in dict.fromkeys(lrows) if r not in rs]
        return rows, lhdr
    out = db.run(node.sql())
    rows, hdr = out if isinstance(out, tuple) else (out, None)
    return [_norm_row(r) for r in rows], hdr


def execute(db, tree, esc):
    rows, hdr = _eval(db, tree, esc)
    order = tree.args.get('order')
    if order is not None:
        for oe in reversed(order.expressions):
            if not isinstance(oe.this, E.Column):
                raise NotImplementedError("set-op ORDER BY supports output column names only")
            nm = oe.this.name
            idx = None
            if hdr:
                low = [str(h).lower() for h in hdr]
                if nm.lower() in low:
                    idx = low.index(nm.lower())
            if idx is None:
                raise NotImplementedError(f"set-op ORDER BY: unknown output column {nm!r}")
            rows.sort(key=lambda r: (r[idx] is None, r[idx]), reverse=bool(oe.args.get('desc')))
    import wdb_sql
    lim = wdb_sql._limit(tree)
    off = wdb_sql._offset(tree) or 0
    if lim is not None or off:
        rows = rows[off: None if lim is None else off + lim]
    return rows, hdr
