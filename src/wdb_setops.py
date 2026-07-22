"""wdb_setops: UNION / UNION ALL / INTERSECT / EXCEPT as controller-level composition.

Each side runs through the full existing pipeline (any read, any table -- sides recurse, so
chains compose), then rows combine by SQL bag/set semantics: UNION ALL concatenates, UNION
dedupes (first-seen order), INTERSECT and EXCEPT operate on distinct rows per the standard.
Outer ORDER BY / LIMIT / OFFSET apply to the combined result. Headers come from the left side;
arity mismatches raise loudly (never silently truncate).
"""
import sqlglot.expressions as E

_SETOPS = (E.Union, E.Intersect, E.Except)


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
            p = WS._scan_flag(seg, col2, fl, 0, seg.N)
            pos = p if pos is None else np.intersect1d(pos, p, assume_unique=True)
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


def _eval(db, node, esc):
    if isinstance(node, _SETOPS):
        if isinstance(node, E.Union) and node.args.get('distinct'):
            fast = _union_codes(db, node)
            if fast is not None:
                return fast
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
