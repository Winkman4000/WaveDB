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


def _eval(db, node, esc):
    if isinstance(node, _SETOPS):
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
