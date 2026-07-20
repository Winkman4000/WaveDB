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
            p = WS._scan_flag(seg, col2, fl, 0, seg.N)
            pos = p if pos is None else np.intersect1d(pos, p, assume_unique=True)
            if pos.size == 0:
                break
    if pos is None:
        cn = np.bincount(np.asarray(seg._raw_codes(C)), minlength=int(cC['V']))
        codes = np.flatnonzero(cn > 0).astype(np.int64)
    else:
        codes = np.unique(np.asarray(seg.codes_at(C, pos)).astype(np.int64))
    has_null = False
    if cC.get('has_null'):
        nullc = int(cC['V']) - 1
        if codes.size and int(codes[-1]) == nullc:
            has_null = True
            codes = codes[:-1]
    return codes, has_null


def has_subquery(tree):
    w = tree.args.get('where')
    if w is None:
        return False
    return (any(True for _ in w.find_all(E.Subquery))
            or any(True for _ in w.find_all(E.Exists)))


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


def _corr_eq(sel, outer_names):
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
        return bool(x.table) and x.table not in inner_quals
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
    ak = wdb_sql._agg_kind(inner.expressions[0])
    if ak is None or ak[0] not in ('SUM', 'AVG', 'MIN', 'MAX', 'COUNT_STAR'):
        return None
    ce = _corr_eq(inner, None)
    if ce is None or ce[2]:
        return None                              # v1: pure eq-correlation, no extra conds
    icol, ocol = ce[0], ce[1]
    if icol != ocol:
        return None                              # partition key must be the same column
    fname = {'SUM': 'SUM', 'AVG': 'AVG', 'MIN': 'MIN', 'MAX': 'MAX', 'COUNT_STAR': 'COUNT'}[ak[0]]
    arg = '*' if ak[0] == 'COUNT_STAR' else ak[1]
    win_sql = f"{fname}({arg}) OVER (PARTITION BY {icol}) AS __corr0"
    new = tree.copy()
    new.set('where', None)
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
        ce = _corr_eq(inner, None)
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
            node.set('_codes', codes)             # query arg stays: codes-unaware paths
            continue                              # decline as before (fail closed)
        rows = _run_inner(db, sub)
        if rows and len(rows[0]) != 1:
            raise ValueError("IN subquery must return one column")
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
