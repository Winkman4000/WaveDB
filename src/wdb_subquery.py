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


def has_subquery(tree):
    w = tree.args.get('where')
    return w is not None and any(True for _ in w.find_all(E.Subquery))


def _run_inner(db, sub):
    inner = sub.this
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


def rewrite(db, tree):
    """Substitute every uncorrelated WHERE subquery; returns the rewritten tree."""
    w = tree.args.get('where')
    if w is None:
        return tree
    # IN (SELECT ...) -- handle In nodes carrying a query arg (NOT IN arrives as Not(In))
    for node in list(w.find_all(E.In)):
        sub = node.args.get('query')
        if sub is None:
            continue
        rows = _run_inner(db, sub)
        if rows and len(rows[0]) != 1:
            raise ValueError("IN subquery must return one column")
        vals = [r[0] for r in rows]
        has_null = any(v is None for v in vals)
        vals = list(dict.fromkeys(v for v in vals if v is not None))
        if len(vals) > _MAX_IN:
            raise NotImplementedError("IN subquery result too large (%d values)" % len(vals))
        negated = isinstance(node.parent, E.Not)
        if negated and has_null:
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
