"""LITERAL RELATIONS: queries that touch no stored table -- SELECT without FROM,
VALUES as a table source, and recursive CTEs over literal seeds. Correctness
faces on sqlglot's Python executor (in-memory tables), never the engine's
scan paths."""
import sqlglot
from sqlglot import exp as E


def references_only_literals(tree, is_stored):
    """True when the query touches NO stored table: every table reference is a
    CTE name or a VALUES source, and at least one relation is literal (a bare
    SELECT, a VALUES, or a recursive CTE). is_stored(name) asks the live catalog."""
    if not isinstance(tree, (E.Select, E.Union, E.Intersect, E.Except)):
        return False
    cte_names = {c.alias for c in tree.find_all(E.CTE)}
    tables = list(tree.find_all(E.Table))
    for t in tables:
        if t.name not in cte_names and is_stored(t.name):
            return False
        if t.name not in cte_names and not is_stored(t.name):
            return False                     # an unknown table is an ERROR for the engine to raise, not ours
    with_ = tree.args.get('with') or tree.args.get('with_')
    has_values = tree.find(E.Values) is not None
    bare = not tables and (tree.args.get('from') is None and tree.args.get('from_') is None)
    recursive = with_ is not None and bool(with_.args.get('recursive'))
    return has_values or bare or recursive


def _rows_of(table):
    return [tuple(r) for r in table.rows], list(table.columns)


def _values_tables(tree):
    """FROM (VALUES ...) AS v(a, b)  ->  {'v': [ {a:..,b:..}, ... ]} and the tree with FROM v."""
    from sqlglot.executor import execute
    tables = {}
    for vals in list(tree.find_all(E.Values)):
        holder = vals.parent if isinstance(vals.parent, E.Subquery) else vals
        alias = vals.alias or (holder.alias if holder is not vals else None) or 'v'
        al = vals.args.get('alias') or (holder.args.get('alias') if holder is not vals else None)
        cols = [c.name for c in (al.columns if al is not None and al.columns else [])]
        rows = []
        for tup in vals.expressions:
            items = list(tup.expressions) if isinstance(tup, E.Tuple) else [tup]
            r = [execute('SELECT %s AS x' % it.sql()).rows[0][0] for it in items]
            rows.append(r)
        if not cols:
            cols = ['col%d' % i for i in range(len(rows[0]) if rows else 0)]
        tables[alias] = [dict(zip(cols, r)) for r in rows]
        holder.replace(E.Table(this=E.Identifier(this=alias, quoted=False)))
    return tables


def _recursive(tree):
    """WITH RECURSIVE r(n) AS (seed UNION ALL step) outer -> materialise r."""
    from sqlglot.executor import execute
    with_ = tree.args.get('with') or tree.args.get('with_')
    if with_ is None or not with_.args.get('recursive'):
        return None
    tables = {}
    for cte in with_.expressions:
        name = cte.alias
        cols = [c.name for c in (cte.args.get('alias').columns if cte.args.get('alias') is not None and cte.args['alias'].columns else [])]
        body = cte.this
        if not isinstance(body, E.Union):
            raise NotImplementedError('recursive CTE body must be seed UNION ALL step')
        seed, step = body.this, body.expression
        seed_t = execute(seed.sql(), tables=tables)
        rows = [dict(zip(cols or seed_t.columns, r)) for r in seed_t.rows]
        allrows = list(rows)
        cur = rows
        for _ in range(100_000):
            if not cur:
                break
            t2 = execute(step.sql(), tables={**tables, name: cur})
            cur = [dict(zip(cols or t2.columns, r)) for r in t2.rows]
            allrows.extend(cur)
        tables[name] = allrows
    outer = tree.copy(); outer.set('with', None); outer.set('with_', None)
    return execute(outer.sql(), tables=tables)


def run_literal(tree):
    """Returns (rows, names)."""
    from sqlglot.executor import execute
    t = tree.copy()
    rec = _recursive(t)
    if rec is not None:
        return _rows_of(rec)
    tables = _values_tables(t)
    out = execute(t.sql(), tables=tables)
    return _rows_of(out)
