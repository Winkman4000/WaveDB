"""wdb_cte: WITH-clause support by flattening -- views merge, they don't materialize.

A CTE that only filters and projects is a lens over its base table: outer references
substitute through the alias map (name -> underlying expression -- scalar expressions ride
along free), FROMs collapse, WHEREs conjoin. An AGGREGATING cte with a filtering outer query
flattens the other way: outer conditions on aggregate aliases become HAVING, conditions on
key aliases join the inner WHERE, and the outer's ORDER/LIMIT take over. Chained CTEs flatten
iteratively until the FROM names a real table. Shapes outside this (a CTE used twice under a
join, recursive CTEs) decline loudly -- materialization is a later war.
"""
import sqlglot.expressions as E


def has_cte(tree):
    return tree.args.get('with_') is not None or tree.args.get('with') is not None


def _with_arg(tree):
    return tree.args.get('with_') or tree.args.get('with')


def _from_arg(tree):
    return tree.args.get('from_') or tree.args.get('from')


def _table_name(tree):
    f = _from_arg(tree)
    if f is None or not isinstance(f.this, E.Table):
        return None
    return f.this.name


def _alias_map(cte_sel):
    amap = {}
    for p in cte_sel.expressions:
        if isinstance(p, E.Star):
            return amap, True
        inner = p.this if isinstance(p, E.Alias) else p
        nm = p.alias if isinstance(p, E.Alias) else (p.name if isinstance(p, E.Column) else None)
        if nm is None:
            return None, False
        amap[nm] = inner
    return amap, False


def _subst(node, amap):
    for col in list(node.find_all(E.Column)):
        if col.name in amap and col.parent is not None:
            col.replace(amap[col.name].copy())
    return node


def _and_where(sel, cond):
    w = sel.args.get('where')
    if w is None:
        sel.set('where', E.Where(this=cond))
    else:
        w.set('this', E.And(this=w.this, expression=cond))


def _and_having(sel, cond):
    h = sel.args.get('having')
    if h is None:
        sel.set('having', E.Having(this=cond))
    else:
        h.set('this', E.And(this=h.this, expression=cond))


def _conjuncts(node):
    if isinstance(node, E.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _flatten_once(outer, cte_sel):
    cte_sel = cte_sel.copy()
    if cte_sel.args.get('joins') or _from_arg(cte_sel) is None:
        raise NotImplementedError("CTE shape not flattenable (joins)")
    amap, star = _alias_map(cte_sel)
    if amap is None:
        raise NotImplementedError("CTE projection not flattenable")
    has_agg = (cte_sel.args.get('group') is not None
               or any(E and _is_agg(p) for p in cte_sel.expressions))
    if not has_agg and not any(cte_sel.args.get(k) for k in
                               ('limit', 'distinct', 'order', 'having')):
        # simple view: substitute outer refs, collapse FROM, conjoin WHERE
        for key in ('expressions',):
            for p in outer.args.get(key) or []:
                _subst(p, amap) if not star else None
        for key in ('where', 'group', 'order', 'having', 'qualify'):
            if outer.args.get(key) is not None and not star:
                _subst(outer.args[key], amap)
        outer.set('from_', _from_arg(cte_sel).copy())
        cw = cte_sel.args.get('where')
        if cw is not None:
            _and_where(outer, cw.this)
        return outer
    # aggregating CTE + filtering/selecting outer: flatten to HAVING/WHERE on the inner
    if (outer.args.get('group') is not None or outer.args.get('joins')
            or any(_is_agg(p) for p in outer.expressions)):
        raise NotImplementedError("CTE shape not flattenable (nested aggregation)")
    agg_aliases = set()
    for p in cte_sel.expressions:
        if isinstance(p, E.Alias) and _is_agg(p):
            agg_aliases.add(p.alias)
    out_names = []
    for p in outer.expressions:
        if isinstance(p, E.Star):
            out_names = None
            break
        if not isinstance(p, E.Column):
            raise NotImplementedError("CTE outer projection must be plain columns or *")
        out_names.append(p.name)
    ow = outer.args.get('where')
    if ow is not None:
        for c in _conjuncts(ow.this):
            names = {col.name for col in c.find_all(E.Column)}
            if names & agg_aliases:
                _and_having(cte_sel, _subst(c.copy(), amap))   # alias -> aggregate expression
            else:
                _and_where(cte_sel, _subst(c.copy(), amap))
    if out_names is not None:
        keep, order_map = [], {}
        for p in cte_sel.expressions:
            nm = p.alias if isinstance(p, E.Alias) else (p.name if isinstance(p, E.Column) else None)
            order_map[nm] = p
        for nm in out_names:
            if nm not in order_map:
                raise NotImplementedError(f"CTE outer column {nm!r} not in CTE output")
            keep.append(order_map[nm].copy())
        cte_sel.set('expressions', keep)
    if outer.args.get('order') is not None:
        cte_sel.set('order', outer.args['order'].copy())
    for k in ('limit', 'offset'):
        if outer.args.get(k) is not None:
            cte_sel.set(k, outer.args[k].copy())
        elif cte_sel.args.get(k) is not None and outer.args.get(k) is None:
            pass                                      # keep the CTE's own bound
    if outer.args.get('qualify') is not None:
        raise NotImplementedError("QUALIFY over aggregating CTE")
    return cte_sel


def _is_agg(p):
    import wdb_sql
    return wdb_sql._agg_kind(p) is not None


def rewrite(tree):
    """Flatten every CTE reference; returns a WITH-free tree or raises loudly."""
    w = _with_arg(tree)
    if w is None:
        return tree
    if any(c.args.get('recursive') for c in w.expressions) or w.args.get('recursive'):
        raise NotImplementedError("recursive CTEs")
    ctes = {}
    for c in w.expressions:
        if not isinstance(c.this, E.Select):
            raise NotImplementedError("non-SELECT CTE")
        ctes[c.alias] = c.this
    outer = tree.copy()
    outer.set('with_', None)
    outer.set('with', None)
    guard = 0
    while True:
        nm = _table_name(outer)
        if nm is None or nm not in ctes:
            break
        outer = _flatten_once(outer, ctes[nm])
        guard += 1
        if guard > 16:
            raise NotImplementedError("CTE chain too deep")
    return outer
