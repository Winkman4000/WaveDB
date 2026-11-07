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
        if outer.args.get('joins'):
            raise NotImplementedError("CTE shape not flattenable (joins)")
        # NESTED AGGREGATION: the CTE becomes a FROM subquery -- the FROM door runs
        # the inner to a frame and evaluates the outer over it
        outer.set('from_', E.From(this=E.Subquery(this=cte_sel.copy(), alias=E.TableAlias(this=E.Identifier(this='__cte', quoted=False)))))
        return outer
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


# THE WITH READ TWICE, RUN ONCE (2026-11-02): WITH c AS (body) SELECT ... FROM base, c WHERE base.k = c.ck
# AND c.v = (SELECT MAX(v) FROM c) [AND conditions on base only] -- c is read twice (joined, and inside the
# scalar), so it cannot flatten. Steps: (1) run the body once (its rows live for this query only); (2) the scalar
# reads only those rows: the largest (or smallest) v; (3) keep the body rows whose v equals it -- our own values
# compared with each other, so the equality is exact; (4) the outer becomes a read of base alone: WHERE base.k IN
# (the kept keys) and base's own conditions, c.ck -> base.k, c.v -> the extreme as a literal (named as before);
# ORDER / LIMIT unchanged. Any other use of c, another c column, or an outer join shape: None (as before).
# WDB_CTE_ONCE=0: off. _CTE_ONCE_HITS: diagnostic (tests read it).
# lore: built for TPC-H Q15.
import os as _os
_CTE_ONCE = [_os.environ.get('WDB_CTE_ONCE', '1') == '1']
_CTE_ONCE_HITS = [0]


def _unalias(p):
    return p.this if isinstance(p, E.Alias) else p


def once_rewrite(db, tree):
    """THE WITH READ TWICE, RUN ONCE: the rewritten outer SQL, or None (not this shape)."""
    if not _CTE_ONCE[0]:
        return None
    w = _with_arg(tree)
    if w is None or len(w.expressions) != 1 or w.args.get('recursive'):
        return None
    cte = w.expressions[0]
    C = cte.alias
    body = cte.this
    if not isinstance(body, E.Select):
        return None
    amap, star = _alias_map(body)
    if amap is None or star:
        return None
    cnames = [p.alias if isinstance(p, E.Alias) else p.name for p in body.expressions]
    outer = tree.copy()
    outer.set('with_', None)
    outer.set('with', None)
    f = _from_arg(outer)
    joins = outer.args.get('joins') or []
    if f is None or len(joins) != 1:
        return None
    j = joins[0]
    if j.args.get('on') or j.args.get('using') or j.args.get('side') or j.args.get('kind'):
        return None                                     # comma join only: the WHERE carries the key
    tabs = [f.this, j.this]
    if not all(isinstance(t, E.Table) for t in tabs):
        return None
    names = [t.name for t in tabs]
    if names.count(C) != 1:
        return None
    ct = tabs[names.index(C)]
    bt = tabs[1 - names.index(C)]
    calias, balias = ct.alias_or_name, bt.alias_or_name
    try:
        base_cols = set(db.cat.column_names(bt.name))
    except Exception:
        return None
    def is_c(col):
        return col.table == calias or (not col.table and col.name in cnames and col.name not in base_cols)
    def is_b(col):
        return col.table == balias or (not col.table and col.name in base_cols and col.name not in cnames)
    wh = outer.args.get('where')
    if wh is None:
        return None
    key_eq = None; ext = None; rest = []
    for c in _conjuncts(wh.this):
        if isinstance(c, E.EQ) and isinstance(c.this, E.Column) and isinstance(c.expression, E.Column):
            a, b = c.this, c.expression
            if is_b(a) and is_c(b) and key_eq is None:
                key_eq = (a, b.name); continue
            if is_c(a) and is_b(b) and key_eq is None:
                key_eq = (b, a.name); continue
        if isinstance(c, E.EQ) and ext is None:
            hit = None
            for x, y in ((c.this, c.expression), (c.expression, c.this)):
                if isinstance(x, E.Column) and is_c(x) and isinstance(y, E.Subquery):
                    s = y.this
                    if (isinstance(s, E.Select) and len(s.expressions) == 1 and _table_name(s) == C
                            and not any(s.args.get(k) for k in ('where', 'joins', 'group', 'having', 'limit', 'order'))):
                        ag = _unalias(s.expressions[0])
                        if isinstance(ag, (E.Max, E.Min)) and isinstance(ag.this, E.Column) and ag.this.name == x.name:
                            hit = (x.name, isinstance(ag, E.Max))
                if hit: break
            if hit:
                ext = hit; continue
        cols = list(c.find_all(E.Column))
        if c.find(E.Select) is None and cols and all(is_b(x) for x in cols):
            rest.append(c); continue
        return None
    if key_eq is None or ext is None:
        return None
    bkey, ckey = key_eq
    vname, is_max = ext
    order = outer.args.get('order')
    for nd in list(outer.expressions) + ([order] if order is not None else []):
        for col in nd.find_all(E.Column):
            if is_c(col):
                if col.name not in (ckey, vname): return None
            elif not is_b(col):
                return None
    # (1) the body, once
    out = db.run(body.sql())
    rows = out[0] if isinstance(out, tuple) else out
    if not rows or len(rows[0]) != len(cnames):
        return None
    iv, ik = cnames.index(vname), cnames.index(ckey)
    vals = [r[iv] for r in rows if r[iv] is not None]
    if not vals:
        return None
    # (2) the extreme, (3) the keys whose value equals it
    target = max(vals) if is_max else min(vals)
    keys = list(dict.fromkeys(r[ik] for r in rows if r[iv] == target))
    if not keys or any(k is None for k in keys):
        return None
    # (4) the outer over base alone
    def _lit(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return E.Literal.number(repr(v) if isinstance(v, float) else v)
        return E.Literal.string(str(v))
    new = outer.copy()
    new.set('joins', None)
    new.set('from_', E.From(this=bt.copy()))
    new.set('from', None)
    exprs = []
    for p in new.expressions:
        nm = p.alias_or_name
        inner = _unalias(p)
        if isinstance(inner, E.Column) and is_c(inner):
            rep = _lit(target) if inner.name == vname else bkey.copy()
            exprs.append(E.Alias(this=rep, alias=E.Identifier(this=nm, quoted=False)))
            continue
        for col in list(inner.find_all(E.Column)):
            if is_c(col):
                col.replace(_lit(target) if col.name == vname else bkey.copy())
        exprs.append(p)
    new.set('expressions', exprs)
    if new.args.get('order') is not None:
        for col in list(new.args['order'].find_all(E.Column)):
            if is_c(col):
                col.replace(_lit(target) if col.name == vname else bkey.copy())
    cond = E.In(this=bkey.copy(), expressions=[_lit(k) for k in keys])
    for r in rest:
        cond = E.And(this=cond, expression=r.copy())
    new.set('where', E.Where(this=cond))
    _CTE_ONCE_HITS[0] += 1
    return new.sql()


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
