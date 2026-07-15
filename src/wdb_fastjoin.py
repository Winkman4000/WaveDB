"""wdb_fastjoin: dimension joins through the dictionary -- the introductions happen in
dict space, never in row space.

The analytic join shape is fact-vs-dimension: a giant table whose key column carries dict
codes, and a small table describing those keys. Two rewrites cover it:

  A) Conditions on the dimension become a SEMI-JOIN: run the dim query (small), collect the
     matching keys, and the fact side runs single-table with key IN (...) -- the IN driver
     does the row work as one flag scan. The join evaporates.

  B) Grouping by dimension attributes becomes GROUP-BY-KEY + POST-MAP: the fact side groups
     by its own key column (existing fast reads), then key->attribute mapping and
     re-aggregation happen on the GROUP CELLS (thousands), not the rows (millions).
     COUNT/SUM/MIN/MAX fold exactly (the grouping-sets lemma); AVG declines.

One-to-many dimensions (duplicate keys) would multiply rows -- decline to the mature join
path. Everything here is INNER equi-join; other kinds fall through untouched.
"""
import sqlglot.expressions as E
import wdb_sql

_DIM_CAP = 500_000


def _tables(tree):
    f = tree.args.get('from_') or tree.args.get('from')
    joins = tree.args.get('joins') or []
    if f is None or len(joins) != 1 or not isinstance(f.this, E.Table):
        return None
    j = joins[0]
    if (j.side or '').upper() not in ('', 'INNER') or (j.kind or '').upper() not in ('', 'INNER'):
        return None
    if not isinstance(j.this, E.Table):
        return None
    t1, t2 = f.this, j.this
    on = j.args.get('on')
    if on is None or type(on).__name__ != 'EQ':
        return None
    a, b = on.this, on.expression
    if not (isinstance(a, E.Column) and isinstance(b, E.Column)):
        return None
    return (t1.name, t1.alias or t1.name), (t2.name, t2.alias or t2.name), (a, b)


def _split_where(tree, quals1, quals2):
    import wdb_wherescan as WS
    w = tree.args.get('where')
    c1, c2 = [], []
    if w is None:
        return c1, c2
    for c in WS._conjuncts(w.this):
        tabs = {x.table for x in c.find_all(E.Column) if x.table}
        unq = [x for x in c.find_all(E.Column) if not x.table]
        if unq:
            return None                          # v1: every column must be qualified
        if tabs <= quals1:
            c1.append(c)
        elif tabs <= quals2:
            c2.append(c)
        else:
            return None                          # mixed-side condition: not separable
    return c1, c2


def _strip_qual(node):
    n = node.copy()
    for col in n.find_all(E.Column):
        col.set('table', None)
    return n


def try_execute(db, tree):
    """Rows/headers for a fact-dim join, or None to fall through."""
    t = _tables(tree)
    if t is None:
        return None
    (n1, a1), (n2, a2), (ka, kb) = t
    try:
        sizes = {a1: db.cat.get_table(n1).get('rows'), a2: db.cat.get_table(n2).get('rows')}
    except KeyError:
        return None
    # fact = the side of the ON key we scan; dim = the side we materialize
    for fact_al, dim_al, fact_tn, dim_tn, fk, dk in (
            (a1, a2, n1, n2, ka, kb), (a2, a1, n2, n1, kb, ka)):
        if fk.table != fact_al or dk.table != dim_al:
            continue
        out = _try_orientation(db, tree, fact_al, dim_al, fact_tn, dim_tn, fk.name, dk.name)
        if out is not None:
            return out
    return None


def _try_orientation(db, tree, fact_al, dim_al, fact_tn, dim_tn, fkey, dkey):
    proj = tree.expressions
    dim_cols, fact_cols, aggs = [], [], []
    for pi, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        ak = wdb_sql._agg_kind(p)
        if ak is not None:
            if ak[0] not in ('COUNT_STAR', 'SUM', 'MIN', 'MAX'):
                return None
            if ak[0] != 'COUNT_STAR':
                arg = inner.this
                if not (isinstance(arg, E.Column) and arg.table == fact_al):
                    return None                  # aggregate args live on the fact
            aggs.append((pi, ak[0]))
        elif isinstance(inner, E.Column) and inner.table == dim_al:
            dim_cols.append((pi, inner.name))
        elif isinstance(inner, E.Column) and inner.table == fact_al:
            fact_cols.append((pi, inner.name))
        else:
            return None
    group = tree.args.get('group')
    if group is not None:
        gnames = set()
        for g in group.expressions:
            if not isinstance(g, E.Column):
                return None
            gnames.add((g.table, g.name))
    sw = _split_where(tree, {fact_al, fact_tn}, {dim_al, dim_tn})
    if sw is None:
        return None
    fact_conds, dim_conds = sw
    if tree.args.get('having') is not None or tree.args.get('qualify') is not None:
        return None
    # ---- dim side: key + needed attributes, filtered ----
    need = sorted({nm for _pi, nm in dim_cols})
    dsel = 'SELECT ' + ', '.join([dkey] + need) + ' FROM ' + dim_tn
    if dim_conds:
        dsel += ' WHERE ' + ' AND '.join(_strip_qual(c).sql() for c in dim_conds)
    drows_out = db.run(dsel)
    drows = drows_out[0] if isinstance(drows_out, tuple) else drows_out
    if len(drows) > _DIM_CAP:
        return None
    dmap = {}
    for r in drows:
        if r[0] in dmap:
            return None                          # one-to-many dimension: would multiply rows
        dmap[r[0]] = r[1:]
    if not dmap:
        return [], [wdb_sql._alias(p) for p in proj]
    attr_idx = {nm: i for i, nm in enumerate(need)}
    keys = list(dmap.keys())
    # ---- fact side: single-table, key IN (semi-join), grouped by (key + fact group cols) ----
    def lit(v):
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        return str(v)
    fact_where = [_strip_qual(c).sql() for c in fact_conds]
    fact_where.append(fkey + ' IN (' + ', '.join(lit(k) for k in keys) + ')')
    if aggs:
        gcols = [fkey] + sorted({nm for _pi, nm in fact_cols})
        fexpr = list(gcols)
        for pi, kd in aggs:
            p = proj[pi]
            inner = p.this if isinstance(p, E.Alias) else p
            fexpr.append(_strip_qual(inner).sql() + ' AS __a%d' % pi)
        fsel = ('SELECT ' + ', '.join(fexpr) + ' FROM ' + fact_tn
                + ' WHERE ' + ' AND '.join(fact_where)
                + ' GROUP BY ' + ', '.join(gcols))
        frows_out = db.run(fsel)
        frows = frows_out[0] if isinstance(frows_out, tuple) else frows_out
        # ---- post-map on CELLS: key -> dim attrs, re-aggregate, order, limit ----
        fpos = {nm: i for i, nm in enumerate(gcols)}
        acc = {}
        for r in frows:
            dvals = dmap.get(r[0])
            if dvals is None:
                continue
            gkey = []
            for pi, nm in dim_cols:
                gkey.append(dvals[attr_idx[nm]])
            for pi, nm in fact_cols:
                gkey.append(r[fpos[nm]])
            gkey = tuple(gkey)
            avals = r[len(gcols):]
            cur = acc.get(gkey)
            if cur is None:
                acc[gkey] = list(avals)
            else:
                for i, (_pi, kd) in enumerate(aggs):
                    v = avals[i]
                    if kd in ('COUNT_STAR', 'SUM'):
                        cur[i] = cur[i] + v
                    elif kd == 'MIN':
                        cur[i] = v if v < cur[i] else cur[i]
                    else:
                        cur[i] = v if v > cur[i] else cur[i]
        rows = []
        for gkey, avals in acc.items():
            row = [None] * len(proj)
            ki = 0
            for pi, nm in dim_cols:
                row[pi] = gkey[ki]; ki += 1
            for pi, nm in fact_cols:
                row[pi] = gkey[ki]; ki += 1
            for i, (pi, _kd) in enumerate(aggs):
                row[pi] = avals[i]
            rows.append(tuple(row))
    else:
        # plain projection dump: bounded only
        lim = wdb_sql._limit(tree)
        if lim is None or lim > 1_000_000:
            return None
        fexpr = [fkey] + sorted({nm for _pi, nm in fact_cols})
        fsel = ('SELECT ' + ', '.join(fexpr) + ' FROM ' + fact_tn
                + ' WHERE ' + ' AND '.join(fact_where))
        if tree.args.get('order') is None:
            fsel += ' LIMIT ' + str(lim * 2)
        frows_out = db.run(fsel)
        frows = frows_out[0] if isinstance(frows_out, tuple) else frows_out
        fpos = {nm: i for i, nm in enumerate(fexpr)}
        rows = []
        for r in frows:
            dvals = dmap.get(r[0])
            if dvals is None:
                continue
            row = [None] * len(proj)
            for pi, nm in dim_cols:
                row[pi] = dvals[attr_idx[nm]]
            for pi, nm in fact_cols:
                row[pi] = r[fpos[nm]]
            rows.append(tuple(row))
    order = tree.args.get('order')
    if order is not None:
        names = [wdb_sql._alias(p) for p in proj]
        for oe in reversed(order.expressions):
            nm = oe.this.name if isinstance(oe.this, E.Column) else None
            if nm is None or nm not in names:
                return None
            idx = names.index(nm)
            rows.sort(key=lambda r: (r[idx] is None, r[idx]), reverse=bool(oe.args.get('desc')))
    lim = wdb_sql._limit(tree)
    off = wdb_sql._offset(tree) or 0
    if lim is not None or off:
        rows = rows[off: None if lim is None else off + lim]
    return rows, [wdb_sql._alias(p) for p in proj]
