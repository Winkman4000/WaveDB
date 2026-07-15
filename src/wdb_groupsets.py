"""wdb_groupsets: GROUPING SETS / ROLLUP / CUBE as controller-level composition.

Each grouping set is a plain GROUP BY the engine is already fast at: the sets run through
the full pipeline one by one, rows assemble into full-width tuples with NULL in the absent
key slots (per the standard), and the outer ORDER/LIMIT apply to the union. ROLLUP(a,b)
expands to ((a,b),(a),()); CUBE to all subsets (capped at 4 keys -- 16 sub-queries); the
empty set is the global aggregate. HAVING passes into every sub-query, which is exactly its
per-group semantics.
"""
import sqlglot.expressions as E
import wdb_sql


def has_grouping(tree):
    g = tree.args.get('group')
    if g is None:
        return False
    return bool(g.args.get('rollup') or g.args.get('cube') or g.args.get('grouping_sets'))


def _colname(e):
    if isinstance(e, E.Paren):
        e = e.this
    return e.name if isinstance(e, E.Column) else None


def _sets_of(tree):
    g = tree.args.get('group')
    plain = [x.name for x in g.expressions if isinstance(x, E.Column)]
    out = []
    for r in g.args.get('rollup') or []:
        names = [x.name for x in r.expressions]
        for i in range(len(names), -1, -1):
            out.append(tuple(plain + names[:i]))
    for c in g.args.get('cube') or []:
        names = [x.name for x in c.expressions]
        if len(names) > 4:
            raise NotImplementedError("CUBE past 4 keys (%d sub-queries)" % (2 ** len(names)))
        for mask in range(2 ** len(names) - 1, -1, -1):
            out.append(tuple(plain + [n for i, n in enumerate(names) if mask & (1 << i)]))
    for gs in g.args.get('grouping_sets') or []:
        for t in gs.expressions:
            if isinstance(t, E.Tuple):
                out.append(tuple(plain + [x.name for x in t.expressions]))
            elif isinstance(t, E.Paren):
                out.append(tuple(plain + ([t.this.name] if isinstance(t.this, E.Column) else [])))
            elif isinstance(t, E.Column):
                out.append(tuple(plain + [t.name]))
            else:
                raise NotImplementedError(f"grouping set element {type(t).__name__}")
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq


def execute(db, tree):
    proj = tree.expressions
    slots = []                                   # ('key', name) | ('agg', proj_index)
    key_names = []
    for pi, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            slots.append(('key', inner.name)); key_names.append(inner.name)
        elif wdb_sql._agg_kind(p) is not None:
            slots.append(('agg', pi))
        else:
            raise NotImplementedError("grouping-sets projection must be keys or aggregates")
    all_rows = []
    for st in _sets_of(tree):
        sub = tree.copy()
        for k in ('order', 'limit', 'offset'):
            sub.set(k, None)
        keep = [p for p in proj
                if (wdb_sql._agg_kind(p) is not None)
                or ((p.this if isinstance(p, E.Alias) else p).name in st)]
        sub.set('expressions', [p.copy() for p in keep])
        if st:
            sub.set('group', E.Group(expressions=[E.column(n) for n in st]))
        else:
            sub.set('group', None)
        out = db.run(sub.sql())
        rows, hdr = out if isinstance(out, tuple) else (out, None)
        pos, cursor = [], 0
        for kind, ident in slots:
            if kind == 'key':
                pos.append(cursor if ident in st else None)
                cursor += 1 if ident in st else 0
            else:
                pos.append(cursor); cursor += 1
        for r in rows:
            all_rows.append(tuple(None if p is None else r[p] for p in pos))
    order = tree.args.get('order')
    if order is not None:
        names = [wdb_sql._alias(p) for p in proj]
        for oe in reversed(order.expressions):
            nm = oe.this.name if isinstance(oe.this, E.Column) else None
            if nm is None or nm not in names:
                raise NotImplementedError("grouping-sets ORDER BY must use output columns")
            idx = names.index(nm)
            all_rows.sort(key=lambda r: (r[idx] is None, r[idx]),
                          reverse=bool(oe.args.get('desc')))
    lim = wdb_sql._limit(tree)
    off = wdb_sql._offset(tree) or 0
    if lim is not None or off:
        all_rows = all_rows[off: None if lim is None else off + lim]
    return all_rows, [wdb_sql._alias(p) for p in proj]
