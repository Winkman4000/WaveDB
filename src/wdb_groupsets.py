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
    sets = _sets_of(tree)
    finest = max(sets, key=len)
    kinds = [wdb_sql._agg_kind(proj[pi])[0] for k, pi in slots if k == 'agg']
    rollable = (set(key_names) == set(finest)
                and all(s2 and set(s2) <= set(finest) or s2 == () for s2 in sets)
                and all(kd in ('COUNT_STAR', 'SUM', 'MIN', 'MAX') for kd in kinds)
                and tree.args.get('having') is None)
    if rollable:
        return _roll_from_finest(db, tree, proj, slots, key_names, sets, finest, kinds)
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


def _roll_from_finest(db, tree, proj, slots, key_names, sets, finest, kinds):
    """One scan: the FINEST grouping runs through the engine; every coarser set derives from
    its cells in python (COUNT/SUM add, MIN/MAX fold) -- N sub-scans become one."""
    sub = tree.copy()
    for k in ('order', 'limit', 'offset'):
        sub.set(k, None)
    sub.set('group', E.Group(expressions=[E.column(n) for n in finest]))
    out = db.run(sub.sql())
    cells, _h = out if isinstance(out, tuple) else (out, None)
    kpos = {}
    cursor = 0
    for kind, ident in slots:
        kpos[ident if kind == 'key' else ('agg', ident)] = cursor
        cursor += 1
    all_rows = []
    agg_slots = [(pi, kd) for (k, pi), kd in
                 zip([sl for sl in slots if sl[0] == 'agg'], kinds)]
    for st in sets:
        if tuple(st) == tuple(finest):
            for r in cells:
                all_rows.append(tuple(r))
            continue
        acc = {}
        keep_idx = [kpos[n] for n in key_names if n in st]
        st_names = [n for n in key_names if n in st]
        for r in cells:
            key = tuple(r[i] for i in keep_idx)
            cur = acc.get(key)
            if cur is None:
                acc[key] = [r[kpos[('agg', pi)]] for pi, _kd in agg_slots]
            else:
                for ai, (pi, kd) in enumerate(agg_slots):
                    v = r[kpos[('agg', pi)]]
                    if kd in ('COUNT_STAR', 'SUM'):
                        cur[ai] = cur[ai] + v
                    elif kd == 'MIN':
                        cur[ai] = v if v < cur[ai] else cur[ai]
                    else:
                        cur[ai] = v if v > cur[ai] else cur[ai]
        for key, aggs in acc.items():
            row = []
            ki = ai = 0
            for kind, ident in slots:
                if kind == 'key':
                    row.append(key[st_names.index(ident)] if ident in st else None)
                else:
                    row.append(aggs[ai]); ai += 1
            all_rows.append(tuple(row))
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
