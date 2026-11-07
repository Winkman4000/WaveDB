"""THE PROJECTION DOOR: a pointer-chain join that only projects, ordered and limited, answered at the survivors.

A pointer-chain join with no GROUP BY and no aggregate -- every selected and ordered column a plain column of a joined
table -- was handed to the pandas tail. Here:
  1. recognise the shape (inner joins of plain tables, bare column names unique across them);
  2. the survivors once: the same cascade the aggregate path runs, through the pointers, potency order
     (wdb_join.join_rows) -- only when it spends the whole WHERE;
  3. THE PER-KEY EXTREME (x = (SELECT MIN|MAX(x) FROM <some of the same tables> WHERE k = outer key AND <the
     outer's own conditions on those tables>)): the outer key is unique in its table and every other outer condition
     on the tables outside the subquery reads only those tables, so the subquery's rows for a key ARE the outer's
     survivors with that key; the zoom over the key's runs takes each run's extreme, the rows equal to it stay;
  4. ORDER BY on the kept rows (columns read at those rows only, parents through their pointers; a stable sort);
  5. LIMIT; 6. the selected columns decoded at the winners only.
c128: a conjunct 'col [NOT] IN (uncorrelated subquery)' becomes the column's code set before the survivors
(wdb_insub.serve_conj), and a comparison against a correlated scalar subquery is judged at the survivors (THE VALUE PER
ROW, wdb_corrrow, step 2c); a single table comes to the door only with such a comparison (single=True, wdb_db).
Anything else declines to the roads behind it.

Lore: 2026-11-03, built for TPC-H Q2 (steps confirmed with Jackson): Q2's outer query took 1.64 s in the pandas tail
against DuckDB's 22 ms, and step 3 is Q2's correlated-subquery shape. c128 came with TPC-H Q20."""
import os
import numpy as np
import sqlglot.expressions as E

_PROJ_DOOR = [os.environ.get('WDB_PROJ_DOOR', '1') == '1']
_PROJ_DOOR_HITS = [0]
_PER_KEY_HITS = [0]


def _conj(n):
    if n is None:
        return []
    if isinstance(n, E.And):
        return _conj(n.this) + _conj(n.expression)
    if isinstance(n, E.Paren) and isinstance(n.this, E.And):
        return _conj(n.this)
    return [n]


def _tables(sel):
    """the plain tables of a SELECT's FROM / inner JOINs (names, each once), or None"""
    f = sel.args.get('from_') or sel.args.get('from')
    if f is None or not isinstance(f.this, E.Table):
        return None
    tabs = [f.this]
    for j in sel.args.get('joins') or []:
        if not isinstance(j.this, E.Table) or j.args.get('using'):
            return None
        if (j.args.get('side') or '') or (j.args.get('kind') or '').upper() not in ('', 'INNER', 'CROSS'):
            return None
        tabs.append(j.this)
    names = [t.name for t in tabs]
    return names if len(set(names)) == len(names) else None


def _owner(db, names):
    own = {}
    for t in names:
        for c in db.cat.column_names(t):
            if c in own:
                return None                     # a bare name in two tables: decline
            own[c] = t
    return own


def _tabs_of(node, own):
    out = set()
    for c in node.find_all(E.Column):
        t = own.get(c.name)
        if t is None:
            out.add(None)
        else:
            out.add(t)
    return out


def _per_key_spec(db, tree, conjs, own, otabs):
    """the PER-KEY EXTREME conjunct: (index in conjs, value column name, key column name, 'min'|'max'), or None"""
    import wdb_subquery as WS
    found = None
    for ix, cj in enumerate(conjs):
        if cj.find(E.Subquery) is None:
            continue
        if found is not None or not isinstance(cj, E.EQ):
            return None
        a, b = cj.this, cj.expression
        if isinstance(a, E.Subquery):
            a, b = b, a
        if not (isinstance(a, E.Column) and isinstance(b, E.Subquery)) or own.get(a.name) is None:
            return None
        inner = b.this
        if not isinstance(inner, E.Select) or any(x is not inner for x in inner.find_all(E.Select)):
            return None
        for k in ('group', 'having', 'order', 'limit', 'offset', 'distinct', 'with', 'qualify'):
            if inner.args.get(k):
                return None
        if len(inner.expressions) != 1:
            return None
        ag = inner.expressions[0]
        ag = ag.this if isinstance(ag, E.Alias) else ag
        if not isinstance(ag, (E.Min, E.Max)) or not isinstance(ag.this, E.Column) or ag.this.name != a.name:
            return None
        itabs = _tables(inner)
        if itabs is None or not set(itabs) <= set(otabs):
            return None
        iown = {c: t for c, t in own.items() if t in itabs}
        icon = _conj(inner.args['where'].this if inner.args.get('where') else None)
        corr = [c for c in icon if any(col.name not in iown for col in c.find_all(E.Column))]
        if len(corr) != 1 or not isinstance(corr[0], E.EQ):
            return None
        x, y = corr[0].this, corr[0].expression
        if not (isinstance(x, E.Column) and isinstance(y, E.Column)):
            return None
        if x.name in iown and y.name not in iown:
            gk, ok = x, y
        elif y.name in iown and x.name not in iown:
            gk, ok = y, x
        else:
            return None
        extra = set(otabs) - set(itabs)
        okt = own.get(ok.name)
        if okt is None or okt not in extra:
            return None
        rest_inner = {WS._sr_norm9(c) for c in icon if c is not corr[0]}
        corr_n = WS._sr_norm9(corr[0])
        oc = [c for i, c in enumerate(conjs) if i != ix]
        oc_n = [WS._sr_norm9(c) for c in oc]
        if corr_n not in oc_n:
            return None                         # the outer must itself tie the key to the outer key
        oc_inner = set()
        for c, cn in zip(oc, oc_n):
            tt = _tabs_of(c, own)
            if None in tt:
                return None
            if tt <= set(itabs):
                oc_inner.add(cn)
            elif cn != corr_n and not tt <= extra:
                return None                     # a condition joining the subquery's tables to the others
        if oc_inner != rest_inner:
            return None                         # not the same rows
        from wdb_join import _solo_segment
        sk, _ = _solo_segment(db, okt)
        pk = db.cat.phys_map(okt).get(ok.name, ok.name)
        mk = sk.cols.get(pk)
        if mk is None or mk.get('has_null') or int(mk.get('V', -1)) != int(sk.N):
            return None                         # the outer key must be unique in its table
        found = (ix, a.name, gk.name, 'min' if isinstance(ag, E.Min) else 'max')
    return found


def _lit(n):
    if not isinstance(n, E.Literal):
        raise ValueError('not a literal')
    if n.is_string:
        return str(n.this)
    t = str(n.this)
    return float(t) if any(ch in t for ch in '.eE') else int(t)


_OPS = None


def _judge(n, vals):
    """a one-column predicate judged over a list of the column's Python values -> bool array (ValueError: decline)"""
    import operator, re
    global _OPS
    if _OPS is None:
        _OPS = {E.EQ: operator.eq, E.NEQ: operator.ne, E.GT: operator.gt, E.LT: operator.lt,
                E.GTE: operator.ge, E.LTE: operator.le}
    flip = {E.GT: E.LT, E.LT: E.GT, E.GTE: E.LTE, E.LTE: E.GTE, E.EQ: E.EQ, E.NEQ: E.NEQ}
    if isinstance(n, E.Paren):
        return _judge(n.this, vals)
    if isinstance(n, E.And):
        return _judge(n.this, vals) & _judge(n.expression, vals)
    if isinstance(n, E.Or):
        return _judge(n.this, vals) | _judge(n.expression, vals)
    if isinstance(n, E.Not):
        return ~_judge(n.this, vals)
    def each(f):
        try:
            return np.fromiter((v is not None and bool(f(v)) for v in vals), dtype=bool, count=len(vals))
        except TypeError:
            raise ValueError('type')
    if type(n) in _OPS:
        a, b, t = n.this, n.expression, type(n)
        if isinstance(a, E.Literal) and isinstance(b, E.Column):
            a, b, t = b, a, flip[t]
        if not isinstance(a, E.Column):
            raise ValueError('shape')
        x = _lit(b); op = _OPS[t]
        return each(lambda v: op(v, x))
    if isinstance(n, E.Between):
        lo = _lit(n.args['low']); hi = _lit(n.args['high'])
        return each(lambda v: lo <= v <= hi)
    if isinstance(n, E.In):
        if n.args.get('query') is not None or not n.args.get('expressions'):
            raise ValueError('in')
        xs = set(_lit(x) for x in n.args['expressions'])
        return each(lambda v: v in xs)
    if isinstance(n, (E.Like, E.ILike)):
        pat = _lit(n.expression)
        rx = re.compile('^' + re.escape(str(pat)).replace('%', '.*').replace('_', '.') + '$',
                        re.DOTALL | (re.IGNORECASE if isinstance(n, E.ILike) else 0))
        m = each(lambda v: isinstance(v, str) and rx.match(v) is not None)
        return ~m if n.args.get('negate') else m
    if isinstance(n, E.Is) and isinstance(n.expression, E.Null):
        return np.fromiter((v is None for v in vals), dtype=bool, count=len(vals))
    raise ValueError('kind')


def try_door(db, tree, single=False, values=False):
    """THE PROJECTION DOOR: (rows, names), or None. single: a one-table query (taken only with a value-per-row
    comparison). values: one selected numeric column, answered as an array of its values (an IN's inner query,
    wdb_insub -- no Python rows)."""
    import wdb_join, wdb_sql, wdb_corrrow, wdb_insub
    if not _PROJ_DOOR[0] or not isinstance(tree, E.Select) or not (tree.args.get('joins') or single):
        return None
    for k in ('group', 'having', 'distinct', 'qualify', 'offset', 'with'):
        if tree.args.get(k):
            return None
    proj = list(tree.expressions)
    pcols = []
    for p in proj:
        q = p.this if isinstance(p, E.Alias) else p
        if not isinstance(q, E.Column):
            return None
        pcols.append(q)
    otabs = _tables(tree)
    if otabs is None:
        return None
    own = _owner(db, otabs)
    if own is None or any(own.get(c.name) is None for c in pcols):
        return None
    where = tree.args.get('where')
    conjs = _conj(where.this if where is not None else None)
    spec = None
    rowval, insub = {}, set()               # conjunct index -> THE VALUE PER ROW's spec; IN conjuncts served as codes
    if any(c.find(E.Subquery) is not None or c.find(E.Exists) is not None for c in conjs):
        # lore: the per-key extreme is TPC-H Q2's shape
        spec = _per_key_spec(db, tree, conjs, own, otabs)      # THE PER-KEY EXTREME first
    if spec is None and any(c.find(E.Subquery) is not None or c.find(E.Exists) is not None for c in conjs):
        for i, c in enumerate(conjs):
            if c.find(E.Subquery) is None and c.find(E.Exists) is None:
                continue
            sv = wdb_corrrow.spec(db, c, own)
            if sv is not None:
                rowval[i] = sv
            elif wdb_insub.in_conj(c):
                insub.add(i)
        if not (rowval or insub) or any((c.find(E.Subquery) is not None or c.find(E.Exists) is not None)
                                        for i, c in enumerate(conjs) if i not in rowval and i not in insub):
            return None                         # a subquery none of the three takes
    if single and not rowval:
        return None
    # the ORDER BY terms as columns
    oterms = []
    order = tree.args.get('order')
    aliases = {wdb_sql._alias(p): q for p, q in zip(proj, pcols)}
    for oe in (order.expressions if order is not None else []):
        t = oe.this
        if isinstance(t, E.Literal) and not t.is_string:
            i = int(str(t.this)) - 1
            if not 0 <= i < len(pcols):
                return None
            col = pcols[i]
        elif isinstance(t, E.Column) and own.get(t.name) is not None:
            col = t
        elif isinstance(t, E.Column) and t.name in aliases:
            col = aliases[t.name]
        else:
            return None
        oterms.append((col, bool(oe.args.get('desc'))))
    # 2. the survivors
    t2 = tree.copy()
    keep = []
    for i, c in enumerate(_conj(t2.args['where'].this if t2.args.get('where') else None)):
        if (spec is not None and i == spec[0]) or i in rowval:
            continue
        if i in insub:
            c = wdb_insub.serve_conj(db, tree, conjs[i])
            if c is None:
                return None
            keep.append(c)
        else:
            keep.append(c.copy())
    w2 = None
    for c in keep:                              # And by hand: the code sets ride their nodes uncopied
        w2 = c if w2 is None else E.And(this=w2, expression=c)
    t2.set('where', E.Where(this=w2) if keep else None)
    for k in ('order', 'limit'):
        t2.set(k, None)
    try:
        r = wdb_join.join_rows(db, t2)
    except (wdb_join._FastUnsupported, NotImplementedError):
        return None
    rows, resolve = r['rows'], r['resolve']

    def at(col, rws):
        sg, pc, cp = resolve(E.column(col.name))
        if sg._overrides(pc) is not None or sg.cols[pc].get('has_null'):
            raise wdb_join._FastUnsupported
        return sg, pc, (rws if cp is None else np.asarray(wdb_join._at_rows(cp, rws)))
    try:
        # 2b. what the cascade left unjudged: one-column conjuncts, judged over the smaller of the dictionary and the
        # survivors' own values, then the survivors kept by code
        for cj in r.get('left') or []:
            if rows.size == 0:
                break
            if cj.find(E.Select) is not None:
                return None
            names = {c.name for c in cj.find_all(E.Column)}
            if len(names) != 1:
                return None
            col = next(cj.find_all(E.Column))
            sg, pc, pr = at(col, rows)
            V = int(sg.cols[pc]['V'])
            try:
                if V <= rows.size:
                    keep = _judge(cj, list(sg.values_at(pc, np.arange(V))))
                    rows = rows[keep[np.asarray(sg.codes_at(pc, pr)).astype(np.int64)]]
                else:
                    rows = rows[_judge(cj, list(sg.values_at_rows(pc, pr)))]
            except ValueError:
                return None
        # 2c. THE VALUE PER ROW: each comparison against a correlated scalar subquery, judged at the survivors
        for i in sorted(rowval):
            rows = wdb_corrrow.keep(db, rowval[i], lambda nm, rws: at(E.column(nm), rws), rows)
        # 3. the per-key extreme over the key's runs
        if spec is not None:
            _, vname, kname, kind = spec
            sk, pk, prk = at(E.column(kname), rows)
            if prk is not rows:
                return None                     # the key lives on the fact
            K = np.asarray(sk.codes_at(pk, rows)).astype(np.int64)
            sv, pv, prv = at(E.column(vname), rows)
            dv = np.asarray(sv._typed_dict(pv))
            if dv.dtype.kind not in 'iuf':
                return None
            V = dv[np.asarray(sv.codes_at(pv, prv))]
            if K.size:
                if K.size == 1 or bool(np.all(K[1:] >= K[:-1])):        # runs: each key's rows sit together
                    brk = np.empty(K.size, bool); brk[0] = True; brk[1:] = K[1:] != K[:-1]
                    st = np.flatnonzero(brk)
                    ext = (np.minimum if kind == 'min' else np.maximum).reduceat(V, st)
                    rid = np.cumsum(brk) - 1
                else:
                    uk, rid = np.unique(K, return_inverse=True)
                    ext = np.full(uk.size, np.inf if kind == 'min' else -np.inf)
                    (np.minimum if kind == 'min' else np.maximum).at(ext, rid, V)
                rows = rows[V == ext[rid]]
            _PER_KEY_HITS[0] += 1
        # 4. ORDER BY on the kept rows
        if oterms and rows.size:
            keys = []
            for col, desc in oterms:
                sg, pc, pr = at(col, rows)
                vals = sg.values_at_rows(pc, pr)
                a = np.asarray(vals)
                if a.dtype.kind in 'iuf':
                    k = a.astype(np.float64) if a.dtype.kind == 'f' or a.size == 0 or max(int(a.max()), -int(a.min())) < (1 << 53) else None
                    if k is None:
                        return None
                    if np.isnan(k).any():
                        return None
                    k = -k if desc else k
                else:
                    if any(v is None for v in vals):
                        return None
                    _, inv = np.unique(np.asarray(vals, dtype=object), return_inverse=True)
                    k = (-inv if desc else inv).astype(np.int64)
                keys.append(k)
            rows = rows[np.lexsort(tuple(reversed(keys)))]
        # 5. LIMIT
        lim = wdb_sql._limit(tree)
        if lim is not None:
            rows = rows[:int(lim)]
        # 6. the selected columns at the winners only
        if values:
            if len(pcols) != 1:
                return None
            sg, pc, pr = at(pcols[0], rows)
            td = np.asarray(sg._typed_dict(pc))
            if td.dtype.kind not in 'iuf':
                return None
            _PROJ_DOOR_HITS[0] += 1
            return td[np.asarray(sg.codes_at(pc, pr))]
        cols = []
        for col in pcols:
            sg, pc, pr = at(col, rows)
            cols.append(list(sg.values_at_rows(pc, pr)))
    except wdb_join._FastUnsupported:
        return None
    _PROJ_DOOR_HITS[0] += 1
    return list(zip(*cols)) if cols else [], [wdb_sql._alias(p) for p in proj]
