"""SEGMENT PARTIALS FOR JOINS: a join whose FACT table lives in several segments
runs once per segment (a catalog override pins the table to one segment; every
door then sees a solo segment) and the per-segment results merge algebraically --
the single-table path's law (wdb_merge): AVG -> SUM+COUNT, SUM -> SUM, MIN/MAX,
COUNT(*) -> count, partials merged per group key, HAVING/ORDER/LIMIT applied once.
Row-emitting joins concatenate. Exactly one multi-segment table is supported (two
would need segment-pair joins); COUNT(DISTINCT), MEDIAN, windows and other
non-mergeable shapes decline by name."""
import copy
import sqlglot
from sqlglot import exp as E
import wdb_merge


class _NotMergeable(NotImplementedError):
    pass


def multi_segment_tables(db, tree):
    """aliases -> table for every table in the FROM/JOIN with >1 segment"""
    out = {}
    frm = tree.args.get('from') or tree.args.get('from_')
    tabs = []
    if frm is not None and isinstance(frm.this, E.Table): tabs.append(frm.this)
    for jn in (tree.args.get('joins') or []):
        if isinstance(jn.this, E.Table): tabs.append(jn.this)
    for t in tabs:
        try:
            if len(db.cat.segment_paths(t.name)) > 1:
                out[t.alias or t.name] = t.name
        except Exception:
            pass
    return out


class _Pin:
    """Pin one table to one segment for the duration of a block (a catalog override)."""
    def __init__(self, db, name, path):
        self.db, self.name, self.path = db, name, path
    def __enter__(self):
        ov = getattr(self.db.cat, '_seg_override', None)
        if ov is None:
            ov = self.db.cat._seg_override = {}
        self._prev = ov.get(self.name)
        ov[self.name] = [self.path]
        return self
    def __exit__(self, *a):
        ov = self.db.cat._seg_override
        if self._prev is None: ov.pop(self.name, None)
        else: ov[self.name] = self._prev


def execute(db, tree, sql):
    ms = multi_segment_tables(db, tree)
    if not ms: return None
    if len(ms) > 1:
        raise NotImplementedError('segment partials: %d multi-segment tables in one join (%s); one is supported' % (len(ms), ', '.join(ms.values())))
    if tree.find(E.Window) is not None or tree.find(E.Subquery) is not None:
        raise NotImplementedError('segment partials: windows / subqueries over a multi-segment join')
    name = next(iter(ms.values()))
    paths = db.cat.segment_paths(name)
    proj = tree.expressions
    has_agg = any(isinstance(wdb_merge._inner(p), wdb_merge._AGG) for p in proj) or tree.find(E.AggFunc) is not None
    group = tree.args.get('group')
    if not has_agg and group is None:
        # plain row union: strip ORDER/LIMIT per segment, apply once
        base = copy.deepcopy(tree); base.set('order', None); base.set('limit', None); base.set('offset', None)
        rows = []; names = None
        for p in paths:
            with _Pin(db, name, p):
                r = db.run(base.sql(dialect='duckdb'))
            rr, names = (r if isinstance(r, tuple) else (r, None))
            rows += list(rr)
        import wdb_sql
        rows = wdb_sql._apply_order(rows, list(tree.expressions), tree.args.get('order'))
        lim = wdb_sql._limit(tree); off = wdb_sql._offset(tree)
        if lim is not None or off:
            rows = rows[off: off + lim] if lim is not None else rows[off:]
        return rows, names or [wdb_sql._alias(p) for p in proj]
    # aggregates: mergeable partials per segment, merged once
    for p in proj:
        nd = wdb_merge._inner(p)
        if wdb_merge._is_cdistinct(nd) or (isinstance(nd, E.Count) and isinstance(nd.this, E.Distinct)):
            raise NotImplementedError('segment partials: COUNT(DISTINCT) is not mergeable across segments')
        if not isinstance(nd, (E.Column, E.Count, E.Sum, E.Min, E.Max, E.Avg)):
            raise NotImplementedError('segment partials: %s is not mergeable across segments' % type(nd).__name__)
    # keys keep their ORIGINAL (possibly qualified) expressions -- a join's key may be t.title
    keys, plan, partial_exprs = wdb_merge._classify(proj)
    key_nodes = {}
    for p in proj:
        nd = wdb_merge._inner(p)
        if isinstance(nd, E.Column) and nd.name not in key_nodes: key_nodes[nd.name] = nd
    base = copy.deepcopy(tree)
    base.set('having', None); base.set('order', None); base.set('limit', None); base.set('offset', None)
    sel = []
    for i, k in enumerate(keys):
        sel.append(E.Alias(this=key_nodes[k].copy(), alias=E.Identifier(this='_k%d' % i, quoted=False)))
    for j, pe in enumerate(partial_exprs):
        sel.append(E.Alias(this=sqlglot.parse_one('SELECT %s' % pe, read='duckdb').expressions[0], alias=E.Identifier(this='_p%d' % j, quoted=False)))
    base.set('expressions', sel)
    psql = base.sql(dialect='duckdb')
    row_lists = []
    for p in paths:
        with _Pin(db, name, p):
            r = db.run(psql)
        row_lists.append(list(r[0] if isinstance(r, tuple) else r))
    acc = wdb_merge._merge_partials(row_lists, len(keys), plan)
    import wdb_sql
    out = []
    for key, parts in acc.items():
        full = list(key) + list(parts)
        row = []
        for entry in plan:
            if entry[0] == 'key': row.append(key[entry[1]])
            elif entry[0] in ('sum', 'count_star', 'min', 'max'): row.append(full[entry[1]])
            elif entry[0] == 'avg':
                sm = full[entry[1]]; n = full[entry[2]]; row.append(None if not n else sm / n)
        out.append(tuple(row))
    having = tree.args.get('having')
    if having is not None:
        out = wdb_sql._apply_having(out, proj, having.this, lambda x: x)
    out = wdb_sql._apply_order(out, proj, tree.args.get('order'))
    lim = wdb_sql._limit(tree); off = wdb_sql._offset(tree)
    if lim is not None or off:
        out = out[off: off + lim] if lim is not None else out[off:]
    return out, [wdb_sql._alias(p) for p in proj]
