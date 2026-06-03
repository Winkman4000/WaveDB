"""WaveDB JOIN executor — step 1: two-table INNER equi-join.

Decode the needed columns from each side via WaveDB's own read path, join them, then run the SELECT
clauses (projection / WHERE / GROUP BY / aggregates / ORDER BY / LIMIT). The join itself has two
kernels: a hash join (baseline, any equi-join) and -- when the child join key is a pre-resolved
foreign-key POINTER into the parent's rows -- a gather join (array index, no hash build). This module
is the hash baseline + the shared post-join evaluator; the gather kernel layers on top.
Correctness first: verified against DuckDB. Unsupported shapes raise NotImplementedError.
"""
import sqlglot, sqlglot.expressions as E
import numpy as np, pandas as pd, os
from wdb_engine import Segment
import wdb_sql, wdb_dml

_CMP = {E.EQ: '==', E.NEQ: '!=', E.GT: '>', E.LT: '<', E.GTE: '>=', E.LTE: '<='}


def _materialize(db, table, cols):
    """Return {col: np.array} for the given LOGICAL columns of `table`, decoded via WaveDB. Fast
    path (single live segment) reads columns directly; otherwise fall back to the row SELECT path
    so presence/synth/merge correctness is inherited."""
    cols = list(dict.fromkeys(cols))
    phys = db.cat.phys_map(table)
    paths = db.cat.segment_paths(table)
    hp = wdb_dml.hot_path(db.cat, table); hot = os.path.exists(hp)
    if len(paths) == 1 and not hot:
        seg = Segment(paths[0]); wdb_dml.register_synth(db.cat, seg, table)
        if seg.presence_mask() is None:
            return {c: seg.values(phys.get(c, c)) for c in cols}
    rows, _ = db.run(f"SELECT {', '.join(cols)} FROM {table}")
    arrs = list(zip(*rows)) if rows else [()] * len(cols)
    return {c: np.array(arrs[i], dtype=object) for i, c in enumerate(cols)}


def _all_columns(node):
    """Every (table_alias, colname) referenced under a node."""
    return [(c.table, c.name) for c in node.find_all(E.Column)]


def join_query(db, sql):
    tree = sqlglot.parse_one(sql, read='duckdb')
    joins = tree.args.get('joins')
    if not joins or len(joins) != 1:
        raise NotImplementedError("join: exactly one JOIN supported (step 1)")
    jn = joins[0]
    if (jn.args.get('side') or jn.args.get('kind')):
        raise NotImplementedError("join: only INNER JOIN supported (step 1)")
    frm = tree.find(E.From).this
    lt, la = frm.name, (frm.alias or frm.name)
    rt, ra = jn.this.name, (jn.this.alias or jn.this.name)
    on = jn.args.get('on')
    if not isinstance(on, E.EQ):
        raise NotImplementedError("join: ON must be a single equality (step 1)")
    # which side of the ON belongs to which table
    a2t = {la: lt, ra: rt}
    le, re = on.this, on.expression
    if a2t.get(le.table) == lt and a2t.get(re.table) == rt:
        lk, rk = le.name, re.name
    elif a2t.get(le.table) == rt and a2t.get(re.table) == lt:
        lk, rk = re.name, le.name
    else:
        raise NotImplementedError("join: ON columns must reference the two joined tables")

    lcols = {c[0] for c in db.cat.get_table(lt)['schema'].__iter__()} if False else set(db.cat.column_names(lt))
    rcols = set(db.cat.column_names(rt))

    def resolve(tbl_alias, name):
        """(alias, col) -> merged-frame key 'alias.col'. Unqualified resolves by membership."""
        if tbl_alias:
            return f"{tbl_alias}.{name}"
        if name in lcols and name in rcols:
            raise NotImplementedError(f"ambiguous column {name!r}")
        return f"{la}.{name}" if name in lcols else f"{ra}.{name}"

    # collect needed columns per table from the whole statement
    need_l, need_r = {lk}, {rk}
    for part in (tree.expressions, [tree.args.get('where')], (tree.args.get('group').expressions if tree.args.get('group') else []),
                 (tree.args.get('order').expressions if tree.args.get('order') else [])):
        for node in part:
            if node is None: continue
            for talias, cname in _all_columns(node):
                if talias == la or (not talias and cname in lcols and cname not in rcols): need_l.add(cname)
                elif talias == ra or (not talias and cname in rcols and cname not in lcols): need_r.add(cname)
                elif not talias and cname in lcols and cname in rcols:
                    raise NotImplementedError(f"ambiguous column {cname!r}")

    lc = _materialize(db, lt, need_l)
    rc = _materialize(db, rt, need_r)
    ldf = pd.DataFrame({f"{la}.{c}": lc[c] for c in lc})
    rdf = pd.DataFrame({f"{ra}.{c}": rc[c] for c in rc})
    merged = ldf.merge(rdf, left_on=f"{la}.{lk}", right_on=f"{ra}.{rk}", how='inner')

    R = lambda colnode: resolve(colnode.table, colnode.name)
    where = tree.args.get('where')
    if where is not None:
        merged = merged[_mask(merged, where.this, R)]

    proj = tree.expressions
    group = tree.args.get('group')
    has_agg = any(wdb_sql._agg_kind(p) for p in proj)
    if group is not None or has_agg:
        rows = _aggregate(merged, proj, group, R)
    else:
        keys = [R(p.this if isinstance(p, E.Alias) else p) for p in proj]
        rows = [tuple(_render(v) for v in t) for t in merged[keys].itertuples(index=False, name=None)]

    rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))
    lim = wdb_sql._limit(tree)
    if lim is not None: rows = rows[:lim]
    return rows, [wdb_sql._alias(p) for p in proj]


def _render(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    if isinstance(v, pd.Timestamp): v = v.to_numpy()
    return wdb_sql._pyval(v)


def _coerce_lit(series, lit):
    if isinstance(lit, E.Neg):
        return -_coerce_lit(series, lit.this)
    k = series.dtype.kind
    if k == 'M':  # datetime
        return pd.Timestamp(str(lit.this))
    if k in 'iuf':
        return float(lit.this) if (k == 'f' or '.' in str(lit.this)) else int(lit.this)
    s = lit.this
    return s.encode() if isinstance(s, str) else s


def _mask(df, node, R):
    import operator
    if isinstance(node, E.Paren): return _mask(df, node.this, R)
    if isinstance(node, E.And): return _mask(df, node.this, R) & _mask(df, node.expression, R)
    if isinstance(node, E.Or): return _mask(df, node.this, R) | _mask(df, node.expression, R)
    if isinstance(node, E.Not): return ~_mask(df, node.this, R)
    if type(node) in _CMP:
        s = df[R(node.this)]; v = _coerce_lit(s, node.expression)
        op = {E.EQ: operator.eq, E.NEQ: operator.ne, E.GT: operator.gt,
              E.LT: operator.lt, E.GTE: operator.ge, E.LTE: operator.le}[type(node)]
        return op(s, v)
    if isinstance(node, E.Between):
        s = df[R(node.this)]; lo = _coerce_lit(s, node.args['low']); hi = _coerce_lit(s, node.args['high'])
        return (s >= lo) & (s <= hi)
    if isinstance(node, E.In):
        s = df[R(node.this)]; vals = [_coerce_lit(s, L) for L in (node.args.get('expressions') or [])]
        return s.isin(vals)
    raise NotImplementedError(f"join WHERE: {type(node).__name__}")


def _aggregate(merged, proj, group, R):
    _PF = {'SUM': 'sum', 'AVG': 'mean', 'MIN': 'min', 'MAX': 'max', 'COUNT': 'count'}
    specs = []  # per projection: ('key', col) | ('size',) | ('agg', fn, col)
    for i, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        kind = wdb_sql._agg_kind(p)
        if kind is None:
            specs.append(('key', R(inner)))
        elif kind[0] == 'COUNT_STAR':
            specs.append(('size',))
        else:
            specs.append(('agg', kind[0], R(inner.this)))
    if group is not None:
        key_cols = [R(g) for g in group.expressions]
        g = merged.groupby(key_cols, sort=False, dropna=False)
        named = {f"_a{i}": pd.NamedAgg(column=s[2], aggfunc=_PF[s[1]]) for i, s in enumerate(specs) if s[0] == 'agg'}
        agg = g.agg(**named) if named else g.size().to_frame('_dummy')
        if any(s[0] == 'size' for s in specs):
            agg['_size'] = g.size()
        agg = agg.reset_index()
        out = []
        for _, r in agg.iterrows():
            row = []
            for i, s in enumerate(specs):
                if s[0] == 'key': row.append(_render(r[s[1]]))
                elif s[0] == 'size': row.append(int(r['_size']))
                else: row.append(_render(r[f"_a{i}"]))
            out.append(tuple(row))
        return out
    # whole-table aggregate -> single row
    row = []
    for i, s in enumerate(specs):
        if s[0] == 'size': row.append(int(len(merged)))
        elif s[0] == 'agg':
            col = merged[s[2]]
            row.append(_render({'SUM': col.sum(), 'AVG': col.mean(), 'MIN': col.min(),
                                'MAX': col.max(), 'COUNT': col.count()}[s[1]]))
        else:
            raise NotImplementedError("bare column with aggregates but no GROUP BY")
    return [tuple(row)]
