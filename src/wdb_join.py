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
import wdb_sql, wdb_dml, wdb_agg, wdb_fkptr

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
        seg = db.open_segment(paths[0], table)
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

    fast = _fast_detect(db, lt, la, rt, ra, lk, rk)
    if fast is not None:
        try:
            return _fast_pointer_agg(db, tree, **fast)
        except _FastUnsupported:
            pass   # fall back to the always-correct pandas hash-join path below

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


# ── FK-pointer gather fast path ──────────────────────────────────────────────
# When a join's ON matches a stored foreign-key pointer (child.fk = parent.key), the join is already
# resolved: we gather parent columns by the pointer instead of hash-merging, and aggregate on WaveDB's
# integer codes with the bincount kernel. Narrow by design -- single join, one group key, GROUP BY +
# aggregates -- and raises _FastUnsupported for anything else so join_query falls back to pandas.

class _FastUnsupported(Exception):
    pass

_FAST_HITS = 0   # diagnostic: how many queries took the gather fast path


def _fast_detect(db, lt, la, rt, ra, lk, rk):
    fkl = db.cat.fk_pointers(lt)
    if lk in fkl and fkl[lk]['parent'] == rt and fkl[lk]['parent_key'] == rk:
        return dict(child=lt, parent=rt, child_alias=la, parent_alias=ra, fk_col=lk)
    fkr = db.cat.fk_pointers(rt)
    if rk in fkr and fkr[rk]['parent'] == lt and fkr[rk]['parent_key'] == lk:
        return dict(child=rt, parent=lt, child_alias=ra, parent_alias=la, fk_col=rk)
    return None


def _solo_segment(db, name):
    paths = db.cat.segment_paths(name)
    if len(paths) != 1: raise _FastUnsupported
    if os.path.exists(wdb_dml.hot_path(db.cat, name)): raise _FastUnsupported
    seg = db.open_segment(paths[0], name)
    if seg.presence_mask() is not None: raise _FastUnsupported
    return seg, paths[0]


def _code_val(seg, pcol, code):
    c = seg.cols[pcol]
    if c.get('has_null') and int(code) == c['V'] - 1: return None
    return seg.fetch(pcol, int(code))


def _fast_pointer_agg(db, tree, child, parent, child_alias, parent_alias, fk_col):
    import operator
    proj = tree.expressions
    if not any(wdb_sql._agg_kind(p) for p in proj): raise _FastUnsupported   # plain projection -> fallback
    group = tree.args.get('group')
    gnodes = group.expressions if group is not None else []
    if len(gnodes) > 1: raise _FastUnsupported

    cseg, csp = _solo_segment(db, child)
    pseg, _   = _solo_segment(db, parent)
    ptr = db.fk_pointer(csp, fk_col)
    if ptr is None: raise _FastUnsupported

    ccols = set(db.cat.column_names(child)); pcols = set(db.cat.column_names(parent))
    cphys = db.cat.phys_map(child);          pphys = db.cat.phys_map(parent)

    def resolve(node):
        a, nm = node.table, node.name
        if a == child_alias or (not a and nm in ccols and nm not in pcols): return cseg, cphys.get(nm, nm), False
        if a == parent_alias or (not a and nm in pcols and nm not in ccols): return pseg, pphys.get(nm, nm), True
        raise _FastUnsupported
    def col_native(node):
        seg, pcol, gathered = resolve(node)
        arr, nm = wdb_sql._col(seg, pcol)
        if gathered:
            arr = arr[ptr]; nm = nm[ptr] if nm is not None else None
        return arr, nm, seg, pcol

    # ---- WHERE -> boolean mask over child rows ----
    _OPS = {E.EQ: operator.eq, E.NEQ: operator.ne, E.GT: operator.gt, E.LT: operator.lt,
            E.GTE: operator.ge, E.LTE: operator.le}
    def mask_eval(node):
        if isinstance(node, E.Paren): return mask_eval(node.this)
        if isinstance(node, E.And): return mask_eval(node.this) & mask_eval(node.expression)
        if isinstance(node, E.Or):  return mask_eval(node.this) | mask_eval(node.expression)
        if isinstance(node, E.Not): return ~mask_eval(node.this)
        if type(node) in _OPS:
            arr, _, seg, pcol = col_native(node.this)
            v = wdb_sql._lit_for_col(seg, pcol, node.expression, arr.dtype.kind)
            return _OPS[type(node)](arr, v)
        if isinstance(node, E.Between):
            arr, _, seg, pcol = col_native(node.this)
            lo = wdb_sql._lit_for_col(seg, pcol, node.args['low'], arr.dtype.kind)
            hi = wdb_sql._lit_for_col(seg, pcol, node.args['high'], arr.dtype.kind)
            return (arr >= lo) & (arr <= hi)
        if isinstance(node, E.In):
            arr, _, seg, pcol = col_native(node.this)
            vals = [wdb_sql._lit_for_col(seg, pcol, e, arr.dtype.kind) for e in (node.args.get('expressions') or [])]
            return np.isin(arr, vals)
        raise _FastUnsupported
    where = tree.args.get('where')
    mask = mask_eval(where.this) if where is not None else None
    sel = (lambda a: a if mask is None else a[mask])

    # ---- group codes ----
    if gnodes:
        gseg, gpcol, ggath = resolve(gnodes[0])
        if gseg.cols[gpcol]['mode'] == 4: raise _FastUnsupported   # codes not value-identity
        full = gseg.codes(gpcol)
        gcodes = sel(full[ptr] if ggath else full).astype(np.int64)
        if len(gcodes) == 0: return [], [wdb_sql._alias(p) for p in proj]
        K = int(gcodes.max()) + 1
    else:
        n = int(mask.sum()) if mask is not None else len(ptr)
        gcodes = np.zeros(n, dtype=np.int64); K = 1; gseg = gpcol = None

    counts = wdb_agg.group_counts(gcodes, K)
    present = np.nonzero(counts > 0)[0]

    # ---- per-projection results ----
    _CLS = {E.Sum: 'SUM', E.Avg: 'AVG', E.Min: 'MIN', E.Max: 'MAX'}
    col_results = {}
    for i, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Count) and (isinstance(inner.this, E.Star) or inner.this is None):
            col_results[i] = ('count',)
        elif isinstance(inner, E.Count):
            arr, nm, seg, pcol = col_native(inner.this)
            col_results[i] = ('arr', wdb_agg.group_agg(gcodes, K, 'COUNT', sel(arr), sel(nm) if nm is not None else None), False, None)
        elif type(inner) in _CLS:
            fn = _CLS[type(inner)]; arr, nm, seg, pcol = col_native(inner.this)
            res = wdb_agg.group_agg(gcodes, K, fn, sel(arr), sel(nm) if nm is not None else None)
            is_dt = seg.cols[pcol]['dt'] == 3 and fn in ('MIN', 'MAX')
            col_results[i] = ('arr', res, is_dt, (seg.unit(pcol) if is_dt else None))
        else:
            if gseg is None: raise _FastUnsupported   # bare key column without GROUP BY
            col_results[i] = ('key',)

    # ---- assemble rows ----
    rows = []
    for code in present:
        row = []
        for i, p in enumerate(proj):
            r = col_results[i]
            if r[0] == 'key':
                row.append(wdb_sql._pyval(_code_val(gseg, gpcol, code)))
            elif r[0] == 'count':
                row.append(int(counts[code]))
            else:
                v = r[1][code]
                if r[2] and v is not None:   # datetime MIN/MAX: int64 epoch -> datetime64
                    v = np.int64(v).view(f"datetime64[{r[3]}]")
                row.append(wdb_sql._pyval(v))
        rows.append(tuple(row))

    global _FAST_HITS; _FAST_HITS += 1
    rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))
    lim = wdb_sql._limit(tree)
    if lim is not None: rows = rows[:lim]
    return rows, [wdb_sql._alias(p) for p in proj]
