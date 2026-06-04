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
import wdb_sql, wdb_dml, wdb_agg, wdb_fkptr, wdb_exprjit

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
    # FK-pointer fast path: handles 1..N joins as a chain/star of pre-resolved pointers.
    try:
        return _fast_pointer_agg(db, tree, _build_chain(db, tree))
    except _FastUnsupported:
        pass   # not an FK-pointer aggregate query -> fall back to the pandas hash-join path
    if not joins or len(joins) != 1:
        raise NotImplementedError("join: multi-join requires FK pointers; fallback supports one INNER join")
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

    # (FK-pointer fast path already attempted above via _build_chain)

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

MULTI_GROUP_CEIL = 1 << 18   # max composite groups for dense multi-col GROUP BY (else -> hashing/fallback)

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


def _fast_pointer_agg(db, tree, ctx):
    import operator
    proj = tree.expressions
    if not any(wdb_sql._agg_kind(p) for p in proj): raise _FastUnsupported   # plain projection -> fallback
    group = tree.args.get('group')
    gnodes = group.expressions if group is not None else []

    fact = ctx['fact']; alias2t = ctx['alias2t']; seg_of = ctx['seg_of']; composed = ctx['composed']
    cols_of = {a: set(db.cat.column_names(t)) for a, t in alias2t.items()}
    phys_of = {a: db.cat.phys_map(t) for a, t in alias2t.items()}

    _colmemo = {}
    def _col_cached(seg, pcol):                 # decode each column once per query (multi-agg reuse)
        k = (id(seg), pcol)
        if k not in _colmemo: _colmemo[k] = wdb_sql._col(seg, pcol)
        return _colmemo[k]

    def resolve(node):
        a, nm = node.table, node.name
        if not a:                               # unqualified: find the unique table owning the column
            owners = [al for al, cs in cols_of.items() if nm in cs]
            if len(owners) != 1: raise _FastUnsupported
            a = owners[0]
        if a not in alias2t or nm not in cols_of[a]: raise _FastUnsupported
        return seg_of[a], phys_of[a].get(nm, nm), composed[a]   # composed[a] is None for the fact table
    def col_operand(node):
        # parent columns become ('g', arr, composed_ptr) so the gather happens per-chunk inside the
        # threaded kernel instead of materialising the full gathered array here.
        seg, pcol, cptr = resolve(node)
        if cptr is None:
            raw = wdb_sql.raw_dict_col(seg, pcol)         # plain dict numeric col -> defer/fuse the decode
            if raw is not None:
                return ('raw', raw[0], raw[1]), None, seg, pcol
        arr, nm = _col_cached(seg, pcol)
        if cptr is not None:
            return ('g', arr, cptr), (('g', nm, cptr) if nm is not None else None), seg, pcol
        return ('d', arr), (('d', nm) if nm is not None else None), seg, pcol

    _ARITH = {E.Add: operator.add, E.Sub: operator.sub, E.Mul: operator.mul, E.Div: operator.truediv}
    def eval_arith(node):
        # Materialise an arithmetic expression to a per-fact-row value array (+ combined null mask).
        # Each column is resolved through its composed pointer, so expressions may mix tables in the chain.
        if isinstance(node, (E.Paren, E.Cast)): return eval_arith(node.this)
        if isinstance(node, E.Neg):
            a, na = eval_arith(node.this); return -a, na
        if isinstance(node, E.Column):
            seg, pcol, cptr = resolve(node)
            arr, nm = _col_cached(seg, pcol)
            if cptr is not None:
                arr = arr[cptr]; nm = nm[cptr] if nm is not None else None
            return arr, nm
        if isinstance(node, E.Literal):
            if node.is_string: raise _FastUnsupported
            v = node.this
            return (float(v) if ('.' in v or 'e' in v.lower()) else int(v)), None
        if type(node) in _ARITH:
            a, na = eval_arith(node.this); b, nb = eval_arith(node.expression)
            out = _ARITH[type(node)](a, b)
            nm = na if nb is None else (nb if na is None else (na | nb))   # NULL if any operand is NULL
            return out, nm
        raise _FastUnsupported
    _ARITH_STR = {E.Add: '+', E.Sub: '-', E.Mul: '*', E.Div: '/'}
    def fused_expr_build(argnode):
        # Compile an arithmetic tree to (numba-source body over slot vars v0.., [(base, codes) per slot]) so
        # wdb_exprjit can fuse decode+expression+aggregate into one pass (no materialised array). Fact (direct)
        # dict-numeric columns only; raises _FastUnsupported on anything else so the caller materialises instead.
        if not wdb_exprjit.HAS_NUMBA: raise _FastUnsupported
        inputs = []; slot = {}
        def emit(node):
            if isinstance(node, (E.Paren, E.Cast)): return emit(node.this)
            if isinstance(node, E.Neg): return f"(-{emit(node.this)})"
            if isinstance(node, E.Column):
                seg, pcol, cptr = resolve(node)
                if cptr is not None: raise _FastUnsupported          # parent column: gathered decode not fused yet
                raw = wdb_sql.raw_dict_col(seg, pcol)
                if raw is None: raise _FastUnsupported               # nullable / string / computed -> fallback
                key = (id(seg), pcol)
                if key not in slot:
                    slot[key] = len(inputs)
                    inputs.append((np.ascontiguousarray(raw[0]), np.ascontiguousarray(raw[1])))
                return f"v{slot[key]}"
            if isinstance(node, E.Literal):
                if node.is_string: raise _FastUnsupported
                v = node.this
                return f"({float(v)})" if ('.' in v or 'e' in v.lower()) else f"({int(v)})"
            if type(node) in _ARITH_STR:
                return f"({emit(node.this)} {_ARITH_STR[type(node)]} {emit(node.expression)})"
            raise _FastUnsupported
        body = emit(argnode)
        if not inputs: raise _FastUnsupported                        # pure constant -> not an aggregation input
        return body, inputs

    def agg_arg_operand(argnode):
        # SUM/AVG/MIN/MAX/COUNT argument: a bare column keeps its seg/pcol (for datetime MIN/MAX); an
        # arithmetic expression is materialised to a direct ('d', arr) operand the kernel slices per chunk.
        if isinstance(argnode, E.Column):
            return col_operand(argnode)
        arr, nm = eval_arith(argnode)
        if not hasattr(arr, 'shape'): raise _FastUnsupported          # need a per-row array, not a constant
        return ('d', arr), (('d', nm) if nm is not None else None), None, None

    # ---- WHERE -> boolean mask over child rows ----
    # Each predicate is evaluated on the UN-gathered column (the small parent side when it is a parent
    # column) and the resulting bool is gathered to child rows -- and string =, !=, IN compare integer
    # CODES, never materialised strings. Both avoid touching a 6M-row object array (measured 76 -> ~3 ms).
    _OPS = {E.EQ: operator.eq, E.NEQ: operator.ne, E.GT: operator.gt, E.LT: operator.lt,
            E.GTE: operator.ge, E.LTE: operator.le}
    def _str_codes(seg, pcol):
        c = seg.cols[pcol]
        if c['dt'] != 1 or c['mode'] == 4: return None        # only value-identity string dicts
        if c['mode'] == 5: seg._raw_codes(pcol); dv = seg.cols[pcol].get('_idict')
        else:
            try: dv = seg.dict_vals(pcol)
            except Exception: return None
        if dv is None: return None
        code_of = {(v if isinstance(v, (bytes, bytearray)) else str(v).encode()): i for i, v in enumerate(dv)}
        return seg.codes(pcol), code_of, (c['V'] - 1 if c['has_null'] else None)
    def _lit_bytes(seg, pcol, e):
        b = wdb_sql._lit_for_col(seg, pcol, e, 'O')
        return b if isinstance(b, (bytes, bytearray)) else str(b).encode()
    def leaf(colnode, make_bool):
        seg, pcol, cptr = resolve(colnode)       # evaluate un-gathered, then gather the bool via composed ptr
        b = make_bool(seg, pcol)
        return b if cptr is None else b[cptr]
    def mask_eval(node):
        if isinstance(node, E.Paren): return mask_eval(node.this)
        if isinstance(node, E.And): return mask_eval(node.this) & mask_eval(node.expression)
        if isinstance(node, E.Or):  return mask_eval(node.this) | mask_eval(node.expression)
        if isinstance(node, E.Not): return ~mask_eval(node.this)
        if type(node) in _OPS:
            op = type(node)
            def mk(seg, pcol):
                if op in (E.EQ, E.NEQ):
                    sc = _str_codes(seg, pcol)
                    if sc is not None:                       # code comparison, no object materialisation
                        codes, code_of, nullcode = sc
                        tc = code_of.get(_lit_bytes(seg, pcol, node.expression), -1)
                        if op is E.EQ: return codes == tc
                        res = codes != tc
                        if nullcode is not None: res &= (codes != nullcode)   # SQL: NULL != x is not TRUE
                        return res
                arr, _ = _col_cached(seg, pcol)
                v = wdb_sql._lit_for_col(seg, pcol, node.expression, arr.dtype.kind)
                return _OPS[op](arr, v)
            return leaf(node.this, mk)
        if isinstance(node, E.Between):
            def mk(seg, pcol):
                arr, _ = _col_cached(seg, pcol)
                lo = wdb_sql._lit_for_col(seg, pcol, node.args['low'], arr.dtype.kind)
                hi = wdb_sql._lit_for_col(seg, pcol, node.args['high'], arr.dtype.kind)
                return (arr >= lo) & (arr <= hi)
            return leaf(node.this, mk)
        if isinstance(node, E.In):
            def mk(seg, pcol):
                exprs = node.args.get('expressions') or []
                sc = _str_codes(seg, pcol)
                if sc is not None:
                    codes, code_of, _ = sc
                    tcs = [code_of[b] for b in (_lit_bytes(seg, pcol, e) for e in exprs) if b in code_of]
                    return np.isin(codes, tcs)
                arr, _ = _col_cached(seg, pcol)
                vals = [wdb_sql._lit_for_col(seg, pcol, e, arr.dtype.kind) for e in exprs]
                return np.isin(arr, vals)
            return leaf(node.this, mk)
        raise _FastUnsupported
    where = tree.args.get('where')
    mask = mask_eval(where.this) if where is not None else None
    sel = (lambda a: a if mask is None else a[mask])

    # ---- group codes (as an operand; gathered per-chunk in the threaded kernel) ----
    n = ctx['n']
    if n == 0: return [], [wdb_sql._alias(p) for p in proj]
    mask_op = ('d', mask) if mask is not None else None
    gkeys = []                                           # one per GROUP BY column
    for g in gnodes:
        gseg, gpcol, gcptr = resolve(g)
        if gseg.cols[gpcol]['mode'] == 4: raise _FastUnsupported   # codes not value-identity
        full = gseg.codes(gpcol)
        if full.size == 0: return [], [wdb_sql._alias(p) for p in proj]
        gkeys.append({'seg': gseg, 'pcol': gpcol, 'cptr': gcptr, 'full': full, 'K': int(full.max()) + 1})
    if len(gkeys) == 0:
        K = 1; group_op = None
    elif len(gkeys) == 1:                                 # single key: keep the gather-fused operand
        k0 = gkeys[0]; K = k0['K']
        group_op = ('g', k0['full'], k0['cptr']) if k0['cptr'] is not None else ('d', k0['full'])
    else:                                                 # multi-key: mixed-radix composite code
        K = 1
        for k in gkeys: K *= k['K']
        if K > MULTI_GROUP_CEIL: raise _FastUnsupported   # dense composite would blow up -> needs hashing
        comp = np.zeros(n, dtype=np.int64)
        for k in gkeys:
            codes = k['full'][k['cptr']] if k['cptr'] is not None else k['full']
            comp = comp * k['K'] + codes.astype(np.int64, copy=False)
        group_op = ('d', np.ascontiguousarray(comp))

    # ---- per-projection results ----
    # COUNT/SUM/AVG become operand specs computed in one pass (threaded + per-chunk gather above a row
    # threshold); MIN/MAX run on the serial kernel; bare key columns map straight to the group value.
    _CLS = {E.Sum: 'SUM', E.Avg: 'AVG', E.Min: 'MIN', E.Max: 'MAX'}
    col_results = {}
    specs = []          # (i, fn, value_op, nullmask_op) for COUNT/SUM/AVG
    minmax = []         # (i, fn, value_op, nullmask_op) for MIN/MAX
    expr_aggs = []      # (i, fn, body, inputs) -- arithmetic aggregates fused via wdb_exprjit codegen
    numba_ok = wdb_agg.HAS_NUMBA   # cleared below if any value operand is string or nullable
    for i, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Count) and (isinstance(inner.this, E.Star) or inner.this is None):
            col_results[i] = ('count',)
        elif isinstance(inner, E.Count):
            vop, nop, seg, pcol = agg_arg_operand(inner.this)
            if nop is not None: numba_ok = False
            specs.append((i, 'COUNT', vop, nop)); col_results[i] = ('arr', None, False, None)
        elif type(inner) in _CLS:
            fn = _CLS[type(inner)]
            if not isinstance(inner.this, E.Column):
                try:
                    _body, _inputs = fused_expr_build(inner.this)
                    expr_aggs.append((i, fn, _body, _inputs)); col_results[i] = ('arr', None, False, None)
                    continue
                except _FastUnsupported:
                    pass
            vop, nop, seg, pcol = agg_arg_operand(inner.this)
            if nop is not None or (seg is not None and seg.cols[pcol]['dt'] == 1):
                numba_ok = False     # nullable or string operand -> stay on the numpy reduction paths
            if fn in ('MIN', 'MAX'):
                is_dt = (seg is not None and seg.cols[pcol]['dt'] == 3)
                minmax.append((i, fn, vop, nop)); col_results[i] = ('arr', None, is_dt,
                                                                     (seg.unit(pcol) if is_dt else None))
            else:
                specs.append((i, fn, vop, nop)); col_results[i] = ('arr', None, False, None)
        else:
            if not gkeys or not isinstance(inner, E.Column): raise _FastUnsupported  # bare col w/o GROUP BY
            pseg, ppcol, _ = resolve(inner)
            ki = next((j for j, k in enumerate(gkeys) if k['seg'] is pseg and k['pcol'] == ppcol), None)
            if ki is None: raise _FastUnsupported          # projected column is not a GROUP BY key
            col_results[i] = ('key', ki)

    # codegen-fused arithmetic aggregates: one pass each (decode + expression + accumulate), no array.
    # SUM and AVG of the SAME expression share a single pass; MIN/MAX trigger the min/max accumulators.
    expr_counts = None
    if expr_aggs:
        groups = {}
        for (i, fn, body, inputs) in expr_aggs:
            ek = (body, tuple(id(b) for b, _ in inputs))
            g = groups.setdefault(ek, {'body': body, 'inputs': inputs, 'aggs': [], 'mm': False})
            g['aggs'].append((i, fn))
            if fn in ('MIN', 'MAX'): g['mm'] = True
        for g in groups.values():
            cE, sE, mnE, mxE = wdb_exprjit.grouped_expr(group_op, K, g['body'], g['inputs'], mask, n, g['mm'])
            expr_counts = cE; nz = cE > 0
            for (i, fn) in g['aggs']:
                o = np.full(K, None, dtype=object)
                if   fn == 'SUM': o = sE.astype(object); o[~nz] = None
                elif fn == 'AVG': o[nz] = sE[nz] / cE[nz]
                elif fn == 'MIN': o[nz] = mnE[nz]
                else:             o[nz] = mxE[nz]
                cr = col_results[i]; col_results[i] = ('arr', o, cr[2], cr[3])

    # plain (column / star) aggregates via the existing fused/numpy machinery
    if specs or minmax:
        if numba_ok:
            counts, agg_arrays = wdb_agg.fused_numba(group_op, K, specs + minmax, mask_op, n)
        elif n >= wdb_agg.PARALLEL_THRESHOLD and not minmax:
            counts, agg_arrays = wdb_agg.fused_counts_and_aggs(group_op, K, specs, mask_op, n)
        else:
            gc = wdb_agg._slice(group_op, 0, n)
            gcodes = np.zeros(n, dtype=np.int64) if gc is None else gc.astype(np.int64, copy=False)
            if mask is not None: gcodes = gcodes[mask]
            counts = wdb_agg.group_counts(gcodes, K)
            def _materialize(vop, nop):
                v = wdb_agg._slice(vop, 0, n); v = v[mask] if mask is not None else v
                nm = wdb_agg._slice(nop, 0, n)
                if nm is not None and mask is not None: nm = nm[mask]
                return v, nm
            agg_arrays = {}
            for (i, fn, vop, nop) in specs + minmax:
                v, nm = _materialize(vop, nop); agg_arrays[i] = wdb_agg.group_agg(gcodes, K, fn, v, nm)
        for i, arr in agg_arrays.items():
            cr = col_results[i]; col_results[i] = ('arr', arr, cr[2], cr[3])
    elif expr_counts is not None:
        counts = expr_counts                                  # only arithmetic aggregates -> counts from codegen
    else:                                                     # only COUNT(*) / key columns -> counts-only pass
        gc = wdb_agg._slice(group_op, 0, n)
        gcodes = np.zeros(n, dtype=np.int64) if gc is None else gc.astype(np.int64, copy=False)
        if mask is not None: gcodes = gcodes[mask]
        counts = wdb_agg.group_counts(gcodes, K)
    present = np.nonzero(counts > 0)[0]

    # ---- assemble rows ----
    radices = [k['K'] for k in gkeys]                    # decode composite group code -> per-key codes
    rows = []
    for code in present:
        if gkeys:
            kc = [0] * len(gkeys); tmp = int(code)
            for j in range(len(gkeys) - 1, -1, -1):
                kc[j] = tmp % radices[j]; tmp //= radices[j]
        row = []
        for i, p in enumerate(proj):
            r = col_results[i]
            if r[0] == 'key':
                gk = gkeys[r[1]]
                row.append(wdb_sql._pyval(_code_val(gk['seg'], gk['pcol'], kc[r[1]])))
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


# ── Multi-table FK-pointer chain (rung: breadth) ─────────────────────────────
# Generalises the single FK pointer to a walk: lineitem -> orders -> customer -> nation -> region.
# Each join's ON must match a stored FK pointer (child -> parent). The "fact" table is the one that is
# a child but never a parent; from it every other table is reached by composing the per-edge pointers
# (compose by gathering the next pointer through the current one). composed[alias] maps each fact row to
# that table's row (None for the fact itself). Raises _FastUnsupported if the join graph is not an
# FK-pointer-rooted tree, so the caller can fall back.
def _build_chain(db, tree):
    frm = tree.find(E.From).this
    tables = [(frm.name, frm.alias or frm.name)]
    for jn in (tree.args.get('joins') or []):
        if jn.args.get('side') or jn.args.get('kind'): raise _FastUnsupported   # INNER only
        if not isinstance(jn.this, E.Table): raise _FastUnsupported              # no subqueries
        tables.append((jn.this.name, jn.this.alias or jn.this.name))
    alias2t = {a: t for t, a in tables}
    if len(alias2t) != len(tables): raise _FastUnsupported                       # duplicate/self alias

    edges = {}            # child_alias -> (parent_alias, fk_col)
    parents = set()
    for jn in (tree.args.get('joins') or []):
        on = jn.args.get('on')
        if not isinstance(on, E.EQ): raise _FastUnsupported
        le, re = on.this, on.expression
        if not (isinstance(le, E.Column) and isinstance(re, E.Column)): raise _FastUnsupported
        aA, kA, aB, kB = le.table, le.name, re.table, re.name
        tA, tB = alias2t.get(aA), alias2t.get(aB)
        if tA is None or tB is None: raise _FastUnsupported
        fkA, fkB = db.cat.fk_pointers(tA), db.cat.fk_pointers(tB)
        if kA in fkA and fkA[kA]['parent'] == tB and fkA[kA]['parent_key'] == kB:
            child_a, parent_a, fk_col = aA, aB, kA
        elif kB in fkB and fkB[kB]['parent'] == tA and fkB[kB]['parent_key'] == kA:
            child_a, parent_a, fk_col = aB, aA, kB
        else:
            raise _FastUnsupported                                               # edge is not a stored FK pointer
        if child_a in edges: raise _FastUnsupported                              # one parent per child (tree)
        edges[child_a] = (parent_a, fk_col); parents.add(parent_a)

    fact_candidates = [a for a in edges if a not in parents]
    if len(fact_candidates) != 1: raise _FastUnsupported                         # need a single rooted fact
    fact = fact_candidates[0]

    # Linear chain (each child has one parent, single rooted fact): order it fact -> p1 -> p2 -> ...
    order = [fact]; cur = fact
    while cur in edges:
        cur = edges[cur][0]; order.append(cur)

    # JOIN PRUNING. An FK pointer is built only after verifying referential integrity (every child maps to
    # exactly one parent, parent key unique), so each child->parent INNER join is row-preserving -- joining
    # a table the query never reads cannot change the result. So we only need to compose pointers up to the
    # furthest table actually referenced by the projection / WHERE / GROUP BY / ORDER BY (NOT the join ON
    # columns, which are join plumbing). Everything beyond it is dropped, saving a gather per pruned hop.
    cols_of = {a: set(db.cat.column_names(t)) for a, t in alias2t.items()}
    ref = {fact}
    scan = list(tree.expressions)
    for key in ('where', 'group', 'order'):
        node = tree.args.get(key)
        if node is not None: scan.append(node)
    for rootn in scan:
        for col in rootn.find_all(E.Column):
            a = col.table
            if a and a in alias2t:
                ref.add(a)
            elif not a:                                          # unqualified: keep every candidate owner
                ref.update(al for al, cs in cols_of.items() if col.name in cs)
    keep_idx = max((i for i, a in enumerate(order) if a in ref), default=0)
    keep = set(order[:keep_idx + 1])

    seg_of, sp_of = {}, {}
    for _, a in tables:
        seg_of[a], sp_of[a] = _solo_segment(db, alias2t[a])

    composed = {fact: None}                                                      # None = identity (fact rows)
    progress = True
    while progress:
        progress = False
        for child_a, (parent_a, fk_col) in edges.items():
            if parent_a not in keep: continue                                    # pruned hop: skip the gather
            if child_a in composed and parent_a not in composed:
                p = db.fk_pointer(sp_of[child_a], fk_col)
                if p is None: raise _FastUnsupported
                cc = composed[child_a]
                composed[parent_a] = p if cc is None else p[cc]                  # compose by gather
                progress = True
    if any(a not in composed for a in keep): raise _FastUnsupported              # kept tables must connect
    return dict(fact=fact, alias2t=alias2t, seg_of=seg_of, composed=composed, n=seg_of[fact].N)
