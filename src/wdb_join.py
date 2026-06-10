"""WaveDB JOIN executor — step 1: two-table INNER equi-join.

Decode the needed columns from each side via WaveDB's own read path, join them, then run the SELECT
clauses (projection / WHERE / GROUP BY / aggregates / ORDER BY / LIMIT). The join itself has two
kernels: a hash join (baseline, any equi-join) and -- when the child join key is a pre-resolved
foreign-key POINTER into the parent's rows -- a gather join (array index, no hash build). This module
is the hash baseline + the shared post-join evaluator; the gather kernel layers on top.
Correctness first: verified against DuckDB. Unsupported shapes raise NotImplementedError.
"""
import re
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


def _chain_pandas(db, tree, ctx):
    """Fallback that REUSES the FK-chain resolution: gather every referenced column to fact-row space via
    the same composed pointers the fast path uses, then run the pandas WHERE/aggregate tail on that frame.
    So an FK-chain query that the fused fast path can't take (high-card group, non-value-identity column,
    plain projection, exotic predicate) still resolves the SAME join -- no separate single-join engine, no
    'multi-join' cliff. pandas only does the aggregation/predicate part that couldn't be fused."""
    alias2t = ctx['alias2t']; seg_of = ctx['seg_of']; composed = ctx['composed']
    cols_of = {a: set(db.cat.column_names(t)) for a, t in alias2t.items()}
    phys_of = {a: db.cat.phys_map(t) for a, t in alias2t.items()}
    _memo = {}
    def gather(alias, name):
        k = (alias, name)
        if k not in _memo:
            seg = seg_of[alias]; pcol = phys_of[alias].get(name, name); cptr = composed[alias]
            arr, _nm = wdb_sql._col(seg, pcol)
            _memo[k] = arr if cptr is None else arr[cptr]        # gather parent rows to fact rows
        return _memo[k]
    def owner(node):
        a = node.table
        if a:
            if a not in alias2t or node.name not in cols_of[a]: raise _FastUnsupported
            return a
        owners = [al for al, cs in cols_of.items() if node.name in cs]
        if len(owners) != 1: raise _FastUnsupported
        return owners[0]
    R = lambda node: f"{owner(node)}.{node.name}"

    proj = tree.expressions
    frame = {}
    scan = list(proj)
    for key in ('where', 'group', 'order'):
        nd = tree.args.get(key)
        if nd is not None: scan.append(nd)
    for rootn in scan:
        for col in rootn.find_all(E.Column):
            a = owner(col); fk = f"{a}.{col.name}"
            if fk not in frame: frame[fk] = gather(a, col.name)
    df = pd.DataFrame(frame) if frame else pd.DataFrame(index=range(ctx['n']))
    where = tree.args.get('where')
    if where is not None: df = df[_mask(df, where.this, R)]
    group = tree.args.get('group')
    has_agg = any(wdb_sql._agg_kind(p) for p in proj)
    if group is not None or has_agg:
        rows = _aggregate(df, proj, group, R)
    else:
        keys = [R(p.this if isinstance(p, E.Alias) else p) for p in proj]
        rows = [tuple(_render(v) for v in t) for t in df[keys].itertuples(index=False, name=None)]
    having = tree.args.get('having')
    if having is not None:
        rows = wdb_sql._apply_having(rows, proj, having.this, None)   # fused path must filter too
    rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))
    lim = wdb_sql._limit(tree)
    if lim is not None: rows = rows[:lim]
    return rows, [wdb_sql._alias(p) for p in proj]


def table_agg(db, tree):
    """Single-table aggregate routed through the SAME fused engine as joins: a 0-join chain (fact only, every
    cptr is None). Reuses predicate fusion, high-card factorise, and vectorised assembly. Raises
    _FastUnsupported on anything not fusable so the caller falls back to the mature single-table executor."""
    return _fast_pointer_agg(db, tree, _build_chain(db, tree))


def join_query(db, sql, columnar=False):
    tree = sqlglot.parse_one(sql, read='duckdb')
    joins = tree.args.get('joins')
    # FK-pointer fast path: handles 1..N joins as a chain/star of pre-resolved pointers.
    try:
        chain = _build_chain(db, tree)
    except _FastUnsupported:
        chain = None
    if chain is not None:
        try:
            return _fast_pointer_agg(db, tree, chain, columnar)  # fully fused
        except _FastUnsupported:
            return _chain_pandas(db, tree, chain)                # same chain, pandas agg/predicate tail
    # Not an FK chain (e.g. a join that has no stored pointer) -> single-join pandas hash merge.
    if not joins or len(joins) != 1:
        raise NotImplementedError("join: non-FK multi-join needs a hash join (not yet supported)")
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

    having = tree.args.get('having')
    if having is not None:
        rows = wdb_sql._apply_having(rows, proj, having.this, None)   # fused path must filter too
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
    if isinstance(node, E.Is):                         # IS NULL (IS NOT NULL arrives as Not(Is))
        if isinstance(node.expression, E.Null): return df[R(node.this)].isna()
        raise NotImplementedError(f"join WHERE: Is {type(node.expression).__name__}")
    if isinstance(node, (E.Like, E.ILike)):            # LIKE that didn't fuse (e.g. high-card column)
        s = df[R(node.this)].astype('string')
        rx = '^' + re.escape(str(node.expression.this)).replace('%', '.*').replace('_', '.') + '$'
        m = s.str.match(rx, case=not isinstance(node, E.ILike), na=False)
        return (~m) if node.args.get('negate') else m
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
GROUP_CD_CELL_CEIL = 1 << 22  # max (groups x value-cardinality) cells for the one-pass grouped
                              # COUNT(DISTINCT) keep-table; above this the dense 2-D table is too big
                              # and we fall back to the sort-based path.
TALLY_MAX_RATIO = 0.9         # engage the value-frequency tally for whole-table SUM/AVG/MIN/MAX only
                              # when n_dict < N * this (the column actually has repeats to exploit);
                              # a near-all-distinct column has none, so we scan instead. Self-gated on
                              # the column's own measured cardinality -- never on the workload.

class _FastUnsupported(Exception):
    pass

_FAST_HITS = 0   # diagnostic: how many queries took the gather fast path
_TOPK_HITS = 0   # diagnostic: how many queries had `present` pruned by the bounded top-K prefilter
_SLICE_SCALAR_HITS = 0   # diagnostic: how many queries took the per-slice scalar agg (cluster-key GROUP BY + predicate)
def _bump_fast():
    global _FAST_HITS
    _FAST_HITS += 1
LUT_MAX_CARD = 65536   # code-LUT predicate fusion (LIKE / string-ordering / IS NULL) precomputes
                       # keep[code]=pred(dict_value) over the dictionary; viable only while the dict
                       # is small. Measured precompute (LIKE regex over D dict values): 0.02ms@200,
                       # 0.88ms@10k, 90ms@1M -- so cap at low-card categoricals; high-card (name/
                       # comment) columns fall back to the mask/row path.
FUSE_STR_PRED = True   # string '='/'!=' -> inline code comparison (codes[i]==target). Measured to
                       # beat both the identity-base trick and the materialised-mask path at every
                       # cardinality, fact AND gathered-parent, at sf=1 -- so no runtime switch is
                       # warranted yet. This flag is where a parent-cache-thrash threshold would go
                       # if a large-parent (sf>=10) workload ever shows the gather losing to a mask.


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
    import wdb_override
    if wdb_override.load(seg.path): raise _FastUnsupported                  # column overrides (post-UPDATE)
    if any(c.get('mode') == 6 for c in seg.cols.values()): raise _FastUnsupported   # synthetic ADD COLUMN
    return seg, paths[0]


def _code_val(seg, pcol, code):
    c = seg.cols[pcol]
    if c.get('has_null') and int(code) == c['V'] - 1: return None
    return seg.fetch(pcol, int(code))


def _bulk_keyvals(seg, pcol, codes):
    """Vectorised decode of an array of group-key dictionary codes -> list of python values. Group keys are
    value-identity (mode 0/2/5/6 -- mode-4 is gated out), so a single dict index replaces a per-row fetch()."""
    c = seg.cols[pcol]; dt = c['dt']
    td = seg._typed_dict(pcol)
    if not isinstance(td, np.ndarray):
        td = np.array(td, dtype=object)
    codes = np.asarray(codes)
    nc = (c['V'] - 1) if c['has_null'] else None
    if len(td) == 0:                                  # all-null column -> every key is NULL
        return [None] * len(codes)
    safe = np.where(codes == nc, 0, codes) if nc is not None else codes   # null code -> dummy idx (fixed below)
    picked = td[safe]
    if dt == 3:                                       # int64 epochs -> datetime64 -> _pyval string
        unit = seg.unit(pcol)
        out = [wdb_sql._pyval(x) for x in picked.astype(np.int64).view(f'datetime64[{unit}]')]
    elif dt == 1:                                     # bytes -> str
        out = [wdb_sql._pyval(x) for x in picked]
    else:                                             # int / float -> python scalars (C-level tolist)
        out = picked.tolist()
    if nc is not None:
        cl = codes.tolist()
        out = [None if cl[i] == nc else out[i] for i in range(len(out))]
    return out


def _mode4_group(seg, pcol):
    """Affine (mode-4) GROUP BY key -> (dense gids per row, K, gid->value labels), memoised on the
    immutable segment. Factorising the column's values is invariant for a static segment, so do it
    once per (segment, column) instead of on every query (the cost the affine-key fix introduced)."""
    cache = getattr(seg, '_mode4_group_cache', None)
    if cache is None:
        cache = {}
        try: seg._mode4_group_cache = cache
        except Exception: pass
    hit = cache.get(pcol)
    if hit is not None: return hit
    vals = np.asarray(seg.values(pcol))
    gids, uniq = pd.factorize(vals, sort=False)
    full = np.ascontiguousarray(gids.astype(np.int64))
    if seg.cols[pcol]['dt'] == 3:
        u = np.asarray(uniq).astype(np.int64).view(f"datetime64[{seg.unit(pcol)}]")
        labels = [wdb_sql._pyval(x) for x in u]
    else:
        labels = [wdb_sql._pyval(x) for x in np.asarray(uniq).tolist()]
    res = (full, len(uniq), labels)
    if isinstance(cache, dict): cache[pcol] = res
    return res


def _topk_prefilter(tree, proj, col_results, counts, present, gkeys):
    """ORDER BY <projected aggregate>[DESC] LIMIT k over a GROUP BY: shrink `present` to a provable
    superset of the top-k groups with ONE numpy partition on the primary order array, so the row
    assembly materialises ~k rows instead of every group. The downstream _apply_order over the shrunk
    set stays the source of truth for exact ordering (tie-breaks, null handling), so the result is
    identical to the full path -- this only drops groups that provably cannot enter the top-k.
    Engages only when the PRIMARY order key maps to a finite numeric aggregate/COUNT column and
    k < #groups; otherwise returns `present` unchanged (full path). Pruning by primary key alone is
    a valid superset: every true top-k row has a primary value at least as good as the k-th best, so
    `a >= thresh` (desc) / `a <= thresh` (asc), with all boundary ties kept, can never exclude one."""
    if not gkeys:
        return present
    order = tree.args.get('order')
    lim = wdb_sql._limit(tree)
    if order is None or lim is None or lim <= 0:
        return present
    n = len(present)
    if lim >= n:
        return present
    o0 = order.expressions[0]
    desc = bool(o0.args.get('desc'))
    target = o0.this
    idx = None
    for i, p in enumerate(proj):                       # same match rule as _apply_order
        inner = p.this if isinstance(p, E.Alias) else p
        tname = target.name if isinstance(target, E.Column) else None
        if inner.sql() == target.sql() or wdb_sql._alias(p) == tname:
            idx = i; break
    if idx is None:
        return present
    r = col_results[idx]
    if r[0] == 'count':
        a = counts[present]
    elif r[0] == 'arr' and not r[2]:                   # numeric aggregate array (not datetime)
        a = r[1][present]
    else:
        return present                                 # key column / datetime primary -> full path
    a = np.asarray(a)
    if a.dtype.kind == 'O':                            # Decimal/object agg array -> float proxy for the
        try:                                           # partition only; equal values map to identical floats
            a = a.astype(np.float64)                   # and distinct sums differ far more than float error,
        except (TypeError, ValueError):                # so the >=thresh superset stays exact. None -> bail.
            return present
    if a.dtype.kind not in 'iuf':
        return present
    if a.dtype.kind == 'f' and not np.isfinite(a).all():   # NaN/inf -> null-ordering risk, full path
        return present
    k = int(lim)
    if desc:
        thresh = np.partition(a, n - k)[n - k]         # k-th largest value
        sel = np.nonzero(a >= thresh)[0]
    else:
        thresh = np.partition(a, k - 1)[k - 1]         # k-th smallest value
        sel = np.nonzero(a <= thresh)[0]
    global _TOPK_HITS; _TOPK_HITS += 1
    return present[sel]


def _slice_scalar_agg(group_keys, inputs, exprs, mask, n, pred, offsets):
    """GROUP BY the cluster key (single, direct fact column) with a fused or materialised predicate:
    each cluster range is exactly one group, so run the fast register-accumulator scalar kernel once
    per slice instead of grouped_multi's per-group indexed-write accumulators -- measured ~5x on a
    predicated SUM (33.9ms -> 6.7ms) because the scalar kernel keeps the accumulators in registers and
    SIMD-reduces. Returns (counts[K], results) in grouped_multi's shape (SUM/AVG/COUNT only; the caller
    gates out MIN/MAX, which scalar_multi does not handle). Bit-identical group sums to grouped_multi."""
    codes, K, _ = group_keys[0]
    global _SLICE_SCALAR_HITS; _SLICE_SCALAR_HITS += 1
    counts = np.zeros(K, dtype=np.int64)
    sums = [np.zeros(K, dtype=np.float64) for _ in exprs]
    for gi in range(len(offsets) - 1):
        lo, hi = int(offsets[gi]), int(offsets[gi + 1])
        if hi <= lo: continue
        g = int(codes[lo])                               # cluster range is constant in the key code
        si = [(b, c[lo:hi], (None if p is None else np.ascontiguousarray(p[lo:hi]))) for (b, c, p) in inputs]
        cnt, out = wdb_exprjit.scalar_multi(si, exprs, (None if mask is None else mask[lo:hi]), hi - lo, pred)
        counts[g] += int(cnt[0])                         # += (not =) so a key split across runs still sums
        for e in range(len(exprs)):
            sums[e][g] += out[e][0][0]
    return counts, [(sums[e], None, None) for e in range(len(exprs))]


def _fast_pointer_agg(db, tree, ctx, columnar=False):
    import operator
    proj = tree.expressions
    group = tree.args.get('group')
    has_agg = any(wdb_sql._agg_kind(p) for p in proj)
    cd_col = None                                                            # COUNT(DISTINCT col), sole, no GROUP BY
    if group is None and len(proj) == 1:
        _i0 = proj[0].this if isinstance(proj[0], E.Alias) else proj[0]
        if isinstance(_i0, E.Count) and isinstance(_i0.this, E.Distinct):
            _dx = _i0.this.expressions
            if len(_dx) == 1 and isinstance(_dx[0], E.Column): cd_col = _dx[0]
            else: raise _FastUnsupported                                     # COUNT(DISTINCT expr / multi) -> fallback
    if cd_col is not None:
        gnodes = []                                                          # computed directly after mask setup
    elif tree.args.get('distinct') is not None and not has_agg and group is None:
        cols = [(p.this if isinstance(p, E.Alias) else p) for p in proj]      # SELECT DISTINCT cols == GROUP BY cols
        if not all(isinstance(c, E.Column) for c in cols): raise _FastUnsupported  # DISTINCT * / over expr
        gnodes = cols
    elif not has_agg:
        raise _FastUnsupported                                                # plain projection -> fallback
    else:
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
        if seg.cols[pcol]['dt'] == 3 and getattr(arr, 'dtype', None) is not None and arr.dtype.kind == 'M':
            arr = arr.view('int64')                       # datetime64 -> epoch ints for the numba kernel
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
                seg, pcol, cptr = resolve(node)               # cptr: fact->parent pointer (None for the fact)
                raw = wdb_sql.raw_dict_col(seg, pcol)
                if raw is None: raise _FastUnsupported               # nullable / string / computed -> fallback
                key = (id(seg), pcol, id(cptr) if cptr is not None else None)
                if key not in slot:
                    slot[key] = len(inputs)
                    inputs.append((np.ascontiguousarray(raw[0]), np.ascontiguousarray(raw[1]),
                                   None if cptr is None else np.ascontiguousarray(cptr)))
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
                if arr.dtype.kind == 'M': arr = arr.view('int64')   # datetime: compare as epoch ints
                return _OPS[op](arr, v)
            return leaf(node.this, mk)
        if isinstance(node, E.Between):
            def mk(seg, pcol):
                arr, _ = _col_cached(seg, pcol)
                lo = wdb_sql._lit_for_col(seg, pcol, node.args['low'], arr.dtype.kind)
                hi = wdb_sql._lit_for_col(seg, pcol, node.args['high'], arr.dtype.kind)
                if arr.dtype.kind == 'M': arr = arr.view('int64')   # datetime: compare as epoch ints
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
        if isinstance(node, E.Is) and isinstance(node.expression, E.Null):   # IS NULL (Not(Is) = IS NOT NULL)
            def mk(seg, pcol):
                c = seg.cols[pcol]; codes = seg.codes(pcol)
                if not c['has_null']: return np.zeros(len(codes), dtype=bool)  # non-nullable -> nothing
                return codes == (c['V'] - 1)                                   # null is reserved code V-1
            return leaf(node.this, mk)
        if isinstance(node, (E.Like, E.ILike)):              # LIKE -> code-LUT over the dict, gathered
            negate = bool(node.args.get('negate')); ci = isinstance(node, E.ILike); pnode = node.expression
            def mk(seg, pcol):
                sc = _str_codes(seg, pcol)
                if sc is None: raise _FastUnsupported       # non-value-identity dict -> can't map codes
                codes, code_of, nullcode = sc
                pb = _lit_bytes(seg, pcol, pnode)
                patt = pb.decode('utf-8', 'replace') if isinstance(pb, (bytes, bytearray)) else str(pb)
                rx = re.compile('^' + re.escape(patt).replace('%', '.*').replace('_', '.') + '$',
                                re.IGNORECASE if ci else 0)
                ncodes = max(max(code_of.values(), default=-1),
                             nullcode if nullcode is not None else -1) + 1
                keep = np.zeros(ncodes, dtype=bool)
                for vb, cd in code_of.items():
                    v = vb.decode('utf-8', 'replace') if isinstance(vb, (bytes, bytearray)) else str(vb)
                    if rx.match(v): keep[cd] = True
                m = keep[codes]
                return ~m if negate else m
            return leaf(node.this, mk)
        raise _FastUnsupported
    where = tree.args.get('where')
    _maskc = {}
    def get_mask():                # materialise the WHERE bool mask lazily -- only non-fused paths need it
        if 'm' not in _maskc:
            _maskc['m'] = mask_eval(where.this) if where is not None else None
        return _maskc['m']
    def _mask_op():
        m = get_mask(); return ('d', m) if m is not None else None

    def _dict_keep(node, seg, pcol, td):
        # Evaluate a WHERE predicate over the (tiny) dictionary value array td -> bool[len(td)].
        # Single column only; returns None on any shape not reducible to td (caller falls back).
        if isinstance(node, E.Paren): return _dict_keep(node.this, seg, pcol, td)
        if isinstance(node, E.And):
            a = _dict_keep(node.this, seg, pcol, td); b = _dict_keep(node.expression, seg, pcol, td)
            return None if a is None or b is None else (a & b)
        if isinstance(node, E.Or):
            a = _dict_keep(node.this, seg, pcol, td); b = _dict_keep(node.expression, seg, pcol, td)
            return None if a is None or b is None else (a | b)
        if isinstance(node, E.Not):
            a = _dict_keep(node.this, seg, pcol, td); return None if a is None else ~a
        if type(node) in _OPS:
            if not (isinstance(node.this, E.Column) and isinstance(node.expression, (E.Literal, E.Neg))):
                return None                                    # col OP literal only (not expr/flip)
            v = wdb_sql._lit_for_col(seg, pcol, node.expression, td.dtype.kind)
            return _OPS[type(node)](td, v)
        if isinstance(node, E.Between):
            if not isinstance(node.this, E.Column): return None
            lo = wdb_sql._lit_for_col(seg, pcol, node.args['low'], td.dtype.kind)
            hi = wdb_sql._lit_for_col(seg, pcol, node.args['high'], td.dtype.kind)
            return (td >= lo) & (td <= hi)
        if isinstance(node, E.In):
            if not isinstance(node.this, E.Column): return None
            exprs = node.args.get('expressions') or []
            if not exprs: return None
            vals = [wdb_sql._lit_for_col(seg, pcol, e, td.dtype.kind) for e in exprs]
            return np.isin(td, vals)
        return None

    def _dict_count(wnode):
        # COUNT(*) WHERE wnode, when wnode is a predicate on a single value-identity numeric dict
        # FACT column -> sum cached per-code counts over qualifying dict entries. None -> normal path.
        try:
            colnodes = list(wnode.find_all(E.Column))
            if not colnodes: return None
            seg0 = pcol0 = None
            for cn in colnodes:
                s, p, cp = resolve(cn)
                if cp is not None: return None                 # parent/join column -> not a solo fact col
                if seg0 is None: seg0, pcol0 = s, p
                elif id(s) != id(seg0) or p != pcol0: return None   # more than one column
            c = seg0.cols[pcol0]
            if c['mode'] == 4 or c['dt'] not in (0, 2): return None  # value-identity int/float only (v1)
            td = np.asarray(seg0._typed_dict(pcol0))
            if td.dtype.kind not in 'iuf' or len(td) == 0: return None
            keep = _dict_keep(wnode, seg0, pcol0, td)
            if keep is None: return None
            counts = seg0.code_counts(pcol0)                   # cached bincount, length V
            return int(counts[:len(td)][keep].sum())           # [:len(td)] excludes the null bin
        except Exception:
            return None

    # ---- group codes (as an operand; gathered per-chunk in the threaded kernel) ----
    n = ctx['n']
    if n == 0 and gnodes: return [], [wdb_sql._alias(p) for p in proj]   # GROUP BY over 0 rows -> no groups
    # (no GROUP BY over 0 rows falls through: SQL still emits one grand-total row -- COUNT=0, SUM/MIN/MAX=NULL)
    if cd_col is not None:                       # COUNT(DISTINCT col) == # distinct non-null codes among matches
        cseg, cpcol, ccptr = resolve(cd_col)
        cc = cseg.cols[cpcol]
        if cc['mode'] == 4: raise _FastUnsupported                         # codes not value-identity -> fallback
        m = get_mask()
        if m is None and ccptr is None:           # no filter: distinct count == dictionary cardinality (O(1))
            _bump_fast()
            return [(int(cc['V'] - cc['has_null']),)], [wdb_sql._alias(proj[0])]
        codes = cseg.codes(cpcol); codes = codes if ccptr is None else codes[ccptr]
        if m is not None: codes = codes[m]
        uniq = np.unique(codes) if codes.size else np.empty(0, dtype=np.int64)
        if cc['has_null']: uniq = uniq[uniq != cc['V'] - 1]                # COUNT(DISTINCT) ignores NULL
        _bump_fast()
        return [(int(uniq.size),)], [wdb_sql._alias(proj[0])]

    # ---- pure COUNT(*) with a single value-identity dict-column predicate -------------------------
    # COUNT(*) WHERE P(col) == sum of per-code row counts over the codes whose dict value satisfies P.
    # O(distinct) over the (tiny) dictionary instead of materialising + scanning N values. One bincount
    # per column, cached on the segment. Numeric int/float dict columns; any other shape returns None
    # and falls through to the normal scan. Fail-safe: any unexpected node -> None -> normal path.
    if (group is None and where is not None and cd_col is None and len(proj) == 1
            and not gnodes):
        _p0 = proj[0].this if isinstance(proj[0], E.Alias) else proj[0]
        if (isinstance(_p0, E.Count) and not isinstance(_p0.this, E.Distinct)
                and (_p0.this is None or isinstance(_p0.this, E.Star))):
            _cnt = _dict_count(where.this)
            if _cnt is not None:
                _bump_fast()
                return [(int(_cnt),)], [wdb_sql._alias(proj[0])]

    # ---- whole-table SUM/AVG/MIN/MAX (+COUNT*) from the per-value tally (no GROUP BY, no WHERE) ------
    # Every reduction over a value-identity numeric dict column is a reduction over (value, frequency):
    # SUM = dict . code_counts (one BLAS dot), AVG = SUM / total, MIN/MAX = extreme value with a nonzero
    # count. O(distinct), no row scan and no gather. Self-gated: engaged only when the column actually
    # compresses (n_dict < N * TALLY_MAX_RATIO); a near-all-distinct column has no repeats to exploit and
    # falls through to the scan kernel below with identical results. Floats only in v1 (dt==2).
    def _whole_tally():
        plan = []                                          # ('sum'|'avg'|'min'|'max', seg, pcol) | ('cnt',_,_)
        has_red = False
        for p in proj:
            node = p.this if isinstance(p, E.Alias) else p
            if (isinstance(node, E.Count) and not isinstance(node.this, E.Distinct)
                    and (node.this is None or isinstance(node.this, E.Star))):
                plan.append(('cnt', None, None)); continue
            if not isinstance(node, (E.Sum, E.Avg, E.Min, E.Max)): return None
            col = node.this
            if not isinstance(col, E.Column): return None              # SUM(a*b) etc -> needs the scan
            s, pc, cp = resolve(col)
            if cp is not None: return None                             # gathered/joined column -> not whole-table
            cc = s.cols[pc]
            if cc['mode'] != 0 or cc['dt'] != 2: return None           # value-identity float dict only (v1)
            if cc['n_dict'] >= n * TALLY_MAX_RATIO: return None        # ~no repeats -> scan instead
            plan.append((node.key, s, pc)); has_red = True
        return plan if has_red else None                              # leave pure COUNT(*) to the scalar path

    if group is None and where is None and cd_col is None and not gnodes and proj and n > 0:
        _tp = _whole_tally()
        if _tp is not None:
            row = []
            for op, s, pc in _tp:
                if op == 'cnt': row.append(int(n)); continue
                nd = s.cols[pc]['n_dict']
                counts = s.code_counts(pc)[:nd]
                dv = np.asarray(s._typed_dict(pc), dtype=np.float64)[:nd]
                if op == 'sum':   row.append(float(np.dot(dv, counts)))
                elif op == 'avg':
                    tot = float(counts.sum()); row.append(float(np.dot(dv, counts) / tot) if tot else None)
                else:
                    pres = counts > 0
                    row.append(None if not pres.any() else
                               (float(dv[pres].min()) if op == 'min' else float(dv[pres].max())))
            _bump_fast()
            return [tuple(row)], [wdb_sql._alias(p) for p in proj]

    gkeys = []                                           # one per GROUP BY column
    for g in gnodes:
        gseg, gpcol, gcptr = resolve(g)
        if gseg.cols[gpcol]['mode'] == 4:
            # Affine-coded (mode 4): dense gids + labels, memoised on the immutable segment.
            full, _K, _labels = _mode4_group(gseg, gpcol)
            if full.size == 0: return [], [wdb_sql._alias(p) for p in proj]
            gkeys.append({'seg': gseg, 'pcol': gpcol, 'cptr': gcptr, 'full': full,
                          'K': _K, 'labels': _labels})
            continue
        full = gseg.codes(gpcol)
        if full.size == 0: return [], [wdb_sql._alias(p) for p in proj]
        gkeys.append({'seg': gseg, 'pcol': gpcol, 'cptr': gcptr, 'full': full, 'K': int(full.max()) + 1})
    gid_to_comp = None                                    # set when a high-card composite is hash-factorised
    if len(gkeys) == 0:
        K = 1; group_op = None; group_keys = []
    elif len(gkeys) == 1:                                 # single key: keep the gather-fused operand
        k0 = gkeys[0]; K = k0['K']
        group_op = ('g', k0['full'], k0['cptr']) if k0['cptr'] is not None else ('d', k0['full'])
        group_keys = [(k0['full'], K, k0['cptr'])]
    else:                                                 # multi-key: mixed-radix composite
        K = 1
        for k in gkeys: K *= k['K']
        if K <= MULTI_GROUP_CEIL:                          # dense: codegen composes the code INLINE (no array)
            group_op = None
            group_keys = [(k['full'], k['K'], k['cptr']) for k in gkeys]
        else:                                              # high-card: hash-factorise the composite to dense ids
            prod = 1
            for k in gkeys: prod *= k['K']
            if prod > (1 << 62): raise _FastUnsupported    # mixed-radix code would overflow int64
            comp = np.zeros(n, dtype=np.int64)
            for k in gkeys:
                codes = k['full'][k['cptr']] if k['cptr'] is not None else k['full']
                comp = comp * k['K'] + codes.astype(np.int64, copy=False)
            gids, gid_to_comp = pd.factorize(comp, sort=False)   # hash-factorise -> only groups present
            gids = np.ascontiguousarray(gids.astype(np.int64))
            K = len(gid_to_comp); group_op = ('d', gids); group_keys = [(gids, K, None)]

    # The codegen value path composes the composite group code INLINE (no comp array). The non-fused
    # plain / numpy / counts-only paths get a materialised group operand on demand via _group_op().
    _go = {}
    def _group_op():
        if group_op is not None or not gkeys: return group_op
        if 'op' not in _go:
            comp = np.zeros(n, dtype=np.int64)
            for k in gkeys:
                codes = k['full'][k['cptr']] if k['cptr'] is not None else k['full']
                comp = comp * k['K'] + codes.astype(np.int64, copy=False)
            _go['op'] = ('d', np.ascontiguousarray(comp))
        return _go['op']

    # ---- per-projection results ----
    # COUNT/SUM/AVG become operand specs computed in one pass (threaded + per-chunk gather above a row
    # threshold); MIN/MAX run on the serial kernel; bare key columns map straight to the group value.
    _CLS = {E.Sum: 'SUM', E.Avg: 'AVG', E.Min: 'MIN', E.Max: 'MAX'}
    col_results = {}

    # ---- UNIFIED fused path -------------------------------------------------------------------------
    # Build every value aggregate as an expression over shared dict-column slots, then a single codegen
    # kernel computes the composite group (inline), decodes each slot once, and accumulates COUNT + every
    # expression's SUM (and MIN/MAX where needed) in ONE pass -- no comp array, no (n,V) value matrix.
    # Falls back wholesale to the per-spec machinery below if any operand is not a numeric dict column or
    # arithmetic over them (string / nullable / computed), or numba is unavailable.
    slots = {}; slot_list = []                       # (id(seg), pcol, id(cptr)) -> global slot index
    def build_fused(node):
        if isinstance(node, (E.Paren, E.Cast)): return build_fused(node.this)
        if isinstance(node, E.Neg): return f"(-{build_fused(node.this)})"
        if isinstance(node, E.Column):
            bseg, bpcol, bcptr = resolve(node)
            raw = wdb_sql.raw_dict_col(bseg, bpcol)
            if raw is None: raise _FastUnsupported
            bkey = (id(bseg), bpcol, id(bcptr) if bcptr is not None else None)
            if bkey not in slots:
                slots[bkey] = len(slot_list)
                slot_list.append((np.ascontiguousarray(raw[0]), np.ascontiguousarray(raw[1]),
                                  None if bcptr is None else np.ascontiguousarray(bcptr)))
            return f"v{slots[bkey]}"
        if isinstance(node, E.Literal):
            if node.is_string: raise _FastUnsupported
            v = node.this
            return f"({float(v)})" if ('.' in v or 'e' in v.lower()) else f"({int(v)})"
        if type(node) in _ARITH_STR:
            return f"({build_fused(node.this)} {_ARITH_STR[type(node)]} {build_fused(node.expression)})"
        raise _FastUnsupported

    _CMP_STR = {E.GT: '>', E.LT: '<', E.GTE: '>=', E.LTE: '<=', E.EQ: '==', E.NEQ: '!='}
    def _is_lit(nd):
        if isinstance(nd, E.Literal): return True
        if isinstance(nd, (E.Neg, E.Cast, E.Paren)): return _is_lit(nd.this)
        return False
    def _str_slot(cseg, cpcol, cptr):                  # raw code slot for a string column (base=None)
        sc = _str_codes(cseg, cpcol)
        if sc is None: raise _FastUnsupported
        codes, code_of, nullcode = sc
        ckey = (id(cseg), cpcol, id(cptr) if cptr is not None else None)
        if ckey not in slots:
            slots[ckey] = len(slot_list)
            slot_list.append((None, np.ascontiguousarray(codes),
                              None if cptr is None else np.ascontiguousarray(cptr)))
        return f"v{slots[ckey]}", code_of, nullcode
    def _code_lut(cseg, cpcol, cptr, fn, mark_null=False):
        # ARBITRARY single-column string predicate -> code-LUT: precompute keep[code]=fn(dict_value) over
        # the (small) dictionary, add it as a value slot whose base IS that bool table, predicate -> 'vk!=0'.
        # Reuses the value-slot kernel verbatim (base[codes[i]]); gated on dict cardinality (precompute cost).
        sc = _str_codes(cseg, cpcol)
        if sc is None: raise _FastUnsupported
        codes, code_of, nullcode = sc
        ncodes = max(max(code_of.values(), default=-1),
                     nullcode if nullcode is not None else -1) + 1
        if ncodes == 0 or ncodes > LUT_MAX_CARD: raise _FastUnsupported
        keep = np.zeros(ncodes, dtype=np.int8)
        if mark_null:
            if nullcode is not None: keep[nullcode] = 1            # IS NULL
        else:
            for vb, cd in code_of.items():
                if fn(vb): keep[cd] = 1
        slot_list.append((keep, np.ascontiguousarray(codes),
                          None if cptr is None else np.ascontiguousarray(cptr)))
        return f"v{len(slot_list) - 1}"
    def _like_fn(cseg, cpcol, pat_node, ci):
        pb = _lit_bytes(cseg, cpcol, pat_node)
        patt = pb.decode('utf-8', 'replace') if isinstance(pb, (bytes, bytearray)) else str(pb)
        rxsrc = '^' + re.escape(patt).replace('%', '.*').replace('_', '.') + '$'   # SQL LIKE -> regex
        rx = re.compile(rxsrc, re.IGNORECASE if ci else 0)
        def _f(vb):
            v = vb.decode('utf-8', 'replace') if isinstance(vb, (bytes, bytearray)) else str(vb)
            return rx.match(v) is not None
        return _f
    def build_pred(node):
        # Compile a WHERE predicate to a numba boolean over the shared slots. Covers AND/OR/NOT, numeric &
        # datetime comparisons (incl. column-vs-column and arithmetic sides), BETWEEN, string '='/'!='/IN
        # via inline code comparison, and arbitrary single-string-column predicates (LIKE, ordering, IS NULL)
        # via a precomputed code-LUT. Raises _FastUnsupported otherwise -> materialised mask.
        if isinstance(node, E.Paren): return build_pred(node.this)
        if isinstance(node, E.And): return f"({build_pred(node.this)} and {build_pred(node.expression)})"
        if isinstance(node, E.Or):  return f"({build_pred(node.this)} or {build_pred(node.expression)})"
        if isinstance(node, E.Not): return f"(not {build_pred(node.this)})"
        if isinstance(node, (E.Like, E.ILike)):           # LIKE / ILIKE -> code-LUT over the dictionary
            if not FUSE_STR_PRED: raise _FastUnsupported
            col = node.this
            if not isinstance(col, E.Column): raise _FastUnsupported
            cseg, cpcol, cptr = resolve(col)
            if cseg.cols[cpcol]['dt'] != 1: raise _FastUnsupported
            pat = node.expression
            if not isinstance(pat, E.Literal) or not pat.is_string: raise _FastUnsupported
            vk = _code_lut(cseg, cpcol, cptr, _like_fn(cseg, cpcol, pat, isinstance(node, E.ILike)))
            return f"({vk} == 0)" if node.args.get('negate') else f"({vk} != 0)"   # NOT LIKE -> negate=True
        if isinstance(node, E.Is):                        # IS NULL (IS NOT NULL handled via E.Not)
            col = node.this
            if not isinstance(col, E.Column) or not isinstance(node.expression, E.Null):
                raise _FastUnsupported
            cseg, cpcol, cptr = resolve(col)
            if cseg.cols[cpcol]['dt'] != 1: raise _FastUnsupported   # numeric IS NULL: presence -> later
            vk = _code_lut(cseg, cpcol, cptr, None, mark_null=True)
            return f"({vk} != 0)"
        if type(node) in _CMP_STR:
            op = _CMP_STR[type(node)]; a, b = node.this, node.expression
            if not _is_lit(a) and not _is_lit(b):          # value-expr vs value-expr (e.g. col < col)
                return f"({build_fused(a)} {op} {build_fused(b)})"   # numeric/datetime; string -> raises
            if _is_lit(a) and _is_lit(b): raise _FastUnsupported
            col, lit, left = (a, b, True) if not _is_lit(a) else (b, a, False)
            if not isinstance(col, E.Column):          # computed expr vs numeric literal -> fuse both
                if not (isinstance(lit, E.Literal) and not lit.is_string): raise _FastUnsupported
                v = build_fused(col)
                lv = float(lit.this) if ('.' in lit.this or 'e' in lit.this.lower()) else int(lit.this)
                return f"({v} {op} {lv})" if left else f"({lv} {op} {v})"
            cseg, cpcol, cptr = resolve(col)
            if cseg.cols[cpcol]['dt'] == 1:                # string
                if not FUSE_STR_PRED or not lit.is_string: raise _FastUnsupported
                if type(node) in (E.EQ, E.NEQ):            # equality -> direct code comparison
                    vk, code_of, nullcode = _str_slot(cseg, cpcol, cptr)
                    target = code_of.get(_lit_bytes(cseg, cpcol, lit), -1)
                    if type(node) is E.EQ: return f"({vk} == {target})"
                    if nullcode is None:   return f"({vk} != {target})"
                    return f"(({vk} != {target}) and ({vk} != {nullcode}))"   # SQL: NULL != x not TRUE
                litb = _lit_bytes(cseg, cpcol, lit)        # ordering -> code-LUT (dict not order-preserving)
                opf = _OPS[type(node)]
                if not left:
                    opf = _OPS[{E.GT: E.LT, E.LT: E.GT, E.GTE: E.LTE, E.LTE: E.GTE}[type(node)]]
                vk = _code_lut(cseg, cpcol, cptr, lambda vb, opf=opf, litb=litb: bool(opf(vb, litb)))
                return f"({vk} != 0)"
            v = build_fused(col)
            kind = 'f' if cseg.cols[cpcol]['dt'] == 2 else 'i'
            lv = wdb_sql._lit_for_col(cseg, cpcol, lit, kind)
            if not isinstance(lv, (int, float, np.integer, np.floating)): raise _FastUnsupported
            lv = float(lv) if kind == 'f' else int(lv)
            return f"({v} {op} {lv})" if left else f"({lv} {op} {v})"
        if isinstance(node, E.In):
            col = node.this; exprs = node.args.get('expressions') or []
            if node.args.get('query') is not None: raise _FastUnsupported   # IN (subquery) -> mask
            if not isinstance(col, E.Column) or not exprs or len(exprs) > 256: raise _FastUnsupported
            cseg, cpcol, cptr = resolve(col)
            if cseg.cols[cpcol]['dt'] == 1:                # string IN -> OR of code equalities
                if not FUSE_STR_PRED: raise _FastUnsupported
                vk, code_of, _nc = _str_slot(cseg, cpcol, cptr)
                tgts = []
                for e in exprs:
                    if not e.is_string: raise _FastUnsupported
                    tgts.append(code_of.get(_lit_bytes(cseg, cpcol, e), -1))
                return "(" + " or ".join(f"({vk} == {t})" for t in tgts) + ")"
            v = build_fused(col); kind = 'f' if cseg.cols[cpcol]['dt'] == 2 else 'i'  # numeric IN
            vals = []
            for e in exprs:
                lv = wdb_sql._lit_for_col(cseg, cpcol, e, kind)
                if not isinstance(lv, (int, float, np.integer, np.floating)): raise _FastUnsupported
                vals.append(float(lv) if kind == 'f' else int(lv))
            return "(" + " or ".join(f"({v} == {x})" for x in vals) + ")"
        if isinstance(node, E.Between):
            col = node.this
            if not isinstance(col, E.Column): raise _FastUnsupported
            cseg, cpcol, _c = resolve(col)
            if cseg.cols[cpcol]['dt'] == 1: raise _FastUnsupported
            v = build_fused(col); kind = 'f' if cseg.cols[cpcol]['dt'] == 2 else 'i'
            lo = wdb_sql._lit_for_col(cseg, cpcol, node.args['low'], kind)
            hi = wdb_sql._lit_for_col(cseg, cpcol, node.args['high'], kind)
            lo = float(lo) if kind == 'f' else int(lo); hi = float(hi) if kind == 'f' else int(hi)
            return f"(({lo} <= {v}) and ({v} <= {hi}))"
        raise _FastUnsupported

    pred_body = None
    if where is not None:
        try:
            pred_body = build_pred(where.this)
        except _FastUnsupported:
            pred_body = None; slots.clear(); slot_list.clear()   # discard any partial predicate slots

    plan = []; fully = wdb_exprjit.HAS_NUMBA
    for i, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Count) and (isinstance(inner.this, E.Star) or inner.this is None):
            plan.append((i, 'count', None, False, None))
        elif type(inner) in _CLS:
            try:
                body = build_fused(inner.this)
            except _FastUnsupported:
                fully = False; break
            is_dt = False; unit = None
            if isinstance(inner.this, E.Column):
                cseg, cpcol, _c = resolve(inner.this)
                if cseg.cols[cpcol]['dt'] == 3: is_dt = True; unit = cseg.unit(cpcol)
            plan.append((i, _CLS[type(inner)], body, is_dt, unit))
        elif isinstance(inner, E.Count):                 # COUNT(col): == group count only if non-nullable
            if not isinstance(inner.this, E.Column): fully = False; break
            cseg, cpcol, _c = resolve(inner.this)
            if cseg.cols[cpcol].get('has_null'): fully = False; break
            plan.append((i, 'count', None, False, None))
        elif gkeys and isinstance(inner, E.Column):      # bare GROUP BY key column
            pseg, ppcol, _r = resolve(inner)
            ki = next((j for j, k in enumerate(gkeys) if k['seg'] is pseg and k['pcol'] == ppcol), None)
            if ki is None: fully = False; break
            plan.append((i, 'key', ki, False, None))
        else:
            fully = False; break

    def _grouped_cd():
        # Grouped COUNT(DISTINCT vcol): when the group key(s) and vcol are value-identity dict columns
        # whose dense (groups x value-cardinality) table is small, count distinct value codes per group
        # in ONE vectorised pass (a 2-D keep-table) instead of the sort-based fallback -- O(N). Returns
        # (counts, col_results) in the same group-id space the row assembler expects, else None.
        cd_idx = vnode = None
        for idx, p in enumerate(proj):
            inner = p.this if isinstance(p, E.Alias) else p
            if isinstance(inner, E.Count) and isinstance(inner.this, E.Distinct):
                dcols = list(inner.this.find_all(E.Column))
                if cd_idx is not None or len(dcols) != 1: return None      # one COUNT(DISTINCT col) only
                cd_idx = idx; vnode = dcols[0]
            elif isinstance(inner, E.Column):
                continue                                                   # bare group key (matched below)
            else:
                return None
        if cd_idx is None: return None
        if any(k['cptr'] is not None for k in gkeys): return None          # v1: fact-only group keys
        vseg, vpcol, vcptr = resolve(vnode)
        vc = vseg.cols[vpcol]
        if vc['mode'] == 4 or vcptr is not None: return None               # need value-identity fact codes
        nv = vc['V']
        if K * nv > GROUP_CD_CELL_CEIL: return None                        # dense table too big -> fallback
        op = _group_op()
        if op is None or op[0] != 'd': return None
        gid = np.asarray(op[1], dtype=np.int64)
        vcodes = vseg.codes(vpcol)
        m = get_mask()
        if m is not None: gid = gid[m]; vcodes = vcodes[m]
        # ONE pass: a (groups x value) cell count. Row totals give presence; nonzero non-null
        # columns per row give the distinct count. The null code is the last dict slot (V-1).
        table = np.bincount(gid * nv + vcodes, minlength=K * nv).reshape(K, nv)
        counts_l = table.sum(axis=1)                                       # rows per group (presence)
        nn = nv - 1 if vc['has_null'] else nv                              # COUNT(DISTINCT) drops NULL
        dist = (table[:, :nn] > 0).sum(axis=1).astype(object)
        cr = {}
        for idx, p in enumerate(proj):
            if idx == cd_idx:
                cr[idx] = ('arr', dist, False, None)
            else:
                pseg, ppcol, _r = resolve(p.this if isinstance(p, E.Alias) else p)
                ki = next((j for j, k in enumerate(gkeys) if k['seg'] is pseg and k['pcol'] == ppcol), None)
                if ki is None: return None
                cr[idx] = ('key', ki)
        return counts_l, cr

    cdist = _grouped_cd() if gkeys else None
    if cdist is not None:
        counts, col_results = cdist
    elif fully:
        ex_index = {}; exprs = []                        # dedup identical expressions; share one pass
        for (i, fn, body, is_dt, unit) in plan:
            if fn in ('SUM', 'AVG', 'MIN', 'MAX'):
                if body not in ex_index: ex_index[body] = len(exprs); exprs.append([body, False])
                if fn in ('MIN', 'MAX'): exprs[ex_index[body]][1] = True
        _mask = None if pred_body else get_mask()
        _cm = gkeys[0]['seg'].cluster_meta() if len(gkeys) == 1 else None
        _no_mm = not any(e[1] for e in exprs)
        if not group_keys and _no_mm:     # no GROUP BY, no MIN/MAX -> lean scalar kernel
            counts, results = wdb_exprjit.scalar_multi(slot_list, exprs, _mask, n, pred_body)
        elif (len(gkeys) == 1 and _cm is not None and gkeys[0]['cptr'] is None and _no_mm
              and _cm.get('key') == gkeys[0]['pcol'] and gkeys[0]['seg'].presence_mask() is None):
            counts, results = _slice_scalar_agg(group_keys, slot_list, exprs,   # cluster-key GROUP BY +
                                                _mask, n, pred_body, _cm['offsets'])  # predicate -> per-slice scalar
        else:
            counts, results = wdb_exprjit.grouped_multi(group_keys, slot_list, exprs,
                                                        _mask, n, pred_body)
        nz = counts > 0
        for (i, fn, body, is_dt, unit) in plan:
            if fn == 'count':
                col_results[i] = ('count',)
            elif fn == 'key':
                col_results[i] = ('key', body)           # body field carries the group-key index
            else:
                s, mn, mx = results[ex_index[body]]
                o = np.full(K, None, dtype=object)
                if   fn == 'SUM': o = s.astype(object); o[~nz] = None
                elif fn == 'AVG': o[nz] = s[nz] / counts[nz]
                elif fn == 'MIN': o[nz] = mn[nz]
                else:             o[nz] = mx[nz]
                col_results[i] = ('arr', o, is_dt, unit)
    else:
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
                ek = (body, tuple(id(b) for b, _, _ in inputs))
                g = groups.setdefault(ek, {'body': body, 'inputs': inputs, 'aggs': [], 'mm': False})
                g['aggs'].append((i, fn))
                if fn in ('MIN', 'MAX'): g['mm'] = True
            for g in groups.values():
                cE, sE, mnE, mxE = wdb_exprjit.grouped_expr(group_keys, g['body'], g['inputs'], get_mask(), n, g['mm'])
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
                counts, agg_arrays = wdb_agg.fused_numba(_group_op(), K, specs + minmax, _mask_op(), n)
            elif n >= wdb_agg.PARALLEL_THRESHOLD and not minmax:
                counts, agg_arrays = wdb_agg.fused_counts_and_aggs(_group_op(), K, specs, _mask_op(), n)
            else:
                gc = wdb_agg._slice(_group_op(), 0, n)
                gcodes = np.zeros(n, dtype=np.int64) if gc is None else gc.astype(np.int64, copy=False)
                _m = get_mask()
                if _m is not None: gcodes = gcodes[_m]
                counts = wdb_agg.group_counts(gcodes, K)
                def _materialize(vop, nop):
                    _m = get_mask()
                    v = wdb_agg._slice(vop, 0, n); v = v[_m] if _m is not None else v
                    nm = wdb_agg._slice(nop, 0, n)
                    if nm is not None and _m is not None: nm = nm[_m]
                    return v, nm
                agg_arrays = {}
                for (i, fn, vop, nop) in specs + minmax:
                    v, nm = _materialize(vop, nop); agg_arrays[i] = wdb_agg.group_agg(gcodes, K, fn, v, nm)
            for i, arr in agg_arrays.items():
                cr = col_results[i]; col_results[i] = ('arr', arr, cr[2], cr[3])
        elif expr_counts is not None:
            counts = expr_counts                                  # only arithmetic aggregates -> counts from codegen
        else:                                                     # only COUNT(*) / key columns -> counts-only pass
            gc = wdb_agg._slice(_group_op(), 0, n)
            gcodes = np.zeros(n, dtype=np.int64) if gc is None else gc.astype(np.int64, copy=False)
            _m = get_mask()
            if _m is not None: gcodes = gcodes[_m]
            counts = wdb_agg.group_counts(gcodes, K)
    # No GROUP BY -> exactly one output row (the grand total), even over zero rows (COUNT=0, SUM/MIN/MAX=NULL,
    # matching SQL). With a GROUP BY, empty groups are dropped.
    present = np.array([0]) if not gkeys else np.nonzero(counts > 0)[0]
    present = _topk_prefilter(tree, proj, col_results, counts, present, gkeys)   # bounded top-K: drop non-winners pre-assembly

    # ---- assemble rows (vectorised) ----
    # Decode every present group's composite code into per-key code arrays in one shot, bulk-decode each key
    # column through its dict, and bulk-pull each aggregate -- then zip columns into row tuples. The old path
    # was a python loop over present groups calling fetch() per cell, which dominated high-card output.
    radices = [k['K'] for k in gkeys]
    kc_arr = []
    if gkeys:
        comp = (present.astype(np.int64, copy=True) if gid_to_comp is None
                else np.asarray(gid_to_comp, dtype=np.int64)[present].copy())   # factorised id -> composite
        kc_arr = [None] * len(gkeys)
        for j in range(len(gkeys) - 1, -1, -1):
            kc_arr[j] = comp % radices[j]; comp //= radices[j]

    # Native columnar result: hand back {name: ndarray} built from the already-native group arrays,
    # skipping the per-cell _pyval decode and the row-tuple zip. Only when no HAVING/ORDER/LIMIT needs
    # row materialisation (those stay on the row path). This is the big lever at high cardinality, where
    # the per-row assembly -- not the gather -- dominated db.run.
    if (columnar and tree.args.get('having') is None and tree.args.get('order') is None
            and wdb_sql._limit(tree) is None):
        names = [wdb_sql._alias(p) for p in proj]
        out = {}
        for i, p in enumerate(proj):
            r = col_results[i]; nm = names[i]
            if r[0] == 'key':
                gk = gkeys[r[1]]
                if 'labels' in gk:
                    out[nm] = np.asarray(gk['labels'], dtype=object)[kc_arr[r[1]]]
                else:
                    out[nm] = np.asarray(_bulk_keyvals(gk['seg'], gk['pcol'], kc_arr[r[1]]), dtype=object)
            elif r[0] == 'count':
                out[nm] = counts[present]
            else:
                picked = r[1][present]
                out[nm] = picked.astype(np.int64).view(f"datetime64[{r[3]}]") if r[2] else picked
        _bump_fast()
        return out, names

    col_lists = []
    for i, p in enumerate(proj):
        r = col_results[i]
        if r[0] == 'key':
            gk = gkeys[r[1]]
            if 'labels' in gk:                            # affine/factorised key: gid -> value via label table
                _lab = gk['labels']; col_lists.append([_lab[c] for c in kc_arr[r[1]].tolist()])
            else:
                col_lists.append(_bulk_keyvals(gk['seg'], gk['pcol'], kc_arr[r[1]]))
        elif r[0] == 'count':
            col_lists.append(counts[present].tolist())
        else:
            picked = r[1][present]
            if r[2]:                                      # datetime epoch -> datetime64 -> _pyval string
                unit = r[3]
                col_lists.append([(wdb_sql._pyval(np.int64(v).view(f'datetime64[{unit}]')) if v is not None
                                   else None) for v in picked.tolist()])
            else:
                col_lists.append([wdb_sql._pyval(v) for v in picked.tolist()])
    rows = list(zip(*col_lists)) if col_lists else [() for _ in present]

    global _FAST_HITS; _FAST_HITS += 1
    having = tree.args.get('having')
    if having is not None:
        rows = wdb_sql._apply_having(rows, proj, having.this, None)   # fused path must filter too
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
def _key_is_unique(db, table, col):
    """True if `col` in `table` is a single clean segment with all-distinct values (a candidate join parent)."""
    try: seg, _ = _solo_segment(db, table)
    except _FastUnsupported: return False
    pc = db.cat.phys_map(table).get(col, col)
    if pc not in seg.cols: return False
    v = wdb_sql._col(seg, pc)[0]
    return v is not None and len(v) == len(np.unique(np.asarray(v)))


def _hash_pointer(db, ctbl, ckey, cseg, ptbl, pkey, pseg):
    """Build a child->parent gather pointer at query time via a hash probe (the parent key must be unique).
    For each child row, the parent row whose key matches. INNER + row-preserving only: a partial match would
    drop child rows, so that raises _FastUnsupported and the query falls back. A hash join becomes a pointer,
    and every downstream gather/predicate/aggregate stays on the same fused chain."""
    cp = db.cat.phys_map(ctbl).get(ckey, ckey); pp = db.cat.phys_map(ptbl).get(pkey, pkey)
    ck = np.asarray(wdb_sql._col(cseg, cp)[0]); pk = np.asarray(wdb_sql._col(pseg, pp)[0])
    pidx = pd.Index(pk)
    if not pidx.is_unique: raise _FastUnsupported                 # many-to-many -> not a pointer
    ptr = pidx.get_indexer(ck)
    if (ptr < 0).any(): raise _FastUnsupported                    # unmatched child rows -> would drop -> fall back
    return ptr.astype(np.int64)


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
        elif _key_is_unique(db, tB, kB):                                         # non-FK: parent = unique-key side
            child_a, parent_a, fk_col = aA, aB, ('hash', kA, kB)                 # built as a runtime hash pointer
        elif _key_is_unique(db, tA, kA):
            child_a, parent_a, fk_col = aB, aA, ('hash', kB, kA)
        else:
            raise _FastUnsupported                                               # neither key unique -> not a pointer
        if child_a in edges: raise _FastUnsupported                              # one parent per child (tree)
        edges[child_a] = (parent_a, fk_col); parents.add(parent_a)

    if not edges:                                                                # 0 joins: single-table query
        if len(tables) != 1: raise _FastUnsupported                              # multiple tables, no FK edge
        fact = tables[0][1]
    else:
        fact_candidates = [a for a in edges if a not in parents]
        if len(fact_candidates) != 1: raise _FastUnsupported                     # need a single rooted fact
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
                if isinstance(fk_col, tuple) and fk_col and fk_col[0] == 'hash':
                    p = _hash_pointer(db, alias2t[child_a], fk_col[1], seg_of[child_a],
                                      alias2t[parent_a], fk_col[2], seg_of[parent_a])
                else:
                    p = db.fk_pointer(sp_of[child_a], fk_col)
                if p is None: raise _FastUnsupported
                cc = composed[child_a]
                composed[parent_a] = p if cc is None else p[cc]                  # compose by gather
                progress = True
    if any(a not in composed for a in keep): raise _FastUnsupported              # kept tables must connect
    return dict(fact=fact, alias2t=alias2t, seg_of=seg_of, composed=composed, n=seg_of[fact].N)
