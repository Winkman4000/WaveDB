#!/usr/bin/env python3
"""WaveDB SQL executor — STEP 1: single-table SELECT with WHERE, GROUP BY, HAVING,
ORDER BY, LIMIT, returning the FULL correct result set. Parses real SQL via sqlglot;
executes against a WVDB3 Segment. Correctness first; optimization second.
Unsupported shapes raise NotImplementedError (honest failure, never silent wrong answer)."""
import sqlglot, sqlglot.expressions as E, numpy as np
from wdb_engine import Segment

def _colname(node):
    if isinstance(node, E.Column): return node.name
    return None

def execute(seg: Segment, sql: str, col_map=None):
    """col_map: optional {sql_name -> segment_name}; default identity."""
    tree = sqlglot.parse_one(sql, read='duckdb')
    if not isinstance(tree, E.Select): raise NotImplementedError(f"top-level {type(tree).__name__}")
    if tree.args.get('joins'): raise NotImplementedError("JOIN (step 2)")
    if tree.args.get('with'): raise NotImplementedError("CTE/WITH (later)")
    def seg_col(nm): return (col_map or {}).get(nm, nm)
    N = seg.N

    # ---- WHERE -> boolean mask ----
    mask = None
    where = tree.args.get('where')
    if where is not None:
        mask = _eval_pred(seg, where.this, seg_col)

    # ---- projections ----
    proj = tree.expressions  # list of selected exprs
    group = tree.args.get('group')
    gcols = [ _colname(g) for g in group.expressions ] if group else []

    if not gcols:
        # no GROUP BY: either pure aggregates over (masked) rows, or row projection
        if any(_is_agg(p) for p in proj):
            row = []
            for p in proj: row.append(_agg_scalar(seg, p, mask, seg_col))
            return [tuple(row)], [_alias(p) for p in proj]
        else:
            # row projection (SELECT cols ... [WHERE]) -> return rows
            cols = [seg_col(_colname(p if not isinstance(p,E.Alias) else p.this)) for p in proj]
            idx = np.nonzero(mask)[0] if mask is not None else np.arange(N)
            lim = _limit(tree)
            if lim is not None: idx = idx[:lim]
            out = []
            colvals = {c: seg.values(c) for c in cols}
            for i in idx: out.append(tuple(_pyval(colvals[c][i]) for c in cols))
            return out, [_alias(p) for p in proj]

    # ---- GROUP BY path ----
    keys = [seg.values(seg_col(g))[mask] if mask is not None else seg.values(seg_col(g)) for g in gcols]
    combo = np.zeros(len(keys[0]), dtype=np.int64); metas=[]
    for k in keys:
        u, inv = np.unique(k, return_inverse=True); metas.append(u); combo = combo*len(u)+inv
    uc, first_idx, counts = np.unique(combo, return_index=True, return_counts=True)
    def decombo(cv):
        out=[]; x=cv
        for u in reversed(metas): out.append(u[x%len(u)]); x//=len(u)
        return list(reversed(out))
    # build each output row
    agg_specs = [(p, _agg_kind(p)) for p in proj]
    # precompute per-group aggregate values where needed
    rows=[]
    # group membership for aggregates
    order = np.argsort(combo, kind='stable'); combo_s=combo[order]
    gstarts = np.searchsorted(combo_s, uc)
    gends = np.r_[gstarts[1:], len(combo_s)]
    aggcache={}
    def groupagg(colname, fn, gi):
        key=(colname,fn)
        if key not in aggcache:
            vals = seg.values(seg_col(colname)).astype(np.float64)
            vals = vals[mask] if mask is not None else vals
            vs = vals[order]
            aggcache[key]=vs
        vs=aggcache[key]; seg_v=vs[gstarts[gi]:gends[gi]]
        return {'SUM':seg_v.sum(),'AVG':seg_v.mean(),'MIN':seg_v.min(),'MAX':seg_v.max(),'COUNT':len(seg_v)}[fn]
    for gi,cv in enumerate(uc):
        keyvals = decombo(cv)
        rowout=[]
        ki=0
        for p,kind in agg_specs:
            if kind is None:  # group key column
                rowout.append(_pyval(keyvals[ki])); ki+=1
            elif kind[0]=='COUNT_STAR':
                rowout.append(int(counts[gi]))
            else:
                fn,cn=kind
                rowout.append(_pyval(groupagg(cn, fn, gi)))
        rows.append(tuple(rowout))

    # ---- HAVING ----
    having = tree.args.get('having')
    if having is not None:
        rows = _apply_having(rows, proj, having.this, seg_col)

    # ---- ORDER BY ----
    rows = _apply_order(rows, proj, tree.args.get('order'))

    # ---- LIMIT ----
    lim = _limit(tree)
    if lim is not None: rows = rows[:lim]
    return rows, [_alias(p) for p in proj]

# ---------- helpers ----------
def _alias(p):
    if isinstance(p, E.Alias): return p.alias
    if isinstance(p, E.Column): return p.name
    return p.sql()
def _is_agg(p):
    return p.find(E.AggFunc) is not None
def _agg_kind(p):
    inner = p.this if isinstance(p,E.Alias) else p
    if isinstance(inner, E.Count):
        if isinstance(inner.this, E.Star) or inner.this is None: return ('COUNT_STAR',)
        return ('COUNT', _colname(inner.this))
    for cls,nm in [(E.Sum,'SUM'),(E.Avg,'AVG'),(E.Min,'MIN'),(E.Max,'MAX')]:
        if isinstance(inner, cls): return (nm, _colname(inner.this))
    return None  # not an aggregate -> group key
def _pyval(v):
    if isinstance(v,(bytes,bytearray)):
        try: return v.decode('utf-8','surrogatepass')
        except: return v
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,)): return float(v)
    return v
def _limit(tree):
    lim = tree.args.get('limit')
    if lim is None: return None
    return int(lim.expression.this) if hasattr(lim.expression,'this') else int(lim.text('expression'))
def _agg_scalar(seg, p, mask, seg_col):
    kind=_agg_kind(p)
    if kind[0]=='COUNT_STAR': return int(mask.sum()) if mask is not None else seg.N
    fn,cn=kind; vals=seg.values(seg_col(cn)).astype(np.float64); vals=vals[mask] if mask is not None else vals
    if kind[0]=='COUNT': return int(len(vals))
    return _pyval({'SUM':vals.sum(),'AVG':vals.mean(),'MIN':vals.min(),'MAX':vals.max()}[fn])

def _eval_pred(seg, node, seg_col):
    if isinstance(node, E.And): return _eval_pred(seg,node.this,seg_col) & _eval_pred(seg,node.expression,seg_col)
    if isinstance(node, E.Or):  return _eval_pred(seg,node.this,seg_col) | _eval_pred(seg,node.expression,seg_col)
    if isinstance(node, E.Not): return ~_eval_pred(seg,node.this,seg_col)
    if isinstance(node, E.Paren): return _eval_pred(seg,node.this,seg_col)
    if isinstance(node, (E.EQ,E.NEQ,E.GT,E.LT,E.GTE,E.LTE)):
        col=_colname(node.this); lit=node.expression
        a=seg.values(seg_col(col))
        if isinstance(lit,E.Literal) and not lit.is_string: v=int(lit.this) if a.dtype.kind in 'iu' else float(lit.this)
        elif isinstance(lit,E.Literal): v=lit.this.encode()
        else: raise NotImplementedError("non-literal RHS")
        if a.dtype.kind not in 'iuf' and isinstance(v,int): v=str(v).encode()
        import operator
        op={E.EQ:operator.eq,E.NEQ:operator.ne,E.GT:operator.gt,E.LT:operator.lt,E.GTE:operator.ge,E.LTE:operator.le}[type(node)]
        return op(a,v)
    if isinstance(node, E.Between):
        col=_colname(node.this); a=seg.values(seg_col(col))
        lo=int(node.args['low'].this); hi=int(node.args['high'].this)
        return (a>=lo)&(a<=hi)
    raise NotImplementedError(f"predicate {type(node).__name__}")

def _apply_having(rows, proj, node, seg_col):
    # map aggregate expr in HAVING to its column index in proj
    def val(row, expr):
        for i,p in enumerate(proj):
            inner=p.this if isinstance(p,E.Alias) else p
            if inner.sql()==expr.sql(): return row[i]
        # COUNT(*) match
        raise NotImplementedError("HAVING references non-projected expr")
    import operator
    def test(row,n):
        if isinstance(n,E.And): return test(row,n.this) and test(row,n.expression)
        if isinstance(n,E.Or): return test(row,n.this) or test(row,n.expression)
        if isinstance(n,(E.GT,E.LT,E.GTE,E.LTE,E.EQ,E.NEQ)):
            lhs=val(row,n.this); rhs=float(n.expression.this)
            op={E.GT:operator.gt,E.LT:operator.lt,E.GTE:operator.ge,E.LTE:operator.le,E.EQ:operator.eq,E.NEQ:operator.ne}[type(n)]
            return op(lhs,rhs)
        raise NotImplementedError("HAVING op")
    return [r for r in rows if test(r,node)]

def _apply_order(rows, proj, order):
    if order is None: return rows
    keys=[]
    for o in order.expressions:
        desc = o.args.get('desc') or False
        target = o.this
        # find column index in proj by sql match or name
        idx=None
        for i,p in enumerate(proj):
            inner=p.this if isinstance(p,E.Alias) else p
            if inner.sql()==target.sql() or _alias(p)==(target.name if isinstance(target,E.Column) else None):
                idx=i; break
        if idx is None: raise NotImplementedError("ORDER BY non-projected expr")
        keys.append((idx,desc))
    for idx,desc in reversed(keys):
        rows=sorted(rows, key=lambda r:(r[idx] is None, r[idx]), reverse=desc)
    return rows
