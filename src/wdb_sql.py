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

    # ---- presence (deleted rows) seeds the mask; WHERE is AND-ed onto it ----
    mask = seg.presence_mask()          # bool[N] True=live, or None if all live
    where = tree.args.get('where')
    if where is not None:
        wm = _eval_pred(seg, where.this, seg_col)
        mask = wm if mask is None else (wm & mask)

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
            # row projection (SELECT cols ... [WHERE] [ORDER BY] [LIMIT]) -> return rows
            cols = [seg_col(_colname(p if not isinstance(p,E.Alias) else p.this)) for p in proj]
            idx = np.nonzero(mask)[0] if mask is not None else np.arange(N)
            order = tree.args.get('order'); lim = _limit(tree)
            if order is None and lim is not None: idx = idx[:lim]   # no ordering: limit early (fast path)
            out = []
            colvals = {c: seg.values(c) for c in cols}
            for i in idx: out.append(tuple(_pyval(colvals[c][i]) for c in cols))
            if order is not None:                                   # ORDER BY: sort full set, then LIMIT
                out = _apply_order(out, proj, order)
                if lim is not None: out = out[:lim]
            return out, [_alias(p) for p in proj]

    # ---- GROUP BY path (group on CODES: NULL becomes its own group naturally) ----
    gnames = [seg_col(g) for g in gcols]
    combo = None; metas=[]   # metas[i] = unique code values for key i
    for gn in gnames:
        gc = seg.codes(gn); gc = gc[mask] if mask is not None else gc
        u, inv = np.unique(gc, return_inverse=True); metas.append((gn, u))
        combo = inv if combo is None else combo*len(u)+inv
    uc, counts = np.unique(combo, return_counts=True)
    def decombo(cv):
        out=[]; x=cv
        for _,u in reversed(metas): out.append(u[x%len(u)]); x//=len(u)
        return list(reversed(out))   # per-key CODE values, original order
    def codeval(colname, code):
        c=seg.cols[colname]
        if c['has_null'] and int(code)==c['V']-1: return None
        return seg.fetch(colname, int(code))
    agg_specs = [(p, _agg_kind(p)) for p in proj]
    order = np.argsort(combo, kind='stable'); combo_s=combo[order]
    gstarts = np.searchsorted(combo_s, uc); gends = np.r_[gstarts[1:], len(combo_s)]
    aggcache={}
    def groupagg(colname, fn, gi):
        if colname not in aggcache:
            arr, nm = _col(seg, seg_col(colname))
            if mask is not None:
                arr = arr[mask]; nm = nm[mask] if nm is not None else None
            arrf = arr.astype(np.float64)
            aggcache[colname]=(arrf[order], (nm[order] if nm is not None else None))
        vs, vn = aggcache[colname]; sl=slice(gstarts[gi],gends[gi]); seg_v=vs[sl]
        if vn is not None: seg_v = seg_v[~vn[sl]]      # SQL: aggregates ignore NULLs
        if fn=='COUNT': return len(seg_v)
        if len(seg_v)==0: return None
        return {'SUM':seg_v.sum(),'AVG':seg_v.mean(),'MIN':seg_v.min(),'MAX':seg_v.max()}[fn]
    rows=[]
    for gi,cv in enumerate(uc):
        keycodes = decombo(cv); rowout=[]; ki=0
        for p,kind in agg_specs:
            if kind is None:
                rowout.append(_pyval(codeval(metas[ki][0], keycodes[ki]))); ki+=1
            elif kind[0]=='COUNT_STAR':
                rowout.append(int(counts[gi]))
            else:
                fn,cn=kind; rowout.append(_pyval(groupagg(cn, fn, gi)))
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
    if isinstance(v, np.datetime64):
        s = str(v)
        # render as plain date when the value is exactly midnight (DATE-like), else full timestamp
        if 'T00:00:00' in s and s.endswith('00:00:00.000000'): return s[:10]
        return s.replace('T',' ')
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,)): return float(v)
    return v
def _limit(tree):
    lim = tree.args.get('limit')
    if lim is None: return None
    return int(lim.expression.this) if hasattr(lim.expression,'this') else int(lim.text('expression'))
def _col(seg, name):
    """Typed array + null mask. arr is int64/float64/object(bytes); nulls filled with a
    sentinel and flagged in nmask (or None if the column has no nulls). Override values are
    appended to the dictionary at synthetic codes V.. so effective codes resolve correctly."""
    import struct as _st
    c = seg.cols[name]; codes = seg.codes(name); dv = seg._typed_dict(name)
    ov = seg._override_vals_typed(name)        # [] if none; sit at codes V, V+1, ...
    if c['has_null']:
        nullcode = c['V'] - 1; nmask = (codes == nullcode)
        if c['dt'] in (0, 3):
            lut = np.array(list(dv) + [0] + list(ov), dtype=np.int64)
        elif c['dt'] == 2:
            lut = np.array(list(dv) + [np.nan] + list(ov), dtype=np.float64)
        else:
            lut = np.empty(c['V'] + len(ov), dtype=object)
            for i,v in enumerate(dv): lut[i]=v
            lut[nullcode] = b''
            for k,v in enumerate(ov): lut[c['V']+k]=v
        return lut[codes], nmask
    if c['dt'] in (0, 3):
        base = np.asarray(dv, dtype=np.int64)
        if ov: base = np.concatenate([base, np.asarray(ov, dtype=np.int64)])
        return base[codes], None
    if c['dt'] == 2:
        base = np.asarray(dv, dtype=np.float64)
        if ov: base = np.concatenate([base, np.asarray(ov, dtype=np.float64)])
        return base[codes], None
    lut = np.empty(len(dv) + len(ov), dtype=object)
    for i,v in enumerate(dv): lut[i]=v
    for k,v in enumerate(ov): lut[len(dv)+k]=v
    return lut[codes], None

def _parse_temporal(litstr, unit):
    return int(np.datetime64(str(litstr).replace(' ','T')).astype(f'datetime64[{unit}]').view('int64'))

def _lit_for_col(seg, colname, lit, arr_kind):
    """Convert a sqlglot Literal (or negated numeric literal -5 -> Neg(Literal 5)) node to a
    value comparable with column `colname`."""
    if isinstance(lit, E.Neg):
        return -_lit_for_col(seg, colname, lit.this, arr_kind)
    c = seg.cols[colname]
    if c['dt'] == 3:                       # datetime: parse string/number to int64 epoch
        return _parse_temporal(lit.this, seg.unit(colname))
    if not lit.is_string:
        return int(lit.this) if arr_kind in 'iu' else float(lit.this)
    v = lit.this.encode()
    return v

def _agg_scalar(seg, p, mask, seg_col):
    kind=_agg_kind(p)
    if kind[0]=='COUNT_STAR': return int(mask.sum()) if mask is not None else seg.N
    fn,cn=kind
    arr, nm = _col(seg, seg_col(cn))
    if mask is not None:
        arr = arr[mask]; nm = nm[mask] if nm is not None else None
    if nm is not None: arr = arr[~nm]            # SQL: aggregates ignore NULLs
    if kind[0]=='COUNT': return int(len(arr))    # COUNT(col) = non-null count
    arr = arr.astype(np.float64)
    if len(arr)==0: return None
    return _pyval({'SUM':arr.sum(),'AVG':arr.mean(),'MIN':arr.min(),'MAX':arr.max()}[fn])

def _eval_pred(seg, node, seg_col):
    if isinstance(node, E.And): return _eval_pred(seg,node.this,seg_col) & _eval_pred(seg,node.expression,seg_col)
    if isinstance(node, E.Or):  return _eval_pred(seg,node.this,seg_col) | _eval_pred(seg,node.expression,seg_col)
    if isinstance(node, E.Not): return ~_eval_pred(seg,node.this,seg_col)
    if isinstance(node, E.Paren): return _eval_pred(seg,node.this,seg_col)
    if isinstance(node, (E.EQ,E.NEQ,E.GT,E.LT,E.GTE,E.LTE)):
        col=_colname(node.this); lit=node.expression
        a, nmask = _col(seg, seg_col(col))
        if not (isinstance(lit,E.Literal) or (isinstance(lit,E.Neg) and isinstance(lit.this,E.Literal))):
            raise NotImplementedError("non-literal RHS")
        v = _lit_for_col(seg, seg_col(col), lit, a.dtype.kind)
        if a.dtype.kind not in 'iuf' and isinstance(v,int) and seg.cols[seg_col(col)]['dt']!=3: v=str(v).encode()
        import operator
        op={E.EQ:operator.eq,E.NEQ:operator.ne,E.GT:operator.gt,E.LT:operator.lt,E.GTE:operator.ge,E.LTE:operator.le}[type(node)]
        res = op(a,v)
        if nmask is not None: res = res & ~nmask   # SQL: NULL fails any comparison
        return res
    if isinstance(node, E.Between):
        col=_colname(node.this); a, nmask = _col(seg, seg_col(col))
        lo=_lit_for_col(seg, seg_col(col), node.args['low'], a.dtype.kind)
        hi=_lit_for_col(seg, seg_col(col), node.args['high'], a.dtype.kind)
        res=(a>=lo)&(a<=hi)
        if nmask is not None: res = res & ~nmask
        return res
    if isinstance(node, E.In):
        col=_colname(node.this); a, nmask = _col(seg, seg_col(col))
        lits=node.args.get('expressions') or []
        vals=[]
        for L in lits:
            if not isinstance(L,E.Literal): raise NotImplementedError("IN with non-literal / subquery")
            if seg.cols[seg_col(col)]['dt']==3: vals.append(_parse_temporal(L.this, seg.unit(seg_col(col))))
            elif L.is_string: vals.append(L.this.encode() if a.dtype.kind not in 'iuf' else L.this)
            else: vals.append(int(L.this) if a.dtype.kind in 'iu' else (float(L.this) if a.dtype.kind=='f' else str(L.this).encode()))
        if a.dtype.kind in 'iuf':
            m=np.isin(a, np.array(vals, dtype=a.dtype))
        else:
            sv=set(vals); m=np.fromiter((x in sv for x in a), dtype=bool, count=len(a))
        if nmask is not None: m = m & ~nmask
        return m
    if isinstance(node, E.Like) or isinstance(node, E.ILike):
        import re
        col=_colname(node.this); a, nmask = _col(seg, seg_col(col))
        pat=node.expression.this  # the LIKE pattern string
        # SQL LIKE -> regex. re.escape leaves % and _ bare (not special), so replace
        # them AFTER escaping: % => .*  (any run),  _ => .  (single char).
        rx='^'+re.escape(pat).replace('%','.*').replace('_','.')+'$'
        flags=re.DOTALL|(re.IGNORECASE if isinstance(node,E.ILike) else 0)
        cre=re.compile(rx, flags)
        def tostr(x): return x.decode('utf-8','surrogatepass') if isinstance(x,(bytes,bytearray)) else ('' if x is None else str(x))
        m = np.fromiter((bool(cre.match(tostr(x))) for x in a), dtype=bool, count=len(a))
        if nmask is not None: m = m & ~nmask
        return m
    if isinstance(node, E.Is):
        col=_colname(node.this); a, nmask = _col(seg, seg_col(col))
        if isinstance(node.expression, E.Null):
            return nmask if nmask is not None else np.zeros(len(a), dtype=bool)  # IS NULL
        raise NotImplementedError("IS <non-null-literal>")
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


def _eval_expr(seg, node):
    """Evaluate a scalar UPDATE RHS expression to a full-column array (length seg.N),
    reading override-aware values. Supports column references, numeric/string literals,
    + - * / %, unary minus, and parentheses. numpy semantics are chosen to match DuckDB
    for these operators (notably '/' is true division in both), so the cold-segment path
    and the DuckDB-evaluated parquet path agree. NULL-in-arithmetic is a deferred edge;
    pure column-copy (SET a=b) is null-safe via object arrays."""
    t = type(node)
    if t is E.Paren:
        return _eval_expr(seg, node.this)
    if t is E.Column:
        return seg.values(node.name)
    if t is E.Neg:
        return -_eval_expr(seg, node.this)
    if t is E.Literal:
        if node.args.get('is_string'):
            return np.full(seg.N, node.this, dtype=object)
        s = node.this
        return np.full(seg.N, float(s) if ('.' in s or 'e' in s.lower()) else int(s))
    if t is E.Add: return _eval_expr(seg, node.left) + _eval_expr(seg, node.right)
    if t is E.Sub: return _eval_expr(seg, node.left) - _eval_expr(seg, node.right)
    if t is E.Mul: return _eval_expr(seg, node.left) * _eval_expr(seg, node.right)
    if t is E.Div: return _eval_expr(seg, node.left) / _eval_expr(seg, node.right)
    if t is E.Mod: return _eval_expr(seg, node.left) % _eval_expr(seg, node.right)
    raise NotImplementedError(f"unsupported UPDATE expression: {node.sql()}")
