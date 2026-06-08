#!/usr/bin/env python3
"""WaveDB SQL executor — STEP 1: single-table SELECT with WHERE, GROUP BY, HAVING,
ORDER BY, LIMIT, returning the FULL correct result set. Parses real SQL via sqlglot;
executes against a WVDB3 Segment. Correctness first; optimization second.
Unsupported shapes raise NotImplementedError (honest failure, never silent wrong answer)."""
import sqlglot, sqlglot.expressions as E, numpy as np
from wdb_engine import Segment

_SLICE_HITS = 0   # count of queries answered via the cluster-slice fast path (tests/telemetry)
SLICE_RESIDENT_BUDGET = 1 << 31   # per-column N*8-byte budget to keep a decoded column resident

def _slice_vals(seg, cn, lo, hi, rmask):
    """Measure values for the cluster slice [lo,hi) under residual rmask. Decode-once-resident when
    the column fits the RAM budget (serving throughput: no per-query dict gather); else lazy partial
    decode. Skips the boolean-index copy when rmask selects the whole slice (the consumed-key case)."""
    base = seg.resident_values(cn) if seg.N * 8 <= SLICE_RESIDENT_BUDGET else None
    a = base[lo:hi] if base is not None else seg.values_range(cn, lo, hi)
    return a if rmask is None else a[rmask]

def _cluster_will_slice(seg, tree, col_map):
    """True iff this query would take the cluster-slice fast path: clustered segment, no GROUP BY,
    no deleted rows, and the WHERE pins a range/eq on the cluster key. Cheap (only inspects the small
    WHERE); the router uses it to divert clustered scalar-aggregate queries off the fused scan."""
    if seg.cluster_meta() is None or seg.presence_mask() is not None: return False
    if tree.args.get('group') is not None: return False
    where = tree.args.get('where')
    if where is None: return False
    def seg_col(nm): return nm if col_map is None else col_map.get(nm, nm)
    try:
        return _cluster_slice(seg, where.this, seg_col) is not None
    except NotImplementedError:
        return False

_GROUP_SLICE_HITS = 0   # count of GROUP BYs answered via the clustered range-walk (tests/telemetry)

def _has_count_distinct(proj):
    """True if any projection is COUNT(DISTINCT col). Such grouped queries keep the fused
    _grouped_cd path (a bincount cell-table) -- it beats the group-slice's per-group np.unique."""
    for p in proj:
        _inn = p.this if isinstance(p, E.Alias) else p
        if isinstance(_inn, E.Count) and isinstance(_inn.this, E.Distinct):
            return True
    return False

def _cluster_will_group_slice(seg, tree, col_map):
    """True iff a single-column GROUP BY on the cluster key with no WHERE/HAVING/ORDER/LIMIT and no
    deleted rows -- the rows are already grouped into the cluster ranges, so no sort/scatter."""
    if seg.cluster_meta() is None or seg.presence_mask() is not None: return False
    if (tree.args.get('where') is not None or tree.args.get('having') is not None
            or tree.args.get('order') is not None or _limit(tree) is not None): return False
    if _has_count_distinct(tree.expressions): return False          # fused _grouped_cd is faster
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 1: return False
    gn = _colname(g.expressions[0])
    if gn is None: return False
    key = gn if col_map is None else col_map.get(gn, gn)
    return seg.cluster_meta()['key'] == key

def _grouped_slice(seg, proj, gn, seg_col):
    """GROUP BY the cluster key as a range walk: each cluster range IS a group, so per-group
    aggregates are sequential reductions over resident slices -- no group-code scan, no scatter,
    no sort. Returns (rows, cols) or raises NotImplementedError on an unsupported agg (caller
    falls back to the general group path). Gated by the caller to no WHERE/HAVING/ORDER/LIMIT."""
    cm = seg.cluster_meta(); off = cm['offsets']; K = len(off) - 1
    agg_specs = [(p, _agg_kind(p)) for p in proj]
    rows = []
    for gi in range(K):
        lo, hi = int(off[gi]), int(off[gi + 1]); rowout = []
        for p, kind in agg_specs:
            _inn = p.this if isinstance(p, E.Alias) else p
            if isinstance(_inn, E.Count) and isinstance(_inn.this, E.Distinct):
                _dx = _inn.this.expressions
                if len(_dx) != 1 or not isinstance(_dx[0], E.Column): raise NotImplementedError
                cn = seg_col(_dx[0].name); c = seg.cols[cn]
                if c['has_null'] or seg._overrides(cn) is not None: raise NotImplementedError
                rc = seg.values_range(cn, lo, hi) if c['mode'] == 4 else seg._raw_codes_range(cn, lo, hi)
                rowout.append(int(np.unique(rc).size))
            elif kind is None:                                  # the group-key column: constant on the slice
                rowout.append(_pyval(seg.values_range(seg_col(gn), lo, lo + 1)[0]))
            elif kind[0] == 'COUNT_STAR':
                rowout.append(int(hi - lo))
            else:
                fn, cn_name = kind; cn = seg_col(cn_name); c = seg.cols[cn]
                if c['has_null'] or seg._overrides(cn) is not None: raise NotImplementedError
                a = seg.resident_values(cn)[lo:hi]
                if len(a) == 0: rowout.append(None)
                elif fn == 'COUNT': rowout.append(int(len(a)))
                elif fn in ('MIN', 'MAX'):
                    v = a.min() if fn == 'MIN' else a.max()
                    if c['dt'] == 3 and isinstance(v, (int, np.integer)):
                        v = np.int64(v).view(f"datetime64[{seg.unit(cn)}]")
                    rowout.append(_pyval(v))
                else:
                    rowout.append(_pyval({'SUM': a.sum(dtype=np.float64), 'AVG': a.mean(dtype=np.float64)}[fn]))
        rows.append(tuple(rowout))
    return rows, [_alias(p) for p in proj]

def _colname(node):
    if isinstance(node, E.Column): return node.name
    return None

def _to_physical(node, col_map):
    """Deep-copy a sqlglot expression with column names remapped logical->physical. Used ONLY for
    the DuckDB-side queries (hot buffer / canonical buffer parquet), whose columns carry the
    physical (storage) name. Identity (returns node unchanged, no copy) when col_map has no
    non-trivial entries -- so pre-ALTER queries pay nothing."""
    import copy
    if not col_map or all(k == v for k, v in col_map.items()):
        return node
    t = copy.deepcopy(node)
    for col in t.find_all(E.Column):
        nm = col.name
        if nm in col_map and col_map[nm] != nm:
            col.this.set('this', col_map[nm])
    return t

def execute(seg: Segment, sql: str, col_map=None, tree=None):
    """col_map: optional {sql_name -> segment_name}; default identity.
    tree: optional pre-parsed sqlglot AST -- lets the router reuse its parse (no second parse)."""
    if tree is None:
        tree = sqlglot.parse_one(sql, read='duckdb')
    if not isinstance(tree, E.Select): raise NotImplementedError(f"top-level {type(tree).__name__}")
    if tree.args.get('joins'): raise NotImplementedError("JOIN (step 2)")
    if tree.args.get('with'): raise NotImplementedError("CTE/WITH (later)")
    def seg_col(nm):
        if col_map is None: return nm                      # direct Segment use: identity, accept all
        if nm not in col_map: raise NotImplementedError(f"unknown column: {nm!r}")
        return col_map[nm]                                 # schema view: complete logical->physical
    N = seg.N

    # ---- projections / shape ----
    proj = tree.expressions  # list of selected exprs
    group = tree.args.get('group')
    gcols = [ _colname(g) for g in group.expressions ] if group else []
    where = tree.args.get('where')

    # ---- cluster-slice fast path: tried BEFORE the full-column WHERE eval, which it avoids.
    # narrow-before-expand -- WHERE pins a range/eq on the cluster key, so decode only that slice
    # off the sorted segment. Falls through on anything unsupported; only when no deleted rows. ----
    if (not gcols and where is not None and any(_is_agg(p) for p in proj)
            and seg.cluster_meta() is not None and seg.presence_mask() is None):
        try:
            sl = _cluster_slice(seg, where.this, seg_col)
            if sl is not None:
                lo, hi, consumed = sl
                rmask = _eval_pred_range(seg, where.this, seg_col, lo, hi, consumed)
                row = [_agg_scalar_range(seg, p, lo, hi, rmask, seg_col) for p in proj]
                global _SLICE_HITS; _SLICE_HITS += 1
                return [tuple(row)], [_alias(p) for p in proj]
        except NotImplementedError:
            pass

    # ---- presence (deleted rows) seeds the mask; WHERE is AND-ed onto it ----
    mask = seg.presence_mask()          # bool[N] True=live, or None if all live
    if where is not None:
        wm = _eval_pred(seg, where.this, seg_col)
        mask = wm if mask is None else (wm & mask)

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
            order = tree.args.get('order'); lim = _limit(tree); distinct = tree.args.get('distinct') is not None
            if order is None and lim is not None and not distinct: idx = idx[:lim]   # no order/distinct: limit early
            out = []
            colvals = {c: seg.values(c) for c in cols}
            for i in idx: out.append(tuple(_pyval(colvals[c][i]) for c in cols))
            if distinct:                                           # SELECT DISTINCT -> dedup (order-preserving)
                seen = set(); ded = []
                for r in out:
                    if r not in seen: seen.add(r); ded.append(r)
                out = ded
            if order is not None: out = _apply_order(out, proj, order)   # ORDER BY: sort, then LIMIT
            if lim is not None: out = out[:lim]
            return out, [_alias(p) for p in proj]

    # ---- GROUP BY path ----
    # Group keys must be VALUE-identity. For dict-style columns (modes 0/1/2/3) seg.codes() is
    # already value-identity, and mode-5 codes are too (factorized). But mode-4 (affine) codes are
    # arange(N) -- identity-per-row -- so grouping on them would put every row in its own group;
    # there we factorize the decoded VALUES instead (vectorized np.unique on int64/datetime64).
    # Each meta carries how to turn a key index back into the emitted value.
    # Clustered single-key GROUP BY: the rows already sit in the cluster ranges, so each group is
    # a contiguous slice -- skip the factorise + sort + scatter and reduce each range directly.
    if (len(gcols) == 1 and where is None and seg.cluster_meta() is not None
            and seg.presence_mask() is None and tree.args.get('having') is None
            and tree.args.get('order') is None and _limit(tree) is None
            and not _has_count_distinct(proj)
            and seg.cluster_meta()['key'] == seg_col(gcols[0])):
        try:
            global _GROUP_SLICE_HITS
            r = _grouped_slice(seg, proj, gcols[0], seg_col)
            _GROUP_SLICE_HITS += 1
            return r
        except NotImplementedError:
            pass
    gnames = [seg_col(g) for g in gcols]
    combo = None; metas=[]   # metas[i] = (gn, kind, table): 'val'->unique values, 'code'->unique codes
    for gn in gnames:
        if seg.cols[gn]['mode'] == 4:
            vals = seg.values(gn); vals = vals[mask] if mask is not None else vals
            u, inv = np.unique(vals, return_inverse=True)      # value-identity keys
            metas.append((gn, 'val', u))
        else:
            gc = seg.codes(gn); gc = gc[mask] if mask is not None else gc
            u, inv = np.unique(gc, return_inverse=True)
            metas.append((gn, 'code', u))
        combo = inv if combo is None else combo*len(u)+inv
    uc, counts = np.unique(combo, return_counts=True)
    def decombo(cv):
        out=[]; x=cv
        for _,_,u in reversed(metas): out.append(int(x%len(u))); x//=len(u)
        return list(reversed(out))   # per-key INDEX into that key's table
    def keyval(ki, idx):
        gn, kk, u = metas[ki]
        if kk == 'val': return u[idx]                          # typed value directly
        code = u[idx]; c = seg.cols[gn]
        if c['has_null'] and int(code) == c['V']-1: return None
        return seg.fetch(gn, int(code))
    agg_specs = [(p, _agg_kind(p)) for p in proj]
    order = np.argsort(combo, kind='stable'); combo_s=combo[order]
    gstarts = np.searchsorted(combo_s, uc); gends = np.r_[gstarts[1:], len(combo_s)]
    aggcache={}
    def groupagg(colname, fn, gi):
        pcol = seg_col(colname)
        if colname not in aggcache:
            arr, nm = _col(seg, pcol)              # cache NATIVE dtype; float cast only for SUM/AVG
            if mask is not None:
                arr = arr[mask]; nm = nm[mask] if nm is not None else None
            aggcache[colname]=(arr[order], (nm[order] if nm is not None else None))
        vs, vn = aggcache[colname]; sl=slice(gstarts[gi],gends[gi]); seg_v=vs[sl]
        if vn is not None: seg_v = seg_v[~vn[sl]]      # SQL: aggregates ignore NULLs
        if fn=='COUNT': return len(seg_v)
        if len(seg_v)==0: return None
        if fn in ('MIN', 'MAX'):                       # native min/max (datetime/string included)
            v = seg_v.min() if fn == 'MIN' else seg_v.max()
            if seg.cols[pcol]['dt'] == 3 and isinstance(v, (int, np.integer)):
                v = np.int64(v).view(f"datetime64[{seg.unit(pcol)}]")
            return v
        segf = seg_v.astype(np.float64)
        return {'SUM': segf.sum(), 'AVG': segf.mean()}[fn]
    def groupdistinct(colname, gi):                # COUNT(DISTINCT col) per group (ignores NULL)
        ck = '#cdarr#' + colname
        if ck not in aggcache:
            pcol = seg_col(colname); c = seg.cols[pcol]; G = len(uc)
            if c['mode'] == 4:                                 # positional codes -> remap values to dense ids
                arr, nm = _col(seg, pcol); arr = arr[mask] if mask is not None else arr
                _u, codes = np.unique(arr, return_inverse=True); Vc = len(_u); nullmask = None
            else:                                              # value-identity codes (no value decode)
                codes = seg.codes(pcol); codes = codes[mask] if mask is not None else codes
                Vc = c['V']; nullmask = (codes == (c['V'] - 1)) if c['has_null'] else None
            gid = np.searchsorted(uc, combo)                   # per-row group index 0..G-1
            if nullmask is not None:
                keep = ~nullmask; gid = gid[keep]; codes = codes[keep]
            if G * Vc <= 64_000_000:                           # dense presence matrix: O(N) scatter, no sort
                Mx = np.zeros((G, Vc), dtype=bool); Mx[gid, codes] = True
                counts = Mx.sum(axis=1)
            else:                                              # huge key space: unique (group,code) pairs
                key = gid.astype(np.int64) * Vc + codes.astype(np.int64)
                counts = np.bincount(np.unique(key) // Vc, minlength=G)
            aggcache[ck] = counts
        return int(aggcache[ck][gi])
    rows=[]
    for gi,cv in enumerate(uc):
        keyidx = decombo(cv); rowout=[]; ki=0
        for p,kind in agg_specs:
            _inn = p.this if isinstance(p, E.Alias) else p
            if isinstance(_inn, E.Count) and isinstance(_inn.this, E.Distinct):   # COUNT(DISTINCT col) per group
                _dx = _inn.this.expressions
                if len(_dx) != 1 or not isinstance(_dx[0], E.Column):
                    raise NotImplementedError("COUNT(DISTINCT) over expression/multiple columns")
                rowout.append(int(groupdistinct(_dx[0].name, gi)))
            elif kind is None:
                rowout.append(_pyval(keyval(ki, keyidx[ki]))); ki+=1
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
    c = seg.cols[name]
    if c['mode'] in (4, 5, 6):
        arr = seg.values(name)                 # computed/inline/constant; overrides already applied
        if c['mode'] == 6 and c['has_null']:   # ADD COLUMN with NULL default -> all-null mask
            return arr, np.array([x is None for x in arr], dtype=bool)
        return arr, None
    codes = seg.codes(name); dv = seg._typed_dict(name)
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

def raw_dict_col(seg, name):
    """For a plain dict-coded NUMERIC column (no nulls, no overrides), return (base, codes) so the caller
    can defer/fuse the base[codes] decode instead of materialising it. base is float64 (dt2) or int64
    (dt0 int / dt3 datetime-epoch); codes index it per row. Returns None for anything that is not this
    simple case (computed/inline/constant modes, nullable, overridden, or string), where the caller must
    fall back to the full _col decode."""
    c = seg.cols[name]
    if c['mode'] in (4, 5, 6) or c['has_null']: return None
    if seg._override_vals_typed(name):          return None
    if c['dt'] == 2:
        base = np.asarray(seg._typed_dict(name), dtype=np.float64)
    elif c['dt'] in (0, 3):
        base = np.asarray(seg._typed_dict(name), dtype=np.int64)
    else:
        return None
    return base, seg.codes(name)

def _parse_temporal(litstr, unit):
    return int(np.datetime64(str(litstr).replace(' ','T')).astype(f'datetime64[{unit}]').view('int64'))

def _lit_for_col(seg, colname, lit, arr_kind):
    """Convert a sqlglot Literal (or negated numeric literal -5 -> Neg(Literal 5)) node to a
    value comparable with column `colname`."""
    if isinstance(lit, E.Neg):
        return -_lit_for_col(seg, colname, lit.this, arr_kind)
    if isinstance(lit, E.Cast):            # DATE '...' / CAST('...' AS DATE) -> unwrap to inner literal
        return _lit_for_col(seg, colname, lit.this, arr_kind)
    c = seg.cols[colname]
    if c['dt'] == 3:                       # datetime: parse string/number to int64 epoch
        return _parse_temporal(lit.this, seg.unit(colname))
    if not lit.is_string:
        return int(lit.this) if arr_kind in 'iu' else float(lit.this)
    v = lit.this.encode()
    return v

def _agg_scalar(seg, p, mask, seg_col):
    _inner = p.this if isinstance(p, E.Alias) else p
    if isinstance(_inner, E.Count) and isinstance(_inner.this, E.Distinct):   # COUNT(DISTINCT col)
        _dx = _inner.this.expressions
        if len(_dx) != 1 or not isinstance(_dx[0], E.Column):
            raise NotImplementedError("COUNT(DISTINCT) over expression/multiple columns")
        cn = seg_col(_dx[0].name); c = seg.cols[cn]
        if mask is None and c['mode'] != 4 and seg._overrides(cn) is None:
            return int(c['V'] - c['has_null'])         # no filter: distinct count == dict cardinality (O(1))
        if c['mode'] == 4:                              # positional codes aren't value-identity -> values
            arr, nm = _col(seg, cn)
            if mask is not None:
                arr = arr[mask]; nm = nm[mask] if nm is not None else None
            if nm is not None: arr = arr[~nm]
            return int(np.unique(arr).size)
        codes = seg.codes(cn)                           # value-identity: count distinct CODES (no value decode)
        if mask is not None: codes = codes[mask]
        if c['has_null']: codes = codes[codes != (c['V'] - 1)]   # COUNT(DISTINCT) ignores NULL
        return int(np.unique(codes).size)
    kind=_agg_kind(p)
    if kind[0]=='COUNT_STAR': return int(mask.sum()) if mask is not None else seg.N
    fn,cn=kind
    arr, nm = _col(seg, seg_col(cn))
    if mask is not None:
        arr = arr[mask]; nm = nm[mask] if nm is not None else None
    if nm is not None: arr = arr[~nm]            # SQL: aggregates ignore NULLs
    if kind[0]=='COUNT': return int(len(arr))    # COUNT(col) = non-null count
    if len(arr)==0: return None
    if fn in ('MIN', 'MAX'):                      # MIN/MAX keep the column's native type
        v = arr.min() if fn == 'MIN' else arr.max()
        c = seg.cols[seg_col(cn)]
        if c['dt'] == 3 and isinstance(v, (int, np.integer)):   # dict-mode datetime is int64 epoch
            v = np.int64(v).view(f"datetime64[{seg.unit(seg_col(cn))}]")
        return _pyval(v)
    arr = arr.astype(np.float64)                  # SUM/AVG are numeric
    return _pyval({'SUM': arr.sum(), 'AVG': arr.mean()}[fn])

# ---------- narrow-before-expand: cluster-key slice fast path (scalar aggregates) ----------
def _cluster_slice(seg, where_node, seg_col):
    """Intersect every top-level-AND range/eq conjunct on the cluster key into one (lo,hi) row
    slice, or None if the WHERE pins nothing on the key. Only descends And/Paren -- Or/Not are
    left for the residual evaluator (we never slice on a non-hard constraint)."""
    cm = seg.cluster_meta()
    if cm is None: return None
    key = cm['key']; los = []; his = []; consumed = set()
    def _kind(dt): return 'i' if dt in (0, 3) else 'f'
    def visit(n):
        if isinstance(n, E.Paren): return visit(n.this)
        if isinstance(n, E.And): visit(n.this); visit(n.expression); return
        if isinstance(n, (E.EQ, E.GT, E.LT, E.GTE, E.LTE)):
            col = _colname(n.this)
            if col is None or seg_col(col) != key: return
            lit = n.expression
            if not (isinstance(lit, E.Literal) or (isinstance(lit, E.Neg) and isinstance(lit.this, E.Literal))):
                return
            op = {E.EQ: '=', E.GT: '>', E.LT: '<', E.GTE: '>=', E.LTE: '<='}[type(n)]
            v = _lit_for_col(seg, key, lit, _kind(seg.cols[key]['dt']))
            b = seg.slice_for_predicate(key, op, v)
            if b is not None: los.append(b[0]); his.append(b[1]); consumed.add(id(n))
        elif isinstance(n, E.Between):
            col = _colname(n.this)
            if col is None or seg_col(col) != key: return
            k = _kind(seg.cols[key]['dt'])
            lov = _lit_for_col(seg, key, n.args['low'], k); hiv = _lit_for_col(seg, key, n.args['high'], k)
            b1 = seg.slice_for_predicate(key, '>=', lov); b2 = seg.slice_for_predicate(key, '<=', hiv)
            if b1 is not None and b2 is not None: los.append(b1[0]); his.append(b2[1]); consumed.add(id(n))
    visit(where_node)
    if not los: return None
    lo = max(los); hi = min(his)
    return (lo, hi, consumed) if lo < hi else (0, 0, consumed)

def _eval_pred_range(seg, node, seg_col, lo, hi, consumed):
    """Evaluate a WHERE predicate over rows [lo,hi) only, returning bool[hi-lo]. Nodes in
    `consumed` (the exact key conjuncts the slice was built from) are all-true on the slice by
    construction, so they short-circuit to ones WITHOUT decoding the key. Same operator set as
    _eval_pred otherwise; raises NotImplementedError on nulls/overrides/unsupported -> caller
    falls back to the full-column path (never a wrong answer)."""
    import operator
    # None == "all-true on [lo,hi), no residual" (node fully consumed by the slice). Propagating None
    # instead of an ones() array lets the pure-key case skip the per-query mask alloc + scan entirely.
    if isinstance(node, E.And):
        a = _eval_pred_range(seg, node.this, seg_col, lo, hi, consumed)
        b = _eval_pred_range(seg, node.expression, seg_col, lo, hi, consumed)
        return b if a is None else (a if b is None else a & b)
    if isinstance(node, E.Or):
        a = _eval_pred_range(seg, node.this, seg_col, lo, hi, consumed)
        b = _eval_pred_range(seg, node.expression, seg_col, lo, hi, consumed)
        return None if (a is None or b is None) else (a | b)            # True OR x = True
    if isinstance(node, E.Not):
        a = _eval_pred_range(seg, node.this, seg_col, lo, hi, consumed)
        return np.zeros(hi - lo, dtype=bool) if a is None else ~a       # NOT(all-true) = all-false
    if isinstance(node, E.Paren): return _eval_pred_range(seg, node.this, seg_col, lo, hi, consumed)
    if id(node) in consumed: return None   # key conjunct already satisfied by the slice: no residual
    def _slice(cn):
        if seg.cols[cn]['has_null'] or seg._overrides(cn) is not None:
            raise NotImplementedError("null/override column in slice predicate")
        return seg.values_range(cn, lo, hi)
    if isinstance(node, (E.EQ, E.NEQ, E.GT, E.LT, E.GTE, E.LTE)):
        cn = seg_col(_colname(node.this)); lit = node.expression
        if not (isinstance(lit, E.Literal) or (isinstance(lit, E.Neg) and isinstance(lit.this, E.Literal))):
            raise NotImplementedError("non-literal RHS")
        a = _slice(cn); v = _lit_for_col(seg, cn, lit, a.dtype.kind)
        if a.dtype.kind not in 'iuf' and isinstance(v, int) and seg.cols[cn]['dt'] != 3: v = str(v).encode()
        if a.dtype.kind == 'M': a = a.view('int64')
        op = {E.EQ: operator.eq, E.NEQ: operator.ne, E.GT: operator.gt, E.LT: operator.lt, E.GTE: operator.ge, E.LTE: operator.le}[type(node)]
        return op(a, v)
    if isinstance(node, E.Between):
        cn = seg_col(_colname(node.this)); a = _slice(cn)
        lov = _lit_for_col(seg, cn, node.args['low'], a.dtype.kind); hiv = _lit_for_col(seg, cn, node.args['high'], a.dtype.kind)
        if a.dtype.kind == 'M': a = a.view('int64')
        return (a >= lov) & (a <= hiv)
    if isinstance(node, E.In):
        cn = seg_col(_colname(node.this)); a = _slice(cn); lits = node.args.get('expressions') or []
        vals = []
        for L in lits:
            if not isinstance(L, E.Literal): raise NotImplementedError("IN non-literal")
            if seg.cols[cn]['dt'] == 3: vals.append(_parse_temporal(L.this, seg.unit(cn)))
            elif L.is_string: vals.append(L.this.encode() if a.dtype.kind not in 'iuf' else L.this)
            else: vals.append(int(L.this) if a.dtype.kind in 'iu' else (float(L.this) if a.dtype.kind == 'f' else str(L.this).encode()))
        if a.dtype.kind == 'M': a = a.view('int64'); return np.isin(a, np.array(vals, dtype=a.dtype))
        if a.dtype.kind in 'iuf': return np.isin(a, np.array(vals, dtype=a.dtype))
        sv = set(vals); return np.fromiter((x in sv for x in a), dtype=bool, count=len(a))
    raise NotImplementedError(f"slice predicate {type(node).__name__}")

def _agg_scalar_range(seg, p, lo, hi, rmask, seg_col):
    """Scalar aggregate over the residual-masked cluster slice (no whole-column decode).
    Non-null columns only; raises NotImplementedError otherwise (caller falls back)."""
    _inner = p.this if isinstance(p, E.Alias) else p
    if isinstance(_inner, E.Count) and isinstance(_inner.this, E.Distinct):
        _dx = _inner.this.expressions
        if len(_dx) != 1 or not isinstance(_dx[0], E.Column): raise NotImplementedError
        cn = seg_col(_dx[0].name); c = seg.cols[cn]
        if c['has_null'] or seg._overrides(cn) is not None: raise NotImplementedError
        if c['mode'] == 4:                                  # positional codes -> count distinct values
            v = seg.values_range(cn, lo, hi)
            return int(np.unique(v if rmask is None else v[rmask]).size)
        rc = seg._raw_codes_range(cn, lo, hi)
        return int(np.unique(rc if rmask is None else rc[rmask]).size)   # value-identity codes
    kind = _agg_kind(p)
    if kind is None: raise NotImplementedError("bare column in aggregate query")
    if kind[0] == 'COUNT_STAR': return int(hi - lo) if rmask is None else int(rmask.sum())
    fn, cn_name = kind; cn = seg_col(cn_name)
    if seg.cols[cn]['has_null'] or seg._overrides(cn) is not None: raise NotImplementedError
    a = _slice_vals(seg, cn, lo, hi, rmask)
    if fn == 'COUNT': return int(len(a))
    if len(a) == 0: return None
    if fn in ('MIN', 'MAX'):
        v = a.min() if fn == 'MIN' else a.max()
        if seg.cols[cn]['dt'] == 3 and isinstance(v, (int, np.integer)):
            v = np.int64(v).view(f"datetime64[{seg.unit(cn)}]")
        return _pyval(v)
    return _pyval({'SUM': a.sum(dtype=np.float64), 'AVG': a.mean(dtype=np.float64)}[fn])

def _seq_eq_mask(seg, name, neg, lit):
    """O(1)-compute equality mask for a clean-affine integer mode-4 column (n_exc==0, no
    overrides). Returns bool[N] for '= X' (or '!= X' if neg), or None to fall back to the
    general path. Exact: a position p with base+stride*p == X exists iff (X-base) is divisible
    by stride and 0<=p<N (stride!=0); for stride==0 the column is constant=base so all or no
    rows match. Python-int arithmetic avoids int64 overflow. Verified vs brute force."""
    c = seg.cols[name]
    if c['mode'] != 4 or c['dt'] != 0:
        return None
    import wdb_seqcodec
    base, stride, n, n_exc = wdb_seqcodec.header(c['seqblob'])
    if n_exc != 0 or seg._overrides(name) is not None:
        return None                                   # not clean-affine -> general path
    v = _lit_for_col(seg, name, lit, 'i')
    if not isinstance(v, (int, np.integer)):
        return None
    v = int(v); N = seg.N
    mask = np.zeros(N, dtype=bool)
    if stride == 0:
        if v == base: mask[:] = True
    else:
        diff = v - base
        if diff % stride == 0:
            p = diff // stride
            if 0 <= p < N: mask[p] = True
    return ~mask if neg else mask

def _eval_pred(seg, node, seg_col):
    if isinstance(node, E.And): return _eval_pred(seg,node.this,seg_col) & _eval_pred(seg,node.expression,seg_col)
    if isinstance(node, E.Or):  return _eval_pred(seg,node.this,seg_col) | _eval_pred(seg,node.expression,seg_col)
    if isinstance(node, E.Not): return ~_eval_pred(seg,node.this,seg_col)
    if isinstance(node, E.Paren): return _eval_pred(seg,node.this,seg_col)
    if isinstance(node, (E.EQ,E.NEQ,E.GT,E.LT,E.GTE,E.LTE)):
        col=_colname(node.this); cn=seg_col(col); lit=node.expression
        if not (isinstance(lit,E.Literal) or (isinstance(lit,E.Neg) and isinstance(lit.this,E.Literal))):
            raise NotImplementedError("non-literal RHS")
        if isinstance(node, (E.EQ, E.NEQ)):
            fm = _seq_eq_mask(seg, cn, isinstance(node, E.NEQ), lit)   # O(1) clean-affine eq
            if fm is not None: return fm
        a, nmask = _col(seg, cn)
        v = _lit_for_col(seg, cn, lit, a.dtype.kind)
        if a.dtype.kind not in 'iuf' and isinstance(v,int) and seg.cols[cn]['dt']!=3: v=str(v).encode()
        import operator
        op={E.EQ:operator.eq,E.NEQ:operator.ne,E.GT:operator.gt,E.LT:operator.lt,E.GTE:operator.ge,E.LTE:operator.le}[type(node)]
        if a.dtype.kind == 'M': a = a.view('int64')   # datetime: compare epoch ints
        res = op(a,v)
        if nmask is not None: res = res & ~nmask   # SQL: NULL fails any comparison
        return res
    if isinstance(node, E.Between):
        col=_colname(node.this); a, nmask = _col(seg, seg_col(col))
        lo=_lit_for_col(seg, seg_col(col), node.args['low'], a.dtype.kind)
        hi=_lit_for_col(seg, seg_col(col), node.args['high'], a.dtype.kind)
        if a.dtype.kind == 'M': a = a.view('int64')   # datetime: compare epoch ints
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


def _eval_expr(seg, node, col_map=None):
    """Evaluate a scalar UPDATE RHS expression to a full-column array (length seg.N),
    reading override-aware values. Supports column references, numeric/string literals,
    + - * / %, unary minus, and parentheses. numpy semantics are chosen to match DuckDB
    for these operators (notably '/' is true division in both), so the cold-segment path
    and the DuckDB-evaluated parquet path agree. NULL-in-arithmetic is a deferred edge;
    pure column-copy (SET a=b) is null-safe via object arrays. col_map maps a logical column
    name to its physical name in the segment (identity when absent)."""
    def rc(nm): return (col_map or {}).get(nm, nm)
    t = type(node)
    if t is E.Paren:
        return _eval_expr(seg, node.this, col_map)
    if t is E.Column:
        return seg.values(rc(node.name))
    if t is E.Neg:
        return -_eval_expr(seg, node.this, col_map)
    if t is E.Literal:
        if node.args.get('is_string'):
            return np.full(seg.N, node.this, dtype=object)
        s = node.this
        return np.full(seg.N, float(s) if ('.' in s or 'e' in s.lower()) else int(s))
    if t is E.Add: return _eval_expr(seg, node.left, col_map) + _eval_expr(seg, node.right, col_map)
    if t is E.Sub: return _eval_expr(seg, node.left, col_map) - _eval_expr(seg, node.right, col_map)
    if t is E.Mul: return _eval_expr(seg, node.left, col_map) * _eval_expr(seg, node.right, col_map)
    if t is E.Div: return _eval_expr(seg, node.left, col_map) / _eval_expr(seg, node.right, col_map)
    if t is E.Mod: return _eval_expr(seg, node.left, col_map) % _eval_expr(seg, node.right, col_map)
    raise NotImplementedError(f"unsupported UPDATE expression: {node.sql()}")
