#!/usr/bin/env python3
"""WaveDB SQL executor — STEP 1: single-table SELECT with WHERE, GROUP BY, HAVING,
ORDER BY, LIMIT, returning the FULL correct result set. Parses real SQL via sqlglot;
executes against a WVDB3 Segment. Correctness first; optimization second.
Unsupported shapes raise NotImplementedError (honest failure, never silent wrong answer)."""
import re
import sqlglot, sqlglot.expressions as E, numpy as np
from wdb_engine import Segment
import wdb_measure_runtime as RT

_SLICE_HITS = 0   # count of queries answered via the cluster-slice fast path (tests/telemetry)
# decoded-column residency budget lives in wdb_measure_runtime: RT.column_fits_resident(n_rows).

def _kind(dt):
    """dtype code -> numpy kind char: 'i' for int/datetime (0,3), else 'f'. Shared (also wdb_bsi_exec)."""
    return 'i' if dt in (0, 3) else 'f'


def _flatten_and(node):
    """Flatten a top-level AND (descending through Paren) into a list of leaf nodes.
    Shared stateless helper (also wdb_compound, wdb_gdsidecar)."""
    if isinstance(node, E.Paren): return _flatten_and(node.this)
    if isinstance(node, E.And): return _flatten_and(node.this) + _flatten_and(node.expression)
    return [node]


def _slice_vals(seg, cn, lo, hi, rmask):
    """Measure values for the cluster slice [lo,hi) under residual rmask. Decode-once-resident when
    the column fits the RAM budget (serving throughput: no per-query dict gather); else lazy partial
    decode. Skips the boolean-index copy when rmask selects the whole slice (the consumed-key case)."""
    base = seg.resident_values(cn) if RT.column_fits_resident(seg.N) else None
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
_DATECOUNT_HITS = 0     # count of date-coarsening COUNT(*) answered via the V->G per-code rollup


def _has_count_star_only(proj):
    """True if exactly one projection is COUNT(*) and the rest are non-aggregate group keys.
    Gates the date-coarsening rollup fast path (which only knows how to produce counts)."""
    naggs = 0
    for p in proj:
        ak = _agg_kind(p)
        if ak is None:
            continue
        if ak[0] != 'COUNT_STAR':
            return False
        naggs += 1
    return naggs == 1

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
                    rowout.append(_pyval(_sum_avg(a, fn)))
        rows.append(tuple(rowout))
    return rows, [_alias(p) for p in proj]

def _colname(node):
    if isinstance(node, E.Column): return node.name
    return None


# ---- scalar functions evaluated over a column's DICTIONARY (value-identity per code) ----------
# f(value) that depends only on the value is a property of the dict entry, not the row: compute it
# over the V distinct values once -> fval_by_code[V], then per-row value is fval_by_code[codes].
# Same shape as the date-coarsening group key. Registry maps SQL func name -> (typed_dict -> int64[V]).

def _fn_length(td, params=None):
    """CHARACTER length of each dict value (matches DuckDB length()); None -> 0 (slot unused, the
    null mask handles real nulls). Bytes decoded as utf-8 for true char count (ascii: == byte len)."""
    out = np.empty(len(td), dtype=np.int64)
    for i, v in enumerate(td):
        if v is None: out[i] = 0
        elif isinstance(v, (bytes, bytearray)): out[i] = len(v.decode('utf-8', 'replace'))
        else: out[i] = len(str(v))
    return out

def _fn_regexp_replace(td, params):
    """REGEXP_REPLACE over each dict value. params=(pattern, replacement, modifiers). Matches DuckDB:
    replace first match (all if 'g' modifier) with the replacement (\\1 = capture group 1); a string
    with no match is returned unchanged; None -> None. Returns an object array of str."""
    pattern, repl, mods = params
    rx = re.compile(pattern)
    count = 0 if (mods and 'g' in mods) else 1               # DuckDB: 'g' = global else first match
    out = np.empty(len(td), dtype=object)
    for i, v in enumerate(td):
        if v is None: out[i] = None; continue
        sv = v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else str(v)
        out[i] = rx.sub(repl, sv, count=count)
    return out


def _factorize_obj(vals):
    """Factorize an object array (str/None) WITHOUT sorting -> (unique_obj_array, codes int64).
    First-seen order (final ORDER BY decides output); tolerates None mixed with strings (np.unique
    would raise comparing None to str)."""
    seen = {}; out = np.empty(len(vals), np.int64); uniq = []
    for i, v in enumerate(vals):
        j = seen.get(v, -1)
        if j < 0: j = len(uniq); seen[v] = j; uniq.append(v)
        out[i] = j
    u = np.empty(len(uniq), dtype=object)
    for i, v in enumerate(uniq): u[i] = v
    return u, out


def _fn_lower(td, params=None):
    """LOWER of each dict value in char space (utf-8 decode -> .lower()); the introductions
    happen once per distinct value, never per row."""
    out = np.empty(len(td), dtype=object)
    for i, v in enumerate(td):
        if v is None: out[i] = None
        elif isinstance(v, (bytes, bytearray)): out[i] = v.decode('utf-8', 'replace').lower()
        else: out[i] = str(v).lower()
    return out


def _fn_upper(td, params=None):
    out = np.empty(len(td), dtype=object)
    for i, v in enumerate(td):
        if v is None: out[i] = None
        elif isinstance(v, (bytes, bytearray)): out[i] = v.decode('utf-8', 'replace').upper()
        else: out[i] = str(v).upper()
    return out


_SCALAR_FNS = { 'LENGTH': _fn_length, 'REGEXP_REPLACE': _fn_regexp_replace,
                'LOWER': _fn_lower, 'UPPER': _fn_upper }

def _scalar_fn(node):
    """Classify a scalar fn over a single column -> ('sfn', fname, colname, params) or None.
    params is None for parameterless fns (LENGTH) or a hashable tuple (REGEXP_REPLACE)."""
    g = node.this if isinstance(node, E.Alias) else node
    if isinstance(g, E.RegexpReplace):                       # REGEXP_REPLACE(col, pattern, repl[, mods])
        col = g.this; pat = g.args.get('expression'); repl = g.args.get('replacement')
        if isinstance(col, E.Column) and isinstance(pat, E.Literal) and isinstance(repl, E.Literal):
            mods = g.args.get('modifiers')
            modv = mods.this if isinstance(mods, E.Literal) else None
            return ('sfn', 'REGEXP_REPLACE', col.name, (pat.this, repl.this, modv))
        return None
    fname = None
    if isinstance(g, E.Length): fname = 'LENGTH'
    elif isinstance(g, E.Lower): fname = 'LOWER'
    elif isinstance(g, E.Upper): fname = 'UPPER'
    elif isinstance(g, (E.Anonymous, E.Func)):
        nm = (getattr(g, 'name', '') or '').upper()
        if nm in _SCALAR_FNS: fname = nm
    if fname is None: return None
    inner = g.this if not isinstance(g, E.Anonymous) else (g.expressions[0] if g.expressions else None)
    if isinstance(inner, E.Column): return ('sfn', fname, inner.name, None)
    return None

_SFN_CACHE = {}   # (id(seg), col, fname) -> fval_by_code int64[V]

def _fval_by_code(seg, fname, cn, params=None):
    key = (id(seg), cn, fname, params)
    m = _SFN_CACHE.get(key)
    if m is None:
        m = _SCALAR_FNS[fname](seg._typed_dict(cn), params)
        _SFN_CACHE[key] = m
    return m

def _ck(cn):
    """Hashable stable key for an aggregate column ref (plain name or ('sfn',fname,col) tuple)."""
    return cn if not isinstance(cn, tuple) else tuple(cn)


def _factorize_nonneg(x, domain=None):
    """Like np.unique(x, return_inverse=True) but for non-negative ints with a known/derivable
    small domain. Returns (present_sorted_values, inverse_indices). Avoids the argsort-based
    return_inverse path (measured ~27x faster on 100M codes): count present via bincount, build a
    dense remap, gather. `domain` = upper bound (exclusive) on values; derived from max+1 if None."""
    x = np.ascontiguousarray(x)
    if x.size == 0:
        return np.empty(0, x.dtype), np.empty(0, np.int64)
    hi = int(domain) if domain is not None else int(x.max()) + 1
    if hi <= 0:
        return np.unique(x, return_inverse=True)
    present = np.nonzero(np.bincount(x, minlength=hi))[0]    # sorted distinct present values
    remap = np.empty(hi, np.int64); remap[present] = np.arange(len(present))
    inv = remap[x]
    return present, inv


def _sfn_array(seg, fname, cn, mask=None, params=None):
    """Derived per-row array fval_by_code[codes] + null mask (null code -> null). Applies mask if
    given. One implementation shared by the group path and the aggregate path."""
    fv = _fval_by_code(seg, fname, cn, params)
    codes = seg.codes(cn)
    c = seg.cols[cn]
    nmask = None
    if c.get('has_null'):
        nmask = (codes == (c['V'] - 1))
    arr = fv[codes]
    if mask is not None:
        arr = arr[mask]; nmask = nmask[mask] if nmask is not None else None
    return arr, nmask


_EXTRACT_UNITS = {'YEAR', 'MONTH', 'QUARTER', 'DAY', 'HOUR', 'MINUTE'}
_TRUNC_UNITS = {'YEAR', 'QUARTER', 'MONTH', 'WEEK', 'DAY', 'HOUR', 'MINUTE'}
_NP_UNIT = {'YEAR': 'Y', 'MONTH': 'M', 'WEEK': 'W', 'DAY': 'D', 'HOUR': 'h', 'MINUTE': 'm'}

def _date_unit(td, unit, dt_unit='D'):
    """Map a date/timestamp column's typed dictionary to an integer group value per code -- the
    coarsening, evaluated over the V dict values, never over N rows. `dt_unit` is the column's stored
    datetime64 unit (from seg.unit); the dict ints are interpreted in THAT unit, preserving sub-day
    resolution for timestamps. `unit` is either an EXTRACT field ('YEAR'..'MINUTE') or 'TRUNC:<U>'
    for DATE_TRUNC. EXTRACT returns the field's integer; TRUNC returns the truncated value in the
    column's own unit (epoch ticks), matching DuckDB's truncated timestamp. Returns int array over
    codes 0..V-1."""
    raw = np.asarray(td)
    base = f'datetime64[{dt_unit}]'
    d = raw.astype(np.int64).view(base) if raw.dtype.kind in ('i', 'u') else raw.astype(base)
    if unit.startswith('TRUNC:'):                          # DATE_TRUNC -> truncated value in column unit
        u = _NP_UNIT[unit.split(':', 1)[1]]
        trunc = d.astype(f'datetime64[{u}]').astype(base)  # floor to U, back to column unit
        return trunc.view(np.int64)                        # epoch ticks (column unit) -- DuckDB's timestamp
    if unit == 'YEAR':    return d.astype('datetime64[Y]').astype(int) + 1970
    if unit == 'QUARTER': return (d.astype('datetime64[M]').astype(int) % 12) // 3 + 1
    if unit == 'MONTH':   return d.astype('datetime64[M]').astype(int) % 12 + 1
    if unit == 'DAY':     return (d.astype('datetime64[D]') - d.astype('datetime64[M]')).astype('timedelta64[D]').astype(int) + 1
    if unit == 'HOUR':    return d.astype('datetime64[h]').astype(int) % 24
    if unit == 'MINUTE':  return d.astype('datetime64[m]').astype(int) % 60
    raise NotImplementedError(f"EXTRACT unit {unit!r}")

_CODECOUNT_CACHE = {}   # (seg.path, col, N) -> full per-code count vector (length V), incl singletons
_DATEMAP_CACHE = {}     # (seg.path, col, unit, N) -> (group keys, per-code inverse) for the V->G rollup

def _code_counts(seg, col):
    """Full per-code count vector aligned to codes 0..V-1 (singletons included), cached per
    (segment, column, N). For a date column V is tiny, so this is small and built once. This is the
    materialized base a date-coarsening COUNT(*) rolls up over -- V->G, no per-row pass."""
    ck = (seg.path, col, int(seg.N))
    hit = _CODECOUNT_CACHE.get(ck)
    if hit is not None:
        return hit
    codes = seg._raw_codes(col)
    K = int(seg.cols[col].get('V') or (int(codes.max()) + 1 if codes.size else 1))
    cc = np.bincount(codes, minlength=K).astype(np.int64)
    _CODECOUNT_CACHE[ck] = cc
    return cc

def _trunc_to_dt(tick, dt_unit):
    """A DATE_TRUNC group value is an epoch tick in the column's unit; render it as a Python
    datetime so the output matches DuckDB's truncated-timestamp type (not a raw int)."""
    import datetime as _dt
    return np.datetime64(int(tick), dt_unit).astype('datetime64[us]').astype(_dt.datetime)

def _is_trunc(unit):
    return isinstance(unit, str) and unit.startswith('TRUNC:')

def _date_count_rollup(seg, col, unit):
    """COUNT(*) grouped by EXTRACT(unit FROM col), answered as a V->G rollup over per-code counts:
    map each code to its group value, sum the per-code counts into those groups. O(V), no N pass.
    The code->group mapping (group keys + inverse) is deterministic per (col,unit) and cached, so a
    warm call is just the final bincount. Returns {group_value: count}. For DATE_TRUNC the key is
    rendered as a datetime (matching DuckDB); for EXTRACT it's the field integer."""
    mk = (seg.path, col, unit, int(seg.N))
    m = _DATEMAP_CACHE.get(mk)
    if m is None:
        td = seg._typed_dict(col)
        gid_of_code = _date_unit(td, unit, seg.unit(col))   # length-V group value per code
        keys, inv = np.unique(gid_of_code, return_inverse=True)
        m = (keys, inv); _DATEMAP_CACHE[mk] = m
    keys, inv = m
    cc = _code_counts(seg, col)                             # length-V counts (cached)
    g = np.bincount(inv, weights=cc, minlength=len(keys)).astype(np.int64)
    if _is_trunc(unit):
        du = seg.unit(col)
        return {_trunc_to_dt(k, du): int(v) for k, v in zip(keys.tolist(), g.tolist())}
    return dict(zip(keys.tolist(), g.tolist()))

def _literal_value(g):
    """Python value of a sqlglot Literal (string -> str, else int then float)."""
    if g.is_string: return g.this
    sv = g.this
    try: return int(sv)
    except (ValueError, TypeError): return float(sv)


def _eval_scalar(node, env):
    """Evaluate a scalar arithmetic expression against a {column_name: value} env. Supports
    column refs, numeric literals, +, -, * and unary minus / parens. Raises (KeyError/TypeError)
    for anything else -- the caller treats that as 'not evaluable from base columns' and falls back."""
    n = node.this if isinstance(node, E.Alias) else node
    if isinstance(n, E.Column):  return env[n.name]            # KeyError if not a base column -> fallback
    if isinstance(n, E.Literal): return _literal_value(n)
    if isinstance(n, E.Paren):   return _eval_scalar(n.this, env)
    if isinstance(n, E.Neg):     return -_eval_scalar(n.this, env)
    if isinstance(n, E.Add):     return _eval_scalar(n.this, env) + _eval_scalar(n.expression, env)
    if isinstance(n, E.Sub):     return _eval_scalar(n.this, env) - _eval_scalar(n.expression, env)
    if isinstance(n, E.Mul):     return _eval_scalar(n.this, env) * _eval_scalar(n.expression, env)
    raise TypeError("not a scalar arithmetic expression")


def _eval_scalar_safe(node, env):
    """(_eval_scalar(node,env), True) or (None, False) if it isn't a base-column arithmetic expr."""
    try: return _eval_scalar(node, env), True
    except (KeyError, TypeError, ValueError): return None, False


def _to_rows(v, nrows):
    """Broadcast a scalar to an object array of length nrows; pass arrays through unchanged."""
    if isinstance(v, np.ndarray): return v
    a = np.empty(nrows, dtype=object); a[:] = v; return a


def _eval_rows(seg, node, mask, resolve=None):
    """Vectorised PER-ROW evaluation of a scalar expression over the (masked) rows -> numpy array of
    length = #selected rows. Used for genuine derived group keys that aren't a simple function of one
    column (e.g. CASE WHEN ...), which must be evaluated per row then factorized. String values are
    produced as bytes (consistent with the engine; _pyval decodes them to str at emit). `resolve` maps
    a logical column name to its physical segment name (col_map); identity if None. Raises TypeError
    for unsupported nodes (caller keeps the prior 'unsupported' behavior)."""
    import operator
    n = node.this if isinstance(node, E.Alias) else node
    if isinstance(n, E.Paren): return _eval_rows(seg, n.this, mask, resolve)
    if isinstance(n, E.Column):
        c = resolve(n.name) if resolve is not None else n.name; col = seg.cols[c]
        if col['dt'] == 1:                                   # string dict col: decode distinct survivors
            rc = seg.codes(c); rc = rc[mask] if mask is not None else rc
            uc, inv = np.unique(rc, return_inverse=True)
            vals = np.empty(len(uc), dtype=object)
            for i, code in enumerate(uc):
                v = seg.fetch(c, int(code))
                vals[i] = v if isinstance(v, (bytes, bytearray)) else (b'' if v is None else str(v).encode())
            return vals[inv]
        v = np.asarray(seg.values(c)); return v[mask] if mask is not None else v
    if isinstance(n, E.Literal):
        return n.this.encode() if n.is_string else _literal_value(n)
    if isinstance(n, E.Null): return None
    if isinstance(n, E.Not): return ~_eval_rows(seg, n.this, mask, resolve)
    if isinstance(n, E.And): return _eval_rows(seg, n.this, mask, resolve) & _eval_rows(seg, n.expression, mask, resolve)
    if isinstance(n, E.Or):  return _eval_rows(seg, n.this, mask, resolve) | _eval_rows(seg, n.expression, mask, resolve)
    if isinstance(n, (E.EQ, E.NEQ, E.GT, E.LT, E.GTE, E.LTE)):
        l = _eval_rows(seg, n.this, mask, resolve); r = _eval_rows(seg, n.expression, mask, resolve)
        op = {E.EQ:operator.eq, E.NEQ:operator.ne, E.GT:operator.gt,
              E.LT:operator.lt, E.GTE:operator.ge, E.LTE:operator.le}[type(n)]
        return op(l, r)
    if isinstance(n, E.Neg): return -_eval_rows(seg, n.this, mask, resolve)
    if isinstance(n, E.Add): return _eval_rows(seg, n.this, mask, resolve) + _eval_rows(seg, n.expression, mask, resolve)
    if isinstance(n, E.Sub): return _eval_rows(seg, n.this, mask, resolve) - _eval_rows(seg, n.expression, mask, resolve)
    if isinstance(n, E.Mul): return _eval_rows(seg, n.this, mask, resolve) * _eval_rows(seg, n.expression, mask, resolve)
    if isinstance(n, E.Case):
        nrows = int(np.count_nonzero(mask)) if mask is not None else seg.N
        default = n.args.get('default')
        acc = _to_rows(_eval_rows(seg, default, mask, resolve), nrows) if default is not None else np.full(nrows, None, object)
        for iff in reversed(n.args.get('ifs') or []):        # last WHEN wins if listed first -> reverse-fold
            cond = _eval_rows(seg, iff.this, mask, resolve)
            then = _to_rows(_eval_rows(seg, iff.args.get('true'), mask, resolve), nrows)
            acc = np.where(cond, then, acc)
        return acc
    raise TypeError(f"_eval_rows: unsupported node {type(n).__name__}")


def _affine_key(g):
    """Detect a group key that is an INJECTIVE function of a single column -- col +/- const,
    const - col, col * nonzero-const, unary minus, and compositions. Injective means grouping by
    the base column makes the identical piles as grouping by the expression (one-to-one), so the
    expression need not be its own partition dimension: group by the base column, compute the
    expression per pile at emit. Returns ('affine', basecol, canonical_sql) or None.
    NOT injective (e.g. length(col), col % k, integer col / k) -> not matched here."""
    cols = set(); ok = [True]
    def walk(n):
        if isinstance(n, E.Column): cols.add(n.name); return
        if isinstance(n, E.Literal): return
        if isinstance(n, E.Paren): walk(n.this); return
        if isinstance(n, E.Neg):   walk(n.this); return
        if isinstance(n, (E.Add, E.Sub)): walk(n.this); walk(n.expression); return
        if isinstance(n, E.Mul):
            l, r = n.this, n.expression
            lit = r if isinstance(r, E.Literal) else (l if isinstance(l, E.Literal) else None)
            if lit is None: ok[0] = False; return            # col*col is not single-col / not injective
            try:
                if float(_literal_value(lit)) == 0.0: ok[0] = False   # *0 collapses -> not injective
            except (ValueError, TypeError): ok[0] = False
            walk(l if lit is r else r); return               # walk the non-literal (column) side
        ok[0] = False                                        # any other op: not provably injective
    walk(g)
    if ok[0] and len(cols) == 1:
        return ('affine', next(iter(cols)), g.sql())
    return None


def _group_key(node, proj=None, _resolve_pos=True, node_sink=None):
    """Classify a GROUP BY expression. Returns:
      ('col', name)        -- a bare value-identity column (today's path, unchanged)
      ('fn', col, unit)    -- EXTRACT(unit FROM col): a date coarsening, grouped in code space
      ('sfn', col, fname)  -- scalar function of a column, e.g. length(col)
      ('const', value)     -- a constant literal key (GROUP BY 1 resolves here when SELECT 1 ...)
    Raises NotImplementedError for anything else, preserving the prior 'unknown column' behavior.
    `proj` (the SELECT list) lets a bare GROUP BY name resolve to a projection alias -- e.g.
    SELECT EXTRACT(year FROM d) AS y ... GROUP BY y, where sqlglot parses `y` as a Column."""
    g = node.this if isinstance(node, E.Alias) else node
    # Positional GROUP BY: an integer literal refers to the Nth SELECT item (1-based). SQL-standard.
    # Resolve once (_resolve_pos guard) so a resolved integer literal (SELECT 1) becomes a const key
    # rather than recursing as another position.
    if _resolve_pos and proj and isinstance(g, E.Literal) and g.is_int:
        pos = int(g.this)
        if 1 <= pos <= len(proj):
            return _group_key(proj[pos-1], proj, _resolve_pos=False)
    nm = _colname(g)
    if nm is not None and proj is not None:
        for p in proj:                                  # bare name matching a SELECT alias -> its expr
            if isinstance(p, E.Alias) and p.alias == nm:
                g = p.this; nm = _colname(g); break
    if nm is not None:
        return ('col', nm)
    if isinstance(g, E.Extract):
        unit = g.this.name.upper() if hasattr(g.this, 'name') else str(g.this).upper()
        col = g.args.get('expression')
        if unit in _EXTRACT_UNITS and isinstance(col, E.Column):
            return ('fn', col.name, unit)
    if isinstance(g, (E.TimestampTrunc, E.DateTrunc)):      # DATE_TRUNC(unit, col) / TIMESTAMP_TRUNC
        u = g.args.get('unit')
        unit = (u.name if hasattr(u, 'name') else str(u)).upper().strip("'\"")
        col = g.this
        if unit in _TRUNC_UNITS and isinstance(col, E.Column):
            return ('fn', col.name, 'TRUNC:' + unit)
    sf = _scalar_fn(g)                                       # length/regexp_replace(col) -> sfn group key
    if sf is not None:
        return ('sfn', sf[2], sf[1], sf[3])                 # ('sfn', colname, fname, params)
    af = _affine_key(g)                                      # col +/- const etc. -> injective single-col
    if af is not None:
        return af                                           # ('affine', basecol, canonical_sql)
    if isinstance(g, E.Literal):                            # constant key (e.g. GROUP BY 1 -> SELECT 1)
        return ('const', _literal_value(g))
    if isinstance(g, E.Case):                               # CASE WHEN ...: genuine derived key,
        if node_sink is not None: node_sink[g.sql()] = g    # evaluated per row then factorized
        return ('rowexpr', g.sql())
    raise NotImplementedError(f"unsupported GROUP BY key: {g.sql()!r}")


def _proj_colname(p):
    """Column name of a projection/group expr, unwrapping a top-level AS alias first.
    The 'unwrap alias then name it' idiom, centralized (was inlined across 6 operators)."""
    return _colname(p.this if isinstance(p, E.Alias) else p)

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
    rowexpr_nodes = {}
    gkeys_raw = [ _group_key(g, proj, node_sink=rowexpr_nodes) for g in group.expressions ] if group else []
    # Effective grouping dimensions: an affine single-column key (ClientIP-1) reduces to its base
    # column (ClientIP) -- injective, so same piles -- and base columns are DEDUPED. This collapses
    # GROUP BY ClientIP, ClientIP-1, ClientIP-2, ClientIP-3 to grouping by ClientIP ONCE (no radix
    # explosion); the minus-variants are functions of the pile's ClientIP, computed at emit. Genuine
    # derived keys (sfn/date/const) stay their own dimension (deduped by identity).
    gkeys = []; _eff_seen = {}
    for _k in gkeys_raw:
        _e = ('col', _k[1]) if _k[0] == 'affine' else _k
        _sig = _e[1] if _e[0] == 'col' else _e
        if _sig in _eff_seen: continue
        _eff_seen[_sig] = len(gkeys); gkeys.append(_e)
    gcols = [ (None if k[0] in ('const','rowexpr') else k[1]) for k in gkeys ]   # None for const/rowexpr (no single base col)
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
            order = tree.args.get('order'); lim = _limit(tree); off = _offset(tree)
            distinct = tree.args.get('distinct') is not None
            early = order is None and not distinct      # window in scan order before materializing
            if early and (lim is not None or off):
                idx = idx[off: off + lim] if lim is not None else idx[off:]
            out = []
            colvals = {c: seg.values(c) for c in cols}
            for i in idx: out.append(tuple(_pyval(colvals[c][i]) for c in cols))
            if distinct:                                           # SELECT DISTINCT -> dedup (order-preserving)
                seen = set(); ded = []
                for r in out:
                    if r not in seen: seen.add(r); ded.append(r)
                out = ded
            if order is not None: out = _apply_order(out, proj, order)   # ORDER BY: sort, then LIMIT/OFFSET
            if not early and (lim is not None or off):
                out = out[off: off + lim] if lim is not None else out[off:]
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
    gnames = [(seg_col(g) if g is not None else None) for g in gcols]
    # --- date-coarsening COUNT(*) fast path: single computed key, count-only, filter-free ---
    # The answer is a V->G rollup over per-code counts (O(V)) instead of an O(N) remap+unique.
    if (len(gkeys) == 1 and gkeys[0][0] == 'fn' and where is None
            and mask is None and tree.args.get('having') is None
            and len(proj) == 2 and _has_count_star_only(proj)):
        gv = _date_count_rollup(seg, gnames[0], gkeys[0][2])
        ci = 0 if _agg_kind(proj[0]) is not None else 1   # which projection is COUNT(*)
        rows = []
        for k, c in gv.items():
            row = [None, None]; row[ci] = int(c); row[1 - ci] = k   # k already typed (int field or datetime)
            rows.append(tuple(row))
        rows = _apply_order(rows, proj, tree.args.get('order'))
        lim = _limit(tree); off = _offset(tree)
        if lim is not None or off:
            rows = rows[off: off + lim] if lim is not None else rows[off:]
        global _DATECOUNT_HITS; _DATECOUNT_HITS += 1
        return rows, [_alias(p) for p in proj]
    combo = None; combo_dom = None; metas=[]   # metas[i] = (gn, kind, table): 'val'->unique values, 'code'->unique codes, 'computed'->(unit, table of typed group values)
    for gn, gk in zip(gnames, gkeys):
        if gk[0] == 'fn':
            # date coarsening: evaluate the unit over the column's DICTIONARY (V values, not N rows),
            # giving a per-code group-id; map row codes through it. No per-row function eval.
            unit = gk[2]
            td = seg._typed_dict(gn)                            # sorted dict values (datetime64 day-numbers)
            gid_of_code = _date_unit(td, unit, seg.unit(gn))   # code -> integer group value, in the column's unit
            gc = seg.codes(gn); gc = gc[mask] if mask is not None else gc
            row_gv = gid_of_code[gc]                            # per-row group value
            u, inv = np.unique(row_gv, return_inverse=True)    # distinct group values + per-row slot
            metas.append((gn, 'computed', u, unit))
        elif gk[0] == 'const':
            # constant key (GROUP BY <int> -> SELECT <literal>): one distinct value, contributes
            # nothing to the grouping (domain 1). inv is all-zeros over the post-mask row count.
            n_rows = len(combo) if combo is not None else (int(np.count_nonzero(mask)) if mask is not None else seg.N)
            u = np.array([gk[1]], dtype=object); inv = np.zeros(n_rows, dtype=np.int64)
            metas.append((None, 'computed', u, None))
        elif gk[0] == 'rowexpr':
            # general per-row expression (e.g. CASE WHEN ...): evaluate over the masked rows, then
            # factorize the resulting values. Not injective / multi-column, so it is its own dimension.
            row_gv = _eval_rows(seg, rowexpr_nodes[gk[1]], mask, seg_col)
            u, inv = np.unique(row_gv, return_inverse=True)
            metas.append((None, 'computed', u, None))
        elif gk[0] == 'sfn':
            # scalar fn over a column (length(col), regexp_replace(col,...)): evaluate over the
            # DICTIONARY (V values), then group. int output -> gather + factorize_nonneg; string output
            # (REGEXP_REPLACE) -> factorize the per-CODE results (V, cheap) and map row codes through it.
            params = gk[3] if len(gk) > 3 else None
            fv = _fval_by_code(seg, gk[2], gn, params)
            if fv.dtype == object:                             # string-valued fn
                codes = seg.codes(gn); codes = codes[mask] if mask is not None else codes
                u, code2g = _factorize_obj(fv)                 # per-code result -> group id (V values)
                inv = code2g[codes]                            # per-row group id (gather)
            else:                                              # int-valued fn (length)
                row_gv, _ = _sfn_array(seg, gk[2], gn, mask, params)
                u, inv = _factorize_nonneg(row_gv)
            metas.append((gn, 'computed', u, None))
        elif seg.cols[gn]['mode'] == 4:
            vals = seg.values(gn); vals = vals[mask] if mask is not None else vals
            u, inv = np.unique(vals, return_inverse=True)      # value-identity keys
            metas.append((gn, 'val', u, None))
        else:
            gc = seg.codes(gn); gc = gc[mask] if mask is not None else gc
            u, inv = _factorize_nonneg(gc)                     # codes dense; domain=max+1 (handles override codes >= V)
            metas.append((gn, 'code', u, None))
        combo = inv if combo is None else combo*len(u)+inv
        combo_dom = len(u) if combo_dom is None else combo_dom*len(u)
    # uc/counts: when the combined key domain is modest, bincount over it (no 100M sort). The sort-based
    # np.unique(return_counts) on 100M combos was ~2.9s; bincount is ~0.1s and also gives dense gids.
    if combo_dom is not None and combo_dom <= 64_000_000:
        cc = np.bincount(combo, minlength=combo_dom)
        uc = np.nonzero(cc)[0]                              # present combined keys (sorted)
        counts = cc[uc]
        _combo_dense = True                                # combo values index directly into a dense space
    else:
        uc, counts = np.unique(combo, return_counts=True)
        _combo_dense = False
    def decombo(cv):
        out=[]; x=cv
        for m in reversed(metas): u=m[2]; out.append(int(x%len(u))); x//=len(u)
        return list(reversed(out))   # per-key INDEX into that key's table
    def keyval(ki, idx):
        gn, kk, u, unit = metas[ki]
        if kk == 'computed':
            v = u[idx]
            return _trunc_to_dt(v, seg.unit(gn)) if _is_trunc(unit) else v
        if kk == 'val': return u[idx]                          # typed value directly
        code = u[idx]; c = seg.cols[gn]
        if c['has_null'] and int(code) == c['V']-1: return None
        return seg.fetch(gn, int(code))
    agg_specs = [(p, _agg_kind(p)) for p in proj]

    # ---- bincount scatter-aggregation fast path (SUM/COUNT/AVG only) ----
    # The argsort-based per-group slicing below sorts all N rows (~8s+ at 100M). When every aggregate
    # is decomposable (COUNT(*), COUNT(col), SUM, AVG) we don't need rows grouped contiguously: scatter
    # each row's value into its group bucket with np.bincount over a dense group id. O(N), no sort.
    # MIN/MAX and COUNT(DISTINCT) still need the argsort/slice path, so we gate them out.
    def _decomposable(kind):
        if kind is None: return True                          # group key, not an agg
        if kind[0] in ('COUNT_STAR', 'COUNT', 'SUM', 'AVG'): return True
        return False
    _has_distinct = any(
        isinstance((p.this if isinstance(p, E.Alias) else p), E.Count)
        and isinstance((p.this if isinstance(p, E.Alias) else p).this, E.Distinct)
        for p, _ in agg_specs)
    _fast = (not _has_distinct) and all(_decomposable(k) for _, k in agg_specs)

    fast_count = None; fast_sum = {}
    if _fast:
        G = len(uc)
        if _combo_dense:
            # combo already indexes the dense combo_dom space; map combo-value -> 0..G-1 via a remap
            # table (gather), avoiding the ~2s searchsorted(uc, combo) over 100M.
            remap = np.empty(combo_dom, np.int64); remap[uc] = np.arange(G)
            gid = remap[combo]
            fast_count = counts.astype(np.int64)              # counts already = cc[uc] from bincount
        else:
            gid = np.searchsorted(uc, combo)                  # dense group id per row, 0..G-1
            fast_count = np.bincount(gid, minlength=G).astype(np.int64)
        for _p, kind in agg_specs:
            if kind is None or kind[0] in ('COUNT_STAR',): continue
            fn, cn = kind
            fkey = (fn, cn if not isinstance(cn, tuple) else cn)
            if fn == 'COUNT' and cn is None: continue
            # build the value array (+ null mask) for cn, then bincount-scatter into groups
            if isinstance(cn, tuple) and cn[0] == 'sfn':
                arr, nm = _sfn_array(seg, cn[1], seg_col(cn[2]), mask, cn[3] if len(cn) > 3 else None)
            else:
                arr, nm = _col(seg, seg_col(cn))
                if mask is not None:
                    arr = arr[mask]; nm = nm[mask] if nm is not None else None
            if fn in ('SUM', 'AVG'):
                is_int = np.issubdtype(arr.dtype, np.integer)
                if nm is not None:                            # NULLs excluded from sum and from the AVG denominator
                    valid = ~nm; gv = gid[valid]; av = arr[valid]
                    cnts = np.bincount(gv, minlength=G).astype(np.int64)
                else:
                    gv = gid; av = arr; cnts = fast_count
                if is_int:                                    # exact integer sum (no float64 mantissa loss)
                    sums = np.zeros(G, np.int64); np.add.at(sums, gv, av.astype(np.int64))
                else:
                    sums = np.bincount(gv, weights=av.astype(np.float64), minlength=G)
                fast_sum[(fn, _ck(cn))] = (sums, cnts, is_int)
            elif fn == 'COUNT':                               # COUNT(col) = non-null count
                cnts = np.bincount(gid[~nm], minlength=G).astype(np.int64) if nm is not None else fast_count
                fast_sum[(fn, _ck(cn))] = (None, cnts, False)

    if _fast:
        order = None; gstarts = gends = None                  # argsort path not needed on fast path
    else:
        order = np.argsort(combo, kind='stable'); combo_s=combo[order]
        gstarts = np.searchsorted(combo_s, uc); gends = np.r_[gstarts[1:], len(combo_s)]
    aggcache={}
    def groupagg(colname, fn, gi):
        is_sfn = isinstance(colname, tuple) and colname[0] == 'sfn'
        ckey = colname if not is_sfn else colname            # tuple is hashable -> fine as cache key
        if ckey not in aggcache:
            if is_sfn:                                        # AVG/SUM/... over length(col) etc.
                arr, nm = _sfn_array(seg, colname[1], seg_col(colname[2]), mask, colname[3] if len(colname) > 3 else None)
            else:
                pcol = seg_col(colname)
                arr, nm = _col(seg, pcol)          # cache NATIVE dtype; float cast only for SUM/AVG
                if mask is not None:
                    arr = arr[mask]; nm = nm[mask] if nm is not None else None
            aggcache[ckey]=(arr[order], (nm[order] if nm is not None else None))
        colname = ckey
        vs, vn = aggcache[colname]; sl=slice(gstarts[gi],gends[gi]); seg_v=vs[sl]
        if vn is not None: seg_v = seg_v[~vn[sl]]      # SQL: aggregates ignore NULLs
        if fn=='COUNT': return len(seg_v)
        if len(seg_v)==0: return None
        if fn in ('MIN', 'MAX'):                       # native min/max (datetime/string included)
            v = seg_v.min() if fn == 'MIN' else seg_v.max()
            if not is_sfn and seg.cols[seg_col(colname)]['dt'] == 3 and isinstance(v, (int, np.integer)):
                v = np.int64(v).view(f"datetime64[{seg.unit(seg_col(colname))}]")
            return v
        return _sum_avg(seg_v, fn)
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
    # --- COUNT(*)-only top-K fast select: when the query is ORDER BY <count> DESC LIMIT k with no
    # HAVING, only k of the (possibly millions of) groups survive. Pick them with argpartition
    # (O(G), no full sort) and build rows for those alone, instead of materialising every group
    # then sorting. Pure structural gate -- counts come from the same `counts` vector either way. ---
    gi_list = range(len(uc))
    _topk = _count_topk_plan(proj, agg_specs, tree)
    if _topk is not None and len(uc) > _topk:
        k = _topk
        part = np.argpartition(counts, -k)[-k:]
        gi_list = part[np.argsort(counts[part])[::-1]].tolist()   # k groups, count-desc
    # match each non-agg projection to its group key by identity (SELECT order may != GROUP BY order)
    _gpos = {}
    for _i, _k in enumerate(gkeys): _gpos.setdefault(_k, _i)
    def _gki_of(pp):
        try: kk = _group_key(pp, proj, _resolve_pos=False)
        except NotImplementedError: return None
        return _gpos.get(kk)
    rows=[]
    for gi in gi_list:
        cv = uc[gi]
        keyidx = decombo(cv); rowout=[]; ki=0
        # base-column values for this pile -> evaluate any arithmetic/bare-column projection directly
        env = {}
        for _mi, (_gn, _kk, _u, _un) in enumerate(metas):
            if _gn is not None and _kk in ('code', 'val'):
                env[_gn] = keyval(_mi, keyidx[_mi])
        for p,kind in agg_specs:
            _inn = p.this if isinstance(p, E.Alias) else p
            if isinstance(_inn, E.Count) and isinstance(_inn.this, E.Distinct):   # COUNT(DISTINCT col) per group
                _dx = _inn.this.expressions
                if len(_dx) != 1 or not isinstance(_dx[0], E.Column):
                    raise NotImplementedError("COUNT(DISTINCT) over expression/multiple columns")
                rowout.append(int(groupdistinct(_dx[0].name, gi)))
            elif kind is None:
                _inn2 = p.this if isinstance(p, E.Alias) else p
                val, ok = _eval_scalar_safe(_inn2, env)         # bare col / arithmetic of base columns
                if not ok:                                      # genuine derived key (sfn/date/const)
                    gk_i = _gki_of(p)
                    if gk_i is None: gk_i = ki                  # defensive positional fallback
                    val = keyval(gk_i, keyidx[gk_i])
                rowout.append(_pyval(val)); ki+=1
            elif kind[0]=='COUNT_STAR':
                rowout.append(int(counts[gi]))
            elif _fast:
                fn,cn=kind
                if fn=='COUNT' and cn is None:
                    rowout.append(int(fast_count[gi]))
                elif fn=='COUNT':
                    rowout.append(int(fast_sum[(fn,_ck(cn))][1][gi]))
                else:
                    sums,cnts,is_int=fast_sum[(fn,_ck(cn))]
                    c=int(cnts[gi])
                    if c==0: rowout.append(None)
                    elif fn=='SUM': rowout.append(int(sums[gi]) if is_int else _pyval(sums[gi]))
                    else: rowout.append(_pyval(float(sums[gi])/c))   # AVG always float
            else:
                fn,cn=kind; rowout.append(_pyval(groupagg(cn, fn, gi)))
        rows.append(tuple(rowout))

    # ---- HAVING ----
    having = tree.args.get('having')
    if having is not None:
        rows = _apply_having(rows, proj, having.this, seg_col)

    # ---- ORDER BY ----
    rows = _apply_order(rows, proj, tree.args.get('order'))

    # ---- LIMIT / OFFSET ----
    lim = _limit(tree); off = _offset(tree)
    if lim is not None or off:
        rows = rows[off: off + lim] if lim is not None else rows[off:]
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
        if isinstance(inner, cls):
            cn = _colname(inner.this)
            if cn is not None: return (nm, cn)
            sf = _scalar_fn(inner.this)               # FN(length(col)) -> carry the sfn spec
            if sf is not None: return (nm, sf)         # (FN, ('sfn', fname, colname))
            return (nm, None)
    return None  # not an aggregate -> group key
def _count_topk_plan(proj, agg_specs, tree):
    """Return k (the LIMIT) when this grouped query is a COUNT(*)-only top-K by that count:
    every aggregate is COUNT(*), there is an ORDER BY whose sole key is that count (or its alias)
    DESC, a LIMIT is present, and no OFFSET / HAVING. Otherwise None. Lets the group path build
    only the k surviving groups (argpartition) instead of every group. Structural -- no data."""
    if tree.args.get('having') is not None or _offset(tree): return None
    lim = _limit(tree)
    if lim is None: return None
    aggs = [k for _, k in agg_specs if k is not None]
    if not aggs or any(k[0] != 'COUNT_STAR' for k in aggs): return None
    order = tree.args.get('order')
    if order is None: return None
    oexprs = order.expressions
    if len(oexprs) != 1: return None
    o = oexprs[0]
    if not bool(o.args.get('desc')): return None            # must be DESC (largest counts)
    key = o.this
    # the ORDER BY key must reference the COUNT(*) -- either COUNT(*) inline or the count's alias
    if isinstance(key, E.Count): return int(lim)
    if isinstance(key, E.Column):
        nm = key.name
        for p in proj:
            if isinstance(p, E.Alias) and p.alias == nm and isinstance(p.this, E.Count):
                return int(lim)
    return None
def _exact_int_sum(arr):
    """Exact Python-int sum of an integer numpy array -- no int64 overflow and no float64 mantissa
    loss (DuckDB sums integers exactly; float64 loses precision past 2^53, e.g. AVG(UserID)).
    Non-negative arrays use a fast hi/lo 32-bit split summed in uint64; signed arrays fall back to
    object accumulation."""
    if arr.size == 0: return 0
    if int(arr.min()) >= 0:
        u = arr.astype(np.uint64, copy=False)
        lo = int(np.bitwise_and(u, np.uint64(0xFFFFFFFF)).sum(dtype=np.uint64))   # each <2^32; sum fits uint64
        hi = int(np.right_shift(u, np.uint64(32)).sum(dtype=np.uint64))
        return (hi << 32) + lo
    return int(arr.sum(dtype=object))


def _sum_avg(arr, fn):
    """SUM or AVG over `arr` (already null-filtered, len>0). Integer inputs: exact integer sum ->
    SUM returns a Python int (matching DuckDB's integer SUM), AVG returns exact_sum / n as a float.
    Float inputs use float64."""
    if np.issubdtype(arr.dtype, np.integer):
        si = _exact_int_sum(arr)
        return si if fn == 'SUM' else (si / len(arr))
    af = arr.astype(np.float64)
    return float(af.sum()) if fn == 'SUM' else float(af.mean())


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
def _offset(tree):
    """OFFSET row count (0 when absent). LIMIT n OFFSET k returns rows [k : k+n]."""
    o = tree.args.get('offset')
    if o is None: return 0
    try: return int(o.expression.this)
    except Exception: return 0
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
        if arr_kind in 'iu':
            try:
                return int(lit.this)
            except ValueError:
                return float(lit.this)   # float literal vs int column: numpy compares fine
        return float(lit.this)
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
    if isinstance(cn, tuple) and cn[0] == 'sfn':              # FN(length(col)) etc., no GROUP BY
        arr, nm = _sfn_array(seg, cn[1], seg_col(cn[2]), mask, cn[3] if len(cn) > 3 else None)
        if nm is not None: arr = arr[~nm]
        if fn == 'COUNT': return int(len(arr))
        if len(arr) == 0: return None
        if fn in ('MIN', 'MAX'): return _pyval(arr.min() if fn == 'MIN' else arr.max())
        return _pyval(_sum_avg(arr, fn))
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
    return _pyval(_sum_avg(arr, fn))

# ---------- narrow-before-expand: cluster-key slice fast path (scalar aggregates) ----------
def _cluster_slice(seg, where_node, seg_col):
    """Intersect every top-level-AND range/eq conjunct on the cluster key into one (lo,hi) row
    slice, or None if the WHERE pins nothing on the key. Only descends And/Paren -- Or/Not are
    left for the residual evaluator (we never slice on a non-hard constraint)."""
    cm = seg.cluster_meta()
    if cm is None: return None
    key = cm['key']; los = []; his = []; consumed = set()
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
    return _pyval(_sum_avg(a, fn))

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

def _dict_eq_mask(seg, name, neg, lit):
    """Code-space equality for a dictionary-coded column: resolve the literal to its dictionary
    code, then compare the CODE array (an integer scan) instead of decoding N values. Pure
    structural logic -- works for any dict column, no dataset knowledge. Returns bool[N] for
    '= X' (or '!= X' if neg), or None to fall back to the value-decode path.

    Correctness: codes index directly into the typed dict (seg.fetch maps code->dict[code]), so
    rows equal to X are exactly the rows whose code == the code(s) of X. A literal absent from the
    dict matches no rows (EQ -> all False, NEQ -> all live). NULL never satisfies = or <> (SQL),
    so the null code is excluded from a positive match and, for NEQ, also excluded."""
    c = seg.cols[name]
    if c['mode'] == 4:                      # affine/value-identity: _seq_eq_mask owns this
        return None
    if seg._overrides(name) is not None:    # override values aren't in the base dict -> value path
        return None
    # PRESENCE-AS-PREDICATE (the sparse dress's dividend): on an enc-8 column, a
    # literal that IS the default value answers from the presence bitmap alone --
    # 12.5MB of bits instead of reconstructing the dense column. sp <> '' becomes
    # the bitmap verbatim; sp = '' its complement. Non-default literals fall through
    # to the code scan as ever.
    if c.get('code_enc') == 8 and not c.get('has_null'):
        import wdb_wherescan as _WS
        dcode = _WS._code_of(seg, name, lit)
        if dcode is None:                       # literal absent from the dictionary
            return np.ones(int(seg.N), bool) if neg else np.zeros(int(seg.N), bool)
        if int(dcode) == int(c['e8d']):
            ck9 = '_e8up_' + name
            up = seg._codes.get(ck9)
            if up is None:
                pb = np.asarray(seg.buf[c['e8pres']:c['e8pres'] + (int(seg.N) + 7) // 8],
                                dtype=np.uint8)
                up = np.unpackbits(pb, count=int(seg.N)).astype(bool)
                seg._codes[ck9] = up
            return up if neg else ~up
        pl = seg.e8_planes(name)                # non-default literal: mark its rows
        if pl is not None:                      # from the planes -- never densify
            ck8 = '_e8m_%s_%d' % (name, int(dcode))
            m8 = seg._codes.get(ck8)            # query-lifetime cache: detect and
            if m8 is None:                      # execute phases share one build
                pos8, lits8, _d8 = pl
                m8 = np.zeros(int(seg.N), bool)
                m8[pos8[lits8 == np.uint32(int(dcode))]] = True
                seg._codes[ck8] = m8
            return ~m8 if neg else m8
    try:
        codes = seg.codes(name)
    except Exception:
        return None
    if codes is None or codes.dtype.kind not in 'iu':
        return None
    td = seg._typed_dict(name)
    td = td if isinstance(td, np.ndarray) else np.asarray(td, dtype=object)
    V = c['V']
    nullcode = (V - 1) if c['has_null'] else None
    kind = 'i' if c['dt'] == 0 else ('f' if c['dt'] == 2 else ('i' if c['dt'] == 3 else 'S'))
    v = _lit_for_col(seg, name, lit, kind)
    if c['dt'] not in (0, 2, 3) and isinstance(v, int):
        v = str(v).encode()
    try:
        match = (td == v)
    except Exception:
        return None
    mcodes = np.nonzero(match)[0]
    if nullcode is not None:
        mcodes = mcodes[mcodes != nullcode]
    if len(mcodes) == 0:                    # literal absent: EQ matches nothing, NEQ matches all live
        res = np.zeros(len(codes), dtype=bool)
    elif len(mcodes) == 1:
        res = (codes == mcodes[0])
    else:
        res = np.isin(codes, mcodes)
    if neg:
        res = ~res
        if nullcode is not None:           # SQL: NULL <> X is not TRUE -> exclude nulls
            res &= (codes != nullcode)
    return res


def _eval_pred(seg, node, seg_col):
    if isinstance(node, E.And): return _eval_pred(seg,node.this,seg_col) & _eval_pred(seg,node.expression,seg_col)
    if isinstance(node, E.Or):  return _eval_pred(seg,node.this,seg_col) | _eval_pred(seg,node.expression,seg_col)
    if isinstance(node, E.Not): return ~_eval_pred(seg,node.this,seg_col)
    if isinstance(node, E.Paren): return _eval_pred(seg,node.this,seg_col)
    if isinstance(node, (E.EQ,E.NEQ,E.GT,E.LT,E.GTE,E.LTE)):
        col=_colname(node.this)
        if col is None:                        # LHS isn't a bare column: try scalar-expression
            import wdb_wherescan as _WS
            sc = _WS._scalar_cmp(seg, node, None)
            if sc is not None:
                sp, op, lit2 = sc
                fl = _WS._scalar_flag(seg, sp, op, lit2)
                scol0 = seg_col(sp['col'])
                pl = seg.e8_planes(scol0) if hasattr(seg, 'e8_planes') else None
                if pl is not None:              # paint the flag onto the planes: the
                    pos8, lits8, d8 = pl        # function ran once per DISTINCT value;
                    out = np.full(int(seg.N), bool(fl[d8]))   # rows never densify
                    out[pos8] = fl[lits8]
                    return out
                return fl[np.asarray(seg._raw_codes(scol0))]   # native width
            raise NotImplementedError(f"predicate LHS {type(node.this).__name__}")
        cn=seg_col(col); lit=node.expression
        if not (isinstance(lit,E.Literal) or (isinstance(lit,E.Neg) and isinstance(lit.this,E.Literal))):
            raise NotImplementedError("non-literal RHS")
        if isinstance(node, (E.EQ, E.NEQ)):
            fm = _seq_eq_mask(seg, cn, isinstance(node, E.NEQ), lit)   # O(1) clean-affine eq
            if fm is not None: return fm
            dm = _dict_eq_mask(seg, cn, isinstance(node, E.NEQ), lit)  # code-space eq (no value decode)
            if dm is not None: return dm
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
        col=_colname(node.this)
        kcs = node.args.get('_codes')
        if kcs is not None:
            # same-column subquery pre-resolved to a CODE SET upstream: membership in code
            # space is a V-sized flag and one NATIVE-WIDTH gather, not a sort (np.isin +
            # int64 cast were 390ms of sq-nested; NULL never matches: not in the set)
            scol8 = seg_col(col)
            pl = seg.e8_planes(scol8) if hasattr(seg, 'e8_planes') else None
            if pl is not None:
                pos8, lits8, d8 = pl
                V0 = int(seg.cols[scol8]['V'])
                fl = np.zeros(V0, bool)
                fl[np.asarray(kcs, dtype=np.int64)] = True
                out = np.full(int(seg.N), bool(fl[d8]))
                out[pos8] = fl[lits8]           # membership painted onto the planes
                return out
            arr = np.asarray(seg._raw_codes(scol8))
            V0 = int(seg.cols[scol8].get('V') or int(arr.max()) + 1)
            fl = np.zeros(V0, bool)
            fl[np.asarray(kcs, dtype=np.int64)] = True
            return fl[arr]
        if node.args.get('query') is not None:
            raise NotImplementedError("IN with unresolved subquery")
        lits=node.args.get('expressions') or []
        c0 = seg.cols.get(seg_col(col))
        if (c0 is not None and c0.get('mode') in (0, 1, 2) and len(lits) > 64
                and all(isinstance(L, E.Literal) for L in lits)):
            # big literal lists on dict columns: bind values to CODES once, isin over raw
            # codes -- never decode the column (a 4k-value NOT IN was minutes of string decode)
            import wdb_wherescan as _WS
            vals0 = [(L.this if L.is_string else str(L.this)) for L in lits]
            tcs = _WS._in_codes(seg, seg_col(col), vals0)
            scol9 = seg_col(col)
            pl = seg.e8_planes(scol9) if hasattr(seg, 'e8_planes') else None
            if pl is not None:
                pos8, lits8, d8 = pl
                V0 = int(seg.cols[scol9]['V'])
                fl = np.zeros(V0, bool)
                tcs2 = np.asarray(tcs, dtype=np.int64)
                fl[tcs2[(tcs2 >= 0) & (tcs2 < V0)]] = True
                out = np.full(int(seg.N), bool(fl[d8]))
                out[pos8] = fl[lits8]
                return out
            arr = np.asarray(seg._raw_codes(scol9))
            V0 = int(seg.cols[scol9].get('V') or int(arr.max()) + 1)
            fl = np.zeros(V0, bool)
            tcs2 = np.asarray(tcs, dtype=np.int64)
            fl[tcs2[(tcs2 >= 0) & (tcs2 < V0)]] = True
            return fl[arr]                       # flag + native gather, not a 100M sort
        a, nmask = _col(seg, seg_col(col))
        vals=[]
        for L in lits:
            neg = isinstance(L, E.Neg) and isinstance(L.this, E.Literal)   # IN (-1, 6): -1 parses as Neg(Literal)
            if neg: L = L.this
            if not isinstance(L,E.Literal): raise NotImplementedError("IN with non-literal / subquery")
            if seg.cols[seg_col(col)]['dt']==3: vals.append(_parse_temporal(L.this, seg.unit(seg_col(col))))
            elif L.is_string: vals.append(L.this.encode() if a.dtype.kind not in 'iuf' else L.this)
            else:
                v = int(L.this) if a.dtype.kind in 'iu' else (float(L.this) if a.dtype.kind=='f' else str(L.this).encode())
                vals.append(-v if (neg and a.dtype.kind in 'iuf') else v)
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
        if node.args.get('negate'):      # sqlglot: NOT LIKE == Like(negate=True), not Not(Like)
            m = ~m
        if nmask is not None: m = m & ~nmask
        return m
    if isinstance(node, E.Is):
        col=_colname(node.this); a, nmask = _col(seg, seg_col(col))
        if isinstance(node.expression, E.Null):
            return nmask if nmask is not None else np.zeros(len(a), dtype=bool)  # IS NULL
        raise NotImplementedError("IS <non-null-literal>")
    if isinstance(node, E.Boolean):
        return np.full(seg.N, bool(node.this), dtype=bool)   # TRUE/FALSE literal predicates
    raise NotImplementedError(f"predicate {type(node).__name__}")

def _apply_having(rows, proj, node, seg_col):
    # Resolve each HAVING leaf's column index ONCE (was: inner.sql()==expr.sql() per row, which
    # re-serialised the AST ~ncols times per group -- O(groups*cols) sqlglot .sql() calls).
    import operator
    proj_inner = [(p.this if isinstance(p, E.Alias) else p) for p in proj]
    proj_sql = [pi.sql() for pi in proj_inner]                 # serialise each projection ONCE
    def col_index(expr):
        es = expr.sql()
        for i, ps in enumerate(proj_sql):
            if ps == es: return i
        raise NotImplementedError("HAVING references non-projected expr")
    OPS = {E.GT:operator.gt,E.LT:operator.lt,E.GTE:operator.ge,E.LTE:operator.le,E.EQ:operator.eq,E.NEQ:operator.ne}
    # compile the predicate tree to a closure over precomputed indices (no per-row .sql())
    def compile_node(n):
        if isinstance(n, E.And):
            a, b = compile_node(n.this), compile_node(n.expression); return lambda r: a(r) and b(r)
        if isinstance(n, E.Or):
            a, b = compile_node(n.this), compile_node(n.expression); return lambda r: a(r) or b(r)
        if isinstance(n, (E.GT,E.LT,E.GTE,E.LTE,E.EQ,E.NEQ)):
            idx = col_index(n.this); rhs = float(n.expression.this); op = OPS[type(n)]
            return lambda r: op(r[idx], rhs)
        raise NotImplementedError("HAVING op")
    pred = compile_node(node)
    return [r for r in rows if pred(r)]

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
