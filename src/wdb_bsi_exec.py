"""Phase 3: BSI filter-aggregate executor.

Fires for single-table scalar/grouped SUM/COUNT aggregates with a WHERE over
non-null, value-sorted, dict-coded numeric/datetime columns. Builds a packed
predicate bitmap via the bit-sliced index (wdb_bsi), combines predicates with
free bitwise AND/OR/NOT, and reduces the measure over set bits with the numba
bitmap-walk kernels (wdb_bsi_kernels) -- no row scan of the predicate columns,
no position materialisation. Any unsupported shape raises _BSIUnsupported and
the caller falls back to the fused path with identical results. Self-gated:
only indexes columns it can prove safe, and bails to the fused scan when the
predicate is too unselective for the bitmap walk to pay (SEL_CEIL).
"""
import numpy as np
import sqlglot.expressions as E
import wdb_sql as S
import wdb_bsi as B
import wdb_bsi_kernels as K
import wdb_measure_runtime as RT

_BSI_HITS = 0
# Path gates (selectivity ceiling, index RAM budget) live in wdb_measure_runtime:
# RT.bsi_too_unselective(cnt, N) and RT.bsi_index_fits(current_bytes, add_bytes).


class _BSIUnsupported(Exception):
    pass


try:
    K.warmup()
except Exception:
    pass


def _kind(dt):
    return 'i' if dt in (0, 3) else 'f'


def _bump():
    global _BSI_HITS
    _BSI_HITS += 1


def _col_index(seg, col):
    """Lazy per-segment BSI for a predicate column, cached on seg._bsi (with the
    running byte total on seg._bsi_bytes). Workload-driven by construction: only
    columns an actual query filters get built. Raises _BSIUnsupported for anything
    not safe to index (computed/inline/constant mode, nullable, string, overridden)
    or once the per-segment index RAM budget is exhausted -- both degrade to the
    fused scan with identical results."""
    cache = getattr(seg, '_bsi', None)
    if cache is None:
        cache = {}; seg._bsi = cache; seg._bsi_bytes = 0
    hit = cache.get(col)
    if hit is not None:
        return hit
    c = seg.cols[col]
    if c['mode'] in (4, 5, 6) or c['has_null'] or c['dt'] == 1:
        raise _BSIUnsupported("col not BSI-indexable")
    if seg._override_vals_typed(col):
        raise _BSIUnsupported("overrides")
    # prospective size = B planes * ceil(N/8) bytes; refuse if it would blow the budget
    plane_bytes = -(-seg.N // 8)
    nplanes = max(1, max(0, c['V'] - 1).bit_length())
    if not RT.bsi_index_fits(seg._bsi_bytes, nplanes * plane_bytes):
        raise _BSIUnsupported("index RAM budget exhausted")
    codes = seg.codes(col)
    dv = np.asarray(seg._typed_dict(col))   # value-sorted (np.unique) => code order == value order
    bsi = B.build_bsi(codes, seg.N)
    seg._bsi_bytes += bsi.nbytes()
    hit = (bsi, dv)
    cache[col] = hit
    return hit


def _is_lit(n):
    return isinstance(n, (E.Literal, E.Neg, E.Cast))


def _cmp(bsi, dv, ntype, v):
    """Map a comparison node + value to a packed bitmap, via searchsorted on the
    value-sorted dict (value-space -> code-space)."""
    z = np.zeros_like(bsi.ones)
    if ntype is E.EQ or ntype is E.NEQ:
        i = int(np.searchsorted(dv, v, 'left'))
        m = bsi.eq(i) if (i < len(dv) and dv[i] == v) else z
        return (bsi.ones & ~m) if ntype is E.NEQ else m
    if ntype is E.GTE:
        return bsi.ge(int(np.searchsorted(dv, v, 'left')))
    if ntype is E.GT:
        return bsi.ge(int(np.searchsorted(dv, v, 'right')))
    if ntype is E.LTE:
        return bsi.range(0, int(np.searchsorted(dv, v, 'right')))
    if ntype is E.LT:
        return bsi.range(0, int(np.searchsorted(dv, v, 'left')))
    raise _BSIUnsupported("cmp")


def _pred(seg, n, sc):
    """Recursively turn a WHERE node into a packed predicate bitmap."""
    if isinstance(n, E.Paren):
        return _pred(seg, n.this, sc)
    if isinstance(n, E.And):
        return _pred(seg, n.this, sc) & _pred(seg, n.expression, sc)
    if isinstance(n, E.Or):
        return _pred(seg, n.this, sc) | _pred(seg, n.expression, sc)
    if isinstance(n, E.Not):
        return B._ones(seg.N) & ~_pred(seg, n.this, sc)
    if isinstance(n, (E.EQ, E.NEQ, E.GT, E.LT, E.GTE, E.LTE)):
        col = S._colname(n.this)
        if col is None:
            raise _BSIUnsupported("non-col lhs")
        col = sc(col); bsi, dv = _col_index(seg, col)
        if not _is_lit(n.expression):
            raise _BSIUnsupported("non-literal rhs")
        v = S._lit_for_col(seg, col, n.expression, _kind(seg.cols[col]['dt']))
        return _cmp(bsi, dv, type(n), v)
    if isinstance(n, E.Between):
        col = S._colname(n.this)
        if col is None:
            raise _BSIUnsupported("between lhs")
        col = sc(col); bsi, dv = _col_index(seg, col); k = _kind(seg.cols[col]['dt'])
        lo = S._lit_for_col(seg, col, n.args['low'], k)
        hi = S._lit_for_col(seg, col, n.args['high'], k)
        a = int(np.searchsorted(dv, lo, 'left')); b = int(np.searchsorted(dv, hi, 'right'))
        return bsi.range(a, b)
    if isinstance(n, E.In):
        if n.args.get('query') is not None:
            raise _BSIUnsupported("IN subquery")
        col = S._colname(n.this)
        if col is None:
            raise _BSIUnsupported("in lhs")
        col = sc(col); bsi, dv = _col_index(seg, col); k = _kind(seg.cols[col]['dt'])
        idxs = []
        for lit in n.expressions:
            if not _is_lit(lit):
                raise _BSIUnsupported("in non-literal")
            v = S._lit_for_col(seg, col, lit, k)
            i = int(np.searchsorted(dv, v, 'left'))
            if i < len(dv) and dv[i] == v:
                idxs.append(i)
        return bsi.in_set(idxs)
    raise _BSIUnsupported(type(n).__name__)


def _agg_spec(e, seg, sc):
    """('KEY',col) | ('COUNT_STAR',) | ('COUNT',col) | ('SUM',col) | ('SUMMUL',c1,c2)."""
    if not S._is_agg(e):
        col = S._colname(e)
        if col is None:
            raise _BSIUnsupported("non-column group expr")
        return ('KEY', sc(col))
    inner = e.this if isinstance(e, E.Alias) else e
    if isinstance(inner, E.Count):
        if inner.find(E.Distinct) is not None:
            raise _BSIUnsupported("COUNT(DISTINCT)")
        if isinstance(inner.this, E.Star) or inner.this is None:
            return ('COUNT_STAR',)
        col = S._colname(inner.this)
        if col is None:
            raise _BSIUnsupported("count expr")
        col = sc(col)
        if seg.cols[col]['has_null']:
            raise _BSIUnsupported("count nullable")
        return ('COUNT', col)
    if isinstance(inner, E.Sum):
        s = inner.this
        if isinstance(s, E.Paren):
            s = s.this
        if isinstance(s, E.Column):
            col = sc(s.name)
            if seg.cols[col]['has_null']:
                raise _BSIUnsupported("nullable measure")
            return ('SUM', col)
        if isinstance(s, E.Mul) and isinstance(s.this, E.Column) and isinstance(s.expression, E.Column):
            c1 = sc(s.this.name); c2 = sc(s.expression.name)
            if seg.cols[c1]['has_null'] or seg.cols[c2]['has_null']:
                raise _BSIUnsupported("nullable measure")
            return ('SUMMUL', c1, c2)
        raise _BSIUnsupported("SUM expr")
    raise _BSIUnsupported("agg kind")


def _scalar(seg, s, mask, N, cnt):
    if s[0] in ('COUNT_STAR', 'COUNT'):
        return int(cnt)
    if cnt == 0:
        return None                       # SQL SUM over zero rows is NULL
    if s[0] == 'SUM':
        return float(K.bw_sum1(mask, seg.resident_values(s[1]), N))
    if s[0] == 'SUMMUL':
        return float(K.bw_sum2(mask, seg.resident_values(s[1]), seg.resident_values(s[2]), N))
    raise _BSIUnsupported("scalar agg")


def _grouped_arr(seg, s, mask, gcodes, nb, N, gcount):
    if s[0] in ('COUNT_STAR', 'COUNT'):
        return gcount
    if s[0] == 'SUM':
        return K.bw_group_sum1(mask, seg.resident_values(s[1]), gcodes, nb, N)
    raise _BSIUnsupported("grouped agg")   # grouped SUM(a*b) etc -> fused


def _grp_val(seg, col, code, dv, nullcode):
    if code == nullcode:
        return None
    v = dv[code]
    if seg.cols[col]['dt'] == 3:
        v = np.int64(v).view(f"datetime64[{seg.unit(col)}]")
    return S._pyval(v)


def execute(seg, tree, col_map):
    """Run the query on the BSI path, or raise _BSIUnsupported to fall back."""
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    if tree.args.get('distinct') is not None:
        raise _BSIUnsupported("distinct")
    if tree.args.get('having') is not None or tree.args.get('qualify') is not None:
        raise _BSIUnsupported("having/qualify")
    where = tree.args.get('where')
    if where is None:
        raise _BSIUnsupported("no where")            # full-scan agg better on fused
    specs = [_agg_spec(e, seg, sc) for e in tree.expressions]
    keys = [s for s in specs if s[0] == 'KEY']
    if len(keys) > 1:
        raise _BSIUnsupported("multi group key")
    grp = tree.args.get('group')
    if (grp is not None) != (len(keys) == 1):
        raise _BSIUnsupported("group/select mismatch")
    if keys and (tree.args.get('order') is not None or S._limit(tree) is not None):
        raise _BSIUnsupported("grouped order/limit")

    header = [S._alias(e) for e in tree.expressions]
    mask = _pred(seg, where.this, sc)
    N = seg.N
    cnt = K.bw_count(mask, N)
    if RT.bsi_too_unselective(cnt, N):
        raise _BSIUnsupported("unselective -> fused")

    if not keys:
        _bump()
        return [tuple(_scalar(seg, s, mask, N, cnt) for s in specs)], header

    gkey = keys[0][1]; gc = seg.cols[gkey]; nb = gc['V']
    gcodes = seg.codes(gkey)
    gcount = K.bw_group_count(mask, gcodes, nb, N)
    arrays = {i: _grouped_arr(seg, s, mask, gcodes, nb, N, gcount)
              for i, s in enumerate(specs) if s[0] != 'KEY'}
    dv = seg._typed_dict(gkey); nullcode = (nb - 1) if gc['has_null'] else -1
    rows = []
    for code in range(nb):
        if gcount[code] == 0:
            continue
        rv = []
        for i, s in enumerate(specs):
            if s[0] == 'KEY':
                rv.append(_grp_val(seg, gkey, code, dv, nullcode))
            else:
                a = arrays[i][code]
                rv.append(int(a) if s[0] in ('COUNT_STAR', 'COUNT') else float(a))
        rows.append(tuple(rv))
    _bump()
    return rows, header


def footprint(seg):
    """(bytes, [indexed columns]) of the BSI index currently built on this segment.
    The index is additive in-memory state; this is what it contributes to RSS."""
    cache = getattr(seg, '_bsi', None)
    if not cache:
        return 0, []
    return getattr(seg, '_bsi_bytes', 0), sorted(cache.keys())
