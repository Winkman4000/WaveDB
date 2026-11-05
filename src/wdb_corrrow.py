"""wdb_corrrow: THE VALUE PER ROW (c128, TPC-H Q20) -- a WHERE comparison against a correlated scalar subquery, judged
at the outer query's survivors.

    ... WHERE ps_availqty > (SELECT 0.5 * SUM(l_quantity) FROM lineitem
                             WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey AND l_shipdate >= .. AND < ..)

The subquery reads one table and returns one aggregate (SUM / AVG / COUNT / MIN / MAX of one column, or COUNT(*),
inside + - * / with numeric literals); its WHERE is equalities 'inner column = outer column' (one or more) and
conditions on the inner table alone. Its value depends only on the outer row's values of the equalities' outer
columns (the KEY), so:
  1. the survivors' key values (integer columns, read at the survivors only); the distinct keys, each survivor's key;
  2. the inner rows: the inner table's own conditions AND, for each equality, the inner column among the survivors'
     values of its outer column (a code set) -- the same cascade every WHERE runs (wdb_join.join_rows);
  3. each inner row's key, found among the distinct keys (a sorted search); rows whose key no survivor has drop;
  4. the aggregate per key, then the subquery's arithmetic -- a key with no inner rows has no value (SQL: the
     aggregate of no rows is NULL; COUNT is 0);
  5. each survivor reads its key's value; the comparison keeps the rows where it is true (no value: not true).
Nothing reads a column name: the shape is the SQL's, the key test is the data's (integer columns, no NULLs).
Anything else returns None / raises wdb_join._FastUnsupported and the roads behind decide.
Switch WDB_CORR_ROW (default on)."""
import os
import numpy as np
from sqlglot import expressions as E

_CORR_ROW = [os.environ.get('WDB_CORR_ROW', '1') == '1']
_HITS = [0]
_AGGS = (E.Sum, E.Avg, E.Count, E.Min, E.Max)
_OPS = (E.GT, E.GTE, E.LT, E.LTE, E.EQ)
_FLIP = {E.GT: E.LT, E.LT: E.GT, E.GTE: E.LTE, E.LTE: E.GTE, E.EQ: E.EQ}

from numba import njit as _njit, prange as _prange


@_njit(parallel=True, cache=True, nogil=True)
def _probe(uk, q, ok):
    """position of each q among the sorted distinct keys uk, -1 when absent or not ok"""
    n = uk.shape[0]
    out = np.empty(q.shape[0], np.int64)
    for i in _prange(q.shape[0]):
        r = -1
        if ok[i] and n > 0:
            a = 0; b = n; v = q[i]
            while a < b:
                h = (a + b) >> 1
                if uk[h] < v: a = h + 1
                else: b = h
            if a < n and uk[a] == v:
                r = a
        out[i] = r
    return out


def _distinct(v, lo, span):
    """the distinct values of v (all in [lo, lo + span)), rising: one mark per value when the range is narrow"""
    if span <= max(4 * v.size, 1 << 16) and span <= (1 << 26):
        mk = np.zeros(span, dtype=np.bool_)
        mk[v - lo] = True
        return np.flatnonzero(mk).astype(np.int64) + lo
    return _uniq_inv(v, inverse=False)


def _conj(n):
    if n is None: return []
    if isinstance(n, E.Paren): return _conj(n.this)
    if isinstance(n, E.And): return _conj(n.this) + _conj(n.expression)
    return [n]


def _arith_ok(nd, ag):
    if nd is ag: return True
    if isinstance(nd, E.Paren): return _arith_ok(nd.this, ag)
    if isinstance(nd, E.Neg): return _arith_ok(nd.this, ag)
    if isinstance(nd, E.Literal): return not nd.is_string
    if type(nd) in (E.Add, E.Sub, E.Mul, E.Div): return _arith_ok(nd.this, ag) and _arith_ok(nd.expression, ag)
    return False


def _arith(nd, ag, vals):
    if nd is ag: return vals
    if isinstance(nd, E.Paren): return _arith(nd.this, ag, vals)
    if isinstance(nd, E.Neg): return -_arith(nd.this, ag, vals)
    if isinstance(nd, E.Literal): return float(nd.this)
    a = _arith(nd.this, ag, vals); b = _arith(nd.expression, ag, vals)
    if isinstance(nd, E.Add): return a + b
    if isinstance(nd, E.Sub): return a - b
    if isinstance(nd, E.Mul): return a * b
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.true_divide(a, b)


def spec(db, cj, own):
    """cj a WHERE conjunct, own {bare column name: outer table}. The shape (see the module), or None:
    dict(op, ocol (the outer side, a bare column), sub, t2, a2, eqs [(inner name, outer name)], rest [conjuncts],
    agg, col (None: COUNT(*)), expr)."""
    if not _CORR_ROW[0] or type(cj) not in _OPS:
        return None
    a, b = cj.this, cj.expression
    op = type(cj)
    if isinstance(a, E.Subquery) and not isinstance(b, E.Subquery):
        a, b, op = b, a, _FLIP[op]
    if not isinstance(b, E.Subquery) or not isinstance(a, E.Column) or own.get(a.name) is None:
        return None
    sub = b.this
    if not isinstance(sub, E.Select) or any(x is not sub for x in sub.find_all(E.Select)):
        return None
    for k in ('joins', 'group', 'having', 'order', 'limit', 'offset', 'distinct', 'with', 'qualify', 'laterals'):
        if sub.args.get(k):
            return None
    frm = sub.args.get('from') or sub.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table) or len(sub.expressions) != 1:
        return None
    t2 = frm.this.name; a2 = frm.this.alias_or_name
    try:
        cols2 = set(db.cat.column_names(t2))
    except Exception:
        return None
    ex = sub.expressions[0]
    ex = ex.this if isinstance(ex, E.Alias) else ex
    aggs = list(ex.find_all(*_AGGS))
    if len(aggs) != 1 or aggs[0].args.get('distinct') or not _arith_ok(ex, aggs[0]):
        return None
    ag = aggs[0]
    arg = ag.this
    if isinstance(ag, E.Count) and (arg is None or isinstance(arg, E.Star)):
        col = None
    elif isinstance(arg, E.Column) and arg.name in cols2 and arg.table in ('', a2):
        col = arg.name
    else:
        return None

    def inner(c):                                   # SQL scoping: the innermost table owns a bare name it has
        if c.table: return c.table == a2
        return c.name in cols2

    eqs, rest, otab = [], [], set()
    for c in _conj(sub.args['where'].this if sub.args.get('where') else None):
        cs = list(c.find_all(E.Column))
        if all(inner(x) for x in cs):
            rest.append(c); continue
        if not (isinstance(c, E.EQ) and isinstance(c.this, E.Column) and isinstance(c.expression, E.Column)):
            return None                             # an outer column anywhere but an equality: not this shape
        x, y = c.this, c.expression
        if inner(x) and not inner(y): ic, oc = x, y
        elif inner(y) and not inner(x): ic, oc = y, x
        else: return None
        if oc.table or own.get(oc.name) is None:
            return None                             # the outer side: a bare name of the outer tables
        eqs.append((ic.name, oc.name)); otab.add(own[oc.name])
    if not eqs or len(otab) != 1:
        return None
    return dict(op=op, ocol=a.name, sub=sub, t2=t2, a2=a2, eqs=eqs, rest=rest, agg=ag, col=col, expr=ex)


def _num_at(seg, pc, rows):
    """float64 values of a numeric, NULL-free dictionary column at rows"""
    c = seg.cols.get(pc)
    if c is None or c.get('has_null') or c.get('dt') == 1 or seg._overrides(pc) is not None:
        raise _unsup()
    td = np.asarray(seg._typed_dict(pc))
    if td.dtype.kind not in 'iuf':
        raise _unsup()
    return td[np.asarray(seg.codes_at(pc, rows))].astype(np.float64)


_ALL_SETS = [os.environ.get('WDB_CORR_ROW_SETS', 'one') == 'all']   # A/B: every equality's code set on the inner


def _uniq_inv(a, inverse=True):
    """np.unique(a, return_inverse) -- in one pass when a already never decreases (survivors in row order over a
    table stored by its key), else the sort"""
    a = np.ascontiguousarray(a)
    if a.size and bool(np.all(a[1:] >= a[:-1])):
        brk = np.empty(a.size, bool); brk[0] = True
        np.not_equal(a[1:], a[:-1], out=brk[1:])
        u = a[brk]
        return (u, np.cumsum(brk) - 1) if inverse else u
    return np.unique(a, return_inverse=True) if inverse else np.unique(a)


def _lit_value(seg, pc, nd):
    """a literal as the column's typed-dictionary value: numbers as is; a date / timestamp (DATE '...', '...' against
    a date column) in the column's own unit, only when exact"""
    dt = seg.cols[pc].get('dt')
    if isinstance(nd, E.Paren):
        return _lit_value(seg, pc, nd.this)
    if isinstance(nd, E.Neg):
        return -_lit_value(seg, pc, nd.this)
    s = None
    if isinstance(nd, E.Cast) and isinstance(nd.this, E.Literal) and nd.this.is_string:
        if not any(k in nd.to.sql().upper() for k in ('DATE', 'TIMESTAMP')):
            raise _unsup()
        s = str(nd.this.this)
    elif isinstance(nd, E.Literal):
        if not nd.is_string:
            if dt == 3: raise _unsup()
            t = str(nd.this)
            return float(t) if any(ch in t for ch in '.eE') else int(t)
        s = str(nd.this)
    else:
        raise _unsup()
    if dt != 3:
        raise _unsup()                              # a string literal against a non-date column: not here
    unit = seg.unit(pc)
    try:
        x = np.datetime64(s.strip())
    except ValueError:
        raise _unsup()
    y = x.astype('datetime64[%s]' % unit)
    if y.astype(x.dtype) != x:
        raise _unsup()                              # the literal is finer than the column's unit
    return int(y.astype(np.int64))


def _keep_left(seg, pm, alias, cj, rows):
    """the rows where a one-column conjunct (compare with a literal, BETWEEN, IN a literal list; AND / OR / NOT of
    those) holds -- judged once per dictionary value, then each row by its code"""
    import operator
    cols = list(cj.find_all(E.Column))
    if not cols or any(c.name != cols[0].name or c.table not in ('', alias) for c in cols):
        raise _unsup()
    pc = pm.get(cols[0].name, cols[0].name)
    c = seg.cols.get(pc)
    if c is None or c.get('has_null') or c.get('dt') == 1 or seg._overrides(pc) is not None:
        raise _unsup()
    td = np.asarray(seg._typed_dict(pc))
    if td.dtype.kind == 'M':
        td = td.view(np.int64)
    if td.dtype.kind not in 'iuf':
        raise _unsup()
    OPS = {E.GT: operator.gt, E.GTE: operator.ge, E.LT: operator.lt, E.LTE: operator.le, E.EQ: operator.eq,
           E.NEQ: operator.ne}
    FL = {E.GT: E.LT, E.LT: E.GT, E.GTE: E.LTE, E.LTE: E.GTE, E.EQ: E.EQ, E.NEQ: E.NEQ}
    def judge(n):
        if isinstance(n, E.Paren): return judge(n.this)
        if isinstance(n, E.And): return judge(n.this) & judge(n.expression)
        if isinstance(n, E.Or): return judge(n.this) | judge(n.expression)
        if isinstance(n, E.Not): return ~judge(n.this)
        if type(n) in OPS:
            a, b, t = n.this, n.expression, type(n)
            if isinstance(b, E.Column) and not isinstance(a, E.Column):
                a, b, t = b, a, FL[t]
            if not isinstance(a, E.Column): raise _unsup()
            return OPS[t](td, _lit_value(seg, pc, b))
        if isinstance(n, E.Between) and isinstance(n.this, E.Column):
            return (td >= _lit_value(seg, pc, n.args['low'])) & (td <= _lit_value(seg, pc, n.args['high']))
        if isinstance(n, E.In) and isinstance(n.this, E.Column) and n.args.get('query') is None \
                and n.args.get('expressions') and n.args.get('_codes') is None:
            return np.isin(td, np.asarray([_lit_value(seg, pc, x) for x in n.args['expressions']]))
        raise _unsup()
    keep = np.ascontiguousarray(judge(cj), dtype=bool)
    if rows.size == 0:
        return rows
    return rows[keep[np.asarray(seg.codes_at(pc, rows)).astype(np.int64)]]


def _unsup():
    import wdb_join
    return wdb_join._FastUnsupported()


def values(db, sp, at, rows):
    """the subquery's value at each survivor row (NaN: no value). at(name, rows) -> (segment, physical column,
    the column's rows for those survivors), the projection door's own reader."""
    import wdb_join as WJ
    import time as _t
    bill = [] if os.environ.get('WDB_CORR_ROW_BILL') else None
    t0 = [_t.perf_counter()]
    def tick(nm):
        if bill is not None:
            t1 = _t.perf_counter(); bill.append('%s=%.1fms' % (nm, (t1 - t0[0]) * 1e3)); t0[0] = t1
    rows = np.asarray(rows, dtype=np.int64)
    m = rows.shape[0]
    # 1. the survivors' keys
    ov = []
    for _ic, oc in sp['eqs']:
        sg, pc, pr = at(oc, rows)
        v = WJ._int_vals(sg, pc, pr)
        if v is None: raise _unsup()
        ov.append(v)
    lo = [int(v.min()) if m else 0 for v in ov]
    span = [int(v.max()) - l + 1 if m else 1 for v, l in zip(ov, lo)]
    mult, tot = [], 1
    for s in reversed(span):
        mult.insert(0, tot); tot *= s
    if tot >= (1 << 62): raise _unsup()
    def key(vs):
        k = np.zeros(vs[0].shape[0], dtype=np.int64)
        ok = np.ones(vs[0].shape[0], dtype=bool)
        for v, l, s, mu in zip(vs, lo, span, mult):
            d = v - l
            ok &= (d >= 0) & (d < s)
            k += np.where(ok, d, 0) * mu
        return k, ok
    tick('outer-keys')
    okey, _ = key(ov)
    uk, inv = _uniq_inv(okey)
    tick('distinct-keys')
    if m == 0:
        return np.zeros(0)
    # 2. the inner rows: its own conditions + each inner key among the survivors' values
    seg2, _ = WJ._solo_segment(db, sp['t2'])
    pm2 = db.cat.phys_map(sp['t2'])
    conds = [c.copy() for c in sp['rest']]
    none = False                                    # an equality no inner value meets: no inner rows at all
    cands = []                                      # (share of the dictionary the survivors can name, column, values)
    for (ic, _oc), v, l, s in zip(sp['eqs'], ov, lo, span):
        pc2 = pm2.get(ic, ic)
        c2 = seg2.cols.get(pc2)
        if c2 is None or c2.get('has_null') or c2.get('dt') not in (0, 3): raise _unsup()
        u = _distinct(v, l, s)
        cands.append((u.size / max(int(c2.get('V') or 1), 1), ic, pc2, u))
    if not _ALL_SETS[0]:
        cands = sorted(cands, key=lambda x: x[0])[:1]   # the narrowest set alone: one lookup per inner row; the
                                                        # key match below is the exact test
    sets = []
    for _sh, ic, pc2, u in cands:
        td = np.asarray(seg2._typed_dict(pc2))
        if td.dtype.kind not in 'iu' or (td.size > 1 and not bool(np.all(td[1:] >= td[:-1]))): raise _unsup()
        ix = np.searchsorted(td, u)
        ok = ix < td.size
        ok[ok] = td[ix[ok]] == u[ok]
        codes = ix[ok].astype(np.int64)             # u is distinct and td sorted: the codes are distinct, rising
        if codes.size == 0:
            none = True; break
        if codes.size < td.size:
            sets.append((ic, codes))
    for ic, codes in sets:
        n = E.In(this=E.Column(this=E.Identifier(this=ic, quoted=False),
                               table=E.Identifier(this=sp['a2'], quoted=False)))
        n.set('_codes', codes)
        conds.append(n)
    tick('code-sets')
    if none:
        irows = np.zeros(0, dtype=np.int64)
    elif conds:
        t = sp['sub'].copy()
        t.set('expressions', [E.Column(this=E.Identifier(this=sp['eqs'][0][0], quoted=False),
                                       table=E.Identifier(this=sp['a2'], quoted=False))])
        t.set('where', E.Where(this=E.and_(*conds)))
        try:
            jr = WJ.join_rows(db, t)
            irows = np.asarray(jr['rows'], dtype=np.int64)
            left = jr.get('left') or []
        except (WJ._FastUnsupported, NotImplementedError):
            irows = np.arange(int(seg2.N), dtype=np.int64)   # the cascade stood down (a filter that keeps much):
            left = sorted(conds, key=lambda c: c.args.get('_codes') is None)   # every condition judged here,
                                                                              # row by code -- the key set first
        for cj in left:                             # what the cascade left unjudged: one column, over its dictionary
            if cj.args.get('_codes') is not None:   # a key's code set
                pc9 = pm2.get(cj.this.name, cj.this.name)
                lut = np.zeros(int(seg2.cols[pc9]['V']) + 1, dtype=bool)
                lut[np.asarray(cj.args['_codes'], dtype=np.int64)] = True
                irows = irows[lut[np.asarray(seg2.codes_at(pc9, irows)).astype(np.int64)]] if irows.size else irows
            else:
                irows = _keep_left(seg2, pm2, sp['a2'], cj, irows)
    else:
        irows = np.arange(int(seg2.N), dtype=np.int64)
    tick('inner-rows')
    # 3. each inner row's key among the survivors' keys
    iv = []
    for ic, _oc in sp['eqs']:
        v = WJ._int_vals(seg2, pm2.get(ic, ic), irows)
        if v is None: raise _unsup()
        iv.append(v)
    ikey, iok = key(iv) if irows.size else (np.zeros(0, np.int64), np.zeros(0, bool))
    pos = _probe(np.ascontiguousarray(uk), np.ascontiguousarray(ikey), np.ascontiguousarray(iok))
    hit = pos >= 0
    gid = pos[hit]
    tick('inner-keys')
    # 4. the aggregate per key, then the arithmetic
    K = uk.size
    ag = sp['agg']
    cnt = np.bincount(gid, minlength=K)[:K]
    if isinstance(ag, E.Count) and sp['col'] is None:
        per = cnt.astype(np.float64)
    else:
        x = _num_at(seg2, pm2.get(sp['col'], sp['col']), irows[hit])
        if isinstance(ag, E.Count):
            per = cnt.astype(np.float64)
        elif isinstance(ag, (E.Sum, E.Avg)):
            s = np.bincount(gid, weights=x, minlength=K)[:K]
            with np.errstate(divide='ignore', invalid='ignore'):
                per = np.where(cnt > 0, s / np.maximum(cnt, 1) if isinstance(ag, E.Avg) else s, np.nan)
        else:
            per = np.full(K, np.inf if isinstance(ag, E.Min) else -np.inf)
            (np.minimum if isinstance(ag, E.Min) else np.maximum).at(per, gid, x)
            per[cnt == 0] = np.nan
    val = np.asarray(_arith(sp['expr'], ag, per), dtype=np.float64)
    if val.ndim == 0:
        val = np.full(K, float(val))
    _HITS[0] += 1
    tick('aggregate')
    if bill is not None:
        print('CORR ROW BILL: %d survivors, %d keys, %d inner rows | %s' % (m, K, irows.size, ' '.join(bill)), flush=True)
    # 5. each survivor's value
    return val[inv]


def keep(db, sp, at, rows):
    """the survivors where the comparison is true"""
    import operator
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return rows
    v = values(db, sp, at, rows)
    sg, pc, pr = at(sp['ocol'], rows)
    o = _num_at(sg, pc, pr)
    f = {E.GT: operator.gt, E.GTE: operator.ge, E.LT: operator.lt, E.LTE: operator.le, E.EQ: operator.eq}[sp['op']]
    with np.errstate(invalid='ignore'):
        k = f(o, v) & ~np.isnan(v)
    return rows[k]
