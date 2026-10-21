"""wdb_keepflag -- one column's WHERE as a keep flag over its dictionary codes (2026-10-21, the Q24-Q26 board).

keep[code] is True when the value with that code passes the predicate; the NULL code (V-1 when the column has
nulls) is False for every comparison, as in SQL. Every dictionary is stored in VALUE ORDER (encode: np.unique /
factorize(sort=True); measured on all 35 column profiles of hits: numbers ascending, text by bytes), so a
range is a run of codes and the bound of a run is one bisect over the dictionary.

Shapes: col =, <>, <, <=, >, >= literal (either side), BETWEEN, [NOT] IN (literals), [NOT] LIKE, IS [NOT] NULL,
and AND / OR / NOT of these over ONE column. Anything else -> None (the caller declines). The flag is the
filter for the reads that walk rows (wdb_firstk) and for the reads that walk the dictionary (wdb_valsort).
"""
import numpy as np
from sqlglot import expressions as E

import wdb_wherescan as WS

_FLIP = {E.GT: E.LT, E.LT: E.GT, E.GTE: E.LTE, E.LTE: E.GTE, E.EQ: E.EQ, E.NEQ: E.NEQ}


def column_of(node, cm=None):
    """the one physical column a predicate reads, or None (several columns, a subquery, none)"""
    if node is None or node.find(E.Select) is not None:
        return None
    cm = cm or {}
    names = {cm.get(c.name, c.name) for c in node.find_all(E.Column)}
    return names.pop() if len(names) == 1 else None


def _lit(node):
    """a literal's Python value, or the sentinel _NOLIT"""
    if isinstance(node, (E.Literal, E.Neg, E.Cast, E.Paren)):
        v = WS._litval(node)
        if v is not None:
            return v
    return _NOLIT


_NOLIT = object()


def _code(seg, col, v):
    """the code of a literal (None when absent); the empty string on a text dictionary is code 0 when present --
    the smallest value of a sorted dictionary -- one point read instead of a bisect (Q25's SearchPhrase <> '')"""
    c = seg.cols[col]
    if c['dt'] == 1 and v == '' and int(c['V']) - (1 if c['has_null'] else 0) > 0:
        return 0 if bytes(seg.fetch(col, 0) or b'') == b'' else None
    return WS._code_of(seg, col, v)


def _key(seg, col, val):
    """the literal in the dictionary's own comparison space, and a getter for the dictionary's value at a code
    in that space; None when the literal cannot compare with the column"""
    c = seg.cols[col]
    dt = c['dt']
    if dt == 1:
        return WS._lit_bytes(val), (lambda i: bytes(seg.fetch(col, i)))
    if dt == 3:
        iv = WS._time_units(seg, col, val)
        if iv is None:
            iv = WS._time_units(seg, col, str(val))
        return (iv, (lambda i: int(np.asarray(seg.fetch(col, i)).astype('int64')))) if iv is not None else None
    try:
        kv = float(val) if dt == 2 else (int(val) if float(val).is_integer() else float(val))
    except (TypeError, ValueError):
        return None
    if c.get('mode') == 0:
        td = np.asarray(seg._typed_dict(col))
        return kv, (lambda i, td=td: td[i])
    return kv, (lambda i: seg.fetch(col, i))


def _bound(seg, col, val, right):
    """first code whose value is > val (right) or >= val (not right): one bisect over the sorted dictionary"""
    c = seg.cols[col]
    Vn = int(c['V']) - (1 if c['has_null'] else 0)
    kg = _key(seg, col, val)
    if kg is None:
        return None
    key, at = kg
    lo, hi = 0, Vn
    while lo < hi:
        mid = (lo + hi) // 2
        v = at(mid)
        if (v <= key) if right else (v < key):
            lo = mid + 1
        else:
            hi = mid
    return lo


def flag(seg, node, col):
    """bool[V]: True where the code's value passes `node` (a predicate over `col` alone); None if unjudged"""
    c = seg.cols.get(col)
    if c is None or c.get('mode') not in (0, 1, 2) or seg._overrides(col) is not None:
        return None
    V = int(c['V'])
    null = (V - 1) if c['has_null'] else None
    Vn = V - (1 if null is not None else 0)

    def run(lo, hi):
        f = np.zeros(V, bool)
        f[max(0, lo):max(0, min(hi, Vn))] = True
        return f

    def go(n):
        if isinstance(n, E.Paren):
            return go(n.this)
        if isinstance(n, E.And):
            a, b = go(n.this), go(n.expression)
            return None if a is None or b is None else (a & b)
        if isinstance(n, E.Or):
            a, b = go(n.this), go(n.expression)
            return None if a is None or b is None else (a | b)
        if isinstance(n, E.Not):
            if isinstance(n.this, E.Is):                      # IS NOT NULL
                a = go(n.this)
                return None if a is None else ~a
            a = go(n.this)
            if a is None:
                return None
            a = ~a
            if null is not None:
                a[null] = False                               # NOT (NULL op x) is not true either
            return a
        if isinstance(n, E.Is) and isinstance(n.expression, E.Null) and isinstance(n.this, E.Column):
            f = np.zeros(V, bool)
            if null is not None:
                f[null] = True
            return f
        if isinstance(n, (E.Like, E.ILike)):
            if isinstance(n, E.ILike):
                return None
            lk = WS._like(n)
            if lk is None:
                return None
            f = np.asarray(WS._like_flags(seg, col, lk[1], lk[2]), bool).copy()
            if lk[3]:
                f = ~f
                if null is not None:
                    f[null] = False
            return f
        if isinstance(n, E.Between) and isinstance(n.this, E.Column):
            lo_v, hi_v = _lit(n.args['low']), _lit(n.args['high'])
            if lo_v is _NOLIT or hi_v is _NOLIT:
                return None
            lo, hi = _bound(seg, col, lo_v, False), _bound(seg, col, hi_v, True)
            return None if lo is None or hi is None else run(lo, hi)
        if isinstance(n, E.In) and isinstance(n.this, E.Column):
            if n.args.get('query') is not None:
                return None
            vals = [_lit(x) for x in (n.expressions or [])]
            if not vals or any(v is _NOLIT for v in vals):
                return None
            f = np.zeros(V, bool)
            cd = WS._in_codes(seg, col, vals)
            cd = np.asarray(cd, np.int64)
            f[cd[(cd >= 0) & (cd < Vn)]] = True
            return f
        if type(n) in _FLIP:
            a, b = n.this, n.expression
            if isinstance(a, E.Column) and _lit(b) is not _NOLIT:
                op, v = type(n), _lit(b)
            elif isinstance(b, E.Column) and _lit(a) is not _NOLIT:
                op, v = _FLIP[type(n)], _lit(a)
            else:
                return None
            if op in (E.EQ, E.NEQ):
                cd = _code(seg, col, v)
                f = np.zeros(V, bool)
                if cd is not None and 0 <= int(cd) < Vn:
                    f[int(cd)] = True
                if op is E.NEQ:
                    f = ~f
                    if null is not None:
                        f[null] = False
                return f
            if op is E.LT:
                b9 = _bound(seg, col, v, False); return None if b9 is None else run(0, b9)
            if op is E.LTE:
                b9 = _bound(seg, col, v, True); return None if b9 is None else run(0, b9)
            if op is E.GT:
                b9 = _bound(seg, col, v, True); return None if b9 is None else run(b9, Vn)
            if op is E.GTE:
                b9 = _bound(seg, col, v, False); return None if b9 is None else run(b9, Vn)
        return None

    try:
        return go(node)
    except (ValueError, TypeError):
        return None


def runs(seg, node, col):
    """the kept codes as sorted, disjoint [lo, hi) runs -- the same judgement as flag() for the shapes that are
    unions of value ranges (=, <>, <, <=, >, >=, BETWEEN, IN and their AND / OR), WITHOUT a V-sized table: a
    WatchID <> x flag is 100M booleans (~80 ms to build and scan) where the runs are two pairs. None for the
    rest (LIKE, IS NULL, NOT ...): the caller builds the flag."""
    c = seg.cols.get(col)
    if c is None or c.get('mode') not in (0, 1, 2) or seg._overrides(col) is not None:
        return None
    V = int(c['V'])
    Vn = V - (1 if c['has_null'] else 0)

    def norm(rs):
        rs = sorted((max(0, a), min(Vn, b)) for a, b in rs if min(Vn, b) > max(0, a))
        out = []
        for a, b in rs:
            if out and a <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        return out

    def both(x, y):
        out = []; i = j = 0
        while i < len(x) and j < len(y):
            a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
            if a < b:
                out.append((a, b))
            if x[i][1] < y[j][1]:
                i += 1
            else:
                j += 1
        return out

    def go(n):
        if isinstance(n, E.Paren):
            return go(n.this)
        if isinstance(n, (E.And, E.Or)):
            a, b = go(n.this), go(n.expression)
            if a is None or b is None:
                return None
            return both(a, b) if isinstance(n, E.And) else norm(a + b)
        if isinstance(n, E.Between) and isinstance(n.this, E.Column):
            lo_v, hi_v = _lit(n.args['low']), _lit(n.args['high'])
            if lo_v is _NOLIT or hi_v is _NOLIT:
                return None
            lo, hi = _bound(seg, col, lo_v, False), _bound(seg, col, hi_v, True)
            return None if lo is None or hi is None else norm([(lo, hi)])
        if isinstance(n, E.In) and isinstance(n.this, E.Column):
            if n.args.get('query') is not None:
                return None
            vals = [_lit(x) for x in (n.expressions or [])]
            if not vals or any(v is _NOLIT for v in vals):
                return None
            cd = np.asarray(WS._in_codes(seg, col, vals), np.int64)
            return norm([(int(x), int(x) + 1) for x in cd if 0 <= x < Vn])
        if type(n) in _FLIP:
            a, b = n.this, n.expression
            if isinstance(a, E.Column) and _lit(b) is not _NOLIT:
                op, v = type(n), _lit(b)
            elif isinstance(b, E.Column) and _lit(a) is not _NOLIT:
                op, v = _FLIP[type(n)], _lit(a)
            else:
                return None
            if op in (E.EQ, E.NEQ):
                cd = _code(seg, col, v)
                if cd is None or not (0 <= int(cd) < Vn):
                    return [] if op is E.EQ else norm([(0, Vn)])
                cd = int(cd)
                return [(cd, cd + 1)] if op is E.EQ else norm([(0, cd), (cd + 1, Vn)])
            b9 = _bound(seg, col, v, op in (E.LTE, E.GT))
            if b9 is None:
                return None
            return norm([(0, b9)]) if op in (E.LT, E.LTE) else norm([(b9, Vn)])
        return None

    try:
        return go(node)
    except (ValueError, TypeError):
        return None


def first_codes(rs, n, desc=False):
    """the first n codes of the runs, in value order (or reversed)"""
    out = []
    for a, b in (reversed(rs) if desc else rs):
        if desc:
            out.extend(range(b - 1, max(a, b - (n - len(out))) - 1, -1))
        else:
            out.extend(range(a, min(b, a + (n - len(out)))))
        if len(out) >= n:
            break
    return np.asarray(out[:n], np.int64)
