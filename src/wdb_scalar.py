"""wdb_scalar: scalar functions computed at the DICTIONARY level -- never per row.

The regexgroup law generalized into engine infrastructure: a supported scalar expression over
one column becomes a code->value table, derived once per DISTINCT value (V-sized work, not
N-sized) and memoized for the process lifetime. Reads gather row codes as usual and index the
table; a 100M-row GROUP BY lower(url) pays the string work 19.7M times, not 100M -- and pays
it once per process.

v1 functions: LENGTH (UTF-8 char count via the continuation-byte trick), LOWER, UPPER, TRIM,
SUBSTR, EXTRACT(year|month|day|hour|minute FROM <datetime dict col>), and +/-/* by an integer
literal over integer dict columns. Keys only (GROUP BY / projection); scalar predicates later.
"""
import numpy as np
import sqlglot.expressions as E

_FN_STR = {'Lower': 'lower', 'Upper': 'upper', 'Trim': 'trim', 'Length': 'length'}
_EXTRACT_OK = {'year', 'month', 'day', 'hour', 'minute'}
_ARITH = {'Add': '+', 'Sub': '-', 'Mul': '*'}


def parse(seg, node, col_map):
    """spec dict for a supported scalar over one existing column, else None.
    spec: {'col', 'fn', 'arg'} with fn in lower|upper|trim|length|substr|extract_<unit>|arith."""
    n = node.this if isinstance(node, E.Alias) else node

    def rescol(cnode):
        if not isinstance(cnode, E.Column):
            return None
        c = col_map.get(cnode.name, cnode.name) if col_map else cnode.name
        return c if c in seg.cols and seg._effective(c) is None else None

    if isinstance(n, E.Column):
        # identity scalar: a bare int-valued dict column IS its own code->value table.
        # Ranges and comparisons on dt0 byte dicts become one V-sized compare (the
        # f-range gate: codes sort by string, values sort by value -- flags fix it)
        col = rescol(n)
        if col is None:
            return None
        c = seg.cols[col]
        if c.get('mode') not in (0, 1, 2) or c.get('has_null') or c.get('dt') != 0:
            return None
        return {'col': col, 'fn': 'id', 'arg': None}
    tn = type(n).__name__
    if tn in _FN_STR:
        col = rescol(n.this)
        if col is None or seg.cols[col].get('mode') not in (0, 1, 2):
            return None
        return {'col': col, 'fn': _FN_STR[tn], 'arg': None}
    if isinstance(n, E.Substring):
        col = rescol(n.this)
        if col is None or seg.cols[col].get('mode') not in (0, 1):
            return None
        try:
            start = int(n.args['start'].this)
            ln = int(n.args['length'].this) if n.args.get('length') is not None else None
        except Exception:
            return None
        return {'col': col, 'fn': 'substr', 'arg': (start, ln)}
    if isinstance(n, E.Extract):
        unit = n.this.name.lower() if hasattr(n.this, 'name') else str(n.this).lower()
        col = rescol(n.expression)
        if unit not in _EXTRACT_OK or col is None:
            return None
        if seg.cols[col].get('dt') != 3 or seg.cols[col].get('mode') not in (0, 1, 2):
            return None
        return {'col': col, 'fn': 'extract_' + unit, 'arg': None}
    if tn in _ARITH:
        a, b = n.this, n.expression
        lit = None
        if isinstance(b, E.Literal):
            col, lit = rescol(a), b
        elif isinstance(a, E.Literal) and tn != 'Sub':
            col, lit = rescol(b), a
        else:
            return None
        if col is None or lit is None:
            return None
        c = seg.cols[col]
        if c.get('dt') != 0 or c.get('mode') not in (0, 2):
            return None
        try:
            v = int(str(lit.this))
        except Exception:
            return None
        return {'col': col, 'fn': 'arith', 'arg': (_ARITH[tn], v)}
    return None


def _key(spec):
    return (spec['col'], spec['fn'], spec['arg'])


def table(seg, spec):
    """code -> derived value, memoized. Numeric fns return np arrays; string fns return
    object arrays of str."""
    memo = seg.__dict__.setdefault('_scalar_memo', {})
    mk = _key(spec)
    if mk in memo:
        return memo[mk]
    col, fn, arg = spec['col'], spec['fn'], spec['arg']
    c = seg.cols[col]
    if fn == 'id':
        import wdb_window as _W
        t = np.asarray(_W._int_table(seg, col), dtype=np.int64)
        memo[mk] = t
        return t
    if fn == 'arith':
        base = np.asarray(seg._dict_ints(c), dtype=np.int64) if c['mode'] == 2 else \
            np.array([int(v) for v in seg._typed_dict(col)], dtype=np.int64)
        op, v = arg
        out = base + v if op == '+' else (base - v if op == '-' else base * v)
    elif fn.startswith('extract_'):
        secs = np.asarray(seg._dict_ints(c), dtype=np.int64)
        dt = secs.astype('datetime64[s]')
        unit = fn[8:]
        if unit == 'year':
            out = dt.astype('datetime64[Y]').astype(np.int64) + 1970
        elif unit == 'month':
            out = (dt.astype('datetime64[M]').astype(np.int64) % 12) + 1
        elif unit == 'day':
            out = (dt - dt.astype('datetime64[M]')).astype('timedelta64[D]').astype(np.int64) + 1
        elif unit == 'hour':
            out = (secs // 3600) % 24
        else:
            out = (secs // 60) % 60
    elif fn == 'length':
        bs = [v if isinstance(v, (bytes, bytearray)) else str(v).encode()
              for v in seg._typed_dict(col)]
        lens = np.fromiter((len(v) for v in bs), np.int64, len(bs))
        offs = np.zeros(len(bs) + 1, np.int64); np.cumsum(lens, out=offs[1:])
        cont = (np.frombuffer(b''.join(bs), dtype=np.uint8) & 0xC0) == 0x80
        cs = np.zeros(offs[-1] + 1, np.int64); np.cumsum(cont, out=cs[1:])
        out = lens - (cs[offs[1:]] - cs[offs[:-1]])
    else:
        vals = [(v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else str(v))
                for v in seg._typed_dict(col)]
        if fn == 'lower':
            out = np.array([v.lower() for v in vals], dtype=object)
        elif fn == 'upper':
            out = np.array([v.upper() for v in vals], dtype=object)
        elif fn == 'trim':
            out = np.array([v.strip() for v in vals], dtype=object)
        else:                                        # substr (1-based SQL semantics)
            start, ln = arg
            i0 = start - 1 if start > 0 else 0
            out = np.array([v[i0:] if ln is None else v[i0:i0 + ln] for v in vals], dtype=object)
    memo[mk] = out
    return out


def surrogate(seg, spec):
    """(code -> surrogate id, id -> value list): int-composable form for radix key folds.
    Distinct derived values get dense ids in SORTED value order (canonical tie order)."""
    memo = seg.__dict__.setdefault('_scalar_surr_memo', {})
    mk = _key(spec)
    if mk in memo:
        return memo[mk]
    t = table(seg, spec)
    uniq, inv = np.unique(t, return_inverse=True)
    memo[mk] = (inv.astype(np.int64), list(uniq))
    return memo[mk]
