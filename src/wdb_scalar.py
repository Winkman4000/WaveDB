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


def _stream(seg, col):
    """(blob, offsets) of a text dictionary laid out as one byte stream (wdb_sql._dict_stream: compiled
    front-coding expansion, tier 1 on the column), or None when the column is not a plain text dictionary"""
    c = seg.cols[col]
    if c.get('dt') != 1 or c.get('mode') not in (0, 1) or c.get('aux') == 9:
        return None
    try:
        import wdb_sql
        blob, off = wdb_sql._dict_stream(seg, col)
    except Exception:
        return None
    return np.ascontiguousarray(blob), np.ascontiguousarray(off, dtype=np.int64)


def table(seg, spec):
    """code -> derived value, memoized. Numeric fns return np arrays; string fns return
    object arrays of str."""
    memo = seg.__dict__.setdefault('_scalar_memo', {})
    mk = _key(spec)
    if mk in memo:
        return memo[mk]
    col, fn, arg = spec['col'], spec['fn'], spec['arg']
    c = seg.cols[col]
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
        st = _stream(seg, col)
        if st is not None:
            # THE LENGTH ON THE STREAM (2026-10-05): one parallel pass counts each value's characters on the
            # dictionary's compiled byte stream -- the Python road unpacked 18-20M URLs/Referers into objects
            # first (LENGTH(URL) > 100: 16 s hot; GROUP BY LENGTH(Referer): 20 s)
            import wdb_kernels as _WKs
            blob, off = st
            out = np.empty(off.size - 1, np.int64)
            _WKs.pstr_charlen(blob, off, out)
        else:
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


class _SliceVals:
    """the distinct derived values, decoded only when asked: value k is the k-th distinct slice of the stream"""
    def __init__(self, blob, ss, se, reps):
        self.blob, self.ss, self.se, self.reps = blob, ss, se, reps
    def __len__(self):
        return int(self.reps.size)
    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self[j] for j in range(*k.indices(len(self)))]
        r = int(self.reps[int(k)])
        return self.blob[int(self.ss[r]):int(self.se[r])].tobytes().decode('utf-8', 'replace')
    def __iter__(self):
        for k in range(len(self)):
            yield self[k]


def _surrogate_prefix(seg, spec):
    """SUBSTR(col, 1, n) -- a prefix -- over a value-sorted dictionary keeps the order, so equal prefixes sit
    side by side: the slices are byte bounds on the dictionary's stream (no copy), a neighbour compare marks
    each new value, and a running count is the sorted surrogate id. No Python string until the answer is
    printed (the Python road sliced 18M URLs and sorted the slices: 28 s of GROUP BY SUBSTR(URL, 1, 20)).
    None when it does not apply (another start, malformed UTF-8, a dictionary not in order)."""
    start, ln = spec['arg']
    if start > 1 or (ln is not None and ln < 0):
        return None
    st = _stream(seg, spec['col'])
    if st is None:
        return None
    import wdb_kernels as K
    blob, off = st
    n = off.size - 1
    ss = np.empty(n, np.int64); se = np.empty(n, np.int64); bad = np.empty(n, np.bool_)
    K.psubstr_bounds(blob, off, 0, -1 if ln is None else int(ln), ss, se, bad)
    if bad.any():
        return None
    chg = np.empty(n, np.uint8)
    K.pslice_change(blob, ss, se, chg)
    if n and (chg == 2).any():
        return None
    inv = np.cumsum(chg != 0) - 1
    return inv.astype(np.int64), _SliceVals(blob, ss, se, np.flatnonzero(chg))


def surrogate(seg, spec):
    """(code -> surrogate id, id -> value list): int-composable form for radix key folds.
    Distinct derived values get dense ids in SORTED value order (canonical tie order)."""
    memo = seg.__dict__.setdefault('_scalar_surr_memo', {})
    mk = _key(spec)
    if mk in memo:
        return memo[mk]
    if spec['fn'] == 'substr':
        got = _surrogate_prefix(seg, spec)
        if got is not None:
            memo[mk] = got
            return got
    t = table(seg, spec)
    if isinstance(t, np.ndarray) and t.dtype.kind in 'iu' and t.size:
        lo, hi = int(t.min()), int(t.max())
        if hi - lo < (1 << 24):
            # INTEGER KEYS NEED NO SORT: presence over the value range gives the sorted distinct values and
            # each value's rank in one pass (np.unique sorted 20M lengths: ~1.5 s of GROUP BY LENGTH)
            pres = np.zeros(hi - lo + 1, np.bool_); pres[t - lo] = True
            rank = np.cumsum(pres) - 1
            memo[mk] = (rank[t - lo].astype(np.int64), list(np.flatnonzero(pres) + lo))
            return memo[mk]
    uniq, inv = np.unique(t, return_inverse=True)
    memo[mk] = (inv.astype(np.int64), list(uniq))
    return memo[mk]
