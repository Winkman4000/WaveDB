"""The shifting board's mutation engine: random, type-aware changes to a query's syntax tree.

Every change keeps the query meaningful against the real table: a column is only swapped for one
of the same kind (number, text, date, timestamp), a literal is replaced by a value that really
occurs in its column, a LIKE pattern is cut from a real string. Which change, where, and with
what value is drawn from the caller's random generator, so a seed reproduces a variant exactly.

    mutate(sql, rng, schema, pool, n_changes) -> (new_sql, [change descriptions]) or None

schema: {column: kind}, kind in 'int', 'text', 'date', 'ts'
pool:   {column: [sample values as Python objects]}   (values that occur in the data)
"""
import datetime
import sqlglot
from sqlglot import expressions as E

UNITS = ['second', 'minute', 'hour', 'day', 'month', 'year']
LIMITS = [1, 3, 5, 10, 25, 50, 100]
NUM_AGGS = [E.Sum, E.Min, E.Max, E.Avg]


def schema_from_create(path):
    """{column: kind} from the board's CREATE TABLE."""
    t = sqlglot.parse_one(open(path).read())
    out = {}
    for cd in t.find_all(E.ColumnDef):
        k = cd.args['kind'].this
        T = E.DataType.Type
        if k in (T.TIMESTAMP, T.DATETIME):
            out[cd.name] = 'ts'
        elif k == T.DATE:
            out[cd.name] = 'date'
        elif k in (T.TEXT, T.VARCHAR, T.CHAR, T.NCHAR, T.NVARCHAR):
            out[cd.name] = 'text'
        else:
            out[cd.name] = 'int'
    return out


# ---------------------------------------------------------------- literals from the data

def _lit(kind, v):
    """A SQL literal for value v of a column of this kind."""
    if v is None:
        return None
    if kind == 'int':
        return E.Literal.number(int(v))
    if kind == 'date':
        d = v if isinstance(v, datetime.date) else datetime.date.fromisoformat(str(v)[:10])
        return E.Literal.string(d.isoformat())
    if kind == 'ts':
        s = v.strftime('%Y-%m-%d %H:%M:%S') if isinstance(v, datetime.datetime) else str(v)[:19]
        return E.Literal.string(s)
    s = v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else str(v)
    return E.Literal.string(s)


def _pick(rng, pool, col, nonempty=False):
    vals = pool.get(col) or []
    if nonempty:
        vals = [v for v in vals if v not in (None, '', b'')]
    return vals[rng.randrange(len(vals))] if vals else None


def _like_pattern(rng, pool, col):
    v = _pick(rng, pool, col, nonempty=True)
    if v is None:
        return None
    s = v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else str(v)
    s = s.replace('%', '').replace('_', '')
    if len(s) < 3:
        return None
    n = rng.randint(3, min(10, len(s)))
    i = rng.randrange(len(s) - n + 1)
    piece = s[i:i + n]
    form = rng.choice(['%{}%', '{}%', '%{}'])
    if form == '{}%':
        piece = s[:n]
    elif form == '%{}':
        piece = s[len(s) - n:]
    return E.Literal.string(form.format(piece))


# ---------------------------------------------------------------- helpers over the tree

def _cols(t, schema):
    return [c for c in t.find_all(E.Column) if c.name in schema]


def _cmp_nodes(t):
    return [n for n in t.find_all(E.EQ, E.NEQ, E.GT, E.GTE, E.LT, E.LTE)
            if isinstance(n.this, E.Column) and isinstance(n.expression, E.Literal)]


def _same_kind(schema, kind, exclude):
    return sorted(c for c, k in schema.items() if k == kind and c not in exclude)


def _fix_literals(t, schema, pool, rng):
    """After a column swap, a literal compared with the new column must be one of ITS values."""
    for n in _cmp_nodes(t):
        col = n.this.name
        if col not in schema:
            continue
        kind = schema[col]
        lit = n.expression
        ok = (kind == 'int' and not lit.is_string) or (kind != 'int' and lit.is_string)
        if not ok or lit.meta.get('stale'):
            nv = _lit(kind, _pick(rng, pool, col, nonempty=True))
            if nv is not None:
                n.set('expression', nv)
    for n in t.find_all(E.Like, E.ILike):
        if isinstance(n.this, E.Column) and n.this.name in schema and schema[n.this.name] != 'text':
            return False                                   # LIKE moved onto a non-text column
    for n in t.find_all(E.In):
        if isinstance(n.this, E.Column) and n.this.name in schema:
            kind = schema[n.this.name]
            new = []
            for e in n.expressions:
                if isinstance(e, E.Literal) and ((kind == 'int') == e.is_string):
                    e = _lit(kind, _pick(rng, pool, n.this.name, nonempty=True)) or e
                new.append(e)
            n.set('expressions', new)
    return True


# ---------------------------------------------------------------- the changes

def m_swap_column(t, rng, schema, pool):
    """Rename one column everywhere in the query to another column of the same kind."""
    cols = sorted({c.name for c in _cols(t, schema)})
    if not cols:
        return None
    a = rng.choice(cols)
    cand = _same_kind(schema, schema[a], set(cols))
    if not cand:
        return None
    b = rng.choice(cand)
    for c in list(t.find_all(E.Column)):
        if c.name == a:
            c.replace(E.column(b))
    for n in _cmp_nodes(t):
        if n.this.name == b:
            n.expression.meta['stale'] = True
    for n in t.find_all(E.Like, E.ILike):
        if isinstance(n.this, E.Column) and n.this.name == b:
            p = _like_pattern(rng, pool, b)
            if p is not None:
                n.set('expression', p)
    return 'column %s -> %s' % (a, b)


def m_literal(t, rng, schema, pool):
    """Replace a compared literal with another value from that column."""
    nodes = [n for n in _cmp_nodes(t) if n.this.name in schema]
    nodes += [n for n in t.find_all(E.In) if isinstance(n.this, E.Column) and n.this.name in schema]
    if not nodes:
        return None
    n = rng.choice(nodes)
    col = n.this.name
    kind = schema[col]
    if isinstance(n, E.In):
        k = rng.randint(1, 4)
        vals = [_lit(kind, _pick(rng, pool, col, nonempty=True)) for _ in range(k)]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        n.set('expressions', vals)
        return '%s IN (%d values from the data)' % (col, len(vals))
    v = _lit(kind, _pick(rng, pool, col, nonempty=rng.random() < 0.8))
    if v is None:
        return None
    n.set('expression', v)
    return '%s literal -> %s' % (col, v.sql()[:40])


def m_like(t, rng, schema, pool):
    nodes = [n for n in t.find_all(E.Like, E.ILike) if isinstance(n.this, E.Column)]
    if not nodes:
        return None
    n = rng.choice(nodes)
    p = _like_pattern(rng, pool, n.this.name)
    if p is None:
        return None
    n.set('expression', p)
    return 'LIKE on %s -> %s' % (n.this.name, p.sql()[:40])


def m_time_unit(t, rng, schema, pool):
    nodes = list(t.find_all(E.Extract)) + list(t.find_all(E.DateTrunc, E.TimestampTrunc))
    if not nodes:
        return None
    n = rng.choice(nodes)
    key = n.sql()
    m = n.copy()
    if isinstance(m, E.Extract):
        old = m.this.name.lower()
        new = rng.choice([u for u in UNITS if u != old])
        m.set('this', E.var(new.upper()))
    else:
        u = m.args.get('unit')
        old = (u.name if u is not None else '').lower()
        new = rng.choice([u2 for u2 in UNITS if u2 != old])
        m.set('unit', E.Literal.string(new))
    for x in list(t.find_all(E.Extract, E.DateTrunc, E.TimestampTrunc)):
        if x.sql() == key:                       # every copy: SELECT, GROUP BY and ORDER BY agree
            x.replace(m.copy())
    return 'time unit %s -> %s' % (old, new)


def m_aggregate(t, rng, schema, pool):
    """Swap one aggregate everywhere it appears (SELECT and ORDER BY stay consistent)."""
    aggs = [a for a in t.find_all(E.Count, E.Sum, E.Min, E.Max, E.Avg)]
    if not aggs:
        return None
    a = rng.choice(aggs)
    key = a.sql()
    if isinstance(a, E.Count) and isinstance(a.this, E.Distinct):
        cols = [c for c in a.this.expressions if isinstance(c, E.Column) and c.name in schema]
        if not cols:
            return None
        cand = sorted(c for c in schema if c != cols[0].name)
        newcol = rng.choice(cand)
        new = E.Count(this=E.Distinct(expressions=[E.column(newcol)]))
        desc = 'COUNT(DISTINCT %s) -> COUNT(DISTINCT %s)' % (cols[0].name, newcol)
    else:
        ints = sorted(c for c, k in schema.items() if k == 'int')
        arg = a.this if isinstance(a.this, E.Column) and a.this.name in schema else None
        if arg is None or schema.get(arg.name) != 'int' or rng.random() < 0.3:
            arg = E.column(rng.choice(ints))
        cls = rng.choice([c for c in NUM_AGGS + [E.Count] if c is not type(a)] or NUM_AGGS)
        new = cls(this=arg.copy())
        desc = '%s -> %s' % (key[:40], new.sql()[:40])
    for x in list(t.find_all(E.Count, E.Sum, E.Min, E.Max, E.Avg)):
        if x.sql() == key:
            x.replace(new.copy())
    return desc


def m_flip_cmp(t, rng, schema, pool):
    nodes = list(t.find_all(E.EQ, E.NEQ, E.GT, E.GTE, E.LT, E.LTE))
    nodes = [n for n in nodes if not isinstance(n.parent, E.Join)]
    if not nodes:
        return None
    n = rng.choice(nodes)
    flip = {E.EQ: E.NEQ, E.NEQ: E.EQ, E.GT: E.GTE, E.GTE: E.GT, E.LT: E.LTE, E.LTE: E.LT}
    new = flip[type(n)](this=n.this.copy(), expression=n.expression.copy())
    n.replace(new)
    return 'comparison %s -> %s' % (type(n).__name__, type(new).__name__)


def m_limit(t, rng, schema, pool):
    lim = t.args.get('limit')
    if lim is None:
        return None
    k = rng.choice(LIMITS)
    t.set('limit', E.Limit(expression=E.Literal.number(k)))
    off = t.args.get('offset')
    if off is not None and rng.random() < 0.5:
        t.set('offset', E.Offset(expression=E.Literal.number(rng.choice([0, 10, 100, 1000]))))
    return 'LIMIT -> %d' % k


def m_order_dir(t, rng, schema, pool):
    order = t.args.get('order')
    if order is None or not order.expressions:
        return None
    o = order.expressions[0]
    o.set('desc', not o.args.get('desc'))
    return 'first ORDER BY key %s' % ('DESC' if o.args.get('desc') else 'ASC')


def m_add_filter(t, rng, schema, pool):
    """AND a new condition on a random column, with a value from the data."""
    col = rng.choice(sorted(schema))
    kind = schema[col]
    v = _lit(kind, _pick(rng, pool, col, nonempty=True))
    if v is None:
        return None
    if kind == 'text' and rng.random() < 0.5:
        p = _like_pattern(rng, pool, col)
        cond = E.Like(this=E.column(col), expression=p) if p is not None else E.NEQ(this=E.column(col), expression=E.Literal.string(''))
    else:
        op = rng.choice([E.EQ, E.NEQ, E.GT, E.LT] if kind != 'text' else [E.EQ, E.NEQ])
        cond = op(this=E.column(col), expression=v)
    w = t.args.get('where')
    t.set('where', E.Where(this=cond if w is None else E.And(this=w.this.copy(), expression=cond)))
    return 'added filter %s' % cond.sql()[:60]


def m_drop_filter(t, rng, schema, pool):
    w = t.args.get('where')
    if w is None:
        return None
    if isinstance(w.this, E.And):
        parts = list(w.this.flatten())
        drop = rng.randrange(len(parts))
        keep = [p for i, p in enumerate(parts) if i != drop]
        node = keep[0]
        for p in keep[1:]:
            node = E.And(this=node, expression=p)
        t.set('where', E.Where(this=node.copy()))
        return 'dropped filter %s' % parts[drop].sql()[:60]
    t.set('where', None)
    return 'dropped the WHERE'


def m_group_key(t, rng, schema, pool, lowcard=None):
    """Add a low-cardinality column to GROUP BY and SELECT, or drop a bare key from both."""
    g = t.args.get('group')
    if g is None or not g.expressions:
        return None
    keys = [k for k in g.expressions if isinstance(k, E.Column)]
    if keys and len(g.expressions) > 1 and rng.random() < 0.5:
        k = rng.choice(keys)

        def names_key(p):                        # the projection IS the key, or is aliased as it
            inner = p.this if isinstance(p, E.Alias) else p
            return inner.sql() == k.sql() or (isinstance(p, E.Alias) and p.alias == k.name)
        gone = [p for p in t.expressions if names_key(p)]
        proj = [p for p in t.expressions if not names_key(p)]
        if not proj or not any(p.find(E.AggFunc) for p in proj):
            return None
        g.set('expressions', [e for e in g.expressions if e is not k])
        t.set('expressions', proj)
        dead = {k.name} | {p.alias for p in gone if isinstance(p, E.Alias)} | \
               {(p.this if isinstance(p, E.Alias) else p).sql() for p in gone}
        order = t.args.get('order')
        if order is not None:
            keep = [o for o in order.expressions if o.this.sql() not in dead]
            t.set('order', E.Order(expressions=keep) if keep else None)
        return 'dropped group key %s' % k.name
    cand = [c for c in (lowcard or []) if c not in {k.name for k in keys}]
    if not cand:
        return None
    c = rng.choice(cand)
    g.set('expressions', list(g.expressions) + [E.column(c)])
    t.set('expressions', [E.column(c)] + list(t.expressions))
    return 'added group key %s' % c


CHANGES = [m_swap_column, m_literal, m_like, m_time_unit, m_aggregate, m_flip_cmp,
           m_limit, m_order_dir, m_add_filter, m_drop_filter, m_group_key]


def mutate(sql, rng, schema, pool, n_changes, lowcard=None):
    t = sqlglot.parse_one(sql.rstrip().rstrip(';'))
    done = []
    tries = 0
    while len(done) < n_changes and tries < 40:
        tries += 1
        f = rng.choice(CHANGES)
        before = t.copy()
        try:
            d = f(t, rng, schema, pool, lowcard) if f is m_group_key else f(t, rng, schema, pool)
        except Exception:
            d = None
        if d is None:
            t = before
            continue
        if not _fix_literals(t, schema, pool, rng):
            t = before
            continue
        done.append(d)
    if not done:
        return None
    out = t.sql()
    if out == sqlglot.parse_one(sql.rstrip().rstrip(';')).sql():
        return None
    return out, done
