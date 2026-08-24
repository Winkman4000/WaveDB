"""wdb_decodespec -- THE DECODE SPEC (planning-layer classifier).

Jackson's law: the decode layer's only contribution is decoding LESS, and
it can only act on information planning gives it. This pass walks the
query tree and stamps every column reference with a USE CLASS and SCOPE,
intersected with the passport's data properties, producing the spec that
tells downstream readers and the emit what may stay in code space:

  IDENTITY  -- GROUP BY keys, join keys, =/IN predicates, DISTINCT:
               needs only same-or-different. Codes suffice ALWAYS.
  ORDER     -- ORDER BY, range predicates, MIN/MAX, top-k: needs
               which-is-bigger. Codes suffice IFF the dictionary is
               order-isomorphic (sorted numeric dicts; front-coded
               string dicts are lexicographic by construction).
  VALUE     -- arithmetic operands and final output: the only true
               decodes. Arithmetic scope = a V-sized LUT (values[code],
               never N rows); output scope = exactly the final k rows.

General by construction: query properties x passport properties, both
known before execution. No per-query heuristics.
"""
import sqlglot
import sqlglot.expressions as E

IDENTITY, ORDER, VALUE = 'identity', 'order', 'value'


def _order_isomorphic(seg, pcol):
    """True when code order == value order, from the passport alone."""
    c = seg.cols.get(pcol)
    if c is None:
        return False
    if c.get('dt') == 1:
        return c.get('mode') == 1        # front-coded string dicts are lexicographic
    return c.get('mode') in (1, 2, 4)    # ordered numeric dicts / identity


def classify(tree):
    """Walk the tree; return {column_name: {'class': str, 'scope': str}}.
    Class escalation: identity < order < value (a column used two ways
    takes the stronger need). Scope for VALUE: 'lut' (arithmetic -- V-sized
    values[code]) or 'rows:k' (output -- decode only the final k)."""
    rank = {IDENTITY: 0, ORDER: 1, VALUE: 2}
    spec = {}

    def stamp(col, cls, scope=None):
        if not isinstance(col, E.Column):
            return
        nm = col.name
        cur = spec.get(nm)
        if cur is None or rank[cls] > rank[cur['class']]:
            spec[nm] = {'class': cls, 'scope': scope}

    grp = tree.args.get('group')
    for g in (grp.expressions if grp else []):
        for c in g.find_all(E.Column):
            stamp(c, IDENTITY)
    ordn = tree.args.get('order')
    for oe in (ordn.expressions if ordn else []):
        for c in oe.find_all(E.Column):
            stamp(c, ORDER)
    w = tree.args.get('where')
    if w is not None:
        for node in w.find_all((E.EQ, E.In)):
            for c in node.find_all(E.Column):
                stamp(c, IDENTITY)
        for node in w.find_all((E.GT, E.GTE, E.LT, E.LTE, E.Between)):
            for c in node.find_all(E.Column):
                stamp(c, ORDER)
    for fn in tree.find_all(E.AggFunc):
        nm9 = fn.key.upper() if hasattr(fn, 'key') else ''
        cls = ORDER if nm9 in ('MIN', 'MAX') else VALUE
        for c in fn.find_all(E.Column):
            stamp(c, cls, 'lut' if cls == VALUE else None)
    lim = tree.args.get('limit')
    k = None
    if lim is not None:
        try:
            k = int(lim.expression.name)
        except Exception:
            k = None
    for p in tree.expressions:                       # projected output columns
        if p.find(E.AggFunc) is None:
            for c in p.find_all(E.Column):
                stamp(c, VALUE, 'rows:%s' % (k if k is not None else 'all'))
    return spec


def resolve(spec, seg_of, pcol_of):
    """Intersect the query spec with passports: ORDER-class columns whose
    dictionaries are NOT order-isomorphic escalate to VALUE('lut'), since
    which-is-bigger then requires values. Returns the final decode spec."""
    out = {}
    for nm, st in spec.items():
        cls, scope = st['class'], st['scope']
        seg = seg_of(nm)
        if cls == ORDER and seg is not None \
                and not _order_isomorphic(seg, pcol_of(nm)):
            cls, scope = VALUE, 'lut'
        out[nm] = {'class': cls, 'scope': scope}
    return out
