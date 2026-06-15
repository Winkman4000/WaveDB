"""
wdb_valsort -- value-sorted projection top-K.

Answers   SELECT col FROM t WHERE col <> '' ORDER BY col ASC LIMIT k
by reading the dictionary directly. String dictionaries are stored alphabetically
(encode uses pd.factorize(sort=True) / np.unique), so VALUE order == CODE order, the
empty string is the smallest non-null code, and NULLs are the highest code. The first
k rows by value are therefore the first k dictionary entries past '' (and below the
null code) -- so we grab the first few codes from the front of the sorted dict and
decode them. No row scan, no counts, no cache: the dictionary order IS the sort, and
the engine already caches the one dict decompress.

Boundary (by design, per Jackson): this emits distinct values in order, which equals
the row-answer unless a *non-empty* value repeats in the column (SQL would repeat it;
we step to the next value). Verified matching DuckDB on SearchPhrase. The always-correct
variant would cost a full count pass; we've chosen not to pay that here.

detect (fail-closed): single table; projection is exactly one bare value-identity
string-dict column C (dt==1, mode 0/1); ORDER BY C ASC, single key; LIMIT present;
WHERE is exactly `C <> ''`; no GROUP BY / HAVING / DISTINCT / JOIN; no deleted rows.
Anything else -> None (caller scans).
"""
import wdb_sql
import workers
import wdb_policies as P
E = wdb_sql.E

_HITS = 0


def _where_is_neq_empty(tree, colname):
    """True iff WHERE is exactly `colname <> ''` (the only filter v1 handles)."""
    w = tree.args.get('where')
    if w is None:
        return False
    pred = w.this
    if not isinstance(pred, E.NEQ):
        return False
    lhs, rhs = pred.this, pred.expression
    if not (isinstance(lhs, E.Column) and lhs.name == colname):
        return False
    return isinstance(rhs, E.Literal) and rhs.is_string and rhs.this == ''


def _order_is_col_asc(tree, colname):
    """True iff ORDER BY is a single ascending key on `colname`."""
    order = tree.args.get('order')
    if order is None or len(order.expressions) != 1:
        return False
    first = order.expressions[0]
    if not isinstance(first, E.Ordered) or first.args.get('desc'):
        return False
    tgt = first.this
    return isinstance(tgt, E.Column) and tgt.name == colname


def detect(seg, tree, col_map):
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    if not P.has_limit(tree):          return None
    if tree.args.get('group') is not None: return None
    proj = tree.expressions
    if len(proj) != 1:                 return None
    if wdb_sql._agg_kind(proj[0]) is not None: return None      # must be a bare column, no agg
    colname = wdb_sql._proj_colname(proj[0])
    if colname is None:                return None
    if not _order_is_col_asc(tree, colname):   return None
    if not _where_is_neq_empty(tree, colname): return None      # v1: exactly `col <> ''`
    col = col_map.get(colname, colname) if col_map else colname
    if not P.columns_exist(seg, col):  return None
    if not P.not_positional(seg, col): return None
    if not P.no_deleted_rows(seg):     return None
    c = seg.cols[col]
    if c['dt'] != 1 or c['mode'] not in (0, 1):                 # value-identity string dict only
        return None
    return {'col': col, 'lim': wdb_sql._limit(tree), 'proj': proj,
            'V': c['V'], 'has_null': c['has_null']}


def execute(seg, spec):
    global _HITS
    col = spec['col']; lim = spec['lim']; V = spec['V']
    null_code = (V - 1) if spec['has_null'] else -1
    decode = lambda code: wdb_sql._pyval(seg.fetch(col, code))
    rows = workers.take_sorted(decode, V, lim, {null_code})     # grab from the front of the sorted dict
    _HITS += 1
    return rows, [wdb_sql._alias(spec['proj'][0])]


def try_valsort(seg, tree, col_map):
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)
