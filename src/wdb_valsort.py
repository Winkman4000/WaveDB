"""
wdb_valsort -- value-sorted projection top-K.

Answers   SELECT col FROM t WHERE col <> '' ORDER BY col ASC LIMIT k
by reading the dictionary directly. String dictionaries are stored alphabetically
(encode uses pd.factorize(sort=True) / np.unique), so VALUE order == CODE order, the
empty string is the smallest non-null code, and NULLs are the highest code. We walk
the dictionary from the front (decode only the few values we emit, via seg.fetch),
skipping '' and the null code, and emit each value as many times as it occurs.

Multiplicity comes from the per-column dup-count map: codes that occur more than once,
with their counts (singletons implicit = 1). This is exactly the sidecar wdb_gbcount
already builds and persists as <seg>.<col>.gbc (count>=2, count-descending), so we
reuse it -- built once over the column, then read from disk forever, no query-time
scan. A popped dup consumes `count` of the LIMIT budget (emit value x count); this is
correct for any LIMIT, not just the unique-prefix case.

detect (fail-closed): single table; projection is exactly one bare value-identity
string-dict column C (dt==1, mode 0/1); ORDER BY C ASC, single key; LIMIT present;
WHERE is exactly `C <> ''`; no GROUP BY / HAVING / DISTINCT / JOIN; no deleted rows.
Anything else -> None (caller scans).
"""
import numpy as np
import wdb_sql
import workers
import wdb_gbcount
import wdb_policies as P
E = wdb_sql.E

_HITS = 0
_CACHE = {}   # (seg.path, col, N) -> (dup_codes_asc, dup_counts) | None


def _dupcounts(seg, col):
    """Code->count for codes occurring >1, as arrays sorted by code (for searchsorted).
    Reuses the persisted wdb_gbcount .gbc sidecar (built once, count>=2). None if absent."""
    key = (seg.path, col, int(seg.N))
    if key in _CACHE:
        return _CACHE[key]
    loaded = wdb_gbcount._load(seg, col)        # (codes count-desc, counts) | None
    if loaded is None:
        _CACHE[key] = None
        return None
    hc, hn = loaded
    o = np.argsort(hc, kind='stable')           # re-sort by code for lookup
    res = (np.ascontiguousarray(hc[o]), np.ascontiguousarray(hn[o]))
    _CACHE[key] = res
    return res


def _where_is_neq_empty(tree, colname):
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
    if wdb_sql._agg_kind(proj[0]) is not None: return None
    colname = wdb_sql._proj_colname(proj[0])
    if colname is None:                return None
    if not _order_is_col_asc(tree, colname):   return None
    if not _where_is_neq_empty(tree, colname): return None
    col = col_map.get(colname, colname) if col_map else colname
    if not P.columns_exist(seg, col):  return None
    if not P.not_positional(seg, col): return None
    if not P.no_deleted_rows(seg):     return None
    c = seg.cols[col]
    if c['dt'] != 1 or c['mode'] not in (0, 1):
        return None
    return {'col': col, 'lim': wdb_sql._limit(tree), 'proj': proj,
            'V': c['V'], 'has_null': c['has_null']}


def execute(seg, spec):
    global _HITS
    col = spec['col']; lim = spec['lim']; V = spec['V']
    null_code = (V - 1) if spec['has_null'] else -1
    dc = _dupcounts(seg, col)
    if dc is None:
        count_of = lambda code: 1
    else:
        ca, na = dc
        def count_of(code, ca=ca, na=na):
            i = np.searchsorted(ca, code)
            return int(na[i]) if (i < ca.size and ca[i] == code) else 1
    decode = lambda code: wdb_sql._pyval(seg.fetch(col, code))
    rows = workers.take_sorted(decode, count_of, V, lim, {null_code})
    _HITS += 1
    return rows, [wdb_sql._alias(spec['proj'][0])]


def try_valsort(seg, tree, col_map):
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)
