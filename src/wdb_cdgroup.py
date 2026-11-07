"""wdb_cdgroup -- Jackson's bounded hunt for COUNT(DISTINCT u) GROUP BY key WHERE key <> lit.

Visits (rows) bound uniques (distinct(u) per group <= group's row count),
so walk groups biggest-first computing exact uniques and STOP when the next
group's visit count falls strictly below the k-th exact answer. The row list
is `key <> lit` (free from the planes when lit is a sparse key's default); one scatter pass
buckets rows by group; only the k winners' strings are ever decoded.

Lore: ClickBench Q13's lane."""
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') or tree.args.get('distinct'):
        return None
    if tree.args.get('having') is not None:
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 1 or not isinstance(g.expressions[0], E.Column):
        return None
    key = (col_map or {}).get(g.expressions[0].name, g.expressions[0].name)
    ucol = None
    aggs = []
    alias_cd = None
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            if (col_map or {}).get(inner.name, inner.name) != key:
                return None
            aggs.append(('KEY',))
            continue
        if isinstance(inner, E.Count) and inner.args.get('distinct') is None \
                and isinstance(inner.this, E.Distinct):
            exprs = inner.this.expressions
            if len(exprs) != 1 or not isinstance(exprs[0], E.Column):
                return None
            ucol = (col_map or {}).get(exprs[0].name, exprs[0].name)
            aggs.append(('CD',))
            if isinstance(p, E.Alias):
                alias_cd = p.alias
            continue
        return None
    if ucol is None:
        return None
    w = tree.args.get('where')
    # THE ROW LIST IS `key <> lit`, any literal of the key's kind: wdb_wherescan.neq_rows hands it (the planes, free,
    # when the key is stored sparse and its default IS lit; else one pass over the codes; NULL keys never).
    # lore: v1 (2026-09-29) served only WHERE key <> '' with '' the sparse default -- without the WHERE it dropped
    # the biggest group (a = 0, 160,025 distinct) from the podium. Widened 2026-11-07 ("make them work for any").
    if w is None:
        return None
    n2 = w.this
    if not (isinstance(n2, E.NEQ) and isinstance(n2.this, E.Column)
            and (col_map or {}).get(n2.this.name, n2.this.name) == key):
        return None
    import wdb_wherescan as _WS
    if _WS._litval(n2.expression) is None:
        return None
    kc = seg.cols.get(key)
    uc = seg.cols.get(ucol)
    if kc is None or uc is None:
        return None
    if uc.get('mode') == 4:
        return None                              # codes are row positions: distinct would count rows
    if seg._overrides(ucol) is not None:
        return None                              # the raw codes would miss the overriding values
    if uc.get('has_null') and uc.get('mode') not in (0, 1, 2):
        return None                              # NULL's code is known (V-1) only in the dictionary modes
    ox = tree.args.get('order')
    lim = None
    lx = tree.args.get('limit')
    if lx is not None:
        try:
            lim = int(lx.expression.this)
        except Exception:
            return None
    if ox is None or lim is None or len(ox.expressions) != 1:
        return None                              # the hunt needs ORDER cd DESC LIMIT k
    o = ox.expressions[0]
    if not o.args.get('desc') or not isinstance(o.this, E.Column):
        return None
    if alias_cd is None or o.this.name != alias_cd:
        return None
    return {'key': key, 'ucol': ucol, 'aggs': aggs, 'lim': lim, 'lit': n2.expression,
            'proj': tree.expressions}


def execute(seg, spec):
    global _HITS
    key, ucol, k = spec['key'], spec['ucol'], spec['lim']
    import wdb_wherescan as _WS
    rl = _WS.neq_rows(seg, key, spec['lit'])
    if rl is None:
        return None                              # not a code filter (a literal of the other kind, a time key, ...)
    pos8, lits8 = rl
    KV = int(seg.cols[key]['V'])
    import wdb_kernels as WK
    uc = np.asarray(seg._raw_codes(ucol))[pos8].astype(np.int64)
    c9 = seg.cols[ucol]
    all9 = None
    if c9.get('has_null'):                       # COUNT(DISTINCT u) counts no NULL: its rows leave the list
        nn9 = uc != int(c9['V']) - 1
        if not nn9.all():
            all9 = lits8                         # (a group of only NULL u still answers 0)
            lits8, uc = lits8[nn9], uc[nn9]
    counts = np.bincount(lits8, minlength=KV)
    offs = np.zeros(KV + 1, np.int64)
    np.cumsum(counts, out=offs[1:])
    bucketed = np.empty(lits8.size, np.int64)
    WK.cd_scatter(lits8, uc, offs, offs[:-1].copy(), bucketed)
    big = np.argsort(counts, kind='stable')[::-1].astype(np.int64)
    bc, bd, filled = WK.cd_hunt(bucketed, offs, big, counts.astype(np.int64),
                                np.int64(k))
    if int(filled) < k and all9 is not None and np.unique(all9).size > int(filled):
        return None                              # the podium reaches the 0-distinct groups the hunt never visits
    best = sorted(((int(bd[t]), int(bc[t])) for t in range(int(filled))),
                  key=lambda t: (-t[0], t[1]))
    rows = []
    for d, code in best[:k]:
        v = seg.fetch(key, int(code))            # only the winners' strings, ever
        if isinstance(v, (bytes, bytearray)):
            v = v.decode('utf-8', 'replace')
        row = []
        for kind in spec['aggs']:
            row.append(v if kind[0] == 'KEY' else d)
        rows.append(tuple(row))
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in spec['proj']]
