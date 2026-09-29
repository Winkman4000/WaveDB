"""wdb_cdgroup -- Jackson's bounded hunt for COUNT(DISTINCT u) GROUP BY sparse col.

Q13's lane: visits bound uniques (distinct(u) per group <= group's row count),
so walk groups biggest-first computing exact uniques and STOP when the next
group's visit count falls strictly below the k-th exact answer. The sparse
dress hands the non-default rows (WHERE col <> '' is free); one scatter pass
buckets rows by group; only the k winners' strings are ever decoded."""
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
    # THE DEFAULT'S GROUP (2026-09-29): the planes hand only the NON-default rows, so the lane is
    # exact only when the query itself drops the default -- WHERE key <> '' with '' the default.
    # Without the WHERE it dropped the biggest group (a = 0, 160,025 distinct) from the podium.
    if w is None:
        return None
    n2 = w.this
    if not (isinstance(n2, E.NEQ) and isinstance(n2.this, E.Column)
            and (col_map or {}).get(n2.this.name, n2.this.name) == key
            and isinstance(n2.expression, E.Literal)
            and str(n2.expression.this) == ''):
        return None
    kc = seg.cols.get(key)
    uc = seg.cols.get(ucol)
    if kc is None or uc is None:
        return None
    if uc.get('mode') == 4:
        return None                              # codes are row positions: distinct would count rows
    if kc.get('code_enc') not in (8, 9):
        return None                              # v1: the sparse dress only --
    d9 = int(kc.get('e8d' if kc.get('code_enc') == 8 else 'e9d', -1))
    if d9 < 0:
        return None
    dv9 = seg.fetch(key, d9)
    if isinstance(dv9, (bytes, bytearray)):
        dv9 = dv9.decode('utf-8', 'replace')
    if dv9 != '':
        return None                              # the default is a real value: its rows are a group
    ox = tree.args.get('order')                  # its planes ARE the row list
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
    return {'key': key, 'ucol': ucol, 'aggs': aggs, 'lim': lim,
            'proj': tree.expressions}


def execute(seg, spec):
    global _HITS
    key, ucol, k = spec['key'], spec['ucol'], spec['lim']
    pl = seg.e8_planes(key)
    if pl is None:
        return None
    pos8, lits8 = pl[0], pl[1]
    lits8 = np.asarray(lits8, np.int64)
    KV = int(seg.cols[key]['V'])
    import wdb_kernels as WK
    counts = np.bincount(lits8, minlength=KV)
    uc = np.asarray(seg._raw_codes(ucol))[np.asarray(pos8)].astype(np.int64)
    offs = np.zeros(KV + 1, np.int64)
    np.cumsum(counts, out=offs[1:])
    bucketed = np.empty(lits8.size, np.int64)
    WK.cd_scatter(lits8, uc, offs, offs[:-1].copy(), bucketed)
    big = np.argsort(counts, kind='stable')[::-1].astype(np.int64)
    bc, bd, filled = WK.cd_hunt(bucketed, offs, big, counts.astype(np.int64),
                                np.int64(k))
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
