"""
wdb_grid2 -- plain 2-key grouped COUNT(*) as one composite bincount.

The shape GROUP BY k1, k2 [WHERE col = lit] COUNT(*) paid generic argsort grouping
(g-2key in the open; j-mixed-grp's inner fact query through fastjoin). Both keys are
dict codes, so the pair is one composite integer and the whole group-by is a fused
grid pass: per-thread boards, winners decoded through V-sized LUTs, never a sort.
Small grids only (K <= 4M); everything transient.
"""
import numpy as np
import wdb_sql
import wdb_policies as P
import wdb_kernels as WK
import sqlglot.expressions as E

_HITS = 0
_K_CAP = 4_000_000


def detect(seg, tree, col_map):
    if not P.no_joins(tree) or not P.no_having(tree) or not P.no_select_distinct(tree):
        return None
    if tree.args.get('qualify') is not None:
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 2:
        return None
    if not all(isinstance(x, E.Column) for x in g.expressions):
        return None
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    gnames = [sc(x.name) for x in g.expressions]
    proj = tree.expressions
    ki = {}
    ci = None
    for pi, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        ak = wdb_sql._agg_kind(p)
        if ak is not None:
            if ak[0] != 'COUNT_STAR' or ci is not None:
                return None
            ci = pi
        elif isinstance(inner, E.Column) and sc(inner.name) in gnames:
            ki[sc(inner.name)] = pi
        else:
            return None
    if ci is None or len(ki) != 2:
        return None
    w = tree.args.get('where')
    fcol = None; flit = None
    if w is not None:
        cj = w.this
        if isinstance(cj, E.Paren):
            cj = cj.this
        if not isinstance(cj, E.EQ) or not isinstance(cj.expression, E.Literal) \
                or not isinstance(cj.this, E.Column):
            return None
        fcol = sc(cj.this.name)
        flit = cj.expression.this
        fcc = seg.cols.get(fcol)
        if fcc is None or fcc.get('mode') not in (0, 1, 2) or fcc.get('has_null'):
            return None
    K = 1
    for nm in gnames:
        c = seg.cols.get(nm)
        if c is None or c.get('mode') not in (0, 1, 2) or c.get('has_null'):
            return None
        K *= int(c.get('V') or 1 << 30)
    if K > _K_CAP:
        return None
    order = tree.args.get('order')
    if order is not None:
        if len(order.expressions) != 1 or not bool(order.expressions[0].args.get('desc')):
            return None
        onm = wdb_sql._colname(order.expressions[0].this)
        if onm != wdb_sql._alias(proj[ci]):
            return None
    if order is None and wdb_sql._limit(tree) is not None:
        return None                              # unordered LIMIT: the subset choice belongs to the
                                                 # engine's canonical encounter order, not the grid
    if not P.no_deleted_rows(seg):
        return None
    return {'g': gnames, 'ki': ki, 'ci': ci, 'proj': proj, 'fcol': fcol, 'flit': flit,
            'lim': wdb_sql._limit(tree), 'ordered': order is not None}


def execute(seg, spec):
    global _HITS
    import wdb_wherescan
    g1, g2 = spec['g']
    c1 = np.asarray(seg._raw_codes(g1))
    c2 = np.asarray(seg._raw_codes(g2))
    V2 = int(seg.cols[g2]['V'])
    K = int(seg.cols[g1]['V']) * V2
    if spec['fcol'] is not None:
        lit = wdb_wherescan._code_of(seg, spec['fcol'], spec['flit'])
        if lit is None:
            rows = []
            return rows, [wdb_sql._alias(p) for p in spec['proj']]
        fc = np.asarray(seg._raw_codes(spec['fcol']))
    else:
        lit = -1
        fc = c1
    cells = WK.grid2_count(np.ascontiguousarray(c1), np.ascontiguousarray(c2),
                           np.ascontiguousarray(fc),
                           np.int64(lit if lit is not None else -1), np.int64(V2), np.int64(K))
    nz = np.flatnonzero(cells)
    if spec['ordered']:
        nz = nz[np.lexsort((nz, -cells[nz]))]
    if spec['lim'] is not None:
        if spec['ordered'] and spec['lim'] < nz.size \
                and int(cells[nz[spec['lim'] - 1]]) == int(cells[nz[spec['lim']]]):
            return None                          # tie straddles LIMIT: defer for consistency
        nz = nz[:spec['lim']]
    lut1 = {}; lut2 = {}
    rows = []
    ki = spec['ki']; ci = spec['ci']
    i1, i2 = ki[g1], ki[g2]
    for cell in nz.tolist():
        a = cell // V2; b = cell % V2
        if a not in lut1:
            v = wdb_sql._pyval(seg.fetch(g1, a))
            lut1[a] = v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else v
        if b not in lut2:
            v = wdb_sql._pyval(seg.fetch(g2, b))
            lut2[b] = v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else v
        row = [None, None, None]
        row[i1] = lut1[a]; row[i2] = lut2[b]; row[ci] = int(cells[cell])
        rows.append(tuple(row))
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in spec['proj']]
