"""wdb_valtopk: ordered row dumps without sorting the world.

SELECT <cols> FROM t ORDER BY <numeric/temporal col(s)> LIMIT k was falling to the general
scan -- a full materialization (137 s on 100M) for 100 rows. The answer's shape: one
np.partition finds the k-th boundary value of the primary key, the candidate set is
everything at-or-inside the boundary (ties included), a lexsort over CANDIDATES ONLY
settles the full key order, and only the k winning rows ever decode. No WHERE (v1),
plain-column projections, numeric or temporal keys.
"""
import numpy as np
import sqlglot.expressions as E
import wdb_sql
import wdb_policies as P

_K_CAP = 100_000
_CAND_CAP = 5_000_000


def _order_keys(tree):
    o = tree.args.get('order')
    if o is None:
        return None
    out = []
    for oe in o.expressions:
        if not isinstance(oe.this, E.Column):
            return None
        out.append((oe.this.name, bool(oe.args.get('desc'))))
    return out


def detect(seg, tree, col_map):
    if not P.no_joins(tree) or not P.no_select_distinct(tree) or not P.no_having(tree):
        return None
    if tree.args.get('group') is not None or tree.args.get('qualify') is not None:
        return None
    if tree.args.get('where') is not None:
        return None                              # v1: bare dumps (the trench shape)
    if wdb_sql._offset(tree):
        return None
    lim = wdb_sql._limit(tree)
    if lim is None or not (1 <= lim <= _K_CAP):
        return None
    keys = _order_keys(tree)
    if not keys:
        return None
    cmap = col_map or {}
    resolve = lambda n: cmap.get(n, n)
    proj = tree.expressions
    if not proj:
        return None
    pcols = []
    for p in proj:
        if wdb_sql._agg_kind(p) is not None:
            return None
        nm = wdb_sql._proj_colname(p)
        if nm is None:
            return None
        c = resolve(nm)
        if not P.columns_exist(seg, c) or seg.cols[c].get('mode') not in (0, 1, 2):
            return None
        pcols.append(c)
    rkeys = []
    for kn, kd in keys:
        c = resolve(kn)
        if not P.columns_exist(seg, c):
            return None
        cc = seg.cols[c]
        if cc.get('dt') not in (0, 3) or cc.get('mode') not in (0, 1, 2) or cc.get('has_null'):
            return None
        rkeys.append((c, kd))
    return {'pcols': pcols, 'proj': proj, 'keys': rkeys, 'lim': lim}


def execute(seg, spec):
    import wdb_window as WN
    keys, lim = spec['keys'], spec['lim']
    kcol, kdesc = keys[0]
    v0 = WN._numvals(seg, kcol, exact_int=True)
    N = v0.size
    k = min(lim, N)
    if k == N:
        cand_idx = np.arange(N)
    else:
        if kdesc:
            thr = np.partition(v0, N - k)[N - k]
            cand = v0 >= thr
        else:
            thr = np.partition(v0, k - 1)[k - 1]
            cand = v0 <= thr
        cand_idx = np.nonzero(cand)[0]
        if cand_idx.size > _CAND_CAP:
            return None                          # boundary tie explosion: fall through
    arrs = []
    for c, d in keys:
        a = v0[cand_idx] if c == kcol else WN._numvals(seg, c, exact_int=True)[cand_idx]
        if a.dtype.kind == 'f':
            arrs.append(-a if d else a)
        else:
            arrs.append(-a.astype(np.int64) if d else a.astype(np.int64))
    order = np.lexsort(list(reversed(arrs)))     # primary decides first
    idx = cand_idx[order[:k]]
    # decode only the winners
    outcols = []
    for c in spec['pcols']:
        codes = np.asarray(seg._raw_codes(c)).astype(np.int64)[idx]
        cc = seg.cols[c]
        cache = {}
        vals = []
        for code in codes:
            code = int(code)
            v = cache.get(code, cache)
            if v is cache:
                v = seg.fetch(c, code)
                if cc.get('dt') == 3 and isinstance(v, (int, np.integer)):
                    v = np.int64(v).view(f"datetime64[{seg.unit(c)}]")
                v = wdb_sql._pyval(v)
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode('utf-8', 'replace')
                cache[code] = v
            vals.append(v)
        outcols.append(vals)
    rows = [tuple(oc[i] for oc in outcols) for i in range(len(idx))]
    return rows, [wdb_sql._alias(p) for p in spec['proj']]
