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


def _counts_lane(seg, kcol, kdesc, need):
    """Jackson's scale-first cut: the k-th boundary code from shelved counts (gbc), then
    only blocks whose cmax/cmin admit a candidate are decompressed. The full code stream
    is never read. mode-2 keys only (int dict: code order == value order, provably)."""
    import os
    import wdb_gbcount
    import wdb_blockstats as BS
    import wdb_window as WN
    if not os.path.exists(seg.path + '.' + kcol + '.gbc'):
        return None                              # peek only: the lane must never lazy-build
    t = np.asarray(WN._int_table(seg, kcol), dtype=np.int64)
    if t.size < 2 or not bool(np.all(np.diff(t) >= 0)):
        return None                              # code order must PROVABLY equal value order
    got = wdb_gbcount._load(seg, kcol)
    if got is None:
        return None
    hc, hn = got
    V = int(seg.cols[kcol]['V'])
    cn = np.zeros(V + 1, dtype=np.int64)
    cn[np.asarray(hc, dtype=np.int64)] = np.asarray(hn, dtype=np.int64)
    cn = cn[:V]
    if kdesc:
        j = V - 1 - int(np.searchsorted(np.cumsum(cn[::-1]), need))
    else:
        j = int(np.searchsorted(np.cumsum(cn), need))
    st = BS.build(seg, kcol)
    if st is None:
        return None
    cmax, cmin = np.asarray(st['cmax']), np.asarray(st['cmin'])
    touched = np.flatnonzero(cmax >= j) if kdesc else np.flatnonzero(cmin <= j)
    c = seg.cols[kcol]
    BR = c['BR']; base = c['cstart']; bo = c['boffs']
    wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
    from wdb_blockstats import _BR as SBR
    if BR != SBR:
        # stats blocks and storage blocks share the grid only when _BR == BR;
        # translate stats blocks -> storage blocks conservatively
        ratio = SBR / BR
        sb = set()
        for b in touched.tolist():
            lo = int(b * ratio); hi = int(((b + 1) * SBR - 1) // BR)
            for x in range(lo, min(hi + 1, int(bo.size) - 1)):
                sb.add(x)
        touched = np.array(sorted(sb), dtype=np.int64)
    import zstandard as _zs
    tl = touched.tolist()
    parts = [None] * len(tl)
    def _blk(i):                                 # zstd drops the GIL: 8 lanes
        b = tl[i]
        raw = _zs.ZstdDecompressor().decompress(
            seg.buf[base + int(bo[b]):base + int(bo[b + 1])].tobytes())
        codes = np.frombuffer(raw, dtype=wdt)
        loc = np.flatnonzero(codes >= j) if kdesc else np.flatnonzero(codes <= j)
        if loc.size:
            parts[i] = (loc.astype(np.int64) + b * BR, codes[loc].astype(np.int64))
    if len(tl) > 4:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as _ex:
            list(_ex.map(_blk, range(len(tl))))
    else:
        for i in range(len(tl)):
            _blk(i)
    pos_parts = [p[0] for p in parts if p is not None]
    code_parts = [p[1] for p in parts if p is not None]
    if not pos_parts:
        return None
    cand_idx = np.concatenate(pos_parts)
    cand_codes = np.concatenate(code_parts)
    if cand_idx.size > _CAND_CAP:
        return None
    return cand_idx, cand_codes


def execute(seg, spec):
    import wdb_window as WN
    keys, lim = spec['keys'], spec['lim']
    kcol, kdesc = keys[0]
    N = int(seg.N)
    k = min(lim, N)
    codes0 = None
    cand = None
    if k < N and len(keys) == 1 and seg.cols[kcol].get('mode') in (0, 1, 2) \
            and seg.cols[kcol].get('dt') == 0 and seg.cols[kcol].get('code_enc') == 3:
        cand = _counts_lane(seg, kcol, kdesc, k)
    if cand is not None:
        cand_idx, cand_codes = cand
    elif k == N:
        codes0 = np.asarray(seg._raw_codes(kcol)).astype(np.int64)
        cand_idx = np.arange(N)
        cand_codes = codes0
    else:
        # codes are ranks: the k-th boundary VALUE is the k-th boundary CODE -- one
        # bincount + cumsum on the board replaces np.partition over 100M materialized
        # values (2.3s -> ~0.3s; the primary's values are never built at all)
        codes0 = np.asarray(seg._raw_codes(kcol)).astype(np.int64)
        V = int(seg.cols[kcol]['V'])
        cn = np.bincount(codes0, minlength=V)
        if kdesc:
            j = int(np.searchsorted(np.cumsum(cn[::-1]), k))
            cand = codes0 >= (V - 1 - j)
        else:
            j = int(np.searchsorted(np.cumsum(cn), k))
            cand = codes0 <= j
        cand_idx = np.nonzero(cand)[0]
        cand_codes = codes0[cand_idx]
        if cand_idx.size > _CAND_CAP:
            return None                          # boundary tie explosion: fall through
    arrs = []
    for c, d in keys:
        if c == kcol:
            a = cand_codes                       # rank order == value order, ties == ties
        else:
            a = WN._numvals(seg, c, exact_int=True)[cand_idx]
        if a.dtype.kind == 'f':
            arrs.append(-a if d else a)
        else:
            arrs.append(-a.astype(np.int64) if d else a.astype(np.int64))
    order = np.lexsort(list(reversed(arrs)))     # primary decides first
    idx = cand_idx[order[:k]]
    # decode only the winners
    outcols = []
    for c in spec['pcols']:
        cc = seg.cols[c]
        if c == kcol:
            codes = cand_codes[order[:k]]
        else:
            # the winners' name tags only: block-targeted fetch of |idx| positions
            # instead of decompressing the column's full code stream (0.75s -> ~0.02s
            # for LIMIT 100). codes_at wants ascending positions; unsort after.
            srt = np.argsort(idx, kind='stable')
            got = np.asarray(seg.codes_at(c, idx[srt])).astype(np.int64)
            codes = np.empty_like(got)
            codes[srt] = got
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
