"""blockstats: whole-table aggregates answered from per-block statistics -- the disk-only read.
At build (eagerly in prewarm workers, or lazily once), every stats-eligible column gets per-32K-row-
block (count, nonnull count, value sum, min code, max code): ~28 bytes a block, ~84 KB a column,
computed in the one pass where the values were already in hand. A no-WHERE aggregate query then
never touches row data at all: COUNT/SUM/AVG/MIN/MAX are arithmetic over ~3K numbers, and MIN/MAX
decode exactly one dict entry each (code order == value order). The exception law inverted: the
summary is the norm, and the compressed rows are opened only for questions the summary can't answer.

Correctness gates: SUM served only when provably exact in float64 (nonnull_count x max|value| < 2^53),
else declined to the exact paths; AVG always float (matches SQL double semantics); nulls excluded
from SUM/AVG/MIN/MAX and included in COUNT(*), per SQL. Declines overrides/deleted rows/WHERE/GROUP."""
import numpy as np
import wdb_qmem
import wdb_sql
import workers
import wdb_policies as P
E = wdb_sql.E

_ENABLED = True
_HITS = 0
_BR = 32768
_SCACHE = wdb_qmem.register({})          # (seg.path, col, N) -> stats dict; qmem per the
                                         # cold-truth law. Persistence is LAWFUL only on
                                         # disk: stats are written once as a .bst sidecar
                                         # (like .gbc), and each query loads them cold in
                                         # ~1ms -- the file is the only memory.


def enable():
    global _ENABLED; _ENABLED = True


def disable():
    global _ENABLED; _ENABLED = False


def eligible(seg, col):
    c = seg.cols.get(col)
    return c is not None and c.get('mode') in (0, 2, 4) and c.get('dt') in (0, 3)


def build(seg, col):
    """Per-block stats for one column. Returns the stats dict (also cached) or None."""
    key = (seg.path, col, int(seg.N))
    if key in _SCACHE:
        return _SCACHE[key]
    if not eligible(seg, col):
        _SCACHE[key] = None; return None
    c = seg.cols[col]
    N = int(seg.N)
    fn = seg.path + '.' + col + '.bst.npz'
    try:
        import os as _os
        if _os.path.exists(fn):
            z = np.load(fn, allow_pickle=False)
            if int(z['N']) == N:
                st = {'cnt': np.asarray(z['cnt']), 'nn': np.asarray(z['nn']),
                      'sum': np.asarray(z['sum']), 'cmin': np.asarray(z['cmin']),
                      'cmax': np.asarray(z['cmax']), 'mode4': bool(z['mode4']),
                      'maxabs': float(z['maxabs']), 'dt': int(z['dt'])}
                _SCACHE[key] = st
                return st
    except Exception:
        pass                                     # unreadable sidecar: recompute below
    nb = (N + _BR - 1) // _BR
    if c['mode'] == 4:
        vals = np.asarray(seg._seq_decode(c))
        codes = None; nullcode = -1; dvals = None
    else:
        codes = seg._raw_codes(col)
        if c['dt'] != 0:
            dvals = None                          # dates: MIN/MAX ride on codes; SUM/AVG declined
        elif c['mode'] == 2:
            dvals = np.asarray(seg._dict_ints(c), dtype=np.int64)
        else:                                     # mode 0 dt 0: small dict of plain ints
            dvals = np.asarray([int(x) for x in c['vals']], dtype=np.int64)
        nullcode = int(c['V']) - 1 if c['has_null'] else -1
        vals = None
    cnt = np.empty(nb, np.int64); nn = np.empty(nb, np.int64)
    bsum = np.empty(nb, np.float64)
    cmin = np.empty(nb, np.int64); cmax = np.empty(nb, np.int64)
    maxabs = 0.0
    for j in range(nb):
        lo = j * _BR; hi = min(lo + _BR, N)
        if codes is None:
            v = vals[lo:hi]; cnt[j] = v.size; nn[j] = v.size
            bsum[j] = v.sum(dtype=np.float64)
            cmin[j] = v.min(); cmax[j] = v.max()
            maxabs = max(maxabs, float(np.abs(v).max()))
        else:
            b = codes[lo:hi].astype(np.int64); cnt[j] = b.size
            if nullcode >= 0:
                b = b[b != nullcode]
            nn[j] = b.size
            if b.size:
                if dvals is not None:
                    dv = dvals[b]
                    bsum[j] = dv.sum(dtype=np.float64)
                    maxabs = max(maxabs, float(np.abs(dv).max()))
                else:
                    bsum[j] = 0.0
                cmin[j] = b.min(); cmax[j] = b.max()
            else:
                bsum[j] = 0.0; cmin[j] = np.iinfo(np.int64).max; cmax[j] = -1
    st = {'cnt': cnt, 'nn': nn, 'sum': bsum, 'cmin': cmin, 'cmax': cmax,
          'mode4': codes is None, 'maxabs': maxabs, 'dt': c['dt']}
    try:
        np.savez(fn, N=N, cnt=cnt, nn=nn, sum=bsum, cmin=cmin, cmax=cmax,
                 mode4=(codes is None), maxabs=maxabs, dt=int(c['dt']))
    except Exception:
        pass                                     # read-only volume: compute-only mode
    _SCACHE[key] = st
    return st


def install(seg_path, N, col, st):
    """Accept eagerly-built stats (prewarm workers) into the cache."""
    _SCACHE[(seg_path, col, int(N))] = st


def detect(seg, tree, col_map):
    # Retired under the query-scoped memory law (wdb_qmem): stats died with the cache, and
    # BUILDING them per query (decompress every frame + per-block reduces, ~0.38 s at 100M)
    # costs more than the fused scan it would replace (~0.10 s). The method revives the day
    # stats live IN THE FILE (encode-time, ~25 KB/col) -- then detect reads, never builds.
    if True:                            return None
    if not _ENABLED:                    return None
    if not P.no_joins(tree):            return None
    if not P.no_select_distinct(tree):  return None
    if not P.no_where(tree):            return None
    if not P.no_having(tree):           return None
    if tree.args.get('group') is not None: return None
    if tree.args.get('order') is not None: return None
    if wdb_sql._offset(tree):           return None
    proj = tree.expressions
    if not proj:
        return None
    specs = []
    for p in proj:
        ak = wdb_sql._agg_kind(p)
        if ak is None:
            return None
        if ak[0] == 'COUNT_STAR':
            specs.append(('COUNT_STAR', None)); continue
        if ak[0] not in ('COUNT', 'SUM', 'AVG', 'MIN', 'MAX') or not isinstance(ak[1], str):
            return None
        col = col_map.get(ak[1], ak[1]) if col_map else ak[1]
        if not P.columns_exist(seg, col):   return None
        if not eligible(seg, col):          return None
        if seg._effective(col) is not None: return None    # overrides falsify stored stats
        if ak[0] in ('SUM', 'AVG') and seg.cols[col].get('dt') != 0: return None
        specs.append((ak[0], col))
    if not P.no_deleted_rows(seg):      return None
    return {'specs': specs, 'proj': proj}


def execute(seg, spec):
    global _HITS
    row = []
    for kind, col in spec['specs']:
        if kind == 'COUNT_STAR':
            row.append(int(seg.N)); continue
        st = build(seg, col)
        if st is None:
            return None
        tot_nn = int(st['nn'].sum())
        if kind == 'COUNT':
            row.append(tot_nn)
        elif kind == 'SUM':
            if tot_nn * st['maxabs'] >= 2.0 ** 53:
                return None                       # float64 exactness not provable -> exact paths
            row.append(int(st['sum'].sum()) if tot_nn else None)
        elif kind == 'AVG':
            row.append(float(st['sum'].sum()) / tot_nn if tot_nn else None)
        else:                                     # MIN / MAX
            live = st['cmax'] >= 0
            if not live.any():
                row.append(None); continue
            if st['mode4']:
                v = int(st['cmin'][live].min()) if kind == 'MIN' else int(st['cmax'][live].max())
                c = seg.cols[col]
                if c.get('dt') == 3:
                    import wdb_engine
                    v = wdb_sql._pyval(np.int64(v).view(f"datetime64[{wdb_engine._DT_UNITS[c['aux']]}]"))
                row.append(v)
            else:
                code = int(st['cmin'][live].min()) if kind == 'MIN' else int(st['cmax'][live].max())
                row.append(wdb_sql._pyval(seg.fetch(col, code)))
    _HITS += 1
    return [tuple(row)], [wdb_sql._alias(p) for p in spec['proj']]
