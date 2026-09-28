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
try:
    import sqlglot.expressions as E
except Exception:
    E = None
import wdb_qmem
import wdb_sql
import wdb_sidecar
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


def compute(seg, col):
    """The block statistics of one column, computed with the kernel (blocks across threads).
    Returns the stats dict: cnt, nn, sum (float64 of dictionary values), cmin, cmax (codes),
    mode4, maxabs (an upper bound: the dictionary's largest magnitude), dt."""
    import wdb_kernels as _WK
    c = seg.cols[col]
    N = int(seg.N)
    nb = (N + _BR - 1) // _BR
    if c['mode'] == 4:
        codes = np.ascontiguousarray(np.asarray(seg._seq_decode(c)), dtype=np.int64)
        dvals = np.zeros(1, np.int64); mode = 2; nullcode = np.int64(-1)
        maxabs = float(np.abs(codes).max()) if codes.size else 0.0
    else:
        codes = np.ascontiguousarray(np.asarray(seg._raw_codes(col)))
        if c['dt'] != 0:
            dvals = np.zeros(1, np.int64); mode = 0; maxabs = 0.0    # dates: MIN/MAX ride on codes
        else:
            if c['mode'] == 2:
                dvals = np.ascontiguousarray(np.asarray(seg._dict_ints(c), dtype=np.int64))
            else:                                                     # mode 0 dt 0: plain-int dict
                dvals = np.asarray([int(x) for x in c['vals']], dtype=np.int64)
            mode = 1
            maxabs = float(np.abs(dvals).max()) if dvals.size else 0.0
        nullcode = np.int64(int(c['V']) - 1) if c['has_null'] else np.int64(-1)
        if mode == 1 and dvals.size < int(c['V']):                    # the null code (skipped) still indexes
            dvals = np.concatenate([dvals, np.zeros(int(c['V']) - dvals.size, np.int64)])
    cnt = np.empty(nb, np.int64); nn = np.empty(nb, np.int64)
    bsum = np.empty(nb, np.float64)
    cmin = np.empty(nb, np.int64); cmax = np.empty(nb, np.int64)
    if nb:
        _WK.block_stats(codes, dvals, np.int64(mode), nullcode, np.int64(_BR), cnt, nn, bsum, cmin, cmax)
    return {'cnt': cnt, 'nn': nn, 'sum': bsum, 'cmin': cmin, 'cmax': cmax,
            'mode4': c['mode'] == 4, 'maxabs': maxabs, 'dt': int(c['dt'])}


def stats_path(seg_path):
    """THE STATISTICS OF THE LOAD: one file beside the segment, DATA (not a sidecar) -- written by
    the encoder, counted in load time, kept by `wdb sidecars drop`, ignored by the sentinel."""
    return seg_path + '.stats.npz'


_LOADED = {}                                     # seg path -> (mtime_ns, npz) -- the file is mmap-light


def _from_load(seg, col):
    p = stats_path(seg.path)
    import os as _os
    try:
        mt = _os.stat(p).st_mtime_ns
    except OSError:
        return None
    hit = _LOADED.get(p)
    if hit is None or hit[0] != mt:
        try:
            hit = _LOADED[p] = (mt, np.load(p, allow_pickle=False))
        except Exception:
            return None
    z = hit[1]
    k = col + '.'
    if (k + 'cnt') not in z.files or int(z['N']) != int(seg.N):
        return None
    return {'cnt': np.asarray(z[k + 'cnt']), 'nn': np.asarray(z[k + 'nn']), 'sum': np.asarray(z[k + 'sum']),
            'cmin': np.asarray(z[k + 'cmin']), 'cmax': np.asarray(z[k + 'cmax']),
            'mode4': bool(z[k + 'mode4']), 'maxabs': float(z[k + 'maxabs']), 'dt': int(z[k + 'dt'])}


def differentiator_rows(seg, col):
    """THE EXCEPTION LIST of a near-unique column (Jackson, 2026-09-20: a column statistic, the same
    species as a distinct count or a discovered unique constraint -- not a query's answer). For a
    dictionary column with V >= N/2 and V >= 1024: the rows whose code occurs more than once, when
    they are under 0.75% of V (WatchID: 4 rows in 100M). None when the column does not qualify."""
    import wdb_kernels as _WK
    c = seg.cols.get(col)
    if c is None or c.get('mode') not in (0, 1, 2):
        return None
    V = int(c.get('V') or 0); N = int(seg.N)
    if V * 2 < N or V < 1024:
        return None
    codes = np.asarray(seg._raw_codes(col))
    cnt = _WK.bincount_par(codes, V)
    rep = np.flatnonzero(cnt[codes] >= 2)
    if rep.size >= 0.0075 * V:
        return None
    return rep.astype(np.uint32)


VCNT_MAX = 65536


def value_counts(seg, col):
    """THE CENSUS OF THE LOAD (Jackson, 2026-09-24): rows per dictionary code, for dictionary
    columns of at most VCNT_MAX codes -- a column statistic of the same species as a distinct
    count, born with the load, a few hundred bytes. COUNT(*) WHERE k <op> v and GROUP BY k
    COUNT(*) become arithmetic over V numbers instead of a pass over N rows: Q01 (AdvEngineID <> 0)
    is N minus the count of code 0. None when the column does not qualify."""
    c = seg.cols.get(col)
    if c is None or c.get('mode') not in (0, 1, 2):
        return None
    V = int(c.get('V') or 0)
    if V < 1 or V > VCNT_MAX:
        return None
    import wdb_kernels as _WK
    codes = np.asarray(seg._raw_codes(col))
    cnt = np.ascontiguousarray(_WK.bincount_par(codes, V), dtype=np.int64)
    assert cnt.size == V and int(cnt.sum()) == int(seg.N), ('census does not cover the rows', col, cnt.size, V)
    return cnt


def _load_npz(seg):
    p = stats_path(seg.path)
    import os as _os
    try:
        mt = _os.stat(p).st_mtime_ns
    except OSError:
        return None
    hit = _LOADED.get(p)
    if hit is None or hit[0] != mt:
        try:
            hit = _LOADED[p] = (mt, np.load(p, allow_pickle=False))
        except Exception:
            return None
    z = hit[1]
    if int(z['N']) != int(seg.N):
        return None
    return z


SIGNPOST_EVERY = 128


def e19_signposts(seg, col):
    """THE SIGNPOSTS (Jackson, 2026-09-27): for a block-dictionary (tag 19) column, every 128th entry
    of each block's sorted dictionary kept whole -- born with the load beside the census, ~1/32 the
    size of the dictionaries themselves (UserID: 1.08 MB beside 58.1 MB). 'Is code k in this block?'
    becomes a halving among ~150 signposts and at most 127 gaps, instead of walking ~20,000 entries:
    Q19 (UserID = k) touches one block of 1,526 by its label, never the other 1,525. Returns
    (sp, spo): the signposts of block b are sp[spo[b]:spo[b + 1]]. None for any other column."""
    c = seg.cols.get(col)
    if c is None or c.get('code_enc') != 19 or c.get('e19dc') is None or 'e19R' in c:
        return None                                  # (shelved labels: a lookup already reads one shelf)
    import wdb_kernels as _WK
    pw, dw = seg._e19_words(c)
    dc = np.asarray(c['e19dc'], np.int64)
    S = SIGNPOST_EVERY
    spo = np.zeros(dc.size + 1, np.int64)
    np.cumsum((dc + S - 1) // S, out=spo[1:])
    sp = np.zeros(int(spo[-1]), np.int64)
    _WK.e19_signposts(dw, np.int64(c['e19bits']), c['e19gw'], dc, c['e19doff'], np.int64(S), spo, sp)
    return (sp.astype(np.uint32) if int(c.get('V') or 0) < (1 << 32) else sp), spo


_SPC = wdb_qmem.register_tier1({})              # (stats path, mtime, col) -> (sp int64, spo, S): the load's
                                                 # signposts decoded (tier 1: kept for hot runs, WDB_HOT_KEEP=0 drops)


def signposts_from_load(seg, col):
    """the load's signposts of a tag-19 column as (sp int64, spo, S), or None (absent, or the shape does
    not match the column's blocks)"""
    z = _load_npz(seg)
    if z is None or (col + '.sp19') not in z.files:
        return None
    key = (stats_path(seg.path), _LOADED[stats_path(seg.path)][0], col)
    hit = _SPC.get(key)
    if hit is not None:
        return hit
    c = seg.cols.get(col)
    if c is None or c.get('code_enc') != 19 or c.get('e19dc') is None or 'e19R' in c:
        return None
    sp = np.asarray(z[col + '.sp19'], np.int64); spo = np.asarray(z[col + '.sp19o'], np.int64)
    S = int(z[col + '.sp19s']) if (col + '.sp19s') in z.files else SIGNPOST_EVERY
    dc = np.asarray(c['e19dc'], np.int64)
    if spo.size != dc.size + 1 or int(spo[-1]) != sp.size or not np.array_equal(np.diff(spo), (dc + S - 1) // S):
        return None
    if len(_SPC) > 16:
        _SPC.clear()
    _SPC[key] = hit = (sp, spo, S)
    return hit


def vcnt_from_load(seg, col):
    """the load's rows-per-code of a column, or None (absent, or the shape does not match)"""
    z = _load_npz(seg)
    k = col + '.vcnt'
    if z is None or k not in z.files:
        return None
    a = np.asarray(z[k], dtype=np.int64)
    c = seg.cols.get(col)
    if c is None or a.size != int(c['V']):
        return None
    return a


def rep_from_load(seg, col):
    """the exception list from the load statistics, or None (absent or the column did not qualify)"""
    p = stats_path(seg.path)
    import os as _os
    try:
        mt = _os.stat(p).st_mtime_ns
    except OSError:
        return None
    hit = _LOADED.get(p)
    if hit is None or hit[0] != mt:
        try:
            hit = _LOADED[p] = (mt, np.load(p, allow_pickle=False))
        except Exception:
            return None
    z = hit[1]
    k = col + '.rep'
    if k not in z.files or int(z['N']) != int(seg.N):
        return None
    return np.asarray(z[k], dtype=np.int64)


def write_for_segment(seg_path, verbose=False):
    """Compute and write the block statistics of every eligible column of a segment (the encoder's
    last step) -- and the exception lists of its differentiator columns. Returns the number of
    columns written."""
    import os as _os
    from wdb_engine import Segment
    seg = Segment(seg_path)
    out = {'N': np.int64(seg.N)}
    n = 0
    for col in seg.order:
        rep = differentiator_rows(seg, col)  # FAIL-LOUD: eligibility is decided inside, never by an exception
        if rep is not None:
            out[col + '.rep'] = rep
            if verbose: print('  stats: %s is a differentiator, %d exception rows' % (col, rep.size), flush=True)
        vc = value_counts(seg, col)          # THE CENSUS OF THE LOAD (small dictionaries)
        if vc is not None:
            out[col + '.vcnt'] = vc
        spp = e19_signposts(seg, col)        # THE SIGNPOSTS (block-dictionary columns)
        if spp is not None:
            out[col + '.sp19'], out[col + '.sp19o'] = spp
            out[col + '.sp19s'] = np.int64(SIGNPOST_EVERY)
        seg._codes.pop(col, None)
        if not eligible(seg, col):
            continue
        try:
            st = compute(seg, col)
        except Exception as ex:
            if verbose: print('  stats: %s declined (%s)' % (col, ex), flush=True)
            continue
        for kk in ('cnt', 'nn', 'sum', 'cmin', 'cmax'):
            out[col + '.' + kk] = st[kk]
        out[col + '.mode4'] = np.bool_(st['mode4']); out[col + '.maxabs'] = np.float64(st['maxabs'])
        out[col + '.dt'] = np.int64(st['dt'])
        seg._codes.pop(col, None); n += 1
    p = stats_path(seg_path)
    tmp = p + '.partial.npz'
    np.savez(tmp, **out)
    _os.replace(tmp, p)
    _LOADED.pop(p, None)
    if verbose: print('  stats: %d columns, %.1f KB -> %s' % (n, _os.path.getsize(p) / 1024, _os.path.basename(p)), flush=True)
    return n


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
    # THE STATS ON THE SHELF: the per-query flush (qmem) forgets _SCACHE by law, and re-reading the
    # .npz each query cost 4 ms of Q06's 6 (a zip of nine arrays on the network volume). The shelf is
    # the lawful resident form -- bounded, evictable -- and 3,052 blocks x 6 arrays is 150 KB
    import wdb_shelf
    try: _mt = __import__('os').stat(seg.path).st_mtime_ns
    except Exception: _mt = 0
    _sk = ('bst', seg.path, col, N, _mt)               # the segment's identity is in the key
    st = wdb_shelf.SHELF.get(_sk)
    if st is not None:
        _SCACHE[key] = st
        return st
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
                try: wdb_shelf.SHELF.put(_sk, st, int(sum(v.nbytes for v in st.values() if hasattr(v, 'nbytes'))), kind='block-stats')
                except Exception: pass
                return st
    except Exception:
        pass                                     # unreadable sidecar: recompute below
    # THE STATISTICS OF THE LOAD (Jackson, B): the encoder writes every eligible column's block
    # stats beside the segment as DATA (<seg>.stats.npz) -- metadata per block, a few bytes each,
    # counted in load time, never born by a query. Read before any birth is considered.
    st = _from_load(seg, col)
    if st is not None:
        _SCACHE[key] = st
        try: wdb_shelf.SHELF.put(_sk, st, int(sum(v.nbytes for v in st.values() if hasattr(v, 'nbytes'))), kind='block-stats')
        except Exception: pass
        return st
    if not wdb_sidecar.may_build(fn):            # THE VANILLA LAW: no census built to answer
        _SCACHE[key] = None; return None
    st = compute(seg, col)
    cnt, nn, bsum, cmin, cmax, maxabs = st['cnt'], st['nn'], st['sum'], st['cmin'], st['cmax'], st['maxabs']
    codes = None if st['mode4'] else True
    try:
        import os as _os9
        if wdb_sidecar.births_on(_os9.path.dirname(fn)):                    # THE SWITCH
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
    # REVIVED: stats now live beside the file (.bst sidecars, written once, loaded
    # cold in ~1ms per the sidecar precedent) -- detect reads, never builds, exactly
    # as the retirement note prophesied. First-ever query per column pays the one-time
    # build+write, like a .gbc birth.
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
