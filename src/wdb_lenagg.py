"""wdb_lenagg -- grouped string-measure aggregates from the charlens table.

Q27's lane: AVG/SUM(LENGTH(c)) [+ COUNT(*)] GROUP BY key [WHERE c <> ''] [HAVING
COUNT(*) > lit] [ORDER BY agg] [LIMIT n]  ==  two weighted bincounts over the code
streams. The lengths live in the V-sized charlens table; no string is ever born.
Fail-closed: any unrecognized limb -> None, the general scan serves."""
import numpy as np
from sqlglot import expressions as E
import wdb_sql

_HITS = 0
import wdb_qmem
_PF = wdb_qmem.register({})   # id(seg) -> the unpacking thread; query-scoped (a detect that declines
                              # after starting it leaves nothing behind the query)
_PRICED = []     # (chosen, fused_ms_est, chain_ms_est) -- the honesty loop's seed


def _fn_table(seg, col, lkind):
    """The V-table for F over col's dictionary. Fast providers for the length
    family (header arithmetic, no decode); the GENERAL provider evaluates any
    registered scalar fn once per distinct value. Numeric outputs only."""
    if not isinstance(lkind, tuple):
        return None
    fname, params = lkind
    if fname == 'LENGTH' and hasattr(seg, 'dict_charlens'):
        t = seg.dict_charlens(col)
        if t is not None:
            return t
    if fname == 'STRLEN' and hasattr(seg, 'dict_bytelens'):
        t = seg.dict_bytelens(col)
        if t is not None:
            return t
    try:
        arr = wdb_sql._fval_by_code(seg, fname, col, params)
        if arr is not None and np.issubdtype(np.asarray(arr).dtype, np.number):
            return np.asarray(arr)
    except Exception:
        pass
    return None


def _price(seg, key, lcol):
    """Cost arithmetic from the cards: fused frame-pour vs gather+bincount chain.
    Returns ('fused'|'chain', est_ms_fused, est_ms_chain)."""
    try:
        import wdb_calib
        mc = wdb_calib.machine_card(getattr(seg, 'db_dir', '.') or '.')
        N = float(seg.N)
        fused = (N / 1e6) / max(mc.get('fused_pour_mrps') or 1, 1) * 1000
        chain = (N / 1e6) / max(mc.get('gather_i64_mrps') or 1, 1) * 1000 \
            + 2 * (N / 1e6) / max(mc.get('bincount_w_mrps') or 1, 1) * 1000
        pick = 'fused' if fused <= chain else 'chain'
        _PRICED.append((pick, round(fused, 1), round(chain, 1)))
        return pick, fused, chain
    except Exception:
        return 'fused', 0.0, 0.0


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') or tree.args.get('distinct'):
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 1 or not isinstance(g.expressions[0], E.Column):
        return None
    key = (col_map or {}).get(g.expressions[0].name, g.expressions[0].name)
    aggs = []
    lcol = None
    lkind = None
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            if (col_map or {}).get(inner.name, inner.name) != key:
                return None
            aggs.append(('KEY', None))
            continue
        kd = wdb_sql._agg_kind(inner)
        if kd is not None and kd[0] == 'COUNT_STAR':
            aggs.append(('CNT', None))
            continue
        if isinstance(inner, (E.Avg, E.Sum)):
            fn = inner.this
            kf = None
            cn2 = None
            sf = wdb_sql._scalar_fn(fn)          # THE ALGEBRA IDENTITY, general form:
            if sf is not None:                   # agg(F(dictcol)) GROUP BY key ==
                kf = (sf[1], sf[3])              # weighted pour over F's V-table.
                cn2 = sf[2]                      # F = ANY registered scalar fn.
            if kf is not None:
                c2 = (col_map or {}).get(cn2, cn2)
                if lcol is not None and (c2 != lcol or kf != lkind):
                    return None
                lcol, lkind = c2, kf
                aggs.append(('AVGL' if isinstance(inner, E.Avg) else 'SUML', None))
                continue
        return None
    if lcol is None:
        return None
    w = tree.args.get('where')
    excl_empty = False
    if w is not None:
        n2 = w.this
        if (isinstance(n2, E.NEQ) and isinstance(n2.this, E.Column)
                and (col_map or {}).get(n2.this.name, n2.this.name) == lcol
                and isinstance(n2.expression, E.Literal) and str(n2.expression.this) == ''):
            excl_empty = True
        else:
            return None
    hv = tree.args.get('having')
    hmin = None
    if hv is not None:
        h = hv.this
        if not isinstance(h, E.GT):
            return None
        hk = wdb_sql._agg_kind(h.this)
        if hk is None or hk[0] != 'COUNT_STAR' or not isinstance(h.expression, E.Literal):
            return None
        hmin = int(str(h.expression.this))
    for c in (key, lcol):
        if c not in seg.cols or seg.cols[c].get('code_enc') not in (0, 3, 5, 8, 10, 12, 19):
            return None
    # THE OVERLAP: the V-table (the dictionary's lengths) and the two row columns are independent
    # reads -- unpack the columns on a thread while the dictionary is read here
    # THE ROW LENGTHS (the operator's --row-lengths): when the load stored lcol's character length
    # per row, a LENGTH aggregate reads that column and the key -- never lcol's dictionary numbers
    import wdb_lens as _L9
    rowl = bool(isinstance(lkind, tuple) and lkind[0] == 'LENGTH' and hasattr(seg, 'path')
                and _L9.has_row_lens(seg, lcol) and _L9.clean(seg))
    import threading
    def _pf(s=seg, a=key, b=lcol, r=rowl):
        s._raw_codes(a)
        if not r:
            s._raw_codes(b)
    t9 = threading.Thread(target=_pf, daemon=True); t9.start()
    _PF[id(seg)] = t9
    if not rowl:
        lens = _fn_table(seg, lcol, lkind)
        if lens is None:
            return None
    lim = None
    lx = tree.args.get('limit')
    if lx is not None:
        try:
            lim = int(lx.expression.this)
        except Exception:
            return None
    oi = None
    ox = tree.args.get('order')
    if ox is not None:
        if len(ox.expressions) != 1 or not ox.expressions[0].args.get('desc'):
            return None
        onm = ox.expressions[0].this
        if not isinstance(onm, E.Column):
            return None
        alias_names = [ (p.alias if isinstance(p, E.Alias) else None) for p in tree.expressions ]
        if onm.name not in alias_names:
            return None
        oi = alias_names.index(onm.name)
    return {'key': key, 'lcol': lcol, 'lkind': lkind, 'aggs': aggs, 'excl_empty': excl_empty,
            'hmin': hmin, 'lim': lim, 'oi': oi, 'proj': tree.expressions, 'rowl': rowl}


def _row_sums(seg, spec):
    """per key: length sum and row count from the stored row lengths (the key unpacks on the thread
    detect started, while the lengths are read)"""
    import wdb_lens as _L9
    key, lcol = spec['key'], spec['lcol']
    L = _L9.row_lens(seg, lcol)
    t9 = _PF.pop(id(seg), None)
    if t9 is not None:
        t9.join()
    if L is None:
        return None
    kc = np.asarray(seg._raw_codes(key))
    S, C = _L9.row_pour(kc, L, int(seg.cols[key]['V']), bool(spec['excl_empty']), 16)
    return S.astype(np.float64), C.astype(np.float64)


def _dict_sums(seg, spec):
    key, lcol = spec['key'], spec['lcol']
    lens = _fn_table(seg, lcol, spec.get('lkind'))
    if lens is None:
        return None
    KV = int(seg.cols[key]['V'])
    ec = -1
    if spec['excl_empty']:
        # '' sorts first in a sorted dictionary: when present it IS code 0 -- one fetch, not a
        # binary search through the front-coded chunks (measured 71 ms on URL)
        z0 = seg.fetch(lcol, 0) if int(seg.cols[lcol]['V']) else None
        if isinstance(z0, (bytes, bytearray)):
            z0 = z0.decode('utf-8', 'replace')
        if z0 == '':
            ec = 0
        elif z0 is None or not isinstance(z0, str):
            import wdb_wherescan as WS
            ec0 = WS._code_of(seg, lcol, '')
            ec = int(ec0) if ec0 is not None else -1
    N = int(seg.N)
    if int(lens.max() if lens.size else 0) < 65536:
        lens16 = lens.astype(np.uint16)          # the tiny alphabet rides a u16 bus
    else:
        lens16 = lens.astype(np.int64)
    import wdb_kernels as WK
    # (the bidder's ledger, _price, is no longer asked here (2026-10-06): its pick was never used, and asking it
    # measured this machine once and wrote a .calib.json into the working directory -- a file a query wrote)
    # every row of both columns is read: ONE full decode each through the engine's own fastest
    # reader (pipelined frames / block dictionaries / tag 20), then the workers slice the codes.
    # Window-by-window reads of an uncached column measured 1.25 s against 0.31 s this way (Q27).
    t9 = _PF.pop(id(seg), None)
    if t9 is not None:
        t9.join()
    seg._raw_codes(key)
    seg._raw_codes(lcol)
    BR = 524288
    nfr = (N + BR - 1) // BR
    from concurrent.futures import ThreadPoolExecutor
    def _work(t, T=8):
        j = np.zeros(KV, np.int64)
        c = np.zeros(KV, np.int64)
        for f in range(t, nfr, T):               # striped frames, ONE dispatch:
            lo, hi = f * BR, min((f + 1) * BR, N)
            kcf = np.asarray(seg._raw_codes_range(key, lo, hi))
            ucf = np.asarray(seg._raw_codes_range(lcol, lo, hi))
            WK.lenagg_pour(kcf, ucf, lens16, j, c, np.int64(ec))
        return j, c                              # one partial pair per worker
    sums = np.zeros(KV, np.float64)
    cnt = np.zeros(KV, np.float64)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for j, c in ex.map(_work, range(8)):     # eight workers, no future churn
            sums += j
            cnt += c
    return sums, cnt


def execute(seg, spec):
    global _HITS
    key = spec['key']
    r9 = _row_sums(seg, spec) if spec.get('rowl') else _dict_sums(seg, spec)
    if r9 is None:
        return None
    sums, cnt = r9
    keep = cnt > (spec['hmin'] if spec['hmin'] is not None else 0)
    gs = np.flatnonzero(keep)
    rows = []
    vals = seg.values_at(key, gs) if gs.size else []
    for i2, g0 in enumerate(gs.tolist()):
        kv = vals[i2]
        if isinstance(kv, (bytes, bytearray)):
            kv = kv.decode('utf-8', 'replace')
        row = []
        for kind, _ in spec['aggs']:
            if kind == 'KEY':
                row.append(kv)
            elif kind == 'CNT':
                row.append(int(cnt[g0]))
            elif kind == 'SUML':
                row.append(float(sums[g0]))
            else:
                row.append(float(sums[g0] / cnt[g0]) if cnt[g0] else None)
        rows.append(tuple(row))
    if spec['oi'] is not None:
        rows.sort(key=lambda r: (r[spec['oi']] is None, r[spec['oi']]), reverse=True)
    if spec['lim'] is not None:
        rows = rows[:spec['lim']]
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in spec['proj']]
