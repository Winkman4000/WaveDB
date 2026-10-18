"""wdb_lenagg -- grouped string-measure aggregates (Q27's shape) from the strings' stored lengths.

  SELECT K, AGG(F(S)) [, COUNT(*)] FROM t [WHERE S <> ''] GROUP BY K [HAVING COUNT(*) op n]
  [ORDER BY <a projection> [ASC|DESC]] [LIMIT n]          AGG = AVG | SUM, F any registered numeric scalar fn

== per key, the sum of F over its rows and their count: one pour over two code streams. F's value per row
comes from the V-table of F over S's dictionary (the dictionary's character lengths for LENGTH), or, for
LENGTH, from the row lengths the load stored (wdb_lens) -- then S's numbers are never read. No string is born.

2026-10-18 (the Q27 board, every key profile x every text profile, 680 queries): this was a hidden first try
inside general_scan, so the route map could not see it, and its frame-range reader carried an encoding gate
(code_enc 0/3/5/8/10/12/19 for both columns) that sent 386 of the 680 queries to the scan at 2.2x DuckDB.
Now a registered read; the pour runs over the decoded codes (_raw_codes reads every encoding) on every core:
per-thread padded boards for a key up to BIG_KV values, the range partition (bincount_big's method) past it.
Fail-closed: any unrecognized limb -> None, the next read serves."""
import os
import numpy as np
from sqlglot import expressions as E
import wdb_sql
import wdb_policies as P

_HITS = 0
_PF = {}                      # kept for older callers (tests cleared it after a detect); detect starts no thread
# past this many keys: the range partition. Measured (pod 2, 100M rows, stored row lengths; boards / partition ms):
# KV 9K 13 / 406 (URL dense) 16 / 91 (SearchPhrase sparse); 1.4M 117 / 293, 91 / 78; 6M 353 / 350; 9.8M 631 / 362,
# 572 / 84; 17.6M 1058 / 358, 989 / 90. The boards grow with the keys, the partition with the live rows.
BIG_KV = int(os.environ.get('WDB_LENAGG_BIG_KV', str(4 << 20)))
_HAVING = {E.GT: lambda c, n: c > n, E.GTE: lambda c, n: c >= n, E.LT: lambda c, n: c < n,
           E.LTE: lambda c, n: c <= n, E.EQ: lambda c, n: c == n}


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


def _plain_col(seg, col):
    """a dictionary column the pour can read as stored: present, not positional, no NULL, no deleted rows,
    no overrides (the pour reads the stored codes)"""
    return (P.columns_exist(seg, col) and P.not_positional(seg, col) and P.no_deleted_rows(seg)
            and seg.cols[col].get('mode') in (0, 1, 2) and not seg.cols[col].get('has_null')
            and seg._effective(col) is None)


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
    having = None
    if hv is not None:
        h = hv.this
        op = _HAVING.get(type(h))
        if op is None:
            return None
        hk = wdb_sql._agg_kind(h.this)
        if hk is None or hk[0] != 'COUNT_STAR' or not isinstance(h.expression, E.Literal) or h.expression.is_string:
            return None
        try:
            having = (op, int(str(h.expression.this)))
        except ValueError:
            return None
    if key == lcol or not _plain_col(seg, key) or not _plain_col(seg, lcol):
        return None
    lim = None
    lx = tree.args.get('limit')
    if lx is not None:
        try:
            lim = int(lx.expression.this)
        except Exception:
            return None
    if tree.args.get('offset') is not None:
        return None
    oi = None
    odesc = True
    ox = tree.args.get('order')
    if ox is not None:
        if len(ox.expressions) != 1:
            return None
        o0 = ox.expressions[0]
        odesc = bool(o0.args.get('desc'))
        onm = o0.this
        if not isinstance(onm, E.Column):
            return None
        alias_names = [(p.alias if isinstance(p, E.Alias) else None) for p in tree.expressions]
        if onm.name in alias_names:
            oi = alias_names.index(onm.name)
        elif (col_map or {}).get(onm.name, onm.name) == key and ('KEY', None) in aggs:
            oi = aggs.index(('KEY', None))
        else:
            return None
    # THE ROW LENGTHS (the operator's --row-lengths): when the load stored lcol's character length
    # per row, a LENGTH aggregate reads that column and the key -- never lcol's dictionary numbers
    import wdb_lens as _L9
    rowl = bool(isinstance(lkind, tuple) and lkind[0] == 'LENGTH' and hasattr(seg, 'path')
                and _L9.has_row_lens(seg, lcol) and _L9.clean(seg))
    if not rowl and _fn_table(seg, lcol, lkind) is None:
        return None
    return {'key': key, 'lcol': lcol, 'lkind': lkind, 'aggs': aggs, 'excl_empty': excl_empty,
            'having': having, 'lim': lim, 'oi': oi, 'odesc': odesc, 'proj': tree.expressions, 'rowl': rowl}


def _empty_code(seg, lcol):
    """the code of '' in lcol's dictionary, or -1: '' sorts first in a sorted dictionary, so when present it IS
    code 0 -- one fetch, not a binary search through the front-coded chunks (measured 71 ms on URL)"""
    if not int(seg.cols[lcol]['V']):
        return -1
    z0 = seg.fetch(lcol, 0)
    if isinstance(z0, (bytes, bytearray)):
        z0 = z0.decode('utf-8', 'replace')
    if z0 == '':
        return 0
    if z0 is None or not isinstance(z0, str):
        import wdb_wherescan as WS
        ec0 = WS._code_of(seg, lcol, '')
        return int(ec0) if ec0 is not None else -1
    return -1


def _sums(seg, spec):
    """(sums, counts) per key code: the stored row lengths, or the V-table of F over S's dictionary, poured on
    every core against the key's codes"""
    import wdb_kernels as WK
    key, lcol = spec['key'], spec['lcol']
    KV = int(seg.cols[key]['V'])
    T = WK._nt()
    L = tab = sc = None
    ec = -1
    # THE OVERLAP: the key's codes (and S's, without row lengths) unpack on a thread while the lengths are
    # read. Started here, joined below, in one call (2026-10-18: started in detect, it was left running whenever
    # execute did not follow -- the route map's markers -- and died with the interpreter mid-decode)
    import threading
    def _pf(s=seg, a=key, b=lcol, r=spec['rowl']):
        s._raw_codes(a)
        if not r:
            s._raw_codes(b)
    t9 = threading.Thread(target=_pf, daemon=True); t9.start()
    try:
        if spec['rowl']:
            import wdb_lens as _L9
            L = _L9.row_lens(seg, lcol)
        else:
            tab = _fn_table(seg, lcol, spec['lkind'])
            if tab is not None:
                tab = np.asarray(tab)
                tab = tab.astype(np.float64) if tab.dtype.kind == 'f' else tab.astype(np.int64)
                if spec['excl_empty']:
                    ec = _empty_code(seg, lcol)
    finally:
        t9.join()
    if L is None and tab is None:
        return None
    kc = np.asarray(seg._raw_codes(key))
    if not spec['rowl']:
        sc = np.asarray(seg._raw_codes(lcol))
    acc = np.float64 if (tab is not None and tab.dtype.kind == 'f') else np.int64
    if KV > BIG_KV:
        bits = max(1, (KV - 1).bit_length())
        outS = np.zeros(KV, acc); outC = np.zeros(KV, np.int64)
        WK.lenpour_big(kc, L if L is not None else np.zeros(1, np.uint16),
                       sc if sc is not None else np.zeros(1, np.uint8),
                       tab if tab is not None else np.zeros(1, np.int64),
                       tab is not None, bool(spec['excl_empty']), np.int64(ec),
                       np.int64(max(0, bits - 12)), np.int64(min(T, 16)), outS, outC)
        return outS, outC
    if L is not None:
        import wdb_lens as _L9
        return _L9.row_pour(kc, L, KV, bool(spec['excl_empty']), T)
    S = np.zeros((T, KV + 16), acc)              # rows padded apart (THE PADDED BOARDS)
    C = np.zeros((T, KV + 16), np.int64)
    WK.lenpour_dict(kc, sc, tab, np.int64(ec), S, C)
    return S.sum(0)[:KV], C.sum(0)[:KV]


def execute(seg, spec):
    global _HITS
    key = spec['key']
    r9 = _sums(seg, spec)
    if r9 is None:
        return None
    sums, cnt = r9
    keep = cnt > 0
    if spec['having'] is not None:
        op, n = spec['having']
        keep &= op(cnt, n)
    gs = np.flatnonzero(keep)
    rows = []
    vals = seg.values_at(key, gs) if gs.size else []
    isint = sums.dtype.kind in 'iu'
    for i2, g0 in enumerate(gs.tolist()):
        kv = vals[i2]
        if isinstance(kv, (bytes, bytearray)):
            kv = kv.decode('utf-8', 'replace')
        kv = wdb_sql._pyval(kv)
        row = []
        for kind, _ in spec['aggs']:
            if kind == 'KEY':
                row.append(kv)
            elif kind == 'CNT':
                row.append(int(cnt[g0]))
            elif kind == 'SUML':
                row.append(int(sums[g0]) if isint else float(sums[g0]))
            else:
                row.append(float(sums[g0]) / float(cnt[g0]))
        rows.append(tuple(row))
    if spec['oi'] is not None:
        oi = spec['oi']
        rows.sort(key=lambda r: (r[oi] is None, r[oi]), reverse=spec['odesc'])
    if spec['lim'] is not None:
        rows = rows[:spec['lim']]
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in spec['proj']]
