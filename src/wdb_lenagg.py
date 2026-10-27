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
# under WHERE S <> '': live rows up to N / SPARSE_DIV are grouped at their rows (0: never) for a key past SPARSE_KV
# values, up to N / SPARSE_DIV_SMALL for a smaller key. Measured (pod 2, the Q27 board): a big key's pour costs its
# boards / partition -- 1.56% live sorted and grouped in 31-44 ms, 13% live 285-450 ms where the pour is cheaper;
# a small key's pour is cheap and its read at 1.5M rows is not (Robotness x UTMSource 98 -> 205 ms), while 2,263
# live rows still win (ResolutionHeight x ParamOrderID 100 -> 35 ms)
SPARSE_KV = 1 << 20
# THE LIVE ROWS FIRST (2026-10-27): an enc-10 measured column with at most N >> LIVE_SHIFT non-empty rows is read at
# those rows only. Measured on the Q27 board (this PC): SocialSourcePage (177,085 non-empty) 68 of 68 variants faster,
# ~155 ms -> 4-67 ms (0.03-0.7x DuckDB, were 1.5-3.1x); OpenstatAdID / OpenstatServiceName (460K) 30 of 30 faster. Other
# encodings lose (their non-empty rows cost more to find than the current read: ParamOrderID enc 3, 2,263 rows, 27 ->
# 50 ms; UTMSource enc 3 30 -> 115; MobilePhoneModel enc 9 47 -> 270; SocialNetwork enc 2 20 -> 350; text 17-85%
# non-empty 160 ms -> 0.7-5.5 s). Nothing past 460K non-empty was measured: the line sits at N / 128 (781K).
# Only where the read would inflate the stored ROW LENGTHS (rowl: 134 ms for 100M rows whatever the column): the
# OpenstatAdID / OpenstatServiceName columns carry none, their read already works off the codes (30-70 ms), and the
# live rows lost there on the board -- CounterID x OpenstatAdID open 67 -> 87 ms, SocialNetwork x OpenstatServiceName
# filtered 30 -> 39 (re-timed back to back, c83 vs c84).
LIVE_SHIFT = int(os.environ.get('WDB_LENAGG_LIVE_SHIFT', '7'))      # -1: never
LIVE_ENC = (10,)
_LIVE_HITS = [0]
SPARSE_DIV = int(os.environ.get('WDB_LENAGG_SPARSE_DIV', '50'))
SPARSE_DIV_SMALL = int(os.environ.get('WDB_LENAGG_SPARSE_DIV_SMALL', '1024'))
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


def _key_at(seg, key, pos):
    """the key's codes at the live rows only, by the grab's rule (wdb_gridwalk): point reads while the rows touch
    few frames (frame-coded) or are few (the rest); a one-stream column, or many rows, decoded whole and gathered"""
    import wdb_gridwalk as GW
    if key in seg._codes:
        return np.asarray(seg._codes[key])[pos]
    N = int(seg.N)
    oc = seg.cols[key]
    enc = oc.get('code_enc', 0)
    BR = int(oc.get('BR') or oc.get('e20BR') or 0) if enc in (3, 18, 20) else 0
    if enc in GW.GRAB_STREAM:
        at = False
    elif BR > 0:
        fr = pos // BR
        touched = 1 + int(np.count_nonzero(fr[1:] != fr[:-1])) if pos.size else 0
        at = touched <= GW.GRAB_FRAMES * ((N + BR - 1) // BR)
    else:
        at = pos.size * GW.GRAB_AT_DIV <= N
    if at:
        return np.asarray(seg.codes_at(key, pos))
    return np.asarray(seg._raw_codes(key))[pos]


def _group(kk, w):
    """(the keys present, ascending; the sum of w; the count) over the few live (key, value) pairs: one parallel
    bucketed sort of (key << 16 | value) when the values are lengths under 65536, else a stable argsort"""
    import wdb_kernels as WK
    if kk.size == 0:
        return np.empty(0, np.int64), np.empty(0, w.dtype), np.empty(0, np.int64)
    if w.dtype.kind in 'iu' and int(w.min()) >= 0 and int(w.max()) < 65536 and int(kk.max()) < (1 << 46):
        p = (np.asarray(kk, np.int64) << 16) | np.asarray(w, np.int64)
        ps = WK.sort_keys_par(p, np.int64(max(0, int(p.max()).bit_length() - 12)))
        ks = ps >> 16
        ws = ps & 0xFFFF
    else:
        o = np.argsort(kk, kind='stable')
        ks = np.asarray(kk, np.int64)[o]
        ws = w[o]
    brk = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    return ks[brk], np.add.reduceat(ws, brk), np.diff(np.r_[brk, ks.size]).astype(np.int64)


def _live_sums(seg, spec):
    """THE LIVE ROWS FIRST (Jackson's steps): 1. S's census: non-empty rows = N - rows(''); past N >> LIVE_SHIFT:
    None (the current read). 2. their positions (the encoding's own exceptions), their codes, F of each code (the
    dictionary table), the key's codes at them. 3. WHERE S <> '': grouped (the codes present, ascending). 4. no WHERE:
    the sum per key from them alone (an empty row adds F('') -- 0 for length), the count per key from the key's census
    over every row."""
    key, lcol = spec['key'], spec['lcol']
    c = seg.cols[lcol]
    if not spec['rowl'] or c.get('code_enc') not in LIVE_ENC or LIVE_SHIFT < 0:
        return None                              # the win is the stored row lengths not inflated (see LIVE_SHIFT)
    N = int(seg.N)
    V = int(c['V'])
    ec = _empty_code(seg, lcol)
    if ec < 0 or V <= 0:
        return None
    cen = np.asarray(seg.raw_census(lcol, V))
    live = N - int(cen[ec])
    if live > (N >> LIVE_SHIFT):
        return None
    tab = _fn_table(seg, lcol, spec['lkind'])
    if tab is None:
        return None
    tab = np.asarray(tab)
    tab = tab.astype(np.float64) if tab.dtype.kind == 'f' else tab.astype(np.int64)
    import wdb_wherescan as WS
    pos = np.asarray(WS._scan_eq(seg, lcol, int(ec), 0, N, negate=True), np.int64)
    w = tab[np.asarray(seg.codes_at(lcol, pos)).astype(np.int64)]
    gk, s, n9 = _group(_key_at(seg, key, pos), w)
    _LIVE_HITS[0] += 1
    if spec['excl_empty']:
        return gk, s, n9
    KV = int(seg.cols[key]['V'])
    cnt = np.asarray(seg.raw_census(key, KV)).astype(np.int64)[:KV]
    sums = np.zeros(KV, s.dtype)
    sums[gk] = s
    if float(tab[ec]) != 0.0:                    # F('') is not 0: each empty row adds it
        lv = np.zeros(KV, np.int64)
        lv[gk] = n9
        sums = sums + (cnt - lv) * tab[ec]
    return None, sums, cnt


def _sums(seg, spec):
    """(key codes, sums, counts): the stored row lengths, or the V-table of F over S's dictionary, poured on every
    core against the key's codes (key codes None: dense, indexed by code) -- or, under WHERE S <> '' with few live
    rows, grouped at those rows (the codes present, ascending)"""
    import wdb_kernels as WK
    r9 = _live_sums(seg, spec)
    if r9 is not None:
        return r9
    key, lcol = spec['key'], spec['lcol']
    KV = int(seg.cols[key]['V'])
    N = int(seg.N)
    T = WK._nt()
    L = tab = sc = None
    ec = -1
    rowl = spec['rowl']
    # THE OVERLAP: with row lengths the key's codes unpack on a thread while the lengths are read; without, S's
    # codes unpack while F's table is made (the key waits: few live rows read it at those rows only). Started
    # here, joined below, in one call (2026-10-18: started in detect, it was left running whenever execute did
    # not follow -- the route map's markers -- and died with the interpreter mid-decode)
    import threading
    def _pf(s=seg, a=key, b=lcol, r=rowl):
        s._raw_codes(a if r else b)
    t9 = threading.Thread(target=_pf, daemon=True); t9.start()
    try:
        if rowl:
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
    if not rowl:
        sc = np.asarray(seg._raw_codes(lcol))
    # THE GRAB (2026-10-18, the Q27 board's corner): under WHERE S <> '' a mostly-empty S leaves few live rows --
    # a big key's whole decode and its KV-sized boards were most of the cost (WatchID x ParamOrderID 2,263 live
    # rows: 211 ms). Count the live rows (one parallel pass); none: no group; few: read the key at them only and
    # group them by a sort. Past N / SPARSE_DIV the pour below is cheaper (measured: 13% live, a sort 285-450 ms)
    div = SPARSE_DIV if KV > SPARSE_KV else SPARSE_DIV_SMALL
    if spec['excl_empty'] and div > 0 and (rowl or ec >= 0):
        # a strided sample (~100K rows, ~1 ms) says when the live rows are plainly too many, so the dense side skips
        # the count pass (12-30 ms on 100M; UserID x SearchPhrase, 13% live, had paid it for nothing)
        smp = (L if rowl else sc)[::1021]
        live_est = float(np.count_nonzero(smp) if rowl else np.count_nonzero(smp != ec)) / max(1, smp.size)
        pos = None
        if live_est * N <= 2.0 * (N // div):
            pos = (WK.live_rows(L, 1, ge=True, limit=N // div) if rowl
                   else WK.live_rows(sc, ec, limit=N // div))
        if pos is not None:
            return _group(_key_at(seg, key, pos), L[pos] if rowl else tab[sc[pos]])
    kc = np.asarray(seg._raw_codes(key))
    acc = np.float64 if (tab is not None and tab.dtype.kind == 'f') else np.int64
    sums = cnt = None
    if KV > BIG_KV:
        bits = max(1, (KV - 1).bit_length())
        outS = np.zeros(KV, acc); outC = np.zeros(KV, np.int64)
        WK.lenpour_big(kc, L if L is not None else np.zeros(1, np.uint16),
                       sc if sc is not None else np.zeros(1, np.uint8),
                       tab if tab is not None else np.zeros(1, np.int64),
                       tab is not None, bool(spec['excl_empty']), np.int64(ec),
                       np.int64(max(0, bits - 12)), np.int64(min(T, 16)), outS, outC)
        sums, cnt = outS, outC
    elif L is not None:
        import wdb_lens as _L9
        sums, cnt = _L9.row_pour(kc, L, KV, bool(spec['excl_empty']), T)
    else:
        S = np.zeros((T, KV + 16), acc)          # rows padded apart (THE PADDED BOARDS)
        C = np.zeros((T, KV + 16), np.int64)
        WK.lenpour_dict(kc, sc, tab, np.int64(ec), S, C)
        sums, cnt = S.sum(0)[:KV], C.sum(0)[:KV]
    return None, sums, cnt                       # dense: the key code IS the index (no compaction of KV slots)


def execute(seg, spec):
    global _HITS
    key = spec['key']
    r9 = _sums(seg, spec)
    if r9 is None:
        return None
    gk, sums, cnt = r9
    keep = cnt > 0
    if spec['having'] is not None:
        op, n = spec['having']
        keep &= op(cnt, n)
    sel = np.flatnonzero(keep)
    gs = sel if gk is None else gk[sel]
    rows = []
    vals = seg.values_at(key, gs) if gs.size else []
    isint = sums.dtype.kind in 'iu'
    for i2, g0 in enumerate(sel.tolist()):
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
