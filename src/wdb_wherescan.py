"""wherescan: the disk-first WHERE read -- predicate scan over blocked (enc=3) code frames.

The shape this serves is the ClickBench trench: a conjunction of selective predicates feeding a
GROUP BY or aggregate, e.g. WHERE CounterID = 62 AND EventDate BETWEEN .. AND IsRefresh = 0.
Today those queries full-decode every involved column (0.6-1.9 s each; Q39 times out). Here:

  1. STAIR predicates (range/eq on a staircase column) become exact row spans via searchsorted
     over the step rows -- no bytes touched.
  2. The first dict-equality predicate becomes a PARALLEL FRAME SCAN: decompress each enc=3 frame
     inside the span (thread pool, per-thread zstd via the thread-local property), compare, keep
     positions. Measured primitive: 100M rows in ~95 ms.
  3. Remaining dict-eq predicates check via codes_at (touched frames only); mode-4 flag
     predicates check in place (bitpacked values, cached seq decode).
  4. Survivors group/aggregate: keys gathered by codes_at, composite built factorize-then-combine
     (overflow-safe), counts by bincount, canonical count-desc order, OFFSET/LIMIT, winners
     decoded one value each. Supports the CASE WHEN <eq-conj> THEN col ELSE '' derived key (Q39).

Zero columns enter the codes cache; the pipeline's residency is its survivors. Declines: joins,
distinct, having, deleted rows, overrides on involved columns, non-conjunctive WHERE, LIKE/regex.
Prototype of this exact pipeline ran Q39 (a 45 s timeout) in ~1 s steady-state, exact vs DuckDB.
"""
import numpy as np
import sqlglot.expressions as E
from concurrent.futures import ThreadPoolExecutor
import wdb_sql
import wdb_policies as P
import wdb_gdsidecar
import os

_ENABLED = True
_HITS = 0
_SCAN_THREADS = 14


def enable():
    global _ENABLED; _ENABLED = True


def disable():
    global _ENABLED; _ENABLED = False


# ---------------------------------------------------------------- literal -> code
def _lit_bytes(v):
    return v if isinstance(v, (bytes, bytearray)) else str(v).encode()


def _litval(node):
    """Python value of a Literal, unwrapping CAST('..' AS DATE/TIMESTAMP) and unary minus."""
    n = node
    neg = False
    while isinstance(n, (E.Cast, E.Neg, E.Paren)):
        if isinstance(n, E.Neg):
            neg = not neg
        n = n.this
    if isinstance(n, E.Literal):
        v = wdb_sql._literal_value(n)
        return -v if neg and isinstance(v, (int, float)) else v
    return None


def _code_of(seg, col, val):
    """Dict code of a literal, or None when absent. Dicts are value-sorted: mode 2 answers by
    searchsorted over the int dict; string dicts by fetch-bisect (O(log V) point fetches, each
    sub-ms against chunked dicts -- never materializes the dictionary)."""
    c = seg.cols[col]
    V = int(c['V']) - (1 if c['has_null'] else 0)
    if V <= 0:
        return None
    if c['mode'] == 2:
        arr = seg._dict_ints(c)
        try:
            iv = int(val)
        except (TypeError, ValueError):
            return None
        k = int(np.searchsorted(arr[:V], iv))
        return k if k < V and int(arr[k]) == iv else None
    if c['dt'] == 0:
        try:
            iv = int(val)
        except (TypeError, ValueError):
            return None
        lo, hi = 0, V - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            fv = int(seg.fetch(col, mid))
            if fv == iv: return mid
            if fv < iv: lo = mid + 1
            else: hi = mid - 1
        return None
    tgt = _lit_bytes(val)
    lo, hi = 0, V - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        fv = seg.fetch(col, mid)
        fv = fv if isinstance(fv, (bytes, bytearray)) else _lit_bytes(fv)
        if fv == tgt: return mid
        if fv < tgt: lo = mid + 1
        else: hi = mid - 1
    return None


# ---------------------------------------------------------------- WHERE parsing
def _conjuncts(node):
    if isinstance(node, E.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    if isinstance(node, E.Paren):
        return _conjuncts(node.this)
    return [node]


def _col_lit(node):
    """(colname, literal, op) for col <op> literal, either side. op in ('=','<>','>=','<=')."""
    ops = {E.EQ: '=', E.NEQ: '<>', E.GTE: '>=', E.LTE: '<='}
    op = ops.get(type(node))
    if op is None:
        return None
    a, b = node.this, node.expression
    if isinstance(a, E.Column) and isinstance(b, (E.Literal, E.Cast, E.Neg)):
        return a.name, _litval(b), op
    if isinstance(b, E.Column) and isinstance(a, (E.Literal, E.Cast, E.Neg)):
        flip = {'>=': '<=', '<=': '>='}
        return b.name, _litval(a), flip.get(op, op)
    return None


def _in_list(node):
    """(colname, [literals]) for `col IN (lit, ...)` -- every member must be a literal."""
    if not isinstance(node, E.In) or not isinstance(node.this, E.Column):
        return None
    vals = []
    for x in node.expressions:
        v = _litval(x)
        if v is None:
            return None
        vals.append(v)
    return node.this.name, vals

def _case_key(p, seg, col_map):
    """The Q39 pattern: CASE WHEN <conj of col = int-literal> THEN <column> ELSE <literal>."""
    inner = p.this if isinstance(p, E.Alias) else p
    if not isinstance(inner, E.Case) or len(inner.args.get('ifs', [])) != 1:
        return None
    iff = inner.args['ifs'][0]
    default = inner.args.get('default')
    if default is not None and not isinstance(default, E.Literal):
        return None
    then = iff.args.get('true')
    if not isinstance(then, E.Column):
        return None
    conds = []
    for cn in _conjuncts(iff.this):
        cl = _col_lit(cn)
        if cl is None or cl[2] != '=':
            return None
        col = col_map.get(cl[0], cl[0]) if col_map else cl[0]
        if not P.columns_exist(seg, col):
            return None
        if seg.cols[col].get('mode') not in (0, 1, 2, 4):
            return None
        conds.append((col, cl[1]))       # literal resolved at execute (code for dicts, int for mode 4)
    src = col_map.get(then.name, then.name) if col_map else then.name
    if not P.columns_exist(seg, src):
        return None
    dflt = _litval(default) if default is not None else ''
    return {'kind': 'case', 'conds': conds, 'src': src, 'default': dflt}


def _count_distinct(p):
    """colname for COUNT(DISTINCT col)."""
    inner = p.this if isinstance(p, E.Alias) else p
    if isinstance(inner, E.Count) and isinstance(inner.this, E.Distinct):
        exprs = inner.this.expressions
        if len(exprs) == 1 and isinstance(exprs[0], E.Column):
            return exprs[0].name
    return None


def _like(node):
    """(col, needle, negate) for `col [NOT] LIKE '%needle%'` -- contains form only."""
    neg = False
    n = node
    if isinstance(n, E.Not):
        n = n.this; neg = True
    if not isinstance(n, E.Like) or not isinstance(n.this, E.Column):
        return None
    neg = neg or bool(n.args.get('negate'))      # sqlglot: NOT LIKE == Like(negate=True)
    pat = _litval(n.expression)
    if not isinstance(pat, str) or len(pat) < 3:
        return None
    if not (pat.startswith('%') and pat.endswith('%')):
        return None
    needle = pat[1:-1]
    if '%' in needle or '_' in needle:
        return None                      # only the plain contains form
    return n.this.name, needle, neg


def _like_flags(seg, col, needle):
    """Boolean flag[code] = dict value contains needle. THE DICT IS THE HAYSTACK: the row data
    is never string-compared -- all distinct values are scanned once (C-speed buffer find with
    per-string skip), and the predicate collapses to a code-set membership test."""
    vals = seg._typed_dict(col)
    V = len(vals)
    bs = [v if isinstance(v, (bytes, bytearray)) else
          (v.encode() if isinstance(v, str) else bytes(v)) for v in vals]
    lens = np.fromiter((len(v) for v in bs), np.int64, V)
    offs = np.zeros(V + 1, np.int64); np.cumsum(lens, out=offs[1:])
    hay = b''.join(bs)
    nd = needle.encode() if isinstance(needle, str) else needle
    flag = np.zeros(V, bool)
    pos = hay.find(nd)
    while pos >= 0:
        i = int(np.searchsorted(offs, pos, side='right')) - 1
        if pos + len(nd) <= offs[i + 1]:
            flag[i] = True
        pos = hay.find(nd, int(offs[i + 1]))     # skip the rest of this string either way
    return flag


def detect(seg, tree, col_map):
    if not _ENABLED:                    return None
    if not P.no_joins(tree):            return None
    if not P.no_select_distinct(tree):  return None
    if not P.no_having(tree):           return None
    if not P.no_deleted_rows(seg):      return None
    where = tree.args.get('where')
    if where is None:
        return None
    spans, eqs, flags, ins, likes = [], [], [], [], []
    for cn in _conjuncts(where.this):
        cl = _col_lit(cn)
        if cl is None:
            il = _in_list(cn)
            if il is not None:
                col = col_map.get(il[0], il[0]) if col_map else il[0]
                if not P.columns_exist(seg, col):   return None
                if seg._effective(col) is not None: return None
                if seg.cols[col].get('mode') not in (0, 1, 2, 4): return None
                ins.append((col, il[1]))
                continue
            lk = _like(cn)
            if lk is None:
                return None
            col = col_map.get(lk[0], lk[0]) if col_map else lk[0]
            if not P.columns_exist(seg, col):   return None
            if seg._effective(col) is not None: return None
            if seg.cols[col].get('mode') not in (0, 1):
                return None              # substring haystack needs a string dict
            likes.append((col, lk[1], lk[2]))
            continue
        col = col_map.get(cl[0], cl[0]) if col_map else cl[0]
        if not P.columns_exist(seg, col):   return None
        if seg._effective(col) is not None: return None
        c = seg.cols[col]
        if c.get('mode') == 4 and cl[2] in ('=', '<>'):
            flags.append((col, int(cl[1]), cl[2]))
        elif seg.stairs(col) is not None and cl[2] in ('>=', '<=', '='):
            spans.append((col, cl[1], cl[2]))
        elif c.get('mode') in (0, 1, 2) and cl[2] in ('=', '<>'):
            eqs.append((col, cl[1], cl[2]))
        else:
            return None
    if not eqs and not spans and not likes:
        return None                      # unselective flag-only shapes stay with the scan family
    drive_eq = next((i for i, e in enumerate(eqs) if e[2] == '='), None)
    drive_like = next((i for i, l in enumerate(likes) if not l[2]), None)
    if drive_eq is None and drive_like is None and not spans:
        return None                      # need a positive driver (= or LIKE) or a stair span
    # projections: plain key columns, aggregates, the CASE derived key, or bare * (rows mode)
    proj = tree.expressions
    if len(proj) == 1 and isinstance(proj[0], E.Star):
        if tree.args.get('group') is not None:
            return None
        order = tree.args.get('order')
        if order is None or not order.expressions:
            return None
        oe = order.expressions[0]
        ocol = oe.this.name if isinstance(oe.this, E.Column) else None
        ocol = col_map.get(ocol, ocol) if (col_map and ocol) else ocol
        if ocol is None or oe.args.get('desc') or seg.stairs(ocol) is None:
            return None                  # rows mode rides the cluster order: ASC on a stair column
        tiebreak = []
        for oe2 in order.expressions[1:]:
            if oe2.args.get('desc') or not isinstance(oe2.this, E.Column):
                return None
            tc = oe2.this.name
            tc = col_map.get(tc, tc) if col_map else tc
            if not P.columns_exist(seg, tc):
                return None
            tiebreak.append(tc)
        lim = wdb_sql._limit(tree)
        if lim is None or lim <= 0 or lim > 100000:
            return None
        return {'spans': spans, 'eqs': eqs, 'flags': flags, 'ins': ins, 'likes': likes,
                'drive_eq': drive_eq, 'drive_like': drive_like, 'mode': 'rows',
                'ocol': ocol, 'tiebreak': tiebreak,
                'lim': int(lim), 'off': int(wdb_sql._offset(tree) or 0)}
    keys, aggs = [], []
    for pi, p in enumerate(proj):
        cd = _count_distinct(p)
        if cd is not None:
            col = col_map.get(cd, cd) if col_map else cd
            if not P.columns_exist(seg, col):   return None
            if seg.cols[col].get('mode') not in (0, 1, 2): return None
            aggs.append((pi, 'COUNT_D', col, )); continue
        ak = wdb_sql._agg_kind(p)
        if ak is not None:
            if ak[0] == 'COUNT_STAR':
                aggs.append((pi, 'COUNT_STAR', None)); continue
            if ak[0] not in ('SUM', 'AVG', 'MIN') or not isinstance(ak[1], str):
                return None
            col = col_map.get(ak[1], ak[1]) if col_map else ak[1]
            if not P.columns_exist(seg, col):   return None
            if ak[0] == 'MIN' and seg.cols[col].get('mode') in (0, 1, 2):
                aggs.append((pi, 'MIN_DICT', col)); continue
            if seg.cols[col].get('dt') != 0:    return None
            aggs.append((pi, ak[0], col)); continue
        ck = _case_key(p, seg, col_map)
        if ck is not None:
            keys.append((pi, ck)); continue
        nm = wdb_sql._proj_colname(p)
        if nm is None:
            return None
        col = col_map.get(nm, nm) if col_map else nm
        if not P.columns_exist(seg, col):   return None
        if seg._effective(col) is not None: return None
        keys.append((pi, {'kind': 'col', 'src': col}))
    group = tree.args.get('group')
    if group is None and keys:
        return None
    if group is not None and len(group.expressions) != len(keys):
        return None                      # every key projection must be grouped (and vice versa)
    # resident structures outrank the scan: a gd sidecar serving this exact
    # (group key, COUNT DISTINCT target) shape keeps the query (doctrine: min(point, pop)).
    # LIKE conjuncts disqualify the sidecar (it can only filter eq/in shapes), so no defer.
    if not likes and len(keys) == 1 and keys[0][1].get('kind') == 'col':
        for a in aggs:
            if a[1] == 'COUNT_D' and os.path.exists(
                    wdb_gdsidecar.sidecar_path(seg.path, keys[0][1]['src'], a[2])):
                return None
    order = tree.args.get('order')
    if order is not None:
        if len(order.expressions) != 1 or not order.expressions[0].args.get('desc'):
            return None
        onm = order.expressions[0].this
        cnt_idx = [pi for pi, k, _ in aggs if k == 'COUNT_STAR']
        if not cnt_idx:
            return None
        tgt = onm.name if isinstance(onm, E.Column) else None
        if tgt is None or tgt not in (wdb_sql._alias(proj[cnt_idx[0]]), 'COUNT', 'count'):
            return None
    lim = wdb_sql._limit(tree)
    off = wdb_sql._offset(tree) or 0
    return {'spans': spans, 'eqs': eqs, 'flags': flags, 'ins': ins, 'likes': likes,
            'drive_eq': drive_eq, 'drive_like': drive_like, 'mode': 'group',
            'keys': keys, 'aggs': aggs, 'proj': proj, 'ordered': order is not None,
            'lim': lim, 'off': int(off)}

# ---------------------------------------------------------------- execution
def _bound_code(seg, col, val, side):
    """Insertion point of a literal in the value-sorted dict: the first code whose value is
    >= (side='left') or > (side='right') the literal. Handles int, datetime (dt==3, compared as
    numpy datetime64), and byte-string dicts via fetch-bisect -- O(log V) point fetches."""
    c = seg.cols[col]
    V = int(c['V']) - (1 if c['has_null'] else 0)
    dt = c['dt']
    if dt == 3:
        try:
            tgt = np.datetime64(str(val))
        except Exception:
            return None
        def key(k):
            return np.datetime64(seg.fetch(col, k))
    elif dt == 0:
        try:
            tgt = int(val)
        except (TypeError, ValueError):
            return None
        def key(k):
            return int(seg.fetch(col, k))
    else:
        tgt = _lit_bytes(val)
        def key(k):
            fv = seg.fetch(col, k)
            return fv if isinstance(fv, (bytes, bytearray)) else _lit_bytes(fv)
    lo, hi = 0, V
    while lo < hi:
        mid = (lo + hi) // 2
        kv = key(mid)
        if kv < tgt or (side == 'right' and kv == tgt):
            lo = mid + 1
        else:
            hi = mid
    return lo


def _span_rows(seg, spans):
    lo, hi = 0, int(seg.N)
    for col, val, op in spans:
        st = seg.stairs(col)
        V = int(seg.cols[col]['V']) - (1 if seg.cols[col]['has_null'] else 0)
        def row_of(code):                        # first row of a code (code==V -> N)
            if code <= 0: return 0
            return int(seg.N) if code >= V else (int(st[code - 1]) if code - 1 < st.size else int(seg.N))
        if op == '=':
            k0 = _bound_code(seg, col, val, 'left')
            k1 = _bound_code(seg, col, val, 'right')
            if k0 is None or k1 is None or k0 == k1:
                return 0, 0                      # literal absent -> empty
            lo, hi = max(lo, row_of(k0)), min(hi, row_of(k1))
        elif op == '>=':
            k = _bound_code(seg, col, val, 'left')
            if k is None: return 0, 0
            lo = max(lo, row_of(k))
        else:                                    # '<='
            k = _bound_code(seg, col, val, 'right')
            if k is None: return 0, 0
            hi = min(hi, row_of(k))
    return lo, max(lo, hi)


def _scan_eq(seg, col, code, lo, hi, negate=False):
    """Positions in [lo, hi) where the column's code equals (or differs from) `code`.
    enc=3: parallel decompress of the frames covering the span; other encodings fall back to the
    range read (bitpack touches covering bytes; sealed uses the cached full decode)."""
    c = seg.cols[col]
    if c.get('code_enc', 0) == 3 and col not in seg._codes:
        wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
        BR = c['BR']; base = c['cstart']; bo = c['boffs']; buf = seg.buf
        j0, j1 = lo // BR, (hi - 1) // BR + 1
        def scan(js):
            import zstandard as zstd
            dz = zstd.ZstdDecompressor(); out = []
            for j in js:
                raw = np.frombuffer(dz.decompress(buf[base+int(bo[j]):base+int(bo[j+1])].tobytes()), dtype=wdt)
                a, b = max(lo, j*BR), min(hi, j*BR + raw.size)
                seg_ = raw[a-j*BR:b-j*BR]
                h = np.nonzero(seg_ != code)[0] if negate else np.nonzero(seg_ == code)[0]
                if h.size: out.append(h + a)
            return np.concatenate(out) if out else np.empty(0, np.int64)
        W = min(_SCAN_THREADS, max(1, j1 - j0))
        with ThreadPoolExecutor(W) as ex:
            parts = list(ex.map(scan, np.array_split(np.arange(j0, j1), W)))
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.empty(0, np.int64)
    cc = np.asarray(seg._raw_codes_range(col, lo, hi))
    h = np.nonzero(cc != code)[0] if negate else np.nonzero(cc == code)[0]
    return h + lo

def _scan_flag(seg, col, flag, lo, hi):
    """Positions in [lo, hi) where flag[code] is set -- the LIKE frame scan: parallel enc=3
    decompress, one fancy-index per frame; the substring test happened once, in the dict."""
    c = seg.cols[col]
    if c.get('code_enc', 0) == 3 and col not in seg._codes:
        wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
        BR = c['BR']; base = c['cstart']; bo = c['boffs']; buf = seg.buf
        j0, j1 = lo // BR, (hi - 1) // BR + 1
        def scan(js):
            import zstandard as zstd
            dz = zstd.ZstdDecompressor(); out = []
            for j in js:
                raw = np.frombuffer(dz.decompress(buf[base+int(bo[j]):base+int(bo[j+1])].tobytes()), dtype=wdt)
                a, b = max(lo, j*BR), min(hi, j*BR + raw.size)
                h = np.nonzero(flag[raw[a-j*BR:b-j*BR]])[0]
                if h.size: out.append(h + a)
            return np.concatenate(out) if out else np.empty(0, np.int64)
        W = min(_SCAN_THREADS, max(1, j1 - j0))
        with ThreadPoolExecutor(W) as ex:
            parts = list(ex.map(scan, np.array_split(np.arange(j0, j1), W)))
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.empty(0, np.int64)
    cc = np.asarray(seg._raw_codes_range(col, lo, hi))
    return np.nonzero(flag[cc])[0] + lo


def execute(seg, spec):
    global _HITS
    lo, hi = _span_rows(seg, spec['spans'])
    likes = spec.get('likes', [])
    lflags = [(col, _like_flags(seg, col, needle), neg) for col, needle, neg in likes] if likes else []
    de, dl = spec.get('drive_eq'), spec.get('drive_like')
    if hi <= lo:
        pos = np.empty(0, np.int64)
    elif de is not None:
        col, val, _op = spec['eqs'][de]
        code = _code_of(seg, col, val)
        pos = _scan_eq(seg, col, int(code), lo, hi) if code is not None else np.empty(0, np.int64)
    elif dl is not None:
        col, fl, _n = lflags[dl]
        pos = _scan_flag(seg, col, fl, lo, hi)
    else:
        pos = np.arange(lo, hi, dtype=np.int64)
    for i, (col, val, op) in enumerate(spec['eqs']):
        if i == de or pos.size == 0: continue
        code = _code_of(seg, col, val)
        if code is None:
            if op == '=': pos = np.empty(0, np.int64)
            continue
        cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
        pos = pos[cc != code] if op == '<>' else pos[cc == code]
    for i, (col, fl, neg) in enumerate(lflags):
        if i == dl or pos.size == 0: continue
        cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
        m = fl[cc]
        pos = pos[~m] if neg else pos[m]
    for col, vals in spec.get('ins', ()):
        if pos.size == 0: break
        c = seg.cols[col]
        if c['mode'] == 4:
            want = np.array([int(v) for v in vals], dtype=np.int64)
            vv = np.asarray(seg._seq_decode(c))[pos]
            pos = pos[np.isin(vv, want)]
        else:
            codes = [_code_of(seg, col, v) for v in vals]
            codes = np.array([k for k in codes if k is not None], dtype=np.int64)
            if codes.size == 0:
                pos = np.empty(0, np.int64); break
            cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
            pos = pos[np.isin(cc, codes)]
    for col, val, op in spec['flags']:
        if pos.size == 0: break
        vv = np.asarray(seg._seq_decode(seg.cols[col]))[pos]
        pos = pos[vv != val] if op == '<>' else pos[vv == val]

    # ---- rows mode: SELECT * ordered by the cluster column -- positions ARE the order
    if spec.get('mode') == 'rows':
        K = spec['off'] + spec['lim']
        if spec.get('tiebreak') and pos.size > K:
            # total order: extend to the full plateau of the K-th row's cluster value,
            # decode, sort by the complete ORDER BY column list, then slice
            oc = np.asarray(seg.codes_at(spec['ocol'], pos)).astype(np.int64)
            kc = oc[K - 1]
            cut = int(np.searchsorted(oc, kc, side='right'))
            sel = pos[:cut]
        else:
            sel = pos[spec['off']: spec['off'] + spec['lim']]
        cols = list(seg.cols.keys())
        vals = []
        for cn in cols:
            c = seg.cols[cn]
            if c['mode'] == 4:
                vals.append(np.asarray(seg._seq_decode(c))[sel])
            else:
                cc = np.asarray(seg.codes_at(cn, sel)).astype(np.int64)
                vals.append([wdb_sql._pyval(seg.fetch(cn, int(k))) for k in cc])
        out = [tuple(vals[ci][ri] for ci in range(len(cols))) for ri in range(sel.size)]
        if spec.get('tiebreak') and len(out) > spec['lim']:
            oi = [cols.index(spec['ocol'])] + [cols.index(c) for c in spec['tiebreak']]
            out.sort(key=lambda r: tuple(('' if r[i] is None else r[i]) for i in oi))
            out = out[spec['off']: spec['off'] + spec['lim']]
        _HITS += 1
        return out, cols

    # ---- aggregates only (no GROUP BY)
    if not spec['keys']:
        row = []
        for _pi, kind, col in spec['aggs']:
            if kind == 'COUNT_STAR':
                row.append(int(pos.size)); continue
            if pos.size == 0:
                row.append(None); continue
            c = seg.cols[col]
            if kind == 'MIN_DICT':
                cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
                row.append(wdb_sql._pyval(seg.fetch(col, int(cc.min())))); continue
            if kind == 'COUNT_D':
                cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
                row.append(int(np.unique(cc).size)); continue
            if c['mode'] == 4:
                v = np.asarray(seg._seq_decode(c))[pos]
            else:
                cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
                v = np.asarray(seg._dict_ints(c))[cc] if c['mode'] == 2 else \
                    np.asarray([int(seg.fetch(col, int(k))) for k in np.unique(cc)])[
                        np.searchsorted(np.unique(cc), cc)]
            row.append(int(v.sum(dtype=np.int64)) if kind == 'SUM'
                       else float(v.sum(dtype=np.float64)) / v.size)
        _HITS += 1
        return [tuple(row)], [wdb_sql._alias(p) for p in spec['proj']]

    # ---- GROUP BY: gather keys at survivor positions, factorize-then-combine (overflow-safe)
    kcols = {}
    for _pi, k in spec['keys']:
        srcs = [k['src']] if k['kind'] == 'col' else [k['src']] + [c for c, _ in k['conds']]
        for s in srcs:
            if s in kcols: continue
            c = seg.cols[s]
            if c['mode'] == 4:
                kcols[s] = np.asarray(seg._seq_decode(c))[pos]
            else:
                kcols[s] = np.asarray(seg.codes_at(s, pos)).astype(np.int64)
    keyarr = []
    for _pi, k in spec['keys']:
        if k['kind'] == 'col':
            keyarr.append((k, kcols[k['src']]))
        else:
            m = np.ones(pos.size, bool)
            for c, v in k['conds']:
                if seg.cols[c]['mode'] == 4:
                    m &= kcols[c] == int(v)
                else:                            # dict cond: compare in CODE space
                    code = _code_of(seg, c, v)
                    m &= (kcols[c] == code) if code is not None else False
            keyarr.append((k, np.where(m, kcols[k['src']], -1)))
    comp = np.zeros(pos.size, dtype=np.int64)
    locals_ = []
    for _k, a in keyarr:
        u, inv = np.unique(a, return_inverse=True)
        comp = comp * u.size + inv
        locals_.append(u)
    g, ginv = np.unique(comp, return_inverse=True)
    cnt = np.bincount(ginv)
    order = np.lexsort((g, -cnt)) if spec['ordered'] else np.arange(g.size)
    sel = order[spec['off']: spec['off'] + spec['lim']] if spec['lim'] is not None else order[spec['off']:]
    rep = np.empty(sel.size, np.int64)
    for i, gi in enumerate(sel):
        rep[i] = int(np.nonzero(ginv == gi)[0][0])
    rows_out = []
    for i, gi in enumerate(sel):
        r = int(rep[i]); row = []
        for _pi, kind_or_key in sorted(
                [(pi, ('agg', kind, col)) for pi, kind, col in spec['aggs']] +
                [(pi, ('key', kk, aa)) for (pi, kk), (_x, aa) in zip(spec['keys'], keyarr)]):
            tag = kind_or_key[0]
            if tag == 'agg':
                _t, kind, col = kind_or_key
                if kind == 'COUNT_STAR':
                    row.append(int(cnt[gi]))
                elif kind == 'MIN_DICT':
                    grows = pos[ginv == gi]
                    cc2 = np.asarray(seg.codes_at(col, grows)).astype(np.int64)
                    row.append(wdb_sql._pyval(seg.fetch(col, int(cc2.min()))))
                elif kind == 'COUNT_D':
                    grows = pos[ginv == gi]
                    cc2 = np.asarray(seg.codes_at(col, grows)).astype(np.int64)
                    row.append(int(np.unique(cc2).size))
                else:
                    c = seg.cols[col]
                    grows = pos[ginv == gi]
                    if c['mode'] == 4:
                        v = np.asarray(seg._seq_decode(c))[grows]
                    else:
                        cc2 = np.asarray(seg.codes_at(col, grows)).astype(np.int64)
                        v = np.asarray(seg._dict_ints(c))[cc2]
                    row.append(int(v.sum(dtype=np.int64)) if kind == 'SUM'
                               else float(v.sum(dtype=np.float64)) / v.size)
            else:
                _t, kk, aa = kind_or_key
                a = int(aa[r])
                if kk['kind'] == 'case' and a < 0:
                    row.append(kk['default'] if isinstance(kk['default'], str) else str(kk['default']))
                else:
                    src = kk['src']; c = seg.cols[src]
                    row.append(int(a) if c['mode'] == 4 else wdb_sql._pyval(seg.fetch(src, a)))
        rows_out.append(tuple(row))
    _HITS += 1
    return rows_out, [wdb_sql._alias(p) for p in spec['proj']]
