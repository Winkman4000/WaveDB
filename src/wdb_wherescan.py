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
    """Python value of a Literal, unwrapping CAST('..' AS DATE/TIMESTAMP) shells."""
    n = node
    while isinstance(n, E.Cast):
        n = n.this
    if isinstance(n, E.Literal):
        return wdb_sql._literal_value(n)
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
    if isinstance(a, E.Column) and isinstance(b, (E.Literal, E.Cast)):
        return a.name, _litval(b), op
    if isinstance(b, E.Column) and isinstance(a, (E.Literal, E.Cast)):
        flip = {'>=': '<=', '<=': '>='}
        return b.name, _litval(a), flip.get(op, op)
    return None

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


def detect(seg, tree, col_map):
    if not _ENABLED:                    return None
    if not P.no_joins(tree):            return None
    if not P.no_select_distinct(tree):  return None
    if not P.no_having(tree):           return None
    if not P.no_deleted_rows(seg):      return None
    where = tree.args.get('where')
    if where is None:
        return None
    spans, eqs, flags = [], [], []
    for cn in _conjuncts(where.this):
        cl = _col_lit(cn)
        if cl is None:
            return None
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
    if not eqs and not spans:
        return None                      # unselective flag-only shapes stay with the scan family
    if eqs and eqs[0][2] != '=':
        return None                      # the driving scan predicate must be an equality
    # projections: plain key columns, COUNT(*)/SUM/AVG aggregates, or the CASE derived key
    proj = tree.expressions
    keys, aggs = [], []
    for pi, p in enumerate(proj):
        ak = wdb_sql._agg_kind(p)
        if ak is not None:
            if ak[0] == 'COUNT_STAR':
                aggs.append((pi, 'COUNT_STAR', None)); continue
            if ak[0] not in ('SUM', 'AVG') or not isinstance(ak[1], str):
                return None
            col = col_map.get(ak[1], ak[1]) if col_map else ak[1]
            if not P.columns_exist(seg, col):   return None
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
    return {'spans': spans, 'eqs': eqs, 'flags': flags, 'keys': keys, 'aggs': aggs,
            'proj': proj, 'ordered': order is not None, 'lim': lim, 'off': int(off)}

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

def execute(seg, spec):
    global _HITS
    lo, hi = _span_rows(seg, spec['spans'])
    if hi <= lo:
        pos = np.empty(0, np.int64)
    elif spec['eqs']:
        col, val, _op = spec['eqs'][0]
        code = _code_of(seg, col, val)
        if code is None:
            pos = np.empty(0, np.int64)
        else:
            pos = _scan_eq(seg, col, int(code), lo, hi)
        for col, val, op in spec['eqs'][1:]:
            if pos.size == 0: break
            code = _code_of(seg, col, val)
            if code is None:
                if op == '=': pos = np.empty(0, np.int64)
                continue
            cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
            pos = pos[cc != code] if op == '<>' else pos[cc == code]
    else:
        pos = np.arange(lo, hi, dtype=np.int64)
    for col, val, op in spec['flags']:
        if pos.size == 0: break
        vv = np.asarray(seg._seq_decode(seg.cols[col]))[pos]
        pos = pos[vv != val] if op == '<>' else pos[vv == val]

    # ---- aggregates only (no GROUP BY)
    if not spec['keys']:
        row = []
        for _pi, kind, col in spec['aggs']:
            if kind == 'COUNT_STAR':
                row.append(int(pos.size)); continue
            if pos.size == 0:
                row.append(None); continue
            c = seg.cols[col]
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
