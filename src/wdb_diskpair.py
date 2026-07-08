"""diskpair: disk-only N-key GROUP BY COUNT top-K -- the scan-merge read.

The proof that pair structures are optional, generalized to any key count. The bare shape
(SELECT k1..kN, COUNT(*) GROUP BY k1..kN ORDER BY count DESC LIMIT k, no WHERE) is answered by
streaming the blocked (enc=3) code frames -- never touching the codes cache, never consulting a
structure. Measured: Q16 (2-key) 0.70-0.84 s vs DuckDB 0.79-0.89, exact, wins.

Keys may be blocked dict columns or ONE derived minute-of(EventTime) key: the stair column's
step rows give each block's codes by searchsorted (no decode), and a 1.4M-entry code->minute
table (dict seconds // 60 % 60) turns them into the derived key -- the stripe insight: row
position IS the timestamp, so the third key is nearly free.

Pipeline: parallel per-thread block scan -> progressive radix key fold (declines if the key
space overflows i64) -> [2-key only] NORM SPLIT: a dominant first-block code (SearchPhrase's ''
at 86.8%) collapses its rows to the other key: dense bincount lane, vectorized merge -> per-
thread sorted partials -> wdb_kernels.kway_topk: numba loser-tree merge streaming equal-key
accumulation into a top-K heap, merged stream never materialized -> winners decoded by fetches.

Placement: AFTER gridwalk. Resident structures answer in ~2.5 ms and should; this is the floor
beneath them. Residency stays a promotion, never a requirement.
"""
import numpy as np
import zstandard as zstd
import sqlglot.expressions as E
from concurrent.futures import ThreadPoolExecutor
import wdb_sql
import wdb_policies as P
import wdb_kernels as K

_ENABLED = True
_HITS = 0
_SCAN_THREADS = 14
_NORM_MIN_SHARE = 0.5
_MAX_KEYS = 4


def enable():
    global _ENABLED; _ENABLED = True


def disable():
    global _ENABLED; _ENABLED = False


def _minute_key(p, seg, col_map):
    """extract(minute FROM col) over a stair datetime column -> derived key spec."""
    inner = p.this if isinstance(p, E.Alias) else p
    if not isinstance(inner, E.Extract):
        return None
    unit = inner.this.name.lower() if hasattr(inner.this, 'name') else str(inner.this).lower()
    if unit != 'minute' or not isinstance(inner.expression, E.Column):
        return None
    col = inner.expression.name
    col = col_map.get(col, col) if col_map else col
    if not P.columns_exist(seg, col):
        return None
    if seg.stairs(col) is None or seg.cols[col].get('dt') != 3:
        return None
    return {'kind': 'minute', 'src': col, 'V': 60}


def detect(seg, tree, col_map):
    if not _ENABLED:                    return None
    if not P.no_joins(tree):            return None
    if not P.no_select_distinct(tree):  return None
    if not P.no_having(tree):           return None
    if not P.no_deleted_rows(seg):      return None
    if tree.args.get('where') is not None:
        return None
    proj = tree.expressions
    keys, cnt_pi = [], None
    for pi, p in enumerate(proj):
        ak = wdb_sql._agg_kind(p)
        if ak is not None:
            if ak[0] != 'COUNT_STAR' or cnt_pi is not None:
                return None
            cnt_pi = pi
            continue
        mk = _minute_key(p, seg, col_map)
        if mk is not None:
            keys.append((pi, mk))
            continue
        nm = wdb_sql._proj_colname(p)
        if nm is None:
            return None
        col = col_map.get(nm, nm) if col_map else nm
        if not P.columns_exist(seg, col):   return None
        if seg._effective(col) is not None: return None
        c = seg.cols[col]
        if c.get('mode') not in (0, 1, 2) or c.get('code_enc') != 3:
            return None
        keys.append((pi, {'kind': 'col', 'src': col, 'V': int(c['V'])}))
    if cnt_pi is None or not (2 <= len(keys) <= _MAX_KEYS):
        return None
    if sum(1 for _pi, k in keys if k['kind'] == 'minute') > 1:
        return None
    span = 1
    for _pi, k in keys:
        span *= k['V']
        if span > (1 << 62):
            return None                 # composite would overflow the i64 key fold
    group = tree.args.get('group')
    if group is None or len(group.expressions) != len(keys):
        return None
    order = tree.args.get('order')
    if order is None or len(order.expressions) != 1:
        return None
    oe = order.expressions[0]
    if not oe.args.get('desc'):
        return None
    onm = oe.this
    if isinstance(onm, E.Column):
        if onm.name not in (wdb_sql._alias(proj[cnt_pi]), 'COUNT', 'count'):
            return None
    elif wdb_sql._agg_kind(onm) is None or wdb_sql._agg_kind(onm)[0] != 'COUNT_STAR':
        return None
    lim = wdb_sql._limit(tree)
    if lim is None or lim <= 0 or lim > 10000:
        return None
    off = wdb_sql._offset(tree) or 0
    return {'keys': keys, 'cnt_pi': cnt_pi, 'lim': int(lim), 'off': int(off), 'proj': proj}


def _modal_share(seg, col):
    c = seg.cols[col]
    dz = zstd.ZstdDecompressor()
    bo = c['boffs']; base = c['cstart']
    wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
    raw = np.frombuffer(dz.decompress(seg.buf[base + int(bo[0]):base + int(bo[1])].tobytes()), dtype=wdt)
    bc = np.bincount(raw)
    m = int(bc.argmax())
    return m, bc[m] / raw.size


def _block_geometry(seg, keys):
    """(NB, BR) from the first blocked key column -- all enc=3 columns share BLOCK_ROWS."""
    for _pi, k in keys:
        if k['kind'] == 'col':
            c = seg.cols[k['src']]
            return c['boffs'].size - 1, int(c['BR'])
    return 0, 0


def execute(seg, spec):
    global _HITS
    keys = spec['keys']
    NB, BR = _block_geometry(seg, keys)
    if NB == 0:
        return None
    N = int(seg.N)
    # per-key block readers
    readers = []
    minute_tab = None
    for _pi, k in keys:
        if k['kind'] == 'minute':
            st = seg.stairs(k['src'])
            secs = np.asarray(seg._dict_ints(seg.cols[k['src']]), dtype=np.int64)
            minute_tab = ((secs // 60) % 60).astype(np.int64)
            readers.append(('minute', st))
        else:
            c = seg.cols[k['src']]
            readers.append(('col', (c['boffs'], c['cstart'],
                                    {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']])))
    # norm lane: 2 plain dict keys only
    norm_idx = -1; norm_code = -1
    if len(keys) == 2 and all(k['kind'] == 'col' for _pi, k in keys):
        m0, s0 = _modal_share(seg, keys[0][1]['src'])
        m1, s1 = _modal_share(seg, keys[1][1]['src'])
        if s1 >= _NORM_MIN_SHARE and s1 >= s0:
            norm_idx, norm_code = 1, m1
        elif s0 >= _NORM_MIN_SHARE:
            norm_idx, norm_code = 0, m0
    radix = [k['V'] for _pi, k in keys]
    buf = seg.buf
    K10 = spec['off'] + spec['lim']

    def work(js):
        dz = zstd.ZstdDecompressor()
        norm_vals, exc = [], []
        for j in js:
            a, b = j * BR, min(N, (j + 1) * BR)
            cols = []
            for kind, meta in readers:
                if kind == 'minute':
                    codes = np.searchsorted(meta, np.arange(a, b), side='right')
                    cols.append(minute_tab[codes])
                else:
                    bo, base, wdt = meta
                    cols.append(np.frombuffer(
                        dz.decompress(buf[base + int(bo[j]):base + int(bo[j + 1])].tobytes()),
                        dtype=wdt))
            key = cols[0].astype(np.int64)
            for ci in range(1, len(cols)):
                key = key * radix[ci] + cols[ci]
            if norm_idx >= 0:
                nm = cols[norm_idx] == norm_code
                oth = cols[1 - norm_idx]
                norm_vals.append(np.asarray(oth[nm]))
                exc.append(key[~nm])
            else:
                exc.append(key)
        tab = (np.bincount(np.concatenate(norm_vals), minlength=radix[1 - norm_idx]).astype(np.int32)
               if norm_idx >= 0 and norm_vals else None)
        ek = np.concatenate(exc) if exc else np.empty(0, np.int64)
        g, c = np.unique(ek, return_counts=True)
        return tab, g, c.astype(np.int64)

    W = min(_SCAN_THREADS, NB) or 1
    with ThreadPoolExecutor(W) as ex:
        parts = list(ex.map(work, np.array_split(np.arange(NB), W)))
    kk = np.concatenate([p[1] for p in parts])
    vv = np.concatenate([p[2] for p in parts])
    offs = np.zeros(len(parts) + 1, np.int64)
    np.cumsum([p[1].size for p in parts], out=offs[1:])
    tc, tk = K.kway_topk(kk, vv, offs, K10)
    cand = [(int(tc[i]), int(tk[i])) for i in range(K10) if tc[i] > 0]
    if norm_idx >= 0:
        etab = None
        for p in parts:
            if p[0] is not None:
                etab = p[0] if etab is None else etab + p[0]
        if etab is not None:
            if K.HAVE_NUMBA:
                ec, ek2 = K.top10_i32(etab, K10)
            else:
                ti = np.argpartition(-etab, min(K10, etab.size - 1))[:K10]
                ec, ek2 = etab[ti].astype(np.int64), ti.astype(np.int64)
            for i in range(len(ec)):
                if ec[i] > 0:
                    oc = int(ek2[i])
                    comp = (oc * radix[1] + norm_code) if norm_idx == 1 else (norm_code * radix[1] + oc)
                    cand.append((int(ec[i]), comp))
    cand.sort(key=lambda x: (-x[0], x[1]))
    cand = cand[spec['off']: spec['off'] + spec['lim']]
    out = []
    for n, comp in cand:
        parts_k = []
        rem = comp
        for ci in range(len(keys) - 1, -1, -1):
            parts_k.append(rem % radix[ci]); rem //= radix[ci]
        parts_k.reverse()
        row = [None] * len(spec['proj'])
        for (pi, k), code in zip(keys, parts_k):
            if k['kind'] == 'minute':
                row[pi] = int(code)
            else:
                row[pi] = wdb_sql._pyval(seg.fetch(k['src'], int(code)))
        row[spec['cnt_pi']] = n
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
