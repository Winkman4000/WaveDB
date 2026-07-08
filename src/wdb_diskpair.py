"""diskpair: disk-only two-key GROUP BY COUNT top-K -- the scan-merge read.

The proof that pair structures are optional: the bare Q16 shape (SELECT a, b, COUNT(*) GROUP BY
a, b ORDER BY count DESC LIMIT k, no WHERE) answered by streaming the blocked (enc=3) code
frames -- never touching the codes cache, never consulting a structure. Measured on ClickBench
Q16: 0.70-0.75 s vs DuckDB's 0.79-0.80, exact, from a 2.58 s first draft in five profiled rounds.

Pipeline:
  1. Parallel block scan (per-thread zstd): decompress both columns' frames.
  2. NORM SPLIT -- the norm/exception law applied to counting: if one column has a dominant code
     (SearchPhrase's '' covers 86.8% of rows), those rows' pair keys collapse to the other
     column's code alone, so they take a dense bincount lane (per-thread table, one vectorized
     sum to merge) and never enter the sort. Exceptions (13.2M rows) compose real pair keys.
  3. Per-thread np.unique -> K sorted partial runs.
  4. wdb_kernels.kway_topk: loser-tree merge streaming straight into a top-K heap -- the merged
     array is never materialized. Norm-lane top-K via compiled dense scan.
  5. Winners (<= 2K codes) decode by point fetches.

Placement: AFTER gridwalk -- a resident structure answers in ~2.5 ms and should. This read is
the floor beneath it: any segment without structures gets DuckDB-beating pair queries anyway.
Residency stays a promotion, never a requirement.

Declines: WHERE (wherescan's territory), >2 keys (v2), aggregates beyond COUNT(*), non-dict
key columns, deleted rows, overrides.
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
_NORM_MIN_SHARE = 0.5              # first-block modal share needed to open the bincount lane


def enable():
    global _ENABLED; _ENABLED = True


def disable():
    global _ENABLED; _ENABLED = False


def detect(seg, tree, col_map):
    if not _ENABLED:                    return None
    if not P.no_joins(tree):            return None
    if not P.no_select_distinct(tree):  return None
    if not P.no_having(tree):           return None
    if not P.no_deleted_rows(seg):      return None
    if tree.args.get('where') is not None:
        return None
    proj = tree.expressions
    if len(proj) != 3:
        return None
    keys, cnt_pi = [], None
    for pi, p in enumerate(proj):
        ak = wdb_sql._agg_kind(p)
        if ak is not None:
            if ak[0] != 'COUNT_STAR' or cnt_pi is not None:
                return None
            cnt_pi = pi
            continue
        nm = wdb_sql._proj_colname(p)
        if nm is None:
            return None
        col = col_map.get(nm, nm) if col_map else nm
        if not P.columns_exist(seg, col):   return None
        if seg._effective(col) is not None: return None
        c = seg.cols[col]
        if c.get('mode') not in (0, 1, 2) or c.get('code_enc') != 3:
            return None                 # both keys must be blocked dict columns
        keys.append((pi, col))
    if len(keys) != 2 or cnt_pi is None:
        return None
    group = tree.args.get('group')
    if group is None or len(group.expressions) != 2:
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
    """(modal_code, share) from the column's first block -- the norm-lane probe."""
    c = seg.cols[col]
    dz = zstd.ZstdDecompressor()
    bo = c['boffs']; base = c['cstart']
    wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
    raw = np.frombuffer(dz.decompress(seg.buf[base + int(bo[0]):base + int(bo[1])].tobytes()), dtype=wdt)
    bc = np.bincount(raw)
    m = int(bc.argmax())
    return m, bc[m] / raw.size


def execute(seg, spec):
    global _HITS
    (piA, colA), (piB, colB) = spec['keys']
    cA, cB = seg.cols[colA], seg.cols[colB]
    VA, VB = int(cA['V']), int(cB['V'])
    # norm lane: pick the key column with a dominant first-block code (or none)
    mB, shB = _modal_share(seg, colB)
    mA, shA = _modal_share(seg, colA)
    if shB >= _NORM_MIN_SHARE and shB >= shA:
        norm_col, norm_code, other = colB, mB, colA
    elif shA >= _NORM_MIN_SHARE:
        norm_col, norm_code, other = colA, mA, colB
    else:
        norm_col, norm_code, other = None, -1, None
    wA = {1: np.uint8, 2: np.uint16, 4: np.uint32}[cA['cwidth']]
    wB = {1: np.uint8, 2: np.uint16, 4: np.uint32}[cB['cwidth']]
    boA, baA = cA['boffs'], cA['cstart']
    boB, baB = cB['boffs'], cB['cstart']
    buf = seg.buf
    NB = boA.size - 1
    Vnorm = int(seg.cols[other]['V']) if norm_col else 0
    K10 = spec['off'] + spec['lim']

    def work(js):
        dz = zstd.ZstdDecompressor()
        norm_vals, exc = [], []
        for j in js:
            a = np.frombuffer(dz.decompress(buf[baA + int(boA[j]):baA + int(boA[j + 1])].tobytes()), dtype=wA)
            b = np.frombuffer(dz.decompress(buf[baB + int(boB[j]):baB + int(boB[j + 1])].tobytes()), dtype=wB)
            key = a.astype(np.int64) * VB + b
            if norm_col is not None:
                nm = (b == norm_code) if norm_col == colB else (a == norm_code)
                oth = a if norm_col == colB else b
                norm_vals.append(oth[nm])
                exc.append(key[~nm])
            else:
                exc.append(key)
        tab = (np.bincount(np.concatenate(norm_vals), minlength=Vnorm).astype(np.int32)
               if norm_col is not None and norm_vals else None)
        ek = np.concatenate(exc) if exc else np.empty(0, np.int64)
        g, c = np.unique(ek, return_counts=True)
        return tab, g, c.astype(np.int64)

    W = min(_SCAN_THREADS, NB) or 1
    with ThreadPoolExecutor(W) as ex:
        parts = list(ex.map(work, np.array_split(np.arange(NB), W)))
    keys = np.concatenate([p[1] for p in parts])
    vals = np.concatenate([p[2] for p in parts])
    offs = np.zeros(len(parts) + 1, np.int64)
    np.cumsum([p[1].size for p in parts], out=offs[1:])
    tc, tk = K.kway_topk(keys, vals, offs, K10)
    cand = [(int(tc[i]), int(tk[i] // VB), int(tk[i] % VB)) for i in range(K10) if tc[i] > 0]
    if norm_col is not None:
        etab = None
        for p in parts:
            if p[0] is not None:
                etab = p[0] if etab is None else etab + p[0]
        if etab is not None:
            ec, ek2 = (K.top10_i32(etab, K10) if K.HAVE_NUMBA else
                       (lambda ti: (etab[ti].astype(np.int64), ti.astype(np.int64)))(
                           np.argpartition(-etab, min(K10, etab.size - 1))[:K10]))
            for i in range(min(K10, len(ec))):
                if ec[i] > 0:
                    oc = int(ek2[i])
                    if norm_col == colB:
                        cand.append((int(ec[i]), oc, norm_code))
                    else:
                        cand.append((int(ec[i]), norm_code, oc))
    cand.sort(key=lambda x: (-x[0], x[1], x[2]))
    cand = cand[spec['off']: spec['off'] + spec['lim']]
    out = []
    for n, ka, kb in cand:
        row = [None, None, None]
        row[spec['keys'][0][0]] = wdb_sql._pyval(seg.fetch(colA, ka))
        row[spec['keys'][1][0]] = wdb_sql._pyval(seg.fetch(colB, kb))
        row[spec['cnt_pi']] = n
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
