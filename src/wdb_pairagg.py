"""wdb_pairagg -- filtered 2-key GROUP BY, top-K by COUNT(*), with COUNT/SUM/AVG payload, served by a
parallel SPARSE hash aggregate (numba).

The shape the dense group-by (fused_agg) loses on: two group keys with millions of distinct pairs, an
optional single-column WHERE, COUNT(*) plus per-group SUM/AVG, ordered by count with a LIMIT. The dense
path builds a bin for every possible group and sorts them all; here we keep a bin only for pairs that
actually occur (open-addressing hash), fold the payload sums into the same pass, and track the top-K
live so the winners fall out with no post-scan. Rows are partitioned across shards by key so each core
owns a disjoint, cache-resident table -- no cross-core merge of counts.

Scope (v1, conservative -- declines to fused_agg on anything else, which stays correct):
  - exactly 2 bare dictionary-coded group keys (mode != 4/affine);
  - projections = the 2 keys + COUNT(*) and any number of SUM(col)/AVG(col) over int-valued columns;
  - optional WHERE `col = literal` or `col <> literal` on a dict column (NULL-safe exact code match);
  - ORDER BY COUNT(*) DESC and a LIMIT;
  - declines when the LIMIT boundary is tied (returned set is the unique correct top-K), or on any
    non-int payload / unsupported predicate / other shape.
"""
import numpy as np
import wdb_sql
import workers
import wdb_policies as P
from numba import njit, prange
E = wdb_sql.E

_ENABLED = True
_HITS = 0
_NT = 16
_NSHARD = 64
_TSIZE = 1 << 18


def enable():
    global _ENABLED; _ENABLED = True


def disable():
    global _ENABLED; _ENABLED = False


def is_enabled():
    return _ENABLED


@njit(parallel=True, cache=True)
def _mask_to_idx(mask, T):
    n = mask.size; chunk = (n + T - 1) // T; counts = np.zeros(T, np.int64)
    for t in prange(T):
        lo = t * chunk; hi = min(lo + chunk, n); cc = 0
        for i in range(lo, hi):
            if mask[i]: cc += 1
        counts[t] = cc
    off = np.zeros(T + 1, np.int64)
    for t in range(T): off[t + 1] = off[t] + counts[t]
    idx = np.empty(off[T], np.int64)
    for t in prange(T):
        lo = t * chunk; hi = min(lo + chunk, n); pos = off[t]
        for i in range(lo, hi):
            if mask[i]: idx[pos] = i; pos += 1
    return idx


@njit(parallel=True, cache=True)
def _gather_partition(idx, ca, cb, Vb, pay, S, NT):
    n = idx.size; Pn = pay.shape[1]
    G = np.empty(n, np.int64); PY = np.empty((n, Pn), np.int64)
    chunk = (n + NT - 1) // NT
    lhist = np.zeros((NT, S), np.int64)
    for w in prange(NT):
        lo = w * chunk; hi = min(lo + chunk, n)
        for j in range(lo, hi):
            r = idx[j]; g = ca[r] * Vb + cb[r]; G[j] = g
            for p in range(Pn): PY[j, p] = pay[r, p]
            lhist[w, g % S] += 1
    base = np.zeros(S + 1, np.int64)
    for s in range(S):
        tot = 0
        for w in range(NT): tot += lhist[w, s]
        base[s + 1] = base[s] + tot
    woff = np.zeros((NT, S), np.int64)
    for s in range(S):
        acc = base[s]
        for w in range(NT):
            woff[w, s] = acc; acc += lhist[w, s]
    PG = np.empty(n, np.int64); PPY = np.empty((n, Pn), np.int64)
    for w in prange(NT):
        lo = w * chunk; hi = min(lo + chunk, n); cur = woff[w].copy()
        for j in range(lo, hi):
            g = G[j]; s = g % S; pos = cur[s]; PG[pos] = g
            for p in range(Pn): PPY[pos, p] = PY[j, p]
            cur[s] = pos + 1
    return PG, PPY, base


@njit(parallel=True, cache=True)
def _agg_shards(PG, PPY, base, S, tsize, K):
    Pn = PPY.shape[1]
    rk = np.full((S, K), -1, np.int64); rc = np.zeros((S, K), np.int64)
    rp = np.zeros((S, K, Pn), np.int64)
    for s in prange(S):
        lo = base[s]; hi = base[s + 1]; m = tsize - 1
        keys = np.full(tsize, -1, np.int64); cnts = np.zeros(tsize, np.int64)
        psum = np.zeros((tsize, Pn), np.int64); tpos = np.full(tsize, -1, np.int64)
        tk_h = np.full(K, -1, np.int64); tk_c = np.zeros(K, np.int64)
        filled = 0; curmin = 0; minpos = 0
        for j in range(lo, hi):
            g = PG[j]; h = (g * 2654435761) & m
            while True:
                k = keys[h]
                if k == -1: keys[h] = g; cnts[h] = 1; break
                elif k == g: cnts[h] += 1; break
                else: h = (h + 1) & m
            for p in range(Pn): psum[h, p] += PPY[j, p]
            c = cnts[h]; pos = tpos[h]
            if pos >= 0:
                tk_c[pos] = c
                if pos == minpos:
                    mn = 0
                    for jj in range(1, filled):
                        if tk_c[jj] < tk_c[mn]: mn = jj
                    minpos = mn; curmin = tk_c[mn]
            elif filled < K:
                tk_c[filled] = c; tk_h[filled] = h; tpos[h] = filled; filled += 1
                if filled == K:
                    mn = 0
                    for jj in range(1, K):
                        if tk_c[jj] < tk_c[mn]: mn = jj
                    minpos = mn; curmin = tk_c[mn]
            elif c > curmin:
                oh = tk_h[minpos]; tpos[oh] = -1
                tk_c[minpos] = c; tk_h[minpos] = h; tpos[h] = minpos
                mn = 0
                for jj in range(1, K):
                    if tk_c[jj] < tk_c[mn]: mn = jj
                minpos = mn; curmin = tk_c[mn]
        for j in range(filled):
            hh = tk_h[j]; rk[s, j] = keys[hh]; rc[s, j] = cnts[hh]
            for p in range(Pn): rp[s, j, p] = psum[hh, p]
    return rk, rc, rp


def _dict_col(seg, phys):
    c = seg.cols.get(phys)
    return c is not None and c.get('mode') != 4


def _single_col_eq_predicate(tree):
    w = tree.args.get('where')
    if w is None:
        return ('__none__', None, None)
    cond = w.this
    if isinstance(cond, (E.EQ, E.NEQ)):
        left, right = cond.this, cond.expression
        if isinstance(left, E.Column) and isinstance(right, E.Literal):
            lit = right.this
            if not right.args.get('is_string'):
                try: lit = int(lit)
                except Exception:
                    try: lit = float(lit)
                    except Exception: return None
            op = '=' if isinstance(cond, E.EQ) else '<>'
            return (left.name, op, lit)
    return None


def _order_targets_count(o0, proj, ci):
    ak = wdb_sql._agg_kind(o0.this)
    return ak is not None and ak[0] == 'COUNT_STAR'


def detect(seg, tree, col_map):
    if not _ENABLED:
        return None
    if not P.no_deleted_rows(seg):
        return None
    if not P.has_limit(tree) or wdb_sql._offset(tree):
        return None
    if tree.args.get('having') or tree.args.get('distinct'):
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 2:
        return None
    proj = tree.expressions
    keys = []; aggs = []
    for i, p in enumerate(proj):
        ak = wdb_sql._agg_kind(p)
        if ak is None:
            nm = wdb_sql._proj_colname(p)
            if nm is None:
                return None
            keys.append((nm, i))
        elif ak[0] == 'COUNT_STAR':
            aggs.append(('COUNT_STAR', None, i))
        elif ak[0] in ('SUM', 'AVG') and isinstance(ak[1], str):
            aggs.append((ak[0], ak[1], i))
        else:
            return None
    if len(keys) != 2 or not any(a[0] == 'COUNT_STAR' for a in aggs):
        return None
    knames = [k[0] for k in keys]
    gnames = [wdb_sql._proj_colname(ge if not isinstance(ge, E.Alias) else ge.this)
              for ge in g.expressions]
    if any(x is None for x in gnames) or set(knames) != set(gnames):
        return None
    order = tree.args.get('order')
    if order is None or not order.expressions:
        return None
    ci = next(a[2] for a in aggs if a[0] == 'COUNT_STAR')
    o0 = order.expressions[0]
    if not o0.args.get('desc'):
        return None
    on = wdb_sql._proj_colname(o0.this) if isinstance(o0.this, E.Column) else None
    if on != wdb_sql._alias(proj[ci]) and on != 'count' and not _order_targets_count(o0, proj, ci):
        return None
    kcols = [col_map.get(n, n) if col_map else n for n in knames]
    if not all(_dict_col(seg, c) for c in kcols):
        return None
    paycols = [col_map.get(a[1], a[1]) if col_map else a[1] for a in aggs if a[0] in ('SUM', 'AVG')]
    for pc in paycols:                       # payload: ANY storage mode (values decoded later), but int-valued
        if seg.cols.get(pc) is None:
            return None
        if np.asarray(seg._typed_dict(pc)).dtype.kind not in ('i', 'u'):
            return None
    pred = _single_col_eq_predicate(tree)
    if pred is None:
        return None
    if pred[0] != '__none__':
        pcol = col_map.get(pred[0], pred[0]) if col_map else pred[0]
        if not _dict_col(seg, pcol):
            return None
    return {'proj': proj, 'keys': keys, 'aggs': aggs, 'ci': ci, 'knames': knames, 'kcols': kcols,
            'lim': wdb_sql._limit(tree), 'order': order, 'pred': pred, 'col_map': col_map}


def _build_mask(seg, pcol, op, lit):
    V = np.asarray(seg._typed_dict(pcol))
    key = lit.encode('utf-8', 'surrogatepass') if (V.dtype.kind == 'S' and isinstance(lit, str)) else lit
    codes = seg._raw_codes(pcol)
    pos = int(np.searchsorted(V, key))
    found = 0 <= pos < V.size and V[pos] == key
    if op == '=':
        return (codes == pos) if found else np.zeros(codes.size, dtype=bool)
    return (codes != pos) if found else np.ones(codes.size, dtype=bool)


def execute(seg, spec):
    global _HITS
    kcols = spec['kcols']; a, b = kcols[0], kcols[1]
    lim = spec['lim']
    caF = seg._raw_codes(a).astype(np.int64)
    cbF = seg._raw_codes(b).astype(np.int64)
    Vb = int(cbF.max()) + 1 if cbF.size else 1
    if caF.size and int(caF.max()) * Vb + int(cbF.max()) >= (1 << 62):
        return None                          # gid would overflow int64 -> fused_agg
    pred = spec['pred']
    if pred[0] == '__none__':
        idx = np.arange(caF.size, dtype=np.int64)
    else:
        pcol = spec['col_map'].get(pred[0], pred[0]) if spec['col_map'] else pred[0]
        mask = _build_mask(seg, pcol, pred[1], pred[2])
        idx = _mask_to_idx(mask, _NT)
    if idx.size == 0:
        return []
    pay_specs = [(a2[0], a2[1]) for a2 in spec['aggs'] if a2[0] in ('SUM', 'AVG')]
    payphys = [spec['col_map'].get(c, c) if spec['col_map'] else c for _, c in pay_specs]
    Pn = max(1, len(payphys))
    payF = np.zeros((caF.size, Pn), np.int64)
    for p, pc in enumerate(payphys):
        vals = np.asarray(seg.values(pc))    # canonical mode-agnostic decode (handles affine/mode-4)
        if vals.dtype.kind not in ('i', 'u'):
            return None                      # non-integer payload -> fused_agg
        payF[:, p] = vals.astype(np.int64)
    PG, PPY, base = _gather_partition(idx, caF, cbF, Vb, payF, _NSHARD, _NT)
    rk, rc, rp = _agg_shards(PG, PPY, base, _NSHARD, _TSIZE, lim + 1)
    fk = rk.reshape(-1); fc = rc.reshape(-1); fp = rp.reshape(-1, Pn)
    good = fc > 0
    fk = fk[good]; fc = fc[good]; fp = fp[good]
    if fk.size == 0:
        return []
    take = min(lim + 1, fk.size)
    part = np.argpartition(fc, -take)[-take:]
    o = part[np.argsort(fc[part], kind='stable')[::-1]]
    if o.size > lim and int(fc[o[lim - 1]]) == int(fc[o[lim]]):
        return None
    o = o[:lim]
    sel_gid = fk[o]; sel_cnt = fc[o].astype(np.int64); sel_pay = fp[o]
    codesA = sel_gid // Vb; codesB = sel_gid - codesA * Vb
    VA = np.asarray(seg._typed_dict(a)); VB = np.asarray(seg._typed_dict(b))
    valA = [wdb_sql._pyval(x) for x in VA[codesA.astype(np.intp)]]
    valB = [wdb_sql._pyval(x) for x in VB[codesB.astype(np.intp)]]
    keyval = {a: valA, b: valB}
    kn2phys = dict(zip(spec['knames'], kcols))
    proj = spec['proj']
    cnt_list = sel_cnt.tolist()
    rows = []
    for r in range(sel_gid.size):
        row = []; payptr = 0
        for p in proj:
            ak = wdb_sql._agg_kind(p)
            if ak is None:
                nm = wdb_sql._proj_colname(p); phys = kn2phys[nm]
                row.append(keyval[phys][r])
            elif ak[0] == 'COUNT_STAR':
                row.append(cnt_list[r])
            elif ak[0] == 'SUM':
                row.append(int(sel_pay[r, payptr])); payptr += 1
            else:
                row.append(float(sel_pay[r, payptr]) / float(cnt_list[r])); payptr += 1
        rows.append(tuple(row))
    rows = workers.finalize(rows, proj, spec['order'], lim)
    _HITS += 1
    return rows
