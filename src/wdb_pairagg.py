"""wdb_pairagg -- filtered 2-key GROUP BY, top-K by COUNT(*), with COUNT/SUM/AVG payload, served by a
parallel SPARSE hash aggregate (numba).

The shape the dense group-by (fused_agg) loses on: two group keys with millions of distinct pairs, an
optional single-column WHERE, COUNT(*) plus per-group SUM/AVG, ordered by count with a LIMIT. The dense
path builds a bin for every possible group and sorts them all; here we keep a bin only for pairs that
actually occur (open-addressing hash), fold the payload sums into the same pass, then take the exact
top-K over the (few million) real pairs in numpy. Rows are partitioned across shards by key so each
core owns a disjoint, cache-resident table -- no cross-core merge of counts.

Scope (v1, conservative -- declines to fused_agg on anything else, which stays correct):
  - exactly 2 bare dictionary-coded group keys (mode != 4/affine);
  - projections = the 2 keys + COUNT(*) and any number of SUM(col)/AVG(col) over int-valued columns;
  - optional WHERE `col = literal` or `col <> literal` on a dict column (NULL-safe exact code match);
  - ORDER BY COUNT(*) DESC and a LIMIT;
  - boundary ties resolved deterministically (count DESC, then gid ASC) over the full sparse distinct
    set -- exact top-K even when the LIMIT boundary is tied.
Declines (execute -> None -> fused_agg) only on non-integer payload, gid overflow, empty survivors,
or any shape outside the above; detect declines the rest.
"""
import numpy as np
import wdb_sql
import wdb_compound
import workers
import wdb_policies as P
from numba import njit, prange
E = wdb_sql.E

_ENABLED = True
_HITS = 0
_NT = 16
_NSHARD = 64
_TSIZE = 1 << 18
_SORT_MIN = 30_000_000   # above this many survivors the sparse hash oversubscribes (near-unique keys);
                         # switch to sort + adjacent-run scan, which never melts. Q32 is the case.


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
            r = idx[j]; g = np.int64(ca[r]) * Vb + np.int64(cb[r]); G[j] = g
            for p in range(Pn): PY[j, p] = pay[j, p]     # pay is SURVIVOR-aligned (compact)
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
def _agg_shards(PG, PPY, base, S, tsize):
    """Per-shard open-addressing hash agg (count + P payload sums), then compact ALL distinct pairs.
    Shard s writes its distinct (gid, count, payload) into its own partition region starting at
    base[s]; ndist[s] records how many. Returning the full distinct set (not a per-shard top-K) lets
    the exact top-K + deterministic tie-break happen in numpy over the sparse pairs -- correct on
    boundary ties, which a per-shard live top-K cannot guarantee."""
    n = PG.size; Pn = PPY.shape[1]
    out_g = np.empty(n, np.int64); out_c = np.empty(n, np.int64); out_p = np.empty((n, Pn), np.int64)
    ndist = np.zeros(S, np.int64)
    for s in prange(S):
        lo = base[s]; hi = base[s + 1]; m = tsize - 1
        keys = np.full(tsize, -1, np.int64); cnts = np.zeros(tsize, np.int64)
        psum = np.zeros((tsize, Pn), np.int64)
        for j in range(lo, hi):
            g = PG[j]; h = (g * 2654435761) & m
            while True:
                k = keys[h]
                if k == -1: keys[h] = g; cnts[h] = 1; break
                elif k == g: cnts[h] += 1; break
                else: h = (h + 1) & m
            for p in range(Pn): psum[h, p] += PPY[j, p]
        w = lo
        for h in range(tsize):
            if keys[h] != -1:
                out_g[w] = keys[h]; out_c[w] = cnts[h]
                for p in range(Pn): out_p[w, p] = psum[h, p]
                w += 1
        ndist[s] = w - lo
    return out_g, out_c, out_p, ndist


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
    # Secondary ORDER BY keys are honored ONLY if they match this read's deterministic tiebreak exactly:
    # count DESC then gid ASC, i.e. the keys ascending in PROJECTION order (kcols). Any other secondary
    # order would change plateau membership, so decline (the scan serves it exactly).
    extra = order.expressions[1:]
    if extra:
        tb = []
        for oe in extra:
            if oe.args.get('desc'):
                return None
            nm = wdb_sql._colname(oe.this)
            if nm is None:
                return None
            tb.append(col_map.get(nm, nm) if col_map else nm)
        if tb != kcols:
            return None
    paycols = [col_map.get(a[1], a[1]) if col_map else a[1] for a in aggs if a[0] in ('SUM', 'AVG')]
    for pc in paycols:                       # payload: any storage mode, but must be integer-typed (dt==0)
        c = seg.cols.get(pc)
        if c is None or c.get('dt') != 0:    # cheap metadata check -- NO column decode in detect
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
    """Row mask for `pcol = lit` / `pcol <> lit`. Find the literal's code CHEAPLY via _code_of
    (O(1) for empty string / int dicts -- no full-dict materialize); only a non-empty string
    literal falls back to the full sorted-dict searchsorted."""
    dt = seg.cols[pcol].get('dt')
    key = lit.encode('utf-8', 'surrogatepass') if (dt == 1 and isinstance(lit, str)) else lit
    code = wdb_compound._code_of(seg, pcol, key)
    if code is None:                             # non-empty string literal: fall back to full dict
        V = np.asarray(seg._typed_dict(pcol)); pos = int(np.searchsorted(V, key))
        code = pos if (0 <= pos < V.size and V[pos] == key) else -1
    codes = seg._raw_codes(pcol)
    if op == '=':
        return (codes == code) if code >= 0 else np.zeros(codes.size, dtype=bool)
    return (codes != code) if code >= 0 else np.ones(codes.size, dtype=bool)


def _survivor_payload(seg, pc, idx):
    """Decode payload values for the SURVIVOR rows only (late materialization). A clean dict column
    gathers Vd[codes[idx]] -- ~13M values instead of decoding all 100M. Columns with overrides, or
    affine/const modes (whose full decode is already cheap), take the correct seg.values()[idx] path."""
    c = seg.cols[pc]
    if c['mode'] not in (4, 6) and seg._overrides(pc) is None:
        Vd = np.asarray(seg._typed_dict(pc))
        return Vd[seg._raw_codes(pc)[idx]]
    return seg.values(pc)[idx]


def _sort_counts(gid):
    """Distinct gids + COUNT per pair via sort + adjacent-run scan -- gid only, payload deferred. A run
    of length c is one pair occurring c times. This is the whole high-cardinality aggregate: no hash to
    oversubscribe, and payload never moves here."""
    sg = np.sort(gid)
    change = np.ones(sg.size, dtype=bool)
    np.not_equal(sg[1:], sg[:-1], out=change[1:])
    starts = np.nonzero(change)[0]
    return sg[starts], np.diff(np.append(starts, sg.size)).astype(np.int64)


def _winner_payload(seg, payphys, gid, sel_gid, idx):
    """SUM of each payload column over the rows of each winning pair, materialized for the <=K winners
    ONLY. searchsorted locates the few winner gids across all rows (one vectorized pass), then payload is
    DECODED for just the matched member rows (~K + duplicate members) -- never the full 100M column.
    idx maps survivor index -> original row index (None = identity, i.e. no filter -- gridwalk's case).
    Returns (K, Pn) aligned to sel_gid, so AVG = sum/count stays exact (integer payload, tiny totals)."""
    order = np.argsort(sel_gid); ws = sel_gid[order]
    pos = np.searchsorted(ws, gid)
    valid = pos < ws.size
    pos2 = np.where(valid, pos, 0)
    match = valid & (ws[pos2] == gid)
    srows = np.nonzero(match)[0]                 # survivor indices of the winning pairs' member rows
    wp = pos2[match]                             # which winner (ws order) each matched row belongs to
    Pn = len(payphys)
    sums_ws = np.zeros((ws.size, Pn), np.int64)
    if Pn:
        orows = srows if idx is None else idx[srows]   # survivor index -> original row index
        for p, pc in enumerate(payphys):
            vals = _survivor_payload(seg, pc, orows)
            sums_ws[:, p] = np.rint(np.bincount(wp, weights=vals.astype(np.float64),
                                                minlength=ws.size)).astype(np.int64)
    inv = np.empty(order.size, np.int64); inv[order] = np.arange(order.size)
    return sums_ws[inv]


def execute(seg, spec):
    global _HITS
    kcols = spec['kcols']; a, b = kcols[0], kcols[1]
    lim = spec['lim']
    caF = seg._raw_codes(a)                  # native-width codes; cast to int64 inside the kernel
    cbF = seg._raw_codes(b)
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
    if idx.size > _SORT_MIN:                              # near-unique keys -> sort, hash would melt
        if idx.size == caF.size:                          # no filter: idx is arange, skip the gather
            gid = caF.astype(np.int64) * Vb + cbF.astype(np.int64)
        else:
            gid = caF[idx].astype(np.int64) * Vb + cbF[idx].astype(np.int64)
        allg, allc = _sort_counts(gid)                    # payload deferred to the winners (below)
        allp = None
    else:
        gid = None
        pay_s = np.zeros((idx.size, Pn), np.int64)         # SURVIVOR-aligned payload (hash path needs it all)
        for p, pc in enumerate(payphys):
            vals = _survivor_payload(seg, pc, idx)
            if vals.dtype.kind not in ('i', 'u'):
                return None                                # non-integer payload -> fused_agg
            pay_s[:, p] = vals
        PG, PPY, base = _gather_partition(idx, caF, cbF, Vb, pay_s, _NSHARD, _NT)
        out_g, out_c, out_p, ndist = _agg_shards(PG, PPY, base, _NSHARD, _TSIZE)
        segs_g = []; segs_c = []; segs_p = []             # gather each shard's occupied region
        for s in range(_NSHARD):
            lo = int(base[s]); dct = int(ndist[s])
            if dct:
                segs_g.append(out_g[lo:lo + dct]); segs_c.append(out_c[lo:lo + dct]); segs_p.append(out_p[lo:lo + dct])
        if not segs_g:
            return []
        allg = np.concatenate(segs_g); allc = np.concatenate(segs_c); allp = np.concatenate(segs_p)
    if allg.size <= lim:                                  # whole distinct set fits -> just order it
        o = np.lexsort((allg, -allc))
    elif allp is None:                                    # SORT path: allg is gid-ascending, counts small
        maxc = int(allc.max())                            # count-bucket top-K: no O(n) partition
        hist = np.bincount(allc, minlength=maxc + 1)      # how many pairs at each (tiny) count value
        cum = 0; boundary = 1
        for c in range(maxc, 0, -1):
            cum += int(hist[c])
            if cum >= lim:
                boundary = c; break
        above = np.nonzero(allc > boundary)[0]            # every pair above the boundary count (< lim of them)
        at = np.nonzero(allc == boundary)[0][:lim - above.size]   # allg gid-ascending -> first = smallest gid
        sel_idx = np.concatenate((above, at))
        o = sel_idx[np.lexsort((allg[sel_idx], -allc[sel_idx]))]  # count DESC, gid ASC over the few winners
    else:                                                 # HASH path: few distinct pairs, partition is cheap
        cbnd = np.partition(allc, allg.size - lim)[allg.size - lim]
        cand = np.nonzero(allc >= cbnd)[0]                # top-lim plus every pair tied at the boundary
        o = cand[np.lexsort((allg[cand], -allc[cand]))[:lim]]
    sel_gid = allg[o]; sel_cnt = allc[o].astype(np.int64)
    sel_pay = allp[o] if allp is not None else _winner_payload(seg, payphys, gid, sel_gid, idx)
    codesA = sel_gid // Vb; codesB = sel_gid - codesA * Vb
    valA = [wdb_sql._pyval(seg.fetch(a, int(cd))) for cd in codesA]   # O(1) per winner -- no full-dict materialize
    valB = [wdb_sql._pyval(seg.fetch(b, int(cd))) for cd in codesB]
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
