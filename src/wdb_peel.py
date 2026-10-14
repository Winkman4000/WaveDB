"""THE SINGLETON PEEL (2026-10-12, Jackson: "every repeating WatchID gets an R tag ... then whenever we
group or aggregate we can jump to multiplication without having to eval every instance of the 100M rows").

A group repeats only if EVERY part of its key repeats. So when one group key is nearly unique (WatchID:
99,997,493 values over 99,997,497 rows) almost every group is one row: its COUNT is 1 and its SUM, MIN,
MAX and AVG are that row's own value. The peel, all at query time (nothing stored):

  1. CENSUS: count the near-unique key's codes over the survivors -- who repeats (the R tag);
  2. group ONLY the repeater rows, exactly, on the whole key;
  3. every other survivor is its own group -- no hash, no accumulator, no 100M-row emit;
  4. pick the k winners of the ORDER BY over (repeater groups + singleton rows), one sort term at a
     time, holding the singletons as a block while a term is constant on them (COUNT is 1 on every
     singleton, so ORDER BY COUNT(*) never touches them row by row);
  5. decode keys and aggregates at the k winners only.

It declines (returns None) when repeaters are more than PEEL_MAX_REP of the survivors: then grouping
is real work and belongs to the fused kernel. Called from wdb_join._fast_pointer_agg's peel door, which
owns the shape gates and hands over the survivors the cascade already found."""
import numpy as np

import wdb_sql

PEEL_MIN_UNIQ = 0.8      # metadata gate: the peel key's dictionary holds >= 80% as many values as the table rows
PEEL_MAX_REP = 0.25      # execute gate: repeater rows at most a quarter of the survivors

_HITS = 0                # answers served by the peel (tests assert the road was taken)


def pick_key(keys):
    """Index of the key to peel on (largest dictionary relative to its table), or None if none is
    near-unique. keys: [{'seg', 'pcol'}] (+ 'map', 'labels' for an expression key, never peeled).
    Metadata only."""
    best = None; bestf = 0.0
    for j, k in enumerate(keys):
        if k.get('map') is not None:
            continue
        c = k['seg'].cols[k['pcol']]
        N = int(k['seg'].N)
        if N <= 0 or c.get('mode') not in (0, 1, 2):
            continue
        f = int(c['V']) / N
        if f >= PEEL_MIN_UNIQ and f > bestf:
            best, bestf = j, f
    return best


def _census(codes, V):
    # per-thread boards cost threads x V x 4 bytes: a near-unique key's V is about its row count, so past
    # 8M values one numpy board (V x 8 bytes) is the smaller bill
    if V > (1 << 23):
        return np.bincount(codes, minlength=V)
    import wdb_kernels
    return wdb_kernels.bincount_par(codes, V)


def _kcodes(kk, rows):
    """a key's codes at base rows: the dictionary code, or for an expression key its index into the distinct
    results (code order = value order either way)"""
    c = np.asarray(kk['seg'].codes_at(kk['pcol'], rows), dtype=np.int64)
    m = kk.get('map')
    return c if m is None else m[c]


def _values(seg, pcol, rows):
    """Numeric values of a plain column at base rows (int64 or float64)."""
    raw = wdb_sql.raw_dict_col(seg, pcol, want_codes=False)
    if raw is None:
        raise ValueError('not a plain numeric column')
    return raw[0][np.asarray(seg.codes_at(pcol, rows), dtype=np.int64)]


class _Block:
    """The candidates: repeater groups (ids into the repeater tables) plus singletons (indices into the
    singleton rows). `s` is an index array, or None for 'all S singletons' (never materialised)."""
    __slots__ = ('r', 's', 'S')

    def __init__(self, r, s, S):
        self.r = r; self.s = s; self.S = S

    def ns(self):
        return self.S if self.s is None else int(self.s.size)

    def sidx(self):
        return np.arange(self.S, dtype=np.int64) if self.s is None else self.s


def run(rows, keys, peel, plan, order, k):
    """rows: int64 base-row positions of the survivors. keys: [{'seg','pcol'}] (the GROUP BY columns,
    each read at base rows). peel: index into keys. plan: entries, visible ones first --
      ('key', j) | ('count',) | ('agg', 'SUM'|'MIN'|'MAX'|'AVG', seg, pcol)
    order: [(plan index, desc)]. Returns the k emitted rows (visible plan entries) or None."""
    global _HITS
    rows = np.asarray(rows, dtype=np.int64)
    n = int(rows.size)
    nvis = sum(1 for e in plan if not (len(e) > 4 and e[4] == 'hidden'))
    if n == 0 or k <= 0:
        _HITS += 1
        return []
    P = keys[peel]
    pseg, ppc = P['seg'], P['pcol']
    V = int(pseg.cols[ppc]['V'])
    pc = np.asarray(pseg.codes_at(ppc, rows), dtype=np.int64)
    cnt = _census(pc, V)
    rep = cnt[pc] > 1                                   # THE R TAG: this row's peel value repeats
    del cnt
    nrep = int(np.count_nonzero(rep))
    if nrep > PEEL_MAX_REP * n:
        return None
    if nrep:
        rrows = rows[rep]
        srows = rows[~rep]
    else:
        rrows = np.empty(0, np.int64)
        srows = rows
    del rep
    S = int(srows.size)

    # ---- the repeaters, grouped exactly on the whole key ----
    nk = len(keys)
    if nrep:
        kc = np.empty((nrep, nk), np.int64)
        for j, kk in enumerate(keys):
            kc[:, j] = _kcodes(kk, rrows)
        ukeys, gid = np.unique(kc, axis=0, return_inverse=True)
        gid = np.asarray(gid, dtype=np.int64).ravel()
    else:
        ukeys = np.empty((0, nk), np.int64); gid = np.empty(0, np.int64)
    G = int(ukeys.shape[0])
    rcount = np.bincount(gid, minlength=G).astype(np.int64)
    ragg = {}                                           # plan index -> per-group value
    for i, e in enumerate(plan):
        if e[0] != 'agg':
            continue
        kind, sg, pc_ = e[1], e[2], e[3]
        v = _values(sg, pc_, rrows) if nrep else np.empty(0, np.float64)
        if v.dtype.kind == 'f' and np.isnan(v).any():
            return None                                 # NaN ordering/aggregation: the general road's
        acc_dt = np.float64 if v.dtype.kind == 'f' else np.int64
        if kind in ('SUM', 'AVG'):
            a = np.zeros(G, acc_dt); np.add.at(a, gid, v.astype(acc_dt, copy=False))
            if kind == 'AVG':
                a = a / np.maximum(rcount, 1)
        elif kind == 'MIN':
            a = np.full(G, np.inf if acc_dt is np.float64 else np.iinfo(np.int64).max, acc_dt)
            np.minimum.at(a, gid, v.astype(acc_dt, copy=False))
        else:
            a = np.full(G, -np.inf if acc_dt is np.float64 else np.iinfo(np.int64).min, acc_dt)
            np.maximum.at(a, gid, v.astype(acc_dt, copy=False))
        ragg[i] = a

    # ---- a sort term over the two parts: (values on repeater ids, values on singleton indices | const) ----
    def term(pi):
        e = plan[pi]
        if e[0] == 'count':
            return (lambda r: rcount[r]), 1
        if e[0] == 'key':
            j = e[1]; kk = keys[j]
            return (lambda r: ukeys[r, j]), \
                (lambda s: _kcodes(kk, srows[s]))
        a = ragg[pi]; sg, pc_ = e[2], e[3]
        def sv(s):
            v = _values(sg, pc_, srows[s])
            if v.dtype.kind == 'f' and np.isnan(v).any():
                raise ValueError('NaN')
            return v.astype(np.float64) if e[1] == 'AVG' else v
        return (lambda r: a[r]), sv
    terms = []
    for pi, desc in order:
        rf, sf = term(pi)
        terms.append((rf, sf, desc))

    def norm(v, desc):
        if not desc:
            return v
        return -v if v.dtype.kind == 'f' else -(v.astype(np.int64))

    # ---- THE PICK: one term at a time; the strictly-better rows are in, the ties go to the next term ----
    blk = _Block(np.arange(G, dtype=np.int64), None, S)
    need = int(k)
    taken_r = []; taken_s = []
    try:
        for rf, sf, desc in terms:
            tot = blk.r.size + blk.ns()
            if tot <= need or need == 0:
                break
            rv = norm(np.asarray(rf(blk.r)), desc)
            if callable(sf):
                sv = norm(np.asarray(sf(blk.sidx())), desc)
                if rv.dtype.kind == 'f' or sv.dtype.kind == 'f':
                    rv = rv.astype(np.float64); sv = sv.astype(np.float64)
                allv = np.concatenate([rv, sv])
                kth = np.partition(allv, need - 1)[need - 1]
                sidx = blk.sidx()
                lt_r = rv < kth; lt_s = sv < kth
                taken_r.append(blk.r[lt_r]); taken_s.append(sidx[lt_s])
                need -= int(lt_r.sum()) + int(lt_s.sum())
                blk = _Block(blk.r[rv == kth], sidx[sv == kth], S)
            else:
                c = -sf if desc else sf                 # the singletons' one shared value
                srt = np.sort(rv); m = blk.ns()
                a = int(np.searchsorted(srt, c, 'left'))    # repeater values strictly below the block
                if need <= a:
                    kth = srt[need - 1]
                elif need <= a + m:
                    kth = c
                else:
                    kth = srt[need - 1 - m]
                lt_r = rv < kth
                taken_r.append(blk.r[lt_r]); need -= int(lt_r.sum())
                if c < kth:
                    taken_s.append(blk.sidx()); need -= m
                    blk = _Block(blk.r[rv == kth], np.empty(0, np.int64), S)
                elif c == kth:
                    blk = _Block(blk.r[rv == kth], blk.s, S)
                else:
                    blk = _Block(blk.r[rv == kth], np.empty(0, np.int64), S)
    except ValueError:
        return None
    if need > 0:                                        # ties past every term (or no terms): any of them
        tr = blk.r[:need]; taken_r.append(tr); need -= int(tr.size)
        if need > 0:
            if blk.s is None:
                taken_s.append(np.arange(min(need, S), dtype=np.int64))
            else:
                taken_s.append(blk.s[:need])
    pr = np.concatenate(taken_r) if taken_r else np.empty(0, np.int64)
    ps = np.concatenate(taken_s) if taken_s else np.empty(0, np.int64)
    nr = int(pr.size); m = nr + int(ps.size)
    if m == 0:
        _HITS += 1
        return []

    # ---- the winners, in ORDER BY order ----
    def at(pi):
        e = plan[pi]
        if e[0] == 'count':
            return np.concatenate([rcount[pr], np.ones(ps.size, np.int64)])
        if e[0] == 'key':
            j = e[1]; kk = keys[j]
            return np.concatenate([ukeys[pr, j],
                                   _kcodes(kk, srows[ps])])
        sv = _values(e[2], e[3], srows[ps])
        if e[1] == 'AVG':
            sv = sv.astype(np.float64)
        ra = ragg[pi][pr]
        if ra.dtype != sv.dtype:
            ra = ra.astype(np.float64); sv = sv.astype(np.float64)
        return np.concatenate([ra, sv])
    cache = {}
    def col(pi):
        if pi not in cache:
            cache[pi] = at(pi)
        return cache[pi]
    if order:
        sel = np.lexsort(tuple(norm(col(pi), desc) for pi, desc in reversed(order)))[:k]
    else:
        sel = np.arange(m)[:k]

    out_cols = []
    for i in range(nvis):
        e = plan[i]
        v = col(i)[sel]
        if e[0] == 'key':
            kk = keys[e[1]]
            import wdb_join
            if kk.get('labels') is not None:
                out_cols.append([kk["labels"][int(x)] for x in v])
            else:
                out_cols.append(wdb_join._bulk_keyvals(kk['seg'], kk['pcol'], v))
        elif e[0] == 'count':
            out_cols.append([int(x) for x in v])
        elif v.dtype.kind == 'f':
            out_cols.append([float(x) for x in v])
        else:
            out_cols.append([int(x) for x in v])
    _HITS += 1
    return [tuple(c[r] for c in out_cols) for r in range(int(sel.size))]
