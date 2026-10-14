"""THE BOUND HUNT (2026-10-13, the Q18 board; Jackson's double bound from tripletop, made general).

A group's COUNT can never exceed the row count of any one of its key values: count(a, m, b) <= min(rows with
a, rows with m, rows with b). Every one of those per-value counts is one census away. So for
    GROUP BY K1, ..., Kn  ORDER BY COUNT(*) DESC [, more terms]  LIMIT k
a row whose BOUND (the smallest of its key values' counts) is below a threshold theta belongs to a group that
cannot reach theta. Keep only rows with bound >= theta, group them exactly, and read the k-th count c_k:
  c_k >= theta  -> exact: every dropped group counts < theta <= c_k, so none of them is in the top k, nor
                   ties with it (ties at c_k are all kept, for the later ORDER BY terms to decide);
  c_k <  theta  -> run once more at theta = c_k (that keeps every group that can reach c_k: exact).
Any number of keys, any profiles: bare columns by dictionary code, expression keys (minute of EventTime)
through their dictionary map -- the same key list the singleton peel takes. tripletop did this for exactly
one pair of encodings and the minute; this is the atom.

Declines (None) when the bounds do not prune: theta's survivors past MAX_SHARE of the rows mean grouping is
real work for the fused kernel. Aggregates beyond COUNT are computed exactly on the kept rows (every kept
group is whole: its rows share its key values, hence its bound)."""
import os, time
import numpy as np

import wdb_peel

MAX_SHARE = 1.0         # decline when the kept rows pass this share of the survivors
SAMPLE = 1 << 20        # rows in the threshold's sample
DENSE_MAX = 1 << 24     # a composite space this small is counted in one parallel census of its cells instead
                        # (2 x 60 minutes x OpenstatAdID's 47K: the bounds cannot prune -- '' holds 99.5% --
                        # and the factorise over every row was 2.2 s against DuckDB's 0.63)
LEAN = 0.7              # theta = this x the sample's estimate of the k-th count (low: one round, nearly always)
TARGET = 1 << 21        # aim the first threshold at about this many kept rows
_HITS = 0
_ROUNDS = 0
_HEAVY = 0             # answers by the heavy split
_FLAT = 0              # answers by the flat scan
FLAT_SHARE = 0.5       # the bounds keep more than this share of the rows: they cannot prune -- scan flat instead
FLAT_BUCKET = 1 << 20  # the flat key's biggest scatter bucket: its count table (2x, 12 bytes a slot) per thread


def _census(codes, V):
    import wdb_kernels
    return wdb_kernels.bincount_par(codes, V)


def _bound(codes_list, cnts):
    """per row: the smallest of its key values' counts (int32: a count never passes the row count)"""
    import wdb_kernels
    b = np.empty(codes_list[0].size, np.int32)
    for j, (c, n) in enumerate(zip(codes_list, cnts)):
        wdb_kernels.bound_min(b, np.ascontiguousarray(n, dtype=np.int32), np.ascontiguousarray(c), j == 0)
    return b


HIST_CAP = 1 << 20      # bounds past this share the histogram's last bin (a threshold that high keeps ~nothing)


def _theta_for(b, budget):
    """the smallest threshold whose kept rows fit the budget (a histogram of the bounds, never a sort)"""
    import wdb_kernels
    if b.size == 0:
        return 1
    h = wdb_kernels.bound_hist(b, HIST_CAP)
    tail = np.cumsum(h[::-1])[::-1]                      # rows with bound >= t (the last bin: >= HIST_CAP)
    ok = np.flatnonzero(tail <= budget)
    return max(1, int(ok[0])) if ok.size else HIST_CAP


def _group(kc):
    """exact groups of the kept rows: (unique key rows, group id per row) -- the codes packed by bit width
    into one or two words and factorised on every core (wdb_kernels.parallel_factorize; np.unique's sort
    was 3.7 s for 21M kept rows)"""
    import wdb_kernels
    widths = [max(1, int(c.max()).bit_length()) if c.size else 1 for c in kc]
    if sum(widths) > 126:
        mat = np.stack(kc, axis=1)
        u, gid = np.unique(mat, axis=0, return_inverse=True)
        return u, np.asarray(gid, np.int64).ravel()
    words = []; cur = None; bit = 0; place = []
    for c, w in zip(kc, widths):
        if cur is None or bit + w > 63:
            cur = np.zeros(c.size, np.int64); words.append(cur); bit = 0
        wdb_kernels.pack_or(cur, np.ascontiguousarray(c), bit)
        place.append((len(words) - 1, bit, w)); bit += w
    if len(words) > 2:                                   # the greedy packing spilled into a third word
        mat = np.stack(kc, axis=1)
        u, gid = np.unique(mat, axis=0, return_inverse=True)
        return u, np.asarray(gid, np.int64).ravel()
    gid, g0, g1 = wdb_kernels.parallel_factorize(words[0], words[1] if len(words) > 1 else None)
    gw = [g0, g1]
    ukeys = np.empty((g0.size, len(kc)), np.int64)
    for j, (wi, b0, w) in enumerate(place):
        ukeys[:, j] = (gw[wi] >> b0) & ((1 << w) - 1)
    return ukeys, np.asarray(gid, np.int64)


def _kth_largest(cnt, k):
    """the k-th largest of a count array, by a census of the counts (np.partition on 90M small, heavily tied
    counts took 4 s a call); counts past the census cap are settled exactly among themselves"""
    import wdb_kernels
    G = int(cnt.size)
    if G < k:
        return 0
    cap = max(1, min(1 << 20, int(cnt.max())))
    h = wdb_kernels.bound_hist(cnt, cap)                 # groups of every count, the last bin = count >= cap
    if int(h[cap]) >= k:
        big = cnt[cnt >= cap]
        return int(np.partition(big, big.size - k)[big.size - k])
    tail = np.cumsum(h[::-1])                            # groups with count >= cap, cap-1, ...
    return int(cap - int(np.flatnonzero(tail >= k)[0]))


def _theta_sampled(codes, n, k):
    """THE SAMPLED THRESHOLD: group a strided sample exactly and scale its k-th count up to the whole -- an
    estimate of the true k-th count c_k; theta sits a little under it, so the first round nearly always
    proves itself (a theta read off the bounds alone overshoots when one heavy value -- Title's '', Referer's
    '' -- holds a mass of rows at one bound, and pays a second round)"""
    st = max(1, n // SAMPLE)
    smp = [c[::st] for c in codes]
    _u, gid = _group(smp)
    if gid.size == 0:
        return None
    cnt = np.bincount(gid)
    if cnt.size < k:
        return 1
    sk = float(_kth_largest(cnt, k))
    # under the sample's k-th count by LEAN, and by two standard deviations of a count that small (a k-th of
    # 2 in the sample is noise: scaled up it overshot the true 38 five-fold and paid a second round)
    low = min(sk * LEAN, sk - 2.0 * np.sqrt(sk))
    return max(1, int(low * (n / float(gid.size))))


def run(rows, keys, plan, order, k):
    """rows: base rows of the survivors (None = every row). keys / plan / order as wdb_peel.run takes them;
    order[0] must be COUNT descending. Returns the k emitted rows or None (decline)."""
    global _HITS, _ROUNDS, _HEAVY, _FLAT
    if not order or plan[order[0][0]][0] != 'count' or not order[0][1] or len(keys) < 2:
        return None
    seg0 = keys[0]['seg']
    if rows is None:
        rows = np.arange(int(seg0.N), dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    n = int(rows.size)
    if n == 0 or k <= 0:
        _HITS += 1
        return []
    bill = [] if os.environ.get('WDB_JOIN_BILL') else None
    t0 = time.perf_counter()
    def tick(nm):
        nonlocal t0
        if bill is not None:
            t1 = time.perf_counter(); bill.append('%s=%.0fms' % (nm, (t1 - t0) * 1e3)); t0 = t1
    full = n == int(seg0.N)                              # every row survives: whole-column codes, no gather
    codes = [_kcodes_all(kk) if full else wdb_peel._kcodes(kk, rows) for kk in keys]
    tick('codes')
    Vs = [(len(kk['labels']) if kk.get('labels') is not None else int(kk['seg'].cols[kk['pcol']]['V']))
          for kk in keys]
    prod = 1
    for V in Vs:
        prod *= V
    if prod <= DENSE_MAX:
        out = _dense(rows, codes, Vs, prod, keys, plan, order, k, tick)
        _say(bill, ' '.join(bill or []))
        if out is not None:
            _HITS += 1
        return out
    cnts = [_census(c, V) for c, V in zip(codes, Vs)]
    tick('census')
    if all(e[0] in ('key', 'count') for e in plan):
        out = _heavy_split(rows, codes, Vs, cnts, keys, plan, order, k, tick)
        if out is not None:
            _say(bill, ' '.join(bill or []))
            _HITS += 1
            _HEAVY += 1
            return out
    theta = _theta_sampled(codes, n, k)
    tick('theta(%s)' % theta)
    if theta is not None and os.environ.get('WDB_FLAT', '1') != '0' and all(e[0] in ('key', 'count') for e in plan):
        # THE FLAT GATE: the share of rows the threshold would keep, read off the sample's bounds (the full
        # bound pass is the hunt's own cost -- the flat scan never needs it)
        st = max(1, n // SAMPLE)
        sb = _bound([c[::st] for c in codes], cnts)
        share = 1.0 if theta <= 1 else float(np.count_nonzero(sb >= theta)) / max(1, sb.size)
        if share > FLAT_SHARE:
            tick('flat gate(%.2f kept)' % share)
            out = _flat(rows, codes, Vs, cnts, keys, plan, order, k, tick)
            if out is not None:
                _say(bill, ' '.join(bill or []))
                _HITS += 1
                _FLAT += 1
                return out
    b = _bound(codes, cnts)
    tick('bound')
    if theta is None:
        theta = _theta_for(b, min(TARGET, int(MAX_SHARE * n)))
    for _round in range(2):
        _ROUNDS += 1
        if theta <= 1:
            keep = None; kc = codes; nk = n                 # every row kept: no filter, no gathers
        else:
            keep = np.flatnonzero(b >= theta); nk = int(keep.size)
            if nk > MAX_SHARE * n:
                _say(bill, 'decline: theta %d keeps %d of %d' % (theta, nk, n))
                return None
            kc = [c[keep] for c in codes]
        tick('keep(%d)' % nk)
        ukeys, gid = _group(kc) if nk else (np.empty((0, len(keys)), np.int64), np.empty(0, np.int64))
        tick('group')
        G = int(ukeys.shape[0])
        cnt = np.bincount(gid, minlength=G).astype(np.int64)
        ck = _kth_largest(cnt, k)
        if ck >= theta or theta <= 1:
            break
        theta = max(1, ck)                               # one more pass keeps every group that can reach c_k
    else:
        return None
    out = _emit(rows if keep is None else rows[keep], keys, ukeys, gid, cnt, plan, order, k)
    tick('emit')
    _say(bill, ' '.join(bill or []))
    if out is not None:
        _HITS += 1
    return out


def _dense(rows, codes, Vs, prod, keys, plan, order, k, tick):
    """THE DENSE CELLS: every group's count is a cell of the mixed-radix composite, one parallel census; the
    winners decode from their cell numbers, and the rows are revisited only for aggregates beyond COUNT"""
    import wdb_kernels
    comp = np.empty(codes[0].size, np.int64)
    for j, (c, V) in enumerate(zip(codes, Vs)):
        wdb_kernels.radix_acc(comp, np.ascontiguousarray(c), V, j == 0)
    tick('composite')
    cells = wdb_kernels.bincount_par(comp, prod)
    nz = np.flatnonzero(cells)
    cnt = np.asarray(cells[nz], dtype=np.int64)
    tick('census(%d cells, %d groups)' % (prod, nz.size))
    ukeys = np.empty((nz.size, len(Vs)), np.int64)
    r = nz.copy()
    for j in range(len(Vs) - 1, -1, -1):
        ukeys[:, j] = r % Vs[j]; r //= Vs[j]

    def gid_of():
        pos = np.full(prod, -1, np.int64); pos[nz] = np.arange(nz.size, dtype=np.int64)
        return pos[comp]
    out = _emit(rows, keys, ukeys, gid_of, cnt, plan, order, k)
    tick('emit')
    return out


HEAVY = 0.5             # a key value holding this share of the rows is the key's heavy value


def _heavy_split(rows, codes, Vs, cnts, keys, plan, order, k, tick):
    """THE HEAVY SPLIT: when keys have a value holding most rows (OpenstatAdID's '' 99.5%, SocialSourcePage's
    ''), the rows where every such key sits on its heavy value differ only in the OTHER keys -- their groups
    are cells of those keys' (small) composite, counted densely; the few remaining rows are grouped exactly.
    The two sets of groups are disjoint (a light row leaves at least one heavy value). COUNT-only plans.
    None when no split pays (the caller hunts as before)."""
    import wdb_kernels
    n = codes[0].size
    H = [j for j in range(len(Vs)) if cnts[j].size and int(cnts[j].max()) >= HEAVY * n]
    if not H:
        return None
    rest = [j for j in range(len(Vs)) if j not in H]
    R = 1
    for j in rest:
        R *= Vs[j]
    if R > DENSE_MAX:
        return None
    tops = [int(np.argmax(cnts[j])) for j in H]
    heavy = wdb_kernels.eq_all([np.ascontiguousarray(codes[j]) for j in H], tops)
    nh = int(np.count_nonzero(heavy))
    if nh < HEAVY * n:
        return None
    tick('heavy mask(%d of %d)' % (nh, n))
    if rest:
        comp = np.empty(n, np.int64)
        for i, j in enumerate(rest):
            wdb_kernels.radix_acc(comp, np.ascontiguousarray(codes[j]), Vs[j], i == 0)
        cells = wdb_kernels.masked_count(comp, heavy, R)
        del comp
    else:
        cells = np.array([nh], np.int64)
    nz = np.flatnonzero(cells)
    hk = np.empty((nz.size, len(Vs)), np.int64)
    r = nz.copy()
    for j in reversed(rest):
        hk[:, j] = r % Vs[j]; r //= Vs[j]
    for j, d in zip(H, tops):
        hk[:, j] = d
    tick('heavy cells(%d)' % nz.size)
    light = np.flatnonzero(~heavy)
    if light.size:
        lk, lgid = _group([c[light] for c in codes])
        lcnt = np.bincount(lgid, minlength=lk.shape[0]).astype(np.int64)
    else:
        lk = np.empty((0, len(Vs)), np.int64); lcnt = np.empty(0, np.int64)
    tick('light(%d rows, %d groups)' % (light.size, lk.shape[0]))
    ukeys = np.concatenate([hk, lk]); cnt = np.concatenate([np.asarray(cells[nz], np.int64), lcnt])
    return _emit(rows, keys, ukeys, None, cnt, plan, order, k)


def _flat(rows, codes, Vs, cnts, keys, plan, order, k, tick):
    """THE FLAT SCAN (2026-10-14, Jackson: "if we have to read everything then we just need to scan as fast as
    possible and get it over with"): when no bound prunes (a time key as the person: every value a handful of
    rows), the hunt's filter and factorise are pure overhead. Instead one key -- the FLAT key, whose values
    spread the rows most evenly -- scatters every row into 4096 buckets by its top bits, carrying the other keys
    packed into one 32-bit word; every group then lives in one bucket, counted in a small table that stays in
    cache: a census of the counts gives the k-th count, and only the buckets holding a group that reaches it
    are counted again to emit (ties kept). Every row is read once into the scatter and once in cache -- no
    filter, no table the size of the groups.
    COUNT-only plans. None when no key fits the shape (the caller hunts as before)."""
    import wdb_kernels
    n = codes[0].size
    widths = [max(1, (int(V) - 1).bit_length()) for V in Vs]
    best = None
    for j in range(len(Vs)):
        if Vs[j] >= (1 << 32) or sum(widths) - widths[j] > 32:
            continue
        SH = max(0, widths[j] - 12)
        c = np.asarray(cnts[j], np.int64)
        big = int(np.add.reduceat(c, np.arange(0, c.size, 1 << SH)).max()) if c.size else 0
        if best is None or big < best[1]:
            best = (j, big, SH)
    if best is None or best[1] > max(FLAT_BUCKET, n // 4096):
        tick('flat: no key fits (%s)' % (best[1] if best else '-'))
        return None
    f, _big, SH = best
    rest = [j for j in range(len(Vs)) if j != f]
    rc = np.zeros(n, np.uint32)                          # the other keys, packed by bit width (<= 32 bits)
    place = []; bit = 0
    for j in rest:
        wdb_kernels.pack_or(rc, np.ascontiguousarray(codes[j]), bit)
        place.append((j, bit, widths[j])); bit += widths[j]
    tick('flat pack(key %d, %d bits)' % (f, bit))
    ku, kr, offs = wdb_kernels.gd_pass1(np.ascontiguousarray(codes[f]), rc, np.int64(SH), np.int64(8))
    del rc
    tick('flat scatter(biggest bucket %d)' % _big)
    pk, cnt = wdb_kernels.flat_count(ku, kr, offs, int(k))
    del ku, kr
    tick('flat sort+runs(%d candidates)' % pk.size)
    ukeys = np.empty((pk.size, len(Vs)), np.int64)
    ukeys[:, f] = (pk >> np.uint64(32)).astype(np.int64)
    lo = (pk & np.uint64(0xFFFFFFFF)).astype(np.int64)
    for j, b0, w in place:
        ukeys[:, j] = (lo >> b0) & ((1 << w) - 1)
    out = _emit(rows, keys, ukeys, None, cnt.astype(np.int64), plan, order, k)
    tick('emit')
    return out


def _say(bill, msg):
    if bill is not None:
        print('BOUND BILL: ' + msg, flush=True)


def _kcodes_all(kk):
    c = np.asarray(kk['seg'].codes(kk['pcol']))
    m = kk.get('map')
    return c if m is None else m[c]


def _emit(krows, keys, ukeys, gid, cnt, plan, order, k):
    import wdb_join
    G = int(ukeys.shape[0])
    # THE CANDIDATES FIRST: only groups whose count reaches the k-th count can be emitted (order[0] is COUNT
    # descending); aggregates and the full ORDER BY are worked for them alone -- a lexsort and np.add.at over
    # every group were the bill when nearly every row is kept (90M groups)
    ck = _kth_largest(cnt, k) if G > k else 0
    cand = np.flatnonzero(cnt >= ck) if G > k else np.arange(G, dtype=np.int64)
    C = int(cand.size)
    vals = {}
    pos = None
    for i, e in enumerate(plan):
        if e[0] == 'key':
            vals[i] = ukeys[cand, e[1]]
        elif e[0] == 'count':
            vals[i] = cnt[cand]
        else:
            if pos is None:
                if callable(gid):
                    gid = gid()
                pos = np.full(G, -1, np.int64); pos[cand] = np.arange(C, dtype=np.int64)
                cg = pos[gid]; on = cg >= 0
                crows = krows[on]; cg = cg[on]
            kind, sg, pc = e[1], e[2], e[3]
            v = wdb_peel._values(sg, pc, crows)
            if v.dtype.kind == 'f' and np.isnan(v).any():
                return None
            acc = np.float64 if v.dtype.kind == 'f' else np.int64
            if kind in ('SUM', 'AVG'):
                if acc is np.float64:
                    a = np.bincount(cg, weights=v, minlength=C)
                else:
                    a = np.zeros(C, np.int64); np.add.at(a, cg, v.astype(np.int64, copy=False))
                if kind == 'AVG':
                    a = a / np.maximum(cnt[cand], 1)
            elif kind == 'MIN':
                a = np.full(C, np.inf if acc is np.float64 else np.iinfo(np.int64).max, acc)
                np.minimum.at(a, cg, v.astype(acc, copy=False))
            else:
                a = np.full(C, -np.inf if acc is np.float64 else np.iinfo(np.int64).min, acc)
                np.maximum.at(a, cg, v.astype(acc, copy=False))
            vals[i] = a

    def norm(v, desc):
        if not desc:
            return v
        return -v if v.dtype.kind == 'f' else -(v.astype(np.int64))
    if len(order) == 1:
        # COUNT alone orders: every candidate above the k-th count, then any of those on it (SQL's ties)
        cc = vals[order[0][0]]
        above = np.flatnonzero(cc > ck)
        above = above[np.argsort(-cc[above], kind='stable')]
        sel = np.concatenate([above, np.flatnonzero(cc == ck)[:max(0, k - above.size)]])[:k]
    else:
        sel = np.lexsort(tuple(norm(vals[pi], desc) for pi, desc in reversed(order)))[:k]
    nvis = sum(1 for e in plan if not (len(e) > 4 and e[4] == 'hidden'))
    out_cols = []
    for i in range(nvis):
        e = plan[i]
        v = vals[i][sel]
        if e[0] == 'key':
            kk = keys[e[1]]
            if kk.get('labels') is not None:
                out_cols.append([kk['labels'][int(x)] for x in v])
            else:
                out_cols.append(wdb_join._bulk_keyvals(kk['seg'], kk['pcol'], v))
        elif e[0] == 'count':
            out_cols.append([int(x) for x in v])
        elif v.dtype.kind == 'f':
            out_cols.append([float(x) for x in v])
        else:
            out_cols.append([int(x) for x in v])
    return [tuple(c[r] for c in out_cols) for r in range(int(sel.size))]
