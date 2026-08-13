"""Compiled aggregation kernels (numba soft-dependency).

kway_topk: single-pass loser-tree merge of K sorted (key, count) runs with on-the-fly equal-key
accumulation feeding a running top-K min-heap -- the merged stream is never materialized. This is
the counting core of the disk-only GROUP BY path: measured on ClickBench Q16 (100M rows, 13.2M
non-norm keys in 14 runs) at ~0.21 s, completing a 0.70 s total vs DuckDB's 0.79 s best.

top10_i32: single-pass top-K over a dense int32 count table (the norm lane's per-code counts).

Without numba both fall back to numpy (pairwise sorted-merge tournament / argpartition) --
correct, ~2-4x slower.
"""
import numpy as np

try:
    from numba import njit, prange
    import numba
    HAVE_NUMBA = True
except Exception:                                     # pragma: no cover
    HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f):
            return f
        return deco


@njit(nogil=True, cache=True)
def _kway_topk_nb(keys, vals, offs, K):
    nr = offs.size - 1
    cur = offs[:nr].copy()
    INF = np.int64(0x7FFFFFFFFFFFFFFF)
    P = 1
    while P < nr:
        P *= 2
    lk = np.full(P, INF, np.int64)
    for r in range(nr):
        lk[r] = keys[cur[r]] if cur[r] < offs[r + 1] else INF
    tree = np.full(2 * P, -1, np.int32)
    win = np.empty(2 * P, np.int32)
    for i in range(P):
        win[P + i] = i if i < nr else -1
    for i in range(P - 1, 0, -1):
        a, b = win[2 * i], win[2 * i + 1]
        ka = lk[a] if a >= 0 else INF
        kb = lk[b] if b >= 0 else INF
        if ka <= kb:
            win[i] = a; tree[i] = b
        else:
            win[i] = b; tree[i] = a
    topc = np.zeros(K, np.int64); topk = np.zeros(K, np.int64)
    curkey = np.int64(-1); acc = np.int64(0)
    w = win[1]
    while w >= 0 and lk[w] != INF:
        r = w
        k = lk[r]
        if k != curkey:
            if acc > topc[0]:
                topc[0] = acc; topk[0] = curkey
                j = 0
                while True:
                    l = 2 * j + 1; rt = 2 * j + 2; m = j
                    if l < K and topc[l] < topc[m]: m = l
                    if rt < K and topc[rt] < topc[m]: m = rt
                    if m == j: break
                    topc[j], topc[m] = topc[m], topc[j]
                    topk[j], topk[m] = topk[m], topk[j]
                    j = m
            curkey = k; acc = vals[cur[r]]
        else:
            acc += vals[cur[r]]
        cur[r] += 1
        lk[r] = keys[cur[r]] if cur[r] < offs[r + 1] else INF
        i = (P + r) >> 1
        wcur = r
        while i >= 1:
            lo = tree[i]
            klo = lk[lo] if lo >= 0 else INF
            if klo < (lk[wcur] if wcur >= 0 else INF):
                tree[i] = wcur; wcur = lo
            i >>= 1
        w = wcur
    if acc > topc[0]:
        topc[0] = acc; topk[0] = curkey
    return topc, topk


@njit(nogil=True, cache=True)
def top10_i32(tab, K):
    topc = np.zeros(K, np.int64); topk = np.zeros(K, np.int64)
    for i in range(tab.size):
        v = np.int64(tab[i])
        if v > topc[0]:
            topc[0] = v; topk[0] = i
            j = 0
            while True:
                l = 2 * j + 1; rt = 2 * j + 2; m = j
                if l < K and topc[l] < topc[m]: m = l
                if rt < K and topc[rt] < topc[m]: m = rt
                if m == j: break
                topc[j], topc[m] = topc[m], topc[j]
                topk[j], topk[m] = topk[m], topk[j]
                j = m
    return topc, topk


def kway_topk(keys, vals, offs, K):
    """Top-K (count, key) pairs from K sorted runs. Returns (counts, keys) unsorted heaps;
    zero-count slots are empty. Numba path streams; numpy fallback merges then partitions."""
    if HAVE_NUMBA:
        return _kway_topk_nb(keys, vals, offs, np.int64(K))
    parts = [(keys[offs[i]:offs[i + 1]], vals[offs[i]:offs[i + 1]])
             for i in range(offs.size - 1) if offs[i + 1] > offs[i]]
    while len(parts) > 1:
        nxt = []
        for i in range(0, len(parts) - 1, 2):
            k = np.concatenate([parts[i][0], parts[i + 1][0]])
            v = np.concatenate([parts[i][1], parts[i + 1][1]])
            o = np.argsort(k, kind='stable'); k = k[o]; v = v[o]
            new = np.ones(k.size, bool); new[1:] = k[1:] != k[:-1]
            idx = np.nonzero(new)[0]
            nxt.append((k[idx], np.add.reduceat(v, idx)))
        if len(parts) % 2:
            nxt.append(parts[-1])
        parts = nxt
    g, c = parts[0] if parts else (np.empty(0, np.int64), np.empty(0, np.int64))
    kk = min(K, g.size)
    if kk == 0:
        return np.zeros(K, np.int64), np.zeros(K, np.int64)
    ti = np.argpartition(-c, kk - 1)[:kk]
    tc = np.zeros(K, np.int64); tk = np.zeros(K, np.int64)
    tc[:kk] = c[ti]; tk[:kk] = g[ti]
    return tc, tk


@njit(nogil=True, parallel=True, cache=True)
def _part_scatter_nb(codes, K):
    """Stable parallel counting scatter: perm such that codes[perm] is grouped by code with
    original order preserved inside each group -- the fused motion, compiled. Two passes:
    per-chunk histograms -> exclusive global/chunk offsets -> stable scatter."""
    N = codes.size
    T = numba.get_num_threads()
    chunk = (N + T - 1) // T
    hist = np.zeros((T, K), np.int64)
    for t in numba.prange(T):
        lo = t * chunk
        hi = min(N, lo + chunk)
        for i in range(lo, hi):
            hist[t, codes[i]] += 1
    offs = np.zeros((T, K), np.int64)
    run = np.int64(0)
    for k in range(K):
        for t in range(T):
            offs[t, k] = run
            run += hist[t, k]
    perm = np.empty(N, np.int64)
    for t in numba.prange(T):
        lo = t * chunk
        hi = min(N, lo + chunk)
        cur = offs[t].copy()
        for i in range(lo, hi):
            c = codes[i]
            perm[cur[c]] = i
            cur[c] += 1
    return perm


def part_scatter(codes, K):
    """Stable grouping permutation by small-int key; numba parallel, numpy fallback."""
    if HAVE_NUMBA:
        return _part_scatter_nb(codes, np.int64(K))
    return np.argsort(codes, kind='stable')


@njit(nogil=True, cache=True)
def _group_min_nb(pc, oc, acc):
    for i in range(pc.size):
        v = oc[i]
        if v < acc[pc[i]]:
            acc[pc[i]] = v
    return acc


def group_min(pc, oc, K):
    """Per-group minimum of oc keyed by pc (one pass, no sort): the rn=1 window shape.
    acc[g] = min over rows of group g; untouched groups stay at int64 max."""
    acc = np.full(K, np.iinfo(np.int64).max, np.int64)
    if HAVE_NUMBA:
        return _group_min_nb(pc.astype(np.int64), oc.astype(np.int64), acc)
    np.minimum.at(acc, pc, oc)
    return acc


@njit(nogil=True, cache=True)
def _group_max_nb(pc, oc, acc):
    for i in range(pc.size):
        v = oc[i]
        if v > acc[pc[i]]:
            acc[pc[i]] = v
    return acc


def group_max(pc, oc, K):
    """Per-group maximum: the DESC rn=1 mirror. Codes are >= 0, so -1 marks empty."""
    acc = np.full(K, np.int64(-1), np.int64)
    if HAVE_NUMBA:
        return _group_max_nb(pc.astype(np.int64), oc.astype(np.int64), acc)
    np.maximum.at(acc, pc, oc)
    return acc


@njit(nogil=True, cache=True)
def _frame_sum_keyed_nb(pc, vals, k, ring, rsum, rcnt, rpos, osum, ocnt):
    W = k + 1
    for i in range(pc.size):
        p = pc[i]
        v = vals[i]
        if rcnt[p] >= W:
            rsum[p] -= ring[p * W + rpos[p]]
        else:
            rcnt[p] += 1
        ring[p * W + rpos[p]] = v
        rpos[p] = (rpos[p] + 1) % W
        rsum[p] += v
        osum[i] = rsum[p]
        ocnt[i] = rcnt[p]


def frame_sum_keyed(pc, vals, k, K):
    """ROWS k PRECEDING running sum per partition, in WALK order: per-partition ring
    of the last k+1 values plus a running sum -- no permutation, no scatter-back.
    Returns (framesum, framecount) per row. Requires walk order == frame order
    (stairs-ascending order column); numba only (caller declines otherwise)."""
    if not HAVE_NUMBA:
        return None
    W = k + 1
    ring = np.zeros(K * W, np.int64)
    rsum = np.zeros(K, np.int64)
    rcnt = np.zeros(K, np.int32)
    rpos = np.zeros(K, np.int32)
    osum = np.empty(pc.size, np.int64)
    ocnt = np.empty(pc.size, np.int32)
    _frame_sum_keyed_nb(pc.astype(np.int64), vals.astype(np.int64), np.int64(k),
                        ring, rsum, rcnt, rpos, osum, ocnt)
    return osum, ocnt


@njit(nogil=True, cache=True)
def _frame_ext_keyed_nb(pc, vals, k, ring, rcnt, rpos, do_max, out):
    W = k + 1
    for i in range(pc.size):
        p = pc[i]
        if rcnt[p] < W:
            rcnt[p] += 1
        ring[p * W + rpos[p]] = vals[i]
        rpos[p] = (rpos[p] + 1) % W
        n = rcnt[p]
        base = p * W
        best = ring[base]
        if do_max:
            for j in range(1, n):
                if ring[base + j] > best:
                    best = ring[base + j]
        else:
            for j in range(1, n):
                if ring[base + j] < best:
                    best = ring[base + j]
        out[i] = best


def frame_ext_keyed(pc, vals, k, K, do_max):
    """ROWS k PRECEDING running min/max per partition, in WALK order: ring of the
    last k+1 values, extremum rescanned over <= k+1 slots per row (k is small)."""
    if not HAVE_NUMBA:
        return None
    W = k + 1
    ring = np.zeros(K * W, np.int64)
    rcnt = np.zeros(K, np.int32)
    rpos = np.zeros(K, np.int32)
    out = np.empty(pc.size, np.int64)
    _frame_ext_keyed_nb(pc.astype(np.int64), vals.astype(np.int64), np.int64(k),
                        ring, rcnt, rpos, do_max, out)
    return out


@njit(nogil=True, cache=True)
def _rnk_mark_nb(pc, k, cnt, rn):
    for i in range(pc.size):
        c = cnt[pc[i]] + 1
        cnt[pc[i]] = c
        if c <= k:
            rn[i] = c


def rnk_mark(pc, k, K):
    """rn <= k in walk order: a counter per cup, rows arriving while the cup holds
    fewer than k are winners, stamped with their arrival position. rn never needed
    order codes -- only arrival. Returns int8 rn per row (0 = not a winner)."""
    if not HAVE_NUMBA:
        return None
    cnt = np.zeros(K, np.int32)
    rn = np.zeros(pc.size, np.int8)
    _rnk_mark_nb(pc.astype(np.int64), np.int64(k), cnt, rn)
    return rn


@njit(nogil=True, parallel=True, cache=True)
def _seg_cumminmax_nb(vals, lane_start, lane_of, do_min):
    """Segmented cumulative min/max: running extreme within each lane (parallel over lanes)."""
    out = np.empty(vals.size, vals.dtype)
    L = lane_start.size
    for li in numba.prange(L):
        lo = lane_start[li]
        hi = lane_start[li + 1] if li + 1 < L else vals.size
        cur = vals[lo]
        out[lo] = cur
        for i in range(lo + 1, hi):
            v = vals[i]
            if (v < cur) == do_min and v != cur:
                cur = v
            elif do_min and v < cur:
                cur = v
            elif not do_min and v > cur:
                cur = v
            out[i] = cur
    return out


def seg_cummin(vals, lane_start):
    if HAVE_NUMBA:
        return _seg_cumminmax_nb(vals, lane_start, None, True)
    out = vals.copy()
    for li in range(lane_start.size):
        lo = lane_start[li]
        hi = lane_start[li + 1] if li + 1 < lane_start.size else vals.size
        out[lo:hi] = np.minimum.accumulate(vals[lo:hi])
    return out


def seg_cummax(vals, lane_start):
    if HAVE_NUMBA:
        return _seg_cumminmax_nb(vals, lane_start, None, False)
    out = vals.copy()
    for li in range(lane_start.size):
        lo = lane_start[li]
        hi = lane_start[li + 1] if li + 1 < lane_start.size else vals.size
        out[lo:hi] = np.maximum.accumulate(vals[lo:hi])
    return out


@njit(nogil=True, parallel=True, cache=True)
def _seg_slidext_nb(vals, lane_start, k, do_min):
    """Sliding window min/max over ROWS k PRECEDING .. CURRENT ROW, per lane: the classic
    monotonic deque, one deque per lane, parallel over lanes."""
    N = vals.size
    out = np.empty(N, vals.dtype)
    L = lane_start.size
    for li in numba.prange(L):
        lo = lane_start[li]
        hi = lane_start[li + 1] if li + 1 < L else N
        m = hi - lo
        dq = np.empty(m, np.int64)          # indices, front..back monotonic
        head = 0; tail = 0                   # deque in dq[head:tail]
        for i in range(lo, hi):
            lo_w = i - k
            while head < tail and dq[head] < lo_w:
                head += 1
            v = vals[i]
            if do_min:
                while head < tail and vals[dq[tail - 1]] >= v:
                    tail -= 1
            else:
                while head < tail and vals[dq[tail - 1]] <= v:
                    tail -= 1
            dq[tail] = i; tail += 1
            out[i] = vals[dq[head]]
    return out


def seg_slidmin(vals, lane_start, k):
    if HAVE_NUMBA:
        return _seg_slidext_nb(vals, lane_start, np.int64(k), True)
    out = vals.copy()
    for li in range(lane_start.size):
        lo = lane_start[li]
        hi = lane_start[li + 1] if li + 1 < lane_start.size else vals.size
        for i in range(lo, hi):
            out[i] = vals[max(lo, i - k):i + 1].min()
    return out


def seg_slidmax(vals, lane_start, k):
    if HAVE_NUMBA:
        return _seg_slidext_nb(vals, lane_start, np.int64(k), False)
    out = vals.copy()
    for li in range(lane_start.size):
        lo = lane_start[li]
        hi = lane_start[li + 1] if li + 1 < lane_start.size else vals.size
        for i in range(lo, hi):
            out[i] = vals[max(lo, i - k):i + 1].max()
    return out


@njit(nogil=True, cache=True)
def _running_nb(pc, vals, stairs, s, c, m, lr, lv, fwd, kind):
    # kind: 0=avg 1=sum 2=min 3=max 4=count. Forward: each partition's running pair
    # (p updated every round, divisor = cards absorbed). Tie run-id rides the stairs
    # in O(1) per row (rr = boundaries crossed == the order code). Backward: promote
    # tie-group peers to the group-final value (RANGE frame: the shared plate).
    n = pc.size
    run = np.empty(n, np.int32)
    rr = 0
    for i in range(n):
        while rr < stairs.size and i >= stairs[rr]:
            rr += 1
        run[i] = rr
        p = pc[i]
        v = vals[i]
        if kind == 2:
            if c[p] == 0 or v < m[p]:
                m[p] = v
        elif kind == 3:
            if c[p] == 0 or v > m[p]:
                m[p] = v
        s[p] += v
        c[p] += 1
        if kind == 0:
            fwd[i] = s[p] / c[p]
        elif kind == 1:
            fwd[i] = s[p]
        elif kind == 4:
            fwd[i] = c[p]
        else:
            fwd[i] = m[p]
    for i in range(n - 1, -1, -1):
        p = pc[i]
        if lr[p] == run[i]:
            fwd[i] = lv[p]
        else:
            lr[p] = run[i]
            lv[p] = fwd[i]


def running_agg(pc, vals, stairs, K, kind):
    """Running aggregate over (PARTITION p ORDER stairs-col), RANGE default frame:
    two walks of the room, no hallways. Returns float64 per-row values (exact for
    sum/min/max/count: integers below 2**53; avg = exact-sum / exact-count, duck's
    own arithmetic). kind: 'avg'|'sum'|'min'|'max'|'count'."""
    if not HAVE_NUMBA:
        return None
    kd = {'avg': 0, 'sum': 1, 'min': 2, 'max': 3, 'count': 4}[kind]
    s = np.zeros(K, np.float64)
    c = np.zeros(K, np.int64)
    m = np.zeros(K, np.float64)
    lr = np.full(K, -1, np.int64)
    lv = np.zeros(K, np.float64)
    fwd = np.empty(pc.size, np.float64)
    _running_nb(pc.astype(np.int64), vals.astype(np.float64),
                np.asarray(stairs).astype(np.int64), s, c, m, lr, lv, fwd, kd)
    return fwd


@njit(nogil=True, cache=True)
def _rank_walk_nb(pc, stairs, last_run, cnt, cur, out, dense):
    rr = 0
    for i in range(pc.size):
        while rr < stairs.size and i >= stairs[rr]:
            rr += 1
        p = pc[i]
        if last_run[p] != rr:
            last_run[p] = rr
            if dense:
                cur[p] += 1                  # dense: how many day-changes my pile has seen
            else:
                cur[p] = cnt[p] + 1          # rank: rows already in my pile when my day began
        out[i] = cur[p]
        cnt[p] += 1


def rank_walk(pc, stairs, K, dense):
    """RANK/DENSE_RANK over (PARTITION p ORDER stairs-col): one walk, no backward
    pass -- a batch's rank is fixed the instant the batch begins, nothing later can
    revise it. Day boundaries ride the stairs in O(1)/row; no order codes exist."""
    if not HAVE_NUMBA:
        return None
    last_run = np.full(K, -1, np.int64)
    cnt = np.zeros(K, np.int64)
    cur = np.zeros(K, np.int64)
    out = np.empty(pc.size, np.int64)
    _rank_walk_nb(pc.astype(np.int64), np.asarray(stairs).astype(np.int64),
                  last_run, cnt, cur, out, dense)
    return out


@njit(nogil=True, cache=True)
def _grid2_nb(a, b, Vb, board):
    for i in range(a.size):
        board[np.int64(a[i]) * Vb + np.int64(b[i])] += 1


@njit(nogil=True, cache=True)
def _grid3_nb(a, b, c, Vb, Vc, board):
    for i in range(a.size):
        board[(np.int64(a[i]) * Vb + np.int64(b[i])) * Vc + np.int64(c[i])] += 1


@njit(nogil=True, cache=True)
def _grid2_nb32(a, b, Vb, board):
    for i in range(a.size):
        board[np.int64(a[i]) * Vb + np.int64(b[i])] += np.int32(1)


@njit(nogil=True, cache=True)
def _grid3_nb32(a, b, c, Vb, Vc, board):
    for i in range(a.size):
        board[(np.int64(a[i]) * Vb + np.int64(b[i])) * Vc + np.int64(c[i])] += np.int32(1)


@njit(nogil=True, cache=True)
def _grid3_split(a, b, c, Vb, Vc, dcode, board2, board3):
    """The sparse grid (Jackson's dress on a hot loop): rows carrying the dominant
    third label pour into a small cache-resident 2-D wall; only the few percent that
    differ touch the big 3-D wall. Merge puts the 2-D wall into its slice."""
    for i in range(a.size):
        ab = np.int64(a[i]) * Vb + np.int64(b[i])
        if np.int64(c[i]) == dcode:
            board2[ab] += np.int32(1)
        else:
            board3[ab * Vc + np.int64(c[i])] += np.int32(1)


def grid_count(codes, spans):
    """The fused small-K board: one trip over the raw code streams (native dtypes,
    no astype, no composed key array) dropping beans on one composite board. Only
    populated cells go cache-hot, so even a 48M-cell board behaves small. Falls
    back to staged compose+bincount without numba."""
    total = 1
    for v in spans:
        total *= int(v)
    if HAVE_NUMBA:
        # HALF-WIDTH WALL (the weather-proofing cut): counts fit int32 at N=100M;
        # a 72MB board doubles the cache-resident share of the scatter vs 145MB.
        board = np.zeros(total, np.int32)
        if len(codes) == 2:
            _grid2_nb32(codes[0], codes[1], np.int64(spans[1]), board)
        else:
            c2 = np.asarray(codes[2])
            SMP = min(c2.size, 2_000_000)
            samp = np.bincount(c2[:SMP], minlength=int(spans[2]))
            dcode = int(samp.argmax()) if samp.size else -1
            if dcode >= 0 and int(samp[dcode]) * 5 > SMP * 3:   # dominant >60%: split
                board2 = np.zeros(int(spans[0]) * int(spans[1]), np.int32)
                _grid3_split(codes[0], codes[1], c2, np.int64(spans[1]),
                             np.int64(spans[2]), np.int64(dcode), board2, board)
                board.reshape(-1, int(spans[2]))[:, dcode] += board2
            else:
                _grid3_nb32(codes[0], codes[1], codes[2],
                            np.int64(spans[1]), np.int64(spans[2]), board)
        return board
    key = np.asarray(codes[0]).astype(np.int64)
    for i in range(1, len(codes)):
        key = key * int(spans[i]) + np.asarray(codes[i]).astype(np.int64)
    return np.bincount(key, minlength=total)


@njit(nogil=True, cache=True)
def _pd_stamp_nb(pidx, uid, jar, cnt):
    """First-touch distinct counting, sort-free (Jackson's guillotine finisher):
    jar[uid] holds a bitmask of pairs that already saw this user (u16: up to 16
    candidate pairs per batch); one pass, exact even when a user spans pairs."""
    for i in range(pidx.size):
        b = np.uint16(1) << np.uint16(pidx[i])
        w = jar[uid[i]]
        if w & b == 0:
            jar[uid[i]] = w | b
            cnt[pidx[i]] += 1


@njit(nogil=True, cache=True)
def _pd_hunt_nb(key, ut, lut, jar, cnt):
    """The fused hunt: filter + pair->batch map + first-touch stamp, one pass.
    lut[pairkey] = batch index or -1; no isin, no searchsorted, no gathers."""
    for i in range(key.size):
        p = lut[key[i]]
        if p >= 0:
            b = np.uint16(1) << np.uint16(p)
            w = jar[ut[i]]
            if w & b == 0:
                jar[ut[i]] = w | b
                cnt[p] += 1


def pd_hunt(key, ut, lut, V, npairs):
    jar = np.zeros(V, np.uint16)
    cnt = np.zeros(npairs, np.int64)
    if HAVE_NUMBA:
        _pd_hunt_nb(key, ut.astype(np.int64), lut, jar, cnt)
        return cnt
    m = lut[key] >= 0
    pidx = lut[key[m]]
    for p in range(npairs):
        cnt[p] = np.unique(ut[m][pidx == p]).size
    return cnt


@njit(nogil=True, cache=True)
def _mx_fold_nb(kc, ac, dv, acc):
    """One-pass weighted fold: acc[key] += dv[a-code]. Native dtypes in, no
    100M casts, no chunk ceremony -- the mixed board's sums in one walk."""
    for i in range(kc.size):
        acc[kc[i]] += dv[ac[i]]


@njit(nogil=True, parallel=True, cache=True)
def bp0_scan_count(buf, base, bits, lo, hi, flag, counts, CH):
    """Rule 3 pass 1, TILE-READER: fused unpack+flag-test; the value lives
    in a register, every packed byte loads once, matches count per chunk."""
    mask = (np.int64(1) << bits) - 1
    for cix in prange(counts.size):
        a = lo + cix * CH
        b = min(hi, a + CH)
        bo = a * bits
        p = base + (bo >> 3)
        acc = np.uint64(buf[p]) & np.uint64(0xFF >> (bo & 7))
        nb = 8 - (bo & 7)
        p += 1
        cnt = 0
        for i in range(a, b):
            while nb < bits:
                acc = (acc << np.uint64(8)) | np.uint64(buf[p])
                p += 1
                nb += 8
            nb -= bits
            if flag[np.int64(acc >> np.uint64(nb)) & mask]:
                cnt += 1
            acc &= (np.uint64(1) << np.uint64(nb)) - np.uint64(1)
        counts[cix] = cnt


@njit(nogil=True, parallel=True, cache=True)
def bp0_scan_fill(buf, base, bits, lo, hi, flag, offs, out, CH):
    """Rule 3 pass 2, TILE-READER: the same fused walk writing matching
    POSITIONS into contention-free slices."""
    mask = (np.int64(1) << bits) - 1
    for cix in prange(offs.size - 1):
        a = lo + cix * CH
        b = min(hi, a + CH)
        bo = a * bits
        p = base + (bo >> 3)
        acc = np.uint64(buf[p]) & np.uint64(0xFF >> (bo & 7))
        nb = 8 - (bo & 7)
        p += 1
        w9 = offs[cix]
        for i in range(a, b):
            while nb < bits:
                acc = (acc << np.uint64(8)) | np.uint64(buf[p])
                p += 1
                nb += 8
            nb -= bits
            if flag[np.int64(acc >> np.uint64(nb)) & mask]:
                out[w9] = i
                w9 += 1
            acc &= (np.uint64(1) << np.uint64(nb)) - np.uint64(1)


@njit(nogil=True, parallel=True, cache=True)
def bp0_decode(buf, base, bits, N, out):
    """Plain bitpack FULL decode, TILE-READER: every packed byte loads
    exactly once into a rolling accumulator; one shift+mask per value.
    Total loads = the packed bytes themselves -- the bandwidth floor."""
    mask = (np.int64(1) << bits) - 1
    CH = 1 << 18
    nch = (N + CH - 1) // CH
    for cix in prange(nch):
        a = cix * CH
        b = min(N, a + CH)
        bo = a * bits
        p = base + (bo >> 3)
        acc = np.uint64(buf[p]) & np.uint64(0xFF >> (bo & 7))
        nb = 8 - (bo & 7)
        p += 1
        for i in range(a, b):
            while nb < bits:
                acc = (acc << np.uint64(8)) | np.uint64(buf[p])
                p += 1
                nb += 8
            nb -= bits
            out[i] = np.int64(acc >> np.uint64(nb)) & mask
            acc &= (np.uint64(1) << np.uint64(nb)) - np.uint64(1)


@njit(nogil=True, parallel=True, cache=True)
def bp0_gather(buf, base, bits, rows, out):
    """Plain bitpack's random access, WORD-WISE: one u64 assembly per row,
    shift+mask. Rows near the buffer's end take the per-bit path."""
    n = rows.size
    mask = (np.int64(1) << bits) - 1
    limit = buf.size - 8
    for i in prange(n):
        o = rows[i] * bits
        j = base + (o >> 3)
        if j <= limit:
            w = np.uint64(0)
            for t in range(8):
                w = (w << np.uint64(8)) | np.uint64(buf[j + t])
            out[i] = np.int64(w >> np.uint64(64 - (o & 7) - bits)) & mask
        else:
            v = np.int64(0)
            for bi in range(bits):
                jb = o + bi
                byte = buf[base + (jb >> 3)]
                v = (v << 1) | ((np.int64(byte) >> (7 - (jb & 7))) & 1)
            out[i] = v


@njit(nogil=True, cache=True)
def _bp10_at(buf, dirX, pay, bits, row):
    """One enc-10 value at one row -- block-local, used by the fused walk."""
    B = 4096
    b = row // B
    o = pay + (dirX[b] >> 1)
    if dirX[b] & 1:
        nr = np.int64(buf[o]) | (np.int64(buf[o + 1]) << 8)
        p = o + 2
        acc = b * B
        for r in range(nr):
            cnt = np.int64(buf[p]) | (np.int64(buf[p + 1]) << 8)
            if row < acc + cnt:
                return np.int64(buf[p + 2]) | (np.int64(buf[p + 3]) << 8)
            acc += cnt
            p += 4
        return np.int64(0)
    r = row - b * B
    v = np.int64(0)
    base = r * bits
    for bi in range(bits):
        idx = base + bi
        v = (v << 1) | ((np.int64(buf[o + (idx >> 3)]) >> (7 - (idx & 7))) & 1)
    return v


@njit(nogil=True, parallel=True, cache=True)
def bp10_hygiene2(buf, d1, p1, b1, c1, eq1, d2, p2, b2, c2, eq2, rows, keep):
    """Jackson's fused hygiene: TWO enc-10 flag tests in ONE walk with
    per-row short-circuit -- the moment a row fails the first test, the
    second never runs (the crossing scheme at row granularity)."""
    n = rows.size
    for i in prange(n):
        v1 = _bp10_at(buf, d1, p1, b1, rows[i])
        ok = (v1 == c1) if eq1 else (v1 != c1)
        if ok:
            v2 = _bp10_at(buf, d2, p2, b2, rows[i])
            ok = (v2 == c2) if eq2 else (v2 != c2)
        keep[i] = ok


@njit(nogil=True, parallel=True, cache=True)
def bp10_gather(buf, dirX, pay, bits, rows, out):
    """enc-10's random access honored: decode ONLY the requested rows.
    rows must be sorted. Bitpack blocks are O(1) bit arithmetic per row;
    run blocks walk their few tokens once per touched block (merged with
    the block's requested rows, two-pointer)."""
    n = rows.size
    if n == 0:
        return
    # partition requested rows by block via prange over blocks touched
    nblk = dirX.size
    B = 4096
    for b in prange(nblk):
        lo_i = np.searchsorted(rows, b * B)
        hi_i = np.searchsorted(rows, (b + 1) * B)
        if hi_i <= lo_i:
            continue
        o = pay + (dirX[b] >> 1)
        if dirX[b] & 1:
            nr = np.int64(buf[o]) | (np.int64(buf[o + 1]) << 8)
            p = o + 2
            acc = b * B                          # running row cursor
            j = lo_i
            for r in range(nr):
                cnt = np.int64(buf[p]) | (np.int64(buf[p + 1]) << 8)
                val = np.int64(buf[p + 2]) | (np.int64(buf[p + 3]) << 8)
                p += 4
                nxt = acc + cnt
                while j < hi_i and rows[j] < nxt:
                    out[j] = val
                    j += 1
                acc = nxt
                if j >= hi_i:
                    break
        else:
            for j in range(lo_i, hi_i):
                r = rows[j] - b * B
                v = np.int64(0)
                base = r * bits
                for bi in range(bits):
                    idx = base + bi
                    byte = buf[o + (idx >> 3)]
                    v = (v << 1) | ((np.int64(byte) >> (7 - (idx & 7))) & 1)
                out[j] = v


@njit(nogil=True, parallel=True, cache=True)
def bp10_decode(buf, dirX, pay, bits, N, out):
    """enc-10's walker: prange over 4096-row blocks; run blocks repeat-fill
    from u16 (count,value) pairs, bitpack blocks extract MSB-first bit runs.
    The 24K-block python loop dies here."""
    B = 4096
    nblk = dirX.size
    for b in prange(nblk):
        lo = b * B
        rows = min(B, N - lo)
        o = pay + (dirX[b] >> 1)
        if dirX[b] & 1:
            nr = np.int64(buf[o]) | (np.int64(buf[o + 1]) << 8)
            p = o + 2
            w = lo
            for r in range(nr):
                cnt = np.int64(buf[p]) | (np.int64(buf[p + 1]) << 8)
                val = np.int64(buf[p + 2]) | (np.int64(buf[p + 3]) << 8)
                p += 4
                for i in range(cnt):
                    out[w] = val
                    w += 1
        else:
            for r in range(rows):
                v = np.int64(0)
                base = r * bits
                for bi in range(bits):
                    idx = base + bi
                    byte = buf[o + (idx >> 3)]
                    v = (v << 1) | ((np.int64(byte) >> (7 - (idx & 7))) & 1)
                out[lo + r] = v


@njit(nogil=True, parallel=True, cache=True)
def pf_prune(pos, bcol, heavy):
    """Q30's early stop: keep only typed positions whose b-code is HEAVY
    (cnt_ip bound: a pair can never outscore its IP's total). Two prange
    passes -- count, then stable scatter -- return the surviving crumb."""
    n = pos.size
    T = heavy.size and 16 or 16
    T = 16
    pc = np.zeros(T + 1, np.int64)
    for t in prange(T):
        lo = t * n // T
        hi = (t + 1) * n // T
        c = 0
        for i in range(lo, hi):
            if heavy[bcol[pos[i]]]:
                c += 1
        pc[t + 1] = c
    for t in range(T):
        pc[t + 1] += pc[t]
    out = np.empty(pc[T], np.int64)
    for t in prange(T):
        lo = t * n // T
        hi = (t + 1) * n // T
        w = pc[t]
        for i in range(lo, hi):
            p = pos[i]
            if heavy[bcol[p]]:
                out[w] = p
                w += 1
    return out


@njit(nogil=True, parallel=True, cache=True)
def pr_scatter(pos, acol, bcol, xcol, ycol, SH, T):
    """Q30's fused gather+radix: walk the planes' typed positions ONCE, read all
    four columns at the row, pack key=(b<<8|a) and pay=(x<<16|y), bucket by b's
    top bits. No intermediate gathers ever materialize."""
    n = pos.size
    NB = 1 << 12
    pc = np.zeros((T, NB), np.int64)
    for t in prange(T):
        lo = t * n // T
        hi = (t + 1) * n // T
        for i in range(lo, hi):
            pc[t, np.int64(bcol[pos[i]]) >> SH] += 1
    offs = np.zeros(NB + 1, np.int64)
    for b in range(NB):
        s0 = 0
        for t in range(T):
            v = pc[t, b]
            pc[t, b] = s0
            s0 += v
        offs[b + 1] = offs[b] + s0
    key = np.empty(n, np.int64)
    pay = np.empty(n, np.int64)
    for t in prange(T):
        lo = t * n // T
        hi = (t + 1) * n // T
        for i in range(lo, hi):
            p = pos[i]
            bb = np.int64(bcol[p])
            j = offs[bb >> SH] + pc[t, bb >> SH]
            pc[t, bb >> SH] += 1
            key[j] = (bb << 8) | np.int64(acol[p])
            pay[j] = (np.int64(xcol[p]) << 16) | np.int64(ycol[p])
    return key, pay, offs


@njit(nogil=True, parallel=True, cache=True)
def _mx_fold2_nb(kc, ac1, dv1, ac2, dv2, cnt2, s1, s2):
    """Jackson's parallel fold: ONE pass over the column, prange-split with
    thread-local accumulators (race-free), producing count + both weighted
    sums together. The gather parallelizes because nothing is sequential."""
    T = s1.shape[0]
    n = kc.size
    step = (n + T - 1) // T
    for t in prange(T):
        lo = t * step
        hi = min(n, lo + step)
        for i in range(lo, hi):
            k = kc[i]
            cnt2[t, k] += 1
            s1[t, k] += dv1[ac1[i]]
            s2[t, k] += dv2[ac2[i]]


def mx_fold2(kc, ac1, dv1, ac2, dv2, KV):
    if HAVE_NUMBA:
        import numba
        T = max(1, numba.get_num_threads())
        cnt2 = np.zeros((T, KV), np.int64)
        s1 = np.zeros((T, KV), np.float64)
        s2 = np.zeros((T, KV), np.float64)
        _mx_fold2_nb(kc, ac1, dv1, ac2, dv2, cnt2, s1, s2)
        return cnt2.sum(0), s1.sum(0), s2.sum(0)
    cnt = np.bincount(kc, minlength=KV)
    return cnt, mx_fold(kc, ac1, dv1, KV), mx_fold(kc, ac2, dv2, KV)


def mx_fold(kc, ac, dv, KV):
    acc = np.zeros(KV, np.float64)
    if HAVE_NUMBA:
        _mx_fold_nb(kc, ac, dv, acc)
        return acc
    CH = 1 << 23
    for lo in range(0, kc.size, CH):
        sl = slice(lo, min(kc.size, lo + CH))
        acc += np.bincount(kc[sl], weights=dv[ac[sl]], minlength=KV)
    return acc


def pd_stamp(pidx, uid, V, npairs):
    jar = np.zeros(V, np.uint16)
    cnt = np.zeros(npairs, np.int64)
    if HAVE_NUMBA:
        _pd_stamp_nb(pidx.astype(np.int64), uid.astype(np.int64), jar, cnt)
        return cnt
    for p in range(npairs):                        # numpy fallback: per-pair unique
        uu = np.unique(uid[pidx == p])
        cnt[p] = uu.size
    return cnt


def warm():
    """JIT-compile the kernels (call from prewarm; ~1 s once, cached on disk after)."""
    kway_topk(np.array([1, 2], np.int64), np.array([1, 1], np.int64),
              np.array([0, 1, 2], np.int64), 4)
    if HAVE_NUMBA:
        top10_i32(np.array([1, 2], np.int32), 4)
        part_scatter(np.array([1, 0, 1], np.int64), 2)
        _grid2_nb(np.array([0, 1], np.uint8), np.array([1, 0], np.uint8), 2, np.zeros(4, np.int64))
        _grid3_nb(np.array([0, 1], np.uint8), np.array([1, 0], np.uint8), np.array([0, 1], np.uint8), 2, 2, np.zeros(8, np.int64))
        pd_stamp(np.array([0, 1, 0], np.int64), np.array([3, 3, 3], np.int64), 8, 2)
        pd_hunt(np.array([0, 1, 0], np.int64), np.array([3, 3, 3], np.int64), np.array([0, 1], np.int16), 8, 2)
        mx_fold(np.array([0, 1, 0], np.int64), np.array([0, 1, 1], np.int64), np.array([2.0, 5.0]), 2)
        mx_fold2(np.array([0, 1, 0], np.int64), np.array([0, 1, 1], np.int64), np.array([2.0, 5.0]), np.array([1, 0, 1], np.int64), np.array([3.0, 4.0]), 2)
        _k9, _p9, _o9 = pr_scatter(np.array([0, 1, 2], np.int64), np.array([0, 1, 0], np.int64),
                                   np.array([1, 1, 2], np.int64), np.array([0, 1, 0], np.int64),
                                   np.array([0, 0, 1], np.int64), 0, 2)
        pf_prune(np.array([0, 1, 2], np.int64), np.array([1, 0, 1], np.int64), np.array([True, False]))
        _bw = np.zeros(8, np.uint8); _bw[0] = 2; _bw[1] = 0; _bw[2] = 3; _bw[3] = 0; _bw[4] = 1; _bw[5] = 0
        _bo = np.zeros(3, np.uint16)
        bp10_decode(_bw, np.array([1], np.int64), 0, 1, 3, _bo)
        _bg = np.zeros(2, np.uint16)
        _b0 = np.zeros(2, np.uint32)
        bp0_gather(_bw, 0, 1, np.array([0, 2], np.int64), _b0)
        _b1 = np.zeros(3, np.uint32)
        bp0_decode(_bw, 0, 1, 3, _b1)
        _kp = np.zeros(2, np.bool_)
        bp10_hygiene2(_bw, np.array([1], np.int64), 0, 1, 0, True,
                      np.array([1], np.int64), 0, 1, 0, True,
                      np.array([0, 2], np.int64), _kp)
        _fl = np.zeros(4, np.bool_); _fl[1] = True
        _ct = np.zeros(1, np.int64)
        bp0_scan_count(_bw, 0, 1, 0, 3, _fl, _ct, 4096)
        _po = np.zeros(max(1, int(_ct[0])), np.int64)
        bp0_scan_fill(_bw, 0, 1, 0, 3, _fl, np.array([0, int(_ct[0])], np.int64), _po, 4096)
        bp10_gather(_bw, np.array([1], np.int64), 0, 1, np.array([0, 2], np.int64), _bg)
        _c9 = np.zeros(3, np.int64); _s19 = np.zeros(3, np.float64); _s29 = np.zeros(3, np.float64)
        _u9 = np.zeros(3, np.int64); _n9 = np.zeros(1 << 12, np.int64)
        pr_fold(_k9, _p9, _o9, np.array([0.0, 1.0]), np.array([2.0, 3.0]), _c9, _s19, _s29, _u9, _n9)


@njit(nogil=True, parallel=True, cache=True)
def grouped_sum_codes(kc, vc, vt, K):
    """sums[k] += vt[vc[i]] for k = kc[i]: the single-key SUM board, fused gather+
    accumulate, per-thread partials (no atomics, no dtype casts, no factorize)."""
    T = numba.get_num_threads()
    part = np.zeros((T, K), np.int64)
    n = kc.size
    for t in prange(T):
        lo = t * n // T
        hi = (t + 1) * n // T
        for i in range(lo, hi):
            part[t, kc[i]] += vt[vc[i]]
    out = np.zeros(K, np.int64)
    for t in range(T):
        for k in range(K):
            out[k] += part[t, k]
    return out


@njit(nogil=True, parallel=True, cache=True)
def enc5_stream(packed, hot, patches, esc_off, N, BR):
    """Patched-bucket decode: 4-bit hot pointers unpack + escape patches, per-block
    independent (the escape-offset table is what makes prange lawful here)."""
    out = np.empty(N, dtype=np.uint16)
    nb = (N + BR - 1) // BR
    for b in prange(nb):
        lo = b * BR
        hi = lo + BR
        if hi > N:
            hi = N
        pi = esc_off[b]
        for i in range(lo, hi):
            byte = packed[i >> 1]
            if (i & 1) == 0:
                v = np.int64(byte & 0x0F)
            else:
                v = np.int64(byte >> 4)
            if v < 15:
                out[i] = hot[v]
            else:
                out[i] = patches[pi]
                pi += 1
    return out


@njit(nogil=True, cache=True)
def enc5_at(packed, hot, patches, esc_off, rows, BR):
    """Point reads on the patched-bucket encoding: one nibble per row; escapes rank
    themselves with a bounded within-block scan."""
    out = np.empty(rows.size, dtype=np.int64)
    for k in range(rows.size):
        i = rows[k]
        byte = packed[i >> 1]
        if (i & 1) == 0:
            v = np.int64(byte & 0x0F)
        else:
            v = np.int64(byte >> 4)
        if v < 15:
            out[k] = hot[v]
        else:
            b = i // BR
            seen = np.int64(0)
            for jj in range(b * BR, i):
                byte2 = packed[jj >> 1]
                if (jj & 1) == 0:
                    v2 = np.int64(byte2 & 0x0F)
                else:
                    v2 = np.int64(byte2 >> 4)
                if v2 == 15:
                    seen += 1
            out[k] = patches[esc_off[b] + seen]
    return out


@njit(nogil=True, parallel=True, cache=True)
def enc5_findpos(packed, esc_off, pidx, blk, N, BR):
    """Row positions of the given patch-array indices: each escape ranks itself with a
    bounded within-block nibble scan (blocks independent -> prange)."""
    pos = np.empty(pidx.size, dtype=np.int64)
    for k in prange(pidx.size):
        b = blk[k]
        want = pidx[k] - esc_off[b]
        seen = np.int64(0)
        lo = b * BR
        hi = lo + BR
        if hi > N:
            hi = N
        for i in range(lo, hi):
            byte = packed[i >> 1]
            if (i & 1) == 0:
                v = np.int64(byte & 0x0F)
            else:
                v = np.int64(byte >> 4)
            if v == 15:
                if seen == want:
                    pos[k] = i
                    break
                seen += 1
    return pos


@njit(nogil=True, parallel=True, cache=True)
def enc5_at2(packed, hot, patches, esc_off, rows_sorted, N, BR):
    """Block-grouped point reads: one nibble sweep per touched block serves every
    requested row in it (the v1 per-row escape re-scan was quadratic within blocks:
    738K stage-B positions cost 0.46s; this is one bounded pass per block, prange)."""
    out = np.empty(rows_sorted.size, dtype=np.int64)
    nreq = rows_sorted.size
    # block boundaries within the sorted request list
    nb = (N + BR - 1) // BR
    starts = np.empty(nreq, dtype=np.int64)
    for k in range(nreq):
        starts[k] = rows_sorted[k] // BR
    # find contiguous runs of equal block id
    run_lo = np.empty(nreq, dtype=np.int64)
    nruns = 0
    for k in range(nreq):
        if k == 0 or starts[k] != starts[k - 1]:
            run_lo[nruns] = k
            nruns += 1
    for r in prange(nruns):
        k0 = run_lo[r]
        k1 = run_lo[r + 1] if r + 1 < nruns else nreq
        b = starts[k0]
        lo = b * BR
        last = rows_sorted[k1 - 1]
        pi = esc_off[b]
        k = k0
        for i in range(lo, last + 1):
            byte = packed[i >> 1]
            if (i & 1) == 0:
                v = np.int64(byte & 0x0F)
            else:
                v = np.int64(byte >> 4)
            while k < k1 and i == rows_sorted[k]:
                if v < 15:
                    out[k] = hot[v]
                else:
                    out[k] = patches[pi]
                k += 1
            if v == 15:
                pi += 1
    return out


@njit(nogil=True, parallel=True, cache=True)
def pr_fold(key, pay, offs, dv1, dv2, ucnt, us1, us2, ukey, nruns):
    """Per-bucket argsort + run walk: fold count and both dictionary-valued sums
    per (b, a) pair. Each bucket writes runs into ITS OWN slice region -- no
    cursors, no contention."""
    NB = offs.size - 1
    for b in prange(NB):
        lo = offs[b]
        hi = offs[b + 1]
        if hi <= lo:
            nruns[b] = 0
            continue
        sl = np.argsort(key[lo:hi], kind='mergesort')
        w = lo
        r = 0
        i = 0
        m = hi - lo
        while i < m:
            kv = key[lo + sl[i]]
            c = 0
            s1 = 0.0
            s2 = 0.0
            while i < m and key[lo + sl[i]] == kv:
                pv = pay[lo + sl[i]]
                s1 += dv1[(pv >> 16) & 0xFFFF]
                s2 += dv2[pv & 0xFFFF]
                c += 1
                i += 1
            ukey[w + r] = kv
            ucnt[w + r] = c
            us1[w + r] = s1
            us2[w + r] = s2
            r += 1
        nruns[b] = r


@njit(nogil=True, parallel=True, cache=True)
def gd_pass1(uc, rc, SH, T):
    """MSD scatter pass 1: group (target, key) rows by the target code's top bits.
    4096 cursor buckets stay cache-resident; per-thread private counts merge into
    stable offsets so the pass parallelizes without contention."""
    n = uc.size
    NB = 1 << 12
    pc = np.zeros((T, NB), np.int64)
    for t in prange(T):
        lo = t * n // T
        hi = (t + 1) * n // T
        for i in range(lo, hi):
            pc[t, np.int64(uc[i]) >> SH] += 1
    offs = np.zeros(NB + 1, np.int64)
    for b in range(NB):
        s = 0
        for t in range(T):
            v = pc[t, b]
            pc[t, b] = s
            s += v
        offs[b + 1] = offs[b] + s
    ku = np.empty(n, np.uint32)                  # target codes fit u32 (V < 2^32): half the
    kr = np.empty(n, np.uint32)                  # cardboard, half the scatter bandwidth
    for t in prange(T):
        lo = t * n // T
        hi = (t + 1) * n // T
        cur = np.empty(NB, np.int64)
        for b in range(NB):
            cur[b] = offs[b] + pc[t, b]
        for i in range(lo, hi):
            b = np.int64(uc[i]) >> SH
            p = cur[b]
            ku[p] = uc[i]
            kr[p] = rc[i]
            cur[b] = p + 1
    return ku, kr, offs


@njit(nogil=True, parallel=True, cache=True)
def gd_pass2_count(ku, kr, offs, SH, VR):
    """MSD pass 2 + marker dedup, fused per bucket: scatter by the low bits (each
    bucket's targets fully contained), then the L1-resident marker table counts a
    target's first touch of each key. Per-bucket accumulator rows: no races."""
    NB = offs.size - 1
    LOW = 1 << SH
    ans = np.zeros((NB, VR), np.int64)
    for b in prange(NB):
        lo = offs[b]
        hi = offs[b + 1]
        if hi <= lo:
            continue
        cnt = np.zeros(LOW + 1, np.int64)
        for i in range(lo, hi):
            cnt[(np.int64(ku[i]) & (LOW - 1)) + 1] += 1
        loffs = np.cumsum(cnt)
        cur = loffs[:-1].copy()
        lr = np.empty(hi - lo, np.uint32)
        for i in range(lo, hi):
            u = np.int64(ku[i]) & (LOW - 1)
            lr[cur[u]] = kr[i]
            cur[u] += 1
        seen = np.full(VR, -1, np.int64)
        for u in range(LOW):
            for i in range(loffs[u], loffs[u + 1]):
                r = lr[i]
                if seen[r] != u:
                    seen[r] = u
                    ans[b, r] += 1
    total = np.zeros(VR, np.int64)
    for b in range(NB):
        for r in range(VR):
            total[r] += ans[b, r]
    return total


@njit(nogil=True, parallel=True, cache=True)
def grid2_count(c1, c2, fc, lit, V2, K):
    """Composite 2-key COUNT grid with optional eq-filter (lit<0 = unfiltered):
    per-thread boards, one fused pass, no factorize, no sorts."""
    T = numba.get_num_threads()
    part = np.zeros((T, K), np.int64)
    n = c1.size
    for t in prange(T):
        lo = t * n // T
        hi = (t + 1) * n // T
        for i in range(lo, hi):
            if lit >= 0 and np.int64(fc[i]) != lit:
                continue
            part[t, np.int64(c1[i]) * V2 + np.int64(c2[i])] += 1
    out = np.zeros(K, np.int64)
    for t in range(T):
        for k in range(K):
            out[k] += part[t, k]
    return out


@njit(nogil=True, parallel=True, cache=True)
def enc6_stream(packed, hot, warm, wbytes, patches, e1off, e2off, N, BR):
    """Two-tier patched buckets: 4-bit hot -> 8-bit warm -> u16 cold. Both escape
    levels carry per-block offsets, so every block decodes independently (prange)."""
    out = np.empty(N, dtype=np.uint16)
    nb = (N + BR - 1) // BR
    for b in prange(nb):
        lo = b * BR
        hi = lo + BR
        if hi > N:
            hi = N
        p1 = e1off[b]
        p2 = e2off[b]
        for i in range(lo, hi):
            byte = packed[i >> 1]
            if (i & 1) == 0:
                v = np.int64(byte & 0x0F)
            else:
                v = np.int64(byte >> 4)
            if v < 15:
                out[i] = hot[v]
            else:
                w = np.int64(wbytes[p1])
                p1 += 1
                if w < 255:
                    out[i] = warm[w]
                else:
                    out[i] = patches[p2]
                    p2 += 1
    return out


@njit(nogil=True, parallel=True, cache=True)
def enc6_at2(packed, hot, warm, wbytes, patches, e1off, e2off, rows_sorted, N, BR):
    """Block-grouped point reads for the two-tier encoding: one bounded nibble+warm
    sweep per touched block serves every requested row in it."""
    out = np.empty(rows_sorted.size, dtype=np.int64)
    nreq = rows_sorted.size
    starts = np.empty(nreq, dtype=np.int64)
    for k in range(nreq):
        starts[k] = rows_sorted[k] // BR
    run_lo = np.empty(nreq, dtype=np.int64)
    nruns = 0
    for k in range(nreq):
        if k == 0 or starts[k] != starts[k - 1]:
            run_lo[nruns] = k
            nruns += 1
    for r in prange(nruns):
        k0 = run_lo[r]
        k1 = run_lo[r + 1] if r + 1 < nruns else nreq
        b = starts[k0]
        lo = b * BR
        last = rows_sorted[k1 - 1]
        p1 = e1off[b]
        p2 = e2off[b]
        k = k0
        for i in range(lo, last + 1):
            byte = packed[i >> 1]
            if (i & 1) == 0:
                v = np.int64(byte & 0x0F)
            else:
                v = np.int64(byte >> 4)
            val = np.int64(-1)
            if v < 15:
                val = np.int64(hot[v])
            else:
                w = np.int64(wbytes[p1])
                p1 += 1
                if w < 255:
                    val = np.int64(warm[w])
                else:
                    val = np.int64(patches[p2])
                    p2 += 1
            while k < k1 and i == rows_sorted[k]:
                out[k] = val
                k += 1
    return out


@njit(nogil=True, parallel=True, cache=True)
def unpack24_be(b, n):
    """Byte-aligned 24-bit MSB-first lanes -> uint32: the throughput law's read."""
    out = np.empty(n, dtype=np.uint32)
    for i in prange(n):
        j = i * 3
        out[i] = (np.uint32(b[j]) << 16) | (np.uint32(b[j + 1]) << 8) | np.uint32(b[j + 2])
    return out


@njit(nogil=True, parallel=True, cache=True)
def unpack_any(b, n, bits):
    """MSB-first bitpack -> uint32 at any width <= 25, parallel. The sparse dress's
    literal lane; the generic sliding-window unpacker cost 400ms where this costs ~40."""
    out = np.empty(n, dtype=np.uint32)
    mask = np.uint64((1 << bits) - 1)
    safe = n - 3 if n > 3 else 0
    for i in prange(safe):
        o = i * bits
        j = o >> 3
        sh = o & 7
        acc = np.uint64(0)
        for kk in range(5):
            acc = (acc << np.uint64(8)) | np.uint64(b[j + kk])
        out[i] = np.uint32((acc >> np.uint64(40 - sh - bits)) & mask)
    for i in range(safe, n):
        o = i * bits
        j = o >> 3
        sh = o & 7
        acc = np.uint64(0)
        nb = len(b) - j
        for kk in range(5):
            v = np.uint64(b[j + kk]) if kk < nb else np.uint64(0)
            acc = (acc << np.uint64(8)) | v
        out[i] = np.uint32((acc >> np.uint64(40 - sh - bits)) & mask)
    return out


@njit(nogil=True, parallel=True, cache=True)
def e8_pos(pres_bytes, ck, n, pos):
    """Pass 1 of the sparse reconstruct: collect present-row positions per 64K chunk,
    each chunk writing its own slice from its checkpoint rank. Zero bytes identify
    eight blanks in one compare (Jackson's principle, done plainly this time)."""
    nck = (n + 65535) >> 16
    for cb in prange(nck):
        row0 = cb << 16
        row1 = min(row0 + 65536, n)
        w = np.int64(ck[cb])
        nb = (row1 - row0) >> 3
        for bi in range(nb):
            b8 = pres_bytes[(row0 >> 3) + bi]
            if b8 == 0:
                continue
            base = row0 + (bi << 3)
            for bit in range(8):
                if (b8 >> (7 - bit)) & 1:
                    pos[w] = base + bit
                    w += 1
        for r in (row0 + (nb << 3)), row1:
            pass
        for r in range(row0 + (nb << 3), row1):
            if (pres_bytes[r >> 3] >> (7 - (r & 7))) & 1:
                pos[w] = r
                w += 1


@njit(nogil=True, parallel=True, cache=True)
def e8_scatter(pos, lits, n, default):
    """Pass 3: default-fill + scatter, parallel."""
    out = np.full(n, np.uint32(default), dtype=np.uint32)
    for i in prange(pos.size):
        out[pos[i]] = lits[i]
    return out


@njit(nogil=True, parallel=True, cache=True)
def unpack_any_off(b, n, bits, bit0):
    """unpack_any with a starting bit offset: range reads begin mid-byte."""
    out = np.empty(n, dtype=np.uint32)
    mask = np.uint64((1 << bits) - 1)
    for i in prange(n):
        o = bit0 + i * bits
        j = o >> 3
        sh = o & 7
        acc = np.uint64(0)
        for kk in range(5):
            acc = (acc << np.uint64(8)) | np.uint64(b[j + kk])
        out[i] = np.uint32((acc >> np.uint64(40 - sh - bits)) & mask)
    return out


@njit(nogil=True, cache=True)
def fc_charlens(a, R, out):
    """Front-coded walk in pure arithmetic: per entry <HH cp,sl>+suffix; char length =
    (cp+sl) - continuation bytes, with the prefix chain's continuation counts carried
    in a cumulative buffer. No bytes object is ever built."""
    o = np.int64(0); i = np.int64(0); n = np.int64(0)
    cum = np.zeros(4096, np.int64)               # cum[j] = continuation bytes in prev[:j]
    plen = np.int64(0)
    while o < a.size:
        if i % R == 0:
            plen = 0
        cp = np.int64(a[o]) | (np.int64(a[o + 1]) << 8)
        sl = np.int64(a[o + 2]) | (np.int64(a[o + 3]) << 8)
        o += 4
        base = cum[cp] if cp <= plen else cum[plen]
        j = cp
        for t in range(sl):
            b = a[o + t]
            cont = np.int64(1) if (b & 0xC0) == 0x80 else np.int64(0)
            base2 = base + cont
            if j + 1 < 4096:
                cum[j + 1] = base2
            base = base2
            j += 1
        out[n] = (cp + sl) - base + (cum[cp] if cp <= plen else cum[plen])
        # rewrite: charlen = total_bytes - total_cont; total_cont = cum[cp] + suffix_cont
        out[n] = (cp + sl) - base
        o += sl
        plen = cp + sl
        n += 1
        i += 1
    return n


@njit(nogil=True, cache=True)
def fc_bytelens(a, R, out):
    """Front-coded walk, BYTE lengths: per entry <HH cp,sl>, length = cp + sl."""
    o = np.int64(0); n = np.int64(0)
    while o < a.size:
        cp = np.int64(a[o]) | (np.int64(a[o + 1]) << 8)
        sl = np.int64(a[o + 2]) | (np.int64(a[o + 3]) << 8)
        out[n] = cp + sl
        o += 4 + sl
        n += 1
    return n


@njit(nogil=True, cache=True)
def lenagg_pour(kc, uc, lens, jars, cnts, ec):
    """Q27's fused pour: one pass, both jars, integer math, no temporaries.
    jars[k] += lens[u]; cnts[k] += 1 for live rows (u != ec; ec = -1 counts all)."""
    for i in range(kc.size):
        u = np.int64(uc[i])
        if u != ec:
            kk = np.int64(kc[i])
            jars[kk] += np.int64(lens[u])
            cnts[kk] += 1


@njit(nogil=True, cache=True)
def fc_hostruns(a, R, brk, hend, labuf, laboff, meta):
    """Jackson's prefix-run walk with the full regex law: byte-exact http(s)
    schemes, www backtracking, and newlines -- '.' never matches \n, so any
    \n strictly after the host slash kills the match (a single TRAILING \n
    survives, labelled host+\n). Runs continue only while the copy-prefix
    clears the slash AND the tail stays newline-clean."""
    o = np.int64(0); i = np.int64(0); nr = np.int64(0); lw = np.int64(0)
    prev = np.zeros(8192, np.uint8)
    plen = np.int64(0)
    W = np.int64(-2)          # current run's host-end (the slash); -2 = no run
    Whs = np.int64(-1)        # current run's host start
    pnl = np.int64(-1)        # first \n strictly after W in prev, else -1
    while o < a.size:
        cp = np.int64(a[o]) | (np.int64(a[o + 1]) << 8)
        sl = np.int64(a[o + 2]) | (np.int64(a[o + 3]) << 8)
        o += 4
        if i % R == 0:
            cp = np.int64(0)
        # first \n after W contributed by the copied prefix
        enl = np.int64(-1)
        if W >= 0 and pnl >= 0 and pnl < cp:
            enl = pnl
        for t in range(sl):
            b9 = a[o + t]
            if cp + t < 8192:
                prev[cp + t] = b9
            if b9 == 10 and enl < 0 and W >= 0 and cp + t > W:
                enl = cp + t
        o += sl
        plen = cp + sl
        cont = (W >= 0) and (cp > W) and (enl < 0)
        if cont:
            brk[i] = 0
            hend[i] = W
            pnl = np.int64(-1)
            i += 1
            continue
        brk[i] = 1
        hs = np.int64(-1); he = np.int64(-1)
        okh = (plen > 8 and prev[0] == 104 and prev[1] == 116
               and prev[2] == 116 and prev[3] == 112)
        if okh and prev[4] == 58 and prev[5] == 47 and prev[6] == 47:
            hs = np.int64(7)
        elif (okh and plen > 9 and prev[4] == 115 and prev[5] == 58
              and prev[6] == 47 and prev[7] == 47):
            hs = np.int64(8)
        if hs >= 0:
            hs0 = hs
            if plen > hs + 4 and prev[hs] == 119 and prev[hs+1] == 119 and prev[hs+2] == 119 and prev[hs+3] == 46:
                hs += 4
            j2 = hs
            while j2 < plen and j2 < 8192:
                if prev[j2] == 47:
                    he = j2
                    break
                j2 += 1
            if he <= hs and hs != hs0:
                hs = hs0
                j2 = hs
                he = np.int64(-1)
                while j2 < plen and j2 < 8192:
                    if prev[j2] == 47:
                        he = j2
                        break
                    j2 += 1
        # newline law at the break: first \n strictly after he
        bnl = np.int64(-1)
        if he > hs:
            j3 = he + 1
            while j3 < plen and j3 < 8192:
                if prev[j3] == 10:
                    bnl = j3
                    break
                j3 += 1
        matched = (he > hs) and (bnl < 0)   # RE2 law: $ is absolute end --
                                             # ANY newline after the slash kills it
        if matched:
            hend[i] = he
            W = he
            Whs = hs
            pnl = np.int64(-1)
            a0, b0 = hs, he
        else:
            hend[i] = -1
            W = np.int64(-2)
            Whs = np.int64(-1)
            pnl = np.int64(-1)
            a0 = np.int64(0)
            b0 = plen if plen < 8192 else np.int64(8192)
        if lw + (b0 - a0) > labuf.size or nr + 1 >= laboff.size:
            meta[0] = -1
            return nr
        laboff[nr] = lw
        for t in range(b0 - a0):
            labuf[lw + t] = prev[a0 + t]
        lw += b0 - a0
        nr += 1
        i += 1
    laboff[nr] = lw
    meta[0] = i
    return nr


@njit(nogil=True, cache=True)
def cd_scatter(lits, uc, offs, cur, out):
    """Jackson's bucket pass: one walk scatters each row's user-code into its
    phrase's slice. Replaces the 13.2M argsort."""
    for i in range(lits.size):
        g = lits[i]
        out[cur[g]] = uc[i]
        cur[g] += 1


@njit(nogil=True, cache=True)
def cd_hunt(bucketed, offs, big, counts, k):
    """The bounded hunt in one kernel: walk groups biggest-first, count uniques
    by sorting each group's small slice, stop when the next bound falls
    strictly below the k-th exact answer. Returns (codes, uniques, filled)."""
    bestd = np.zeros(k, np.int64)
    bestc = np.zeros(k, np.int64)
    filled = 0
    kth = np.int64(-1)
    for bi in range(big.size):
        code = big[bi]
        cnt = counts[code]
        if cnt == 0 or cnt < kth:
            break
        s = np.sort(bucketed[offs[code]:offs[code] + cnt])
        d = np.int64(1)
        for t in range(1, s.size):
            if s[t] != s[t - 1]:
                d += 1
        if filled < k:
            bestd[filled] = d
            bestc[filled] = code
            filled += 1
            if filled == k:
                # establish kth
                m = bestd[0]
                for t in range(k):
                    if bestd[t] < m:
                        m = bestd[t]
                kth = m
        elif d > kth:
            # replace the current minimum
            mi = 0
            for t in range(1, k):
                if bestd[t] < bestd[mi]:
                    mi = t
            bestd[mi] = d
            bestc[mi] = code
            m = bestd[0]
            for t in range(k):
                if bestd[t] < m:
                    m = bestd[t]
            kth = m
    return bestc, bestd, filled


@njit(nogil=True, parallel=True, cache=True)
def cd_alldistinct(bucketed, offs):
    """Distinct count for EVERY group: per-slice sort + adjacent-diff, groups
    across threads. Feeds the gdc sidecar's big-key birth."""
    G = offs.size - 1
    out = np.zeros(G, np.int64)
    for g in prange(G):
        a, b = offs[g], offs[g + 1]
        if b > a:
            s = np.sort(bucketed[a:b])
            d = np.int64(1)
            for t in range(1, s.size):
                if s[t] != s[t - 1]:
                    d += 1
            out[g] = d
    return out


@njit(nogil=True, parallel=True, cache=True)
def tt_survivors(uc, ucnt, pos8, lits8, spc, ec, mt, e0, emptyc, theta, outs, lens, has_m):
    """Q18's fused pass: per-thread chunks walk the rows once -- user bound
    from the census, phrase bound via two-pointer merge with the sorted
    sparse planes (no 100M scratch arrays), survivors emit packed 54-bit
    keys (uid<<29 | minute<<23 | sp) straight into per-thread buffers."""
    T = outs.shape[0]
    N = uc.size
    step = (N + T - 1) // T
    for t in prange(T):
        a = t * step
        b = min(N, a + step)
        # position the plane cursor at the first plane row >= a
        lo = np.int64(0); hi = np.int64(pos8.size)
        while lo < hi:
            mid = (lo + hi) >> 1
            if pos8[mid] < a:
                lo = mid + 1
            else:
                hi = mid
        p = lo
        w = np.int64(0)
        for i in range(a, b):
            u = np.int64(uc[i])
            if ucnt[u] < theta:
                if p < pos8.size and pos8[p] == i:
                    p += 1
                continue
            if p < pos8.size and pos8[p] == i:
                s = np.int64(lits8[p])
                p += 1
                if spc[s] < theta:
                    continue
            else:
                s = e0
                if emptyc < theta:
                    continue
            if has_m:
                outs[t, w] = (u << 29) | (np.int64(mt[ec[i]]) << 23) | s
            else:
                outs[t, w] = (u << 29) | s   # pair shape: minute bits stay clear
            w += 1
        lens[t] = w


@njit(nogil=True, cache=True)
def pt_census(uc, jar):
    """u8 saturating census: 'seen once / seen again' is all the singleton-
    discard law needs. Sequential to keep counts exact (no write races)."""
    for i in range(uc.size):
        c = jar[uc[i]]
        if c < 2:
            jar[uc[i]] = c + 1


@njit(nogil=True, parallel=True, cache=True)
def pt_collect(uc, jar, outs, lens):
    """Parallel survivor collection: rows whose ticket repeats (jar==2)."""
    T = outs.shape[0]
    N = uc.size
    step = (N + T - 1) // T
    for t in prange(T):
        a = t * step
        b = min(N, a + step)
        w = np.int64(0)
        for i in range(a, b):
            if jar[uc[i]] >= 2:
                outs[t, w] = i
                w += 1
        lens[t] = w


@njit(nogil=True, parallel=True, cache=True)
def pt_census_bucketed(ku, kr, offs, SH, outs, lens):
    """Q32's parallel census on gd_pass1's buckets: each bucket owns a disjoint
    code range, so its jar slice never races. Saturate at 2, collect the rows
    (companions) of every repeated ticket -- one prange body, no shared writes."""
    NB = offs.size - 1
    LOW = 1 << SH
    for b in prange(NB):
        lo = offs[b]
        hi = offs[b + 1]
        w = np.int64(0)
        if hi > lo:
            jar = np.zeros(LOW, np.uint8)
            for i in range(lo, hi):
                c = jar[np.int64(ku[i]) & (LOW - 1)]
                if c < 2:
                    jar[np.int64(ku[i]) & (LOW - 1)] = c + 1
            for i in range(lo, hi):
                if jar[np.int64(ku[i]) & (LOW - 1)] >= 2:
                    outs[b, w] = kr[i]
                    w += 1
        lens[b] = w
