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


def _nt():
    """THE CACHEABLE THREAD COUNT (2026-09-24): numba.get_num_threads() or get_thread_id() INSIDE a
    compiled kernel embeds a pointer into the threading layer, which numba cannot write to its
    cache ("uses dynamic globals") -- so the kernel recompiled in EVERY process: gd_pass2_count
    2.2 s, sort_keys_par 2.8 s, group_fold_dict 1.9 s, _count_ge 0.9 s of ClickBench's cold runs
    (Q08/Q09/Q10/Q15/Q16/Q18). The count is read here, in Python, and passed in as an argument;
    the arithmetic is unchanged (per-thread boards summed, exact for any T)."""
    return np.int64(numba.get_num_threads()) if HAVE_NUMBA else np.int64(1)


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
def _part_scatter_nb(codes, K, T):
    """Stable parallel counting scatter: perm such that codes[perm] is grouped by code with
    original order preserved inside each group -- the fused motion, compiled. Two passes:
    per-chunk histograms -> exclusive global/chunk offsets -> stable scatter. T = the thread
    count, passed in (see THE CACHEABLE THREAD COUNT)."""
    N = codes.size
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
        return _part_scatter_nb(codes, np.int64(K), _nt())
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


@njit(nogil=True, cache=True)
def _t64(a):
    """Hacker's Delight 64x64 bit-matrix transpose, in place -- the tapes'
    un-rotation: 64 rows' values from bits-many plane words."""
    j = np.uint64(32)
    m = np.uint64(0x00000000FFFFFFFF)
    while j != np.uint64(0):
        k = 0
        while k < 64:
            for i in range(k, k + int(j)):
                t = (a[i] ^ (a[i + int(j)] >> j)) & m
                a[i] ^= t
                a[i + int(j)] ^= (t << j)
            k = k + int(j) * 2
        j >>= np.uint64(1)
        m ^= (m << j)


@njit(nogil=True, parallel=True, cache=True)
def vp_window(pl, nwords, bits, lo, hi, out):
    """Jackson's tapes: decode rows [lo,hi) from flat planes via the
    64x64 transpose -- the realm's fastest windowed reader."""
    w_lo = lo // 64
    w_hi = (hi + 63) // 64
    sh = np.uint64(64 - bits)
    for wb in prange(w_hi - w_lo):
        w = w_lo + wb
        a = np.zeros(64, np.uint64)
        for p in range(bits):
            a[p] = pl[p * nwords + w]
        _t64(a)
        r0 = w * 64
        for j in range(64):
            r = r0 + j
            if r >= lo and r < hi:
                out[r - lo] = a[63 - j] >> sh


@njit(nogil=True, parallel=True, cache=True)
def radix_hist12(keys, lo_shift, hists, CH):
    """Pass histogram: 12-bit digit counts per chunk (4096 counters, L1)."""
    nch = hists.shape[0]
    n = keys.size
    for cix in prange(nch):
        a = cix * CH
        b = min(n, a + CH)
        for i in range(a, b):
            hists[cix, (keys[i] >> lo_shift) & 0xFFF] += 1


@njit(nogil=True, parallel=True, cache=True)
def radix_scatter12(keys, lo_shift, offs, okeys, CH):
    """Stable scatter by 12-bit digit using per-chunk running offsets."""
    nch = offs.shape[0]
    n = keys.size
    for cix in prange(nch):
        a = cix * CH
        b = min(n, a + CH)
        for i in range(a, b):
            d = (keys[i] >> lo_shift) & 0xFFF
            j = offs[cix, d]
            offs[cix, d] = j + 1
            okeys[j] = keys[i]


@njit(nogil=True, parallel=True, cache=True)
def radix_scatter12p(keys, pay, lo_shift, offs, okeys, opay, CH):
    """Stable scatter carrying a payload word -- Jackson's partition pass:
    the top digit routes rows into buckets, the 62-bit remainder rides
    along, and each bucket then fits the single-word engine."""
    nch = offs.shape[0]
    n = keys.size
    for cix in prange(nch):
        a = cix * CH
        b = min(n, a + CH)
        for i in range(a, b):
            d = (keys[i] >> lo_shift) & 0xFFF
            j = offs[cix, d]
            offs[cix, d] = j + 1
            okeys[j] = keys[i]
            opay[j] = pay[i]


def radix_partition(top, pay):
    """Partition rows by a <=12-bit top digit (stable), payload alongside.
    Returns (sorted_top, sorted_pay, bucket_bounds)."""
    n = top.size
    CH = 1 << 16
    nch = (n + CH - 1) // CH
    k0 = np.ascontiguousarray(top, dtype=np.int64)
    p0 = np.ascontiguousarray(pay, dtype=np.int64)
    hists = np.zeros((nch, 4096), np.int64)
    radix_hist12(k0, 0, hists, CH)
    tot = hists.sum(axis=0)
    base = np.zeros(4096, np.int64)
    np.cumsum(tot[:-1], out=base[1:])
    offs = np.empty((nch, 4096), np.int64)
    run = base
    for cix in range(nch):
        offs[cix] = run
        run = run + hists[cix]
    ok = np.empty(n, np.int64)
    op = np.empty(n, np.int64)
    radix_scatter12p(k0, p0, 0, offs, ok, op, CH)
    nz = np.flatnonzero(tot)
    bounds = [(int(d), int(base[d]), int(base[d] + tot[d])) for d in nz]
    return ok, op, bounds


def radix_sortN(keys, bits):
    """THE RADIX ATOM, general form: sort integer keys of known width in
    ceil(bits/12) LSD passes -- O(n), L1 histogram bowls, never mutates."""
    n = keys.size
    if n == 0:
        return np.asarray(keys, np.int64)
    CH = 1 << 16
    nch = (n + CH - 1) // CH
    k0 = np.array(keys, dtype=np.int64, copy=True)
    t_k = np.empty(n, np.int64)
    sh = 0
    while sh < bits:
        hists = np.zeros((nch, 4096), np.int64)
        radix_hist12(k0, sh, hists, CH)
        tot = hists.sum(axis=0)
        base = np.zeros(4096, np.int64)
        np.cumsum(tot[:-1], out=base[1:])
        offs = np.empty((nch, 4096), np.int64)
        run = base
        for cix in range(nch):
            offs[cix] = run
            run = run + hists[cix]
        radix_scatter12(k0, sh, offs, t_k, CH)
        k0, t_k = t_k, k0
        sh += 12
    return k0


def radix_sort24(keys):
    return radix_sortN(keys, 24)


@njit(nogil=True, cache=True)
def vbits_set(rows, mask):
    """Positions -> bitmap: the crumb in mask currency."""
    for i in range(rows.size):
        r = rows[i]
        mask[r >> 6] |= np.uint64(1) << np.uint64(r & 63)


@njit(nogil=True, parallel=True, cache=True)
def vbits_pop(mask, out):
    """Per-word popcounts: the lockstep gather's free output offsets."""
    for i in prange(mask.size):
        x = mask[i]
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        out[i] = np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


_REV8 = np.array([int('{:08b}'.format(i)[::-1], 2) for i in range(256)],
                 dtype=np.uint8)


@njit(nogil=True, parallel=True, cache=True)
def flag1_pass(buf, d, p, blo, bhi, want_one, rev, fmb):
    """ONE-BIT flags ARE bitmasks: the bitpack payload, byte-reversed to
    the mask's bit order, IS the pass mask. Per block: 512 byte lookups.
    Run-list blocks (rare for 1-bit) expand the slow way."""
    B = 4096
    k0 = blo // B
    k1 = (bhi + B - 1) // B
    for kb in prange(k1 - k0):
        blk = k0 + kb
        o = p + (d[blk] >> 1)
        base = blk * B
        start = base if base > blo else blo
        if d[blk] & 1:
            nr = np.int64(buf[o]) | (np.int64(buf[o + 1]) << 8)
            ptr = o + 2
            r = base
            for _ in range(nr):
                cnt = np.int64(buf[ptr]) | (np.int64(buf[ptr + 1]) << 8)
                val = np.int64(buf[ptr + 2]) | (np.int64(buf[ptr + 3]) << 8)
                ptr += 4
                e = min(r + cnt, bhi)
                ok = (val == 1) if want_one else (val == 0)
                if ok and e > start:
                    for rr in range(max(r, start), e):
                        fmb[(rr - blo) >> 3] |= np.uint8(1 << (rr & 7))
                r = r + cnt
                if r >= bhi:
                    break
        else:
            hi = min(base + B, bhi)
            kk0 = (start - base) >> 3
            nby = (hi - base) >> 3
            ob = (base - blo) >> 3
            if want_one:
                for k in range(kk0, nby):
                    fmb[ob + k] = rev[buf[o + k]]
            else:
                for k in range(kk0, nby):
                    fmb[ob + k] = rev[buf[o + k]] ^ np.uint8(0xFF)


@njit(nogil=True, parallel=True, cache=True)
def _flag_pass(buf, d, p, b, c, eq, blo, bhi, fm):
    """Surgery with the polarity RIGHT: SET bits where the test PASSES --
    the passing majority arrives as long runs, so whole-word fills do
    almost all the work; the AND into the crumb mask follows."""
    B = 4096
    k0 = blo // B
    k1 = (bhi + B - 1) // B
    for kb in prange(k1 - k0):
        blk = k0 + kb
        o = p + (d[blk] >> 1)
        base = blk * B
        a0 = base if base > blo else blo
        a1 = base + B if base + B < bhi else bhi
        if d[blk] & 1:
            nr = np.int64(buf[o]) | (np.int64(buf[o + 1]) << 8)
            ptr = o + 2
            r = base
            for _ in range(nr):
                cnt = np.int64(buf[ptr]) | (np.int64(buf[ptr + 1]) << 8)
                val = np.int64(buf[ptr + 2]) | (np.int64(buf[ptr + 3]) << 8)
                ptr += 4
                e = r + cnt
                ok = (val == c) if eq else (val != c)
                if ok:
                    x0 = r if r > a0 else a0
                    x1 = e if e < a1 else a1
                    rr = x0
                    while rr < x1:
                        w = (rr - blo) >> 6
                        if (rr & 63) == 0 and rr + 64 <= x1:
                            fm[w] = ~np.uint64(0)
                            rr += 64
                        else:
                            fm[w] |= np.uint64(1) << np.uint64(rr & 63)
                            rr += 1
                r = e
                if r >= bhi:
                    break
        else:
            for rr in range(a0, a1):
                idx = (rr - base) * b
                v = np.int64(0)
                for bi in range(b):
                    ix = idx + bi
                    v = (v << 1) | ((np.int64(buf[o + (ix >> 3)]) >> (7 - (ix & 7))) & 1)
                ok = (v == c) if eq else (v != c)
                if ok:
                    fm[(rr - blo) >> 6] |= np.uint64(1) << np.uint64(rr & 63)


@njit(nogil=True, parallel=True, cache=True)
def _flag_mask(buf, d, p, b, c, eq, blo, bhi, mask):
    """Hygiene as MASK SURGERY: decode the flag's band block by block and
    clear mask bits where the test fails. No keep array, no handoff."""
    B = 4096
    k0 = blo // B
    k1 = (bhi + B - 1) // B
    for kb in prange(k1 - k0):
        blk = k0 + kb
        o = p + (d[blk] >> 1)
        base = blk * B
        a0 = base if base > blo else blo
        a1 = base + B if base + B < bhi else bhi
        if d[blk] & 1:
            nr = np.int64(buf[o]) | (np.int64(buf[o + 1]) << 8)
            ptr = o + 2
            r = base
            for _ in range(nr):
                cnt = np.int64(buf[ptr]) | (np.int64(buf[ptr + 1]) << 8)
                val = np.int64(buf[ptr + 2]) | (np.int64(buf[ptr + 3]) << 8)
                ptr += 4
                e = r + cnt
                ok = (val == c) if eq else (val != c)
                if not ok:
                    x0 = r if r > a0 else a0
                    x1 = e if e < a1 else a1
                    rr = x0
                    while rr < x1:
                        w = (rr - blo) >> 6
                        j = np.uint64((rr - blo) & 63)
                        if (rr & 63) == 0 and rr + 64 <= x1:
                            mask[w] = np.uint64(0)
                            rr += 64
                        else:
                            mask[w] &= ~(np.uint64(1) << j)
                            rr += 1
                r = e
                if r >= bhi:
                    break
        else:
            for rr in range(a0, a1):
                idx = (rr - base) * b
                v = np.int64(0)
                for bi in range(b):
                    ix = idx + bi
                    v = (v << 1) | ((np.int64(buf[o + (ix >> 3)]) >> (7 - (ix & 7))) & 1)
                ok = (v == c) if eq else (v != c)
                if not ok:
                    w = (rr - blo) >> 6
                    mask[w] &= ~(np.uint64(1) << np.uint64((rr - blo) & 63))


@njit(nogil=True, cache=True)
def _offsets_from_mask(mask, CHW, ob):
    tot = 0
    nch = ob.size
    for cix in range(nch):
        ob[cix] = tot
        a0 = cix * CHW
        a1 = min(mask.size, a0 + CHW)
        for i in range(a0, a1):
            x = mask[i]
            x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
            x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
            x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
            tot += np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))
    return tot


@njit(nogil=True, cache=True)
def _radix_inplace(k0, t_k, bits, CH):
    n = k0.size
    nch = (n + CH - 1) // CH
    src = k0
    dst = t_k
    sh = 0
    while sh < bits:
        hists = np.zeros((nch, 4096), np.int64)
        radix_hist12(src, sh, hists, CH)
        run = np.zeros(4096, np.int64)
        tot = 0
        for d in range(4096):
            run[d] = tot
            for cix in range(nch):
                tot += hists[cix, d]
        offs = np.empty((nch, 4096), np.int64)
        for cix in range(nch):
            for d in range(4096):
                offs[cix, d] = run[d]
                run[d] += hists[cix, d]
        radix_scatter12(src, sh, offs, dst, CH)
        src, dst = dst, src
        sh += 12
    return src


@njit(nogil=True, parallel=True, cache=True)
def mask_to_rows(mask, w0, ob, out, CHW):
    """Surviving mask bits -> absolute row positions, compacted per chunk."""
    nch = ob.size
    for cix in prange(nch):
        a0 = cix * CHW
        a1 = min(mask.size, a0 + CHW)
        k = ob[cix]
        for w in range(a0, a1):
            mw = mask[w]
            base = (w0 + w) * 64
            while mw != np.uint64(0):
                t = mw & ((~mw) + np.uint64(1))
                x = t - np.uint64(1)
                x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
                x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
                x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
                j = int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))
                mw ^= t
                out[k] = base + j
                k += 1


@njit(nogil=True, cache=True)
def _morsel_body(crumb, lo, hi, buf,
                 d1, p1, b1, c1, eq1, d2, p2, b2, c2, eq2, nflags,
                 emask, ebase, has_em,
                 pl1, nw1, kb1, pl2, nw2, kb2, pl3, nw3, kb3, nkeys,
                 okeys, obase):
    """One cook's COMPLETE pipeline over its crumb slice: kill-ordered
    tests (snowball mask bit, then monotone flag cursors), key bits read
    only for survivors, packed keys appended locally. No barriers inside."""
    B = 4096
    blkA = np.int64(-1); ptrA = np.int64(0); endA = np.int64(-1)
    valA = np.int64(0); bpoA = np.int64(0); bitA = False
    blkB = np.int64(-1); ptrB = np.int64(0); endB = np.int64(-1)
    valB = np.int64(0); bpoB = np.int64(0); bitB = False
    k = obase
    for i in range(lo, hi):
        r = crumb[i]
        if has_em:
            rr = r - ebase
            if ((emask[rr >> 6] >> np.uint64(rr & 63)) & np.uint64(1)) == np.uint64(0):
                continue
        ok = True
        if nflags >= 1:
            bb = r // B
            if bb != blkA:
                blkA = bb
                o = p1 + (d1[bb] >> 1)
                if d1[bb] & 1:
                    bitA = False
                    ptrA = o + 2
                    endA = bb * B
                    valA = -1
                else:
                    bitA = True
                    bpoA = o
            if bitA:
                rr2 = (r - blkA * B) * b1
                v1 = np.int64(0)
                for bi in range(b1):
                    ix = rr2 + bi
                    v1 = (v1 << 1) | ((np.int64(buf[bpoA + (ix >> 3)]) >> (7 - (ix & 7))) & 1)
            else:
                while r >= endA:
                    cnt = np.int64(buf[ptrA]) | (np.int64(buf[ptrA + 1]) << 8)
                    valA = np.int64(buf[ptrA + 2]) | (np.int64(buf[ptrA + 3]) << 8)
                    ptrA += 4
                    endA += cnt
                v1 = valA
            ok = (v1 == c1) if eq1 else (v1 != c1)
        if ok and nflags >= 2:
            bb = r // B
            if bb != blkB:
                blkB = bb
                o = p2 + (d2[bb] >> 1)
                if d2[bb] & 1:
                    bitB = False
                    ptrB = o + 2
                    endB = bb * B
                    valB = -1
                else:
                    bitB = True
                    bpoB = o
            if bitB:
                rr2 = (r - blkB * B) * b2
                v2 = np.int64(0)
                for bi in range(b2):
                    ix = rr2 + bi
                    v2 = (v2 << 1) | ((np.int64(buf[bpoB + (ix >> 3)]) >> (7 - (ix & 7))) & 1)
            else:
                while r >= endB:
                    cnt = np.int64(buf[ptrB]) | (np.int64(buf[ptrB + 1]) << 8)
                    valB = np.int64(buf[ptrB + 2]) | (np.int64(buf[ptrB + 3]) << 8)
                    ptrB += 4
                    endB += cnt
                v2 = valB
            ok = (v2 == c2) if eq2 else (v2 != c2)
        if not ok:
            continue
        if nkeys == 0:
            okeys[k] = r                         # rows mode: emit the row
            k += 1
            continue
        w = r // 64
        j = np.uint64(r % 64)
        key = np.int64(0)
        for p in range(kb1):
            key = (key << 1) | np.int64((pl1[p * nw1 + w] >> j) & np.uint64(1))
        if nkeys >= 2:
            for p in range(kb2):
                key = (key << 1) | np.int64((pl2[p * nw2 + w] >> j) & np.uint64(1))
        if nkeys >= 3:
            for p in range(kb3):
                key = (key << 1) | np.int64((pl3[p * nw3 + w] >> j) & np.uint64(1))
        okeys[k] = key
        k += 1
    return k - obase


@njit(nogil=True, cache=True)
def _local_sort_walk(keys, n, bits, tmp, uk, uc):
    """One cook's private count: serial LSD radix on its survivors, then
    the run walk -- L1 bowls, no contention with any other cook."""
    src = keys
    dst = tmp
    sh = 0
    while sh < bits:
        hist = np.zeros(4096, np.int64)
        for i in range(n):
            hist[(src[i] >> sh) & 0xFFF] += 1
        tot = 0
        for d in range(4096):
            h = hist[d]
            hist[d] = tot
            tot += h
        for i in range(n):
            dg = (src[i] >> sh) & 0xFFF
            dst[hist[dg]] = src[i]
            hist[dg] += 1
        src, dst = dst, src
        sh += 12
    g = 0
    i = 0
    while i < n:
        v = src[i]
        c = 1
        i += 1
        while i < n and src[i] == v:
            c += 1
            i += 1
        uk[g] = v
        uc[g] = c
        g += 1
    return g


@njit(nogil=True, parallel=True, cache=True)
def morsel_group(crumb, buf,
                 d1, p1, b1, c1, eq1, d2, p2, b2, c2, eq2, nflags,
                 emask, ebase, has_em,
                 pl1, nw1, kb1, pl2, nw2, kb2, pl3, nw3, kb3, nkeys,
                 kbits, NT, okeys, tmp, uks, ucs, gcounts):
    """JACKSON'S MORSEL DOCTRINE: partition the crumb ONCE, then every
    thread runs the WHOLE pipeline -- kill-ordered filters, survivor-only
    key reads, private radix count -- with a single barrier at the end.
    The reading-aloud collapses to the combine."""
    n = crumb.size
    per = (n + NT - 1) // NT
    for t in prange(NT):
        lo = t * per
        hi = min(n, lo + per)
        if lo >= hi:
            gcounts[t] = 0
            continue
        m = _morsel_body(crumb, lo, hi, buf,
                         d1, p1, b1, c1, eq1, d2, p2, b2, c2, eq2, nflags,
                         emask, ebase, has_em,
                         pl1, nw1, kb1, pl2, nw2, kb2, pl3, nw3, kb3, nkeys,
                         okeys, lo)
        if nkeys >= 1:
            gcounts[t] = _local_sort_walk(okeys[lo:lo + m], m, kbits,
                                          tmp[lo:lo + m], uks[lo:lo + m],
                                          ucs[lo:lo + m])
        else:
            gcounts[t] = m                       # rows mode: survivors only


@njit(nogil=True, cache=True)
def _swar_u64(pk8, w):
    x = np.uint64(pk8[w]) | (np.uint64(pk8[w + 1]) << np.uint64(8)) \
        | (np.uint64(pk8[w + 2]) << np.uint64(16)) | (np.uint64(pk8[w + 3]) << np.uint64(24)) \
        | (np.uint64(pk8[w + 4]) << np.uint64(32)) | (np.uint64(pk8[w + 5]) << np.uint64(40)) \
        | (np.uint64(pk8[w + 6]) << np.uint64(48)) | (np.uint64(pk8[w + 7]) << np.uint64(56))
    return x


@njit(nogil=True, cache=True)
def _swar_body(pk8, hot, warm, wb, pt, o1, o2, rows, BR, out, i0, i1):
    """JACKSON'S SLICE-LAB LAW: never decode what you can count past.
    Escape nibbles (==15) are SWAR-counted 16 at a time to position the
    warm cursor; warm bytes are scanned lazily for the cold cursor; only
    survivor rows are ever read. 17x over bucket sweeps, measured."""
    NIB = np.uint64(0x1111111111111111)
    curb = np.int64(-1)
    base = np.int64(0)
    esc = np.int64(0)
    scan = np.int64(0)
    wscan = np.int64(0)
    w255 = np.int64(0)
    for i in range(i0, i1):
        r = rows[i]
        b = r // BR
        if b != curb:
            curb = b
            base = b * BR
            esc = 0
            scan = base
            wscan = o1[b]
            w255 = 0
        while scan < r:
            wq = (scan >> 4) << 3
            x = _swar_u64(pk8, wq)
            t = x & (x >> np.uint64(1)) & (x >> np.uint64(2)) & (x >> np.uint64(3)) & NIB
            lo = scan & 15
            hi = r - (scan & ~np.int64(15))
            if hi > 16:
                hi = 16
            if lo > 0 or hi < 16:
                if hi < 16:
                    m = (np.uint64(1) << np.uint64(4 * hi)) - np.uint64(1)
                else:
                    m = ~np.uint64(0)
                m &= ~((np.uint64(1) << np.uint64(4 * lo)) - np.uint64(1))
                t &= m
            x = t
            x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
            x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
            x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
            esc += np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))
            scan = (scan & ~np.int64(15)) + hi
        nib = (pk8[r >> 1] >> (4 * (r & 1))) & 15
        if nib < 15:
            out[i] = np.int64(hot[nib])
        else:
            p1 = o1[curb] + esc
            while wscan < p1:
                if wb[wscan] == 255:
                    w255 += 1
                wscan += 1
            wv = np.int64(wb[p1])
            if wv < 255:
                out[i] = np.int64(warm[wv])
            else:
                out[i] = np.int64(pt[o2[curb] + w255])
        if nib == 15:
            esc += 1
            if wscan == o1[curb] + esc - 1:
                if wb[wscan] == 255:
                    w255 += 1
                wscan += 1
        scan = r + 1


@njit(nogil=True, parallel=True, cache=True)
def enc6_range(packed, hot, warm, wbytes, patches, e1off, e2off,
               blo, bhi, BR, out):
    """THE STREAMING MORSEL's missing machine: decode enc-6 rows [blo,bhi)
    sequentially, bucket by bucket -- escape cursors walk from each
    bucket's start, writes land band-relative."""
    b0 = blo // BR
    b1 = (bhi + BR - 1) // BR
    for bb in prange(b1 - b0):
        b = b0 + bb
        lo = b * BR
        hi = lo + BR
        if hi > bhi:
            hi = bhi
        p1 = e1off[b]
        p2 = e2off[b]
        for i in range(lo, hi):
            byte = packed[i >> 1]
            if (i & 1) == 0:
                v = np.int64(byte & 0x0F)
            else:
                v = np.int64(byte >> 4)
            if v < 15:
                x = hot[v]
            else:
                w = np.int64(wbytes[p1])
                p1 += 1
                if w < 255:
                    x = warm[w]
                else:
                    x = patches[p2]
                    p2 += 1
            if i >= blo:
                out[i - blo] = x


@njit(nogil=True, parallel=True, cache=True)
def enc5_range(packed, hot, patches, eoff, blo, bhi, BR, out):
    """enc-5's band decode: hot nibble -> u16 patch, one escape tier."""
    b0 = blo // BR
    b1 = (bhi + BR - 1) // BR
    for bb in prange(b1 - b0):
        b = b0 + bb
        lo = b * BR
        hi = lo + BR
        if hi > bhi:
            hi = bhi
        p2 = eoff[b]
        for i in range(lo, hi):
            byte = packed[i >> 1]
            if (i & 1) == 0:
                v = np.int64(byte & 0x0F)
            else:
                v = np.int64(byte >> 4)
            if v < 15:
                x = hot[v]
            else:
                x = patches[p2]
                p2 += 1
            if i >= blo:
                out[i - blo] = x


@njit(nogil=True, parallel=True, cache=True)
def enc6_swar_pick(pk8, hot, warm, wb, pt, o1, o2, rows, BR, out):
    """The swar pick, morselized: survivor chunks in prange, each chunk
    re-deriving its cursor state from its first bucket -- Jackson's
    partition-once doctrine applied to the pick itself."""
    NT = 32
    n = rows.size
    per = (n + NT - 1) // NT
    for t in prange(NT):
        i0 = t * per
        i1 = min(n, i0 + per)
        if i0 < i1:
            _swar_body(pk8, hot, warm, wb, pt, o1, o2, rows, BR, out, i0, i1)


@njit(nogil=True, parallel=True, cache=True)
def e8_unpack(buf, base, n, bits, out):
    """Sparse-default literal stream -> codes, MSB-packed, 16-wide."""
    for i in prange(n):
        bo = i * bits
        by = base + (bo >> 3)
        sh = bo & 7
        acc = np.int64(0)
        for k in range(4):
            acc = (acc << np.int64(8)) | np.int64(buf[by + k])
        out[i] = (acc >> np.int64(32 - bits - sh)) \
            & ((np.int64(1) << np.int64(bits)) - 1)


@njit(nogil=True, parallel=True, cache=True)
def pres_pair_count(pres, ck, codes, eng, cand, ncand, ne, bowls):
    """THE MARGINAL-BOUND LAW's walk: stride the presence bitmap by its
    64K checkpoints (each chunk knows its starting rank), and for every
    present row whose code is a candidate, count (candidate, partner)
    into a cache-resident table. One pass, sixteen-wide, no expansion
    of anything that cannot win."""
    nch = ck.size
    for c in prange(nch):
        r0 = c << 16
        rank = ck[c]
        b0 = r0 >> 3
        b1 = min(pres.size, b0 + 8192)
        for bi in range(b0, b1):
            pv = pres[bi]
            if pv == 0:
                continue
            row = bi << 3
            for j in range(8):
                if (pv >> (7 - j)) & 1:
                    code = codes[rank]
                    ci = cand[code]
                    if ci < ncand:
                        bowls[c % 16, ci * ne + eng[row + j]] += 1
                    rank += 1


@njit(nogil=True, parallel=True, cache=True)
def codes_test_mask(v, c1, c2, mode, fm):
    """Band codes -> pass-mask words, sixteen-wide: each thread reads 64
    codes and emits one word. mode 0: ==c1 | ==c2 (IN pair / eq when
    c2==c1); mode 1: != c1. Replaces the serial isin+packbits tail."""
    nw = fm.size
    n = v.size
    for w in prange(nw):
        base = w * 64
        hi = base + 64
        if hi > n:
            hi = n
        m = np.uint64(0)
        if mode == 0:
            for j in range(base, hi):
                x = v[j]
                if x == c1 or x == c2:
                    m |= np.uint64(1) << np.uint64(j - base)
        else:
            for j in range(base, hi):
                if v[j] != c1:
                    m |= np.uint64(1) << np.uint64(j - base)
        fm[w] = m


@njit(nogil=True, parallel=True, cache=True)
def band_pick(band, mask, ob, out, CHW):
    """Compact a band-relative value array by the surviving mask --
    sequential reads of freshly written cache-warm bytes."""
    nch = ob.size
    for cix in prange(nch):
        a0 = cix * CHW
        a1 = min(mask.size, a0 + CHW)
        k = ob[cix]
        for w in range(a0, a1):
            mw = mask[w]
            base = w * 64
            while mw != np.uint64(0):
                t = mw & ((~mw) + np.uint64(1))
                x = t - np.uint64(1)
                x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
                x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
                x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
                j = int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))
                mw ^= t
                out[k] = np.int64(band[base + j])
                k += 1


def fused_band_group(buf, fl, keys, crumb, blo, bhi, emask=None):
    """THE STREAMING MORSEL: every stage a sequential band pass, zero
    per-row random touches. fl: enc-10 flags (mask surgery). emask: an
    optional pre-built snowball match mask (enc-12 equality), ANDed in.
    keys: ('p', planes, nwords, bits) or ('e6', packed, hot, warm,
    wbytes, patches, o1, o2, BR, bits). Returns (packed keys, counts,
    widths)."""
    w0 = blo // 64
    w1 = (bhi + 63) // 64
    mask = np.zeros(w1 - w0, np.uint64)
    vbits_set(np.ascontiguousarray(crumb) - np.int64(w0 * 64), mask)
    for (d, p, b, c, eq) in fl:
        fm = np.zeros(w1 - w0, np.uint64)
        if b == 1 and c in (0, 1):
            want_one = (c == 1) if eq else (c == 0)
            flag1_pass(buf, d, p, w0 * 64, bhi, want_one, _REV8,
                       fm.view(np.uint8))
        else:
            _flag_pass(buf, d, p, b, c, eq, w0 * 64, bhi, fm)
        np.bitwise_and(mask, fm, out=mask)
    if emask is not None:
        np.bitwise_and(mask, emask, out=mask)
    CHW = 1024
    nch = (mask.size + CHW - 1) // CHW
    ob = np.zeros(nch, np.int64)
    tot = _offsets_from_mask(mask, CHW, ob)
    widths = [k[-1] for k in keys]
    if tot == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), widths
    comp = np.zeros(tot, np.int64)
    tmp = np.empty(tot, np.int64)
    band6 = None
    rows2 = None
    accb = 0
    for k in keys:
        if k[0] == 'p':
            _, pl, nw, b = k
            vp_gather_band(pl, nw, b, mask, w0, w1, ob, tmp)
        elif k[0] == 'e6':
            _, pk, hot, warm, wb, pt, o1, o2, BR, b = k
            if rows2 is None:
                rows2 = np.empty(tot, np.int64)
                mask_to_rows(mask, w0, ob, rows2, CHW)
            enc6_swar_pick(pk, hot, warm, wb, pt, o1, o2, rows2,
                           np.int64(BR), tmp)
        else:
            _, pk, hot, pt, o1, BR, b = k
            if rows2 is None:
                rows2 = np.empty(tot, np.int64)
                mask_to_rows(mask, w0, ob, rows2, CHW)
            v5 = enc5_at2(pk, hot, pt, o1, rows2, np.int64(bhi), np.int64(BR))
            tmp[:] = v5.astype(np.int64)
        np.left_shift(comp, b, out=comp)
        np.bitwise_or(comp, tmp, out=comp)
        accb += b
    t_k = np.empty(tot, np.int64)
    ks = _radix_inplace(comp, t_k, accb, 1 << 16)
    bnd = np.flatnonzero(np.diff(ks) != 0)
    st = np.concatenate([[0], bnd + 1])
    uk = ks[st]
    uc = np.diff(np.concatenate([st, [ks.size]])).astype(np.int64)
    return uk, uc, widths


@njit(nogil=True, parallel=True, cache=True)
def vp_gather_band(pl, nwords, bits, mask, w0, w1, obase, out):
    """JACKSON'S LOCKSTEP GATHER: walk the band's word positions once,
    all planes advancing shoulder-to-shoulder, transpose per word, emit
    ONLY the masked rows' codes compacted -- offsets pre-paid by mask
    popcounts, so every chunk writes its own slice with no contention.
    Every plane word is read exactly once, sequentially."""
    sh = np.uint64(64 - bits)
    CHW = 1024
    nch = (w1 - w0 + CHW - 1) // CHW
    for cix in prange(nch):
        a0 = w0 + cix * CHW
        a1 = min(w1, a0 + CHW)
        a = np.zeros(64, np.uint64)
        k = obase[cix]
        for w in range(a0, a1):
            mw = mask[w - w0]
            if mw == np.uint64(0):
                continue
            for p in range(bits):
                a[p] = pl[p * nwords + w]
            for p in range(bits, 64):
                a[p] = np.uint64(0)
            _t64(a)
            while mw != np.uint64(0):
                t = mw & ((~mw) + np.uint64(1))
                x = t - np.uint64(1)
                x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
                x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
                x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
                j = int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))
                mw ^= t
                out[k] = np.int64(a[63 - j] >> sh)
                k += 1


@njit(nogil=True, parallel=True, cache=True)
def vp_gather_span(pl, nwords, p0, p1, rows, out):
    """Gather only planes [p0,p1) at the rows -- HALF A KEY IS FREE
    vertically. Jackson's prefix-group law: read bits in the order
    that prunes."""
    n = rows.size
    for i in prange(n):
        r = rows[i]
        w = r // 64
        j = np.uint64(r % 64)
        v = np.int64(0)
        for p in range(p0, p1):
            v = (v << 1) | np.int64((pl[p * nwords + w] >> j) & np.uint64(1))
        out[i] = np.uint16(v)


@njit(nogil=True, parallel=True, cache=True)
def vp_gather(pl, nwords, bits, rows, out):
    """Scattered gather from planes -- measured FASTER than horizontal at
    DRAM scale (independent plane streams pipeline)."""
    n = rows.size
    for i in prange(n):
        r = rows[i]
        w = r // 64
        j = np.uint64(r % 64)
        v = np.int64(0)
        for p in range(bits):
            v = (v << 1) | np.int64((pl[p * nwords + w] >> j) & np.uint64(1))
        out[i] = v


@njit(nogil=True, cache=True)
def _vp_eq_block(pl, nwords, w0, w1, bits, target):
    """One block of the snowball: word-granularity active list."""
    sz = w1 - w0
    m = np.full(sz, ~np.uint64(0), np.uint64)
    alive = np.empty(sz, np.int64)
    for w in range(sz):
        alive[w] = w
    na = sz
    for p in range(bits):
        tbit = (target >> (bits - 1 - p)) & 1
        k = 0
        if tbit:
            for ai in range(na):
                w = alive[ai]
                m[w] &= pl[p * nwords + w0 + w]
                if m[w] != np.uint64(0):
                    alive[k] = w
                    k += 1
        else:
            for ai in range(na):
                w = alive[ai]
                m[w] &= ~pl[p * nwords + w0 + w]
                if m[w] != np.uint64(0):
                    alive[k] = w
                    k += 1
        na = k
        if na == 0:
            return 0
    c = 0
    for ai in range(na):
        x = m[alive[ai]]
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        c += np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))
    return c


@njit(nogil=True, parallel=True, cache=True)
def vp_scan_eq_mask(pl, nwords, bits, target, w0, w1, mask):
    """The snowball EMITTING: per-word match bitmasks for rows == target
    over word window [w0,w1) -- Jackson's progressive pruning, but the
    survivors land as a bitmap ready to AND into any crumb."""
    BLK = 4096
    nblk = (w1 - w0 + BLK - 1) // BLK
    for b in prange(nblk):
        a0 = w0 + b * BLK
        a1 = min(w1, a0 + BLK)
        for w in range(a0, a1):
            m = ~np.uint64(0)
            for p in range(bits):
                tbit = (target >> (bits - 1 - p)) & 1
                pw = pl[p * nwords + w]
                m &= pw if tbit else ~pw
                if m == np.uint64(0):
                    break
            mask[w - w0] = m


@njit(nogil=True, parallel=True, cache=True)
def vp_scan_eq(pl, nwords, bits, target, counts, BLK):
    """COUNT rows == target: Jackson's progressive pruning at word
    granularity -- runs data touches ~25-30% of bytes."""
    nblk = counts.size
    for b in prange(nblk):
        w0 = b * BLK
        w1 = min(nwords, w0 + BLK)
        counts[b] = _vp_eq_block(pl, nwords, w0, w1, bits, target)


@njit(nogil=True, parallel=True, cache=True)
def vp_scan_flag_count(pl, nwords, bits, lo, hi, flag, counts, CH):
    """Membership scan pass 1 over planes: transpose 64 rows, test flag,
    count per chunk (CH multiple of 64)."""
    nch = counts.size
    sh = np.uint64(64 - bits)
    for cix in prange(nch):
        a0 = lo + cix * CH
        b0 = min(hi, a0 + CH)
        a = np.zeros(64, np.uint64)
        cnt = 0
        w = a0 // 64
        wend = (b0 + 63) // 64
        while w < wend:
            for p in range(bits):
                a[p] = pl[p * nwords + w]
            for p in range(bits, 64):
                a[p] = np.uint64(0)
            _t64(a)
            r0 = w * 64
            for j in range(64):
                r = r0 + j
                if r >= a0 and r < b0:
                    if flag[np.int64(a[63 - j] >> sh)]:
                        cnt += 1
            w += 1
        counts[cix] = cnt


@njit(nogil=True, parallel=True, cache=True)
def vp_scan_flag_fill(pl, nwords, bits, lo, hi, flag, offs, out, CH):
    """Membership scan pass 2: same walk, positions into slices."""
    nch = offs.size - 1
    sh = np.uint64(64 - bits)
    for cix in prange(nch):
        a0 = lo + cix * CH
        b0 = min(hi, a0 + CH)
        a = np.zeros(64, np.uint64)
        w9 = offs[cix]
        w = a0 // 64
        wend = (b0 + 63) // 64
        while w < wend:
            for p in range(bits):
                a[p] = pl[p * nwords + w]
            for p in range(bits, 64):
                a[p] = np.uint64(0)
            _t64(a)
            r0 = w * 64
            for j in range(64):
                r = r0 + j
                if r >= a0 and r < b0:
                    if flag[np.int64(a[63 - j] >> sh)]:
                        out[w9] = r
                        w9 += 1
            w += 1


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
def bp10_range(buf, d, p, b, blo, bhi, out):
    """THE BAND-DECODE LAW's enc-10 half: expand run-lists / bitpack for
    every row in [blo,bhi) -- streaming beats per-row probes when the
    crumb is dense inside its band."""
    B = 4096
    k0 = blo // B
    k1 = (bhi + B - 1) // B
    for kb in prange(k1 - k0):
        blk = k0 + kb
        o = p + (d[blk] >> 1)
        base = blk * B
        if d[blk] & 1:
            nr = np.int64(buf[o]) | (np.int64(buf[o + 1]) << 8)
            ptr = o + 2
            r = base
            for _ in range(nr):
                cnt = np.int64(buf[ptr]) | (np.int64(buf[ptr + 1]) << 8)
                val = np.int64(buf[ptr + 2]) | (np.int64(buf[ptr + 3]) << 8)
                ptr += 4
                e = r + cnt
                a0 = r if r > blo else blo
                a1 = e if e < bhi else bhi
                for rr in range(a0, a1):
                    out[rr - blo] = np.int16(val)
                r = e
                if r >= bhi:
                    break
        else:
            a0 = base if base > blo else blo
            a1 = base + B if base + B < bhi else bhi
            for rr in range(a0, a1):
                idx = (rr - base) * b
                v = np.int64(0)
                for bi in range(b):
                    ix = idx + bi
                    v = (v << 1) | ((np.int64(buf[o + (ix >> 3)]) >> (7 - (ix & 7))) & 1)
                out[rr - blo] = np.int16(v)


@njit(nogil=True, cache=True)
def _h2_chunk(buf, d1, p1, b1, c1, eq1, d2, p2, b2, c2, eq2, rows, keep, a0, a1):
    """Jackson's monotone cursor: rows ascend, so each flag column's run
    walk only ever moves FORWARD -- O(rows + runs), no per-row restart."""
    B = 4096
    blkA = np.int64(-1); nrA = 0; ptrA = np.int64(0); endA = np.int64(-1)
    valA = np.int64(0); bpoA = np.int64(0); bitA = False
    blkB = np.int64(-1); nrB = 0; ptrB = np.int64(0); endB = np.int64(-1)
    valB = np.int64(0); bpoB = np.int64(0); bitB = False
    for i in range(a0, a1):
        r = rows[i]
        # --- column A ---
        bb = r // B
        if bb != blkA:
            blkA = bb
            o = p1 + (d1[bb] >> 1)
            if d1[bb] & 1:
                bitA = False
                nrA = np.int64(buf[o]) | (np.int64(buf[o + 1]) << 8)
                ptrA = o + 2
                endA = bb * B
                valA = -1
            else:
                bitA = True
                bpoA = o
        if bitA:
            rr = (r - blkA * B) * b1
            v1 = np.int64(0)
            for bi in range(b1):
                idx = rr + bi
                v1 = (v1 << 1) | ((np.int64(buf[bpoA + (idx >> 3)]) >> (7 - (idx & 7))) & 1)
        else:
            while r >= endA:
                cntr = np.int64(buf[ptrA]) | (np.int64(buf[ptrA + 1]) << 8)
                valA = np.int64(buf[ptrA + 2]) | (np.int64(buf[ptrA + 3]) << 8)
                ptrA += 4
                endA += cntr
            v1 = valA
        ok = (v1 == c1) if eq1 else (v1 != c1)
        if ok:
            # --- column B ---
            if bb != blkB:
                blkB = bb
                o = p2 + (d2[bb] >> 1)
                if d2[bb] & 1:
                    bitB = False
                    nrB = np.int64(buf[o]) | (np.int64(buf[o + 1]) << 8)
                    ptrB = o + 2
                    endB = bb * B
                    valB = -1
                else:
                    bitB = True
                    bpoB = o
            if bitB:
                rr = (r - blkB * B) * b2
                v2 = np.int64(0)
                for bi in range(b2):
                    idx = rr + bi
                    v2 = (v2 << 1) | ((np.int64(buf[bpoB + (idx >> 3)]) >> (7 - (idx & 7))) & 1)
            else:
                while r >= endB:
                    cntr = np.int64(buf[ptrB]) | (np.int64(buf[ptrB + 1]) << 8)
                    valB = np.int64(buf[ptrB + 2]) | (np.int64(buf[ptrB + 3]) << 8)
                    ptrB += 4
                    endB += cntr
                v2 = valB
            ok = (v2 == c2) if eq2 else (v2 != c2)
        keep[i] = ok


@njit(nogil=True, parallel=True, cache=True)
def bp10_hygiene2(buf, d1, p1, b1, c1, eq1, d2, p2, b2, c2, eq2, rows, keep):
    """Two enc-10 flag tests, one walk, short-circuit, MONOTONE CURSORS --
    prange over row chunks, each chunk's cursors seek once then only
    advance. A skipped col-B block never even loads its directory."""
    n = rows.size
    CH = 1 << 15
    nch = (n + CH - 1) // CH
    for cix in prange(nch):
        a0 = cix * CH
        a1 = min(n, a0 + CH)
        _h2_chunk(buf, d1, p1, b1, c1, eq1, d2, p2, b2, c2, eq2, rows, keep, a0, a1)


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


@njit(nogil=True, cache=True)
def gather_frame(raw, rows, j0, out):
    """out[i] = raw[rows[i] - j0]: one frame's rows picked from its inflated codes (nogil: the lanes of a
    sorted gather run it side by side, each into its own slice of the output)"""
    for i in range(rows.size):
        out[i] = raw[rows[i] - j0]


@njit(nogil=True, parallel=True, cache=True)
def bp10_counts(buf, dirX, pay, bits, N, L, T):
    """THE CENSUS FROM THE DRESS, enc 10 (2026-10-01): per-code row counts straight from the
    blocks -- a run block adds each run's length to its value, a bitpack block counts its codes
    (bp10_decode's own walk, counting instead of writing). No 100M-row array. Length L."""
    B = 4096
    nblk = dirX.size
    part = np.zeros((T, L), np.int64)
    for t in prange(T):
        b0 = nblk * t // T
        b1 = nblk * (t + 1) // T
        for b in range(b0, b1):
            lo = b * B
            rows = min(B, N - lo)
            o = pay + (dirX[b] >> 1)
            if dirX[b] & 1:
                nr = np.int64(buf[o]) | (np.int64(buf[o + 1]) << 8)
                p = o + 2
                for r in range(nr):
                    cnt = np.int64(buf[p]) | (np.int64(buf[p + 1]) << 8)
                    val = np.int64(buf[p + 2]) | (np.int64(buf[p + 3]) << 8)
                    p += 4
                    part[t, val] += cnt
            else:
                for r in range(rows):
                    v = np.int64(0)
                    base = r * bits
                    for bi in range(bits):
                        idx = base + bi
                        byte = buf[o + (idx >> 3)]
                        v = (v << 1) | ((np.int64(byte) >> (7 - (idx & 7))) & 1)
                    part[t, v] += 1
    out = np.zeros(L, np.int64)
    for k in prange(L):
        s = np.int64(0)
        for t in range(T):
            s += part[t, k]
        out[k] = s
    return out


@njit(nogil=True, parallel=True, cache=True)
def enc5_counts(packed, hot, patches, N, L, T):
    """THE CENSUS FROM THE DRESS, enc 5 (2026-10-01): per-code row counts from the patched buckets
    without the 100M-row decode -- a 16-bin census of the nibbles (a hot pointer k counts for
    hot[k]; nibble 15 is an escape) plus a census of the escape patches, which are exactly the
    escaped rows' codes. The pad nibble of an odd row count is not a row. Length L."""
    nb = N >> 1                                    # bytes whose two nibbles are both rows
    part = np.zeros((T, 16), np.int64)
    for t in prange(T):
        a = nb * t // T
        b = nb * (t + 1) // T
        for i in range(a, b):
            x = np.int64(packed[i])
            part[t, x & 15] += 1
            part[t, x >> 4] += 1
    nib = np.zeros(16, np.int64)
    for t in range(T):
        for k in range(16):
            nib[k] += part[t, k]
    if N & 1:
        nib[np.int64(packed[nb]) & 15] += 1
    np_ = patches.size
    pp = np.zeros((T, L), np.int64)
    for t in prange(T):
        a = np_ * t // T
        b = np_ * (t + 1) // T
        for j in range(a, b):
            pp[t, np.int64(patches[j])] += 1
    out = np.zeros(L, np.int64)
    for k in prange(L):
        s = np.int64(0)
        for t in range(T):
            s += pp[t, k]
        out[k] = s
    for k in range(min(15, hot.size)):
        out[np.int64(hot[k])] += nib[k]
    return out


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
        _vp = np.zeros(2 * 1, np.uint64); _vp[0] = np.uint64(3)
        _vo = np.zeros(4, np.uint64)
        vp_window(_vp, 1, 2, 0, 4, _vo)
        _vg = np.zeros(2, np.int64)
        vp_gather(_vp, 1, 2, np.array([0, 1], np.int64), _vg)
        _vs = np.zeros(2, np.uint16)
        vp_gather_span(_vp, 1, 0, 1, np.array([0, 1], np.int64), _vs)
        _mb = np.zeros(1, np.uint64)
        vbits_set(np.array([0, 2], np.int64), _mb)
        _ob = np.zeros(1, np.int64)
        _og = np.zeros(2, np.int64)
        vp_gather_band(_vp, 1, 2, _mb, 0, 1, _ob, _og)
        _flag_mask(_bw, np.array([1], np.int64), 0, 1, 0, True, 0, 3, _mb)
        _offsets_from_mask(_mb, 1024, np.zeros(1, np.int64))
        _radix_inplace(np.array([3, 1, 2], np.int64), np.empty(3, np.int64), 12, 65536)
        _rk = np.array([5, 1, 3, 1], np.int64)
        _rh = np.zeros((1, 4096), np.int64)
        radix_hist12(_rk, 0, _rh, 65536)
        _ro = np.zeros((1, 4096), np.int64)
        np.cumsum(_rh[0][:-1], out=_ro[0][1:])
        _tk = np.empty(4, np.int64)
        radix_scatter12(_rk, 0, _ro, _tk, 65536)
        _ro2 = np.zeros((1, 4096), np.int64)
        np.cumsum(_rh[0][:-1], out=_ro2[0][1:])
        _tp = np.zeros(4, np.int64)
        radix_scatter12p(_rk, np.array([9, 8, 7, 6], np.int64), 0, _ro2, _tk, _tp, 65536)
        _vc = np.zeros(1, np.int64)
        vp_scan_eq(_vp, 1, 2, 1, _vc, 4096)
        _fl2 = np.zeros(4, np.bool_); _fl2[3] = True
        vp_scan_flag_count(_vp, 1, 2, 0, 4, _fl2, _vc, 64)
        _po2 = np.zeros(max(1, int(_vc[0])), np.int64)
        vp_scan_flag_fill(_vp, 1, 2, 0, 4, _fl2, np.array([0, int(_vc[0])], np.int64), _po2, 64)
        _mc = np.array([0, 1, 2, 3], np.int64)
        _mo = np.empty(4, np.int64); _mt = np.empty(4, np.int64)
        _mu = np.empty(4, np.int64); _mv = np.empty(4, np.int64)
        _mg = np.zeros(2, np.int64)
        _sp = np.zeros(1, np.int64)
        _cm = np.zeros(1, np.uint64)
        codes_test_mask(np.array([1, 2, 3], np.int64), 2, 2, 0, _cm)
        _e8 = np.zeros(2, np.int64)
        e8_unpack(_bw, 0, 2, 3, _e8)
        _bl = np.zeros((16, 4), np.int64)
        pres_pair_count(np.array([255], np.uint8), np.array([0], np.int64),
                        np.zeros(8, np.int64), np.zeros(8, np.uint8),
                        np.zeros(8, np.uint8), 2, 2, _bl)
        enc6_swar_pick(np.zeros(16, np.uint8), np.zeros(15, np.int64),
                       np.zeros(255, np.int64), np.zeros(1, np.uint8),
                       np.zeros(1, np.int64), np.zeros(2, np.int64),
                       np.zeros(2, np.int64), np.array([3], np.int64),
                       np.int64(32), _sp)
        morsel_group(_mc, _bw,
                     np.array([1], np.int64), 0, 1, 0, True,
                     np.array([1], np.int64), 0, 1, 0, True, 0,
                     np.zeros(1, np.uint64), 0, False,
                     _vp, 1, 2, _vp, 1, 2, _vp, 1, 2, 1,
                     2, 2, _mo, _mt, _mu, _mv, _mg)
        _sm = np.zeros(1, np.uint64)
        vp_scan_eq_mask(_vp, 1, 2, 1, 0, 1, _sm)
        _br = np.zeros(3, np.int16)
        bp10_range(_bw, np.array([1], np.int64), 0, 1, 0, 3, _br)
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


def grouped_sum_codes(kc, vc, vt, K):
    return _grouped_sum_codes_nb(kc, vc, vt, K, _nt())


@njit(nogil=True, parallel=True, cache=True)
def _grouped_sum_codes_nb(kc, vc, vt, K, T):
    """sums[k] += vt[vc[i]] for k = kc[i]: the single-key SUM board, fused gather+
    accumulate, per-thread partials (no atomics, no dtype casts, no factorize)."""
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


def gd_pass2_count(ku, kr, offs, SH, VR):
    return _gd_pass2_count_nb(ku, kr, offs, SH, VR, _nt())


@njit(nogil=True, parallel=True, cache=True)
def _gd_pass2_count_nb(ku, kr, offs, SH, VR, T):
    """MSD pass 2 + marker dedup, fused per bucket: scatter by the low bits (each
    bucket's targets fully contained), then the L1-resident marker table counts a
    target's first touch of each key. Per-thread accumulator rows: no races. Thread t takes
    the buckets [NB*t/T, NB*(t+1)/T) -- prange's own static split, with the row index known
    without asking numba for it (get_thread_id made the kernel uncacheable)."""
    NB = offs.size - 1
    LOW = 1 << SH
    ans = np.zeros((T, VR), np.int64)            # per-THREAD rows (was per-bucket: 4096 x VR, 296 MB
    for t in prange(T):                           # zeroed and reduced serially for RegionID's 9,040)
        for b in range(NB * t // T, NB * (t + 1) // T):
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
                        ans[t, r] += 1
    total = np.zeros(VR, np.int64)
    for r in prange(VR):
        s = 0
        for t in range(T):
            s += ans[t, r]
        total[r] = s
    return total


def grid2_count(c1, c2, fc, lit, V2, K):
    return _grid2_count_nb(c1, c2, fc, lit, V2, K, _nt())


@njit(nogil=True, parallel=True, cache=True)
def _grid2_count_nb(c1, c2, fc, lit, V2, K, T):
    """Composite 2-key COUNT grid with optional eq-filter (lit<0 = unfiltered):
    per-thread boards, one fused pass, no factorize, no sorts."""
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


@njit(inline='always')
def _pop8(v):
    v = v - ((v >> 1) & 0x55)
    v = (v & 0x33) + ((v >> 2) & 0x33)
    return (v + (v >> 4)) & 0x0F


@njit(nogil=True, cache=True)
def e8_rank_at(pres, ck, rows, rank, present):
    """THE POINT READ BY RANK (2026-10-01), tags 8 and 9: for SORTED rows, each row's rank among the
    present rows (its literal's index) and whether it is present -- the 64K checkpoint, then the set
    presence bits (MSB-first) from the block start up to the row, counted on from the previous row.
    Reads only the presence bytes up to the rows asked, never the whole column."""
    cur = np.int64(-1); cnt = np.int64(0); byte = np.int64(0)
    for i in range(rows.size):
        r = rows[i]
        b = r >> 16
        if b != cur:
            cur = b; cnt = np.int64(ck[b]); byte = b << 13
        tb = r >> 3
        while byte < tb:
            cnt += _pop8(np.int64(pres[byte])); byte += 1
        v = np.int64(pres[tb]); k = r & 7
        rank[i] = cnt + (_pop8(v >> (8 - k)) if k else 0)
        present[i] = (v >> (7 - k)) & 1


@njit(nogil=True, cache=True)
def e8_lits_at(lane, bits, idx, out):
    """MSB-first literals of the given width at the given indices (any order): five bytes per read, as
    unpack_any, the lane's end guarded."""
    mask = np.int64((1 << bits) - 1); L = lane.size
    for i in range(idx.size):
        o = idx[i] * bits; j = o >> 3; sh = o & 7
        acc = np.int64(0)
        for kk in range(5):
            acc = (acc << 8) | (np.int64(lane[j + kk]) if j + kk < L else np.int64(0))
        out[i] = (acc >> (40 - sh - bits)) & mask


@njit(nogil=True, cache=True)
def e9_tier_at(tb, idx, hit, nxt):
    """One tier of tag 9 for SORTED literal indices: hit[i] = the tier's bit at idx[i]; nxt[i] = the index
    among the literals this tier leaves = idx[i] - the tier's set bits before idx[i], counted on."""
    cnt = np.int64(0); byte = np.int64(0)
    for i in range(idx.size):
        a = idx[i]; t = a >> 3
        while byte < t:
            cnt += _pop8(np.int64(tb[byte])); byte += 1
        v = np.int64(tb[t]); k = a & 7
        hit[i] = (v >> (7 - k)) & 1
        nxt[i] = a - (cnt + (_pop8(v >> (8 - k)) if k else 0))


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
    cum = np.zeros(131072, np.int64)             # cum[j] = continuation bytes in prev[:j]; a string is at most 65535 + 65535 bytes (u16 cp + u16 sl)
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
    prev = np.zeros(131072, np.uint8)   # the longest string the format can hold (u16 cp + u16 sl)
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
            while j2 < plen and j2 < 131072:
                if prev[j2] == 47:
                    he = j2
                    break
                j2 += 1
            if he <= hs and hs != hs0:
                hs = hs0
                j2 = hs
                he = np.int64(-1)
                while j2 < plen and j2 < 131072:
                    if prev[j2] == 47:
                        he = j2
                        break
                    j2 += 1
        # newline law at the break: first \n strictly after he
        bnl = np.int64(-1)
        if he > hs:
            j3 = he + 1
            while j3 < plen and j3 < 131072:
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
            b0 = plen                    # no match: REGEXP_REPLACE returns the whole string
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


@njit(cache=True, parallel=True)
def e14_reconstruct(Y, M, D, ybase, inv, dmin, out):
    """FIELD-PLANE reconstruction: y/m/d u8 planes -> epoch days (Hinnant) ->
    dict codes via the inverse LUT. One parallel pass, no temporaries."""
    n = Y.shape[0]
    for i in prange(n):
        mm = np.int64(M[i])
        y = np.int64(ybase) + np.int64(Y[i])
        if mm <= 1:
            y -= 1
        if y >= 0:
            era = y // 400
        else:
            era = (y - 399) // 400
        yoe = y - era * 400
        mp = (mm + 10) % 12
        doe = yoe * 365 + yoe // 4 - yoe // 100 + (153 * mp + 2) // 5 + np.int64(D[i])
        days = era * 146097 + doe - 719468
        out[i] = inv[days - dmin]


@njit(cache=True, parallel=True)
def e14_band_test(Y, M, D, ybase, ylo, mlo, dlo, yhi, mhi, dhi, out):
    """FIELD-PLANE band test (the plane-test read): keep rows whose calendar
    tuple sits in [ (ylo,mlo,dlo), (yhi,mhi,dhi) ) -- three u8 compares per
    row, NO reconstruction, no dictionary, no gather."""
    n = Y.shape[0]
    for i in prange(n):
        y = np.int64(ybase) + np.int64(Y[i]); m = np.int64(M[i]); d = np.int64(D[i])
        ge = (y > ylo) or (y == ylo and (m > mlo or (m == mlo and d >= dlo)))
        lt = (y < yhi) or (y == yhi and (m < mhi or (m == mhi and d < dhi)))
        out[i] = ge and lt


@njit(cache=True, parallel=True)
def e14_year_band(Y, ybase, ylo, yhi, out):
    """Year-aligned calendar band: the y-plane alone decides."""
    lo = ylo - ybase; hi = yhi - ybase
    n = Y.shape[0]
    for i in prange(n):
        v = np.int64(Y[i])
        out[i] = (v >= lo) and (v < hi)


@njit(cache=True, nogil=True)
def e14_reconstruct_chunk(Y, M, D, ybase, inv, dmin, out):
    """FIELD-PLANE reconstruction, ONE CHUNK, nogil: called from t frame
    workers so the whole decode runs with t sets of eyes end-to-end --
    decompress-then-reconstruct per chunk, cache-hot, no barrier, no
    materialised planes (Jackson's shape)."""
    n = Y.shape[0]
    for i in range(n):
        mm = np.int64(M[i])
        y = np.int64(ybase) + np.int64(Y[i])
        if mm <= 1:
            y -= 1
        if y >= 0:
            era = y // 400
        else:
            era = (y - 399) // 400
        yoe = y - era * 400
        mp = (mm + 10) % 12
        doe = yoe * 365 + yoe // 4 - yoe // 100 + (153 * mp + 2) // 5 + np.int64(D[i])
        days = era * 146097 + doe - 719468
        out[i] = inv[days - dmin]


@njit(cache=True, nogil=True)
def e15_reconstruct_chunk(Y, M, D, DL, BB, role, ybase, inv, dmin, out):
    """CLOCK-DRESS reconstruction (Jackson's dial), one chunk, nogil:
    anchor y/m/d planes -> civil days; the delta arm swings by the
    orientation bit; role says which column this is (1: the min when
    bit=1). inv-LUT gather lands this column's own dict codes."""
    n = Y.shape[0]
    for i in range(n):
        mm = np.int64(M[i])
        y = np.int64(ybase) + np.int64(Y[i])
        if mm <= 1:
            y -= 1
        if y >= 0:
            era = y // 400
        else:
            era = (y - 399) // 400
        yoe = y - era * 400
        mp = (mm + 10) % 12
        doe = yoe * 365 + yoe // 4 - yoe // 100 + (153 * mp + 2) // 5 + np.int64(D[i])
        days = era * 146097 + doe - 719468
        b = (BB[i >> 3] >> (7 - (i & 7))) & 1
        if b != role:
            days += np.int64(DL[i])
        out[i] = inv[days - dmin]


@njit(cache=True, nogil=True)
def e15_band_chunk(Y, M, D, DL, BB, role, ybase, dlo, dhi, out):
    """CLOCK-DRESS band test, one chunk, nogil: anchor civil days + the
    delta arm (when the bit disagrees with this column's role), then a
    plain numeric [dlo, dhi) test. No dict, no gather."""
    n = Y.shape[0]
    for i in range(n):
        mm = np.int64(M[i])
        y = np.int64(ybase) + np.int64(Y[i])
        if mm <= 1:
            y -= 1
        if y >= 0:
            era = y // 400
        else:
            era = (y - 399) // 400
        yoe = y - era * 400
        mp = (mm + 10) % 12
        doe = yoe * 365 + yoe // 4 - yoe // 100 + (153 * mp + 2) // 5 + np.int64(D[i])
        days = era * 146097 + doe - 719468
        b = (BB[i >> 3] >> (7 - (i & 7))) & 1
        if b != role:
            days += np.int64(DL[i])
        out[i] = (days >= dlo) and (days < dhi)


@njit(cache=True, parallel=True, nogil=True)
def exists_scatter(m, ptr, okeep, yes):
    """Jackson's three-scans glue, fused: child verdict AND parent gate AND
    scatter in ONE pass. The racing bool store is idempotent."""
    n = m.shape[0]
    for i in prange(n):
        if m[i]:
            p = np.int64(ptr[i])
            if okeep[p]:
                yes[p] = True


@njit(cache=True, nogil=True)
def pk32_pack(codes, bits, out):
    """enc-18 PACKED FRAMES: LE bit-stream of `bits`-wide codes (bits <= 32), one frame; out needs
    ceil(n*bits/8) + 8 bytes"""
    for i in range(codes.size):
        bitpos = np.int64(i) * bits
        b = bitpos >> 3; sh = bitpos & 7
        v = np.int64(codes[i]) << sh
        out[b] |= np.uint8(v & 0xFF); out[b + 1] |= np.uint8((v >> 8) & 0xFF)
        out[b + 2] |= np.uint8((v >> 16) & 0xFF); out[b + 3] |= np.uint8((v >> 24) & 0xFF)
        out[b + 4] |= np.uint8((v >> 32) & 0xFF)


def pk32_words(buf8):
    """THE WORD VIEW: an inflated enc-18 frame (uint8, 8 slack bytes) as aligned u64 words for the
    readers below -- one load per code, a second only when the code straddles a word. Measured on
    Title's scan: five byte loads 67.5 ms, one word 48.8 (enc-3's u32 read 63.1). Safe on the
    slack: a straddled word ends before the eight slack bytes do, for any prefix length the
    point reader asks (ceil(n*bits/8) + 8 bytes)."""
    b = np.frombuffer(buf8, np.uint8) if not isinstance(buf8, np.ndarray) else buf8
    return b[:b.size // 8 * 8].view(np.uint64)


@njit(cache=True, parallel=True, nogil=True)
def pk32_unpack(w64, bits, n, out):
    """enc-18: LE bit-stream (as u64 words, pk32_words) -> n codes (bits <= 32), parallel"""
    mask = (np.int64(1) << bits) - 1
    for i in prange(n):
        bitpos = np.int64(i) * bits
        w = bitpos >> 6; o = bitpos & 63
        v = np.int64(w64[w] >> np.uint64(o))
        if o + bits > 64:
            v |= np.int64(w64[w + 1] << np.uint64(64 - o))
        out[i] = v & mask


@njit(cache=True, nogil=True)
def pk32_unpack_serial(w64, bits, n, out):
    """the same, one thread (a frame inside a thread pool)"""
    mask = (np.int64(1) << bits) - 1
    for i in range(n):
        bitpos = np.int64(i) * bits
        w = bitpos >> 6; o = bitpos & 63
        v = np.int64(w64[w] >> np.uint64(o))
        if o + bits > 64:
            v |= np.int64(w64[w + 1] << np.uint64(64 - o))
        out[i] = v & mask


@njit(cache=True, nogil=True)
def pk32_gather(w64, bits, rows, out):
    """enc-18 codes AT ROWS of one inflated frame (rows relative to the frame)"""
    mask = (np.int64(1) << bits) - 1
    for j in range(rows.shape[0]):
        bitpos = np.int64(rows[j]) * bits
        w = bitpos >> 6; o = bitpos & 63
        v = np.int64(w64[w] >> np.uint64(o))
        if o + bits > 64:
            v |= np.int64(w64[w + 1] << np.uint64(64 - o))
        out[j] = v & mask


@njit(cache=True, nogil=True)
def pk32_flag_hits(w64, bits, s0, e0, base, fbits, hits, codes):
    """enc-18 frame scan: positions (base + i) and codes for i in [s0, e0) whose code's bit is set
    in the packed flag -- unpack and test in one loop, no intermediate array"""
    mask = (np.int64(1) << bits) - 1
    n = 0
    for i in range(s0, e0):
        bitpos = np.int64(i) * bits
        w = bitpos >> 6; o = bitpos & 63
        v = np.int64(w64[w] >> np.uint64(o))
        if o + bits > 64:
            v |= np.int64(w64[w + 1] << np.uint64(64 - o))
        v &= mask
        if (fbits[v >> 3] >> (v & 7)) & 1:
            hits[n] = base + (i - s0); codes[n] = v; n += 1
    return n


@njit(cache=True, parallel=True, nogil=True)
def pk_unpack(buf, bits, n, out):
    """enc-17 RAW PACKED CODES (Jackson's deal law): LE bit-stream ->
    codes, one parallel pass. The dress that charges no toll."""
    mask = (np.int64(1) << bits) - 1
    for i in prange(n):
        bitpos = np.int64(i) * bits
        b = bitpos >> 3
        sh = bitpos & 7
        w = (np.int64(buf[b]) | (np.int64(buf[b + 1]) << 8)
             | (np.int64(buf[b + 2]) << 16))
        out[i] = (w >> sh) & mask


@njit(cache=True, parallel=True, nogil=True)
def e14_reconstruct_at(Y, M, D, rows, ybase, inv, dmin, out):
    """FIELD-PLANE codes AT ROWS (the survivor read): streams decompress
    once (cached per query), the math runs only where asked."""
    n = rows.shape[0]
    for j in prange(n):
        i = rows[j]
        mm = np.int64(M[i])
        y = np.int64(ybase) + np.int64(Y[i])
        if mm <= 1:
            y -= 1
        if y >= 0:
            era = y // 400
        else:
            era = (y - 399) // 400
        yoe = y - era * 400
        mp = (mm + 10) % 12
        doe = yoe * 365 + yoe // 4 - yoe // 100 + (153 * mp + 2) // 5 + np.int64(D[i])
        out[j] = inv[era * 146097 + doe - 719468 - dmin]


@njit(cache=True, parallel=True, nogil=True)
def e15_reconstruct_at(Y, M, D, DL, BB, rows, role, ybase, inv, dmin, out):
    """CLOCK codes AT ROWS: anchor civil + delta swing, only where asked."""
    n = rows.shape[0]
    for j in prange(n):
        i = rows[j]
        mm = np.int64(M[i])
        y = np.int64(ybase) + np.int64(Y[i])
        if mm <= 1:
            y -= 1
        if y >= 0:
            era = y // 400
        else:
            era = (y - 399) // 400
        yoe = y - era * 400
        mp = (mm + 10) % 12
        doe = yoe * 365 + yoe // 4 - yoe // 100 + (153 * mp + 2) // 5 + np.int64(D[i])
        days = era * 146097 + doe - 719468
        b = (BB[i >> 3] >> (7 - (i & 7))) & 1
        if b != role:
            days += np.int64(DL[i])
        out[j] = inv[days - dmin]


@njit(cache=True, parallel=True, nogil=True)
def e15_band_from_streams(Y, M, D, DL, BB, role, ybase, lo, hi, out):
    """Clock band over ALREADY-DECOMPRESSED streams (the sibling share)."""
    n = out.shape[0]
    for i in prange(n):
        mm = np.int64(M[i])
        y = np.int64(ybase) + np.int64(Y[i])
        if mm <= 1:
            y -= 1
        if y >= 0:
            era = y // 400
        else:
            era = (y - 399) // 400
        yoe = y - era * 400
        mp = (mm + 10) % 12
        doe = yoe * 365 + yoe // 4 - yoe // 100 + (153 * mp + 2) // 5 + np.int64(D[i])
        days = era * 146097 + doe - 719468
        b = (BB[i >> 3] >> (7 - (i & 7))) & 1
        if b != role:
            days += np.int64(DL[i])
        out[i] = (days >= lo) and (days < hi)


@njit(cache=True, parallel=True, nogil=True)
def pk_gather(buf, bits, rows, out):
    """enc-17 codes AT ROWS: direct bit arithmetic, 1:1 granularity."""
    mask = (np.int64(1) << bits) - 1
    n = rows.shape[0]
    for j in prange(n):
        bitpos = np.int64(rows[j]) * bits
        b = bitpos >> 3
        sh = bitpos & 7
        w = (np.int64(buf[b]) | (np.int64(buf[b + 1]) << 8)
             | (np.int64(buf[b + 2]) << 16))
        out[j] = (w >> sh) & mask


@njit(cache=True, parallel=True, nogil=True)
def e15_band_lut(Y, M, D, DL, BB, role, ystart, mcum, lo, hi, out):
    """Clock band, TABLE-DRIVEN: days = ystart[y] + mcum[leap, m] + d - 1.
    Two lookups replace the civil division chain (5x less CPU per row)."""
    n = out.shape[0]
    for i in prange(n):
        yi = np.int64(Y[i])
        mi = np.int64(M[i])                     # planes: M 0..11, D 0..30
        days = ystart[yi] + mcum[mi] + np.int64(D[i])
        if mi >= 2 and (ystart[yi + 1] - ystart[yi]) == 366:
            days += 1                           # leap day sits before March
        b = (BB[i >> 3] >> (7 - (i & 7))) & 1
        if b != role:
            days += np.int64(DL[i])
        out[i] = (days >= lo) and (days < hi)


@njit(cache=True, parallel=True, nogil=True)
def pand(a, b):
    """Parallel in-place AND of two bool masks (60M serial ANDs were
    extending the shadow's wall by ~15ms each)."""
    n = a.shape[0]
    for i in prange(n):
        a[i] = a[i] and b[i]


@njit(cache=True, parallel=True, nogil=True)
def _hist_par(codes, V, part, out):
    """per-thread int32 boards over row chunks, then a parallel reduce over the codes"""
    T = part.shape[0]
    n = codes.size
    for t in prange(T):
        a = n * t // T; b = n * (t + 1) // T
        for i in range(a, b):
            part[t, np.int64(codes[i])] += np.int32(1)
    for v in prange(V):
        s = np.int64(0)
        for t in range(T):
            s += part[t, v]
        out[v] = s


@njit(cache=True, parallel=True, nogil=True)
def block_stats(codes, dvals, has_dvals, nullcode, BR, cnt, nn, bsum, cmin, cmax):
    """THE BLOCK STATISTICS (Jackson's metadata per block, written at load): per BR rows the row
    count, the non-null count, the sum of dictionary values (float64), and the min and max CODE.
    has_dvals: 0 = no sum, 1 = sum dvals[code], 2 = the codes ARE the values (mode 4), sum them.
    Blocks across threads; one pass. Was a Python loop over 3,052 blocks (0.91 s on UserID)."""
    n = codes.size
    nb = cnt.size
    for j in prange(nb):
        lo = j * BR
        hi = min(lo + BR, n)
        c = 0; s = 0.0
        mn = np.int64(9223372036854775807); mx = np.int64(-9223372036854775807 - 1)
        for i in range(lo, hi):
            v = np.int64(codes[i])
            if v == nullcode:
                continue
            c += 1
            if v < mn: mn = v
            if v > mx: mx = v
            if has_dvals == 1:
                s += dvals[v]
            elif has_dvals == 2:
                s += v
        cnt[j] = hi - lo
        nn[j] = c
        bsum[j] = s
        cmin[j] = mn
        cmax[j] = mx if c > 0 else np.int64(-1)   # an empty block: the old convention


def group_fold_dict(grp, codes, dvals, G, out):
    return _group_fold_dict_nb(grp, codes, dvals, G, out, _nt())


@njit(cache=True, parallel=True, nogil=True)
def _group_fold_dict_nb(grp, codes, dvals, G, out, T):
    """THE FOLD ON CODES: out[g] = sum over rows of dvals[codes[i]] for grp[i] == g -- a SUM or AVG
    numerator per group straight from two code streams and a dictionary, per-thread boards
    (T x G float64) and a reduce. No value array is ever built (Q09 built three, 630 ms)."""
    n = grp.size
    part = np.zeros((T, G), np.float64)
    for t in prange(T):
        a = n * t // T; b = n * (t + 1) // T
        for i in range(a, b):
            part[t, np.int64(grp[i])] += dvals[np.int64(codes[i])]
    for g in prange(G):
        s = 0.0
        for t in range(T):
            s += part[t, g]
        out[g] = s


def sort_keys_par(keys, SH):
    return _sort_keys_par_nb(keys, SH, _nt())


@njit(cache=True, parallel=True, nogil=True)
def _sort_keys_par_nb(keys, SH, T):
    """THE BUCKETED SORT: non-negative int64 keys, MSD-partitioned by (key >> SH) into 4096
    buckets (per-thread counts -> stable offsets -> scatter), each bucket sorted in place across
    threads. Sorted output, one pass + small sorts: np.sort on 6.2M keys was 217 ms on one
    thread (Q18's survivors); this is ~30. SH must put every key under 4096 buckets."""
    n = keys.size
    NB = 1 << 12
    pc = np.zeros((T, NB), np.int64)
    for t in prange(T):
        for i in range(n * t // T, n * (t + 1) // T):
            pc[t, keys[i] >> SH] += 1
    offs = np.zeros(NB + 1, np.int64)
    for b in range(NB):
        s = 0
        for t in range(T):
            v = pc[t, b]; pc[t, b] = s; s += v
        offs[b + 1] = offs[b] + s
    out = np.empty(n, np.int64)
    for t in prange(T):
        cur = np.empty(NB, np.int64)
        for b in range(NB):
            cur[b] = offs[b] + pc[t, b]
        for i in range(n * t // T, n * (t + 1) // T):
            b = keys[i] >> SH
            out[cur[b]] = keys[i]
            cur[b] += 1
    for b in prange(NB):
        lo = offs[b]; hi = offs[b + 1]
        if hi - lo > 1:
            out[lo:hi].sort()
    return out


def _count_ge(a, bar):
    return _count_ge_nb(a, bar, _nt())


@njit(cache=True, parallel=True, nogil=True)
def _count_ge_nb(a, bar, T):
    n = a.size
    part = np.zeros(T, np.int64)
    for t in prange(T):
        c = 0
        for i in range(n * t // T, n * (t + 1) // T):
            if a[i] >= bar:
                c += 1
        part[t] = c
    return part.sum()


def topk_bar(a, k):
    """THE BAR (Jackson: stop after inspecting the top hundred, not the top seventeen million):
    indices of the k largest of `a`, ordered value desc then index asc, without partitioning
    the whole array. Lower a bar from the max until at least k values clear it (each probe a
    parallel compare, ~10 ms on 17.6M), then sort only those. Exact: everything below the bar
    is below the k-th value."""
    a = np.ascontiguousarray(a)
    n = int(a.size); k = int(k)
    if k <= 0 or n == 0:
        return np.empty(0, np.int64)
    if k >= n:
        return np.argsort(-a, kind='stable')
    mx = int(a.max()); mn = int(a.min())
    if mx == mn:
        return np.arange(k)
    bar = mx
    step = max(1, (mx - mn) // 64)
    while bar > mn and _count_ge(a, bar) < k:
        bar -= step
        step *= 2
    if bar < mn:
        bar = mn
    cand = np.flatnonzero(a >= bar)
    return cand[np.argsort(-a[cand], kind='stable')][:k]


@njit(nogil=True, cache=True)
def _counting_order_nb(u, span, out):
    cnt = np.zeros(span + 2, np.int64)
    for i in range(u.size):
        cnt[u[i] + 1] += 1
    for j in range(1, span + 2):
        cnt[j] += cnt[j - 1]
    for i in range(u.size):
        v = u[i]
        out[cnt[v]] = i
        cnt[v] += 1


COUNTING_ORDER_MAX_SPAN = 1 << 24


def counting_order(key, out_dtype=np.int64):
    """THE CLUSTER ORDER BY COUNTING (2026-09-26): the stable row order of one integer (or datetime)
    key -- the same permutation np.lexsort((key,)) / argsort(kind='stable') gives: rows of equal key
    keep their file order. One count pass, one prefix, one scatter: EventTime's 100M rows in 0.75 s
    against lexsort's 8.4 s. Returns None when the key's span is past COUNTING_ORDER_MAX_SPAN (the
    count array would be too large) or the key is not integral: the caller sorts as before."""
    k = np.asarray(key)
    if k.dtype.kind == 'M':
        k = k.view(np.int64)
    if k.dtype.kind not in 'iu' or k.ndim != 1 or k.dtype == np.uint64:
        return None                               # (a u64 key could exceed int64: the caller's sort)
    n = k.size
    out = np.empty(n, out_dtype)
    if n == 0:
        return out
    lo = int(k.min()); span = int(k.max()) - lo
    if span > COUNTING_ORDER_MAX_SPAN:
        return None
    u = k.astype(np.int64)                        # widened before the subtraction: a narrow signed key
    u -= lo                                       # minus its minimum can overflow its own width
    u = u.astype(np.int32)
    _counting_order_nb(u, span, out)
    return out


def bincount_par(codes, minlength):
    """THE PARALLEL CENSUS: np.bincount of 100M codes is one thread for 300 ms (plus an astype
    copy); this is per-thread boards and a reduce, measured ~60 ms. Boards cost T*V*4 bytes, so
    a dictionary past 64M codes falls back to numpy. Codes are assumed < minlength (a null code
    past the dictionary is the caller's to trim)."""
    codes = np.asarray(codes)
    V = int(minlength)
    if codes.size < (1 << 22) or V > (64 << 20) or codes.dtype.kind not in 'iu':
        return np.bincount(codes, minlength=V)
    if V < int(codes.max()) + 1:
        return np.bincount(codes, minlength=V)
    T = min(numba.get_num_threads(), 16)
    part = np.zeros((T, V), np.int32)
    out = np.empty(V, np.int64)
    _hist_par(np.ascontiguousarray(codes), V, part, out)
    return out


def count_codes(codes, K):
    """COUNT(*) per code in the codes' own width -- no widening copy (np.bincount casts uint8 to
    int64 first: 2.2 ms on Q07's 630k codes against 0.2 ms for the generated parallel loop). Codes
    past K are not counted (the generated kernel's contract: a trailing null slot is dropped).
    The thread count rides in as an argument so the kernel stays cacheable."""
    return _count_codes_nb(codes, np.int64(K), _nt())


@njit(cache=True, parallel=True, nogil=True)
def _count_codes_nb(codes, K, T):
    n = codes.shape[0]
    part = np.zeros((T, K), np.int64)
    chunk = (n + T - 1) // T
    for t in prange(T):
        lo = t * chunk
        hi = min(lo + chunk, n)
        for i in range(lo, hi):
            c = codes[i]
            if c < K:
                part[t, c] += 1
    out = np.zeros(K, np.int64)
    for t in range(T):
        for k in range(K):
            out[k] += part[t, k]
    return out


@njit(cache=True, parallel=True, nogil=True)
def pcount_chunks(mask, counts, chunk):
    nc = counts.shape[0]
    n = mask.shape[0]
    for c in prange(nc):
        lo = c * chunk
        hi = min(n, lo + chunk)
        t = 0
        for i in range(lo, hi):
            if mask[i]:
                t += 1
        counts[c] = t


@njit(cache=True, parallel=True, nogil=True)
def pfill_rows(mask, offs, out, chunk):
    nc = offs.shape[0] - 1
    n = mask.shape[0]
    for c in prange(nc):
        lo = c * chunk
        hi = min(n, lo + chunk)
        w = offs[c]
        for i in range(lo, hi):
            if mask[i]:
                out[w] = i
                w += 1


@njit(cache=True, parallel=True, nogil=True)
def plut_u8(codes, lut, out):
    """Parallel bool LUT over u8/u16/int codes (kx[codes] at 60M was a serial
    ~90ms numpy gather in the cascade's native keeps)."""
    n = codes.shape[0]
    for i in prange(n):
        out[i] = lut[np.int64(codes[i])]


@njit(cache=True, parallel=True, nogil=True)
def pkeep_via_ptr(rows, ptr, keep, out):
    """out[j] = keep[ptr[rows[j]]] -- a parent keep applied through a road at
    survivor rows, in parallel (serial numpy chained gathers cost ~118ms at 15M)."""
    n = rows.shape[0]
    for j in prange(n):
        out[j] = keep[np.int64(ptr[np.int64(rows[j])])]


@njit(cache=True, parallel=True, nogil=True)
def pgather_ptr(p, cc, out):
    """out[i] = p[cc[i]] -- composing a grandparent road through a parent road
    at fact scale, in parallel (serial numpy fancy-index cost ~100ms per 60M)."""
    n = cc.shape[0]
    for i in prange(n):
        out[i] = p[np.int64(cc[i])]


@njit(cache=True, parallel=True, nogil=True)
def plike2(blob, off, n1, n2, out):
    """Ordered two-needle LIKE over a mode-5 inline stream: out[r] = row r's
    bytes contain n1 and then n2 after it (n2 empty => single needle). The
    Python-string version of this scan cost 15s on Q13; bytes cost ~150ms."""
    R = off.shape[0] - 1
    L1 = n1.shape[0]; L2 = n2.shape[0]
    for r in prange(R):
        a = off[r]; b = off[r + 1]
        p1 = np.int64(-1)
        i = a
        while i <= b - L1:
            k = 0
            while k < L1 and blob[i + k] == n1[k]:
                k += 1
            if k == L1:
                p1 = i + L1
                break
            i += 1
        if p1 < 0:
            out[r] = False
            continue
        if L2 == 0:
            out[r] = True
            continue
        ok = False
        i = p1
        while i <= b - L2:
            k = 0
            while k < L2 and blob[i + k] == n2[k]:
                k += 1
            if k == L2:
                ok = True
                break
            i += 1
        out[r] = ok


@njit(cache=True, parallel=True, nogil=True)
def pprefix2(blob, off, out):
    """out[r] = first two bytes of row r as u16 (b0<<8|b1) -- a 2-byte read,
    never a Python string (Q22's country code)."""
    R = off.shape[0] - 1
    for r in prange(R):
        a = off[r]
        if off[r + 1] - a >= 2:
            out[r] = (np.uint16(blob[a]) << 8) | np.uint16(blob[a + 1])
        else:
            out[r] = 0


def plike_fc(buf, restarts, R, V, n1, n2, keep):
    return _plike_fc_nb(buf, restarts, R, V, n1, n2, keep, _nt())


@njit(cache=True, parallel=True, nogil=True)
def _plike_fc_nb(buf, restarts, R, V, n1, n2, keep, NT):
    """Ordered-needle LIKE over a FRONT-CODED dict: restart blocks walk
    sequentially (prefix carry), blocks run in parallel. keep[code]=contains."""
    nb = (V + R - 1) // R
    L1 = n1.shape[0]; L2 = n2.shape[0]
    T = min(nb, NT * 8)                            # one carry buffer per chunk of blocks, not per
    for t in prange(T):                            # block (1.1M mallocs of 128 KB on an 18M URL dict)
        prev = np.empty(131072, np.uint8)          # cp, sl are u16: a value is at most 131070 bytes
        for b in range(nb * t // T, nb * (t + 1) // T):
            o = np.int64(restarts[b])
            hi = min(R, V - b * R)
            mend = np.int64(-1)                    # THE PREFIX CARRY: where the previous value's
            for step in range(hi):                 # first match ENDED; a match inside the shared
                cp = np.int64(buf[o]) | (np.int64(buf[o + 1]) << 8)      # prefix is still a match,
                sl = np.int64(buf[o + 2]) | (np.int64(buf[o + 3]) << 8)  # so only the new bytes
                o += 4                                                    # are searched (2 GB of
                for q in range(sl):                                       # decoded text -> ~500 MB)
                    prev[cp + q] = buf[o + q]
                o += sl
                plen = cp + sl
                p1 = np.int64(-1)
                if mend >= 0 and mend <= cp:
                    p1 = mend
                    i = plen                       # nothing to search
                else:
                    i = cp - L1 + 1
                    if i < 0:
                        i = 0
                while i <= plen - L1:
                    k = 0
                    while k < L1 and prev[i + k] == n1[k]:
                        k += 1
                    if k == L1:
                        p1 = i + L1
                        break
                    i += 1
                mend = p1
                ok = False
                if p1 >= 0:
                    if L2 == 0:
                        ok = True
                    else:
                        i = p1
                        while i <= plen - L2:
                            k = 0
                            while k < L2 and prev[i + k] == n2[k]:
                                k += 1
                            if k == L2:
                                ok = True
                                break
                            i += 1
                keep[b * R + step] = ok


@njit(cache=True, parallel=True, nogil=True)
def pruns_distinct(starts, ptr, supp, flag, nsupp, nflag):
    """Per contiguous run of ptr (a sorted road): count distinct supp values,
    and distinct supp values among flagged rows (Q21's blame censuses)."""
    S = starts.shape[0] - 1
    for s in prange(S):
        i = starts[s]; j = starts[s + 1]
        c = 0; cf = 0
        for a in range(i, j):
            seen = False
            for b in range(i, a):
                if supp[b] == supp[a]:
                    seen = True; break
            if not seen:
                c += 1
            if flag[a]:
                seenf = False
                for b in range(i, a):
                    if flag[b] and supp[b] == supp[a]:
                        seenf = True; break
                if not seenf:
                    cf += 1
        nsupp[ptr[i]] = c; nflag[ptr[i]] = cf


@njit(cache=True, nogil=True)
def hcomposite2(w0, w1, gid, tk0, tk1, tg):
    """THE HASHED COMPOSITE: open-addressing group ids over a 128-bit packed
    key (two int64 words). Table arrays sized to a power of two >= 2n, tg
    initialised to -1. Returns the number of groups; tk0/tk1 hold each
    slot's key words, tg its gid (group g's words are found via rep below)."""
    n = w0.shape[0]
    mask = tk0.shape[0] - 1
    ng = 0
    for i in range(n):
        a = w0[i]; b = w1[i]
        h = (a * np.int64(-7046029254386353131)) ^ ((b + np.int64(0x9E3779B97F4A7C15)) * np.int64(-4265267296055464877))
        h ^= (h >> np.int64(29))
        j = np.int64(h) & mask
        while True:
            g = tg[j]
            if g < 0:
                tk0[j] = a; tk1[j] = b; tg[j] = ng; gid[i] = ng; ng += 1
                break
            if tk0[j] == a and tk1[j] == b:
                gid[i] = g
                break
            j = (j + 1) & mask
    return ng


@njit(cache=True, parallel=True, nogil=True)
def hcomposite_rep(tg, tk0, tk1, g0, g1):
    """Scatter each slot's key words to its group id (g0/g1 sized ngroups)."""
    m = tg.shape[0]
    for j in prange(m):
        g = tg[j]
        if g >= 0:
            g0[g] = tk0[j]; g1[g] = tk1[j]


@njit(cache=True, parallel=True, nogil=True)
def pack2(codes, shifts, words, w0, w1):
    """Pack k code arrays (int64[k, n] as a 2-D view) into two words by the
    given bit shifts; words[j] says which word column j lands in."""
    k = codes.shape[0]; n = codes.shape[1]
    for i in prange(n):
        a = np.int64(0); b = np.int64(0)
        for j in range(k):
            v = codes[j, i] << shifts[j]
            if words[j] == 0:
                a |= v
            else:
                b |= v
        w0[i] = a; w1[i] = b


@njit(cache=True, nogil=True)
def pscatter_by_gid(gid, vals, offs, out):
    """Counting scatter: place each value into its group's slice (offs = start
    per group, advanced in place). Sequential, O(n)."""
    n = gid.shape[0]
    for i in range(n):
        g = gid[i]
        out[offs[g]] = vals[i]
        offs[g] += 1


@njit(cache=True, parallel=True, nogil=True)
def pgroup_median(out_vals, starts, ends, med):
    """Median per group over group-contiguous values: sort each slice
    (groups in parallel), mean of the two middles on even counts."""
    G = starts.shape[0]
    for g in prange(G):
        a = starts[g]; b = ends[g]
        cnt = b - a
        if cnt <= 0:
            med[g] = np.nan
            continue
        sl = np.sort(out_vals[a:b])
        lo = (cnt - 1) // 2
        hi = cnt // 2
        med[g] = (sl[lo] + sl[hi]) / 2.0


@njit(cache=True, parallel=True, nogil=True)
def pgroup_topk(idx_placed, vals, starts, ends, k, desc, out_rows, out_cnt):
    """Top-k rows per group: idx_placed holds row indices grouped by gid;
    each group's slice sorts by vals[row] (desc if desc), the first k row
    indices land in out_rows[g*k : g*k+cnt]; out_cnt[g] = cnt."""
    G = starts.shape[0]
    for g in prange(G):
        a = starts[g]; b = ends[g]
        m = b - a
        if m <= 0:
            out_cnt[g] = 0
            continue
        sl = idx_placed[a:b]
        v = np.empty(m, np.float64)
        for t in range(m):
            v[t] = -vals[sl[t]] if desc else vals[sl[t]]
        order = np.argsort(v, kind='mergesort')
        c = k if m > k else m
        for t in range(c):
            out_rows[g * k + t] = sl[order[t]]
        out_cnt[g] = c


# ------------------------------------------------------------------------------------------------
# enc 19 = THE BLOCK DICTIONARIES (Jackson, 2026-09-23: "store each distinct value once and
# populate the rows with pointers, so the scope of V collapses to the distinct count" -- applied
# one level down, to the row pointers themselves). Per block of BR rows: the sorted distinct
# global codes present (first code at gbits, then gaps at the block's gap width), and one local
# pointer per row at the block's own width. Every block starts on a u64 word, so the encoder and
# the decoder run block-parallel with no shared words. Decode is one jump per row.

@njit(inline='always')
def _e19_put(w, pos, v, nbits):
    if nbits > 0:
        i = pos >> 6; sh = pos & 63
        w[i] |= np.uint64(v) << np.uint64(sh)
        if sh + nbits > 64:
            w[i + 1] |= np.uint64(v) >> np.uint64(64 - sh)


@njit(inline='always')
def _e19_get(w, pos, nbits):
    if nbits == 0:
        return np.int64(0)
    i = pos >> 6; sh = pos & 63
    v = w[i] >> np.uint64(sh)
    if sh + nbits > 64:
        v |= w[i + 1] << np.uint64(64 - sh)
    return np.int64(v & ((np.uint64(1) << np.uint64(nbits)) - np.uint64(1)))


@njit(cache=True, parallel=True, nogil=True)
def e19_plan(codes, BR, gbits, lb, gw, dcnt, pwords, dwords):
    """per block: distinct count, local pointer width, dictionary gap width, and the u64 words
    the block's pointers and dictionary occupy -- the exact size, before a byte is written"""
    N = codes.size; nb = lb.size
    for b in prange(nb):
        lo = b * BR; hi = min(lo + BR, N)
        s = np.sort(codes[lo:hi])
        d = 1; mg = np.int64(0)
        for i in range(1, s.size):
            if s[i] != s[i - 1]:
                g = np.int64(s[i]) - np.int64(s[i - 1])
                if g > mg:
                    mg = g
                d += 1
        l = 0
        while (np.int64(1) << l) < d:
            l += 1
        w = 0
        while (np.int64(1) << w) <= mg:
            w += 1
        lb[b] = l; gw[b] = w; dcnt[b] = d
        pwords[b] = (np.int64(hi - lo) * l + 63) // 64
        dwords[b] = (np.int64(gbits) + np.int64(d - 1) * w + 63) // 64


@njit(cache=True, parallel=True, nogil=True)
def e19_write(codes, BR, gbits, lb, gw, poff, doff, pw, dw):
    """write every block's dictionary (first code + gaps) and its row pointers (the local ids)"""
    N = codes.size; nb = lb.size
    for b in prange(nb):
        lo = b * BR; hi = min(lo + BR, N)
        blk = codes[lo:hi]
        s = np.unique(blk)
        pos = doff[b] * 64
        _e19_put(dw, pos, s[0], gbits); pos += gbits
        g = np.int64(gw[b])
        for k in range(1, s.size):
            _e19_put(dw, pos, np.int64(s[k]) - np.int64(s[k - 1]), g); pos += g
        l = np.int64(lb[b]); pos = poff[b] * 64
        for i in range(blk.size):
            _e19_put(pw, pos, np.searchsorted(s, blk[i]), l); pos += l


@njit(inline='always')
def _e19_dict(dw, b, gbits, gw, dcnt, doff, upto):
    """a block's dictionary entries [0, upto]: the first code, then the running sum of the gaps"""
    loc = np.empty(upto + 1, np.int64)
    pos = doff[b] * 64; g = np.int64(gw[b])
    v = _e19_get(dw, pos, gbits); pos += gbits; loc[0] = v
    for k in range(1, upto + 1):
        v += _e19_get(dw, pos, g); pos += g; loc[k] = v
    return loc


@njit(cache=True, parallel=True, nogil=True)
def e19_decode(pw, dw, BR, N, gbits, lb, gw, dcnt, poff, doff, out):
    """the full column: per block, its dictionary, then one jump per row"""
    nb = lb.size
    for b in prange(nb):
        lo = b * BR; hi = min(lo + BR, N)
        loc = _e19_dict(dw, b, gbits, gw, dcnt, doff, np.int64(dcnt[b]) - 1)
        l = np.int64(lb[b]); pos = poff[b] * 64
        for i in range(lo, hi):
            out[i] = loc[_e19_get(pw, pos, l)]; pos += l


@njit(inline='always')
def _valsum_block(h, loc, dc, tab, ntab):
    """one block's exact share of SUM(value): its pointer counts h against its dictionary loc, each
    value split in two 32-bit halves so no partial sum can wrap (a block holds at most BR rows)"""
    sh = np.int64(0); sl = np.int64(0); nn = np.int64(0)
    for j in range(dc):
        k = loc[j]
        if k >= ntab:
            continue                                # the null code: no value, not counted
        v = tab[k]; cnt = h[j]
        sh += cnt * (v >> 32)
        sl += cnt * (v & np.int64(0xFFFFFFFF))
        nn += cnt
    return sh, sl, nn


@njit(cache=True, parallel=True, nogil=True)
def e19_valsum(pw, dw, BR, N, gbits, lb, gw, dcnt, poff, doff, tab, ntab, hi_out, lo_out, nn_out):
    """THE SUM FROM THE BLOCK DICTIONARIES, enc 19 (2026-10-01): per block, count its row pointers on
    a board the size of its own dictionary (cache-resident), then weigh each entry once by its value
    -- no 100M-row code array and no V-sized census. SUM = sum(hi) * 2^32 + sum(lo), exact."""
    nb = lb.size
    for b in prange(nb):
        lo = b * BR; hi = min(lo + BR, N)
        dc = np.int64(dcnt[b])
        loc = _e19_dict(dw, b, gbits, gw, dcnt, doff, dc - 1)
        h = np.zeros(dc, np.int64)
        l = np.int64(lb[b]); pos = poff[b] * 64
        for i in range(lo, hi):
            h[_e19_get(pw, pos, l)] += 1; pos += l
        a, s, n = _valsum_block(h, loc, dc, tab, ntab)
        hi_out[b] = a; lo_out[b] = s; nn_out[b] = n


@njit(cache=True, parallel=True, nogil=True)
def e19s_valsum(pw, dw, BR, N, W, wb, lb, gw, dcnt, poff, pre, soff, SW, tab, ntab, hi_out, lo_out, nn_out):
    """e19_valsum over the shelves (the labels by code range)"""
    nb = lb.size
    for b in prange(nb):
        lo = b * BR; hi = min(lo + BR, N)
        dc = np.int64(dcnt[b])
        loc = _e19s_dict(dw, W, wb, gw, pre, soff, SW, b, dc - 1)
        h = np.zeros(dc, np.int64)
        l = np.int64(lb[b]); pos = poff[b] * 64
        for i in range(lo, hi):
            h[_e19_get(pw, pos, l)] += 1; pos += l
        a, s, n = _valsum_block(h, loc, dc, tab, ntab)
        hi_out[b] = a; lo_out[b] = s; nn_out[b] = n


@njit(cache=True, parallel=True, nogil=True)
def e19_signposts(dw, gbits, gw, dcnt, doff, S, spo, sp):
    """THE SIGNPOSTS (Jackson, 2026-09-27): per block, every S-th entry of its sorted dictionary kept
    whole (entries 0, S, 2S, ...), written at sp[spo[b]:spo[b+1]] -- a lookup can then halve among a
    block's signposts and walk at most S-1 gaps instead of the whole dictionary."""
    for b in prange(dcnt.size):
        pos = doff[b] * 64; g = np.int64(gw[b])
        v = _e19_get(dw, pos, gbits); pos += gbits
        o = spo[b]; sp[o] = v; o += 1
        for k in range(1, np.int64(dcnt[b])):
            v += _e19_get(dw, pos, g); pos += g
            if k % S == 0:
                sp[o] = v; o += 1


@njit(cache=True, nogil=True)
def e19_eq_blocks_sp(dw, gbits, gw, dcnt, doff, S, spo, sp, t, b0, b1, lid):
    """per block in [b0, b1): the local id of code t in its dictionary, or -1 -- through the signposts
    (the last signpost <= t, then at most S-1 gaps). One thread: 1,526 blocks x ~150 steps is
    0.15 ms, less than a thread pool's start on a busy host (measured 7 ms with 16 threads)."""
    for b in range(b0, b1):
        a = spo[b]; e = spo[b + 1]
        lo = a; hi = e
        while lo < hi:
            m = (lo + hi) // 2
            if sp[m] <= t:
                lo = m + 1
            else:
                hi = m
        lid[b] = -1
        j = lo - 1 - a
        if j < 0:
            continue
        k = j * S; v = sp[a + j]
        if v == t:
            lid[b] = k
            continue
        g = np.int64(gw[b]); pos = doff[b] * 64 + gbits + k * g      # the gap that makes entry k + 1
        stop = min(np.int64(dcnt[b]), k + S)
        for kk in range(k + 1, stop):
            v += _e19_get(dw, pos, g); pos += g
            if v >= t:
                if v == t:
                    lid[b] = kk
                break


@njit(cache=True, parallel=True, nogil=True)
def e19_eq_blocks(dw, gbits, gw, dcnt, doff, t, b0, b1, lid):
    """the same answer without signposts: walk each block's sorted dictionary until it reaches t"""
    for b in prange(b0, b1):
        pos = doff[b] * 64; g = np.int64(gw[b])
        v = _e19_get(dw, pos, gbits); pos += gbits
        lid[b] = -1
        if v == t:
            lid[b] = 0
            continue
        if v > t:
            continue
        for k in range(1, np.int64(dcnt[b])):
            v += _e19_get(dw, pos, g); pos += g
            if v >= t:
                if v == t:
                    lid[b] = k
                break


@njit(cache=True, nogil=True)
def e19_rows_eq(pw, BR, N, lb, poff, lid, b0, b1, lo, hi, out):
    """the rows in [lo, hi) of the blocks with lid >= 0 whose pointer is that local id; returns the count"""
    k = 0
    for b in range(b0, b1):
        j = lid[b]
        if j < 0:
            continue
        l = np.int64(lb[b]); r0 = b * BR; r1 = min(r0 + BR, N)
        a = max(r0, lo); e = min(r1, hi)
        pos = poff[b] * 64 + (a - r0) * l
        for i in range(a, e):
            if _e19_get(pw, pos, l) == j:
                if k < out.size:
                    out[k] = i
                k += 1
            pos += l
    return k


@njit(cache=True, parallel=True, nogil=True)
def e19_gather(pw, dw, BR, gbits, lb, gw, dcnt, poff, doff, rows, starts, out):
    """codes at sorted rows, grouped by block (starts: the group boundaries). A block's
    dictionary is decoded only as far as the highest local id its rows point at."""
    for t in prange(starts.size - 1):
        s0 = starts[t]; s1 = starts[t + 1]
        b = rows[s0] // BR
        l = np.int64(lb[b]); base = poff[b] * 64; r0 = b * BR
        mx = np.int64(0)
        for j in range(s0, s1):
            lid = _e19_get(pw, base + (rows[j] - r0) * l, l)
            out[j] = lid
            if lid > mx:
                mx = lid
        loc = _e19_dict(dw, b, gbits, gw, dcnt, doff, mx)
        for j in range(s0, s1):
            out[j] = loc[out[j]]


# THE SHELVES (Jackson, 2026-09-27): the same block labels, laid out by CODE RANGE instead of by block.
# Shelf r holds, block after block, each block's label entries whose code is in [r*W, (r+1)*W): the
# block's first entry there as (code - r*W) in wb bits, then its gaps at the block's own width gw[b].
# pre[r, b] = how many of block b's entries sit on shelves before r (local id = pre[r, b] + k);
# soff[r, b] = the bit where block b's piece starts on shelf r; SW[r] = shelf r's first u64 word.
# 'Which blocks hold code t' reads ONE shelf (+ one row of each table) instead of a piece of every
# label: UserID cold 69-78 ms (the whole 58 MB, in parallel) -> 5.5-7 ms (~0.25 MB), measured.

@njit(cache=True, parallel=True, nogil=True)
def e19s_write(codes, BR, lb, poff, pw, LS, L):
    """per block: its sorted distinct codes into L[LS[b]:LS[b+1]], and its row pointers (as e19_write)"""
    N = codes.size; nb = lb.size
    for b in prange(nb):
        lo = b * BR; hi = min(lo + BR, N)
        blk = codes[lo:hi]
        s = np.unique(blk)
        for k in range(s.size):
            L[LS[b] + k] = s[k]
        l = np.int64(lb[b]); pos = poff[b] * 64
        for i in range(blk.size):
            _e19_put(pw, pos, np.searchsorted(s, blk[i]), l); pos += l


@njit(cache=True, parallel=True, nogil=True)
def e19s_counts(L, LS, W, cnt):
    """cnt[r + 1, b] = how many of block b's label entries fall on shelf r"""
    for b in prange(LS.size - 1):
        for i in range(LS[b], LS[b + 1]):
            cnt[np.int64(L[i]) // W + 1, b] += 1


@njit(cache=True, parallel=True, nogil=True)
def e19s_shelve(L, LS, W, wb, gw, pre, soff, SW, dw):
    """write every shelf (one thread per shelf: shelves start on their own u64 word)"""
    R = SW.size - 1; nb = LS.size - 1
    for r in prange(R):
        base = SW[r] * 64
        for b in range(nb):
            p0 = np.int64(pre[r, b]); c = np.int64(pre[r + 1, b]) - p0
            if c == 0:
                continue
            i0 = LS[b] + p0; pos = base + np.int64(soff[r, b])
            _e19_put(dw, pos, np.int64(L[i0]) - r * W, wb); pos += wb
            g = np.int64(gw[b])
            for k in range(1, c):
                _e19_put(dw, pos, np.int64(L[i0 + k]) - np.int64(L[i0 + k - 1]), g); pos += g


@njit(inline='always')
def _e19s_dict(dw, W, wb, gw, pre, soff, SW, b, upto):
    """block b's label entries [0, upto], shelf after shelf"""
    loc = np.empty(upto + 1, np.int64)
    k = 0; g = np.int64(gw[b])
    for r in range(SW.size - 1):
        c = np.int64(pre[r + 1, b]) - np.int64(pre[r, b])
        if c == 0:
            continue
        pos = SW[r] * 64 + np.int64(soff[r, b])
        v = r * W + _e19_get(dw, pos, wb); pos += wb
        loc[k] = v; k += 1
        if k > upto:
            break
        for j in range(1, c):
            v += _e19_get(dw, pos, g); pos += g
            loc[k] = v; k += 1
            if k > upto:
                break
        if k > upto:
            break
    return loc


@njit(cache=True, parallel=True, nogil=True)
def e19s_decode(pw, dw, BR, N, W, wb, lb, gw, dcnt, poff, pre, soff, SW, b0, b1, out):
    """the blocks [b0, b1) of the column into out (global rows): per block its label, one jump per row"""
    for b in prange(b0, b1):
        lo = b * BR; hi = min(lo + BR, N)
        loc = _e19s_dict(dw, W, wb, gw, pre, soff, SW, b, np.int64(dcnt[b]) - 1)
        l = np.int64(lb[b]); pos = poff[b] * 64
        for i in range(lo, hi):
            out[i] = loc[_e19_get(pw, pos, l)]; pos += l


@njit(cache=True, parallel=True, nogil=True)
def e19s_gather(pw, dw, BR, W, wb, lb, gw, poff, pre, soff, SW, rows, starts, out, per_row):
    """e19_gather over the shelves. A block with at most per_row rows asked: each row goes straight to
    its entry -- the shelf holding local id j is found by halving block b's column of pre, then a walk
    inside that one piece (~dc/R entries) -- so a row touches ONE shelf page, not all R. A block with
    more rows: its label decoded once, only as far as its highest pointer."""
    R = SW.size - 1
    for t in prange(starts.size - 1):
        s0 = starts[t]; s1 = starts[t + 1]
        b = rows[s0] // BR
        l = np.int64(lb[b]); base = poff[b] * 64; r0 = b * BR
        mx = np.int64(0)
        for j in range(s0, s1):
            lid = _e19_get(pw, base + (rows[j] - r0) * l, l)
            out[j] = lid
            if lid > mx:
                mx = lid
        if s1 - s0 <= per_row:
            g = np.int64(gw[b])
            for j in range(s0, s1):
                k = out[j]
                lo = 0; hi = R                          # the last shelf r with pre[r, b] <= k
                while hi - lo > 1:
                    m = (lo + hi) // 2
                    if np.int64(pre[m, b]) <= k:
                        lo = m
                    else:
                        hi = m
                r = lo
                pos = SW[r] * 64 + np.int64(soff[r, b])
                v = r * W + _e19_get(dw, pos, wb); pos += wb
                for s in range(k - np.int64(pre[r, b])):
                    v += _e19_get(dw, pos, g); pos += g
                out[j] = v
        else:
            loc = _e19s_dict(dw, W, wb, gw, pre, soff, SW, b, mx)
            for j in range(s0, s1):
                out[j] = loc[out[j]]


@njit(cache=True, nogil=True)
def e19s_eq_blocks(dw, W, wb, gw, pre, soff, SW, t, b0, b1, lid):
    """per block in [b0, b1): the local id of code t, or -1 -- reading only shelf t // W. One thread
    (1,526 blocks x ~40 steps: 0.13 ms measured)"""
    r = t // W
    for b in range(b0, b1):
        lid[b] = -1
    if t < 0 or r >= SW.size - 1:
        return
    base = SW[r] * 64; lo = r * W
    for b in range(b0, b1):
        p0 = np.int64(pre[r, b]); c = np.int64(pre[r + 1, b]) - p0
        if c == 0:
            continue
        pos = base + np.int64(soff[r, b])
        v = lo + _e19_get(dw, pos, wb); pos += wb
        if v >= t:
            if v == t:
                lid[b] = p0
            continue
        g = np.int64(gw[b])
        for k in range(1, c):
            v += _e19_get(dw, pos, g); pos += g
            if v >= t:
                if v == t:
                    lid[b] = p0 + k
                break


# ------------------------------------------------------------------------------------------------
# THE THREE READS (Jackson, 2026-09-23): differentiation (codes), identification (a decision per
# distinct string, decisive bytes only, inherited across a shared prefix), return (answer strings
# only). The kernels below serve IDENTIFICATION over one front-coded dictionary chunk at a time --
# a chunk's own bytes, as decompressed, never glued into a whole-dictionary blob.

@njit(cache=True, nogil=True)
def plike_fc_serial(buf, restarts, R, V, n1, n2, keep):
    """plike_fc for ONE chunk on one thread (the chunk loop runs in a thread pool): keep[i] =
    value i of this chunk contains n1 (then n2 after it). restarts are chunk-local offsets. THE
    PREFIX CARRY: a match that ended inside the shared prefix is inherited -- only new bytes searched."""
    nb = (V + R - 1) // R
    L1 = n1.shape[0]; L2 = n2.shape[0]
    prev = np.empty(131072, np.uint8)
    for b in range(nb):
        o = np.int64(restarts[b])
        hi = min(R, V - b * R)
        mend = np.int64(-1)
        for step in range(hi):
            cp = np.int64(buf[o]) | (np.int64(buf[o + 1]) << 8)
            sl = np.int64(buf[o + 2]) | (np.int64(buf[o + 3]) << 8)
            o += 4
            for q in range(sl):
                prev[cp + q] = buf[o + q]
            o += sl
            plen = cp + sl
            p1 = np.int64(-1)
            if mend >= 0 and mend <= cp:
                p1 = mend
                i = plen
            else:
                i = cp - L1 + 1
                if i < 0:
                    i = 0
            while i <= plen - L1:
                k = 0
                while k < L1 and prev[i + k] == n1[k]:
                    k += 1
                if k == L1:
                    p1 = i + L1
                    break
                i += 1
            mend = p1
            ok = False
            if p1 >= 0:
                if L2 == 0:
                    ok = True
                else:
                    i = p1
                    while i <= plen - L2:
                        k = 0
                        while k < L2 and prev[i + k] == n2[k]:
                            k += 1
                        if k == L2:
                            ok = True
                            break
                        i += 1
            keep[b * R + step] = ok


@njit(cache=True, parallel=True, nogil=True)
def label_hash(blob, offs, out):
    """FNV-1a 64 over each label blob[offs[r]:offs[r+1]] -- a bucket, never an identity"""
    for r in prange(out.size):
        h = np.uint64(14695981039346656037)
        for i in range(offs[r], offs[r + 1]):
            h ^= np.uint64(blob[i])
            h *= np.uint64(1099511628211)
        out[r] = h


@njit(inline='always')
def _label_eq(blob, offs, a, b):
    la = offs[a + 1] - offs[a]
    if la != offs[b + 1] - offs[b]:
        return False
    oa = offs[a]; ob = offs[b]
    for k in range(la):
        if blob[oa + k] != blob[ob + k]:
            return False
    return True


@njit(cache=True, nogil=True)
def label_groups(blob, offs, h, order, gid, rep):
    """EXACT label groups: walk labels in hash order; inside one equal-hash bucket every label is
    compared BYTE FOR BYTE with the bucket's representatives, so a collision can never merge two
    different labels. gid[run] = group, rep[group] = a run carrying it. Returns the group count."""
    n = order.size; G = 0; i = 0
    while i < n:
        j = i
        while j < n and h[order[j]] == h[order[i]]:
            j += 1
        base = G
        for t in range(i, j):
            r = order[t]; found = -1
            for g in range(base, G):
                if _label_eq(blob, offs, r, rep[g]):
                    found = g
                    break
            if found < 0:
                rep[G] = r; found = G; G += 1
            gid[r] = found
        i = j
    return G


@njit(cache=True, nogil=True)
def first_index(g, out):
    """out[k] = the first position whose group is k (out pre-filled with -1): one pass, no sort"""
    for i in range(g.size):
        k = g[i]
        if out[k] < 0:
            out[k] = i


# ---------------------------------------------------------------------------------------------
# THE THREE STREAMS (Jackson, 2026-09-23): a front-coded dictionary chunk is stored as
#   headers  <cp u16><sl u16> per entry          (what differentiates, and every byte length)
#   mask     one bit per text byte, 1 = the byte starts a character   (every character length)
#   text     the suffix bytes, back to back       (what identification reads when it must)
# "capture the length by having a fixed width and then measuring the mask of each entry": the
# width is fixed in the READING -- the mask is taken 64 bits at a time and a string's length is
# the count of its set bits. The interleaved form is byte-for-byte recoverable (fc3_join).

@njit(nogil=True, cache=True)
def fc3_split(a, n, hdr, text, mask):
    """one interleaved chunk (n entries) -> hdr (4n bytes), text, packed mask (little bit order).
    mask must be zeroed and sized >= ceil(text/64)*8. Returns the text length."""
    o = np.int64(0); t = np.int64(0)
    for e in range(n):
        hdr[4 * e] = a[o]; hdr[4 * e + 1] = a[o + 1]; hdr[4 * e + 2] = a[o + 2]; hdr[4 * e + 3] = a[o + 3]
        sl = np.int64(a[o + 2]) | (np.int64(a[o + 3]) << 8)
        o += 4
        for q in range(sl):
            x = a[o + q]
            text[t] = x
            if (x & 0xC0) != 0x80:
                mask[t >> 3] |= np.uint8(1 << (t & 7))
            t += 1
        o += sl
    assert o == a.size, 'fc3_split: chunk bytes not consumed exactly'
    return t


@njit(nogil=True, cache=True)
def fc3_join(hdr, text, out):
    """headers + text -> the interleaved chunk, byte-identical to what fc3_split was given"""
    n = hdr.size // 4; o = np.int64(0); t = np.int64(0)
    for e in range(n):
        out[o] = hdr[4 * e]; out[o + 1] = hdr[4 * e + 1]; out[o + 2] = hdr[4 * e + 2]; out[o + 3] = hdr[4 * e + 3]
        sl = np.int64(hdr[4 * e + 2]) | (np.int64(hdr[4 * e + 3]) << 8)
        o += 4
        for q in range(sl):
            out[o + q] = text[t + q]
        o += sl; t += sl
    assert t == text.size, 'fc3_join: text not consumed exactly'
    return o


@njit(inline='always')
def _pc64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(inline='always')
def _ones(w, a, b):
    """set bits of the mask between bit a (inclusive) and bit b (exclusive), 64 at a time"""
    if b <= a:
        return np.int64(0)
    wa = a >> 6; wb = (b - 1) >> 6
    lo = np.uint64(0xFFFFFFFFFFFFFFFF) << np.uint64(a & 63)
    hib = np.uint64(0xFFFFFFFFFFFFFFFF) >> np.uint64(63 - ((b - 1) & 63))
    if wa == wb:
        return _pc64(w[wa] & lo & hib)
    s = _pc64(w[wa] & lo) + _pc64(w[wb] & hib)
    for k in range(wa + 1, wb):
        s += _pc64(w[k])
    return s


@njit(nogil=True, cache=True)
def fc3_charlens(hdr16, w, R, out):
    """character length of every entry of one chunk from its headers and mask -- no text byte read.
    THE STRING AS PIECES: the previous string is a stack of pieces (where the piece starts in the
    string, where its bits start in the mask, characters before it). An entry keeps [0, cp): pieces
    starting at or after cp are dropped, the last kept piece is counted up to cp, the suffix is
    pushed. The stack empties at every restart, so it never holds more than R pieces."""
    n = hdr16.size // 2
    ps = np.zeros(R + 1, np.int64); pb = np.zeros(R + 1, np.int64); pc = np.zeros(R + 1, np.int64)
    top = 0; bit = np.int64(0)
    for e in range(n):
        cp = np.int64(hdr16[2 * e]); sl = np.int64(hdr16[2 * e + 1])
        if e % R == 0:
            top = 0
        while top > 0 and ps[top - 1] >= cp:
            top -= 1
        before = np.int64(0)
        if top > 0:
            k = top - 1
            before = pc[k] + _ones(w, pb[k], pb[k] + (cp - ps[k]))
        ps[top] = cp; pb[top] = bit; pc[top] = before; top += 1
        out[e] = before + _ones(w, bit, bit + sl)
        bit += sl
    return n


@njit(nogil=True, cache=True)
def plike_fc3(hdr16, text, R, n1, n2, keep):
    """plike_fc_serial on the split streams: keep[i] = entry i contains n1 (then n2 after it).
    Headers say where each suffix lands; the text is read in place. THE PREFIX CARRY: a match that
    ended inside the shared prefix is inherited -- only new bytes are searched."""
    n = hdr16.size // 2
    L1 = n1.shape[0]; L2 = n2.shape[0]
    prev = np.empty(131072, np.uint8)
    t = np.int64(0); mend = np.int64(-1)
    for e in range(n):
        cp = np.int64(hdr16[2 * e]); sl = np.int64(hdr16[2 * e + 1])
        if e % R == 0:
            mend = np.int64(-1)
        for q in range(sl):
            prev[cp + q] = text[t + q]
        t += sl
        plen = cp + sl
        p1 = np.int64(-1)
        if mend >= 0 and mend <= cp:
            p1 = mend
            i = plen
        else:
            i = cp - L1 + 1
            if i < 0:
                i = 0
        while i <= plen - L1:
            k = 0
            while k < L1 and prev[i + k] == n1[k]:
                k += 1
            if k == L1:
                p1 = i + L1
                break
            i += 1
        mend = p1
        ok = False
        if p1 >= 0:
            if L2 == 0:
                ok = True
            else:
                i = p1
                while i <= plen - L2:
                    k = 0
                    while k < L2 and prev[i + k] == n2[k]:
                        k += 1
                    if k == L2:
                        ok = True
                        break
                    i += 1
        keep[e] = ok
    return n


# ---------------------------------------------------------------------------------------------
# IDENTIFICATION AT THE SURVIVORS (Jackson, 2026-09-23: "if we know that what we are going to do
# will cost like 500ms but there is a filter that is 48ms that cuts it in half then we need to run
# the filter before it"). A filter that runs after others need not decide the whole dictionary:
# only the distinct codes still alive. Each needed entry is rebuilt from its restart (the prefix
# chain starts there) and decided alone; groups holding no needed code are never walked.

@njit(inline='always')
def _contains2(prev, plen, n1, n2):
    L1 = n1.shape[0]; L2 = n2.shape[0]
    i = 0
    while i <= plen - L1:
        k = 0
        while k < L1 and prev[i + k] == n1[k]:
            k += 1
        if k == L1:
            if L2 == 0:
                return True
            j = i + L1
            while j <= plen - L2:
                k = 0
                while k < L2 and prev[j + k] == n2[k]:
                    k += 1
                if k == L2:
                    return True
                j += 1
            return False
        i += 1
    return False


@njit(nogil=True, cache=True)
def plike_sel_fc3(hdr16, text, gto, R, need, n1, n2, out):
    """three streams: out[i] = chunk-local entry need[i] (sorted, unique) contains n1 (then n2).
    gto[g] = where restart group g's suffixes begin in the text. Returns entries walked."""
    prev = np.empty(131072, np.uint8)
    g_cur = np.int64(-1); e = np.int64(0); t = np.int64(0); plen = np.int64(0); walked = np.int64(0)
    for i in range(need.size):
        ne = np.int64(need[i]); g = ne // R
        if g != g_cur:
            g_cur = g; e = g * R; t = np.int64(gto[g])
        while e <= ne:
            cp = np.int64(hdr16[2 * e]); sl = np.int64(hdr16[2 * e + 1])
            for q in range(sl):
                prev[cp + q] = text[t + q]
            t += sl; plen = cp + sl; e += 1; walked += 1
        out[i] = _contains2(prev, plen, n1, n2)
    return walked


@njit(nogil=True, cache=True)
def plike_sel_fc(a, gro, R, need, n1, n2, out):
    """plike_sel_fc3 on an interleaved chunk: gro[g] = restart group g's byte offset in a"""
    prev = np.empty(131072, np.uint8)
    g_cur = np.int64(-1); e = np.int64(0); o = np.int64(0); plen = np.int64(0); walked = np.int64(0)
    for i in range(need.size):
        ne = np.int64(need[i]); g = ne // R
        if g != g_cur:
            g_cur = g; e = g * R; o = np.int64(gro[g])
        while e <= ne:
            cp = np.int64(a[o]) | (np.int64(a[o + 1]) << 8)
            sl = np.int64(a[o + 2]) | (np.int64(a[o + 3]) << 8)
            o += 4
            for q in range(sl):
                prev[cp + q] = a[o + q]
            o += sl; plen = cp + sl; e += 1; walked += 1
        out[i] = _contains2(prev, plen, n1, n2)
    return walked


# ---------------------------------------------------------------------------------------------
# THE HOST FROM THE HEADERS (Jackson, 2026-09-23: read per block by the most restrictive position
# -- in a sorted dictionary the leading positions of a block are one byte each, so the block's
# shared prefix decides its members at once). The host lives at fixed bytes after "://": while an
# entry's shared prefix reaches past the host's slash, its host is its predecessor's -- decided by
# its header alone; its suffix is only looked at for a newline ('.' never matches \n, so a newline
# after the slash breaks the match). Only a BREAK rebuilds the string, from a stack of pieces that
# point into the text (the previous string is never copied byte by byte).

@njit(inline='always')
def _fc3_fill(ps, pt, top, plen, text, prev):
    """rebuild prev[0:plen] from the pieces: piece k covers [ps[k], ps[k+1]) and reads text at pt[k]"""
    for k in range(top):
        a = ps[k]; b = ps[k + 1] if k + 1 < top else plen
        src = pt[k]
        for x in range(a, b):
            prev[x] = text[src + (x - a)]


@njit(nogil=True, cache=True)
def fc3_hostruns(hdr16, text, R, nl_free, brk, hend, labuf, laboff, meta):
    """fc_hostruns on the three streams, same law and same outputs (brk, hend, labels): a run
    continues while cp > W (the host's slash) and the suffix is newline-clean; nl_free = the chunk's
    text holds no newline at all, so no suffix is even looked at while a run continues."""
    n = hdr16.size // 2
    prev = np.zeros(131072, np.uint8)
    ps = np.zeros(R + 2, np.int64); pt = np.zeros(R + 2, np.int64); top = 0
    W = np.int64(-2); nr = np.int64(0); lw = np.int64(0); t = np.int64(0)
    for i in range(n):
        cp = np.int64(hdr16[2 * i]); sl = np.int64(hdr16[2 * i + 1])
        if i % R == 0:
            cp = np.int64(0); top = 0
        while top > 0 and ps[top - 1] >= cp:
            top -= 1
        ps[top] = cp; pt[top] = t; top += 1
        plen = cp + sl
        if W >= 0 and cp > W:
            clean = True
            if not nl_free:
                for q in range(sl):
                    if text[t + q] == 10:
                        clean = False
                        break
            if clean:
                brk[i] = 0; hend[i] = W
                t += sl
                continue
        t += sl
        brk[i] = 1
        _fc3_fill(ps, pt, top, plen, text, prev)
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
            if plen > hs + 4 and prev[hs] == 119 and prev[hs + 1] == 119 and prev[hs + 2] == 119 and prev[hs + 3] == 46:
                hs += 4
            j2 = hs
            while j2 < plen:
                if prev[j2] == 47:
                    he = j2
                    break
                j2 += 1
            if he <= hs and hs != hs0:
                hs = hs0
                j2 = hs
                he = np.int64(-1)
                while j2 < plen:
                    if prev[j2] == 47:
                        he = j2
                        break
                    j2 += 1
        bnl = np.int64(-1)
        if he > hs:
            j3 = he + 1
            while j3 < plen:
                if prev[j3] == 10:
                    bnl = j3
                    break
                j3 += 1
        if (he > hs) and (bnl < 0):
            hend[i] = he; W = he
            a0 = hs; b0 = he
        else:
            hend[i] = -1; W = np.int64(-2)
            a0 = np.int64(0); b0 = plen          # no match: the label is the whole string
        if lw + (b0 - a0) > labuf.size or nr + 1 >= laboff.size:
            meta[0] = -1
            return nr
        laboff[nr] = lw
        for q in range(b0 - a0):
            labuf[lw + q] = prev[a0 + q]
        lw += b0 - a0
        nr += 1
    laboff[nr] = lw
    meta[0] = n
    return nr


# ---------------------------------------------------------------------------------------------
# tag 20 = THE BACK-REFERENCE (Jackson, 2026-09-24): an operator-declared HASH column. Per block of
# BR rows (blocks start on a byte; a start table), per row: bit 0 = flag. Flag 0: the code's `bits`
# bits follow. Flag 1: a 4-bit class k, then the k low bits of the gap back to the previous copy of
# the same code INSIDE THE BLOCK (the gap has k+1 bits, its leading 1 implicit). A block never
# reads another; any row is one jump to its block plus a walk to the row. One 8-byte load decodes
# one row (the widest row is 1 + 32 bits after at most 7 bits of shift).
# ---------------------------------------------------------------------------------------------
@njit(cache=True, inline='always')
def _e20_load(buf, byte):
    w = np.uint64(0)
    for k in range(8):
        w |= np.uint64(buf[byte + k]) << np.uint64(8 * k)
    return w


@njit(cache=True, parallel=True, nogil=True)
def e20_write(x, gap, BR, bits, boff, base, buf):
    nb = boff.size - 1
    for b in prange(nb):
        p = (base + boff[b]) * 8
        lo = b * BR
        hi = min(x.size, lo + BR)
        for r in range(lo, hi):
            g = gap[r]
            if g <= 0:
                v = np.int64(x[r]) << 1
                n = 1 + bits
            else:
                k = 0
                while (g >> (k + 1)) > 0:
                    k += 1
                v = 1 | (k << 1) | ((g & ((1 << k) - 1)) << 5)
                n = 5 + k
            for i in range(n):
                if (v >> i) & 1:
                    q = p + i
                    buf[q >> 3] |= np.uint8(1 << (q & 7))
            p += n


@njit(cache=True, inline='always')
def _e20_block(buf, p, n, bits, out, o0):
    vm = np.uint64((1 << bits) - 1)
    for i in range(n):
        w = _e20_load(buf, p >> 3) >> np.uint64(p & 7)
        if (w & np.uint64(1)) == 0:
            out[o0 + i] = (w >> np.uint64(1)) & vm
            p += 1 + bits
        else:
            k = np.int64((w >> np.uint64(1)) & np.uint64(15))
            g = (np.int64(1) << k) | np.int64((w >> np.uint64(5)) & np.uint64((1 << k) - 1))
            out[o0 + i] = out[o0 + i - g]
            p += 5 + k


@njit(cache=True, parallel=True, nogil=True)
def e20_decode(buf, base, BR, N, bits, boff, out):
    nb = boff.size - 1
    for b in prange(nb):
        n = min(BR, N - b * BR)
        _e20_block(buf, (base + boff[b]) * 8, n, bits, out, b * BR)


@njit(cache=True, parallel=True, nogil=True)
def e20_decode_blocks(buf, base, BR, N, bits, boff, b0, b1, out):
    """blocks b0 .. b1-1 only, block j of them written at out[(j) * BR]"""
    for j in prange(b1 - b0):
        b = b0 + j
        n = min(BR, N - b * BR)
        _e20_block(buf, (base + boff[b]) * 8, n, bits, out, j * BR)


@njit(cache=True, parallel=True, nogil=True)
def e20_gather(buf, base, BR, N, bits, boff, blocks, starts, rows, out):
    """rows sorted; starts[j]..starts[j+1] are the rows in blocks[j]. Each touched block is walked
    only as far as its last wanted row (gaps only point back)."""
    for j in prange(blocks.size):
        b = blocks[j]
        last = rows[starts[j + 1] - 1] - b * BR
        tmp = np.empty(last + 1, np.int64)
        _e20_block(buf, (base + boff[b]) * 8, last + 1, bits, tmp, 0)
        for i in range(starts[j], starts[j + 1]):
            out[i] = tmp[rows[i] - b * BR]
