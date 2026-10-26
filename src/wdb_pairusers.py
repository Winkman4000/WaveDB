"""wdb_pairusers -- THE PAIR USERS read (2026-10-25): Q11's shape (before pairdistinct, which takes what it declines).

    SELECT a, b, COUNT(DISTINCT u) AS n FROM t [WHERE a|b <> literal] GROUP BY a, b ORDER BY n DESC LIMIT k

Jackson's instant cases (the counts stop it early; a near-unique column is solved almost at once), the filter first:

  THE KEPT CASE -- the filter keeps few rows (the filter column's census: N minus the excluded value's rows, at most
  KEPT_LINE). The whole query is those rows: every pair's distinct users counted exactly over them.

  THE DENSE CASE -- any other pairs whose space Va x Vb is at most 2^24: rows per pair in one pass over a and b,
  then the pairs by rows, biggest first, 64 a round, distinct users counted, stopping at the first pair whose rows
  are at most the k-th best users (users <= rows). Defers to pairdistinct's narrow read where that is quicker.

  THE WIDE CASE (2026-10-26) -- pairs past the dense line. users(x, y) <= rows_kept(x, y) <= rows_kept(x) and
  <= rows_kept(y): 1. rows per value of each key over the kept rows (the kept census); 2. the top values of each
  side renumbered into a box (pair space 2^20, then 2^22, 2^24; every other value passed over like the excluded
  one) and the dense steps run inside it; 3. done when the k-th best users >= max(next_a, next_b), the most kept
  rows of a value left outside. THE SPARSE STEP: when the box the k-th best would need (values of a with more
  kept rows than it, times values of b with more) is past SPARSE_LINE, rows for every occurring pair instead --
  hashed into 2^12 parts, each counted in its own table, the pairs with more rows than the box's k-th kept -- and
  the dense walk over them (users <= rows). LocalEventTime x WindowClientHeight needed a box of 2.2 billion
  pairs; 1,926 pairs have more rows than its 10th best.

  THE SELF CASE -- u is one of the keys. A group fixes its user: every group holds exactly 1 distinct user, so
  any k groups that pass the filter are a legal top k (all tie at 1). Read the first rows until k groups.

  THE NEAR-UNIQUE CASE -- a key K whose N - V is tiny (WatchID: 99,997,497 rows, 99,997,493 values, 4 extra).
  users(x, y) <= rows(x, y) <= rows(x): only a value held by 2+ rows can make a group of 2+ users, and every other
  group is one row = one user. A mode-2 dictionary is exactly the column's values, so every code 0..V-1 holds a
  row: count rows per band of 2^12 codes and extra(band) = rows(band) - codes(band) is that band's repeated rows
  (the extras must add to N - V -- the check that every code is used). Only the bands with extra > 0 are read
  again; their codes held twice are the repeated values, counted exactly; k - (groups of 2+) one-row groups fill
  the rest.

A filter `key <> literal` names a group key, so it removes whole groups (that key's value) and leaves every other
group's count alone (group_distinct's rule). Lever: WDB_PAIRUSERS=0.
"""
import os
import numpy as np
import numba
from sqlglot import expressions as E
import wdb_sql
import wdb_policies as P
import wdb_groupdistinct as GD

ON = [os.environ.get('WDB_PAIRUSERS', '1') != '0']
BAND_SHIFT = 12            # bands of 2^12 codes: measured 40.7 ms on WatchID (2^16: 54.5, 2^10: 56.7, 2^8: 60.2)
NEAR_SHIFT = 10            # near unique: N - V <= N / 2^10 (PROVISIONAL: on this data only WatchID qualifies, extra 4;
                           # the line is to be measured against the hashed counts path once that exists)
FIRST_ROWS = 4096          # the first read for one-row / self groups; x8 until k groups
CENSUS_V = 1 << 20         # the filter column's census (rows per value) is read when its V is at most this
KEPT_LINE = 1 << 20        # THE KEPT CASE when the filter keeps at most this many rows. Measured, the exact count over
                           # the kept rows vs DuckDB: 177,085 kept 0.32x (x Age) / 0.40x (x Title); 333,781 0.60 /
                           # 0.29; 460,217 0.91 / 1.07; 630,500 0.92 / 0.42; 1,561,991 5.1 / 1.29; 5,563,212 3.9 /
                           # 1.93 -- the line sits between 630,500 and 1,561,991 (nothing measured in between)
_HITS = [0, 0, 0, 0, 0, 0] # telemetry: self, near-unique, kept, dense, wide box, wide sparse answers
WIDE_ON = [os.environ.get('WDB_PAIRUSERS_WIDE', '1') != '0']
WIDE_BOX = (1 << 20, 1 << 22, 1 << 24)   # the box's pair space, growing. Measured on 26 wide Q11 holes: 21 stopped
                                         # in 2^20, 1 in 2^22 (Referer x ClientEventTime)
SPARSE_LINE = 1 << 24     # THE SPARSE STEP when the box the k-th best needs is past this (or the last box fails)
SPARSE_PARTS = 12         # 2^12 hash parts: a part's table stays small


def detect(seg, tree, col_map):
    if not ON[0]:
        return None
    if not (P.no_joins(tree) and P.no_select_distinct(tree) and P.no_having(tree)):
        return None
    if tree.args.get('offset') is not None:
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 2 or not all(isinstance(x, E.Column) for x in g.expressions):
        return None
    gl = [x.name for x in g.expressions]
    if gl[0] == gl[1]:
        return None
    proj = tree.expressions
    if len(proj) != 3:
        return None
    di = GD._distinct_index(proj)
    if di is None:
        return None
    ci, tname = di
    bare = []
    for i, p in enumerate(proj):
        if i == ci:
            continue
        if wdb_sql._agg_kind(p) is not None:
            return None
        nm = wdb_sql._proj_colname(p)
        if nm is None:
            return None
        bare.append(nm)
    if sorted(bare) != sorted(gl):
        return None
    order = tree.args.get('order')
    if order is None or len(order.expressions) != 1 or not GD._order_is_distinct_desc(tree, proj, ci):
        return None                              # a second ORDER BY key would ask for one order among the ties
    lx = tree.args.get('limit')
    try:
        k = int(lx.expression.this)
    except Exception:
        return None
    if k <= 0:
        return None
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    excl = None
    w = tree.args.get('where')
    if w is not None:
        n9 = w.this
        if not (isinstance(n9, E.NEQ) and isinstance(n9.this, E.Column) and n9.this.name in gl
                and GD._is_literal(n9.expression)):
            return None
        excl = (sc(n9.this.name), n9.expression)
    a, b, u = sc(gl[0]), sc(gl[1]), sc(tname)
    if not P.columns_exist(seg, a, b, u) or not P.no_deleted_rows(seg):
        return None
    for c9 in (a, b, u):
        if not P.key_not_nullable(seg, c9) or not P.not_positional(seg, c9):
            return None
    N = int(seg.N)
    if u in (a, b):
        case = ('self', None)
    else:
        best = None
        for kc in (a, b):
            cm = seg.cols[kc]
            V = int(cm.get('V') or 0)
            if cm.get('mode') != 2 or V <= 0 or int(cm.get('n_dict') or -1) != V:
                continue
            extra = N - V
            if 0 <= extra <= (N >> NEAR_SHIFT) and (best is None or extra < best[1]):
                best = (kc, extra)
        if best is not None:
            case = ('near', best[0])
        else:
            keptable = excl is not None and (int(seg.cols[excl[0]]['V']) <= CENSUS_V
                                             or seg.cols[excl[0]].get('code_enc') in (8, 9))
            dense = int(seg.cols[a]['V']) * int(seg.cols[b]['V']) <= DENSE_CASE
            if not (keptable or dense or WIDE_ON[0]):
                return None
            case = ('ord', dense)                # THE KEPT CASE when the filter keeps few rows, else THE DENSE CASE
    pcols = [None if i == ci else sc(wdb_sql._proj_colname(p)) for i, p in enumerate(proj)]
    defer = False
    if case[0] == 'ord' and excl is not None and seg.cols[excl[0]].get('code_enc') in (8, 9):
        import wdb_pairdistinct as _PD             # its narrow read takes this shape quicker (_defer_narrow)
        defer = _PD.detect(seg, tree, col_map) is not None
    return {'a': a, 'b': b, 'u': u, 'k': k, 'excl': excl, 'case': case, 'ci': ci, 'proj': proj, 'pcols': pcols,
            'defer': defer}


def _excl_code(seg, spec):
    """(column, code) the filter removes; code None when no row holds the literal (it removes nothing)"""
    if spec['excl'] is None:
        return None, None
    col, lit = spec['excl']
    return col, GD._excl_code(seg, col, lit)


PASS_BATCH = 65536         # past the first rows, the filter's passing rows are read in batches this size


def _row_batches(seg, xcol, xcode):
    """THE FILTER FIRST (Jackson's ordering): batches of candidate rows, ascending. The first FIRST_ROWS rows
    as they are (the caller drops the filter's value); past them, with a filter, ONLY the rows that pass it --
    wherescan's <> scan, frame by frame -- because a filter that removes a tiny column's common value
    (IsOldCounter <> 0) leaves rows the first reads would wade through: growing range reads cost 0.3-0.4 s
    there, DuckDB 0.04. Without a filter, growing ranges."""
    import wdb_wherescan as WS
    N = int(seg.N)
    lo, n9 = 0, 64                               # 64 rows usually hold k groups: codes_at on 4,096 rows cost
    while lo < min(N, FIRST_ROWS):               # 107 ms on MobilePhoneModel (64 rows: ~1 ms)
        hi = min(N, FIRST_ROWS, lo + n9)
        yield np.arange(lo, hi, dtype=np.int64)
        lo = hi; n9 *= 8
    if lo >= N:
        return
    if xcode is None:
        n9 = FIRST_ROWS * 8
        while lo < N:
            hi = min(N, lo + n9)
            yield np.arange(lo, hi, dtype=np.int64)
            lo = hi; n9 *= 8
        return
    pos = np.asarray(WS._scan_eq(seg, xcol, int(xcode), lo, N, negate=True), np.int64)
    for i in range(0, pos.size, PASS_BATCH):
        yield pos[i:i + PASS_BATCH]


def _first_groups(seg, a, b, k, xcol, xcode):
    """up to k (code a, code b) groups among the rows that pass the filter (THE SELF CASE: each holds one user)"""
    Vb = int(seg.cols[b]['V'])
    found, seen = [], set()
    for rows in _row_batches(seg, xcol, xcode):
        if len(found) >= k:
            break
        if not rows.size:
            continue
        ca = np.asarray(seg.codes_at(a, rows)).astype(np.int64)
        cb = np.asarray(seg.codes_at(b, rows)).astype(np.int64)
        if xcode is not None:
            m = (ca if xcol == a else cb) != xcode
            ca, cb = ca[m], cb[m]
        for x in np.unique(ca * Vb + cb).tolist():
            pr = (x // Vb, x % Vb)
            if pr not in seen:
                seen.add(pr); found.append(pr)
                if len(found) >= k:
                    break
    return found


def _near(seg, spec, xcol, xcode):
    """[(users, code a, code b)] best first: the repeated values' groups counted exactly, then one-row groups"""
    import wdb_kernels as WK
    a, b, u, k = spec['a'], spec['b'], spec['u'], spec['k']
    K = spec['case'][1]; O = b if K == a else a
    N = int(seg.N); VK = int(seg.cols[K]['V'])
    ck = np.ascontiguousarray(np.asarray(seg._raw_codes(K)))
    sh = BAND_SHIFT
    B = ((VK - 1) >> sh) + 1
    T = numba.get_num_threads()
    h = WK.band_hist(ck, B, sh, T)
    size = np.full(B, 1 << sh, np.int64)
    size[-1] = VK - ((B - 1) << sh)
    ext = h - size
    if (ext < 0).any() or int(ext.sum()) != N - VK:
        return None                              # some code holds no row: the band count cannot see repeats
    rows = WK.band_rows(ck, ext > 0, sh, T)
    cc = ck[rows].astype(np.int64)
    uq, inv, cnt = np.unique(cc, return_inverse=True, return_counts=True)
    rr = rows[cnt[inv] >= 2]
    repc = uq[cnt >= 2]
    board = []
    if rr.size:
        kc = ck[rr].astype(np.int64)
        oc = np.asarray(seg.codes_at(O, rr)).astype(np.int64)
        uc = np.asarray(seg.codes_at(u, rr)).astype(np.int64)
        if xcode is not None:
            m = (kc if xcol == K else oc) != xcode
            kc, oc, uc = kc[m], oc[m], uc[m]
        if kc.size:
            trip = np.unique(np.stack([kc, oc, uc], 1), axis=0)
            pairs, users = np.unique(trip[:, :2], axis=0, return_counts=True)
            for (x, y), n in zip(pairs.tolist(), users.tolist()):
                board.append((int(n), x, y) if K == a else (int(n), y, x))
    board.sort(key=lambda t: -t[0])
    board = board[:k]
    if len(board) < k:
        # THE ONE-ROW FILL: K's codes are in hand, so the candidates are the first rows whose K value is held
        # once; the other key is read at those rows only (codes_at -- a range read decodes far more: enc 5's
        # stream cost 43 ms for 4,096 rows). Each such row is its own group of one user.
        board += _one_row_groups(seg, ck, repc, K, O, a, k - len(board), xcol, xcode)
    _HITS[1] += 1
    return board


def _one_row_groups(seg, ck, repc, K, O, a, need, xcol, xcode):
    out = []
    for rows in _row_batches(seg, xcol, xcode):
        if len(out) >= need:
            break
        if not rows.size:
            continue
        kc = ck[rows].astype(np.int64)
        m = ~np.isin(kc, repc) if repc.size else np.ones(kc.size, np.bool_)
        rows, kc = rows[m], kc[m]
        if not rows.size:
            continue
        oc = np.asarray(seg.codes_at(O, rows)).astype(np.int64)
        if xcode is not None:
            m = (kc if xcol == K else oc) != xcode
            kc, oc = kc[m], oc[m]
        for x, y in zip(kc.tolist(), oc.tolist()):      # K held once: every row a distinct group
            out.append((1, x, y) if K == a else (1, y, x))
            if len(out) >= need:
                break
    return out


def _kept_count(seg, xcol, xcode):
    """rows the filter keeps. THE CHEAP KEPT COUNT (2026-10-25): a sparse column (enc 8 / 9) excluding its
    default value keeps exactly its stored non-default rows (e8n / e9n: MobilePhoneModel <> '' keeps 5,563,212 --
    its census cost 21 ms a query, 209 the first); excluding any other of its values, None (not worth the census).
    Otherwise the census (enc 2 / 10: under 1 ms; enc 3: 10-15), None past CENSUS_V."""
    c = seg.cols[xcol]
    if xcode is None:
        return int(seg.N)
    enc = c.get('code_enc')
    if enc in (8, 9):
        d9, n9 = c.get('e%dd' % enc), c.get('e%dn' % enc)
        if d9 is not None and n9 is not None and int(d9) == int(xcode):
            return int(n9)
        return None
    V = int(c['V'])
    if V > CENSUS_V:
        return None
    cen = np.asarray(seg.raw_census(xcol, V))
    return int(seg.N) - int(cen[xcode])


def _kept(seg, spec, xcol, xcode):
    """THE KEPT CASE (the filter first, Jackson's ordering): the filter keeps few rows, so the whole query is those
    rows -- every pair's distinct users counted exactly over them. The pairs are numbered densely first, so (pair,
    user) is one int64 whatever the keys' widths (a row-wise unique of (a, b, u) cost 0.24-0.42 s with Title)."""
    import wdb_wherescan as WS
    a, b, u, k = spec['a'], spec['b'], spec['u'], spec['k']
    pos = np.asarray(WS._scan_eq(seg, xcol, int(xcode), 0, int(seg.N), negate=True), np.int64)
    _HITS[2] += 1
    if not pos.size:
        return []
    ka = np.asarray(seg.codes_at(a, pos)).astype(np.int64)
    kb = np.asarray(seg.codes_at(b, pos)).astype(np.int64)
    ku = np.asarray(seg.codes_at(u, pos)).astype(np.int64)
    Va, Vb, Vu = int(seg.cols[a]['V']), int(seg.cols[b]['V']), int(seg.cols[u]['V'])
    return _stop_early(ka, kb, ku, Va, Vb, Vu, k)


DENSE_PAIRS = 1 << 24      # rows per pair by bincount over a x Vb + b when Va x Vb is at most this; past it, one unique


def _stop_early(ka, kb, ku, Va, Vb, Vu, k):
    """JACKSON'S STOP-EARLY RULE over the kept rows. 1. rows(pair) for every pair (a bincount when the pair space
    is small, else one unique). 2. users(pair) <= rows(pair). 3. walk the pairs by rows, biggest first, a batch at
    a time: count each batch's distinct users over that batch's rows only. 4. stop at the first pair whose rows are
    at most the k-th best users found: it and every pair after it hold at most that many users (a tie at the cut
    is legal either way). Returns [(users, code a, code b)] best first."""
    if Va * Vb <= DENSE_PAIRS:
        pk = ka * Vb + kb
        rows_per = np.bincount(pk, minlength=Va * Vb)
        live = np.flatnonzero(rows_per)
        order = live[np.argsort(-rows_per[live], kind='stable')]
        keyof = None
    else:
        keyof, pk = np.unique(ka * Vb + kb, return_inverse=True)
        pk = pk.astype(np.int64)
        rows_per = np.bincount(pk)
        order = np.argsort(-rows_per, kind='stable')
    lut = np.full(rows_per.size, -1, np.int32)
    board = []                                   # (users, pair)
    kth = 0
    i, batch = 0, max(4 * k, 64)
    while i < order.size:
        take = order[i:i + batch]
        if len(board) >= k:
            take = take[rows_per[take] > kth]    # a pair with rows <= kth cannot get in
        if not take.size:
            break
        lut[take] = np.arange(take.size, dtype=np.int32)
        li = lut[pk]
        sel = li >= 0
        uu = np.unique(li[sel].astype(np.int64) * Vu + ku[sel])
        users = np.bincount(uu // Vu, minlength=take.size)
        lut[take] = -1
        board += list(zip(users.tolist(), take.tolist()))
        board.sort(key=lambda t: -t[0])
        board = board[:k]
        kth = board[-1][0] if len(board) >= k else 0
        i += batch
        batch *= 4
        if i < order.size and len(board) >= k and rows_per[order[i]] <= kth:
            break                                # THE STOP: nothing after this can climb past the k-th
    out = []
    for n9, p9 in board:
        key = int(p9) if keyof is None else int(keyof[p9])
        out.append((int(n9), key // Vb, key % Vb))
    return out


DENSE_CASE = 1 << 24       # THE DENSE CASE: a pair space Va x Vb up to this. Measured on 14 Q11 holes, 2 to 16,104,704
                           # pairs: every answer DuckDB's, 0.18-1.69x DuckDB (the largest: T count tables of K int32)
_DENSE_PAIRS_ROUND = 64    # candidates a round: one bit each in a u64 per user


def _defer_narrow(seg, spec, xcol, xcode):
    """pairdistinct's narrow read -- a sparse filter column excluding its default, so only the kept rows are read --
    beats the dense case's three whole decodes: Q11 itself 55 ms vs 136, JavaEnable x MobilePhoneModel 59 vs 121.
    (A dense filter column goes the other way: UTMSource-filtered pairs 122-184 ms dense vs 184-274.)"""
    if not spec.get('defer') or xcode is None:
        return False
    c = seg.cols[xcol]
    enc = c.get('code_enc')
    return enc in (8, 9) and int(c.get('e%dd' % enc, -1)) == int(xcode)


def _dense(seg, spec, xcol, xcode):
    """THE DENSE CASE -- Jackson's stop-early steps over every row, the filter's excluded rows passed over (the
    exceptions road: the kept rows are never listed). 2. rows per pair: slot = a x Vb + b, one pass over a and b
    (wdb_kernels.pair_rows: each thread's table on its own cache lines; no slot array -- writing one cost 83-91 ms
    a query). 3. pairs by rows, biggest first, 64 a round: distinct users per pair, the slot computed again from
    a and b (wdb_kernels.hunt64: rows bucketed by user lane, a u64 of pair bits per user). 4. stop at the first
    pair whose rows are at most the k-th best users."""
    import wdb_kernels as WK
    a, b, u, k = spec['a'], spec['b'], spec['u'], spec['k']
    Va, Vb, Vu = int(seg.cols[a]['V']), int(seg.cols[b]['V']), int(seg.cols[u]['V'])
    K = Va * Vb
    which = 1 if xcol == b else 0
    xc = -1 if xcode is None else int(xcode)
    ca = np.asarray(seg._raw_codes(a))
    cb = np.asarray(seg._raw_codes(b))
    cu = np.asarray(seg._raw_codes(u))
    T = numba.get_num_threads()
    SH = max(1, int(Vu - 1).bit_length() - 7)       # 128 user lanes: a lane's table 2^SH u64
    NL = ((Vu - 1) >> SH) + 1
    cnt = WK.pair_rows(ca, cb, which, xc, Vb, K, T)
    live = np.flatnonzero(cnt)
    order = live[np.argsort(-cnt[live], kind='stable')]
    lut = np.full(K, -1, np.int8)
    board, kth, idx = [], 0, 0
    while idx < order.size:
        take = []
        while idx < order.size and (len(board) < k or cnt[order[idx]] > kth):
            take.append(int(order[idx])); idx += 1
            if len(take) >= _DENSE_PAIRS_ROUND:
                break
        if not take:
            break
        tk = np.asarray(take, np.int64)
        lut[tk] = np.arange(tk.size, dtype=np.int8)
        dc = WK.hunt64(ca, cb, which, xc, Vb, cu, lut, tk.size, SH, NL, T)
        lut[tk] = -1
        board += [(int(dc[j]), int(tk[j])) for j in range(tk.size)]
        board.sort(key=lambda t: -t[0])
        board = board[:k]
        kth = board[-1][0] if len(board) >= k else 0
        if idx < order.size and len(board) >= k and kth >= cnt[order[idx]]:
            break                                # THE STOP: nothing after this can climb past the k-th
    _HITS[3] += 1
    return [(n9, p9 // Vb, p9 % Vb) for n9, p9 in board]


def _walk(rows, order, k, hunt):
    """the dense walk: pairs by rows, biggest first, 64 a round (hunt(take) -> users each); stop at the first pair
    whose rows are at most the k-th best users. Returns board [(users, pair)], kth, idx (pairs walked)"""
    board, kth, idx = [], 0, 0
    while idx < order.size:
        take = []
        while idx < order.size and (len(board) < k or rows[order[idx]] > kth):
            take.append(int(order[idx])); idx += 1
            if len(take) >= _DENSE_PAIRS_ROUND:
                break
        if not take:
            break
        dc = hunt(take)
        board += [(int(dc[j]), take[j]) for j in range(len(take))]
        board.sort(key=lambda t: -t[0])
        board = board[:k]
        kth = board[-1][0] if len(board) >= k else 0
        if idx < order.size and len(board) >= k and kth >= rows[order[idx]]:
            break                                # THE STOP: nothing after this can climb past the k-th
    return board, kth, idx


def _kept_census(seg, c, codes, xcol, fcodes, xcode):
    """rows per value of c over the rows the filter keeps: the census (the filter on c: its excluded value zeroed;
    no filter: as is), else one masked pass (wdb_kernels.hist_kept)"""
    V = int(seg.cols[c]['V'])
    if xcol is None or xcode is None:
        return np.asarray(seg.raw_census(c, V), np.int64)[:V]
    if xcol == c:
        cen = np.array(np.asarray(seg.raw_census(c, V))[:V], np.int64)
        cen[int(xcode)] = 0
        return cen
    import wdb_kernels as WK
    return WK.hist_kept(codes, fcodes, int(xcode), V, numba.get_num_threads())


def _box(sa, sb, cap):
    """Ma x Mb <= cap (Ma a power of 2) with the smallest outside bound max(next_a, next_b); sa / sb: each side's
    kept rows, biggest first, values with none left out"""
    best = None
    Ma = 1
    while True:
        Ma = min(Ma, sa.size)
        Mb = min(sb.size, cap // Ma)
        if Mb >= 1:
            na = int(sa[Ma]) if Ma < sa.size else 0
            nb = int(sb[Mb]) if Mb < sb.size else 0
            bnd = max(na, nb)
            if best is None or bnd < best[0]:
                best = (bnd, Ma, Mb)
        if Ma >= sa.size or Ma >= cap:
            break
        Ma *= 2
    return best


_M64 = (1 << 64) - 1


def _wide(seg, spec, xcol, xcode):
    """THE WIDE CASE (Jackson's steps): the kept census of each key; the top values renumbered into a box and the
    dense steps run inside it; done when the k-th best >= max(next_a, next_b) (a pair outside the box has a value
    outside it: users <= that value's kept rows); a bigger box, or THE SPARSE STEP past SPARSE_LINE.
    Returns [(users, code a, code b)], or None (declined)"""
    import wdb_kernels as WK
    a, b, u, k = spec['a'], spec['b'], spec['u'], spec['k']
    Va, Vb, Vu = int(seg.cols[a]['V']), int(seg.cols[b]['V']), int(seg.cols[u]['V'])
    which = 1 if xcol == b else 0
    xc = -1 if xcode is None else int(xcode)
    ca = np.asarray(seg._raw_codes(a))
    cb = np.asarray(seg._raw_codes(b))
    cu = np.asarray(seg._raw_codes(u))
    T = numba.get_num_threads()
    SH = max(1, int(Vu - 1).bit_length() - 7)
    NL = ((Vu - 1) >> SH) + 1
    fc = ca if xcol == a else cb
    cena = _kept_census(seg, a, ca, xcol, fc, xcode)
    cenb = _kept_census(seg, b, cb, xcol, fc, xcode)
    oa = np.argsort(-cena, kind='stable'); sa = cena[oa]; sa = sa[:int(np.count_nonzero(sa))]
    ob = np.argsort(-cenb, kind='stable'); sb = cenb[ob]; sb = sb[:int(np.count_nonzero(sb))]
    if not sa.size or not sb.size:
        return []                                # no row passes the filter
    for cap in WIDE_BOX:
        bnd, Ma, Mb = _box(sa, sb, cap)
        ma = np.full(Va, -1, np.int32); ma[oa[:Ma]] = np.arange(Ma, dtype=np.int32)
        mb = np.full(Vb, -1, np.int32); mb[ob[:Mb]] = np.arange(Mb, dtype=np.int32)
        K = Ma * Mb
        cnt = WK.pair_rows_map(ca, cb, ma, mb, which, xc, Mb, K, T)
        live = np.flatnonzero(cnt)
        order = live[np.argsort(-cnt[live], kind='stable')]
        lut = np.full(K, -1, np.int8)

        def hunt(take):
            tk = np.asarray(take, np.int64)
            lut[tk] = np.arange(tk.size, dtype=np.int8)
            dc = WK.hunt64_map(ca, cb, ma, mb, which, xc, Mb, cu, lut, tk.size, SH, NL, T)
            lut[tk] = -1
            return dc
        board, kth, _ = _walk(cnt, order, k, hunt)
        if bnd == 0 or (len(board) >= k and bnd <= kth):
            _HITS[4] += 1                        # every value with kept rows inside, or the outside bound met
            return [(n9, int(oa[p9 // Mb]), int(ob[p9 % Mb])) for n9, p9 in board]
        need = int(np.count_nonzero(sa > kth)) * int(np.count_nonzero(sb > kth))
        if need > SPARSE_LINE or cap == WIDE_BOX[-1]:
            return _sparse(ca, cb, cu, which, xc, Va, Vb, max(kth, 1), k, SH, NL, T)
    return None


def _sparse(ca, cb, cu, which, xc, Va, Vb, F, k, SH, NL, T):
    """THE SPARSE STEP: rows for every occurring pair over the kept rows (hashed into 2^SPARSE_PARTS parts, each
    counted in its own table), the pairs with more than F rows kept (F: the box's k-th best, which the final k-th
    can only pass); the dense walk over them (users <= rows). Every pair not kept has rows <= F: declined (None)
    unless the walk stops, or ends with the k-th best >= F"""
    import wdb_kernels as WK
    buf, offs, nem = WK.sparse_rows(ca, cb, which, xc, Vb, F, SPARSE_PARTS, T)
    hk, hc = WK.sparse_gather(buf, offs, nem)
    del buf
    o = np.argsort(-hc, kind='stable'); hk = hk[o]; hc = hc[o]
    fa = np.zeros(Va, np.int8)
    C2 = 0xC2B2AE3D27D4EB4F                       # wdb_kernels._WIDE_C2: the same 128-slot hash

    def hunt(take):
        tk = np.full(128, -1, np.int64); tv = np.zeros(128, np.int8)
        for j, q in enumerate(take):
            key = int(hk[q]); h = ((key * C2) & _M64) >> 57
            while tk[h] >= 0:
                h = (h + 1) & 127
            tk[h] = key; tv[h] = j
            fa[key // Vb] = 1
        dc = WK.hunt64_sparse(ca, cb, which, xc, Vb, cu, fa, tk, tv, len(take), SH, NL, T)
        for q in take:
            fa[int(hk[q]) // Vb] = 0
        return dc
    board, kth, idx = _walk(hc, np.arange(hk.size), k, hunt)
    if len(board) < k or (idx >= hk.size and kth < F):
        return None
    _HITS[5] += 1
    return [(n9, int(hk[p9]) // Vb, int(hk[p9]) % Vb) for n9, p9 in board]


def execute(seg, spec):
    a, b, k = spec['a'], spec['b'], spec['k']
    xcol, xcode = _excl_code(seg, spec)
    # the self case reads the filter's passing rows itself (_row_batches) and needs no census: MobilePhoneModel's
    # cost 40 ms a query where the self answer takes 1.3
    kc = (_kept_count(seg, xcol, xcode) if (xcol is not None and xcode is not None and spec['case'][0] != 'self')
          else None)
    if xcol is not None and xcode is not None and int(seg.cols[xcol]['V']) == 1:
        board = []                               # the filter removes the column's only value: no rows pass
    elif kc is not None and kc <= KEPT_LINE:
        board = _kept(seg, spec, xcol, xcode)
    elif spec['case'][0] == 'ord':
        if _defer_narrow(seg, spec, xcol, xcode):
            return None                          # pairdistinct's narrow read is quicker
        if spec['case'][1]:
            board = _dense(seg, spec, xcol, xcode)
        else:
            board = _wide(seg, spec, xcol, xcode) if WIDE_ON[0] else None
            if board is None:
                return None
    elif spec['case'][0] == 'self':
        board = [(1, x, y) for x, y in _first_groups(seg, a, b, k, xcol, xcode)]
        _HITS[0] += 1
    else:
        board = _near(seg, spec, xcol, xcode)
        if board is None:
            return None
    names = [wdb_sql._alias(p) for p in spec['proj']]
    va = GD._keyvals(seg, a, np.asarray([t[1] for t in board], np.int64)) if board else []
    vb = GD._keyvals(seg, b, np.asarray([t[2] for t in board], np.int64)) if board else []
    pc = spec['pcols']
    out = []
    for i, t in enumerate(board):
        out.append(tuple(t[0] if c9 is None else (va[i] if c9 == a else vb[i]) for c9 in pc))
    return out, names
