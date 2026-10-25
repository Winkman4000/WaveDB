"""wdb_pairusers -- THE PAIR USERS read (2026-10-25): Q11's shape on keys pairdistinct does not take.

    SELECT a, b, COUNT(DISTINCT u) AS n FROM t [WHERE a|b <> literal] GROUP BY a, b ORDER BY n DESC LIMIT k

Jackson's instant cases (the counts stop it early; a near-unique column is solved almost at once), the filter first:

  THE KEPT CASE -- the filter keeps few rows (the filter column's census: N minus the excluded value's rows, at most
  KEPT_LINE). The whole query is those rows: every pair's distinct users counted exactly over them.

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
_HITS = [0, 0, 0]          # telemetry: self answers, near-unique answers, kept answers


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
        elif excl is not None and int(seg.cols[excl[0]]['V']) <= CENSUS_V:
            case = ('kept', None)                # THE KEPT CASE alone: execute counts the kept rows (census)
        else:
            return None
    pcols = [None if i == ci else sc(wdb_sql._proj_colname(p)) for i, p in enumerate(proj)]
    return {'a': a, 'b': b, 'u': u, 'k': k, 'excl': excl, 'case': case, 'ci': ci, 'proj': proj, 'pcols': pcols}


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
    """rows the filter keeps, from the filter column's census (None when its V is past CENSUS_V)"""
    V = int(seg.cols[xcol]['V'])
    if V > CENSUS_V:
        return None
    if xcode is None:
        return int(seg.N)
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
    elif spec['case'][0] == 'kept':
        return None                              # the filter keeps too many rows and no other case applies
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
