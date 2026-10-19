"""
wdb_groupdistinct — group-wise exact COUNT(DISTINCT) via single-pass code hashing.

  SELECT key, COUNT(DISTINCT target) FROM t GROUP BY key [ORDER BY <count> DESC] [LIMIT N]

This is the non-foldable aggregate — you can't merge partial counts, you must remember every distinct
thing seen. WaveDB's edge: dictionary codes are distinctness-preserving, so distinct(target) ==
distinct(target_codes) with NO value decode. We pack (group_code, target_code) into one int64 and count
distinct pairs per group in a single pass using an open-addressing hash set: one probe per row, and a
re-seen pair is a no-op. Flips the group-wise-distinct class (the ClickBench Q08/Q09/Q13 shape) from a
~150s scalar fallback to sub-second, single-threaded and exact.

Same contract as the other operators: try_groupdistinct(seg, tree, col_map) -> (rows, colnames) | None
(None => caller falls through to the scan paths). Fail-closed on anything outside the exact shape.

Scope: single segment, no HAVING/JOIN, projection exactly {bare key, COUNT(DISTINCT target)}, both key
and target value-identity (mode 0/1/2, never mode-4 positional codes), group key not nullable, ORDER BY the
distinct-count DESC (or absent), optional LIMIT; WHERE only `key <> literal` (the Q13 board, 2026-10-13:
any literal). The walk is picked by the two columns' profiles (execute): the sparse planes, the dense
boards, the target scatter, the key marker, the key sort -- and the survivor road when the filter removes
nearly every row. Co-aggregates live in wdb_groupmix.
"""
import numpy as np
from sqlglot import expressions as E
import wdb_sql
import workers
import wdb_gbcount
import wdb_policies as P
E = wdb_sql.E

_HITS = 0   # telemetry: queries answered by the group-distinct kernel
LANES = {}  # telemetry: lane -> answers (dense / scatter / keymark / keysort / None = numpy)
DENSE_CELLS = 1 << 22   # dense lane: (groups x targets) cells; per-thread boards cost T x cells bytes
MARK_MAX = 1 << 22      # key-mark lane: target space; per-thread marker tables cost T x targets x 4 bytes
SURVIVOR_ROAD = 64      # read the counted column at the survivors when the filter keeps <= 1/64 of the rows:
                        # a gather costs ~30 ns a row on the sparse encodings (MobilePhoneModel at IsMobile's
                        # 10%: 0.29 s gathered vs 0.08 s whole), so the road pays only when few rows survive
SURVIVOR_HITS = 0       # telemetry: answers that took the survivor road

try:
    from numba import njit

    @njit(cache=True)
    def _walk(grp, tgt, k, gmax, null_tgt, capbits):
        cap = 1 << capbits
        mask = cap - 1
        table = np.full(cap, -1, np.int64)
        counts = np.zeros(gmax, np.int64)
        for i in range(grp.shape[0]):
            tv = tgt[i]
            if tv == null_tgt:                       # COUNT(DISTINCT) ignores NULL
                continue
            key = grp[i] * k + tv                    # collision-free pair id (grp dense, tgt < k)
            h = (key * np.int64(0x9E3779B1)) & mask
            while True:
                slot = table[h]
                if slot == -1:                       # new pair -> mark + count it for its group
                    table[h] = key
                    counts[grp[i]] += 1
                    break
                if slot == key:                      # already seen ("standing twice") -> no-op
                    break
                h = (h + 1) & mask
        return counts
    _HAVE_NUMBA = True
except Exception:                                    # numba absent: NumPy pack+unique fallback (still exact)
    _HAVE_NUMBA = False


def idspace_sig(seg, col):
    """THE ID-SPACE BIRTHMARK: which id space a group column lives in today
    (raw values vs dict codes) and the dict's edges. A sidecar built under a
    different space must be rejected -- a July RegionID shelf built on raw
    values decoded today as codes produced right counts with WRONG labels
    (ClickBench Q08/Q09, 2026-09-04)."""
    r = _ids(seg, col)
    if r is None:
        return 'none'
    _ids_, _nc, kd = r
    V = int(seg.cols[col].get('V') or 0)
    if kd is None:
        return 'raw:%d' % V
    try:
        a = str(kd[0]) if V > 0 else ''
        b = str(kd[max(0, V - 2)]) if V > 1 else ''
    except Exception:
        a = b = '?'
    return 'dict:%d:%s:%s' % (V, a, b)


def _ids(seg, col):
    """Per-row value-identity integer ids for `col`, plus (null_code or -1) and an optional by-code
    decode array. Returns (ids:int64, null_code:int, decode_or_None) or None if not value-identity."""
    c = seg.cols[col]
    if c['mode'] == 4:
        return None
    V = wdb_gbcount._code_values(seg, col)               # dict-backed (string dict, or high-card int)
    if V is not None:
        ids = np.asarray(seg.codes(col)).astype(np.int64, copy=False)
        null_code = (c['V'] - 1) if c.get('has_null') else -1
        return ids, null_code, V
    if c.get('dt') == 0 and c['mode'] in (0, 1):         # raw small-range int (e.g. RegionID): id == value
        ids = np.asarray(seg.values(col)).astype(np.int64, copy=False)
        null_code = -1                                   # raw ints carry no dict null sentinel here
        return ids, null_code, None
    return None


def _distinct_index(proj):
    """(index, colname) of the single COUNT(DISTINCT col) projection, or None if not exactly one."""
    found = None
    for i, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Count) and isinstance(inner.this, E.Distinct):
            dx = inner.this.expressions
            if len(dx) != 1 or not isinstance(dx[0], E.Column):
                return None
            if found is not None:
                return None
            found = (i, dx[0].name)
    return found


def _order_is_distinct_desc(tree, proj, ci):
    """True iff ORDER BY is absent, or its primary key is the distinct-count projection, descending."""
    order = tree.args.get('order')
    if order is None or not order.expressions:
        return True                                      # no ORDER BY: any order is acceptable pre-LIMIT
    first = order.expressions[0]
    if not isinstance(first, E.Ordered) or not first.args.get('desc'):
        return False
    tgt = first.this
    alias = wdb_sql._alias(proj[ci])
    if isinstance(tgt, E.Column) and tgt.name == alias:
        return True
    inner = tgt.this if isinstance(tgt, E.Alias) else tgt
    return isinstance(inner, E.Count) and isinstance(inner.this, E.Distinct)


def _top_sel(counts, present, lim):
    """present groups ordered by count desc, code asc (stable), cut to lim -- via a PARTITION when
    lim is small against millions of groups (Q13: 6M SearchPhrases, two full argsorts = 82 ms).
    Exact to the full sort: the window is extended to the whole plateau of the lim-th count."""
    counts = np.asarray(counts)
    if lim is None or lim <= 0 or lim * 8 >= present.size:
        return present[np.argsort(-counts[present], kind='stable')] if lim is None else \
            present[np.argsort(-counts[present], kind='stable')][:lim]
    cp = counts[present]
    part = np.argpartition(-cp, lim - 1)[:lim]
    kth = int(cp[part].min())                             # the lim-th largest count
    cand = np.flatnonzero(cp >= kth)                      # everything that could be in the window
    cand = cand[np.argsort(-cp[cand], kind='stable')]
    return present[cand][:lim]


def _gdc_path(seg, kcol, tcol):
    return seg.path + '.%s__%s.gdc' % (kcol, tcol)


def _gdc_load(seg, kcol, tcol):
    """The grouped-distinct shelf (Jackson's cut): per-group COUNT(DISTINCT target) is a
    V-sized fact about an immutable file -- kilobytes on disk, reborn only with the file.
    Same law as the gbc: lawful V-sized sidecar, no_deleted_rows-gated, cold-truth clean."""
    import os, pickle
    p = _gdc_path(seg, kcol, tcol)
    if not os.path.exists(p):
        return None
    try:
        blob = pickle.load(open(p, 'rb'))
        if blob.get('n') == int(seg.N):
            return blob
    except Exception:
        pass
    return None


def birth2(seg, kcol, tcol):
    """Standalone big-key gdc birth (the ledger's midwife): planes -> scatter
    -> per-group distinct -> codes-only sidecar. Mirrors the lane's compute."""
    import numpy as np
    import wdb_kernels as _WK
    kc9 = seg.cols.get(kcol)
    if kc9 is None or kc9.get('code_enc') not in (8, 9):
        return False
    pl = seg.e8_planes(kcol)
    pos8, lits8 = np.asarray(pl[0]), np.asarray(pl[1], dtype=np.int64)
    KV9 = int(kc9['V'])
    cnts9 = np.bincount(lits8, minlength=KV9)
    uc9 = np.asarray(seg._raw_codes(tcol))[pos8].astype(np.int64)
    offs9 = np.zeros(KV9 + 1, np.int64)
    np.cumsum(cnts9, out=offs9[1:])
    bucketed9 = np.empty(lits8.size, np.int64)
    _WK.cd_scatter(lits8, uc9, offs9, offs9[:-1].copy(), bucketed9)
    dcounts = _WK.cd_alldistinct(bucketed9, offs9)
    _gdc_save2(seg, kcol, tcol, dcounts)
    return True


def _gdc_save2(seg, kcol, tcol, counts):
    """Codes-only sidecar for giant key spaces: counts indexed by dict code,
    winners decoded at emission. noempty: built from the sparse planes, so the
    default-value group is absent -- only filtered queries may ride it."""
    import pickle
    import wdb_sidecar as _wsc2, os as _os2
    if not _wsc2.births_on(_os2.path.dirname(seg.path)):
        return                                        # THE SWITCH, before the podium is even sorted
    # THE LEADERBOARD (Jackson's constraint chain): this shelf only ever
    # answers ORDER BY u DESC LIMIT k, so persist the podium, not the world.
    T = 65536
    counts = np.asarray(counts)
    if counts.size > T:
        part = np.argpartition(-counts, T - 1)[:T]
    else:
        part = np.arange(counts.size)
    order = part[np.argsort(-counts[part], kind='stable')]
    top_codes = order.astype(np.uint32)
    mx = int(counts[order[0]]) if order.size else 0
    dt = (np.uint8 if mx < 256 else
          np.uint16 if mx < 65536 else np.uint32)   # full ladder: the data climbs
                                                    # only as high as it measures
    top_counts = counts[order].astype(dt)        # the measured max picks the bus
    try:
        import wdb_sidecar, os as _os9
        if not wdb_sidecar.births_on(_os9.path.dirname(seg.path)): return   # THE SWITCH
        pickle.dump({'n': int(seg.N), 'top_codes': top_codes,
                     'top_counts': top_counts, 'keys': None, 'noempty': True},
                    open(_gdc_path(seg, kcol, tcol), 'wb'), protocol=4)
    except Exception:
        pass


def _gdc_save(seg, kcol, tcol, counts, keys):
    import pickle
    try:
        import wdb_sidecar, os as _os9
        if not wdb_sidecar.births_on(_os9.path.dirname(seg.path)): return   # THE SWITCH
        pickle.dump({'n': int(seg.N), 'counts': counts, 'keys': keys},
                    open(_gdc_path(seg, kcol, tcol), 'wb'), protocol=4)
    except Exception:
        pass


def birth_gdc(seg, kcol, tcol):
    """The dense-gdc midwife: replay's birther for recorded recipes -- the
    same scatter pass the lane runs on first touch, minus the query. A
    fresh successor file opens with its inherited shelves already born."""
    if _gdc_load(seg, kcol, tcol) is not None:
        return True                              # already on the shelf
    kinfo = _ids(seg, kcol)
    tinfo = _ids(seg, tcol)
    if kinfo is None or tinfo is None:
        return False
    grp, _knull, kdecode = kinfo
    tgt, tnull, _t = tinfo
    if grp.shape[0] != tgt.shape[0]:
        return False
    N = grp.shape[0]
    VRk = int(grp.max()) + 1 if N else 0
    VTt = int(tgt.max()) + 1 if N else 0
    if not (tnull < 0 and N > 4_000_000 and 0 < VRk <= 262_144 and VTt > 65_536):
        return False                             # the lane's own gates, honored
    import wdb_kernels as _WK
    SH = max(1, VTt.bit_length() - 12)
    ku, kr, offs = _WK.gd_pass1(np.ascontiguousarray(tgt),
                                np.ascontiguousarray(grp),
                                np.int64(SH), np.int64(8))
    counts = _WK.gd_pass2_count(ku, kr, offs, np.int64(SH), np.int64(VRk))
    keys = [wdb_sql._pyval(kdecode[g] if kdecode is not None else np.int64(g))
            for g in range(VRk)]
    keys = [k.decode('utf-8', 'replace') if isinstance(k, (bytes, bytearray)) else k
            for k in keys]
    _gdc_save(seg, kcol, tcol, counts, keys)
    return True


def _is_literal(node):
    """A plain literal, a negated number, or a typed literal (DATE '...', CAST('...' AS ...))."""
    if isinstance(node, E.Literal):
        return True
    if isinstance(node, E.Neg):
        return isinstance(node.this, E.Literal) and not node.this.is_string
    if isinstance(node, E.Cast):
        return isinstance(node.this, E.Literal)
    return False


def _excl_code(seg, kcol, lit):
    """The group code a `key <> literal` filter removes, or None when no row holds that value (then
    the filter removes nothing). Resolved through the column's own dictionary, then decoded back and
    compared, so a literal that only converts near the value never removes a neighbouring group."""
    import wdb_wherescan as _WS
    c = seg.cols[kcol]
    kind = 'f' if c.get('dt') == 2 else 'i'
    try:
        val = wdb_sql._lit_for_col(seg, kcol, lit, kind)
    except Exception:
        return None
    if isinstance(val, float) and c.get('dt') != 2:
        if val != int(val):
            return None                                  # 3.5 against an integer column: no row equals it
        val = int(val)
    def num(v):
        if isinstance(v, np.datetime64):
            return int(v.astype('datetime64[%s]' % seg.unit(kcol)).astype(np.int64))
        return v
    try:
        if c.get('mode') != 2 and c.get('dt') in (0, 2, 3) and not isinstance(val, (bytes, bytearray)):
            # a small numeric or time dictionary: bisect it by point fetches (wherescan's _code_of reads a
            # mode-0 time dictionary as text)
            lo, hi = 0, int(c['V']) - (1 if c.get('has_null') else 0) - 1
            code = None
            while lo <= hi:
                mid = (lo + hi) // 2
                fv = num(seg.fetch(kcol, mid))
                if fv == val:
                    code = mid
                    break
                if fv < val:
                    lo = mid + 1
                else:
                    hi = mid - 1
        else:
            code = _WS._code_of(seg, kcol, val)
    except Exception:
        return None
    if code is None:
        return None
    back = num(seg.fetch(kcol, int(code)))
    if isinstance(val, (bytes, bytearray)):
        back = back if isinstance(back, (bytes, bytearray)) else str(back).encode()
        return int(code) if bytes(back) == bytes(val) else None
    try:
        return int(code) if back == val or int(back) == int(val) else None
    except (TypeError, ValueError):
        return None


def detect(seg, tree, col_map, _allow_group_filter=False):
    """Shape gate shared by the live walk and the materialized sidecar. Returns
    (kcol, tcol, ci, ki, proj) for a `GROUP BY key, COUNT(DISTINCT target)` query inside v1 scope,
    else None. kcol/tcol are col_map-resolved physical names; ci/ki index the distinct-count and the
    bare-key projections. Both serve paths must agree on eligibility, so neither duplicates this."""
    # --- shared shape guards (lifted to wdb_policies; the sidecar relaxes no_where to group-key-only) ---
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    excl = None
    if not P.no_where(tree):
        w9 = tree.args.get('where')
        n9 = w9.this if w9 is not None else None
        gx = tree.args.get('group')
        # WHERE key <> <any literal> (THE Q13 BOARD, 2026-10-13: was key <> '' only -- ClickBench's own
        # literal): the filter names only the group key, so the other groups' counts are untouched and
        # the one group whose value it names leaves at emission
        if (isinstance(n9, E.NEQ) and isinstance(n9.this, E.Column)
                and gx is not None and len(gx.expressions) == 1
                and isinstance(gx.expressions[0], E.Column)
                and n9.this.name == gx.expressions[0].name
                and _is_literal(n9.expression)):
            excl = n9.expression
        elif not _allow_group_filter:
            return None

    if not P.single_group_key(tree):   return None
    group = tree.args.get('group')
    # --- distinct-family shape match (also EXTRACTS the column/projection indices, so it stays here) ---
    proj = tree.expressions
    if len(proj) != 2:
        return None
    di = _distinct_index(proj)
    if di is None:
        return None
    ci, tname = di                                       # ci = distinct-count projection index
    ki = 1 - ci
    kp = proj[ki]
    if wdb_sql._agg_kind(kp) is not None:                # the other projection must be the bare key
        return None
    knm = wdb_sql._proj_colname(kp)
    gnm = wdb_sql._colname(group.expressions[0])
    if knm is None or gnm is None or knm != gnm:
        return None
    if not _order_is_distinct_desc(tree, proj, ci):
        return None
    # --- resolve physical names, then shared segment/column guards ---
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    kcol = sc(knm); tcol = sc(tname)
    if not P.columns_exist(seg, kcol, tcol): return None
    if not P.no_deleted_rows(seg):           return None
    if not P.key_not_nullable(seg, kcol):    return None
    return kcol, tcol, ci, ki, proj, excl


HEAVY_STRIDE = 64      # the sample that finds heavy groups: every 64th row
HEAVY_SHARE = 64       # a group holding more than 1/64 of the rows is heavy: counted apart, on every core
HEAVY_MAX = 16
HEAVY_MIN_N = 4_000_000  # below this many rows one thread finishes any group quickly (tests lower it)
HEAVY_CELLS = 1 << 30  # heavy groups x target space, in bytes of presence table
HEAVY_HITS = 0         # telemetry: answers whose key lane counted heavy groups apart
PRESENCE_GROUPS = 16   # the presence lane: at most this many groups, each a presence row over the targets,
PRESENCE_CELLS = 1 << 26   # or any number of groups whose rows fit one shared table this size (bytes)
PACKED = True          # the scatter lane's one-u32-per-row form (tests turn it off to compare)


def _heavy_groups(grp, VRk, N, ecode):
    """The groups holding more than N / HEAVY_SHARE rows, estimated from a strided sample (heaviness only
    picks how a group is counted, never what is counted), at most HEAVY_MAX of them, the filtered one
    excluded"""
    s = np.asarray(grp[::HEAVY_STRIDE])
    if s.size == 0:
        return np.empty(0, np.int64)
    if VRk <= (1 << 26):
        c = np.bincount(s.astype(np.int64, copy=False), minlength=VRk)
        cand = np.flatnonzero(c * HEAVY_STRIDE > N // HEAVY_SHARE)
        cc = c[cand]
    else:
        u, cc0 = np.unique(s, return_counts=True)
        keep = cc0 * HEAVY_STRIDE > N // HEAVY_SHARE
        cand, cc = u[keep].astype(np.int64), cc0[keep]
    if ecode is not None:
        cc = cc[cand != ecode]; cand = cand[cand != ecode]
    if cand.size > HEAVY_MAX:
        cand = cand[np.argsort(-cc, kind='stable')[:HEAVY_MAX]]
    return np.sort(cand)


def lane_runs(grp, tgt, tnull, VRk, VTt, ecode=None):
    """THE Q13 BOARD'S LANES (2026-10-13), shared with group_mix (2026-10-19): per-group COUNT(DISTINCT)
    by a walk the two columns' profiles pick --
      DENSE   (groups x targets) cells small: per-thread presence boards, one parallel pass;
      SCATTER few groups, a big target space: MSD by target + a group marker table (Q8/Q9's lane);
      KEYMARK the target space fits a per-thread marker table: MSD by group, one linear marker walk;
      KEYSORT anything else: MSD by group, each group's own run sorted (no table at all).
    On the key lanes the heavy groups (one value holding most rows: the open form's '' or 0) are counted
    apart on every core (gd_heavy) -- left in, a heavy group is one bucket on one thread (SearchPhrase x
    UserID with no WHERE: 7.5 s). Returns (lane, runs): runs[g] = the distinct count, -1 = group absent;
    a group whose targets are all NULL is present with 0. The filtered group (ecode) may hold anything."""
    global HEAVY_HITS
    import wdb_kernels as _WK
    N = grp.shape[0]
    lane = None
    if _HAVE_NUMBA:
        if VRk * VTt <= DENSE_CELLS:
            lane = 'dense'
        elif N > HEAVY_MIN_N and ((VRk <= PRESENCE_GROUPS and VRk * VTt <= HEAVY_CELLS)
                                  or VRk * VTt <= PRESENCE_CELLS):
            lane = 'presence'
        elif tnull < 0 and N > 4_000_000 and VRk <= 262_144 and VTt > 65_536:
            lane = 'scatter'
        elif VTt <= MARK_MAX and VRk < (1 << 31):
            lane = 'keymark'
        elif VRk < (1 << 31) and VTt < (1 << 32):
            lane = 'keysort'
    LANES[lane] = LANES.get(lane, 0) + 1
    grp = np.ascontiguousarray(grp); tgt = np.ascontiguousarray(tgt)
    if lane == 'dense':
        dist, rowsper = _WK.gd_dense(grp, tgt, VRk, VTt, tnull)
        return lane, np.where(rowsper > 0, dist, -1)
    if lane == 'presence':
        # A HANDFUL OF GROUPS over a wide target (GoodEvent's 1 value x SearchPhrase): every group is a heavy
        # one -- one parallel pass marks presence rows, no scatter (the scatter lane: 0.67 s vs DuckDB 0.55)
        dist = _WK.gd_heavy(grp, tgt, np.arange(VRk, dtype=np.int32), VRk, VTt, tnull)
        rowsper = _WK.bincount_par(grp, VRk) if tnull >= 0 else dist
        return lane, np.where(np.asarray(rowsper) > 0, dist, -1)
    if lane == 'scatter':
        SH = max(1, VTt.bit_length() - 12)
        GB = max(1, int(VRk - 1).bit_length())
        if SH + GB <= 32 and PACKED:
            # one u32 per row: the target's low bits over the group (Q8: RegionID's 14 bits + UserID's 13)
            pk, offs = _WK.gd_pass1_packed(tgt, grp, np.int64(SH), np.int64(GB), np.int64(8))
            counts = _WK.gd_pass2_packed(pk, offs, np.int64(SH), np.int64(GB), np.int64(VRk))
            del pk
        else:
            ku, kr, offs = _WK.gd_pass1(tgt, grp, np.int64(SH), np.int64(8))
            counts = _WK.gd_pass2_count(ku, kr, offs, np.int64(SH), np.int64(VRk))
            del ku, kr
        return lane, np.where(counts > 0, counts, -1)    # no nulls on this lane: count > 0 iff present
    if lane in ('keymark', 'keysort'):
        SH = max(0, VRk.bit_length() - 12)
        heavy = _heavy_groups(grp, VRk, N, ecode) if N > HEAVY_MIN_N else np.empty(0, np.int64)
        if heavy.size and heavy.size * VTt <= HEAVY_CELLS:
            skipm = np.zeros(VRk, np.uint8)
            skipm[heavy] = 1
            if ecode is not None:
                skipm[ecode] = 1
            ku, kr, offs = _WK.gd_pass1_skipmask(grp, tgt, np.int64(SH), np.int64(8), skipm)
            runs = _WK.gd_keymark(ku, kr, offs, SH, VRk, VTt, tnull, mark=(lane == 'keymark'))
            del ku, kr
            hidx = np.full(VRk, -1, np.int32)
            hidx[heavy] = np.arange(heavy.size, dtype=np.int32)
            runs[heavy] = _WK.gd_heavy(grp, tgt, hidx, heavy.size, VTt, tnull)
            HEAVY_HITS += 1
            return lane, runs
        ku, kr, offs = _WK.gd_pass1_skip(grp, tgt, np.int64(SH), np.int64(8),
                                         np.int64(-1 if ecode is None else ecode))
        runs = _WK.gd_keymark(ku, kr, offs, SH, VRk, VTt, tnull, mark=(lane == 'keymark'))
        return lane, runs
    if VRk * VTt >= (1 << 62):                           # pack would overflow int64 -> decline
        return lane, None
    keep = (tgt != tnull) if tnull >= 0 else slice(None)
    key = grp[keep].astype(np.int64) * VTt + tgt[keep].astype(np.int64)
    uq = np.unique(key)
    counts = np.bincount((uq // VTt).astype(np.int64), minlength=VRk)
    present = np.bincount(grp.astype(np.int64), minlength=VRk) > 0
    return lane, np.where(present, counts, -1)


def execute(seg, det, tree):
    """THE READ: per-group COUNT(DISTINCT) via one code-hash walk. May decline (None) on
    measured conditions (mismatched lengths, pack overflow)."""
    global _HITS, SURVIVOR_HITS
    kcol, tcol, ci, ki, proj, excl = det
    _ecode = _excl_code(seg, kcol, excl) if excl is not None else None
    excl_empty = (_ecode is not None and isinstance(excl, E.Literal) and excl.is_string
                  and str(excl.this) == '')      # the shelf's own cut: it only ever stored that one


    # THE SHELF FIRST: an unfiltered per-group distinct count is a stored fact.
    if (P.no_where(tree) or excl_empty) and P.no_deleted_rows(seg):
        blob = _gdc_load(seg, kcol, tcol)
        if blob is not None and 'top_codes' in blob and excl_empty:
            lim9 = wdb_sql._limit(tree)
            tc9 = blob['top_codes']
            if lim9 is not None and lim9 <= tc9.size:
                rows = []
                for t9 in range(int(lim9)):
                    row = [None, None]
                    v9 = seg.fetch(kcol, int(tc9[t9]))
                    row[ki] = v9.decode('utf-8', 'replace') if isinstance(v9, (bytes, bytearray)) else v9
                    row[ci] = int(blob['top_counts'][t9])
                    rows.append(tuple(row))
                rows = workers.finalize(rows, proj, tree.args.get('order'), lim9)
                _HITS += 1
                return rows, [wdb_sql._alias(p) for p in proj]
        if blob is not None and 'counts' in blob and (not blob.get('noempty') or excl_empty):
            counts = blob['counts']; keys = blob['keys']
            if excl_empty and _ecode is not None and _ecode < len(counts):
                counts = np.asarray(counts).copy()
                counts[_ecode] = 0
            lim = wdb_sql._limit(tree)
            present = np.nonzero(counts)[0]
            sel = _top_sel(counts, present, lim)
            rows = []
            for gid in sel.tolist():
                row = [None, None]
                if keys is not None:
                    row[ki] = keys[gid]
                else:
                    v9 = seg.fetch(kcol, int(gid))
                    row[ki] = v9.decode('utf-8', 'replace') if isinstance(v9, (bytes, bytearray)) else v9
                row[ci] = int(counts[gid])
                rows.append(tuple(row))
            rows = workers.finalize(rows, proj, tree.args.get('order'), lim)
            _HITS += 1
            return rows, [wdb_sql._alias(p) for p in proj]

    kc9 = seg.cols.get(kcol)
    if (kc9 is not None and kc9.get('code_enc') in (8, 9) and hasattr(seg, 'e8_planes')
            and P.no_deleted_rows(seg) and _ecode is not None
            and _ecode == int(kc9.get('e8d' if kc9.get('code_enc') == 8 else 'e9d', -1))):
        # the planes hold only NON-default rows: this branch serves solely the
        # filtered shape (the default group is exactly what the filter removes)
        # BIG-KEY BRANCH (Q13's shape): sparse planes hand non-default rows;
        # scatter buckets target codes by key; every group's uniques exact;
        # the sidecar births codes-only (noempty) on first touch.
        try:
            import wdb_kernels as _WK
            pl = seg.e8_planes(kcol)
            pos8, lits8 = np.asarray(pl[0]), np.asarray(pl[1], dtype=np.int64)
            KV9 = int(kc9['V'])
            uc9 = np.asarray(seg._raw_codes(tcol))[pos8]
            VT9 = int(seg.cols[tcol].get('V') or 0)
            if KV9 <= 262_144 and VT9 > 65_536 and lits8.size > 4_000_000:
                # THE SCATTER LANE for a SMALL key space (Q10: 166 phone models): per-slice sorts
                # were one giant slice on one thread (336 ms); the MSD lane balances by TARGET
                SH9 = max(1, VT9.bit_length() - 12)
                ku9, kr9, offs9 = _WK.gd_pass1(np.ascontiguousarray(uc9), np.ascontiguousarray(lits8),
                                               np.int64(SH9), np.int64(8))
                dcounts = _WK.gd_pass2_count(ku9, kr9, offs9, np.int64(SH9), np.int64(KV9))
            else:
                cnts9 = np.bincount(lits8, minlength=KV9)
                uc9 = uc9.astype(np.int64)
                offs9 = np.zeros(KV9 + 1, np.int64)
                np.cumsum(cnts9, out=offs9[1:])
                bucketed9 = np.empty(lits8.size, np.int64)
                _WK.cd_scatter(lits8, uc9, offs9, offs9[:-1].copy(), bucketed9)
                dcounts = _WK.cd_alldistinct(bucketed9, offs9)
            _gdc_save2(seg, kcol, tcol, dcounts)
            import wdb_shelves
            wdb_shelves.record(seg, 'gdc2', k=kcol, t=tcol)
            counts = dcounts
            if _ecode is not None and _ecode < counts.size:
                counts = counts.copy(); counts[_ecode] = 0
            lim = wdb_sql._limit(tree)
            present = np.nonzero(counts)[0]
            sel = _top_sel(counts, present, lim)
            rows = []
            for gid in sel.tolist():
                row = [None, None]
                v9 = seg.fetch(kcol, int(gid))
                row[ki] = v9.decode('utf-8', 'replace') if isinstance(v9, (bytes, bytearray)) else v9
                row[ci] = int(counts[gid])
                rows.append(tuple(row))
            rows = workers.finalize(rows, proj, tree.args.get('order'), lim)
            _HITS += 1
            return rows, [wdb_sql._alias(p) for p in proj]
        except Exception:
            pass
    def _codes_or_ids(col):
        # THE KEY IS ITS CODE (Q09's line items, applied here for Q08): a dict column without nulls
        # groups on its codes as stored -- no value array, no astype; decoded only at emission
        c9 = seg.cols.get(col)
        if c9 is not None and c9.get('mode') in (0, 1, 2) and not c9.get('has_null'):
            dec = wdb_gbcount._code_values(seg, col)
            if dec is None:
                import wdb_window as _WN
                dec = np.asarray(_WN._int_table(seg, col), dtype=np.int64)
            return np.ascontiguousarray(np.asarray(seg._raw_codes(col))), -1, dec
        return _ids(seg, col)
    kc0 = seg.cols.get(kcol)
    if kc0 is None or kc0.get('mode') not in (0, 1, 2) or kc0.get('has_null'):
        return None
    if _ecode is not None and int(kc0['V']) == 1:
        _HITS += 1                                       # the key holds one value and the filter removes it
        return [], [wdb_sql._alias(p) for p in proj]
    # the key is its code, decoded at emission for the winners only (_bulk_keyvals: typed -- a time key
    # emits as a time -- and point-fetched from a big dictionary)
    grp = np.ascontiguousarray(np.asarray(seg._raw_codes(kcol)))
    tinfo = None
    t_codes = False
    tc0 = seg.cols.get(tcol)
    if _ecode is not None and tc0 is not None and tc0.get('mode') in (0, 1, 2) and grp.size:
        # THE SURVIVOR ROAD (2026-10-13): when the removed group is nearly every row (IsOldCounter's 0,
        # OpenerName's -1, ParamOrderID's '': ~100%), read the counted column at the survivors only --
        # the whole column decode was the bill (DuckDB 0.08 s, the full lanes 0.4-0.7 s; the road 0.05 s)
        keep = grp != _ecode
        m9 = int(np.count_nonzero(keep))
        if m9 * SURVIVOR_ROAD <= grp.size:
            rows9 = np.flatnonzero(keep)
            grp = np.ascontiguousarray(grp[rows9])
            tgt9 = np.ascontiguousarray(np.asarray(seg.codes_at(tcol, rows9)))
            tinfo = (tgt9, (int(tc0['V']) - 1) if tc0.get('has_null') else -1, None)
            t_codes = True
            SURVIVOR_HITS += 1
        del keep
    if tinfo is None:
        tinfo = _codes_or_ids(tcol)
    if tinfo is None:
        return None
    tgt, tnull, _tdecode = tinfo
    if grp.shape[0] != tgt.shape[0]:
        return None
    N = grp.shape[0]
    # THE SCATTER LANE (Jackson's reorganization): no nulls, dict codes both sides,
    # big target space -> MSD 2-pass scatter groups rows by target, then an
    # L1-resident marker table counts each target's first touch of every key.
    # Measured on a-cd-grp: 2.4s walk -> ~0.65s, exact. All transient: cold-truth.
    # the key is its dictionary code: [0, V) by construction -- no min / max passes over 100M rows (four of
    # them cost ~0.1 s of every query in this family); a target read as codes likewise
    VRk = int(kc0['V']) if N else 0
    t_codes = t_codes or _tdecode is not None            # dictionary codes (raw ids carry no decode)
    if t_codes and tc0 is not None and tc0.get('V'):
        VTt = int(tc0['V']) if N else 0
    else:
        if N and tgt.dtype.kind == 'i' and int(tgt.min()) < 0:
            lo9 = int(tgt.min())                         # raw-value targets below zero: distinctness is all
            tgt = tgt.astype(np.int64) - lo9             # that matters, so shift them onto >= 0
            tnull = tnull - lo9 if tnull >= 0 else -1
        VTt = int(tgt.max()) + 1 if N else 0
    names = [wdb_sql._alias(p) for p in proj]
    if N == 0:
        _HITS += 1
        return [], names
    lane, runs = lane_runs(grp, tgt, tnull, VRk, VTt, _ecode)
    if runs is None:
        return None
    if lane == 'scatter' and P.no_where(tree) and P.no_deleted_rows(seg):
        import wdb_sidecar as _wsc9, os as _os9
        if _wsc9.births_on(_os9.path.dirname(seg.path)):
            keys = _keyvals(seg, kcol, np.arange(VRk, dtype=np.int64))
            _gdc_save(seg, kcol, tcol, np.maximum(runs, 0), keys)   # birth-on-first-touch (switch-gated)
        import wdb_shelves
        wdb_shelves.record(seg, 'gdc', k=kcol, t=tcol)
    import wdb_kernels as _WK
    lim = wdb_sql._limit(tree)
    sel = None
    counts = runs                                        # per-group counts, -1 = absent (every lane)
    if lim is not None and lim > 0 and runs.size > (1 << 20):
        # the census pick, for many groups only: its histogram is as wide as the largest count, and a few
        # groups can each count millions (RegionID x UserID: 0.48 -> 1.1 s); numpy picks among few groups
        sel = _WK.gd_topk(runs, -1 if _ecode is None else _ecode, int(lim))
    if sel is None:
        present = np.flatnonzero(runs >= 0)
        counts = np.maximum(runs, 0)
        if _ecode is not None:                           # WHERE key <> literal: that one group leaves
            present = present[present != _ecode]
        sel = _top_sel(counts, present, lim)
    kv = _keyvals(seg, kcol, np.asarray(sel, dtype=np.int64))
    rows = []
    for j, gid in enumerate(sel.tolist()):
        row = [None, None]
        row[ki] = kv[j]
        row[ci] = int(counts[gid])
        rows.append(tuple(row))
    rows = workers.finalize(rows, proj, tree.args.get('order'), lim)
    _HITS += 1
    return rows, names


def _keyvals(seg, kcol, codes):
    import wdb_join
    out = wdb_join._bulk_keyvals(seg, kcol, codes)
    return [k.decode('utf-8', 'replace') if isinstance(k, (bytes, bytearray)) else k for k in out]


def try_groupdistinct(seg, tree, col_map):
    """Detect + execute, kept as the backward-compatible single-call entry."""
    det = detect(seg, tree, col_map)
    if det is None:
        return None
    return execute(seg, det, tree)
