"""
wdb_gridwalk — grid filled-cell + count-ordered head for 2-key GROUP BY COUNT(*) top-K.

The per-PAIR-but-not-per-ROW structure. wdb_countpos stores the count once per ROW (N entries);
wdb_heavypair stores identity+count per heavy pair (16 B each). This stores each heavy distinct pair
ONCE as a filled cell of the A×B grid -- identity is the cell's code coordinates (codeA,codeB) packed
as a grid-id, the count stored once per cell -- and keeps a count-ordered HEAD (the top-N cells by
count) as the entry point. Top-K is a direct O(K) slice of that pre-sorted head: enter at the
high-count end, read K, decode coordinates back to values. No per-row materialization, no scan.

Measured (100M cb25db, SearchPhrase x UserID): head read top-10 = ~2 us, exact; the full filled-cell
structure is ~27 MB (walked identity 6.9 + counts 20 + head 0.4) vs heavypair's 161 MB and the per-row
countpos column's 200 MB -- 6x lighter, microsecond reads.

v2 scope (this module): serves 2-key COUNT(*) top-K (ORDER BY COUNT(*) DESC, or unordered LIMIT), plus
SUM/AVG(int col) payload computed winner-only at read (one match pass locates the <=K winners' member
rows; only those decode); NO WHERE/HAVING/DISTINCT/JOIN/OFFSET. The head is a CANONICAL prefix (count
DESC, gid ASC), so a count plateau at the LIMIT boundary resolves deterministically -- never a decline
(the plateau is a set of equally-correct answers; we emit the canonical one and the total-order
validator agrees). A LIMIT deeper than the heavy set fills from `ones` (the _ONES_N smallest singleton
gids, count 1) -- exception-defined pairs like WatchID x ClientIP (4 heavy cells in 100M) are served
as: 4 cells + deterministic singleton fill. For LIMIT beyond the head it decodes the gap-encoded bulk
(all heavy cells, byte-block frame-of-reference over dict-order gaps).
Append-tail maintenance is the documented next layer. RAM-resident, never persisted. Live by default
(_ENABLED True); disable() restores the prior routing (heavypair/scan) for the shapes it handles.
"""
import numpy as np
import wdb_qmem
import wdb_sql
import wdb_pairagg
import workers
import wdb_policies as P
E = wdb_sql.E

_ENABLED = True           # live by default: gridwalk is the primary 2-key COUNT(*) top-K read
_HEAD_N = 200000          # count-ordered head depth; serves any LIMIT up to this, declines beyond
_ONES_N = 1024            # singleton fill head: the _ONES_N smallest count-1 gids, for LIMIT > nheavy
_BINC_MAX = 1 << 24     # composite spaces up to 16.7M cells go by bincount (128 MB transient, query-scoped)
_DENSE_PAR = [__import__('os').environ.get('WDB_GW_DENSE_PAR', '1') != '0']   # the dense build on every core
_FUSED_MAX = [int(__import__('os').environ.get('WDB_GW_FUSED_MAX', str(1 << 22)))]   # cells: the fused count's
# line, measured (fusesweep.py, _build both ways on real pairs): fused 0.022-0.028 s up to 252K cells against the
# radix path's flat 0.11-0.12; 1.07M 0.041 vs 0.130; 1.94M 0.082 vs 0.153; 3.96M 0.121 vs 0.161; 7.97M 0.161 vs
# 0.149; 16.1M 0.297 vs 0.176 -- the boards (T x cells x 8 B) grow, the key pass does not. Crossing ~6M; at 4M the
# boards are 512 MB on 16 threads, under the radix path's 800 MB key + its own boards
_POS_MAX = 8192           # store member-row positions when total stored cells <= this (payload O(K))
_POSROWS_MAX = 1 << 16    # ...AND their member rows <= this: positions cost bytes per ROW, and a
                          # low-card pair's handful of cells holds ALL 100M rows (measured 1.5 GB!)
_HITS = 0
_CACHE = wdb_qmem.register({})               # (seg.path,(a,b),N) -> (head_gid, head_cnt, Vb, nheavy, bulk, ones, pos, nd)
_VCACHE = wdb_qmem.register({})              # (seg.path,col) -> by-code value dict (decode)


def enable():
    global _ENABLED
    _ENABLED = True
    import controller
    controller.plan_epoch_bump()


def disable():
    global _ENABLED
    _ENABLED = False
    import controller
    controller.plan_epoch_bump()   # standing a lane down changes routes


def is_enabled():
    return _ENABLED


def _vals(seg, col):
    if seg.cols[col]['mode'] == 4:
        return None
    ck = (seg.path, col)
    hit = _VCACHE.get(ck)
    if hit is not None:
        return hit
    try:
        v = np.asarray(seg._typed_dict(col))
    except Exception:
        return None
    _VCACHE[ck] = v
    return v


_BULK_B = 128             # bulk block size; per-block byte-width frame-of-reference over dict-order gaps


def _bulk_encode(gid, cnt):
    """Gap-encoded bulk: ALL heavy filled cells (not just the head). gid sorted ascending in code
    order; within-block deltas BIT-PACKED at the block's exact bit-width (frame-of-reference), the
    big cross-block gap absorbed by an explicit per-block anchor (per-block streams stay byte-aligned
    because B=128 is divisible by 8). Counts use the NORM-IMPLICIT law when one count value dominates
    the class (>=50%): the modal count is declared once, a bitmap marks deviants, and only deviants
    carry an explicit count -- the exceptions of the exceptions. Otherwise counts stay flat.
    Has per-block anchors for future point-lookup/maintenance. RAM-resident, never persisted."""
    n = int(gid.size)
    B = _BULK_B
    nb = (n + B - 1) // B
    pad = nb * B - n
    gp = np.concatenate([gid, np.full(pad, gid[-1], dtype=np.int64)]).reshape(nb, B)
    wd = np.diff(gp, axis=1, prepend=gp[:, :1])          # within-block deltas; col0 == 0
    anchors = np.ascontiguousarray(gp[:, 0])             # absolute gid at each block start
    maxd = wd.max(axis=1)
    nbits = np.where(maxd > 0, np.floor(np.log2(np.maximum(maxd, 1))).astype(np.int64) + 1,
                     1).astype(np.uint8)                 # exact bits/delta per block, 1..64
    buffers = {}
    for k in np.unique(nbits):
        k = int(k)
        bidx = np.flatnonzero(nbits == k)
        blk = wd[bidx]                                   # (m, B) int64, every value < 2^k
        shifts = np.arange(k - 1, -1, -1, dtype=np.int64)
        bits = ((blk[:, :, None] >> shifts) & 1).astype(np.uint8)   # MSB-first within each delta
        buffers[k] = np.packbits(bits.reshape(-1))
    out = {'n': n, 'B': B, 'nb': nb, 'anchors': anchors, 'nbits': nbits, 'buffers': buffers}
    cdt = np.uint16 if int(cnt.max()) < 65536 else np.uint32
    vals, freq = np.unique(cnt, return_counts=True)
    mi = int(freq.argmax()); mode = int(vals[mi])
    if n >= 4096 and freq[mi] / n >= 0.5:                # norm-implicit: the class's modal count
        exc = cnt != mode
        out['cnorm'] = mode
        out['cbits'] = np.packbits(exc)
        out['cexc'] = np.ascontiguousarray(cnt[exc], dtype=cdt)
    else:
        out['cnt'] = cnt.astype(cdt)
    return out


def _bulk_decode_all(bulk):
    """Reconstruct (gid, cnt) for all heavy cells in code order. Vectorized per bit-width group in
    block slabs (bounds the unpack transient); counts re-expand from the norm + deviants when the
    norm-implicit law applied. Round-trip exact."""
    n = bulk['n']; B = bulk['B']; nb = bulk['nb']
    anchors = bulk['anchors']; nbits = bulk['nbits']; buffers = bulk['buffers']
    out = np.empty(nb * B, dtype=np.int64)
    ar = np.arange(B)
    SLAB = 8192                                          # blocks per unpack slab
    for k, buf in buffers.items():
        k = int(k)
        bidx = np.flatnonzero(nbits == k)
        shifts = np.arange(k - 1, -1, -1, dtype=np.int64)
        bpb = (B * k) // 8                               # bytes per block (byte-aligned: 8 | B)
        for s in range(0, bidx.size, SLAB):
            sl = bidx[s:s + SLAB]; m = sl.size
            bits = np.unpackbits(buf[s * bpb:(s + m) * bpb], count=m * B * k)
            val = (bits.reshape(m, B, k).astype(np.int64) << shifts).sum(axis=2)
            rec = np.cumsum(val, axis=1) + anchors[sl][:, None]
            out[(sl[:, None] * B + ar)] = rec
    if 'cnorm' in bulk:
        cnt = np.full(n, bulk['cnorm'], dtype=np.int64)
        mask = np.unpackbits(bulk['cbits'], count=n).astype(bool)
        cnt[mask] = bulk['cexc']
    else:
        cnt = bulk['cnt']
    return out[:n], cnt


def _bulk_nbytes(bulk):
    """Resident size of the bulk in bytes (anchors + bit-widths + packed deltas + counts)."""
    b = bulk['anchors'].nbytes + bulk['nbits'].nbytes
    for buf in bulk['buffers'].values():
        b += buf.nbytes
    if 'cnorm' in bulk:
        b += bulk['cbits'].nbytes + bulk['cexc'].nbytes + 8
    else:
        b += bulk['cnt'].nbytes
    return int(b)


def _build(seg, cols):
    """Filled-cell head: occurring heavy pairs (count>=2) as grid-ids, the top _HEAD_N in the
    DETERMINISTIC total order (count DESC, gid ASC) -- so the head is a canonical prefix even when a
    count plateau crosses its boundary. Also keeps `ones`: the _ONES_N smallest count-1 gids, so a
    LIMIT deeper than the heavy set fills deterministically from singletons (exception-defined pairs
    like WatchID x ClientIP have nheavy ~ a handful and the answer is mostly fill).
    Returns (head_gid, head_cnt, Vb, nheavy, bulk|None, ones, pos|None, nd) or None; nd is the pair's
    FULL distinct-cell count (heavy + singletons), recorded for the survey's FD post-pass."""
    a, b = sorted(cols)
    ca0 = np.ascontiguousarray(seg._raw_codes(a)); cb0 = np.ascontiguousarray(seg._raw_codes(b))
    if ca0.size == 0:
        return None
    Vb = int(cb0.max()) + 1                          # the stored-width codes' own max (the cell ids' radix)
    Va = int(ca0.max()) + 1
    if Va * Vb <= _BINC_MAX and ca0.dtype.kind in 'iu' and cb0.dtype.kind in 'iu' and _DENSE_PAR[0]:
        # THE GRID ON EVERY CORE (2026-10-23, Q17's tiny x tiny pairs: JavaEnable x Income 0.39 s vs DuckDB 0.09).
        # The build was numpy glue on one thread: two int64 copies of 100M codes (0.12 s), the key a x Vb + b as a
        # new array, a one-thread bincount (0.21 s with the glue) -- 17 ms of it was the decode. The composed count's
        # own atoms instead: radix_acc glues the stored-width codes into the key in place, bincount_par tallies the
        # cells on per-thread boards. The same key and the same cells (a x Vb + b).
        import wdb_kernels
        if Va * Vb <= _FUSED_MAX[0]:
            # THE COUNT FUSED (2026-10-23): a small grid is counted straight from the two code arrays on per-thread
            # boards (grid2's kernel) -- the key is never written: radix_acc's two passes over an 800 MB key were
            # 68 ms of JavaEnable x Income's 125. The key is built below only if the member positions need it.
            key = None
            full = np.asarray(wdb_kernels.grid2_count(ca0, cb0, ca0, np.int64(-1), np.int64(Vb), np.int64(Va * Vb)))
        else:
            key = np.empty(ca0.size, np.int64)
            wdb_kernels.radix_acc(key, ca0, Va, True)
            wdb_kernels.radix_acc(key, cb0, Vb, False)
            full = np.asarray(wdb_kernels.bincount_par(key, Va * Vb))
        gid_all = np.flatnonzero(full)
        cnt_all = full[gid_all]
        del full
    else:
        ca = ca0.astype(np.int64); cb = cb0.astype(np.int64)
        key = ca * Vb + cb
        if Va * Vb <= _BINC_MAX:             # small composite space: bincount the grid directly --
            full = np.bincount(key, minlength=Va * Vb)   # no 100M sort (np.unique sorts)
            gid_all = np.flatnonzero(full)               # ascending = unique's code order
            cnt_all = full[gid_all]
            del full
        else:
            gid_all, cnt_all = np.unique(key, return_counts=True)   # filled cells, code order
    nd = int(gid_all.size)               # the pair's FULL distinct count -- the survey's FD input
    heavy = cnt_all >= 2
    gid = gid_all[heavy]; cnt = cnt_all[heavy]
    ones = np.ascontiguousarray(gid_all[~heavy][:_ONES_N], dtype=np.int64)  # ascending = smallest gids
    if gid.size == 0 and ones.size == 0:
        return None
    if gid.size:
        o = np.lexsort((gid, -cnt))                  # canonical total order: count DESC, gid ASC
        k = min(_HEAD_N, gid.size)
        head_gid = np.ascontiguousarray(gid[o[:k]], dtype=np.int64)
        head_cnt = np.ascontiguousarray(cnt[o[:k]], dtype=np.uint32)   # counts < 2^32 always (N=100M)
        bulk = _bulk_encode(gid, cnt)    # full heavy-cell coverage, gap-encoded, for LIMIT beyond head
    else:                                # all-singleton pair: the fully exception-defined extreme
        head_gid = np.empty(0, np.int64); head_cnt = np.empty(0, np.int64); bulk = None
    # Exception-defined pairs (few cells stored in total): also record the MEMBER ROW POSITIONS of
    # every stored cell, so SUM/AVG payload at read is a direct row gather + decode of ~K rows --
    # no per-query pass over the 100M codes. ~1 KB for WatchID x ClientIP (4 heavy + ones).
    pos = None
    stored = np.concatenate((gid, ones)) if ones.size else gid
    nmember = int(cnt.sum()) + int(ones.size)    # positions scale with MEMBER ROWS, not cells --
    if 0 < stored.size <= _POS_MAX and nmember <= _POSROWS_MAX:   # a low-card pair's few cells hold ALL rows
        st = np.sort(stored)
        if key is None:                  # the fused count wrote no key: built here, for the positions only
            key = ca0.astype(np.int64) * Vb + cb0.astype(np.int64)
        p = np.searchsorted(st, key)
        valid = p < st.size
        p2 = np.where(valid, p, 0)
        match = valid & (st[p2] == key)
        rows = np.nonzero(match)[0]
        rgid = key[rows]
        o2 = np.argsort(rgid, kind='stable')
        pos = (np.ascontiguousarray(rgid[o2]), np.ascontiguousarray(rows[o2]))
    return head_gid, head_cnt, Vb, int(gid.size), bulk, ones, pos, nd


_COMPOSE = [__import__('os').environ.get('WDB_GRIDWALK_COMPOSE', '1') != '0']
_COMPOSED = 0             # top-K answers by the composed atoms (no structure built)


_GRAB_MAX = [int(__import__('os').environ.get('WDB_GW_GRAB_MAX', '-1'))]   # rows; -1: the share below decides
# THE LINE (2026-10-18, measured: cross5.jsonl / cross6.jsonl, 26 pairs): the grab saves the other key's decode and
# a count of N, and pays a gather and a count of S -- so the line is the share where count(S) + gather(S) meets
# count(N), and the count is what differs by lane: the dense census is cheap per row (ties at 17-27% of the rows),
# the flat lane's pack + scatter + bucket count is not (ties at 27-45%; ClientTimeZone x Title still won at 56%)
_GRAB_SHARE = [float(__import__('os').environ.get('WDB_GW_GRAB_DENSE', '0.15')),
               float(__import__('os').environ.get('WDB_GW_GRAB_FLAT', '0.30'))]
GRAB_AT_DIV = 256        # survivors up to N / this are read AT their rows; more: the column decoded whole, gathered
GRAB_STREAM = (1,)       # one zstd stream: a point read costs a decode (calib.jsonl) -- decoded whole
GRAB_SORT_MAX = 1 << 18  # survivors up to this are counted by one sort (np.unique: 0.3 ms at 10K, 3.8 at 100K);
                         # past it the all-core atoms (their fixed cost was 38-53 ms at 2K survivors)
GRAB_FRAMES = 0.5        # frame-coded (enc 3 / 18 / 20): read at rows while the survivors touch at most this share
                         # of the frames (a point read inflates exactly the frames it touches)
_GRABBED = 0


def _grab_limit(N, dense):
    return _GRAB_MAX[0] if _GRAB_MAX[0] >= 0 else int(_GRAB_SHARE[0 if dense else 1] * N)


def _grab_top(seg, a, b, Va, Vb, lim, afilter, tick):
    """THE GRAB (2026-10-18, the Q14 board; Jackson: "is just grabbing the 1M faster than removing the 99M"):
    the axis filter's survivors are found from the filter key's codes alone, the other key is read only for them,
    and only they are counted. The remove lane decodes both keys whole, counts all N rows, then drops the
    filtered cells. Both decode the filter key, so the grab saves (decode of the other key + a count of N) and
    pays (reading the other key at S rows + a count of S):
      the read: point reads (seg.codes_at) while S <= N / GRAB_AT_DIV; past that the frames a point read inflates
        are most of the column anyway (enc 1/3/18/20 reach a full decode by ~20K spread rows, enc 8/9 pass it near
        N/128), so the column is decoded whole and gathered;
      the count: the remove lane's own atoms on S rows -- dense cells (radix_acc + bincount_par) when Va x Vb fits
        DENSE_MAX, the flat lane (wdb_bound._flat_candidates) otherwise -- every core, no sort (np.unique was
        51 ms per 1M survivors on one core).
    A filter key holding one value answers from its dictionary: '<>' that value keeps no row. Grabs while
    S <= _grab_limit(N); returns the canonical top (sel_gid, sel_cnt, Vb), or None (the remove lane answers)."""
    global _GRABBED
    import wdb_kernels, wdb_bound
    fcol, fneg, fcode = afilter
    c = seg.cols[fcol]
    if int(c['V']) - (1 if c.get('has_null') else 0) == 1 and int(fcode) == 0 and fneg:
        tick('grab: one-value key, nothing kept')
        _GRABBED += 1
        return np.empty(0, np.int64), np.empty(0, np.int64), Vb
    N = int(seg.N)
    cf = np.ascontiguousarray(seg._raw_codes(fcol))
    m = (cf != fcode) if fneg else (cf == fcode)
    S = int(np.count_nonzero(m))
    tick('grab survivors(%d)' % S)
    if S > _grab_limit(N, Va * Vb <= wdb_bound.DENSE_MAX):
        return None
    _GRABBED += 1
    if S == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), Vb
    pos = np.flatnonzero(m)
    other = b if fcol == a else a
    oc = seg.cols[other]
    enc = oc.get('code_enc', 0)
    BR = int(oc.get('BR') or oc.get('e20BR') or 0) if enc in (3, 18, 20) else 0
    if enc in GRAB_STREAM:
        at = False
    elif BR > 0:
        fr = pos // BR
        touched = 1 + int(np.count_nonzero(fr[1:] != fr[:-1]))
        at = touched <= GRAB_FRAMES * ((N + BR - 1) // BR)
    else:
        at = S * GRAB_AT_DIV <= N
    if at:
        co = np.ascontiguousarray(seg.codes_at(other, pos))
        how = 'at rows'
    else:
        co = np.ascontiguousarray(np.asarray(seg._raw_codes(other))[pos])
        how = 'decoded, gathered'
    cfp = np.ascontiguousarray(cf[pos])
    tick('grab read %s %s (%d)' % (other, how, S))
    ka, kb = (cfp, co) if fcol == a else (co, cfp)
    prod = Va * Vb
    if S <= GRAB_SORT_MAX:
        u, cnt = np.unique(ka.astype(np.int64) * Vb + kb.astype(np.int64), return_counts=True)
        gid = u.astype(np.int64); cnt = cnt.astype(np.int64)
        tick('grab count sorted(%d pairs)' % gid.size)
    elif prod <= wdb_bound.DENSE_MAX:
        comp = np.empty(S, np.int64)
        wdb_kernels.radix_acc(comp, ka, Va, True)
        wdb_kernels.radix_acc(comp, kb, Vb, False)
        cells = wdb_kernels.bincount_par(comp, prod)
        gid = np.flatnonzero(cells); cnt = np.asarray(cells[gid], np.int64)
        ck = wdb_bound._kth_largest(cnt, lim) if gid.size > lim else 0
        if ck > 0:
            keep = cnt >= ck; gid = gid[keep]; cnt = cnt[keep]
        tick('grab count dense(%d cells, %d occur)' % (prod, gid.size))
    else:
        got = wdb_bound._flat_candidates([ka, kb], [Va, Vb], int(lim), tick)
        if got is None:
            u, cnt = np.unique(ka.astype(np.int64) * Vb + kb.astype(np.int64), return_counts=True)
            gid = u.astype(np.int64); cnt = cnt.astype(np.int64)
            tick('grab count sorted(%d pairs)' % gid.size)
        else:
            uk, cnt = got
            gid = uk[:, 0].astype(np.int64) * Vb + uk[:, 1].astype(np.int64)
            tick('grab count flat(%d candidates)' % gid.size)
    o = np.lexsort((gid, -cnt))[:lim]                 # the canonical prefix: count DESC, gid ASC
    return gid[o], cnt[o], Vb


def _compose_top(seg, cols, lim, afilter=None):
    """THE PAIR COUNT FROM THE ATOMS (2026-10-15, the Q16 board): the canonical top `lim` cells (count DESC,
    gid ASC, gid = codeA * Vb + codeB over the name-sorted columns) WITHOUT building the structure -- the
    structure lives one query (qmem) and its build was the whole bill: codes copied to 8 bytes and glued
    (~280 ms), then ONE core tallied (~190 ms) or sorted all 100M keys (5-8 s) whenever the POSSIBLE pairs
    passed 16.7M (WindowClientHeight x WindowClientWidth: 24.7M possible, 776K occurring), then every heavy
    cell sorted and bulk-encoded for LIMITs nobody asked. Composed instead, on every core:
      possible pairs <= DENSE_MAX: radix_acc glues the stored-width codes, bincount_par tallies the cells;
      otherwise the flat lane (pack_u64 / flat_scatter / flat_count) counts only the pairs that occur;
    then only the cells reaching the lim-th count (ties kept) are sorted canonically.
    THE AXIS FILTER (2026-10-17, the Q14 board): a WHERE that is one '=' / '<>' on ONE of the two keys
    (afilter = (column, neg, code)) is applied to the cells, not the rows -- the dense table's row or column of
    that code is kept alone ('=') or zeroed ('<>'); the flat lane counts only the rows it keeps. Exact: a pair's
    count is all its rows, and the filter takes a pair whole or not at all. (heavypair answered these by
    sorting every pair of all 100M rows each query -- 2-10 s -- because its stored sidecar is off.)
    Returns (sel_gid, sel_cnt, Vb) or None (the structure build answers, as before)."""
    import os, time
    import wdb_kernels, wdb_bound
    bill = os.environ.get('WDB_JOIN_BILL')
    t0 = time.perf_counter(); T = []
    def tick(nm):
        nonlocal t0
        if bill:
            t1 = time.perf_counter(); T.append('%s=%.0fms' % (nm, (t1 - t0) * 1e3)); t0 = t1
    a, b = sorted(cols)
    if seg.cols[a].get('has_null') or seg.cols[b].get('has_null'):
        return None                                   # a null code may sit past the dictionary: the build
    Va = int(seg.cols[a]['V']); Vb = int(seg.cols[b]['V'])
    if Va <= 0 or Vb <= 0:
        return None
    if afilter is not None:
        got = _grab_top(seg, a, b, Va, Vb, int(lim), afilter, tick)
        if got is not None:
            if bill:
                print('GRIDWALK BILL: ' + ' '.join(T), flush=True)
            return got
    ca = np.ascontiguousarray(seg._raw_codes(a)); cb = np.ascontiguousarray(seg._raw_codes(b))
    if ca.size == 0:
        return None
    if ca.dtype.kind not in 'iu' or cb.dtype.kind not in 'iu':
        return None
    tick('codes(%s,%s)' % (ca.dtype, cb.dtype))
    prod = Va * Vb
    if prod <= wdb_bound.DENSE_MAX:
        comp = np.empty(ca.size, np.int64)
        wdb_kernels.radix_acc(comp, ca, Va, True)
        wdb_kernels.radix_acc(comp, cb, Vb, False)
        tick('glue')
        cells = wdb_kernels.bincount_par(comp, prod)
        del comp
        if afilter is not None:
            fcol, fneg, fcode = afilter
            grid = np.asarray(cells).reshape(Va, Vb)
            if fcol == a:
                if fneg:
                    grid[fcode, :] = 0
                else:
                    row = grid[fcode, :].copy(); grid[:] = 0; grid[fcode, :] = row
            else:
                if fneg:
                    grid[:, fcode] = 0
                else:
                    col = grid[:, fcode].copy(); grid[:] = 0; grid[:, fcode] = col
            cells = grid.reshape(-1)
            tick('axis filter')
        gid = np.flatnonzero(cells)
        cnt = np.asarray(cells[gid], np.int64)
        tick('tally(%d cells, %d occur)' % (prod, gid.size))
        ck = wdb_bound._kth_largest(cnt, lim) if gid.size > lim else 0
        if ck > 0:
            keep = cnt >= ck
            gid = gid[keep]; cnt = cnt[keep]
    else:
        keep = None
        if afilter is not None:
            fcol, fneg, fcode = afilter
            fc9 = ca if fcol == a else cb
            keep = (fc9 != fcode) if fneg else (fc9 == fcode)
            tick('axis filter')
        got = wdb_bound._flat_candidates([ca, cb], [Va, Vb], int(lim), tick, keep=keep)
        if got is None:
            if bill:
                print('GRIDWALK BILL: ' + ' '.join(T) + ' -> structure', flush=True)
            return None
        uk, cnt = got
        gid = uk[:, 0].astype(np.int64) * Vb + uk[:, 1].astype(np.int64)
    o = np.lexsort((gid, -cnt))[:lim]                 # the canonical prefix: count DESC, gid ASC
    tick('pick(%d candidates)' % gid.size)
    if bill:
        print('GRIDWALK BILL: ' + ' '.join(T), flush=True)
    return gid[o], cnt[o], Vb


def _load(seg, cols):
    """RAM-resident: cache the count-head for this pair, built on first touch. Never disk."""
    a, b = sorted(cols)
    ck = (seg.path, (a, b), int(seg.N))
    hit = _CACHE.get(ck)
    if hit is not None:
        return hit
    built = _build(seg, cols)
    if built is None:
        return None
    _CACHE[ck] = built
    return built


def build(seg, cols):
    """Eagerly materialize a pair's count-head into the RAM cache. Returns True if available."""
    return _load(seg, cols) is not None


def structure_nbytes(built):
    """Resident bytes of one pair's structure (head + bulk + ones + positions)."""
    head_gid, head_cnt, Vb, nheavy, bulk, ones, pos, _nd = built
    b = head_gid.nbytes + head_cnt.nbytes + ones.nbytes
    if bulk is not None:
        b += _bulk_nbytes(bulk)
    if pos is not None:
        b += pos[0].nbytes + pos[1].nbytes
    return int(b)


def cache_pop(seg, cols):
    """Drop one pair's structure from the RAM cache (budget eviction)."""
    a, b = sorted(cols)
    _CACHE.pop((seg.path, (a, b), int(seg.N)), None)


def _count_index(proj):
    ci = None
    for i, p in enumerate(proj):
        ak = wdb_sql._agg_kind(p)
        if ak is not None and ak[0] == 'COUNT_STAR':
            if ci is not None:
                return None
            ci = i
    return ci


def _order_is_count_desc(tree, proj, ci):
    order = tree.args.get('order')
    if order is None or not order.expressions:
        return False
    first = order.expressions[0]
    if not isinstance(first, E.Ordered) or not first.args.get('desc'):
        return False
    tgt = first.this
    alias = wdb_sql._alias(proj[ci])
    if isinstance(tgt, E.Column) and tgt.name == alias:
        return True
    ak = wdb_sql._agg_kind(tgt)
    return ak is not None and ak[0] == 'COUNT_STAR'


def detect(seg, tree, col_map):
    if not _ENABLED:                   return None
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    if not P.has_limit(tree):          return None
    if wdb_sql._offset(tree):          return None
    where = tree.args.get('where') if P.has_where(tree) else None
    group = tree.args.get('group')
    if group is None or len(group.expressions) != 2:
        return None
    proj = tree.expressions
    keys = []; aggs = []                # 2 bare keys + COUNT(*) + any SUM/AVG(int col) payload
    for i, p in enumerate(proj):
        ak = wdb_sql._agg_kind(p)
        if ak is None:
            if wdb_sql._proj_colname(p) is None:
                return None
            keys.append(i)
        elif ak[0] == 'COUNT_STAR':
            aggs.append(('COUNT_STAR', None, i))
        elif ak[0] in ('SUM', 'AVG') and isinstance(ak[1], str):
            aggs.append((ak[0], ak[1], i))
        else:
            return None
    if len(keys) != 2 or sum(1 for a2 in aggs if a2[0] == 'COUNT_STAR') != 1:
        return None
    ci = next(a2[2] for a2 in aggs if a2[0] == 'COUNT_STAR')
    knames = [wdb_sql._proj_colname(proj[i]) for i in keys]
    gnames = [wdb_sql._colname(g) for g in group.expressions]
    if any(k is None for k in knames) or any(g is None for g in gnames):
        return None
    if set(knames) != set(gnames):
        return None
    cols = [col_map.get(k, k) if col_map else k for k in knames]
    for col in cols:
        if not P.columns_exist(seg, col):  return None
        if not P.not_positional(seg, col): return None
    if not P.no_deleted_rows(seg):         return None
    payphys = [col_map.get(c, c) if col_map else c for kind, c, _ in aggs if kind != 'COUNT_STAR']
    for pc in payphys:                     # payload must be integer-typed; cheap metadata check only
        c = seg.cols.get(pc)
        if c is None or c.get('dt') != 0:
            return None
    order = tree.args.get('order')
    if order is None or not order.expressions:
        unordered = True
    elif _order_is_count_desc(tree, proj, ci):
        unordered = False
        # Secondary ORDER BY keys are honored ONLY if they match this read's stored tiebreak exactly:
        # count DESC then gid ASC, i.e. the keys ascending in name-sorted physical order. Any other
        # secondary order would change plateau MEMBERSHIP, so decline (pairagg/scan serve it).
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
            if tb != sorted(cols):
                return None
    else:
        return None
    Vs = None                              # winners decode via seg.fetch (O(1) per cell) -- NEVER the
    for col in cols:                       # full value dict (SearchPhrase's dict is ~12 GB / 12.8 s to
        if seg.cols[col].get('mode') == 4:  # materialize, for 10 winners). Metadata-only gate here.
            return None
    afilter = None
    if where is not None:
        # only the composed count can apply it: COUNT alone, ordered by count, a LIMIT the head serves
        if payphys or unordered or wdb_sql._limit(tree) > _HEAD_N or not _COMPOSE[0]:
            return None
        afilter = _axis_filter(seg, where, cols, col_map)
        if afilter is None:
            return None
    return {'cols': cols, 'ci': ci, 'lim': wdb_sql._limit(tree), 'proj': proj,
            'knames': knames, 'V': Vs, 'order': tree.args.get('order'), 'unordered': unordered,
            'payphys': payphys, 'afilter': afilter}


def _axis_filter(seg, where, cols, col_map):
    """(column, neg, code) for a WHERE that is one '=' / '<>' between one of the two keys and a literal; None
    otherwise -- and None when the literal's code can't be found (never guess it absent)"""
    import wdb_wherescan as WS
    n = where.this if isinstance(where, E.Where) else where
    if not isinstance(n, (E.EQ, E.NEQ)):
        return None
    l, r = n.this, n.expression
    if not isinstance(l, E.Column):
        l, r = r, l
    if not isinstance(l, E.Column):
        return None
    col = col_map.get(l.name, l.name) if col_map else l.name
    if col not in cols or seg.cols[col].get('has_null'):
        return None
    val = WS._litval(r)
    if val is None:
        return None
    try:
        code = WS._code_of(seg, col, val)
    except Exception:
        return None
    if code is None:
        return None
    return (col, isinstance(n, E.NEQ), int(code))


def execute(seg, spec):
    """Top-K = a direct slice of the canonical count-ordered head (count DESC, gid ASC -- a plateau at
    the LIMIT boundary resolves deterministically instead of declining; the total-order validator
    agrees with this pick). A LIMIT deeper than the heavy set fills from `ones` (smallest singleton
    gids, count 1). SUM/AVG payload is computed for the <=K winners ONLY: one vectorized match pass
    over the cached codes locates the winners' member rows, and only those rows' payload is decoded."""
    global _HITS
    cols = spec['cols']; ci = spec['ci']; lim = spec['lim']; proj = spec['proj']
    knames = spec['knames']
    if lim <= 0:
        return None
    a, b = sorted(cols)
    composed = None
    if spec.get('afilter') is not None:
        composed = _compose_top(seg, cols, lim, spec['afilter'])
        if composed is None:
            return None                      # the structure can't apply a filter: hand on
    elif (_COMPOSE[0] and not spec.get('payphys') and not spec.get('unordered') and lim <= _HEAD_N
            and _CACHE.get((seg.path, (a, b), int(seg.N))) is None):
        composed = _compose_top(seg, cols, lim)
    if composed is not None:
        global _COMPOSED
        _COMPOSED += 1
        sel_gid, sel_cnt, Vb = composed
    else:
        loaded = _load(seg, cols)
        if loaded is None:
            return None
        head_gid, head_cnt, Vb, nheavy, bulk, ones, pos, _nd = loaded
    if composed is not None:
        pass
    elif lim <= head_gid.size:
        # hot path: the head is a canonical prefix -- top-K is a direct slice, plateaus included
        sel_gid = head_gid[:lim]; sel_cnt = head_cnt[:lim]
    elif lim <= nheavy:
        # beyond the head but within the heavy set: decode the gap-encoded bulk, canonical top-lim
        gid_all, cnt_all = _bulk_decode_all(bulk)
        cnt_all = cnt_all.astype(np.int64)
        o = np.lexsort((gid_all, -cnt_all))[:lim]
        sel_gid = gid_all[o]; sel_cnt = cnt_all[o]
    else:
        # LIMIT deeper than the heavy set: every heavy cell + deterministic singleton fill (count 1,
        # smallest gids first). Exception-defined pairs (nheavy ~ handful) live here.
        fill = lim - nheavy
        if ones is None:
            return None
        if fill > ones.size:
            if ones.size >= _ONES_N:
                return None                          # ones truncated: unseen singletons exist -> scan
            fill = ones.size                         # ones COMPLETE: every distinct cell is in hand,
                                                     # so the exact answer simply has < lim rows
        if nheavy == 0:
            hg = np.empty(0, np.int64); hc = np.empty(0, np.int64)
        elif nheavy <= head_gid.size:
            hg, hc = head_gid, head_cnt              # head already holds ALL heavy, canonical order
        else:
            gid_all, cnt_all = _bulk_decode_all(bulk)
            cnt_all = cnt_all.astype(np.int64)
            o = np.lexsort((gid_all, -cnt_all))
            hg, hc = gid_all[o], cnt_all[o]
        sel_gid = np.concatenate((hg, ones[:fill]))
        sel_cnt = np.concatenate((hc, np.ones(fill, np.int64)))
    # winner-only payload. Preferred: stored member-row positions (exception-defined pairs) -> a direct
    # gather+decode of ~K rows, no pass over the codes. Fallback: one vectorized match pass.
    payphys = spec.get('payphys') or []
    if payphys:
        if pos is not None:
            pg, pr = pos
            lo = np.searchsorted(pg, sel_gid, side='left')
            hi = np.searchsorted(pg, sel_gid, side='right')
            wp = np.repeat(np.arange(sel_gid.size), hi - lo)
            rows_idx = (np.concatenate([pr[l:h] for l, h in zip(lo, hi)])
                        if sel_gid.size else np.empty(0, np.int64))
            sel_pay = np.zeros((sel_gid.size, len(payphys)), np.int64)
            for p_i, pc in enumerate(payphys):
                vals = wdb_pairagg._survivor_payload(seg, pc, rows_idx)
                sel_pay[:, p_i] = np.rint(np.bincount(wp, weights=vals.astype(np.float64),
                                                      minlength=sel_gid.size)).astype(np.int64)
        else:
            gid_rows = seg._raw_codes(a).astype(np.int64) * Vb + seg._raw_codes(b).astype(np.int64)
            sel_pay = wdb_pairagg._winner_payload(seg, payphys, gid_rows, sel_gid, None)
    # materialize: BATCH decode each winner key column (values_at: touched dict chunks decompressed
    # once, restart blocks walked once, C-speed scalar conversion) -- no per-row fetch calls
    codesA = sel_gid // Vb
    codesB = sel_gid - codesA * Vb
    code_by_col = {a: codesA, b: codesB}
    decoded = {}
    for pi, p in enumerate(proj):
        if wdb_sql._agg_kind(p) is not None:
            continue
        col = cols[knames.index(wdb_sql._proj_colname(p))]
        if col not in decoded:
            decoded[col] = seg.values_at(col, code_by_col[col])
    cnt64 = sel_cnt.astype(np.int64)
    cnt_list = cnt64.tolist()
    # emission: per-projection COLUMN lists built vectorized, rows assembled by zip (C speed)
    emit = []
    payptr = 0
    for p in proj:
        ak = wdb_sql._agg_kind(p)
        if ak is None:
            emit.append(decoded[cols[knames.index(wdb_sql._proj_colname(p))]])
        elif ak[0] == 'COUNT_STAR':
            emit.append(cnt_list)
        elif ak[0] == 'SUM':
            emit.append(sel_pay[:, payptr].tolist()); payptr += 1
        else:                                        # AVG = exact integer sum / count (float64 both ways)
            emit.append((sel_pay[:, payptr].astype(np.float64) / cnt64).tolist()); payptr += 1
    rows_out = list(zip(*emit)) if emit else []
    rows_out = workers.finalize(rows_out, proj, spec['order'], lim)
    _HITS += 1
    return rows_out, [wdb_sql._alias(p) for p in proj]


def try_gridwalk(seg, tree, col_map):
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)
