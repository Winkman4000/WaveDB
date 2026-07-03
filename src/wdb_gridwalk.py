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
import wdb_sql
import wdb_pairagg
import workers
import wdb_policies as P
E = wdb_sql.E

_ENABLED = True           # live by default: gridwalk is the primary 2-key COUNT(*) top-K read
_HEAD_N = 200000          # count-ordered head depth; serves any LIMIT up to this, declines beyond
_ONES_N = 1024            # singleton fill head: the _ONES_N smallest count-1 gids, for LIMIT > nheavy
_POS_MAX = 8192           # store member-row positions when total stored cells <= this (payload O(K))
_POSROWS_MAX = 1 << 16    # ...AND their member rows <= this: positions cost bytes per ROW, and a
                          # low-card pair's handful of cells holds ALL 100M rows (measured 1.5 GB!)
_HITS = 0
_CACHE = {}               # (seg.path,(a,b),N) -> (head_gid, head_cnt, Vb, nheavy, bulk, ones, pos, nd)
_VCACHE = {}              # (seg.path,col) -> by-code value dict (decode)


def enable():
    global _ENABLED
    _ENABLED = True


def disable():
    global _ENABLED
    _ENABLED = False


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
    ca = seg._raw_codes(a).astype(np.int64); cb = seg._raw_codes(b).astype(np.int64)
    if ca.size == 0:
        return None
    Vb = int(cb.max()) + 1
    gid_all, cnt_all = np.unique(ca * Vb + cb, return_counts=True)   # filled cells, code order
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
        key = ca * Vb + cb
        st = np.sort(stored)
        p = np.searchsorted(st, key)
        valid = p < st.size
        p2 = np.where(valid, p, 0)
        match = valid & (st[p2] == key)
        rows = np.nonzero(match)[0]
        rgid = key[rows]
        o2 = np.argsort(rgid, kind='stable')
        pos = (np.ascontiguousarray(rgid[o2]), np.ascontiguousarray(rows[o2]))
    return head_gid, head_cnt, Vb, int(gid.size), bulk, ones, pos, nd


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
    if P.has_where(tree):              return None
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
    return {'cols': cols, 'ci': ci, 'lim': wdb_sql._limit(tree), 'proj': proj,
            'knames': knames, 'V': Vs, 'order': tree.args.get('order'), 'unordered': unordered,
            'payphys': payphys}


def execute(seg, spec):
    """Top-K = a direct slice of the canonical count-ordered head (count DESC, gid ASC -- a plateau at
    the LIMIT boundary resolves deterministically instead of declining; the total-order validator
    agrees with this pick). A LIMIT deeper than the heavy set fills from `ones` (smallest singleton
    gids, count 1). SUM/AVG payload is computed for the <=K winners ONLY: one vectorized match pass
    over the cached codes locates the winners' member rows, and only those rows' payload is decoded."""
    global _HITS
    cols = spec['cols']; ci = spec['ci']; lim = spec['lim']; proj = spec['proj']
    knames = spec['knames']
    loaded = _load(seg, cols)
    if loaded is None:
        return None
    head_gid, head_cnt, Vb, nheavy, bulk, ones, pos, _nd = loaded
    if lim <= 0:
        return None
    a, b = sorted(cols)
    if lim <= head_gid.size:
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
    # materialize: fetch-decode each winner cell's coordinates -- O(1) per value, no dict materialize
    codesA = sel_gid // Vb
    codesB = sel_gid - codesA * Vb
    code_by_col = {a: codesA, b: codesB}
    decoded = {}
    for pi, p in enumerate(proj):
        if wdb_sql._agg_kind(p) is not None:
            continue
        col = cols[knames.index(wdb_sql._proj_colname(p))]
        if col not in decoded:
            decoded[col] = [wdb_sql._pyval(seg.fetch(col, int(c))) for c in code_by_col[col]]
    cnt_list = sel_cnt.astype(np.int64).tolist()
    # emission plan hoisted OUT of the row loop: per-projection kind/source computed ONCE (at K=1M
    # the per-row _agg_kind/_proj_colname re-parse measured 7s of pure loop-invariant recompute)
    plan = []                       # ('key', column-list) | ('cnt', None) | ('sum'|'avg', payptr)
    payptr = 0
    for p in proj:
        ak = wdb_sql._agg_kind(p)
        if ak is None:
            plan.append(('key', decoded[cols[knames.index(wdb_sql._proj_colname(p))]]))
        elif ak[0] == 'COUNT_STAR':
            plan.append(('cnt', None))
        elif ak[0] == 'SUM':
            plan.append(('sum', payptr)); payptr += 1
        else:
            plan.append(('avg', payptr)); payptr += 1
    rows_out = []
    ap = rows_out.append
    for i in range(sel_gid.size):
        row = []
        for kind, src in plan:
            if kind == 'key':
                row.append(src[i])
            elif kind == 'cnt':
                row.append(cnt_list[i])
            elif kind == 'sum':
                row.append(int(sel_pay[i, src]))
            else:                                    # AVG = exact integer sum / count
                row.append(float(sel_pay[i, src]) / float(cnt_list[i]))
        ap(tuple(row))
    rows_out = workers.finalize(rows_out, proj, spec['order'], lim)
    _HITS += 1
    return rows_out, [wdb_sql._alias(p) for p in proj]


def try_gridwalk(seg, tree, col_map):
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)
