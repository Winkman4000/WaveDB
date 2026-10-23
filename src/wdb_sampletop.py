"""THE SAMPLE LANE: a LIMIT with no ORDER BY asks for a sample, not a
ranking -- so serve a fresh lawful sample each run. Random group keys
are drawn from the big key's plist; each candidate's COMPLETE row slice
makes every returned count exact by construction. Jackson's ruling:
different runs, different ten -- the point of such a query is sampling,
and honest variance comes with the territory."""
import os
import numpy as np
import sqlglot.expressions as E
import wdb_sql

_HITS = 0


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') \
            or tree.args.get('where') or tree.args.get('having') \
            or tree.args.get('distinct') or tree.args.get('order'):
        return None
    lim = tree.args.get('limit')
    if lim is None:
        return None
    try:
        k = int(lim.expression.this)
    except Exception:
        return None
    if not (1 <= k <= 100) or tree.args.get('offset'):
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 2:
        return None
    cm = col_map or {}
    gcols = []
    for ge in g.expressions:
        if not isinstance(ge, E.Column):
            return None
        gcols.append(cm.get(ge.name, ge.name))
    proj = tree.expressions
    kinds = []
    for p in proj:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            nm = cm.get(inner.name, inner.name)
            if nm not in gcols:
                return None
            kinds.append(('K', nm))
            continue
        ak = wdb_sql._agg_kind(inner)
        if ak is None or ak[0] != 'COUNT_STAR':
            return None
        kinds.append(('C',))
    if gcols[0] == gcols[1]:
        return None
    lane = _lane(seg, gcols)
    if lane is not None:
        return {'b': lane[0], 'a': lane[1], 'k': k, 'kinds': kinds, 'proj': proj, 'road': 'lane'}
    if not _CENSUS[0] or not _census_ok(seg, gcols):
        return None
    # THE CENSUS ROAD's picker: the key with more values (below)
    v0, v1 = (int(seg.cols[c]['V']) for c in gcols)
    bcol, acol = (gcols[0], gcols[1]) if v0 >= v1 else (gcols[1], gcols[0])
    return {'b': bcol, 'a': acol, 'k': k, 'kinds': kinds, 'proj': proj, 'road': 'census'}


def _lane(seg, gcols):
    """the plist / stored-planes lanes' gate: (sampled key, partner) or None"""
    # the sampled key needs a plist; the partner needs point reads
    bcol = None
    for c in gcols:
        cc = seg.cols.get(c, {})
        if cc.get('code_enc') in (8, 12) and int(cc.get('V', 0)) >= 1000:
            bcol = c
    if bcol is None:
        return None
    import wdb_funnel as _F
    if not _F.plist_ready(seg, bcol) and seg.cols[bcol].get('code_enc') not in (8, 9):
        return None                      # THE VANILLA LAW: no lists, and no stored planes to draw from
    acol = [c for c in gcols if c != bcol][0]
    if seg.cols.get(acol, {}).get('code_enc') not in (0, 2, 3, 10, 12, 19):   # 19: point reads by block
        return None
    return bcol, acol


def _draws_from_planes(seg, b, k, rng):
    """THE VANILLA DRAW: the sparse dress already stores the non-default rows and their codes.
    A census of those codes (query-own, forgotten), k random small codes, ONE flagged pass for
    all their rows. Returns [(code, rows)] -- every slice COMPLETE, so every count is exact."""
    import wdb_kernels as _WK
    pos, lits, dflt = seg.e8_planes(b)
    lits = np.asarray(lits); V = int(seg.cols[b]['V'])
    cnt = _WK.bincount_par(lits, V)
    if 0 <= int(dflt) < V:
        cnt[int(dflt)] = 0
    pick = np.empty(0, np.int64)
    for cap in (max(64, 4 * k), 200000):         # tiny slices first, then anything countable
        for _ in range(8):                       # rejection draws: never order the whole census
            cand = rng.integers(0, V, 64 * k)    # (a permutation of 6M eligible codes cost 115 ms)
            ok = cand[(cnt[cand] > 0) & (cnt[cand] <= cap)]
            pick = np.unique(np.concatenate([pick, ok]))
            if pick.size >= k:
                break
        if pick.size >= k:
            break
    if pick.size < k:                            # a sparse census: take what exists, exactly
        elig = np.flatnonzero((cnt > 0) & (cnt <= 200000))
        pick = np.unique(np.concatenate([pick, elig[:k]]))
    pick = rng.permutation(pick)[:k]             # each code yields at least one pair
    flag = np.zeros(V, np.bool_); flag[pick] = True
    m = flag[lits]
    rows = np.asarray(pos)[m]; cs = lits[m]
    o = np.argsort(cs, kind='stable'); rows = rows[o]; cs = cs[o]
    out = []
    for c in pick.tolist():
        a = np.searchsorted(cs, c, 'left'); z = np.searchsorted(cs, c, 'right')
        out.append((int(c), rows[a:z].astype(np.int64)))
    return out


def execute(seg, spec):
    global _HITS
    if spec.get('road') == 'census':
        return _census_road(seg, spec)
    import wdb_funnel as _F
    if not _F.plist_ready(seg, spec['b']):
        rng = np.random.default_rng(); k = spec['k']; pairs = {}
        for c, rows9 in _draws_from_planes(seg, spec['b'], k, rng):
            if len(pairs) >= k:
                break
            ac9 = np.asarray(seg.codes_at(spec['a'], rows9)).astype(np.int64)
            u9, n9c = np.unique(ac9, return_counts=True)
            for aa, nn in zip(u9.tolist(), n9c.tolist()):
                if len(pairs) >= k:
                    break
                pairs[(c, aa)] = int(nn)
        return _emit(seg, spec, pairs)
    offsB, plB = _F._plist(seg, spec['b'])
    # a draw's count is two neighbouring offsets -- the census diff over all 6M codes (48 MB of
    # offsets, ~115 ms cold) was read to look up a few dozen of them (Q17)
    V = int(offsB.size) - 1
    rng = np.random.default_rng()
    k = spec['k']
    pairs = {}
    tries = 0
    cap9 = max(64, 4 * k)          # tiny slices first: draws touch few frames
    while len(pairs) < k and tries < 96:
        tries += 1
        if tries == 65:
            cap9 = 200000          # fallback phase: admit anything countable
        c = int(rng.integers(0, V))
        n9 = int(offsB[c + 1]) - int(offsB[c])
        if n9 == 0 or n9 > cap9:
            continue
        rows9 = np.asarray(plB[offsB[c]:offsB[c + 1]]).astype(np.int64)
        ac9 = np.asarray(seg.codes_at(spec['a'], rows9)).astype(np.int64)
        u9, n9c = np.unique(ac9, return_counts=True)
        for aa, nn in zip(u9.tolist(), n9c.tolist()):
            if len(pairs) >= k:
                break
            pairs[(c, aa)] = int(nn)
    return _emit(seg, spec, pairs)


def _emit(seg, spec, pairs):
    global _HITS
    k = spec['k']
    out = []
    items = list(pairs.items())[:k]
    # THE PLUCK IN ONE BATCH (as Q14's, Q24's): values_at decodes each touched dictionary chunk
    # once, in parallel -- 2k point fetches were serial chunk inflates (~61 ms cold, Q17)
    vbs = seg.values_at(spec['b'], np.array([bc for (bc, ac), nn in items], np.int64)) if items else []
    vas = seg.values_at(spec['a'], np.array([ac for (bc, ac), nn in items], np.int64)) if items else []
    for ((bc, ac), nn), vb, va in zip(items, vbs, vas):
        if isinstance(vb, (bytes, bytearray)):
            vb = vb.decode('utf-8', 'replace')
        if isinstance(va, (bytes, bytearray)):
            va = va.decode('utf-8', 'replace')
        row = []
        for kd in spec['kinds']:
            if kd[0] == 'C':
                row.append(nn)
            elif kd[1] == spec['b']:
                row.append(vb)
            else:
                row.append(va)
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]


# ---------------------------------------------------------------------------------------------------------------------
# THE CENSUS ROAD (2026-10-23, Q17's board: 57 of 70 pairs over 2x DuckDB, geomean 3.26x, worst 24.5 s on gridwalk).
# GROUP BY a, b LIMIT k with no ORDER BY asks for any k real groups with exact counts. The lanes above need a plist or
# an enc-8 key's stored planes; every other pair fell to a read that counts EVERY pair. Here, for any encoding:
#   1. the picker is the key with more values (its rarest values have the fewest rows);
#   2. its census (rows per value, one parallel pass);
#   3. k values drawn at random among those with few rows (cap 4k, then 200,000 -- the lanes' own ladder; Jackson's
#      ruling: a fresh lawful sample each run). R = their rows, known before any row is read;
#   4. the road's cost predicted from R (the frames it will touch, each key's encoding and width -- _road_cost below)
#      against the full count's (from the two keys' value counts -- _full_cost): past it, decline (a full count's job). Measured crossing x cost per row read held at
#      ~4 ns over an 8x range of artificial costs (csweep.py) and on 22 real pickers (2.6-5.4 ns);
#   5. every row of the drawn values (one flag scan of the picker), the partner read at those rows only;
#   6. the pairs counted. Every row of each drawn value was read, so every count is exact; k distinct drawn values
#      give at least k distinct pairs.
_CENSUS = [os.environ.get('WDB_SAMPLE_CENSUS', '1') != '0']
_CENSUS_HITS = 0


def _census_ok(seg, cols):
    if type(seg).__name__ != 'Segment':
        return False
    for c in cols:
        cc = seg.cols.get(c)
        if cc is None or cc.get('mode') not in (0, 1, 2) or seg._effective(c) is not None:
            return False
    return True


def _draw(cn, k, rng):
    """k live codes with few rows: random draws under a rising cap; a sparse or heavy census takes the fewest-row
    live codes (sorted only when the live codes are few; past 1M of them, a random choice)"""
    V = int(cn.size)
    pick = np.empty(0, np.int64)
    if V == 0:
        return pick
    for cap in (max(64, 4 * k), 200000):
        for _ in range(8):
            cand = rng.integers(0, V, 64 * k)
            ok = cand[(cn[cand] > 0) & (cn[cand] <= cap)]
            pick = np.unique(np.concatenate([pick, ok]))
            if pick.size >= k:
                break
        if pick.size >= k:
            break
    if pick.size < k:
        live = np.flatnonzero(cn > 0)
        if live.size <= k:
            return live.astype(np.int64)
        rest = np.setdiff1d(live, pick, assume_unique=True)
        need = k - pick.size
        if rest.size > (1 << 20):
            more = rng.choice(rest, need, replace=False)
        else:
            more = rest[np.argsort(cn[rest], kind='stable')[:need]]
        pick = np.concatenate([pick, more])
    return rng.permutation(pick)[:k].astype(np.int64)


def _census_road(seg, spec):
    """steps 2-6 above; None declines (R past the line, or the scan disagreeing with the census)"""
    global _CENSUS_HITS
    import wdb_wherescan as WS
    b, a, k = spec['b'], spec['a'], spec['k']
    N = int(seg.N)
    Vb = int(seg.cols[b]['V'])
    import time
    t0 = time.perf_counter()
    cn = np.asarray(seg.raw_census(b, Vb))[:Vb]
    t_census = time.perf_counter() - t0
    rng = np.random.default_rng()
    pick = _draw(cn, k, rng)
    if pick.size == 0:
        return _emit(seg, spec, {})
    R = int(cn[pick].sum())
    if not _road_pays(seg, b, spec['a'], R, t_census, pick.size):
        return None
    flag = np.zeros(Vb + 1, np.bool_)
    flag[pick] = True
    rows = np.asarray(WS._scan_flag(seg, b, flag, 0, N), np.int64)
    if rows.size != R:
        return None
    bc = np.asarray(seg.codes_at(b, rows)).astype(np.int64)
    ac = np.asarray(seg.codes_at(a, rows)).astype(np.int64)
    ps = np.sort(pick)
    li = np.searchsorted(ps, bc)
    W = int(seg.cols[a]['V']) + 1
    key = li * W + ac
    if ps.size * W <= (1 << 24):
        # the drawn values x the partner's codes fit a small board: counted, not sorted. (Past it -- at most k drawn
        # values against a partner past 1.6M codes -- the drawn values are rare ones and R is small: the sort is.)
        cnt = np.bincount(key, minlength=ps.size * W)
        keys = np.flatnonzero(cnt)
        counts = cnt[keys]
    else:
        keys, counts = np.unique(key, return_counts=True)
    take = rng.permutation(keys.size)[:k]
    pairs = {}
    for i in take.tolist():
        pairs[(int(ps[int(keys[i]) // W]), int(keys[i]) % W)] = int(counts[i])
    _CENSUS_HITS += 1
    return _emit(seg, spec, pairs)


# THE ROAD'S COST (2026-10-23, Jackson: "the number of frames, the encoding and the width in bits"). Reading a column at
# R rows costs (frames touched x the cost of one frame) + R x the cost of one row. R rows scattered over F frames touch
# F x (1 - e^(-R / F)) of them (1,000 rows on 1,526 frames: 733 predicted, 749 measured). Measured per column with
# codes_at at 1K..10M scattered rows, caches flushed (partsweep.py):
#   frame cost: enc 3 69-74 us for a 64K-row frame at any width; larger frames grow with their bytes (512K rows x 4 B
#   318 us, 4M rows x 2 B 384 us); enc 18 92-95 us; enc 20 53 us; enc 19 0.8-5 us (point reads by block); enc 5 5.5 us.
#   row cost: 0.5-3.3 ns for the unframed encodings (0, 1, 2, 9, 10, 17), 7.5 (enc 8), 14.5 (enc 2 past 8 bits),
#   11.5 (enc 5), 7 (enc 18), 3-4 (enc 19 / 20), 1.5-2.3 (enc 3). Unframed enc 1 / 9 carry a fixed 5-14 ms.
# Each entry: (frame base us, bytes decompressed per us past it, row ns, fixed s); an encoding not listed takes the
# dearest row seen (conservative: the road declines sooner).
_COST = {3: (75.0, 6500.0, 2.3, 0.0), 18: (95.0, 6500.0, 7.5, 0.0), 20: (55.0, 6500.0, 3.1, 0.0),
         19: (5.5, 0.0, 4.1, 0.0), 5: (5.6, 0.0, 11.6, 0.0),
         0: (0.0, 0.0, 1.0, 0.0), 1: (0.0, 0.0, 0.8, 0.007), 2: (0.0, 0.0, 3.0, 0.0), 8: (0.0, 0.0, 7.5, 0.012),
         9: (0.0, 0.0, 2.6, 0.014), 10: (0.0, 0.0, 3.3, 0.004), 17: (0.0, 0.0, 0.7, 0.001)}
_COST_DEFAULT = (100.0, 6500.0, 15.0, 0.015)
_COUNT_NS = 6.0                   # the count per row read (board), measured beside a cheap partner
_FIND_NS = 1.0                    # + per row, per halving: which drawn value a row holds (a search of m values;
                                  # the real road draws m <= k values)
_FULL_NS = [float(os.environ.get('WDB_SAMPLE_FULL_NS', '3.7'))]   # the full count's cost per table row: the dense
                                                                  # grid counts measured 3.7-5.5 ns (the cheapest 0.37 s)


def _read_cost(seg, col, R):
    """seconds to read col's codes at R scattered rows: frames touched x frame cost + rows x row cost (+ fixed)"""
    c = seg.cols[col]
    enc = int(c.get('code_enc') or 0)
    if enc == 2 and int(c.get('bits') or 0) > 8:
        base, bpu, row, fix = 0.0, 0.0, 14.5, 0.0
    else:
        base, bpu, row, fix = _COST.get(enc, _COST_DEFAULT)
    t = fix + R * row * 1e-9
    BR = c.get('BR') or c.get('e19BR') or c.get('e20BR')
    if BR and base > 0:
        F = (int(seg.N) + int(BR) - 1) // int(BR)
        touched = F * (1.0 - np.exp(-float(R) / F))
        fbytes = float(BR) * float(c.get('cwidth') or max(1, (int(c.get('bits') or 8) + 7) // 8))
        per = max(base, fbytes / bpu) if bpu else base
        t += touched * per * 1e-6
    return t


def _road_cost(seg, b, a, R, t_census, m):
    """the road past its census: one scan of the picker (a whole-column pass, estimated as the census just paid),
    both keys read at the R rows, the count (each row's drawn value found among m: log2(m) halvings)"""
    find = _FIND_NS * max(1.0, float(np.log2(max(2, int(m)))))
    return t_census + _read_cost(seg, b, R) + _read_cost(seg, a, R) + R * (_COUNT_NS + find) * 1e-9


_FULL_WIDE_NS = [float(os.environ.get('WDB_SAMPLE_FULL_WIDE_NS', '45'))]   # past the dense board: the full count
                                                    # measured 49-86 ns per table row (Title / Referer / UserID / WatchID /
                                                    # RefererHash x IsMobile, 4.9-8.6 s)
_DENSE_CELLS = 1 << 24            # gridwalk's dense board: (values of a) x (values of b) up to 16.7M cells


def _full_cost(seg, b, a):
    """the full count's cost, from the two keys' value counts: the dense grid count when its board fits (measured
    0.37-0.70 s, 3.7-7 ns per table row; the cheapest taken), else the wide count"""
    cells = (int(seg.cols[b]['V']) + 1) * (int(seg.cols[a]['V']) + 1)
    ns = _FULL_NS[0] if cells <= _DENSE_CELLS else _FULL_WIDE_NS[0]
    return ns * 1e-9 * int(seg.N)


def _road_pays(seg, b, a, R, t_census, m):
    return _road_cost(seg, b, a, R, t_census, m) <= _full_cost(seg, b, a)
