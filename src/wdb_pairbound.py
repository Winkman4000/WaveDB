"""wdb_pairbound -- THE PAIR BOUND (Jackson, 2026-10-28, Q32's family 1).

SELECT a, b, COUNT(*) AS c [, SUM(x), AVG(y)] FROM hits GROUP BY a, b ORDER BY c DESC LIMIT k

Every value lives at its rows on one row circle (N rows); a pair's count is the cancellation sum of its two values'
signatures, and its frequency-0 term bounds it: count(a, b) <= min(rows of a, rows of b). So with T a bar:
  1. both censuses, on every core (bincount_par)
  2. T guessed from a row sample (~1M rows: the sample's k-th pair count, scaled up, halved)
  3. one parallel pass keeps the rows whose a AND b both have >= T rows (their pair keys a * Vb + b, positions)
  4. one parallel sort of the kept keys, the runs = the kept pairs' exact counts, the top k
  5. THE CERTIFICATE: the k-th exact count >= T -> every dropped pair (count < T) is below it: exact. Else
     T = max(that k-th count, T // 2) and again from 3 (T = 1 keeps every row: the whole exact count)
  6. the winners' SUM / AVG: one pass over the kept rows (pb_winner_sums)
Measured on the PC (prototype, hot, the sample guess; DuckDB in brackets): ClientIP x RefererHash 11.7 s -> 0.90
(1.22), IPNetworkID x RefererHash 8.3 -> 0.85 (0.98), OriginalURL x ClientIP 10.5 -> 1.02 (1.98), Referer x RefererHash
8.9 -> 1.53 (2.07), LocalEventTime x RefererHash 21.3 -> 2.61 (2.06), three 30 s timeouts -> 1.1-2.5 s; all exact.
The switch: THE LINE below (measured on the Q32 board), neither key near-unique (V >= N/2: pairtop's walk), no WHERE
(this first form).
"""
import os
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0
_ROUNDS = [0]
# THE LINE (measured, Jackson OK'd 2026-10-28): the Q32 board run with the bound on every size (1,118 variants), against
# c88 pair by pair, by the larger key's V x the smaller key's V. The bound wins where
#   1. the larger key has >= 2^24 values (any smaller key):           0.13-0.88x, over 2x 40 -> 1
#   2. both keys >= 2^16 values and the larger >= 2^22:               0.14-0.80x, over 2x 13 -> 0
#   3. the larger key 2^16 .. 2^20 values and the smaller >= 2^10:    0.50x,      over 2x 14 -> 9 (IPNetworkID x a mid key)
# and loses everywhere else (both keys small: every row kept and sorted, 3.5-15x slower than the dense / bucket folds).
# Q32 estimate: over 2x 148 -> 96, timeouts 5 -> 0, 0.82x -> 0.74x. WDB_PAIRBOUND_LINE: line (default), all, off.
LINE = os.environ.get('WDB_PAIRBOUND_LINE', 'line')


def _on_line(Va, Vb):
    lo, hi = min(Va, Vb), max(Va, Vb)
    return hi >= (1 << 24) or (lo >= (1 << 16) and hi >= (1 << 22)) or ((1 << 16) <= hi < (1 << 20) and lo >= (1 << 10))
SAMPLE_ROWS = 1 << 20


def detect(seg, tree, col_map):
    if LINE == 'off':
        return None
    if tree.args.get('joins') or tree.args.get('with') or tree.args.get('having') or tree.args.get('distinct') \
            or tree.args.get('where') is not None or tree.args.get('offset'):
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 2 or not all(isinstance(x, E.Column) for x in g.expressions):
        return None
    cm = col_map or {}
    a, b = (cm.get(x.name, x.name) for x in g.expressions)
    if a == b:
        return None
    ca9, cb9 = seg.cols.get(a), seg.cols.get(b)
    if ca9 is None or cb9 is None:
        return None
    N = int(seg.N)
    for c9 in (ca9, cb9):
        if c9.get('has_null') or c9.get('mode') == 4:
            return None                          # mode 4: codes are row positions, a repeated value's rows never meet
    Va, Vb = int(ca9.get('V') or 0), int(cb9.get('V') or 0)
    if max(Va, Vb) * 2 >= N or Va * Vb >= (1 << 62) or min(Va, Vb) < 1:
        return None                              # a near-unique key is pairtop's walk
    if LINE != 'all' and not _on_line(Va, Vb):
        return None
    proj = []
    calias = None
    scols = []
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            cn = cm.get(inner.name, inner.name)
            if cn not in (a, b):
                return None
            proj.append(('A',) if cn == a else ('B',))
            continue
        ak = wdb_sql._agg_kind(inner)
        if ak is None:
            return None
        if ak[0] == 'COUNT_STAR':
            proj.append(('C',))
            if isinstance(p, E.Alias):
                calias = p.alias
            continue
        if ak[0] in ('SUM', 'AVG') and isinstance(ak[1], str):
            xc = cm.get(ak[1], ak[1])
            c9 = seg.cols.get(xc)
            if c9 is None or c9.get('dt') != 0 or c9.get('has_null') or int(c9.get('V') or 1 << 30) > (1 << 22):
                return None
            if xc not in scols:
                scols.append(xc)
            proj.append(('S' if ak[0] == 'SUM' else 'V', xc))
            continue
        return None
    if len(scols) > 2 or not any(k[0] == 'C' for k in proj):
        return None
    if not scols:
        # THE MEASURED SHAPE (2026-10-28): the line was measured on Q32's form (COUNT + SUM / AVG), where pairfold and
        # gridwalk were the competitors. On COUNT-only pairs (the Q16 board, 217 variants on the line) tripletop and
        # gridwalk's composed tally were faster: over 2x 6 -> 24, Q16 itself 0.33 -> 0.81 s. They keep those.
        return None
    ox = tree.args.get('order'); lx = tree.args.get('limit')
    if ox is None or lx is None or len(ox.expressions) != 1:
        return None
    o = ox.expressions[0]
    if not o.args.get('desc'):
        return None
    io = o.this
    if isinstance(io, E.Column):
        if calias is None or io.name != calias:
            return None
    elif not isinstance(io, E.Count) or isinstance(io.this, E.Distinct) or not isinstance(io.this, E.Star):
        return None
    try:
        lim = int(lx.expression.this)
    except Exception:
        return None
    if lim <= 0:
        return None
    import wdb_policies as P
    if not P.no_deleted_rows(seg):
        return None
    return {'a': a, 'b': b, 'scols': scols, 'lim': lim, 'proj_kinds': proj, 'proj': tree.expressions}


def _dv(seg, col):
    V = int(seg.cols[col]['V'])
    try:
        return np.asarray(seg._dict_ints_at(seg.cols[col], np.arange(V, dtype=np.int64)), np.float64)
    except Exception:
        return np.asarray([float(seg.fetch(col, v9)) for v9 in range(V)], np.float64)


def _kept(WK, ac, bc, ca, cb, T, VB, NT):
    cnt = np.zeros(NT, np.int64)
    WK.pb_survive_count(ac, bc, ca, cb, np.int64(T), np.int64(NT), cnt)
    offs = np.zeros(NT + 1, np.int64)
    offs[1:] = np.cumsum(cnt)
    keys = np.empty(int(offs[-1]), np.int64)
    pos = np.empty(int(offs[-1]), np.int64)
    WK.pb_survive_write(ac, bc, ca, cb, np.int64(T), np.int64(NT), offs, np.int64(VB), keys, pos)
    return keys, pos


def _runs(WK, keys):
    """the kept pairs (ascending keys) and their exact counts"""
    if keys.size == 0:
        return keys, np.empty(0, np.int64)
    s = WK.sort_keys_par(keys, np.int64(max(0, int(keys.max()).bit_length() - 12)))
    brk = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return s[brk], np.diff(np.r_[brk, s.size]).astype(np.int64)


def _guess(ac, bc, VB, k):
    """the bar's first guess: the k-th pair count of a ~1M-row sample (every step-th row), scaled by step, halved"""
    n = ac.size
    step = max(1, n // SAMPLE_ROWS)
    sk = ac[::step].astype(np.int64) * np.int64(VB) + bc[::step].astype(np.int64)
    _, sc = np.unique(sk, return_counts=True)
    if sc.size < k:
        return 1
    kth = int(np.partition(sc, sc.size - k)[sc.size - k])
    return max(1, int(kth * step * 0.5))


def execute(seg, spec):
    global _HITS
    import wdb_kernels as WK
    import numba
    a, b, k = spec['a'], spec['b'], spec['lim']
    NT = max(1, numba.get_num_threads())
    ac = np.ascontiguousarray(seg._raw_codes(a))
    bc = np.ascontiguousarray(seg._raw_codes(b))
    VA, VB = int(seg.cols[a]['V']), int(seg.cols[b]['V'])
    ca = WK.bincount_par(ac, VA)
    cb = WK.bincount_par(bc, VB)
    # THE KEY'S HIGH PART (2026-10-28): the sort buckets by the key's top bits, so the high part is the column whose most
    # common value holds the smaller share of the rows -- RefererHash's '' (19% of rows) high piled a fifth of the keys
    # into one bucket on one thread (RefererHash x EventTime: 18.5 s of 33 s in the sort). key = hi * V(lo) + lo
    swap = int(cb.max()) < int(ca.max())
    hc, lc, hcnt, lcnt, VL = (bc, ac, cb, ca, VA) if swap else (ac, bc, ca, cb, VB)
    T = _guess(hc, lc, VL, k)
    rounds = 0
    while True:
        rounds += 1
        keys, pos = _kept(WK, hc, lc, hcnt, lcnt, T, VL, NT)
        uk, uc = _runs(WK, keys)
        if uc.size >= k:
            # THE BAR (wdb_kernels.topk_bar): the counts are nearly all 1s -- numpy's partition took 4.9 s on them
            topi = WK.topk_bar(uc, k)
            kth = int(uc[topi[-1]])
            if kth >= T:
                break                            # THE CERTIFICATE: every dropped pair has fewer than T <= kth rows
            T = max(kth, T // 2)
        elif T <= 1:
            break                                # every row kept: the count is whole
        else:
            T = max(1, T // 2)
    _ROUNDS[0] = rounds
    m = min(k, uc.size)
    top = np.asarray(WK.topk_bar(uc, m), np.int64) if m else np.empty(0, np.int64)
    top = top[np.lexsort((uk[top], -uc[top]))]   # count DESC, then the pair code (ties: any is legal; this is fixed)
    wk = uk[top]
    wc = uc[top]
    sums = {}
    if spec['scols'] and m:
        x1 = spec['scols'][0]
        x2 = spec['scols'][1] if len(spec['scols']) > 1 else x1
        o = np.argsort(wk, kind='stable')
        wks = np.ascontiguousarray(wk[o])
        S1 = np.zeros((NT, m), np.float64)
        S2 = np.zeros((NT, m), np.float64)
        WK.pb_winner_sums(keys, pos, wks, np.ascontiguousarray(seg._raw_codes(x1)),
                          np.ascontiguousarray(seg._raw_codes(x2)), _dv(seg, x1), _dv(seg, x2) if x2 != x1 else _dv(seg, x1),
                          np.int64(NT), S1, S2)
        s1 = np.empty(m); s2 = np.empty(m)
        s1[o] = S1.sum(0); s2[o] = S2.sum(0)
        sums = {x1: s1, x2: s2}
    hcode = wk // VL
    lcode = wk % VL
    acode, bcode = (lcode, hcode) if swap else (hcode, lcode)
    va = seg.values_at(a, acode.astype(np.int64)) if m else []
    vb = seg.values_at(b, bcode.astype(np.int64)) if m else []
    out = []
    for j in range(m):
        x, y = va[j], vb[j]
        if isinstance(x, (bytes, bytearray)):
            x = x.decode('utf-8', 'replace')
        if isinstance(y, (bytes, bytearray)):
            y = y.decode('utf-8', 'replace')
        row = []
        for kind in spec['proj_kinds']:
            if kind[0] == 'A':
                row.append(wdb_sql._pyval(x))
            elif kind[0] == 'B':
                row.append(wdb_sql._pyval(y))
            elif kind[0] == 'C':
                row.append(int(wc[j]))
            elif kind[0] == 'S':
                row.append(int(round(float(sums[kind[1]][j]))))
            else:
                row.append(float(sums[kind[1]][j]) / max(1, int(wc[j])))
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
