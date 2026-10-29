"""wdb_pairfold -- Q30's one-walk pair board (the eight taxes become one).

SELECT a, b, COUNT(*) c, SUM(x), AVG(y) FROM hits
WHERE f <> '' GROUP BY a, b ORDER BY c DESC LIMIT n

The filter is free (the planes hand the typed positions, memoized). One
fused kernel walks those positions, reads all four columns at the row,
radix-buckets by b, then per-bucket sorts and folds count + both
dictionary-valued sums. Ten pairs decode at the pluck.
"""
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0
_OVERLAP = [__import__('os').environ.get('WDB_PF_OVERLAP', '1') == '1']  # A/B: 0 decodes a after the census, via codes_band
_PFWARM = [__import__('os').environ.get('WDB_PF_WARM', '1') == '1']      # A/B: 0 leaves the plist to page faults
_DENSE = [__import__('os').environ.get('WDB_PF_DENSE', '1') == '1']      # A/B: 0 buckets and sorts every pair space
_UNIQUE_HANDOFF = [__import__('os').environ.get('WDB_PF_UNIQUE', '1') == '1']  # A/B: 0 keeps near-unique partners
SMALL_MAX = 4 << 20          # the smaller key's values for pairfold (both keys past it: pairtop's)
DENSE_MAX = 1 << 18         # pair slots Va x Vb for the dense fold: 16 thread tables x 3 x 8 B x 256K = 100 MB
# THE HEAVY VALUES ON EVERY CORE (Jackson, 2026-10-28, the Q32 board: 242 of its 254 pairfold variants over 2x had one
# bucket holding > n/64 of the rows, sorted or counted by ONE thread while 15 idled -- ResponseEndTiming x OpenstatAdID
# 24.2 s, 22.9 of them in pr_fold2, OpenstatAdID's '' on 99.5M rows). b values with more than n // HEAVY_DIV rows (n =
# the rows read) are counted on every core into the threads' own tables of (heavy values x Va) slots, at most
# HEAVY_SLOTS (past it: the buckets, as before); the other rows go the buckets' way. Prototype (this PC, same counts on
# 13 pairs): 24.2 -> 0.22 s (DuckDB 0.21), SendTiming x UserAgentMajor 1.22 -> 0.29 (0.22). HEAVY_DIV 0: never.
HEAVY_DIV = int(__import__('os').environ.get('WDB_PF_HEAVY_DIV', '64'))
HEAVY_SLOTS = int(__import__('os').environ.get('WDB_PF_HEAVY_SLOTS', str(1 << 18)))
_HEAVY_HITS = [0]
_DENSED = 0
# THE ONE PASS (2026-10-29, Jackson's fix 3): every thread adds its rows into its own hash table, the tables merge
# on every core -- one read of the rows, no census, no prune, no buckets. It wins where FEW pairs occur (the tables
# stay small) and loses where nearly every row is its own pair (the tables grow to the rows). The line is on the
# occurring pairs, estimated from a sample of the rows before the pass.
_HT = [__import__('os').environ.get('WDB_PF_HT', '1') == '1']
HT_LINE = float(__import__('os').environ.get('WDB_PF_HT_LINE', '262144'))
HT_HEAVY = float(__import__('os').environ.get('WDB_PF_HT_HEAVY', '0.95'))   # the heavy share past which the heavy fold wins
HT_SAMPLE = 1 << 17          # rows in the estimate's sample (2^18: 3x the time, the same decisions)
HT_SAMPLE1 = 1 << 15         # THE STAGED ESTIMATE: this many rows first ...
HT_STAGE = 2.0               # ... and an estimate past HT_STAGE x HT_LINE there is past the line
HT_CBMAX = 20                # largest per-thread table, 2^20 slots (past it: pairfold's buckets)
_HT_HITS = [0]
_HT_SPILL = [0]
_HT_TRACE = __import__('os').environ.get('WDB_PF_HT_TRACE')


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') \
            or tree.args.get('having') or tree.args.get('distinct'):
        return None
    w = tree.args.get('where')
    cm = col_map or {}
    if w is None:
        fcol = None                              # NO FILTER (2026-10-17, Q32's form): every row is a position
    else:
        if not isinstance(w.this, E.NEQ):
            return None
        wl, wr = w.this.this, w.this.expression
        if not (isinstance(wl, E.Column) and isinstance(wr, E.Literal)
                and wr.is_string and str(wr.this) == ''):
            return None
        fcol = cm.get(wl.name, wl.name)
        fc = seg.cols.get(fcol)
        if fc is None or fc.get('code_enc') not in (8, 9) or fc.get('has_null'):
            return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 2 \
            or not all(isinstance(x, E.Column) for x in g.expressions):
        return None
    acol = cm.get(g.expressions[0].name, g.expressions[0].name)
    bcol = cm.get(g.expressions[1].name, g.expressions[1].name)
    ac, bc = seg.cols.get(acol), seg.cols.get(bcol)
    if ac is None or bc is None or ac.get('has_null') or bc.get('has_null'):
        return None
    Va0, Vb0 = int(ac.get('V') or 1 << 30), int(bc.get('V') or 1 << 30)
    if Va0 > 256 and Vb0 < Va0:
        acol, bcol = bcol, acol                  # the smaller key rides the low bits
        ac, bc = bc, ac
        Va0, Vb0 = Vb0, Va0
    # THE SMALL KEY'S OWN BITS (2026-10-17): the low field was one byte (a <= 256 values); it is now as
    # wide as the smaller key needs, so a mid key pairs too. Both keys past SMALL_MAX values (WatchID x
    # ClientIP) stay pairtop's; the packed pair must fit 62 bits
    if Va0 > SMALL_MAX:
        return None
    abits = max(1, (Va0 - 1).bit_length())
    if max(1, (Vb0 - 1).bit_length()) + abits > 62:
        return None
    # A NEAR-UNIQUE PARTNER IS PAIRTOP'S (2026-10-17, the c3 Q31 board): with the larger key holding a value
    # per ~row (WatchID: 100M values in 100M rows) every pair counts ~1 and pairtop's walk wins -- SendTiming x
    # WatchID 0.30 s there vs 1.19 s here; the Q31 board's WatchID family 12.5 s vs 25.2 s
    if _UNIQUE_HANDOFF[0] and Vb0 >= max(1, int(getattr(seg, 'N', 0) or 0)) // 2:
        return None
    proj = []                                    # ('A',)('B',)('C',)('S',col)('V',col)
    calias = None
    scols = []
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            cn = cm.get(inner.name, inner.name)
            if cn == acol:
                proj.append(('A',)); continue
            if cn == bcol:
                proj.append(('B',)); continue
            return None
        ak = wdb_sql._agg_kind(inner)
        if ak is None:
            return None
        if ak[0] == 'COUNT_STAR':
            proj.append(('C',))
            if isinstance(p, E.Alias):
                calias = p.alias
            continue
        if ak[0] in ('SUM', 'AVG'):
            xc = cm.get(ak[1], ak[1])
            c9 = seg.cols.get(xc)
            if c9 is None or c9.get('dt') != 0 or c9.get('has_null') \
                    or int(c9.get('V') or 1 << 30) > 65536:
                return None
            if xc not in scols:
                scols.append(xc)
            proj.append(('S' if ak[0] == 'SUM' else 'V', xc)); continue
        return None
    if len(scols) > 2 or not any(k[0] == 'C' for k in proj):
        return None
    if fcol is None and not scols:
        return None          # an unfiltered COUNT-only pair is gridwalk's composed tally (wdb_gridwalk._compose_top)
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
    elif not isinstance(io, E.Count) or isinstance(io.this, E.Distinct):
        return None
    try:
        lim = int(lx.expression.this)
    except Exception:
        return None
    import wdb_policies as P
    if not P.no_deleted_rows(seg):
        return None
    return {'f': fcol, 'a': acol, 'b': bcol, 'scols': scols, 'lim': lim, 'abits': abits,
            'projkinds': proj, 'proj': tree.expressions}


def _dv(seg, col):
    V = int(seg.cols[col]['V'])
    try:
        return np.asarray(seg._dict_ints_at(seg.cols[col],
                                            np.arange(V, dtype=np.int64)),
                          np.float64)
    except Exception:
        return np.asarray([float(seg.fetch(col, v9)) for v9 in range(V)],
                          np.float64)


def execute(seg, spec):
    global _HITS, _DENSED
    import wdb_kernels as _WK
    # THE PLIST PATH (Jackson's audit: the counts were in the data all
    # along). The big key's plist offsets ARE the census (diff), its
    # positions ARE each candidate's rows -- so the marginal-bound law
    # runs on slices: candidates descending, partner gathered at their
    # rows, one tiny bincount, the certificate verbatim. Zero storage,
    # zero N-arrays, nothing to purge.
    if spec['f'] is not None and not spec.get('scols') and seg.cols.get(spec['a'], {}).get('code_enc') == 3 \
            and seg.cols.get(spec['b'], {}).get('code_enc') == 8 \
            and int(seg.cols[spec['a']].get('V') or 999) <= 256 \
            and __import__('wdb_funnel').plist_ready(seg, spec['b']):    # THE VANILLA LAW: lists that may serve
        import wdb_funnel as _F
        from concurrent.futures import ThreadPoolExecutor as _TPa
        # THE OVERLAP (Q14 cold): the small key's full decode is CPU (~70 ms warm or cold through
        # codes_band; the pipelined 16-lane decode ~45) and the census is storage (48 MB of offsets,
        # ~50 ms) -- the decode runs beside the census read instead of after it
        _exa = _TPa(max_workers=1)
        futa = _exa.submit(seg._raw_codes, spec['a']) if _OVERLAP[0] else None
        try:
            offsB, plB = _F._plist(seg, spec['b'])
            if _PFWARM[0]:
                _F.warm_plist(seg, spec['b'], offsets=True)
            offsB = np.asarray(offsB, dtype=np.int64)
            cnt_b = np.diff(offsB)
            dflt9 = int(seg.cols[spec['b']].get('e8d', -1))
            if 0 <= dflt9 < cnt_b.size:
                cnt_b = cnt_b.copy()
                cnt_b[dflt9] = 0                     # the WHERE excludes ''
            BV = cnt_b.size
            k = spec['lim']
            aC = np.ascontiguousarray(np.asarray(
                futa.result() if futa is not None else seg.codes_band(spec['a'], 0, seg.N)).astype(np.uint8, copy=False))
        finally:
            _exa.shutdown(wait=True)
        nz = int((cnt_b > 0).sum())
        # THE BAR, not the sort: the loop reads only the top M candidates and the count just below
        # them -- topk_bar selects those M + 1 (a lowered bar, parallel compares) where a full argsort
        # of all 6M SearchPhrase counts cost ~123 ms hot and cold (Q14)
        M = max(64, 4 * k)
        while True:
            M9 = min(M, nz)
            top9 = _WK.topk_bar(cnt_b, min(M9 + 1, nz))
            cands = top9[:M9]
            max_excl = int(cnt_b[top9[M9]]) if M9 < nz else 0
            if _PFWARM[0]:                       # the candidates' lists: parallel reads, not faults
                _F.warm_plist(seg, spec['b'], codes=cands)
            rows9 = np.concatenate([np.asarray(plB[offsB[c]:offsB[c + 1]])
                                    for c in cands.tolist()]).astype(np.int64)
            cidr = np.repeat(np.arange(M9, dtype=np.int64),
                             cnt_b[cands])
            pk = (cidr << 8) | aC[rows9].astype(np.int64)
            pcnt = np.bincount(pk, minlength=M9 << 8)
            k9 = min(k, int((pcnt > 0).sum()))
            if k9 == 0 and max_excl > 0:
                M *= 4
                continue
            topi = np.argpartition(-pcnt, max(0, k9 - 1))[:k9]
            topi = topi[np.argsort(-pcnt[topi], kind='stable')]
            p10 = int(pcnt[topi[-1]]) if k9 else 0
            if max_excl <= p10 or M9 >= nz:
                break
            M *= 4
        out = []
        tl9 = topi.tolist()
        # THE PLUCK IN ONE BATCH (as Q24's): values_at decodes each touched dictionary chunk once,
        # in parallel -- k point fetches were k serial chunk inflates (~23 ms cold for 10 rows)
        vbs = seg.values_at(spec['b'], np.array([int(cands[j >> 8]) for j in tl9], np.int64)) if tl9 else []
        vas = seg.values_at(spec['a'], np.array([j & 0xFF for j in tl9], np.int64)) if tl9 else []
        for j, va, vb in zip(tl9, vas, vbs):
            if isinstance(va, (bytes, bytearray)):
                va = va.decode('utf-8', 'replace')
            if isinstance(vb, (bytes, bytearray)):
                vb = vb.decode('utf-8', 'replace')
            row = []
            for p in spec['projkinds']:
                if p[0] == 'A':
                    row.append(va)
                elif p[0] == 'B':
                    row.append(vb)
                else:
                    row.append(int(pcnt[j]))
            out.append(tuple(row))
        _HITS += 1
        return out, [wdb_sql._alias(p) for p in spec['proj']]
    if spec['f'] is None:
        pos = None                               # no filter: the kernels walk every row in order
    else:
        pl = seg.e8_planes(spec['f'])
        pos = np.ascontiguousarray(np.asarray(pl[0], dtype=np.int64))
    bc9 = np.asarray(seg._raw_codes(spec['b']))
    BV = int(seg.cols[spec['b']]['V'])
    k = spec['lim']
    scols = spec['scols']
    x1 = scols[0] if scols else spec['a']
    x2 = scols[1] if len(scols) > 1 else x1
    VA = int(seg.cols[spec['a']]['V'])
    # THE SUMS' VALUE TABLES, once (2026-10-17): with no SUM/AVG the payload is never read, so no table is
    # decoded -- the first version decoded the small key's values as numbers anyway and a TEXT key crashed
    # (MobilePhoneModel x SearchPhrase, COUNT only, WHERE SearchPhrase <> '')
    dv1 = _dv(seg, x1) if scols else np.zeros(max(1, VA), np.float64)
    dv2 = _dv(seg, x2) if len(scols) > 1 else dv1
    if _DENSE[0] and VA * BV <= DENSE_MAX:
        # THE SMALL PAIR SPACE: every pair gets its own slot, counted exactly -- no census, no
        # prune, no certificate needed, no buckets, no sort
        ac9 = np.asarray(seg._raw_codes(spec['a']))
        cnt, s1, s2 = _WK.pf_dense(pos, ac9, bc9, np.asarray(seg._raw_codes(x1)), np.asarray(seg._raw_codes(x2)),
                                   dv1, dv2, VA, VA * BV)
        nzk = np.flatnonzero(cnt)
        ucnt = cnt[nzk]; us1 = s1[nzk]; us2 = s2[nzk]
        ukey = ((nzk // VA) << spec['abits']) | (nzk % VA)
        order = np.lexsort((ukey, -ucnt))[:min(k, ucnt.size)]      # count DESC, then (b, a) ASC
        _DENSED += 1
        return _pluck(seg, spec, order, ukey, ucnt, {x1: us1, x2: us2})
    if _HT[0]:
        # THE ONE PASS first: under the line (few pairs occur) it is the whole answer
        try:
            import numba
            T8 = max(1, numba.get_num_threads())
        except Exception:
            T8 = 1
        ac9 = np.asarray(seg._raw_codes(spec['a']))     # decoded once: the buckets reuse them when it stands aside
        xc9 = np.asarray(seg._raw_codes(x1))
        yc9 = xc9 if x2 == x1 else np.asarray(seg._raw_codes(x2))
        r8 = _onepass(pos, ac9, bc9, xc9, yc9, dv1, dv2, spec['abits'], 3 if scols else 1, T8,
                      tag='%s|%s|%s' % (spec['a'], spec['b'], spec['f']))
        if r8 is not None:
            ukey, ucnt, us1, us2 = r8
            k9 = min(k, ucnt.size)
            if k9:
                order = np.argpartition(-ucnt, k9 - 1)[:k9]
                order = order[np.argsort(-ucnt[order], kind='stable')]
            else:
                order = np.zeros(0, np.int64)
            _HT_HITS[0] += 1
            return _pluck(seg, spec, order, ukey, ucnt, {x1: us1, x2: us2})
    memo = seg.__dict__.setdefault('_censusmemo', {})
    ckey = ('pf', spec['f'], spec['b'])
    cnt_b = memo.get(ckey)
    if cnt_b is None:                        # the bound: a pair can never
        # (2026-10-28) on every core: numpy's bincount was one thread, 170-224 ms of the Q32 pairs (THE HEAVY VALUES)
        cnt_b = _WK.bincount_par(bc9 if pos is None else bc9[pos], BV)    # outscore its b's total
        memo[ckey] = cnt_b
    if not _HT[0]:
        ac9 = np.asarray(seg._raw_codes(spec['a']))
        xc9 = np.asarray(seg._raw_codes(x1))
        yc9 = xc9 if x2 == x1 else np.asarray(seg._raw_codes(x2))
    SH = max(0, BV.bit_length() - 12)
    try:
        import numba
        T9 = max(1, numba.get_num_threads())
    except Exception:
        T9 = 1
    hv = None                                    # the heavy values' pairs: (key, count, sum 1, sum 2)
    if HEAVY_DIV > 0:
        nrows9 = int(seg.N) if pos is None else int(pos.size)
        H9 = np.flatnonzero(cnt_b > nrows9 // HEAVY_DIV)
        if H9.size and H9.size * VA <= HEAVY_SLOTS:
            # THE HEAVY VALUES ON EVERY CORE: 1. their pairs counted exactly, every thread its own table;
            # 2. the other rows (the light ones) are the positions the buckets walk; 3. the heavy values leave the
            # census the bound reads (counted exactly, they are never "excluded")
            hidx9 = np.full(BV, -1, np.int64)
            hidx9[H9] = np.arange(H9.size, dtype=np.int64)
            W9 = int(H9.size) * VA
            C9 = np.zeros((T9, W9 + 16), np.int64)           # rows padded apart (THE PADDED BOARDS)
            A9 = np.zeros((T9, W9 + 16), np.float64)
            B9 = np.zeros((T9, W9 + 16), np.float64)
            lc9 = np.zeros(T9, np.int64)
            allr9 = pos is None
            pos9 = np.zeros(1, np.int64) if allr9 else pos
            _WK.pf_heavy_fold(pos9, allr9, ac9, bc9, xc9, yc9, hidx9, np.int64(VA), dv1, dv2, C9, A9, B9, lc9)
            c9 = C9.sum(0)[:W9]
            nzs = np.flatnonzero(c9)
            hv = ((H9[nzs // VA].astype(np.int64) << spec['abits']) | (nzs % VA).astype(np.int64),
                  c9[nzs], A9.sum(0)[:W9][nzs], B9.sum(0)[:W9][nzs])
            off9 = np.zeros(T9 + 1, np.int64)
            off9[1:] = np.cumsum(lc9)
            lpos9 = np.empty(int(off9[-1]), np.int64)
            _WK.pf_light_rows(pos9, allr9, bc9, hidx9, off9, lpos9)
            pos = lpos9
            cnt_b = cnt_b.copy()
            cnt_b[H9] = 0
            _HEAVY_HITS[0] += 1
    M = 4096
    nz = int((cnt_b > 0).sum())
    while True:
        M9 = min(M, nz)
        if M9 >= nz:                         # no prune possible: whole board
            spos = pos
            max_excl = 0
        else:
            hkey = ('pfh', spec['f'], spec['b'], M9) if hv is None else ('pfhL', spec['f'], spec['b'], M9, HEAVY_DIV)
            hit = memo.get(hkey)                   # the heavy set is static
            if hit is None:                        # per (filter, column, M):
                thr_idx = np.argpartition(-cnt_b, M9 - 1)[:M9]   # select once,
                heavy = np.zeros(BV, np.bool_)                   # remember
                heavy[thr_idx] = True
                max_excl = int(cnt_b[~heavy].max()) if M9 < BV else 0
                spos9 = _WK.pf_prune_all(bc9, heavy) if pos is None else _WK.pf_prune(pos, bc9, heavy)
                memo[hkey] = (max_excl, spos9)
                hit = memo[hkey]
            max_excl, spos = hit
        if spos is not None and spos.size == 0:      # every row was a heavy value's
            ucnt = np.zeros(0, np.int64); us1 = np.zeros(0); us2 = np.zeros(0); ukey = np.zeros(0, np.int64)
        else:
            if spos is None:                     # no filter, no prune: every row
                key, pay, offs = _WK.pr_scatter_all(ac9, bc9, xc9, yc9, SH, spec['abits'], T9)
            else:
                key, pay, offs = _WK.pr_scatter2(spos, ac9, bc9, xc9, yc9, SH, spec['abits'], T9)
            n = key.size
            ucnt = np.zeros(n, np.int64)
            us1 = np.zeros(n, np.float64)
            us2 = np.zeros(n, np.float64)
            ukey = np.zeros(n, np.int64)
            nruns = np.zeros(1 << 12, np.int64)
            _WK.pr_fold2(key, pay, offs, SH, spec['abits'], dv1, dv2,    # a heavy bucket is counted, not sorted
                         ucnt, us1, us2, ukey, nruns)
            live = np.zeros(n, bool)
            for b in range(nruns.size):
                if nruns[b]:
                    live[offs[b]:offs[b] + nruns[b]] = True
            ucnt = ucnt[live]; us1 = us1[live]; us2 = us2[live]; ukey = ukey[live]
        if hv is not None:                       # the heavy values' pairs join the light ones'
            ukey = np.concatenate([hv[0], ukey]); ucnt = np.concatenate([hv[1], ucnt])
            us1 = np.concatenate([hv[2], us1]); us2 = np.concatenate([hv[3], us2])
        k9 = min(k, ucnt.size)
        if k9 == 0 and max_excl > 0:
            M *= 4; continue
        order = np.argpartition(-ucnt, max(0, k9 - 1))[:k9]
        order = order[np.argsort(-ucnt[order], kind='stable')]
        p10 = int(ucnt[order[-1]]) if k9 else 0
        if max_excl <= p10 or M9 >= nz:      # THE CERTIFICATE: every excluded
            break                            # b PROVABLY hosts no better pair
        M *= 4                               # widen once, re-check
    # (the loop's last fold IS the answer: the fold that was re-run here from the same buckets,
    # doubling every pair's bill, is gone -- 2026-10-17)
    return _pluck(seg, spec, order, ukey, ucnt, {x1: us1, x2: us2})


def _ht_trace(rec):
    if _HT_TRACE:
        with open(_HT_TRACE, 'a') as f9:
            f9.write(__import__('json').dumps(rec) + '\n')


def _onepass(pos, ac9, bc9, xc9, yc9, dv1, dv2, AB, W, T9, tag=None):
    """THE ONE PASS: (keys, counts, sums 1, sums 2) of every occurring pair -- or None when the estimate of the
    occurring pairs is past HT_LINE, or the tables would pass 2^HT_CBMAX slots (the caller's buckets take it)"""
    import wdb_kernels as _WK
    import time as _tm
    allr = pos is None
    pos9 = np.zeros(1, np.int64) if allr else pos
    n = int(bc9.size) if allr else int(pos.size)
    if n == 0:
        return None
    t0 = _tm.perf_counter()
    def _samp(S9):
        ds9, f19, f29, hb9 = _WK.pf_ht_sample(pos9, allr, ac9, bc9, np.int64(AB), np.int64(S9),
                                              np.int64(max(4, (2 * S9 - 1).bit_length())))   # load <= 1/2
        ds9 = int(ds9); f19 = int(f19); f29 = int(f29)
        # Chao1: the pairs seen plus the unseen ones the singletons imply (with every row sampled, the pairs seen)
        return ds9, f19, f29, int(hb9) / S9, (float(ds9) if S9 >= n else ds9 + f19 * (f19 - 1) / (2.0 * (f29 + 1)))
    # THE STAGED ESTIMATE (measured 2026-10-29 on the 563 pairs the Q30-Q32 boards traced): 2^15 rows first (0.7 ms
    # median against 6.1 for 2^17 -- the sample's cost is its gathers); an estimate already past HT_STAGE x HT_LINE
    # there is past the line at 2^17 too (not one of the 563 decisions flipped at a stage line of 2, 4 or 8)
    S = min(n, HT_SAMPLE1)
    ds, f1, f2, hsh, est = _samp(S)
    if not (S < n and est > HT_STAGE * HT_LINE and not (allr and hsh > HT_HEAVY)):
        S = min(n, HT_SAMPLE)
        ds, f1, f2, hsh, est = _samp(S)
    rec = {'tag': tag, 'n': n, 'S': S, 'ds': ds, 'f1': f1, 'f2': f2, 'est': est, 'heavy': hsh}
    # THE HEAVY LINE (measured 2026-10-29, Q32 vs Q31 boards): over every row, a big key past HT_HEAVY is the heavy
    # fold's (its dense table on contiguous rows beats the hash, 56 of 66 lost at share >= 0.99); under a filter both
    # gather rows by position and the one pass won all 90 at share >= 0.95 -- the line holds only with no filter
    if allr and hsh > HT_HEAVY:
        rec['path'] = 'heavy'
        _ht_trace(rec)
        return None
    if est > HT_LINE:
        rec['path'] = 'line'
        _ht_trace(rec)
        return None
    per = min(est, n / T9 + 1.0)                 # a thread can't see more pairs than its rows
    CB = max(12, int(np.ceil(np.log2(max(per, 1.0) * 2.0))))
    t1 = _tm.perf_counter()
    while True:
        if CB > HT_CBMAX:
            rec['path'] = 'spill'
            _ht_trace(rec)
            return None
        K = np.zeros((T9, 1 << CB), np.int64)
        V = np.zeros((T9, 1 << CB, W), np.float64)
        used = np.zeros(T9, np.int64)
        full = np.zeros(T9, np.uint8)
        _WK.pf_ht_pass(pos9, allr, ac9, bc9, xc9, yc9, dv1, dv2, np.int64(AB), np.int64(CB), K, V, used, full)
        if not full.any():
            break
        _HT_SPILL[0] += 1                         # the estimate was short: larger tables, the pass again
        CB += 2
    t2 = _tm.perf_counter()
    PB = min(8, CB - 6)
    P = 1 << PB
    ub = np.zeros(P, np.int64)
    _WK.pf_ht_bound(K, np.int64(PB), ub)
    offs = np.zeros(P + 1, np.int64)
    offs[1:] = np.cumsum(ub)
    oK = np.empty(int(offs[-1]), np.int64)
    oV = np.empty((int(offs[-1]), W), np.float64)
    nout = np.zeros(P, np.int64)
    _WK.pf_ht_merge(K, V, np.int64(CB), np.int64(PB), offs, oK, oV, nout)
    foffs = np.zeros(P + 1, np.int64)
    foffs[1:] = np.cumsum(nout)
    fK = np.empty(int(foffs[-1]), np.int64)
    fV = np.empty((int(foffs[-1]), W), np.float64)
    _WK.pf_ht_compact(oK, oV, offs, nout, foffs, fK, fV)
    t3 = _tm.perf_counter()
    ucnt = fV[:, 0].astype(np.int64)
    us1 = fV[:, 1].copy() if W > 1 else np.zeros(ucnt.size)
    us2 = fV[:, 2].copy() if W > 1 else us1
    rec.update(path='ht', CB=CB, pairs=int(fK.size), used_max=int(used.max()),
               ms_est=round((t1 - t0) * 1e3, 1), ms_pass=round((t2 - t1) * 1e3, 1), ms_merge=round((t3 - t2) * 1e3, 1))
    _ht_trace(rec)
    return fK, ucnt, us1, us2


def _pluck(seg, spec, order, ukey, ucnt, sums):
    global _HITS
    out = []
    for j in order.tolist():
        kv = int(ukey[j])
        bcode, acode = kv >> spec['abits'], kv & ((1 << spec['abits']) - 1)
        va = seg.fetch(spec['a'], acode)         # THE pluck
        vb = seg.fetch(spec['b'], bcode)
        if isinstance(va, (bytes, bytearray)):
            va = va.decode('utf-8', 'replace')
        if isinstance(vb, (bytes, bytearray)):
            vb = vb.decode('utf-8', 'replace')
        row = []
        for kind in spec['projkinds']:
            if kind[0] == 'A':
                row.append(va)
            elif kind[0] == 'B':
                row.append(vb)
            elif kind[0] == 'C':
                row.append(int(ucnt[j]))
            elif kind[0] == 'S':
                row.append(int(round(float(sums[kind[1]][j]))))
            else:
                row.append(float(sums[kind[1]][j]) / max(1, int(ucnt[j])))
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
