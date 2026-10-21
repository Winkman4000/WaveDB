"""wdb_firstk -- Q23's staircase early-exit (Jackson's design), general since 2026-10-21 (Q24-Q26 board).

SELECT <*, or columns> FROM hits [WHERE <one column's predicate>] ORDER BY staircol [, more keys]
       LIMIT k [OFFSET o]

The staircase IS the sort: when the first ORDER column is non-decreasing in file order, the k earliest
matches are the FIRST k matches in row order. The filter is a keep flag over its column's dictionary codes
(wdb_keepflag: =, <>, ranges, IN, LIKE, IS NULL and their AND / OR / NOT; the plain LIKE '%x%' keeps its
cheaper survivors road); the row data is walked window by window from the top of the file and stops at the
k-th hit -- only the blocks holding the answer's time window ever inflate.

More ORDER keys after the staircase (Q26's ORDER BY EventTime, SearchPhrase): the rows sharing the k-th hit's
staircase value form one unbroken run, so the walk reads on to the end of that run, and only those candidates
are sorted by (staircase code, key codes) -- every dictionary is in value order, so codes compare as values.
Emission decodes the projected columns for exactly k rows.
"""
import numpy as np
from sqlglot import expressions as E

import wdb_sql
import wdb_wherescan as WS
import wdb_keepflag as KF

_HITS = 0
_CHUNK = 1 << 19    # ~one zstd frame: the first pop is usually the only pop


def _staircase_ok(seg, cn):
    memo = seg.__dict__.setdefault('_fk_stair', {})
    if cn in memo:
        return memo[cn]
    c = seg.cols.get(cn)
    ok = False
    if c is not None and c.get('dt') in (2, 3) and int(c.get('V') or 0) > 1:
        rows = np.linspace(0, int(seg.N) - 1, 4096).astype(np.int64)
        ev = np.asarray(seg.codes_at(cn, rows), np.int64)
        ok = bool((np.diff(ev) >= 0).all())
    memo[cn] = ok
    return ok


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') \
            or tree.args.get('having') or tree.args.get('distinct') \
            or tree.args.get('group'):
        return None
    cm = col_map or {}
    # THE PROJECTION: every column (*) or a list of bare columns
    proj = tree.expressions
    if len(proj) == 1 and isinstance(proj[0], E.Star):
        pcols = None
    else:
        pcols = []
        for p in proj:
            inner = p.this if isinstance(p, E.Alias) else p
            if not isinstance(inner, E.Column):
                return None
            pc = cm.get(inner.name, inner.name)
            if seg.cols.get(pc) is None:
                return None
            pcols.append(pc)
    # THE ORDER: the staircase first (ascending), then any keys the dictionaries order by code
    ox = tree.args.get('order'); lx = tree.args.get('limit')
    if ox is None or lx is None or not ox.expressions:
        return None
    o = ox.expressions[0]
    if o.args.get('desc') or not isinstance(o.this, E.Column):
        return None
    ocol = cm.get(o.this.name, o.this.name)
    if seg.cols.get(ocol) is None or not _staircase_ok(seg, ocol):
        return None
    ties = []
    for o2 in ox.expressions[1:]:
        if not isinstance(o2.this, E.Column):
            return None
        tc = cm.get(o2.this.name, o2.this.name)
        c2 = seg.cols.get(tc)
        if c2 is None or c2.get('mode') not in (0, 1, 2) or c2.get('has_null') or seg._overrides(tc) is not None:
            return None                                       # NULLS LAST under DESC is not code order: declined
        ties.append((tc, bool(o2.args.get('desc'))))
    try:
        lim = int(lx.expression.this)
        off = int(tree.args['offset'].expression.this) if tree.args.get('offset') is not None else 0
    except Exception:
        return None
    if lim < 0 or off < 0 or lim + off > (1 << 20):
        return None
    # THE FILTER: one column's predicate, as a keep flag over its codes (or no WHERE at all)
    w9 = tree.args.get('where')
    spec = {}
    if w9 is not None:
        lk = WS._like(w9.this)
        if lk is not None and not lk[3] and lk[2] == 'contains':
            spec.update(col=cm.get(lk[0], lk[0]), needle=lk[1], kind='contains')   # Q23's survivors road
            if seg.cols.get(spec['col']) is None:
                return None
        else:
            fcol = KF.column_of(w9.this, cm)
            if fcol is None or seg.cols.get(fcol) is None or seg.cols[fcol].get('mode') not in (0, 1, 2):
                return None
            fl = KF.flag(seg, w9.this, fcol)                  # judged here: an unjudgeable filter declines
            if fl is None:
                return None
            spec.update(col=fcol, flag=fl)
    import wdb_policies as P
    if not P.no_deleted_rows(seg):
        return None
    spec.update(ocol=ocol, lim=lim, off=off, ties=ties, pcols=pcols, proj=proj)
    return spec


_FIRST_WINDOW = 1 << 18   # the walk's first window of rows (Q23's 10th hit sits at row 214,125)
_SCAN_SWITCH_SHIFT = 5    # the walk turns into one scan of the rest when more than N >> this rows remain to read


def _walk(seg, spec, need):
    """row positions of the first `need` matches in file order (more may come back), and the row the walk
    has read up to"""
    col = spec.get('col')
    N = int(seg.N)
    if col is None:                                           # no WHERE: the first rows are the matches
        return list(range(min(need, N))), min(need, N)
    needle = spec.get('needle'); kind = spec.get('kind', 'contains')
    c0 = seg.cols.get(col) or {}
    flag = spec.get('flag'); known = None; spent = 0.0; nd = None
    if flag is None:
        nd = needle.encode() if isinstance(needle, str) else needle
        if 'restarts' in c0 and len(nd) >= 1:
            known = np.zeros(int(c0['V']), np.int8)           # 0 undecided, 1 matches, 2 does not
        else:
            flag = spec['flag'] = np.asarray(WS._like_flags(seg, col, needle, kind))
    if flag is not None and not flag.any():
        return [], N
    hits = []
    lo = 0; W = _FIRST_WINDOW
    while lo < N and len(hits) < need:
        hi = min(N, lo + W)
        cc = np.asarray(seg.codes_at(col, np.arange(lo, hi, dtype=np.int64)), np.int64)
        if flag is None:
            # THE ORDER IS THE FILTER (Jackson: the cheap filter before the expensive read): the LIKE is decided
            # only for the codes the walk meets; once the walk's running bill would pass the whole dictionary's,
            # the whole dictionary is decided once and the walk continues on its flag
            cand = cc[known[cc] == 0]
            if cand.size:
                import wdb_strings
                u = np.unique(cand)
                sub, bill = wdb_strings.take_subset_road(seg, col, u)
                if sub and spent + bill[0] < wdb_strings.AT_COST_SHARE:
                    known[u] = np.where(wdb_strings.identify_contains_at(seg, col, u, nd), 1, 2)
                    spent += bill[0]
                else:
                    flag = spec['flag'] = np.asarray(WS._like_flags(seg, col, needle, 'contains'))
                    if not flag.any():
                        return [], N
        m9 = np.flatnonzero(flag[cc]) if flag is not None else np.flatnonzero(known[cc] == 1)
        if m9.size:
            hits.extend((lo + m9).tolist())
        lo = hi; W = _CHUNK
        if len(hits) < need and lo < N:
            # THE WALK THAT WOULD NOT STOP (2026-10-19, the LIKE board): at the hit rate seen so far, past
            # _SCAN_SWITCH_SHIFT's share of the table still to read, one scan of the rest through the column's own
            # scanner is cheaper than window after window of point reads
            rate = len(hits) / float(lo)
            left = (need - len(hits)) / rate if rate > 0 else float('inf')
            if left > (N >> _SCAN_SWITCH_SHIFT):
                if flag is None:
                    flag = spec['flag'] = np.asarray(WS._like_flags(seg, col, needle, kind))
                if flag.any():
                    rest = np.asarray(WS._scan_flag(seg, col, flag, lo, N), np.int64)
                    hits.extend(rest.tolist())
                return hits, N
    return hits, lo


def _matches_in(seg, spec, lo, hi):
    """rows in [lo, hi) the filter keeps"""
    col = spec.get('col')
    if col is None:
        return np.arange(lo, hi, dtype=np.int64)
    flag = spec.get('flag')
    if flag is None:
        flag = spec['flag'] = np.asarray(WS._like_flags(seg, col, spec['needle'], spec.get('kind', 'contains')))
    cc = np.asarray(seg.codes_at(col, np.arange(lo, hi, dtype=np.int64)), np.int64)
    return lo + np.flatnonzero(flag[cc])


def _settle_ties(seg, spec, hits, read_to, need):
    """THE RUN CERTIFICATE (Q26): the need-th hit's staircase value t is shared only by the rows of one unbroken
    run; read on to the end of that run, then order the candidates by (staircase code, the other keys' codes)"""
    N = int(seg.N); ocol = spec['ocol']
    hits = np.asarray(hits, np.int64)
    if hits.size >= need:
        r = int(hits[need - 1])
        t = int(np.asarray(seg.codes_at(ocol, np.array([r], np.int64)))[0])
        end = r + 1; W = 1 << 12
        while end < N:                                        # the run's end: the first row whose code passes t
            hi = min(N, end + W)
            oc = np.asarray(seg.codes_at(ocol, np.arange(end, hi, dtype=np.int64)), np.int64)
            j = int(np.searchsorted(oc, t, side='right'))
            end += j
            if j < oc.size:
                break
            W = min(W * 4, _CHUNK)
        cand = hits[hits < end]
        if end > read_to:                                     # the run reaches past what the walk read
            cand = np.union1d(cand, _matches_in(seg, spec, max(read_to, r + 1), end))
    else:
        cand = hits
    keys = [np.asarray(seg.codes_at(ocol, cand), np.int64)]
    for tc, desc in spec['ties']:
        kc = np.asarray(seg.codes_at(tc, cand), np.int64)
        keys.append(-kc if desc else kc)
    order = np.lexsort(tuple(reversed(keys)))                 # the first key is the primary sort
    return cand[order]


def execute(seg, spec):
    global _HITS
    k, off = spec['lim'], spec.get('off', 0)
    need = k + off
    hits, read_to = _walk(seg, spec, need)
    if spec.get('ties'):
        sel = _settle_ties(seg, spec, hits, read_to, need)
    else:
        sel = np.asarray(hits[:need], np.int64)
    sel = np.asarray(sel[off:need], np.int64)
    cols = spec.get('pcols') or list(seg.cols.keys())

    def _colvals(cn):
        c = seg.cols[cn]
        if c.get('mode') == 4:
            v4 = np.asarray(seg._seq_decode(c))[sel]
            if c.get('dt') == 3:
                return [wdb_sql._pyval(x) for x in v4.view('datetime64[%s]' % seg.unit(cn))]
            return [int(x) for x in v4]
        import wdb_pairtop
        cc = wdb_pairtop._pt_codes(seg, cn, sel)   # true point reads: enc-0
        if c.get('dt') == 0 and not (c.get('has_null') and (np.asarray(cc) == int(c['V']) - 1).any()):
            try:
                # plain ints: point dict reads -- values_at would unpack the
                # WHOLE packed dict (WatchID: 100M x 27 bits) for k values
                vv = np.asarray(seg._dict_ints_at(c, cc), np.int64)
                return [int(x) for x in vv]
            except Exception:
                pass
        try:
            vv = seg.values_at(cn, cc)       # the armory's batch fetch:
            return [wdb_sql._pyval(x) for x in vv]   # one dict walk, k values
        except Exception:
            return [wdb_sql._pyval(seg.fetch(cn, int(x))) for x in cc]
    vals = [_colvals(cn) for cn in cols]     # sequential: the work is GIL-bound
    out = [tuple(vals[ci][ri] for ci in range(len(cols)))
           for ri in range(sel.size)]
    _HITS += 1
    names = cols if spec.get('pcols') is None else [wdb_sql._alias(p) for p in spec['proj']]
    return out, names
