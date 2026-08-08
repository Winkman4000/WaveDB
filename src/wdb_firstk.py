"""wdb_firstk -- Q23's staircase early-exit (Jackson's design).

SELECT * FROM hits WHERE strcol LIKE '%needle%' ORDER BY staircol LIMIT k

The staircase IS the sort: when the ORDER column is non-decreasing in file
order, the k earliest matches are the FIRST k matches in row order. The
dict is the haystack (code flags, memoized); the row data is walked frame
by frame from the top of the file and stops at the k-th hit -- only the
blocks holding the answer's time window ever inflate. Emission decodes
every column for exactly k rows.
"""
import numpy as np
from sqlglot import expressions as E

import wdb_sql
import wdb_wherescan as WS

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
    if len(tree.expressions) != 1 or not isinstance(tree.expressions[0], E.Star):
        return None
    w9 = tree.args.get('where')
    if w9 is None:
        return None
    lk = WS._like(w9.this)
    if lk is None or lk[3]:
        return None                              # negated likes stay with wherescan
    cm = col_map or {}
    col, needle, kind = cm.get(lk[0], lk[0]), lk[1], lk[2]
    if kind != 'contains':
        return None
    ox = tree.args.get('order'); lx = tree.args.get('limit')
    if ox is None or lx is None or len(ox.expressions) != 1:
        return None
    o = ox.expressions[0]
    if o.args.get('desc') or not isinstance(o.this, E.Column):
        return None
    ocol = cm.get(o.this.name, o.this.name)
    off = 0
    if tree.args.get('offset') is not None:
        return None
    try:
        lim = int(lx.expression.this)
    except Exception:
        return None
    if seg.cols.get(col) is None or seg.cols.get(ocol) is None:
        return None
    if not _staircase_ok(seg, ocol):
        return None
    return {'col': col, 'needle': needle, 'ocol': ocol, 'lim': lim}


def execute(seg, spec):
    global _HITS
    col, needle, k = spec['col'], spec['needle'], spec['lim']
    N = int(seg.N)
    flag = np.asarray(WS._like_flags(seg, col, needle, 'contains'))
    if not flag.any():
        _HITS += 1
        return [], [c for c in seg.cols.keys()]
    hits = []
    lo = 0
    while lo < N and len(hits) < k:
        hi = min(N, lo + _CHUNK)
        cc = np.asarray(seg.codes_at(col, np.arange(lo, hi, dtype=np.int64)),
                        np.int64)
        m9 = np.flatnonzero(flag[cc])
        if m9.size:
            hits.extend((lo + m9).tolist())
        lo = hi
    sel = np.asarray(hits[:k], np.int64)
    cols = list(seg.cols.keys())
    def _colvals(cn):
        c = seg.cols[cn]
        if c.get('mode') == 4:
            return [int(x) for x in np.asarray(seg._seq_decode(c))[sel]]
        import wdb_pairtop
        cc = wdb_pairtop._pt_codes(seg, cn, sel)   # true point reads: enc-0
        if c.get('dt') == 0:                       # rides the MSB extractor,
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
    vals = [_colvals(cn) for cn in cols]     # sequential: the work is GIL-
    out = [tuple(vals[ci][ri] for ci in range(len(cols)))  # bound; a pool
           for ri in range(sel.size)]        # only buys a lock storm
    _HITS += 1
    return out, cols
