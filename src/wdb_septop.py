"""wdb_septop -- the engine-first hunt (Jackson's design): top-k pairs (low-V key, a dictionary key filtered by `<>`).

SELECT kcol, tcol, COUNT(*) WHERE tcol <> lit GROUP BY kcol, tcol ORDER BY c DESC LIMIT k

The kept rows are the planes: tcol is stored sparse and lit is its stored
default (any default, any literal of tcol's kind), so the filter IS the plane --
wdb_wherescan.neq_rows hands them, no scan. Any other literal is another read's.
Codes differentiate without decoding. A pair's
count <= its phrase's total, so top pairs live among top phrases: take the
top-M phrases by tcnt, read the ENGINE (low V) at only their instance rows,
count (engine, phrase) exactly inside the subset, and accept when the k-th
count >= the M-th phrase total (no outside pair can beat it). Decode only
the k winners.

Lore: built for ClickBench Q14 (`<> ''` on a sparse column whose default is '').
"""
# lore: 2026-11-07 widened from `<> ''` on a default-'' sparse column only, to any literal that is the stored
# lore: default of tcol (stored sparse)
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') \
            or tree.args.get('having') or tree.args.get('distinct'):
        return None
    w9 = tree.args.get('where')
    if w9 is None:
        return None
    n9 = w9.this
    import wdb_wherescan as _WS
    if not (isinstance(n9, E.NEQ) and isinstance(n9.this, E.Column)
            and _WS._litval(n9.expression) is not None):
        return None
    cm = col_map or {}
    fcol = cm.get(n9.this.name, n9.this.name)
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 2:
        return None
    gcols = []
    for ge in g.expressions:
        if not isinstance(ge, E.Column):
            return None
        gcols.append(cm.get(ge.name, ge.name))
    if fcol not in gcols:
        return None
    tcol = fcol
    kcol = [c for c in gcols if c != tcol][0]
    cs = seg.cols.get(kcol)
    cp = seg.cols.get(tcol)
    if cs is None or cp is None:
        return None
    if cp.get('mode') not in (0, 1, 2) or cp.get('dt') not in (0, 1) \
            or (cp.get('dt') == 1) != bool(getattr(n9.expression, 'is_string', False)):
        return None                              # neq_rows answers a dictionary column against a literal of its kind
    import wdb_policies as _P9
    if cp.get('code_enc') not in (8, 9) or not _P9.sparse_default_is(seg, tcol, _WS._litval(n9.expression)):
        return None                              # the kept rows must BE the planes: tcol stored sparse, lit its default
    # lore: 2026-11-07 board on hits (100M): with a full pass over the codes for any other literal, septop took the
    # lore: filtered pair count-tops from gridwalk at 15-45x their time (0.06 -> 2.9 s, DuckDB 0.14); the planes
    # lore: forms stayed at DuckDB's 0.02-0.1 s
    if int(cs.get('V') or 1 << 30) > 4096:
        return None                              # engine-first needs low V
    if int(cp.get('V') or 1 << 30) >= (1 << 50):
        return None                              # the packed key holds the engine code above the phrase code
    aggs = []
    calias = None
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            cn = cm.get(inner.name, inner.name)
            if cn not in gcols:
                return None
            aggs.append(('K', cn)); continue
        ak = wdb_sql._agg_kind(inner)
        if ak is None or ak[0] != 'COUNT_STAR':
            return None
        aggs.append(('C',))
        if isinstance(p, E.Alias):
            calias = p.alias
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
    elif not isinstance(io, E.Count):
        return None
    if tree.args.get('offset') is not None or lx.args.get('offset') is not None:
        return None
    try:
        lim = int(lx.expression.this)
    except Exception:
        return None
    return {'kcol': kcol, 'tcol': tcol, 'lim': lim, 'proj': tree.expressions, 'lit': n9.expression}


def execute(seg, spec):
    global _HITS
    kcol, tcol, k = spec['kcol'], spec['tcol'], spec['lim']
    import wdb_wherescan as _WS
    r9 = _WS.neq_rows(seg, tcol, spec['lit'])
    if r9 is None:
        return None                              # not a code filter this column can answer: the next read
    pos8, lits8 = r9
    Vp = int(seg.cols[tcol]['V'])
    sh = max(1, (Vp - 1).bit_length())           # the phrase code's field in the packed key
    tcnt = np.bincount(lits8, minlength=Vp)
    # Jackson's descent, window-batched: hunt the biggest phrases first in
    # ONE batch per window; stop when the k-th pair beats the best phrase
    # still outside. kth lands high fast, so the first window usually ends it.
    W = max(16, 2 * k)
    while True:
        W = min(W, Vp)
        if W < Vp:
            part = np.argpartition(-tcnt, W)[:W + 1]     # top-W+1, unordered
            part = part[np.argsort(-tcnt[part], kind='stable')]
            topp = part[:W]
            outside = int(tcnt[part[W]])      # the best phrase left outside
        else:
            topp = np.arange(Vp)
            outside = 0
        sel = np.flatnonzero(np.isin(lits8, topp))
        rows9 = pos8[sel]
        pc9 = lits8[sel]
        ec9 = np.asarray(seg.codes_at(kcol, rows9), np.int64)
        key = (ec9 << sh) | pc9
        key.sort(kind='stable')
        brk = np.empty(key.size, bool)
        if key.size:
            brk[0] = True
            np.not_equal(key[1:], key[:-1], out=brk[1:])
        st = np.flatnonzero(brk)
        cnt = np.diff(np.append(st, key.size))
        order = np.argsort(-cnt, kind='stable')[:k]
        kth = int(cnt[order[-1]]) if order.size >= k else 0
        if kth >= outside or W >= Vp:
            break                            # nothing outside can board
        W *= 4
    out = []
    proj = spec['proj']
    for gi in order.tolist():
        kk = int(key[st[gi]]); c9 = int(cnt[gi])
        ev = seg.fetch(kcol, kk >> sh)
        sv = seg.fetch(tcol, kk & ((1 << sh) - 1))
        if isinstance(ev, (bytes, bytearray)):
            ev = ev.decode('utf-8', 'replace')
        if isinstance(sv, (bytes, bytearray)):
            sv = sv.decode('utf-8', 'replace')
        row = []
        for p in proj:
            inner = p.this if isinstance(p, E.Alias) else p
            if isinstance(inner, E.Column):
                cn = inner.name
                row.append(sv if cn == spec['tcol'] or cn.lower() == spec['tcol'].lower() else ev)
            else:
                row.append(c9)
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in proj]
