"""wdb_septop -- Q14's engine-first hunt (Jackson's design).

SELECT se, sp, COUNT(*) WHERE sp <> '' GROUP BY se, sp ORDER BY c DESC LIMIT k

The filter IS the plane; codes differentiate without decoding. A pair's
count <= its phrase's total, so top pairs live among top phrases: take the
top-M phrases by spc, read the ENGINE (low V) at only their instance rows,
count (engine, phrase) exactly inside the subset, and accept when the k-th
count >= the M-th phrase total (no outside pair can beat it). Decode only
the k winners.
"""
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
    if not (isinstance(n9, E.NEQ) and isinstance(n9.this, E.Column)
            and isinstance(n9.expression, E.Literal)
            and str(n9.expression.this) == ''):
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
    sp = fcol
    se = [c for c in gcols if c != sp][0]
    cs = seg.cols.get(se)
    cp = seg.cols.get(sp)
    if cs is None or cp is None:
        return None
    if cp.get('code_enc') != 8:
        return None                              # the filter must BE the plane
    if int(cs.get('V') or 1 << 30) > 4096:
        return None                              # engine-first needs low V
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
    try:
        lim = int(lx.expression.this)
    except Exception:
        return None
    return {'se': se, 'sp': sp, 'lim': lim, 'proj': tree.expressions}


def execute(seg, spec):
    global _HITS
    se, sp, k = spec['se'], spec['sp'], spec['lim']
    pl = seg.e8_planes(sp)
    pos8 = np.asarray(pl[0], np.int64)
    lits8 = np.asarray(pl[1], np.int64)
    Vp = int(seg.cols[sp]['V'])
    spc = np.bincount(lits8, minlength=Vp)
    # Jackson's descent, window-batched: hunt the biggest phrases first in
    # ONE batch per window; stop when the k-th pair beats the best phrase
    # still outside. kth lands high fast, so the first window usually ends it.
    W = max(16, 2 * k)
    while True:
        W = min(W, Vp)
        if W < Vp:
            part = np.argpartition(-spc, W)[:W + 1]     # top-W+1, unordered
            part = part[np.argsort(-spc[part], kind='stable')]
            topp = part[:W]
            outside = int(spc[part[W]])      # the best phrase left outside
        else:
            topp = np.arange(Vp)
            outside = 0
        sel = np.flatnonzero(np.isin(lits8, topp))
        rows9 = pos8[sel]
        pc9 = lits8[sel]
        ec9 = np.asarray(seg.codes_at(se, rows9), np.int64)
        key = (ec9 << 23) | pc9
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
        ev = seg.fetch(se, kk >> 23)
        sv = seg.fetch(sp, kk & ((1 << 23) - 1))
        if isinstance(ev, (bytes, bytearray)):
            ev = ev.decode('utf-8', 'replace')
        if isinstance(sv, (bytes, bytearray)):
            sv = sv.decode('utf-8', 'replace')
        row = []
        for p in proj:
            inner = p.this if isinstance(p, E.Alias) else p
            if isinstance(inner, E.Column):
                cn = inner.name
                row.append(sv if cn == spec['sp'] or cn.lower() == spec['sp'].lower() else ev)
            else:
                row.append(c9)
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in proj]
