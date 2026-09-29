"""wdb_firstsorted -- Q24/Q26: the staircase answers ORDER BY unprojected time.

SELECT p FROM hits WHERE p <> '' ORDER BY et [, p] LIMIT k
  where et is a STAIRCASE column (file order IS et order).

The planes hand the typed positions ascending; the first k ARE the answer
(Q24). With a tiebreak on p (Q26), et codes come free from the steps
(searchsorted, no decode) and p compares by sorted-dict CODE; a boundary
certificate widens until the k-th row's et strictly clears the candidate
window. Ten values decode at the pluck.
"""
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') \
            or tree.args.get('having') or tree.args.get('distinct') \
            or tree.args.get('group'):
        return None
    if len(tree.expressions) != 1:
        return None
    p0 = tree.expressions[0]
    inner = p0.this if isinstance(p0, E.Alias) else p0
    if not isinstance(inner, E.Column):
        return None
    cm = col_map or {}
    pcol = cm.get(inner.name, inner.name)
    w9 = tree.args.get('where')
    if w9 is None or not isinstance(w9.this, E.NEQ):
        return None
    wl, wr = w9.this.this, w9.this.expression
    if not (isinstance(wl, E.Column) and isinstance(wr, E.Literal)
            and wr.is_string and str(wr.this) == ''):
        return None
    if cm.get(wl.name, wl.name) != pcol:
        return None                              # v1: filter col is the projection
    pc = seg.cols.get(pcol)
    if pc is None or pc.get('code_enc') not in (8, 9) or pc.get('has_null'):
        return None
    ox = tree.args.get('order'); lx = tree.args.get('limit')
    if ox is None or lx is None or len(ox.expressions) not in (1, 2):
        return None
    o1 = ox.expressions[0]
    if o1.args.get('desc') or not isinstance(o1.this, E.Column):
        return None
    ocol = cm.get(o1.this.name, o1.this.name)
    steps = seg.stairs(ocol)
    if steps is None:
        return None                              # the staircase is the whole trick
    tiebreak = False
    if len(ox.expressions) == 2:
        o2 = ox.expressions[1]
        if o2.args.get('desc') or not isinstance(o2.this, E.Column) \
                or cm.get(o2.this.name, o2.this.name) != pcol:
            return None
        tiebreak = True
    try:
        lim = int(lx.expression.this)
    except Exception:
        return None
    import wdb_policies as P
    if not P.no_deleted_rows(seg):
        return None
    return {'p': pcol, 'o': ocol, 'k': lim, 'tie': tiebreak,
            'proj': tree.expressions}


def _planes_head(seg, p, K):
    """the first >= K present rows and their codes, and how many the column holds: from the head
    chunks only when the dress allows it (tag 8: Segment.e8_head), else the full planes"""
    import os
    h = seg.e8_head(p, K) if (hasattr(seg, 'e8_head') and os.environ.get('WDB_E8_HEAD', '1') != '0') else None
    if h is not None:
        return h
    pl = seg.e8_planes(p)
    pos = np.asarray(pl[0], dtype=np.int64)
    return pos, np.asarray(pl[1], dtype=np.int64), int(pos.size)


def execute(seg, spec):
    global _HITS
    if not spec['tie']:
        pos, lits, n_all = _planes_head(seg, spec['p'], spec['k'])
        k = min(spec['k'], pos.size)
        pick = lits[:k]                          # file order IS et order (Q24)
    else:
        steps = np.asarray(seg.stairs(spec['o']), dtype=np.int64)
        K = max(64, 4 * spec['k'])
        while True:
            pos, lits, n_all = _planes_head(seg, spec['p'], K)   # THE HEAD: only the chunks the window needs
            k = min(spec['k'], pos.size)
            K9 = min(K, pos.size)
            cand_pos = pos[:K9]
            etc = np.searchsorted(steps, cand_pos, side='right')   # et code, no decode
            order = np.lexsort((lits[:K9], etc))                   # (et, p) by CODE
            pick = lits[:K9][order[:k]]
            # THE CERTIFICATE: the window is complete when the last candidate's
            # et strictly clears the k-th chosen row's et (or the file is spent)
            if K9 >= n_all or (k and int(etc[K9 - 1]) > int(etc[order[k - 1]])):
                break
            K *= 4
    # THE pluck: k values in one batch -- values_at pops the touched dictionary chunks in parallel
    # (ten point fetches were ~20 serial reads + inflates, ~50 ms of the cold run)
    out = [(v9.decode('utf-8', 'replace') if isinstance(v9, (bytes, bytearray)) else v9,)
           for v9 in seg.values_at(spec['p'], np.asarray(pick, dtype=np.int64))]
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
