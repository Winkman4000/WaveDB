"""THE SAMPLE LANE: a LIMIT with no ORDER BY asks for a sample, not a
ranking -- so serve a fresh lawful sample each run. Random group keys
are drawn from the big key's plist; each candidate's COMPLETE row slice
makes every returned count exact by construction. Jackson's ruling:
different runs, different ten -- the point of such a query is sampling,
and honest variance comes with the territory."""
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
    # the sampled key needs a plist; the partner needs point reads
    bcol = None
    for c in gcols:
        cc = seg.cols.get(c, {})
        if cc.get('code_enc') in (8, 12) and int(cc.get('V', 0)) >= 1000:
            bcol = c
    if bcol is None:
        return None
    acol = [c for c in gcols if c != bcol][0]
    if seg.cols.get(acol, {}).get('code_enc') not in (0, 2, 3, 10, 12, 19):   # 19: point reads by block
        return None
    return {'b': bcol, 'a': acol, 'k': k, 'kinds': kinds,
            'proj': proj}


def execute(seg, spec):
    global _HITS
    import wdb_funnel as _F
    offsB, plB = _F._plist(seg, spec['b'])
    offsB = np.asarray(offsB, dtype=np.int64)
    cnt_b = np.diff(offsB)
    V = cnt_b.size
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
        n9 = int(cnt_b[c])
        if n9 == 0 or n9 > cap9:
            continue
        rows9 = np.asarray(plB[offsB[c]:offsB[c + 1]]).astype(np.int64)
        ac9 = np.asarray(seg.codes_at(spec['a'], rows9)).astype(np.int64)
        u9, n9c = np.unique(ac9, return_counts=True)
        for aa, nn in zip(u9.tolist(), n9c.tolist()):
            if len(pairs) >= k:
                break
            pairs[(c, aa)] = int(nn)
    out = []
    for (bc, ac), nn in list(pairs.items())[:k]:
        vb = seg.fetch(spec['b'], bc)
        va = seg.fetch(spec['a'], ac)
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
