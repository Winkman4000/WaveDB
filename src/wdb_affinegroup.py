"""wdb_affinegroup -- Q35's identity (grouping by x wearing four hats).

SELECT f1(col), f2(col), ..., COUNT(*) c FROM hits
GROUP BY f1(col), f2(col), ... ORDER BY c DESC LIMIT n
  where every f is affine: col itself or col +/- literal.

All the group keys are determined by col, so the group IS col: one
bincount over codes, top-n by count, decode exactly n values at the
pluck, re-derive the affine faces with n arithmetic ops. The composite
factorize dies; no row is walked twice.
"""
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0


def _affine(node):
    """col / col+lit / col-lit / lit+col -> (name, shift) or None."""
    if isinstance(node, E.Column):
        return (node.name, 0)
    if isinstance(node, E.Add):
        a, b = node.this, node.expression
        if isinstance(a, E.Column) and isinstance(b, E.Literal) and not b.is_string:
            return (a.name, int(str(b.this)))
        if isinstance(b, E.Column) and isinstance(a, E.Literal) and not a.is_string:
            return (b.name, int(str(a.this)))
    if isinstance(node, E.Sub):
        a, b = node.this, node.expression
        if isinstance(a, E.Column) and isinstance(b, E.Literal) and not b.is_string:
            return (a.name, -int(str(b.this)))
    return None


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') or tree.args.get('where') \
            or tree.args.get('having') or tree.args.get('distinct'):
        return None
    g = tree.args.get('group')
    if g is None or not g.expressions:
        return None
    cm = col_map or {}
    col = None
    ncols = 0
    for ge in g.expressions:
        if isinstance(ge, E.Literal):
            if not ge.is_string and str(ge.this).lstrip('-').isdigit():
                # DuckDB law: a bare integer in GROUP BY is POSITIONAL --
                # resolve it to the select item before judging it
                idx = int(str(ge.this)) - 1
                if idx < 0 or idx >= len(tree.expressions):
                    return None
                item = tree.expressions[idx]
                ge = item.this if isinstance(item, E.Alias) else item
                if isinstance(ge, E.Literal):
                    continue                     # a true constant: no-op
            else:
                continue                         # string/float literal: no-op
        af = _affine(ge)
        if af is None:
            return None
        cn = cm.get(af[0], af[0])
        ncols += 1
        if af[1] != 0 and (seg.cols.get(cm.get(af[0], af[0])) or {}).get('dt') != 0:
            return None                      # arithmetic faces need ints
        if col is None:
            col = cn
        elif cn != col:
            return None                          # one column, many faces
    if ncols == 0:
        return None                              # all-constant group: not ours
    c = seg.cols.get(col)
    if c is None or c.get('has_null'):
        return None
    proj = []                                    # ('F', shift) | ('C',)
    calias = None
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Literal):
            v9 = str(inner.this)
            proj.append(('L', v9 if inner.is_string else
                         (float(v9) if '.' in v9 else int(v9))))
            continue                             # the party hat, glued back on
        af = _affine(inner)
        if af is not None and af[1] != 0 \
                and (seg.cols.get(cm.get(af[0], af[0])) or {}).get('dt') != 0:
            return None                      # arithmetic faces need ints
        if af is not None:
            if cm.get(af[0], af[0]) != col:
                return None
            proj.append(('F', af[1])); continue
        ak = wdb_sql._agg_kind(inner)
        if ak is None or ak[0] != 'COUNT_STAR':
            return None
        proj.append(('C',))
        if isinstance(p, E.Alias):
            calias = p.alias
    if not any(k[0] == 'C' for k in proj):
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
    elif not isinstance(io, E.Count) or isinstance(io.this, E.Distinct):
        return None
    try:
        lim = int(lx.expression.this)
    except Exception:
        return None
    return {'col': col, 'lim': lim, 'projkinds': proj, 'proj': tree.expressions}


def execute(seg, spec):
    global _HITS
    col, k = spec['col'], spec['lim']
    V = int(seg.cols[col]['V'])
    memo = seg.__dict__.setdefault('_censusmemo', {})   # V-sized, planes-memo law
    cnt = memo.get(col)
    if cnt is None:
        cnt = np.bincount(np.asarray(seg._raw_codes(col)), minlength=V)
        memo[col] = cnt
    k9 = min(k, V - 1)
    order = np.argpartition(-cnt, k9)[:k]
    order = order[np.argsort(-cnt[order], kind='stable')]
    out = []
    for code in order.tolist():
        v9 = seg.fetch(col, int(code))           # THE pluck: n values only
        if isinstance(v9, (bytes, bytearray)):
            v9 = v9.decode('utf-8', 'replace')
        row = []
        for kind in spec['projkinds']:
            if kind[0] == 'L':
                row.append(kind[1])
            elif kind[0] == 'F':
                row.append(v9 if kind[1] == 0 else int(v9) + kind[1])
            else:
                row.append(int(cnt[code]))
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
