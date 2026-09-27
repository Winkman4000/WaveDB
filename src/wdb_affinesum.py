"""wdb_affinesum -- Q29's algebra (the ninety-sums comedy).

SELECT SUM(col + k0), SUM(col + k1), ... FROM hits   (no WHERE, no GROUP)

SUM(col + k) = S + k*N: every projection is an affine fact of two numbers.
S comes from the census (V-sized dot product, gbc-shelf material), N from
the segment. Nine billion additions become ninety multiply-adds; no row
is ever walked, nothing decodes beyond a V-sized dictionary.
"""
import numpy as np
from sqlglot import expressions as E

import wdb_sql

_HITS = 0


def _affine(node):
    """SUM(col) / SUM(col + lit) / SUM(lit + col) -> (colname, shift) or None."""
    if not isinstance(node, E.Sum):
        return None
    t = node.this
    if isinstance(t, E.Column):
        return (t.name, 0)
    if isinstance(t, E.Add):
        a, b = t.this, t.expression
        if isinstance(a, E.Column) and isinstance(b, E.Literal) and not b.is_string:
            return (a.name, int(str(b.this)))
        if isinstance(b, E.Column) and isinstance(a, E.Literal) and not a.is_string:
            return (b.name, int(str(a.this)))
    return None


def detect(seg, tree, col_map):
    if tree.args.get('joins') or tree.args.get('with') or tree.args.get('where') \
            or tree.args.get('group') or tree.args.get('having') \
            or tree.args.get('order') or tree.args.get('distinct'):
        return None
    cm = col_map or {}
    shifts = []
    col = None
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        af = _affine(inner)
        if af is None:
            return None
        cn = cm.get(af[0], af[0])
        if col is None:
            col = cn
        elif cn != col:
            return None                          # one column, many shifts
        shifts.append(af[1])
    if col is None or len(shifts) < 2:
        return None
    c = seg.cols.get(col)
    if c is None or c.get('dt') != 0 or c.get('has_null') \
            or int(c.get('V') or 1 << 30) > 1 << 20:
        return None
    return {'col': col, 'shifts': shifts, 'proj': tree.expressions}


def execute(seg, spec):
    global _HITS
    col = spec['col']
    V = int(seg.cols[col]['V'])
    memo = seg.__dict__.setdefault('_censusmemo', {})   # V-sized, planes-memo law
    census = memo.get(col)
    if census is None:
        # THE LOAD'S CENSUS FIRST (2026-09-27): the rows per code were counted at load time
        # (stats.npz '<col>.vcnt', the census of the load, 2026-09-24). Q29 measured: the full
        # decode (53 ms) + bincount (166 ms) re-counted them on every run; the load's copy
        # reads in ~1 ms and equals the bincount entry for entry (checked on the kit).
        try:
            import wdb_blockstats
            vc = wdb_blockstats.vcnt_from_load(seg, col)
            if vc is not None and vc.size == V and int(vc.sum()) == int(seg.N):
                census = np.asarray(vc, np.int64)
        except Exception:
            census = None
    if census is None:
        import wdb_gbshelf
        sh = wdb_gbshelf.open_shelf(seg, col)            # ride it if it exists;
        if sh is not None:                               # never birth here -- the
            try:                                         # u16 tail law refuses wide
                census = np.asarray(wdb_gbshelf.bulk(sh), np.int64)   # columns and a
            except Exception:                            # refused birth re-paid the
                census = None                            # bincount every single run
    if census is None or census.size < V or int(census.sum()) != int(seg.N):
        census = np.bincount(np.asarray(seg._raw_codes(col)), minlength=V)
    memo[col] = census
    dv = memo.get((col, 'dv'))                           # the dictionary's values, once per segment: the
    if dv is None or dv.size != V:                       # per-value fetch fallback was 2,159 calls every run
        try:
            dv = np.asarray(seg._dict_ints_at(seg.cols[col],
                                              np.arange(V, dtype=np.int64)),
                            np.float64)
        except Exception:
            dv = None
        if dv is None and int(seg.cols[col].get('mode', -1)) == 0:
            # a mode-0 integer dictionary holds its values as decimal text (the integer spine is
            # mode 2's): read the whole small dictionary once instead of V single fetches
            try:
                vals = seg.dict_vals(col)
                if len(vals) == V:
                    dv = np.asarray([float(int(x)) for x in vals], np.float64)
            except Exception:
                dv = None
        if dv is None:
            dv = np.asarray([float(seg.fetch(col, v9)) for v9 in range(V)],
                            np.float64)
        memo[(col, 'dv')] = dv
    S = float(census @ dv)
    N = int(census.sum())
    out = tuple(int(round(S + k * N)) for k in spec['shifts'])
    _HITS += 1
    return [out], [wdb_sql._alias(p) for p in spec['proj']]
