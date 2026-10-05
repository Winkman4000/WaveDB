"""wdb_groupexists: THE COUNT INSTEAD OF THE SEARCH (Jackson, 2026-10-04, TPC-H Q21).

    EXISTS     (SELECT * FROM t i WHERE i.k = o.k AND i.x <> o.x AND P(i))
    NOT EXISTS (SELECT * FROM t i WHERE i.k = o.k AND i.x <> o.x AND P(i))

asks, for each outer row: is there another row in my group (same k), different from me on x, that meets P?
That is a count, not a search: (rows in my group meeting P) minus (rows in my group meeting P with my own x).
EXISTS keeps the outer row when the difference is above zero, NOT EXISTS when it is zero.

Each such conjunct is judged ONCE over the outer alias's whole table, into one true/false per row, and the
conjunct is replaced in the tree by a placeholder WDB_ROWMASK(<outer k column>, '<key>'). The join engine's
fused predicate reads that per-row verdict as a slot through the alias's pointer (wdb_join.build_pred), the
mask layer reads it the same way (mask_eval). The verdicts live in a query-scoped registry (wdb_qmem):
nothing outlives the query.

Two roads:
  - THE RUNS: inner table is the outer table, same k and x columns, k non-decreasing on disk (lineitem by
    l_orderkey). Groups are contiguous runs; one parallel pass per run (wdb_kernels.pruns_others).
  - THE CENSUS: anything else of integer keys -- counts per group and per (group, x) over the inner rows
    meeting P, looked up for each outer row by bisection.
A shape outside these (nulls, overrides, strings across tables, a correlation other than one = and one <>)
leaves the node untouched: the query declines as before, never answers wrong."""
import itertools
import numpy as np
from sqlglot import expressions as E

import wdb_qmem

_MASKS = wdb_qmem.register({})          # key -> (bool mask over the outer table's rows, keep fraction)
_RUNS = wdb_qmem.register({})           # (segment, key column) -> run starts, for this query only
_SEQ = itertools.count()
MARK = 'WDB_ROWMASK'


def is_mark(node):
    return isinstance(node, E.Anonymous) and str(node.this).upper() == MARK


def lookup(node):
    """(outer column node, bool mask, keep fraction) for a WDB_ROWMASK placeholder; KeyError if it is stale."""
    col, key = node.expressions[0], node.expressions[1].this
    m, frac = _MASKS[key]
    return col, m, frac


def _flat(x):
    if isinstance(x, E.Paren): return _flat(x.this)
    if isinstance(x, E.And): return _flat(x.this) + _flat(x.expression)
    return [x]


def _and(cjs):
    out = cjs[0]
    for c in cjs[1:]:
        out = E.And(this=out, expression=c)
    return out


def _outer_tables(tree):
    frm = tree.args.get('from') or tree.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table): return None
    tabs = [frm.this] + [j.this for j in (tree.args.get('joins') or [])]
    if any(not isinstance(t, E.Table) for t in tabs): return None
    return {(t.alias or t.name): t.name for t in tabs}


def rewrite(db, tree):
    """Replace every top-level WHERE conjunct [NOT] EXISTS of the counted shape by its per-row verdict.
    Returns how many were replaced; a shape it cannot count stays in the tree untouched."""
    w = tree.args.get('where')
    if w is None or tree.find(E.Exists) is None: return 0
    outer = _outer_tables(tree)
    if outer is None: return 0
    out, done = [], 0
    for cj in _flat(w.this):
        neg = isinstance(cj, E.Not) and isinstance(cj.this, E.Exists)
        ex = cj.this if neg else cj
        rep = None
        if isinstance(ex, E.Exists):
            try:
                import os, time
                t0 = time.perf_counter()
                rep = _serve(db, ex, neg, outer)
                if os.environ.get('WDB_GE_BILL'):
                    print('GROUPEXISTS %s %s %.0f ms' % ('NOT' if neg else 'POS', 'served' if rep is not None
                          else 'declined', (time.perf_counter() - t0) * 1000), flush=True)
            except Exception:
                import os
                if os.environ.get('WDB_GE_DEBUG'):
                    import traceback; traceback.print_exc()
                rep = None
        out.append(cj if rep is None else rep)
        done += rep is not None
    if done:
        w.set('this', _and(out))
    return done


def _serve(db, ex, neg, outer):
    sub = ex.this
    if isinstance(sub, E.Subquery): sub = sub.this
    if not isinstance(sub, E.Select): return None
    for k9 in ('joins', 'group', 'having', 'limit', 'offset', 'with', 'laterals'):
        if sub.args.get(k9): return None
    frm = sub.args.get('from') or sub.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table): return None
    ww = sub.args.get('where')
    if ww is None: return None
    it, ia = frm.this.name, (frm.this.alias or frm.this.name)
    icols = set(db.cat.column_names(it))
    ocols = {a: set(db.cat.column_names(t)) for a, t in outer.items()}

    def side(c):                         # SQL scoping: the inner table answers first
        if not isinstance(c, E.Column): return None
        if c.table:
            if c.table == ia: return ('i', c.name) if c.name in icols else None
            if c.table in outer: return ('o', c.table, c.name) if c.name in ocols[c.table] else None
            return None
        if c.name in icols: return ('i', c.name)
        own = [a for a, cs in ocols.items() if c.name in cs]
        return ('o', own[0], c.name) if len(own) == 1 else None

    eqs, neqs, P = [], [], []
    for c in _flat(ww.this):
        sides = [side(x) for x in c.find_all(E.Column)]
        if not sides or any(s is None for s in sides): return None
        if all(s[0] == 'i' for s in sides):
            P.append(c); continue
        if type(c) in (E.EQ, E.NEQ) and isinstance(c.this, E.Column) and isinstance(c.expression, E.Column):
            a, b = side(c.this), side(c.expression)
            if a[0] == b[0]: return None
            inn, ou = (a, b) if a[0] == 'i' else (b, a)
            (eqs if isinstance(c, E.EQ) else neqs).append((inn[1], ou[1], ou[2]))
            continue
        return None                      # any other use of an outer column: not this shape
    if len(eqs) != 1 or len(neqs) != 1: return None
    (ik, oa, okn), (ix, oa2, oxn) = eqs[0], neqs[0]
    if oa != oa2: return None
    mask = _verdict(db, it, ik, ix, P, outer[oa], okn, oxn, neg)
    if mask is None: return None
    key = 'g%d' % next(_SEQ)
    _MASKS[key] = (mask, float(np.count_nonzero(mask)) / mask.size if mask.size else 0.0)
    return E.Anonymous(this=MARK, expressions=[
        E.Column(this=E.Identifier(this=okn, quoted=False), table=E.Identifier(this=oa, quoted=False)),
        E.Literal.string(key)])


def _ident(seg, p, codes_ok):
    """One integer-comparable identity per row: values for numbers (the shelf's decoded keys for integers),
    dictionary codes for text -- codes only when both sides read this same column (codes_ok)."""
    import wdb_join, wdb_sql
    c = seg.cols[p]
    if c.get('dt') in (0, 3):
        v = np.asarray(wdb_join._key_values(seg, p))
        return v.view(np.int64) if v.dtype.kind == 'M' else v
    if c.get('dt') == 2:
        return np.asarray(wdb_sql._col(seg, p)[0])
    if codes_ok and c.get('mode') in (0, 1, 2):
        return np.asarray(seg.codes(p))
    return None


def _verdict(db, it, ik, ix, P, ot, okn, oxn, neg):
    import wdb_join, wdb_sql, wdb_kernels
    iseg, _ = wdb_join._solo_segment(db, it)
    oseg, _ = wdb_join._solo_segment(db, ot)
    ipm, opm = db.cat.phys_map(it), db.cat.phys_map(ot)
    pik, pix, pok, pox = ipm.get(ik, ik), ipm.get(ix, ix), opm.get(okn, okn), opm.get(oxn, oxn)
    for s, p in ((iseg, pik), (iseg, pix), (oseg, pok), (oseg, pox)):
        c = s.cols.get(p)
        if c is None or c.get('has_null') or s._overrides(p) is not None: return None
    import os, time
    bill = [] if os.environ.get('WDB_GE_BILL') else None
    t0 = time.perf_counter()
    def tick(name):
        nonlocal t0
        if bill is not None:
            t1 = time.perf_counter(); bill.append('%s=%.0fms' % (name, (t1 - t0) * 1000)); t0 = t1
    pm = None
    for c in P:                          # P: the inner-only conjuncts, judged once over the inner table
        c2 = c.copy()
        for col in list(c2.find_all(E.Column)): col.set('table', None)
        m = np.asarray(wdb_sql._eval_pred(iseg, c2, lambda nm: ipm.get(nm, nm)), dtype=bool)
        pm = m if pm is None else (pm & m)
    tick('P')
    samek = iseg is oseg and pik == pok
    samex = iseg is oseg and pix == pox
    ki = _ident(iseg, pik, samek); xi = _ident(iseg, pix, samex)
    if ki is None or xi is None: return None
    tick('idents')
    if samek and samex:
        rk = (id(iseg), pik)             # the pair of EXISTS shares one group key: its runs found once
        st = _RUNS.get(rk)
        if st is None:
            st = _RUNS[rk] = wdb_kernels.run_bounds(ki)
        tick('runs')
        if st.size:                      # THE RUNS: the group is a contiguous stretch of rows
            if xi.dtype.kind == 'f':         # the kernel's long runs sort x as integers: rank floats first
                import pandas as pd
                xi = pd.factorize(xi)[0]
            out = np.empty(int(oseg.N), np.bool_)
            pu = np.ascontiguousarray(pm.view(np.uint8)) if pm is not None else np.ones(1, np.uint8)
            wdb_kernels.pruns_others(st, np.ascontiguousarray(xi), pu, pm is not None, bool(neg), out)
            tick('count')
            if bill is not None: print('GROUPEXISTS BILL ' + ' '.join(bill), flush=True)
            return out
        ko, xo = ki, xi
    else:
        ko = _ident(oseg, pok, samek); xo = _ident(oseg, pox, samex)
        if ko is None or xo is None: return None
    return _census(ki, xi, pm, ko, xo, neg)


def _census(ki, xi, pm, ko, xo, neg):
    """THE CENSUS: count per group and per (group, x) over the inner rows meeting P; each outer row's
    'others' is its group's count minus its own pair's count."""
    import pandas as pd
    if pm is not None: ki, xi = ki[pm], xi[pm]
    if ki.size == 0: return np.full(ko.shape[0], bool(neg))
    n_i = ki.size
    kf, ku = pd.factorize(np.concatenate([ki, ko]))
    xf, xu = pd.factorize(np.concatenate([xi, xo]))
    nx = len(xu)
    pair = kf.astype(np.int64) * nx + xf
    pair_i, pair_o = pair[:n_i], pair[n_i:]
    cg = np.bincount(kf[:n_i], minlength=len(ku))
    pu, pc = np.unique(pair_i, return_counts=True)
    pos = np.minimum(np.searchsorted(pu, pair_o), pu.size - 1)
    own = np.where(pu[pos] == pair_o, pc[pos], 0)
    other = cg[kf[n_i:]] - own
    return (other == 0) if neg else (other > 0)
