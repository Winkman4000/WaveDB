"""wdb_insub: [NOT] IN (an uncorrelated subquery) in a JOIN query, as a filter (c127, TPC-H Q16).

    ... WHERE ps_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%')

The inner query does not read the outer row, so:
  1. run it by itself first (wdb_subquery._run_inner; Q16: 56 suppliers, ~2 ms);
  2. turn its values into the outer column's dictionary codes (wdb_subquery._road_codes);
     NOT IN keeps the complement -- every code of the column NOT in the list -- so the filter is positive;
  3. replace the conjunct by In(column) carrying those codes (the pre-resolved code set the cascade, the fused
     predicate and the mask layer already read), with no subquery left in it.
An empty list: NOT IN keeps every row (the code set is all the column's codes); IN an empty list stands down (the
mask layer cannot take an empty code set yet -- the old road declines it, as before).
Only a plain inner query is taken (one table, a WHERE, one column; no GROUP BY / HAVING / aggregate):
the grouped shapes keep their own roads (Q18's IN ... GROUP BY ... HAVING -> the run sums).
SQL's NULL rule: NOT IN against a list holding a NULL keeps no row -- that case, an outer column with NULLs, a text
column and anything unexpected leave the conjunct untouched (the old road answers or declines).
Switch WDB_IN_SUB (default on)."""
import os
import numpy as np
from sqlglot import expressions as E

_IN_SUB = [os.environ.get('WDB_IN_SUB', '1') == '1']
_HITS = [0]


def _flat(x):
    if isinstance(x, E.Paren): return _flat(x.this)
    if isinstance(x, E.And): return _flat(x.this) + _flat(x.expression)
    return [x]


def _and(cjs):
    out = cjs[0]
    for c in cjs[1:]:
        out = E.And(this=out, expression=c)
    return out


def _outer_owner(db, tree, col):
    """(table, alias) of the OUTER FROM table that owns col (qualified, or the one table holding the name)"""
    frm = tree.args.get('from') or tree.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table): return None
    tabs = [frm.this] + [j.this for j in (tree.args.get('joins') or [])]
    if any(not isinstance(t, E.Table) for t in tabs): return None
    own = [(t.name, t.alias or t.name) for t in tabs
           if (col.table in ('', None) or col.table == (t.alias or t.name)) and col.name in db.cat.column_names(t.name)]
    return own[0] if len(own) == 1 else None


def _plain_inner(q):
    """a plain inner query: one table, an optional WHERE, one column out -- no GROUP BY, HAVING, aggregate, window,
    LIMIT, OFFSET, ORDER, join or nested subquery (those shapes have roads of their own, e.g. Q18's run sums)"""
    sub = q.this if isinstance(q, E.Subquery) else q
    if not isinstance(sub, E.Select) or len(sub.expressions) != 1: return False
    for k9 in ('group', 'having', 'limit', 'offset', 'order', 'joins', 'with', 'laterals', 'qualify'):
        if sub.args.get(k9): return False
    frm = sub.args.get('from') or sub.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table): return False
    if sub.find(E.AggFunc) is not None or sub.find(E.Window) is not None: return False
    w = sub.args.get('where')
    if w is not None and w.find(E.Select) is not None: return False
    return True


def _serve(db, tree, node, neg):
    import wdb_subquery as S, wdb_join as WJ
    if not _plain_inner(node.args['query']): return None
    own = _outer_owner(db, tree, node.this)
    if own is None: return None
    seg, _ = WJ._solo_segment(db, own[0])
    pc = db.cat.phys_map(own[0]).get(node.this.name, node.this.name)
    c = seg.cols.get(pc)
    if c is None or c.get('has_null') or c.get('dt') not in (0, 2, 3): return None
    rows = S._run_inner(db, node.args['query'])                      # step 1
    if rows and len(rows[0]) != 1: return None
    vals = [r[0] for r in rows]
    if any(v is None for v in vals):
        if neg: return None                                          # NOT IN with an inner NULL: the old road's rule
        rows = [(v,) for v in vals if v is not None]
    if rows:
        probe = E.In(this=E.Column(this=E.Identifier(this=node.this.name, quoted=False),
                                   table=E.Identifier(this=own[0], quoted=False)))
        codes = S._road_codes(db, tree, probe, rows)                 # step 2
        if codes is None: return None
        codes = np.unique(np.asarray(codes, dtype=np.int64))
    else:
        if not neg: return None                                      # IN an empty list: stands down (the mask
        codes = np.zeros(0, dtype=np.int64)                          # layer cannot take an empty code set yet);
                                                                     # NOT IN an empty list keeps every row
    if neg:
        V = int(c.get('V') or 0)
        if V <= 0: return None
        keep = np.ones(V, dtype=bool)
        keep[codes[(codes >= 0) & (codes < V)]] = False
        codes = np.flatnonzero(keep).astype(np.int64)
    out = E.In(this=node.this.copy())                                # step 3
    out.set('_codes', codes)
    return out


def rewrite(db, tree):
    """Every top-level WHERE conjunct `col [NOT] IN (uncorrelated subquery)` of a join query that can be served
    becomes In(col) with codes; returns how many."""
    if not _IN_SUB[0]: return 0
    w = tree.args.get('where')
    if w is None or not tree.args.get('joins'): return 0
    out, done = [], 0
    for cj in _flat(w.this):
        neg = isinstance(cj, E.Not) and isinstance(cj.this, E.In)
        node = cj.this if neg else cj
        rep = None
        if isinstance(node, E.In) and node.args.get('query') is not None and isinstance(node.this, E.Column):
            try:
                rep = _serve(db, tree, node, neg)
            except Exception:
                if os.environ.get('WDB_IN_SUB_DEBUG'):
                    import traceback; traceback.print_exc()
                rep = None
        out.append(rep if rep is not None else cj)
        done += rep is not None
    if done:
        w.set('this', _and(out))
        _HITS[0] += done
    return done
