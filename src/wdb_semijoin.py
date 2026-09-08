"""THE SEMI-JOIN FIXPOINT (the JOB shape): a multi-way equi-join projecting ONLY
MIN/MAX. Under MIN/MAX row multiplication is irrelevant -- a table's
contribution is the extreme over its rows that PARTICIPATE in the join.
Participation is a fixpoint: local filters seed each table's keep; every
equality edge prunes both sides to the other's surviving keys (dense-id
lookup tables, O(N)); iterate to stability; MIN/MAX per column over its
table's survivors (strings by the extreme present dictionary code)."""
import numpy as np
import sqlglot
from sqlglot import exp as E


class _Decline(Exception):
    pass


def shape_ok(tree):
    """MIN/MAX-only projections, comma/INNER joins on plain columns, no group/order/limit/subquery/window."""
    if not isinstance(tree, E.Select): return False
    if tree.args.get('group') is not None or tree.args.get('having') is not None: return False
    if tree.args.get('order') is not None or tree.args.get('limit') is not None: return False
    if tree.find(E.Window) is not None or tree.find(E.Subquery) is not None: return False
    if not tree.expressions: return False
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        if not isinstance(nd, (E.Min, E.Max)) or not isinstance(nd.this, E.Column): return False
    joins = tree.args.get('joins') or []
    for jn in joins:
        if (jn.args.get('side') or '') or (jn.args.get('kind') or '').upper() not in ('', 'INNER', 'CROSS'): return False
        if not isinstance(jn.this, E.Table): return False
    frm = tree.args.get('from') or tree.args.get('from_')
    return frm is not None and isinstance(frm.this, E.Table)


def _conjuncts(node):
    if node is None: return []
    if isinstance(node, E.Paren): return _conjuncts(node.this)
    if isinstance(node, E.And): return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def execute(db, tree):
    import wdb_sql, os, time
    _bill = [] if os.environ.get('WDB_SEMI_BILL') else None
    _tk = time.perf_counter; _t0 = _tk()
    from wdb_join import _solo_segment, _FastUnsupported, _bulk_keyvals
    frm = tree.args.get('from') or tree.args.get('from_')
    tabs = [frm.this] + [jn.this for jn in (tree.args.get('joins') or [])]
    alias2t = {}
    for t in tabs:
        alias2t[t.alias or t.name] = t.name
    conds = []
    for c in _conjuncts(tree.args.get('where').this if tree.args.get('where') is not None else None):
        conds.append(c)
    for jn in (tree.args.get('joins') or []):
        if jn.args.get('on') is not None: conds.extend(_conjuncts(jn.args['on']))
    # column ownership
    cols_of = {a: set(db.cat.column_names(t)) for a, t in alias2t.items()}
    def owner(col):
        if col.table: return col.table
        cands = [a for a, cs in cols_of.items() if col.name in cs]
        if len(cands) != 1: raise _Decline('ambiguous column %s' % col.name)
        return cands[0]
    edges, local = [], {a: [] for a in alias2t}
    for c in conds:
        if isinstance(c, E.EQ) and isinstance(c.this, E.Column) and isinstance(c.expression, E.Column):
            a, b = owner(c.this), owner(c.expression)
            if a != b:
                edges.append((a, c.this.name, b, c.expression.name)); continue
        owners = {owner(x) for x in c.find_all(E.Column)}
        if len(owners) != 1: raise _Decline('multi-table non-equality conjunct: %s' % c.sql()[:50])
        local[owners.pop()].append(c)
    # segments, local keeps
    segs, pms, keeps = {}, {}, {}
    for a, t in alias2t.items():
        try:
            seg, _ = _solo_segment(db, t)
        except _FastUnsupported:
            raise _Decline('multi-segment table %s' % t)
        segs[a] = seg; pms[a] = db.cat.phys_map(t)
        m = None
        for c in local[a]:
            c2 = c.copy()
            for col in c2.find_all(E.Column):
                col.set('table', None)
            mm = np.asarray(wdb_sql._eval_pred(seg, c2, lambda nm, pm=pms[a]: pm.get(nm, nm)), dtype=bool)
            m = mm if m is None else (m & mm)
        keeps[a] = m if m is not None else np.ones(int(seg.N), bool)
        if _bill is not None: _bill.append(('local %s(%d) keep=%d' % (a, int(seg.N), int(keeps[a].sum())), _tk() - _t0)); _t0 = _tk()
    # key columns as int64 arrays (NULL -> -1)
    keycache = {}
    def keys(a, col):
        k = (a, col)
        if k in keycache: return keycache[k]
        seg = segs[a]; pc = pms[a].get(col, col); cd = seg.cols.get(pc)
        if cd is None: raise _Decline('no such column %s.%s' % (a, col))
        if cd.get('dt') != 0: raise _Decline('non-integer join key %s.%s' % (a, col))
        raw = wdb_sql.raw_dict_col(seg, pc, want_codes=False)
        codes = np.asarray(seg.codes(pc))
        if raw is not None:
            vals = np.asarray(raw[0]).astype(np.int64)
            if cd.get('has_null'):
                vals = np.append(vals, -1)
            out = vals[codes]
        else:
            arr, nm = wdb_sql._col(seg, pc)
            out = np.asarray(arr).astype(np.int64)
            if nm is not None: out = np.where(nm, -1, out)
        keycache[k] = out
        return out
    # edge merge: several equalities between the same pair -> composite keys
    pair = {}
    for a, ca, b, cb in edges:
        key = (a, b) if a < b else (b, a)
        pair.setdefault(key, []).append((ca, cb) if a < b else (cb, ca))
    def combined(a, cols_a, b, cols_b):
        ka = [keys(a, c) for c in cols_a]; kb = [keys(b, c) for c in cols_b]
        if len(ka) == 1: return ka[0], kb[0]
        # composite: pack (values are ids < 2^31 in IMDB)
        xa = np.zeros_like(ka[0]); xb = np.zeros_like(kb[0])
        for va, vb in zip(ka, kb):
            xa = xa * (1 << 31) + np.where(va < 0, 0, va); xb = xb * (1 << 31) + np.where(vb < 0, 0, vb)
        xa = np.where(np.any(np.stack([v < 0 for v in ka]), axis=0), -1, xa)
        xb = np.where(np.any(np.stack([v < 0 for v in kb]), axis=0), -1, xb)
        return xa, xb
    # fixpoint
    def prune(src_keys, src_keep, dst_keys, dst_keep):
        sk = src_keys[src_keep]
        sk = sk[sk >= 0]
        if sk.size == 0:
            return np.zeros_like(dst_keep)
        mx = int(max(sk.max(), dst_keys.max())) if dst_keys.size else int(sk.max())
        idx = np.flatnonzero(dst_keep)
        if idx.size == 0: return dst_keep
        if mx < 200_000_000:
            lut = np.zeros(mx + 2, bool); lut[sk] = True
            if idx.size * 8 < dst_keys.size:
                # SURVIVORS ONLY: once the keep is small, gather only the kept rows' keys
                dk = dst_keys[idx]
                hit_s = lut[np.where(dk < 0, mx + 1, dk)]
                out = np.zeros_like(dst_keep); out[idx[hit_s]] = True
                return out
            hit = lut[np.where(dst_keys < 0, mx + 1, dst_keys)]
        else:
            hit = np.isin(dst_keys, np.unique(sk))
        return dst_keep & hit
    for _round in range(12):
        changed = False
        for (a, b), pairs in pair.items():
            xa, xb = combined(a, [p[0] for p in pairs], b, [p[1] for p in pairs])
            na = prune(xb, keeps[b], xa, keeps[a])
            if na.sum() != keeps[a].sum(): keeps[a] = na; changed = True
            nb = prune(xa, keeps[a], xb, keeps[b])
            if nb.sum() != keeps[b].sum(): keeps[b] = nb; changed = True
            if _bill is not None: _bill.append(('r%d %s-%s keep %d/%d' % (_round, a, b, int(keeps[a].sum()), int(keeps[b].sum())), _tk() - _t0)); _t0 = _tk()
        if not changed: break
    # any table empty -> every MIN is NULL (a scalar over an empty join)
    empty = any(int(k.sum()) == 0 for k in keeps.values())
    out = []
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        a = owner(nd.this); seg = segs[a]; pc = pms[a].get(nd.this.name, nd.this.name)
        if empty:
            out.append(None); continue
        rows = np.flatnonzero(keeps[a])
        cd = seg.cols.get(pc, {})
        codes = np.asarray(seg.codes(pc))[rows] if cd.get('mode') in (0, 1, 2) else None
        if codes is not None:
            if cd.get('has_null'):
                codes = codes[codes != int(cd['V']) - 1]
            if codes.size == 0:
                out.append(None); continue
            k = int(codes.min()) if isinstance(nd, E.Min) else int(codes.max())     # sorted dictionary
            v = seg._typed_dict(pc)[k]
            out.append(wdb_sql._pyval(v))
        else:
            vals = list(seg.values_at_rows(pc, rows))
            vals = [v for v in vals if v is not None]
            out.append(wdb_sql._pyval(min(vals) if isinstance(nd, E.Min) else max(vals)) if vals else None)
    names = [wdb_sql._alias(p) for p in tree.expressions]
    if _bill is not None:
        _bill.append(('emit', _tk() - _t0))
        print('SEMI BILL: ' + ' | '.join('%s=%.0fms' % (n, v * 1000) for n, v in _bill), flush=True)
    return [tuple(out)], names
