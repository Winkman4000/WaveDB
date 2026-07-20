"""wdb_groupself -- the counting board: GROUP BY K where the WHERE lives on K itself.

The whole query is dictionary arithmetic: one walk drops beans on the board
(bincount of raw codes), the filter is a BIN operation (= keeps one square,
<> sweeps one, IN keeps a set, NOT IN sweeps a set), HAVING masks counts,
ORDER BY count is a top-k on the board, and exactly LIMIT winners are ever
decoded -- by point-fetch, never the dictionary. The rule: only decode when
needed, never more than needed.

Ties straddling the LIMIT boundary decline (execute returns None, controller
falls through) so tie-breaking stays consistent with the scan path -- the
gbcount precedent. A NULL group with no WHERE also declines: SQL emits the
NULL group, and that is the scan path's job (every WHERE op we accept is
null-excluding by SQL semantics, so with terms present the null bin zeroes)."""
import numpy as np
import wdb_sql
import wdb_policies as P
import workers
import wdb_wherescan as WS

E = wdb_sql.E
_HITS = 0


def _term(cj, col):
    """One WHERE conjunct as a bin operation on `col`, or None (not our shape)."""
    if isinstance(cj, E.Not) and isinstance(cj.this, E.In):
        node = cj.this
        if node.args.get('query') is not None or node.args.get('_codes') is not None:
            return None                           # subquery/code-set IN: wherescan's job
        if not (isinstance(node.this, E.Column) and node.this.name == col):
            return None
        vals = [WS._litval(x) for x in node.expressions]
        return None if any(v is None for v in vals) else ('nin', vals)
    if isinstance(cj, E.In):
        if cj.args.get('query') is not None or cj.args.get('_codes') is not None:
            return None                           # subquery/code-set IN: wherescan's job
        if not (isinstance(cj.this, E.Column) and cj.this.name == col):
            return None
        vals = [WS._litval(x) for x in cj.expressions]
        return None if any(v is None for v in vals) else ('in', vals)
    cl = WS._col_lit(cj)
    if cl is None:
        return None
    tcol, val, op = cl[0], cl[1], cl[2]
    if tcol != col:
        return None
    if op == '=':
        return ('in', [val])
    if op == '<>':
        return ('nin', [val])
    return None


def _having(node, count_alias):
    """HAVING COUNT(*) OP literal (or the count's alias) -> (op, value), else None."""
    ops = {E.GT: '>', E.GTE: '>=', E.LT: '<', E.LTE: '<=', E.EQ: '='}
    for k, s in ops.items():
        if isinstance(node, k):
            lhs, rhs = node.this, node.expression
            v = WS._litval(rhs)
            if v is None:
                return None
            ak = wdb_sql._agg_kind(lhs)
            if ak is not None and ak[0] == 'COUNT_STAR':
                return (s, float(v))
            if isinstance(lhs, E.Column) and lhs.name == count_alias:
                return (s, float(v))
    return None


def detect(seg, tree, col_map):
    """ACTIVATION: pure shape decision, touches no row data."""
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.single_group_key(tree):   return None
    if not P.has_limit(tree):          return None
    proj = tree.expressions
    if len(proj) != 2:
        return None
    ci = None
    for i, p in enumerate(proj):
        ak = wdb_sql._agg_kind(p)
        if ak is not None:
            if ak[0] != 'COUNT_STAR' or ci is not None:
                return None
            ci = i
    if ci is None:
        return None
    ki = 1 - ci
    if wdb_sql._agg_kind(proj[ki]) is not None:
        return None
    kp = proj[ki]
    kexpr = kp.this if isinstance(kp, E.Alias) else kp
    lower = isinstance(kexpr, E.Lower)
    ge = tree.args.get('group').expressions[0]
    if lower:
        # LOWER(col) AS l ... GROUP BY l: the key is the lower-collation gid --
        # the lmap folds the board's V bins into G groups, no strings anywhere
        if not isinstance(kexpr.this, E.Column):
            return None
        knm = kexpr.this.name
        kalias = wdb_sql._alias(kp)
        gmatch = ((isinstance(ge, E.Column) and ge.name == kalias)
                  or (isinstance(ge, E.Lower) and isinstance(ge.this, E.Column)
                      and ge.this.name == knm))
        if not gmatch:
            return None
    else:
        knm = wdb_sql._proj_colname(kp)
        gnm = wdb_sql._colname(ge)
        if knm is None or gnm is None or knm != gnm:
            return None
    col = col_map.get(knm, knm) if col_map else knm
    if not P.columns_exist(seg, col):  return None
    if not P.not_positional(seg, col): return None
    if not P.no_deleted_rows(seg):     return None
    c = seg.cols[col]
    if c.get('mode') not in (0, 1):
        return None                                      # dict columns (v1)
    if lower and c.get('has_null'):
        return None                                      # NULL under LOWER: scan path
    if seg._effective(col) is not None:
        return None
    terms = []
    w = tree.args.get('where')
    if w is not None:
        for cj in WS._conjuncts(w.this):
            t = _term(cj, col)
            if t is None:
                return None
            terms.append(t)
    if c.get('has_null') and not terms:
        return None                                      # NULL group must be emitted: scan path
    hv = None
    having = tree.args.get('having')
    if having is not None:
        hv = _having(having.this, wdb_sql._alias(proj[ci]))
        if hv is None:
            return None
    order = tree.args.get('order')
    if order is None or not order.expressions:
        return None
    first = order.expressions[0]
    if not isinstance(first, E.Ordered) or not first.args.get('desc'):
        return None
    tgt = first.this
    ok = isinstance(tgt, E.Column) and tgt.name == wdb_sql._alias(proj[ci])
    if not ok:
        ak = wdb_sql._agg_kind(tgt)
        ok = ak is not None and ak[0] == 'COUNT_STAR'
    if not ok:
        return None
    lim = wdb_sql._limit(tree)
    if lim is None or wdb_sql._offset(tree):
        return None
    return {'col': col, 'ci': ci, 'ki': ki, 'lim': int(lim), 'terms': terms,
            'having': hv, 'proj': proj, 'order': order, 'lower': lower}


def execute(seg, spec):
    """THE READ: one walk feeds the board; everything after happens on V squares."""
    global _HITS
    col = spec['col']
    c = seg.cols[col]
    codes = np.asarray(seg._raw_codes(col))
    V = int(c['V'])
    cn = np.bincount(codes, minlength=V).astype(np.int64)
    if c.get('has_null'):
        cn[V - 1] = 0                # terms present (detect gate): every op excludes NULL
    for op, vals in spec['terms']:
        kcs = [WS._code_of(seg, col, v) for v in vals]
        if op == 'in':
            keep = np.zeros(V, bool)
            for k in kcs:
                if k is not None:
                    keep[k] = True
            cn = np.where(keep, cn, 0)
        else:                        # nin: sweep the named squares
            for k in kcs:
                if k is not None:
                    cn[k] = 0
    rep = None
    if spec.get('lower'):
        # fold the V bins into G lower-collation groups through the memmapped lmap:
        # WHERE already acted in original code space (rows filtered), the fold is V-sized
        import wdb_lmap
        lr = wdb_lmap.load_or_build(seg, col)
        if lr is None:
            return None
        mp, rep = lr
        cg = np.zeros(rep.size, np.int64)
        np.add.at(cg, np.asarray(mp), cn)
        cn = cg
    live = cn > 0
    if spec['having'] is not None:
        op, val = spec['having']
        live &= (cn > val) if op == '>' else (cn >= val) if op == '>=' else \
                (cn < val) if op == '<' else (cn <= val) if op == '<=' else (cn == val)
    idx = np.flatnonzero(live)
    hdrs = [wdb_sql._alias(p) for p in spec['proj']]
    if idx.size == 0:
        _HITS += 1
        return [], hdrs
    k = min(spec['lim'], int(idx.size))
    cnl = cn[idx]
    if idx.size > k:
        part = np.argpartition(-cnl, k - 1)
        top = part[:k]
        rest_max = int(cnl[part[k:]].max())
        top = top[np.argsort(-cnl[top], kind='stable')]
        if int(cnl[top[-1]]) == rest_max:
            return None              # tie straddles the LIMIT boundary: scan path decides
        sel = idx[top]
    else:
        sel = idx[np.argsort(-cnl, kind='stable')]
    rows = []
    for g in sel.tolist():
        row = [None, None]
        if rep is not None:
            import wdb_lmap
            row[spec['ki']] = wdb_lmap._lower(seg.fetch(col, int(rep[g])))
        else:
            row[spec['ki']] = wdb_sql._pyval(seg.fetch(col, int(g)))   # decode ONLY winners
        row[spec['ci']] = int(cn[g])
        rows.append(tuple(row))
    rows = workers.finalize(rows, spec['proj'], spec['order'], spec['lim'])
    _HITS += 1
    return rows, hdrs
