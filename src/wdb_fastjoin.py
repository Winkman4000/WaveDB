"""wdb_fastjoin: dimension joins through the dictionary -- the introductions happen in
dict space, never in row space.

The analytic join shape is fact-vs-dimension: a giant table whose key column carries dict
codes, and a small table describing those keys. Two rewrites cover it:

  A) Conditions on the dimension become a SEMI-JOIN: run the dim query (small), collect the
     matching keys, and the fact side runs single-table with key IN (...) -- the IN driver
     does the row work as one flag scan. The join evaporates.

  B) Grouping by dimension attributes becomes GROUP-BY-KEY + POST-MAP: the fact side groups
     by its own key column (existing fast reads), then key->attribute mapping and
     re-aggregation happen on the GROUP CELLS (thousands), not the rows (millions).
     COUNT/SUM/MIN/MAX fold exactly (the grouping-sets lemma); AVG declines.

One-to-many dimensions (duplicate keys) would multiply rows -- decline to the mature join
path. Everything here is INNER equi-join; other kinds fall through untouched.
"""
import sqlglot.expressions as E
import wdb_sql

_DIM_CAP = 500_000


def _tables(tree):
    f = tree.args.get('from_') or tree.args.get('from')
    joins = tree.args.get('joins') or []
    if f is None or len(joins) != 1 or not isinstance(f.this, E.Table):
        return None
    j = joins[0]
    side = (j.side or '').upper()
    if side not in ('', 'INNER', 'LEFT') or (j.kind or '').upper() not in ('', 'INNER'):
        return None
    if not isinstance(j.this, E.Table):
        return None
    t1, t2 = f.this, j.this
    on = j.args.get('on')
    if on is None or type(on).__name__ != 'EQ':
        return None
    a, b = on.this, on.expression
    if not (isinstance(a, E.Column) and isinstance(b, E.Column)):
        return None
    return (t1.name, t1.alias or t1.name), (t2.name, t2.alias or t2.name), (a, b), side


def _split_where(tree, quals1, quals2):
    import wdb_wherescan as WS
    w = tree.args.get('where')
    c1, c2 = [], []
    if w is None:
        return c1, c2
    for c in WS._conjuncts(w.this):
        tabs = {x.table for x in c.find_all(E.Column) if x.table}
        unq = [x for x in c.find_all(E.Column) if not x.table]
        if unq:
            return None                          # v1: every column must be qualified
        if tabs <= quals1:
            c1.append(c)
        elif tabs <= quals2:
            c2.append(c)
        else:
            return None                          # mixed-side condition: not separable
    return c1, c2


def _strip_qual(node):
    n = node.copy()
    for col in n.find_all(E.Column):
        col.set('table', None)
    return n


def try_execute(db, tree):
    """Rows/headers for a fact-dim join, or None to fall through."""
    t = _tables(tree)
    if t is None:
        return None
    (n1, a1), (n2, a2), (ka, kb), side = t
    try:
        sizes = {a1: db.cat.get_table(n1).get('rows'), a2: db.cat.get_table(n2).get('rows')}
    except KeyError:
        return None
    # fact = the side of the ON key we scan; dim = the side we materialize
    for fact_al, dim_al, fact_tn, dim_tn, fk, dk in (
            (a1, a2, n1, n2, ka, kb), (a2, a1, n2, n1, kb, ka)):
        if fk.table != fact_al or dk.table != dim_al:
            continue
        if side == 'LEFT' and fact_al != (t[0][1]):
            continue                             # LEFT keeps the FROM side's rows: fact must be left
        out = _try_orientation(db, tree, fact_al, dim_al, fact_tn, dim_tn, fk.name, dk.name,
                               left=(side == 'LEFT'))
        if out is not None:
            return out
    return None


def _try_orientation(db, tree, fact_al, dim_al, fact_tn, dim_tn, fkey, dkey, left=False):
    proj = tree.expressions
    dim_cols, fact_cols, aggs = [], [], []
    for pi, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        ak = wdb_sql._agg_kind(p)
        if ak is not None:
            if ak[0] not in ('COUNT_STAR', 'SUM', 'MIN', 'MAX'):
                return None
            if ak[0] != 'COUNT_STAR':
                arg = inner.this
                if not (isinstance(arg, E.Column) and arg.table == fact_al):
                    return None                  # aggregate args live on the fact
            aggs.append((pi, ak[0]))
        elif isinstance(inner, E.Column) and inner.table == dim_al:
            dim_cols.append((pi, inner.name))
        elif isinstance(inner, E.Column) and inner.table == fact_al:
            fact_cols.append((pi, inner.name))
        else:
            return None
    group = tree.args.get('group')
    if group is not None:
        gnames = set()
        for g in group.expressions:
            if not isinstance(g, E.Column):
                return None
            gnames.add((g.table, g.name))
    sw = _split_where(tree, {fact_al, fact_tn}, {dim_al, dim_tn})
    if sw is None:
        return None
    fact_conds, dim_conds = sw
    if left and dim_conds:
        left = False                             # WHERE on the dim side nullifies LEFT
    if tree.args.get('having') is not None or tree.args.get('qualify') is not None:
        return None
    # ---- dim side: key + needed attributes, filtered ----
    need = sorted({nm for _pi, nm in dim_cols})
    dsel = 'SELECT ' + ', '.join([dkey] + need) + ' FROM ' + dim_tn
    if dim_conds:
        dsel += ' WHERE ' + ' AND '.join(_strip_qual(c).sql() for c in dim_conds)
    dpaths = db.cat.segment_paths(dim_tn)
    if len(dpaths) == 1:
        dseg_n = int(db.open_segment(dpaths[0], dim_tn).N)
        if dseg_n > _DIM_CAP:
            if left:
                return None
            return _giant_m2o(db, tree, fact_al, dim_al, fact_tn, dim_tn, fkey, dkey,
                              proj, dim_cols, fact_cols, aggs, fact_conds, dim_conds)
    drows_out = db.run(dsel)
    drows = drows_out[0] if isinstance(drows_out, tuple) else drows_out
    if len(drows) > _DIM_CAP:
        return None
    dmap = {}
    for r in drows:
        dmap.setdefault(r[0], []).append(r[1:])  # one-to-many: each match multiplies the row
    if not dmap and not left:
        return [], [wdb_sql._alias(p) for p in proj]
    multi = any(len(v) > 1 for v in dmap.values())
    if multi and any(kd in ('MIN', 'MAX') for _pi, kd in aggs):
        pass                                     # MIN/MAX are duplication-immune: fine
    attr_idx = {nm: i for i, nm in enumerate(need)}
    keys = list(dmap.keys())
    # ---- fact side: single-table, key IN (semi-join), grouped by (key + fact group cols) ----
    def lit(v):
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        return str(v)
    fact_where = [_strip_qual(c).sql() for c in fact_conds]
    if not left and not aggs:
        # dump path only: the IN drives the row scan. The AGG path never needs it -- the
        # post-map enforces INNER at the CELL level (dmap.get drops unmatched keys among
        # thousands of cells), and a complete dimension's IN filters nothing while pushing
        # a millisecond groupby off every fast read into the general grind.
        fact_where.append(fkey + ' IN (' + ', '.join(lit(k) for k in keys) + ')')
    if aggs:
        gcols = [fkey] + sorted({nm for _pi, nm in fact_cols})
        fexpr = list(gcols)
        for pi, kd in aggs:
            p = proj[pi]
            inner = p.this if isinstance(p, E.Alias) else p
            fexpr.append(_strip_qual(inner).sql() + ' AS __a%d' % pi)
        fsel = ('SELECT ' + ', '.join(fexpr) + ' FROM ' + fact_tn
                + ((' WHERE ' + ' AND '.join(fact_where)) if fact_where else '')
                + ' GROUP BY ' + ', '.join(gcols))
        frows_out = db.run(fsel)
        frows = frows_out[0] if isinstance(frows_out, tuple) else frows_out
        # ---- post-map on CELLS: key -> dim attrs, re-aggregate, order, limit ----
        fpos = {nm: i for i, nm in enumerate(gcols)}
        acc = {}
        for r in frows:
            matches = dmap.get(r[0])
            if matches is None:
                if not left:
                    continue
                matches = [tuple([None] * len(need))]
            avals = r[len(gcols):]
            for dvals in matches:
                gkey = []
                for pi, nm in dim_cols:
                    gkey.append(dvals[attr_idx[nm]])
                for pi, nm in fact_cols:
                    gkey.append(r[fpos[nm]])
                gkey = tuple(gkey)
                cur = acc.get(gkey)
                if cur is None:
                    acc[gkey] = list(avals)
                else:
                    for i, (_pi, kd) in enumerate(aggs):
                        v = avals[i]
                        if kd in ('COUNT_STAR', 'SUM'):
                            cur[i] = cur[i] + v
                        elif kd == 'MIN':
                            cur[i] = v if v < cur[i] else cur[i]
                        else:
                            cur[i] = v if v > cur[i] else cur[i]
        rows = []
        for gkey, avals in acc.items():
            row = [None] * len(proj)
            ki = 0
            for pi, nm in dim_cols:
                row[pi] = gkey[ki]; ki += 1
            for pi, nm in fact_cols:
                row[pi] = gkey[ki]; ki += 1
            for i, (pi, _kd) in enumerate(aggs):
                row[pi] = avals[i]
            rows.append(tuple(row))
    else:
        # plain projection dump: bounded only
        lim = wdb_sql._limit(tree)
        if lim is None or lim > 1_000_000:
            return None
        st = _stream_dump(db, tree, fact_tn, fkey, fact_conds, dmap, need, attr_idx,
                          dim_cols, fact_cols, proj, lim, left)
        if st is not None:
            return st
        fexpr = [fkey] + sorted({nm for _pi, nm in fact_cols})
        fsel = ('SELECT ' + ', '.join(fexpr) + ' FROM ' + fact_tn
                + ((' WHERE ' + ' AND '.join(fact_where)) if fact_where else ''))
        if tree.args.get('order') is None:
            fsel += ' LIMIT ' + str(lim * 2)
        frows_out = db.run(fsel)
        frows = frows_out[0] if isinstance(frows_out, tuple) else frows_out
        fpos = {nm: i for i, nm in enumerate(fexpr)}
        rows = []
        for r in frows:
            matches = dmap.get(r[0])
            if matches is None:
                if not left:
                    continue
                matches = [tuple([None] * len(need))]
            for dvals in matches:
                row = [None] * len(proj)
                for pi, nm in dim_cols:
                    row[pi] = dvals[attr_idx[nm]]
                for pi, nm in fact_cols:
                    row[pi] = r[fpos[nm]]
                rows.append(tuple(row))
    order = tree.args.get('order')
    if order is not None:
        names = [wdb_sql._alias(p) for p in proj]
        for oe in reversed(order.expressions):
            nm = oe.this.name if isinstance(oe.this, E.Column) else None
            if nm is None or nm not in names:
                return None
            idx = names.index(nm)
            rows.sort(key=lambda r: (r[idx] is None, r[idx]), reverse=bool(oe.args.get('desc')))
    lim = wdb_sql._limit(tree)
    off = wdb_sql._offset(tree) or 0
    if lim is not None or off:
        rows = rows[off: None if lim is None else off + lim]
    return rows, [wdb_sql._alias(p) for p in proj]


def _stream_dump(db, tree, fact_tn, fkey, fact_conds, dmap, need, attr_idx,
                 dim_cols, fact_cols, proj, lim, left):
    """The library run, Jackson's pattern: walk the fact table shelf by shelf, glance at
    code stickers against the golden-list flag card, toss matches in the cart, and LEAVE
    the moment it holds LIMIT rows. A prefix of the disk answers the join dump; a walk cap
    falls back to the full path when matches are too rare."""
    import numpy as np
    if tree.args.get('order') is not None or lim > 100_000:
        return None
    fpaths = db.cat.segment_paths(fact_tn)
    if len(fpaths) != 1:
        return None
    fseg = db.open_segment(fpaths[0], fact_tn)
    kc0 = fseg.cols.get(fkey)
    if kc0 is None or kc0.get('mode') not in (0, 1, 2):
        return None
    import wdb_wherescan as WS
    # per-block evaluable conditions: eq / neq against a literal, in code space
    conds = []
    for c in fact_conds:
        tn = type(c).__name__
        if tn not in ('EQ', 'NEQ') or not isinstance(c.this, E.Column) \
                or not isinstance(c.expression, E.Literal):
            return None
        nm = c.this.name
        cc = fseg.cols.get(nm)
        if cc is None or cc.get('mode') not in (0, 1, 2):
            return None
        lv = c.expression.this if c.expression.is_string else str(c.expression.this)
        code = WS._code_of(fseg, nm, lv)
        conds.append((nm, tn, code))
    V = int(kc0['V'])
    flag = np.zeros(V + 1, bool)
    if not left:
        tc = WS._in_codes(fseg, fkey, list(dmap.keys()))   # values AS-IS: the bulk
                                                           # binder matches native types
        tc = np.asarray(tc, dtype=np.int64)
        if tc.size == 0:
            return [], [wdb_sql._alias(p) for p in proj]
        flag[tc] = True
    N = int(fseg.N)
    step = 1 << 18
    cap = min(N, max(step * 64, lim * 512))
    hits = []
    lo = 0
    while lo < cap and len(hits) < lim:
        hi = min(lo + step, cap)
        m = None
        if not left:
            kcb = np.asarray(fseg._raw_codes_range(fkey, lo, hi)).astype(np.int64)
            m = flag[kcb]
        for nm, tn, code in conds:
            cb = np.asarray(fseg._raw_codes_range(nm, lo, hi)).astype(np.int64)
            if code is None:
                cm = np.zeros(cb.size, bool) if tn == 'EQ' else np.ones(cb.size, bool)
            else:
                cm = (cb == code) if tn == 'EQ' else (cb != code)
            m = cm if m is None else (m & cm)
        if m is None:
            m = np.ones(hi - lo, bool)
        for p0 in np.nonzero(m)[0]:
            hits.append(lo + int(p0))
            if len(hits) >= lim:
                break
        lo = hi
    if len(hits) < lim and lo < N:
        return None                              # cart still hungry: full path serves
    pos = np.asarray(hits[:lim], dtype=np.int64)
    def dec_at(nm, positions):
        cc = np.asarray(fseg.codes_at(nm, positions)).astype(np.int64)
        cache = {}
        out = []
        for x in cc:
            x = int(x)
            v = cache.get(x, cache)
            if v is cache:
                v = wdb_sql._pyval(fseg.fetch(nm, x))
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode('utf-8', 'replace')
                cache[x] = v
            out.append(v)
        return out
    kvals = dec_at(fkey, pos)
    fvals = {nm: dec_at(nm, pos) for nm in sorted({nm for _pi, nm in fact_cols})}
    rows = []
    for i in range(pos.size):
        matches = dmap.get(kvals[i])
        if matches is None:
            if not left:
                continue
            matches = [tuple([None] * len(attr_idx))]
        for dvals in matches:
            row = [None] * len(proj)
            for pi, nm in dim_cols:
                row[pi] = dvals[attr_idx[nm]]
            for pi, nm in fact_cols:
                row[pi] = fvals[nm][i]
            rows.append(tuple(row))
            if len(rows) >= lim:
                break
        if len(rows) >= lim:
            break
    return rows, [wdb_sql._alias(p) for p in proj]


def _giant_m2o(db, tree, fact_al, dim_al, fact_tn, dim_tn, fkey, dkey,
               proj, dim_cols, fact_cols, aggs, fact_conds, dim_conds):
    """Both sides giant, many-to-one: the join is three gathers and a bincount. The
    dictionaries meet ONCE (both value-sorted: one searchsorted builds the translation);
    the dim key's raw codes give a row-per-code map; then every fact row's group cell is
    attr_codes[rowof[translate[key_codes]]] -- integers end to end, no row ever decodes."""
    import numpy as np
    if fact_cols or not aggs or not dim_cols or len(dim_cols) > 2:
        return None                              # v1: dim-attr grouping + aggregates only
    if any(kd not in ('COUNT_STAR', 'SUM') for _pi, kd in aggs):
        return None
    fpaths = db.cat.segment_paths(fact_tn)
    dpaths = db.cat.segment_paths(dim_tn)
    if len(fpaths) != 1 or len(dpaths) != 1:
        return None
    fseg = db.open_segment(fpaths[0], fact_tn)
    dseg = db.open_segment(dpaths[0], dim_tn)
    import wdb_window as WN
    try:
        fk_vals = WN._int_table(fseg, fkey)
        dk_vals = WN._int_table(dseg, dkey)
    except Exception:
        return None                              # v1: integer-valued key dictionaries
    # the introduction: two sorted dictionaries, one merge
    t_pos = np.searchsorted(dk_vals, fk_vals)
    t_pos_c = np.minimum(t_pos, dk_vals.size - 1)
    trans = np.where(dk_vals[t_pos_c] == fk_vals, t_pos_c, -1).astype(np.int64)
    # dim row per key code (m2o requires unique keys)
    dkc = np.asarray(dseg._raw_codes(dkey)).astype(np.int64)
    N2 = int(dseg.N)
    kcounts = np.bincount(dkc, minlength=dk_vals.size)
    if kcounts.max() > 1:
        return None                              # duplicate keys: multiplying join
                                                 # (bincount, not np.unique: no 9 s sort)
    keep_dim = np.ones(N2, bool)
    for c in dim_conds:
        keep_dim &= wdb_sql._eval_pred(dseg, _strip_qual(c),
                                       lambda nm: nm)
    rowof = np.full(dk_vals.size, -1, dtype=np.int64)
    rowof[dkc[keep_dim]] = np.nonzero(keep_dim)[0]
    # fact side: key codes -> dim row, masked by fact conditions
    fkc = np.asarray(fseg._raw_codes(fkey)).astype(np.int64)
    frow = np.where(trans[fkc] >= 0, rowof[np.maximum(trans[fkc], 0)], -1)
    keep = frow >= 0
    for c in fact_conds:
        keep &= wdb_sql._eval_pred(fseg, _strip_qual(c), lambda nm: nm)
    frow_k = frow[keep]
    # group cells: composed dim-attr codes (radix fold), aggregated by bincount
    acodes_all, spans, atabs = [], [], []
    for _pi, nm in dim_cols:
        c2 = dseg.cols.get(nm)
        if c2 is None or c2.get('mode') not in (0, 1, 2):
            return None
        ac = np.asarray(dseg._raw_codes(nm)).astype(np.int64)
        acodes_all.append(ac[frow_k]); spans.append(int(c2['V'])); atabs.append(nm)
    comp = acodes_all[0]
    for i in range(1, len(acodes_all)):
        comp = comp * spans[i] + acodes_all[i]
    total = 1
    for v in spans:
        total *= v
    if total > 50_000_000:
        return None
    counts = np.bincount(comp, minlength=total)
    sums = {}
    for pi, kd in aggs:
        if kd == 'SUM':
            p = proj[pi]
            inner = p.this if isinstance(p, E.Alias) else p
            v = WN._numvals(fseg, inner.this.name)[keep]
            sums[pi] = np.bincount(comp, weights=v, minlength=total)
    live = np.nonzero(counts)[0]
    rows = []
    for cell in live:
        rem = int(cell); key_codes = []
        for i in range(len(spans) - 1, -1, -1):
            key_codes.append(rem % spans[i]); rem //= spans[i]
        key_codes.reverse()
        row = [None] * len(proj)
        for (pi, nm), kc in zip(dim_cols, key_codes):
            v = wdb_sql._pyval(dseg.fetch(nm, int(kc)))
            row[pi] = v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else v
        for pi, kd in aggs:
            row[pi] = int(counts[cell]) if kd == 'COUNT_STAR' else float(sums[pi][cell])
        rows.append(tuple(row))
    order = tree.args.get('order')
    if order is not None:
        names = [wdb_sql._alias(p) for p in proj]
        for oe in reversed(order.expressions):
            nm = oe.this.name if isinstance(oe.this, E.Column) else None
            if nm is None or nm not in names:
                return None
            idx = names.index(nm)
            rows.sort(key=lambda r: (r[idx] is None, r[idx]), reverse=bool(oe.args.get('desc')))
    lim = wdb_sql._limit(tree)
    off = wdb_sql._offset(tree) or 0
    if lim is not None or off:
        rows = rows[off: None if lim is None else off + lim]
    return rows, [wdb_sql._alias(p) for p in proj]
