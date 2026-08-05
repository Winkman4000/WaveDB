"""wdb_groupsets: GROUPING SETS / ROLLUP / CUBE as controller-level composition.

Each grouping set is a plain GROUP BY the engine is already fast at: the sets run through
the full pipeline one by one, rows assemble into full-width tuples with NULL in the absent
key slots (per the standard), and the outer ORDER/LIMIT apply to the union. ROLLUP(a,b)
expands to ((a,b),(a),()); CUBE to all subsets (capped at 4 keys -- 16 sub-queries); the
empty set is the global aggregate. HAVING passes into every sub-query, which is exactly its
per-group semantics.
"""
import sqlglot.expressions as E
import wdb_sql


def has_grouping(tree):
    g = tree.args.get('group')
    if g is None:
        return False
    return bool(g.args.get('rollup') or g.args.get('cube') or g.args.get('grouping_sets'))


def _colname(e):
    if isinstance(e, E.Paren):
        e = e.this
    return e.name if isinstance(e, E.Column) else None


def _sets_of(tree):
    g = tree.args.get('group')
    plain = [x.name for x in g.expressions if isinstance(x, E.Column)]
    out = []
    for r in g.args.get('rollup') or []:
        names = [x.name for x in r.expressions]
        for i in range(len(names), -1, -1):
            out.append(tuple(plain + names[:i]))
    for c in g.args.get('cube') or []:
        names = [x.name for x in c.expressions]
        if len(names) > 4:
            raise NotImplementedError("CUBE past 4 keys (%d sub-queries)" % (2 ** len(names)))
        for mask in range(2 ** len(names) - 1, -1, -1):
            out.append(tuple(plain + [n for i, n in enumerate(names) if mask & (1 << i)]))
    for gs in g.args.get('grouping_sets') or []:
        for t in gs.expressions:
            if isinstance(t, E.Tuple):
                out.append(tuple(plain + [x.name for x in t.expressions]))
            elif isinstance(t, E.Paren):
                out.append(tuple(plain + ([t.this.name] if isinstance(t.this, E.Column) else [])))
            elif isinstance(t, E.Column):
                out.append(tuple(plain + [t.name]))
            else:
                raise NotImplementedError(f"grouping set element {type(t).__name__}")
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq


def _fused_2key(db, tree):
    """Stage B: filtered 2-key CUBE/ROLLUP as ONE grid -- count and weighted-sum boards
    over the survivors, every grouping set a sideways sum of the same board. Replaces
    2**k sub-queries (each re-filtering, re-sorting) with one filter + one bincount per
    aggregate. Narrow by design: single table, one eq conjunct, 2 plain CUBE/ROLLUP keys,
    COUNT(*)/SUM aggs; anything else declines to composition."""
    import numpy as np
    import wdb_policies as P
    if not P.no_joins(tree) or not P.no_having(tree) or not P.no_select_distinct(tree):
        return None
    if tree.args.get('qualify') is not None:
        return None
    g = tree.args.get('group')
    if [x for x in g.expressions if isinstance(x, E.Column)]:
        return None                                     # plain prefix keys: compose
    src = (g.args.get('cube') or []) + (g.args.get('rollup') or [])
    if len(src) != 1 or len(src[0].expressions) != 2:
        return None
    knames = [x.name for x in src[0].expressions]
    w = tree.args.get('where')
    if w is None:
        return None
    cj = w.this
    if isinstance(cj, E.Paren):
        cj = cj.this
    if not isinstance(cj, E.EQ) or not isinstance(cj.expression, E.Literal):
        return None
    fcol = wdb_sql._colname(cj.this)
    if fcol is None:
        return None
    flit = cj.expression.this
    frm = tree.args.get('from_') or tree.args.get('from')
    if frm is None:
        return None
    tabs = list(frm.find_all(E.Table))
    if len(tabs) != 1:
        return None
    tname = tabs[0].name
    seg = db.open_segment(db.cat.segment_paths(tname)[0], tname)
    fc = seg.cols.get(fcol)
    if fc is None or fc.get('mode') not in (0, 1, 2) or fc.get('has_null'):
        return None
    aggs = []                                           # (proj_index, kind, col_or_None)
    keyslot = {}
    for pi, p in enumerate(tree.expressions):
        inner = p.this if isinstance(p, E.Alias) else p
        nm = wdb_sql._colname(inner)
        if nm in knames:
            keyslot[nm] = pi
            continue
        kd = wdb_sql._agg_kind(p)
        if kd is None:
            return None
        if kd[0] == 'COUNT_STAR':
            aggs.append((pi, 'COUNT', None))
        elif kd[0] == 'SUM':
            ac = wdb_sql._colname(inner.this)
            if ac is None:
                return None
            aggs.append((pi, 'SUM', ac))
        else:
            return None
    if len(keyslot) != 2 or not aggs:
        return None
    for nm in knames:
        c = seg.cols.get(nm)
        if c is None or c.get('has_null') or c.get('mode') not in (0, 1, 2, 4):
            return None
    for _pi, _kd, ac in aggs:
        if ac is not None:
            c = seg.cols.get(ac)
            if c is None or c.get('has_null') or c.get('mode') not in (0, 1, 2, 4):
                return None
    if not P.no_deleted_rows(seg):
        return None
    # ---- filter: driver scan on the eq column ----
    import wdb_wherescan as WS
    import zstandard
    kc = WS._code_of(seg, fcol, flit)
    if kc is None:
        pos = np.empty(0, np.int64)
    elif fc.get('boffs') is not None and fc.get('BR'):
        base = fc['cstart']; bo = fc['boffs']; BR = int(fc['BR'])
        wdt = np.uint8 if fc['cwidth'] == 1 else (np.uint16 if fc['cwidth'] == 2 else np.uint32)
        dz = zstandard.ZstdDecompressor()
        parts = []
        for j in range(len(bo) - 1):
            raw = np.frombuffer(dz.decompress(seg.buf[base + int(bo[j]):base + int(bo[j + 1])].tobytes()), dtype=wdt)
            hit = np.flatnonzero(raw == int(kc))
            if hit.size:
                parts.append(hit.astype(np.int64) + j * BR)
        pos = np.concatenate(parts) if parts else np.empty(0, np.int64)
    else:
        import wdb_fpm
        pos = wdb_fpm.eq_positions(seg, fcol, int(kc))
        if pos is None:                          # ineligible column: full read
            raw = np.asarray(seg._raw_codes(fcol))
            pos = np.flatnonzero(raw == int(kc)).astype(np.int64)
    # ---- key arrays and numeric gathers at survivors only ----
    import wdb_window as W
    def keys_at(nm):
        c = seg.cols[nm]
        if c.get('mode') == 4:
            v = np.asarray(seg._seq_decode(c))[pos].astype(np.int64)
            return v, (int(v.max()) + 1 if v.size else 1), None
        cc = np.asarray(seg.codes_at(nm, pos)).astype(np.int64) if pos.size else np.empty(0, np.int64)
        return cc, int(c['V']), c
    def nums_at(nm):
        c = seg.cols[nm]
        if c.get('mode') == 4:
            return np.asarray(seg._seq_decode(c))[pos].astype(np.float64)
        cc = np.asarray(seg.codes_at(nm, pos)).astype(np.int64) if pos.size else np.empty(0, np.int64)
        return np.asarray(W._int_table(seg, nm))[cc].astype(np.float64)
    k1, K1, c1 = keys_at(knames[0])
    k2, K2, c2 = keys_at(knames[1])
    if K1 * K2 > 4_000_000:
        return None                                     # grid too big: compose instead
    cells = k1 * K2 + k2
    boards = {}
    boards['COUNT'] = np.bincount(cells, minlength=K1 * K2).reshape(K1, K2)
    for _pi, kd, ac in aggs:
        if kd == 'SUM' and ('SUM', ac) not in boards:
            boards[('SUM', ac)] = np.bincount(cells, weights=nums_at(ac),
                                              minlength=K1 * K2).reshape(K1, K2)
    nz = boards['COUNT'] > 0
    def _val(nm, c, code):
        if c is None:
            return int(code)                            # mode 4: values are themselves
        v = wdb_sql._pyval(seg.fetch(nm, int(code)))
        if isinstance(v, (bytes, bytearray)):
            v = v.decode('utf-8', 'replace')
        return v
    rows = []                                           # (sortkey_tuple, out_row_list)
    nslots = len(tree.expressions)
    def emit(level, i, j, cvals):
        out = [None] * nslots
        if 'a' in level:
            out[keyslot[knames[0]]] = ('K1', i)
        if 'b' in level:
            out[keyslot[knames[1]]] = ('K2', j)
        for (pi, kd, ac), v in zip(aggs, cvals):
            out[pi] = v if v is None else (float(v) if kd == 'SUM' else int(v))
        rows.append(out)
    sets = _sets_of(tree)
    want = set()
    for st in sets:
        key = ''.join(sorted('a' if n == knames[0] else 'b' for n in st))
        want.add(key)
    for level in want:
        if level == 'ab':
            ii, jj = np.nonzero(nz)
            for i, j in zip(ii.tolist(), jj.tolist()):
                emit(level, i, j, [boards['COUNT'][i, j] if kd == 'COUNT' else boards[('SUM', ac)][i, j]
                                   for _pi, kd, ac in aggs])
        elif level == 'a':
            cnt = boards['COUNT'].sum(1)
            s1 = {ac: boards[('SUM', ac)].sum(1) for _pi, kd, ac in aggs if kd != 'COUNT'}
            for i in np.flatnonzero(cnt).tolist():          # sums hoisted: the per-row
                emit(level, i, 0, [cnt[i] if kd == 'COUNT' else s1[ac][i]   # recompute was
                                   for _pi, kd, ac in aggs])                # 213ms on gs-cube
        elif level == 'b':
            cnt = boards['COUNT'].sum(0)
            s0 = {ac: boards[('SUM', ac)].sum(0) for _pi, kd, ac in aggs if kd != 'COUNT'}
            for j in np.flatnonzero(cnt).tolist():
                emit(level, 0, j, [cnt[j] if kd == 'COUNT' else s0[ac][j]
                                   for _pi, kd, ac in aggs])
        else:
            # the () set aggregates the whole (filtered) input: one row ALWAYS,
            # COUNT=0 and SUM=NULL over an empty input (duck-verified semantics)
            emit(level, 0, 0, [int(boards['COUNT'].sum()) if kd == 'COUNT'
                               else (float(boards[('SUM', ac)].sum()) if pos.size else None)
                               for _pi, kd, ac in aggs])
    # ---- outer ORDER BY agg + LIMIT over the union ----
    order = tree.args.get('order')
    lim = wdb_sql._limit(tree)
    if order is not None:
        oe = order.expressions
        if len(oe) != 1:
            return None
        onm = wdb_sql._colname(oe[0].this)
        opi = None
        for pi, p in enumerate(tree.expressions):
            if wdb_sql._alias(p) == onm or wdb_sql._proj_colname(p) == onm:
                opi = pi
        if opi is None or not any(pi == opi for pi, _kd, _ac in aggs):
            return None                                 # order by a key: compose instead
        desc = bool(oe[0].args.get('desc'))
        rows.sort(key=lambda r: (r[opi] if r[opi] is not None else 0), reverse=desc)
    if lim is not None:
        rows = rows[:lim]
    final = []
    for r in rows:
        out = []
        for x in r:
            if isinstance(x, tuple) and x and x[0] in ('K1', 'K2'):
                nm, c = (knames[0], c1) if x[0] == 'K1' else (knames[1], c2)
                out.append(_val(nm, c, x[1]))
            else:
                out.append(x)
        final.append(tuple(out))
    hdr = [wdb_sql._alias(p) for p in tree.expressions]
    return final, hdr


def execute(db, tree):
    fused = _fused_2key(db, tree)
    if fused is not None:
        return fused
    proj = tree.expressions
    slots = []                                   # ('key', name) | ('agg', proj_index)
    key_names = []
    for pi, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Column):
            slots.append(('key', inner.name)); key_names.append(inner.name)
        elif wdb_sql._agg_kind(p) is not None:
            slots.append(('agg', pi))
        else:
            raise NotImplementedError("grouping-sets projection must be keys or aggregates")
    sets = _sets_of(tree)
    finest = max(sets, key=len)
    kinds = [wdb_sql._agg_kind(proj[pi])[0] for k, pi in slots if k == 'agg']
    rollable = (set(key_names) == set(finest)
                and all(s2 and set(s2) <= set(finest) or s2 == () for s2 in sets)
                and all(kd in ('COUNT_STAR', 'SUM', 'MIN', 'MAX') for kd in kinds)
                and tree.args.get('having') is None)
    if rollable:
        return _roll_from_finest(db, tree, proj, slots, key_names, sets, finest, kinds)
    all_rows = []
    for st in _sets_of(tree):
        sub = tree.copy()
        for k in ('order', 'limit', 'offset'):
            sub.set(k, None)
        keep = [p for p in proj
                if (wdb_sql._agg_kind(p) is not None)
                or ((p.this if isinstance(p, E.Alias) else p).name in st)]
        sub.set('expressions', [p.copy() for p in keep])
        if st:
            sub.set('group', E.Group(expressions=[E.column(n) for n in st]))
        else:
            sub.set('group', None)
        out = db.run(sub.sql())
        rows, hdr = out if isinstance(out, tuple) else (out, None)
        pos, cursor = [], 0
        for kind, ident in slots:
            if kind == 'key':
                pos.append(cursor if ident in st else None)
                cursor += 1 if ident in st else 0
            else:
                pos.append(cursor); cursor += 1
        for r in rows:
            all_rows.append(tuple(None if p is None else r[p] for p in pos))
    order = tree.args.get('order')
    if order is not None:
        names = [wdb_sql._alias(p) for p in proj]
        for oe in reversed(order.expressions):
            nm = oe.this.name if isinstance(oe.this, E.Column) else None
            if nm is None or nm not in names:
                raise NotImplementedError("grouping-sets ORDER BY must use output columns")
            idx = names.index(nm)
            all_rows.sort(key=lambda r: (r[idx] is None, r[idx]),
                          reverse=bool(oe.args.get('desc')))
    lim = wdb_sql._limit(tree)
    off = wdb_sql._offset(tree) or 0
    if lim is not None or off:
        all_rows = all_rows[off: None if lim is None else off + lim]
    return all_rows, [wdb_sql._alias(p) for p in proj]


def _roll_from_finest(db, tree, proj, slots, key_names, sets, finest, kinds):
    """One scan: the FINEST grouping runs through the engine; every coarser set derives from
    its cells in python (COUNT/SUM add, MIN/MAX fold) -- N sub-scans become one."""
    sub = tree.copy()
    for k in ('order', 'limit', 'offset'):
        sub.set(k, None)
    sub.set('group', E.Group(expressions=[E.column(n) for n in finest]))
    out = db.run(sub.sql())
    cells, _h = out if isinstance(out, tuple) else (out, None)
    kpos = {}
    cursor = 0
    for kind, ident in slots:
        kpos[ident if kind == 'key' else ('agg', ident)] = cursor
        cursor += 1
    all_rows = []
    agg_slots = [(pi, kd) for (k, pi), kd in
                 zip([sl for sl in slots if sl[0] == 'agg'], kinds)]
    for st in sets:
        if tuple(st) == tuple(finest):
            for r in cells:
                all_rows.append(tuple(r))
            continue
        acc = {}
        keep_idx = [kpos[n] for n in key_names if n in st]
        st_names = [n for n in key_names if n in st]
        for r in cells:
            key = tuple(r[i] for i in keep_idx)
            cur = acc.get(key)
            if cur is None:
                acc[key] = [r[kpos[('agg', pi)]] for pi, _kd in agg_slots]
            else:
                for ai, (pi, kd) in enumerate(agg_slots):
                    v = r[kpos[('agg', pi)]]
                    if kd in ('COUNT_STAR', 'SUM'):
                        cur[ai] = cur[ai] + v
                    elif kd == 'MIN':
                        cur[ai] = v if v < cur[ai] else cur[ai]
                    else:
                        cur[ai] = v if v > cur[ai] else cur[ai]
        for key, aggs in acc.items():
            row = []
            ki = ai = 0
            for kind, ident in slots:
                if kind == 'key':
                    row.append(key[st_names.index(ident)] if ident in st else None)
                else:
                    row.append(aggs[ai]); ai += 1
            all_rows.append(tuple(row))
    order = tree.args.get('order')
    if order is not None:
        names = [wdb_sql._alias(p) for p in proj]
        for oe in reversed(order.expressions):
            nm = oe.this.name if isinstance(oe.this, E.Column) else None
            if nm is None or nm not in names:
                raise NotImplementedError("grouping-sets ORDER BY must use output columns")
            idx = names.index(nm)
            all_rows.sort(key=lambda r: (r[idx] is None, r[idx]),
                          reverse=bool(oe.args.get('desc')))
    lim = wdb_sql._limit(tree)
    off = wdb_sql._offset(tree) or 0
    if lim is not None or off:
        all_rows = all_rows[off: None if lim is None else off + lim]
    return all_rows, [wdb_sql._alias(p) for p in proj]
