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
_NOLIM_MAX = 4096                    # without a LIMIT, at most this many groups (one point-fetch each)
_BULK = [__import__('os').environ.get('WDB_GS_BULK', '1') != '0']   # past it: the bulk path
_LAZY_TEXT_V = 1 << 18            # a text dictionary past this decodes per fetched batch (the cursor)


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
    # NO LIMIT IS A SHAPE TOO (2026-10-01): Q07 (AdvEngineID <> 0, every group, ORDER BY COUNT(*) DESC)
    # fell to the fused cascade -- a 100M-row mask and a row walk, 78-88 ms hot -- for 18 bins.
    # Without a LIMIT every live bin is emitted (execute declines past _NOLIM_MAX of them)
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
    if c.get('mode') not in (0, 1, 2):
        return None                                      # dictionary columns (mode 2: int dicts, 2026-10-18)
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
    if c.get('mode') == 2 and not terms:
        return None                                      # a plain int-dictionary key: dict_count / affinegroup
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
    if wdb_sql._offset(tree):
        return None
    return {'col': col, 'ci': ci, 'ki': ki, 'lim': (int(lim) if lim is not None else None), 'terms': terms,
            'having': hv, 'proj': proj, 'order': order, 'lower': lower}


def execute(seg, spec):
    """THE READ: one walk feeds the board; everything after happens on V squares."""
    global _HITS
    col = spec['col']
    c = seg.cols[col]
    V = int(c['V'])
    # THE CENSUS FROM THE DRESS (2026-10-01): was np.bincount -- one thread -- over the decoded codes
    cn = np.asarray(seg.raw_census(col, V), np.int64)[:V].copy()
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
    nlive = int(np.count_nonzero(live))     # counted, not listed: a near-unique key has V live squares
    hdrs = [wdb_sql._alias(p) for p in spec['proj']]
    if nlive == 0:
        _HITS += 1
        return [], hdrs
    if spec['lim'] is None and nlive > _NOLIM_MAX:
        if (rep is not None or not _BULK[0] or spec['ki'] not in (0, 1) or spec['ci'] != 1 - spec['ki']
                or not _bulk_order_ok(spec)):   # ORDER BY the count alone, or then the key ascending: our order
            return None              # every group, one point-fetch each: the scan path's job
        return _bulk_rows(seg, col, c, np.where(live, cn, 0), nlive, spec, hdrs)
    k = nlive if spec['lim'] is None else min(spec['lim'], nlive)
    if nlive > k and len(spec['order'].expressions) == 1:
        # THE CANONICAL TOP (2026-10-18): ORDER BY the count alone -- a tie at the LIMIT is SQL's to leave
        # unordered, and declining it sent WatchID-like keys (every count 1) to a 4.7 s scan. Count desc, then code
        # asc, by topk_bar over the live board without sorting it (dead squares zeroed sit below every live count)
        import wdb_kernels
        board = np.where(live, cn, 0)
        sel = np.asarray(wdb_kernels.topk_bar(board, k), np.int64)
    elif nlive > k:
        # more ORDER BY terms: the pick must follow them -- a tie straddling the LIMIT declines (the scan decides)
        idx = np.flatnonzero(live)
        cnl = cn[idx]
        part = np.argpartition(-cnl, k - 1)
        top = part[:k]
        rest_max = int(cnl[part[k:]].max())
        top = top[np.argsort(-cnl[top], kind='stable')]
        if int(cnl[top[-1]]) == rest_max:
            return None
        sel = idx[top]
    else:
        idx = np.flatnonzero(live)
        cnl = cn[idx]
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


def _bulk_values(seg, col, c, sel):
    """The values of codes `sel` in bulk, as the per-group fetch would give them (wdb_sql._pyval of seg.fetch): ints and
    floats from the typed dictionary, times through datetime64 of the column's unit (a day -> date, finer -> datetime),
    text decoded from bytes. None: not a shape handled here (the caller declines)."""
    if c.get('mode') not in (0, 1, 2) or seg._effective(col) is not None:
        return None
    dt = c.get('dt')
    td = seg._typed_dict(col)
    if dt in (0, 2):
        if c.get('aux') == 9:
            return None
        return np.asarray(td)[sel].tolist()
    if dt == 3:
        unit = seg.unit(col)
        a = np.asarray(td, np.int64)[sel].astype('datetime64[%s]' % unit)
        return a.tolist() if unit == 'D' else a.astype('datetime64[us]').tolist()
    if dt == 1:
        out = []
        for v in (td[sel] if isinstance(td, np.ndarray) else [td[i] for i in sel.tolist()]):
            if isinstance(v, (bytes, bytearray)):
                try:
                    v = v.decode('utf-8', 'surrogatepass')
                except Exception:
                    pass
            out.append(v)
        return out
    return None


def _bulk_rows(seg, col, c, board, nlive, spec, hdrs):
    """THE COUNTING PATH WITHOUT A CAP (2026-10-22, Q7's board): GROUP BY K ORDER BY COUNT(*) with no LIMIT and more than
    _NOLIM_MAX groups. It declined (one point-fetch per group), and the shape fell to wherescan's row-level grouping:
    ResponseEndTiming's 28K groups 0.68 s (DuckDB 0.077), LocalEventTime / ClientEventTime / UserID / RefererHash past
    10 s. Here the board is ordered by count in two linear passes (wdb_kernels.count_desc_order: count descending, code
    ascending -- the small path's argsort order) and the winners' values come from the dictionary in bulk. The rows are
    built in their final order, so finalize's re-sort is skipped. The working-set guard still names a result too big to
    hold as Python rows (WatchID's 100M groups). WDB_GS_BULK=0 declines as before."""
    global _HITS
    import wdb_govern, wdb_kernels, wdb_result
    cols_ok = wdb_result.WANT[0] and type(seg).__name__ == 'Segment'
    if not cols_ok:
        wdb_govern.ask(nlive, 2, 'grouped result')      # Python rows: the working-set guard names a result too big
    mx = int(board.max())
    if mx > 16 * board.size:
        # one huge count against few groups: the buckets would outnumber the squares -- sort the live squares instead
        # (count descending, code ascending: the same order)
        idx = np.flatnonzero(board > 0)
        sel = idx[np.argsort(-board[idx], kind='stable')]
    else:
        sel = wdb_kernels.count_desc_order(board)
    if cols_ok:
        # THE CURSOR'S ANSWER (2026-10-22): columns, Python values built per fetched batch (wdb_result)
        out = _columns(seg, col, c, sel, board[sel], spec, hdrs)
        if out is not None:
            _HITS += 1
            return out, hdrs
        wdb_govern.ask(nlive, 2, 'grouped result')
    vals = _bulk_values(seg, col, c, sel)
    if vals is None:
        return None
    cnt = board[sel].tolist()
    rows = list(zip(vals, cnt)) if spec['ki'] == 0 else list(zip(cnt, vals))
    _HITS += 1
    return rows, hdrs


def _columns(seg, col, c, sel, cnt_arr, spec, hdrs):
    """the bulk path's answer in columns (wdb_result.Columns), for db.execute: numbers and times as arrays (their
    Python values built per fetched batch), text decoded from the dictionary per fetched batch -- a 20M-group text
    answer decodes the rows a caller fetches, not 20M strings. None: not a shape handled here."""
    import wdb_result as WR
    if c.get('mode') not in (0, 1, 2) or seg._effective(col) is not None or c.get('aux') == 9:
        return None
    dt = c.get('dt')
    if dt in (0, 2):
        kcol = WR.ArrayColumn(np.asarray(seg._typed_dict(col))[sel])
    elif dt == 3:
        unit = seg.unit(col)
        a = np.asarray(seg._typed_dict(col), np.int64)[sel].astype('datetime64[%s]' % unit)
        kcol = WR.ArrayColumn(a, py=(lambda s: s.tolist()) if unit == 'D' else (lambda s: s.astype('datetime64[us]').tolist()))
    elif dt == 1 and int(c['V']) <= _LAZY_TEXT_V:
        # a small dictionary decodes whole faster than a batch of point fetches (SocialSourcePage, 63K values: 0.15 s
        # through point fetches, 0.04 s whole) -- in the query, as the rows path does
        vals = _bulk_values(seg, col, c, sel)
        if vals is None:
            return None
        arr = np.empty(len(vals), dtype=object)
        arr[:] = vals
        kcol = WR.ArrayColumn(arr, py=list)
    elif dt == 1:
        def _dec(lo, hi, _sel=sel):
            out = []
            for g in _sel[lo:hi].tolist():
                out.append(wdb_sql._pyval(seg.fetch(col, int(g))))
            return out
        kcol = WR.LazyColumn(int(sel.size), _dec)
    else:
        return None
    ccol = WR.ArrayColumn(cnt_arr)
    cols = [kcol, ccol] if spec['ki'] == 0 else [ccol, kcol]
    return WR.Columns(hdrs, cols)


def _bulk_order_ok(spec):
    """the bulk path's order is count descending, then code ascending -- the key's value ascending, the dictionary being
    value-sorted. It serves ORDER BY the count alone (ties are SQL's to leave) and ORDER BY the count, then the key
    ascending (the board's tie-break form: Title's had fallen to the row path and timed out)."""
    ex = spec['order'].expressions
    if len(ex) == 1:
        return True
    if len(ex) != 2:
        return False
    o = ex[1]
    if not isinstance(o, E.Ordered) or o.args.get('desc'):
        return False
    t = o.this
    kp = spec['proj'][spec['ki']]
    kname = wdb_sql._alias(kp)
    inner = kp.this if isinstance(kp, E.Alias) else kp
    if isinstance(t, E.Column) and (t.name == kname or (isinstance(inner, E.Column) and t.name == inner.name)):
        return True
    return False
