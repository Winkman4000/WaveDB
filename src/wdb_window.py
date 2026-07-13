"""wdb_window: window functions on the FUSED MOTION -- one placement, two coordinates.

The architecture is the child-language design: walking the file once, each row is placed into
its partition's lane; which lane = the group, position down the lane = the rank. Because the
file is cluster-ordered (EventTime), order within every lane is INHERITED from the walk -- no
comparison sort ever happens. The placement is one stable integer argsort of partition codes
(counting-sort machinery); everything else is vectorized lane arithmetic on the permutation:
ROW_NUMBER is position-minus-lane-start, LAG/LEAD are lane neighbors, running SUM/COUNT are
segmented cumsums with lane-boundary resets.

v1 shapes: plain columns + window functions sharing one window spec, PARTITION BY one dict
column, ORDER BY absent or the cluster stair column ASC. QUALIFY filters on window results
(the top-k-per-partition idiom: QUALIFY rn <= 3 selects lane heads -- tiny output, no world
materialization). Full-output shapes require LIMIT. Everything else declines to the fallback.

Functions: ROW_NUMBER, RANK/DENSE_RANK (over the order column's values), LAG, LEAD,
running SUM, running COUNT(*).
"""
import numpy as np
import sqlglot.expressions as E
import wdb_sql
import wdb_policies as P

_ENABLED = True
_HITS = 0
_MAX_FULL = 1_000_000

_FNS = {'RowNumber': 'row_number', 'Rank': 'rank', 'DenseRank': 'dense_rank',
        'Lag': 'lag', 'Lead': 'lead', 'Sum': 'sum', 'Count': 'count'}


def enable():
    global _ENABLED; _ENABLED = True


def disable():
    global _ENABLED; _ENABLED = False


def _parse_window(seg, p, col_map):
    inner = p.this if isinstance(p, E.Alias) else p
    if not isinstance(inner, E.Window):
        return None
    fn = inner.this
    kind = _FNS.get(type(fn).__name__)
    if kind is None:
        return None
    arg_col = None
    if kind in ('lag', 'lead', 'sum'):
        a = fn.this
        if not isinstance(a, E.Column):
            return None
        arg_col = col_map.get(a.name, a.name) if col_map else a.name
    if kind == 'count':
        if not isinstance(fn.this, E.Star) and fn.this is not None:
            return None
    part = inner.args.get('partition_by') or []
    if len(part) != 1 or not isinstance(part[0], E.Column):
        return None
    pcol = col_map.get(part[0].name, part[0].name) if col_map else part[0].name
    order = inner.args.get('order')
    ocol = None
    if order is not None:
        if len(order.expressions) != 1 or order.expressions[0].args.get('desc'):
            return None
        oc = order.expressions[0].this
        if not isinstance(oc, E.Column):
            return None
        ocol = col_map.get(oc.name, oc.name) if col_map else oc.name
    return {'kind': kind, 'arg': arg_col, 'pcol': pcol, 'ocol': ocol,
            'alias': wdb_sql._alias(p)}


def detect(seg, tree, col_map):
    if not _ENABLED:                    return None
    if not P.no_joins(tree):            return None
    if not P.no_deleted_rows(seg):      return None
    if tree.args.get('where') is not None or tree.args.get('group') is not None:
        return None
    proj = tree.expressions
    cols, wins = [], []
    for pi, p in enumerate(proj):
        w = _parse_window(seg, p, col_map)
        if w is not None:
            wins.append((pi, w)); continue
        inner = p.this if isinstance(p, E.Alias) else p
        if not isinstance(inner, E.Column):
            return None
        nm = col_map.get(inner.name, inner.name) if col_map else inner.name
        if not P.columns_exist(seg, nm):
            return None
        cols.append((pi, nm))
    if not wins:
        return None
    pcols = {w['pcol'] for _pi, w in wins}
    ocols = {w['ocol'] for _pi, w in wins}
    if len(pcols) != 1 or len(ocols) != 1:
        return None                      # v1: one shared window spec
    pcol = next(iter(pcols)); ocol = next(iter(ocols))
    if not P.columns_exist(seg, pcol) or seg._effective(pcol) is not None:
        return None
    if seg.cols[pcol].get('mode') not in (0, 1, 2):
        return None
    if ocol is not None and seg.stairs(ocol) is None:
        return None                      # v1 order: the cluster stair (or file order)
    for _pi, w in wins:
        if w['arg'] is not None:
            if not P.columns_exist(seg, w['arg']):
                return None
            if w['kind'] == 'sum' and seg.cols[w['arg']].get('dt') != 0:
                return None
    qual = tree.args.get('qualify')
    qterms = None
    if qual is not None:
        qterms = []
        aliases = {w['alias'] for _pi, w in wins}
        import wdb_wherescan as WS
        for qc in WS._conjuncts(qual.this):
            tn = type(qc).__name__
            if tn not in WS._SCMP or not isinstance(qc.this, E.Column):
                return None
            if qc.this.name not in aliases or not isinstance(qc.expression, E.Literal):
                return None
            try:
                qterms.append((qc.this.name, WS._SCMP[tn], float(str(qc.expression.this))))
            except Exception:
                return None
    lim = wdb_sql._limit(tree)
    if qterms is None and (lim is None or lim > _MAX_FULL):
        return None                      # unbounded full-window output: fallback's territory
    return {'cols': cols, 'wins': wins, 'pcol': pcol, 'ocol': ocol,
            'qterms': qterms, 'lim': lim, 'off': int(wdb_sql._offset(tree) or 0),
            'proj': proj}


def _numvals(seg, col):
    c = seg.cols[col]
    if c['mode'] == 4:
        return np.asarray(seg._seq_decode(c), dtype=np.float64)
    import wdb_wherescan as WS
    return WS._num_table(seg, col)[np.asarray(seg._raw_codes(col)).astype(np.int64)]


def execute(seg, spec):
    global _HITS
    N = int(seg.N)
    pc = np.asarray(seg._raw_codes(spec['pcol'])).astype(np.int64)
    perm = np.argsort(pc, kind='stable')         # THE FUSED MOTION: one placement, two axes
    ps = pc[perm]
    newlane = np.ones(N, bool); newlane[1:] = ps[1:] != ps[:-1]
    lane_start = np.nonzero(newlane)[0]
    lane_id_sorted = np.cumsum(newlane) - 1
    idx_in_lane = np.arange(N, dtype=np.int64) - lane_start[lane_id_sorted]
    # order-value tie groups (peers): SQL's default frame is RANGE -- peers share the
    # aggregate value of their whole tie group (evaluated at the group's END)
    tie_ends = None; tg = None
    if spec['ocol'] is not None:
        oc_all = np.searchsorted(seg.stairs(spec['ocol']), perm, side='right')
        tie_new = newlane.copy()
        tie_new[1:] |= oc_all[1:] != oc_all[:-1]
        tg = np.cumsum(tie_new) - 1
        st = np.nonzero(tie_new)[0]
        tie_ends = np.append(st[1:] - 1, N - 1)
    lane_len = np.diff(np.append(lane_start, N))
    results = {}
    for _pi, w in spec['wins']:
        k = w['kind']
        if k == 'row_number':
            out_s = idx_in_lane + 1
        elif k in ('rank', 'dense_rank'):
            if spec['ocol'] is None:
                out_s = idx_in_lane + 1
            elif k == 'dense_rank':
                out_s = tg - tg[lane_start[lane_id_sorted]] + 1
            else:
                st2 = np.nonzero(np.r_[True, tg[1:] != tg[:-1]])[0]
                out_s = st2[tg] - lane_start[lane_id_sorted] + 1
        elif k in ('lag', 'lead'):
            v = np.asarray(seg._raw_codes(w['arg'])).astype(np.int64) \
                if seg.cols[w['arg']]['mode'] != 4 else None
            vals_s = (_numvals(seg, w['arg'])[perm] if v is None
                      else v[perm])
            shifted = np.empty(N, dtype=object)
            if k == 'lag':
                shifted[1:] = vals_s[:-1]; shifted[0] = None
                shifted[newlane] = None
            else:
                shifted[:-1] = vals_s[1:]; shifted[-1] = None
                last = np.zeros(N, bool); last[:-1] = newlane[1:]; last[-1] = True
                shifted[last] = None
            if v is not None:
                dec = shifted.copy()
                for i in np.nonzero(shifted != None)[0]:
                    dec[i] = wdb_sql._pyval(seg.fetch(w['arg'], int(shifted[i])))
                out_s = dec
            else:
                out_s = shifted
        elif k == 'sum':
            vv = _numvals(seg, w['arg'])[perm]
            cs = np.cumsum(vv)
            base = np.where(lane_start > 0, cs[lane_start - 1], 0.0)
            if spec['ocol'] is None:
                lane_tot = np.add.reduceat(vv, lane_start)
                out_s = lane_tot[lane_id_sorted]         # no ORDER: whole-partition aggregate
            else:
                out_s = cs[tie_ends[tg]] - base[lane_id_sorted]   # RANGE: peers share group-end
        else:                                    # count
            if spec['ocol'] is None:
                out_s = lane_len[lane_id_sorted].astype(np.int64)
            else:
                out_s = (tie_ends[tg] - lane_start[lane_id_sorted] + 1).astype(np.int64)
        results[w['alias']] = out_s
    # selection: QUALIFY on window results (lane-space), else first off+lim rows
    if spec['qterms'] is not None:
        m = np.ones(N, bool)
        for al, op, val in spec['qterms']:
            arr = results[al]
            if arr.dtype == object:
                arr = np.array([float(x) if x is not None else np.nan for x in arr])
            m &= (arr > val if op == '>' else arr >= val if op == '>=' else
                  arr < val if op == '<' else arr <= val if op == '<=' else
                  arr == val if op == '=' else arr != val)
        sel_s = np.nonzero(m)[0]
    else:
        sel_s = np.arange(N)
    sel_file = perm[sel_s]
    o = np.argsort(sel_file, kind='stable')      # emit in file (cluster) order
    sel_s, sel_file = sel_s[o], sel_file[o]
    if spec['lim'] is not None:
        sel_s = sel_s[spec['off']: spec['off'] + spec['lim']]
        sel_file = sel_file[spec['off']: spec['off'] + spec['lim']]
    # emission: one column list per projection slot, assembled by zip (C-speed) --
    # per-row python loops died here at 28.6M output rows
    slots = [None] * len(spec['proj'])
    for pi, nm in spec['cols']:
        c = seg.cols[nm]
        if c['mode'] == 4:
            slots[pi] = np.asarray(seg._seq_decode(c))[sel_file].tolist()
        else:
            cc = np.asarray(seg.codes_at(nm, sel_file)).astype(np.int64)
            if sel_file.size > 10000:
                dv = seg._typed_dict(nm)
                dvals = np.array([wdb_sql._pyval(x) for x in dv], dtype=object)
                slots[pi] = dvals[cc].tolist()           # one bulk gather, never per-fetch
            else:
                slots[pi] = [wdb_sql._pyval(seg.fetch(nm, int(x))) for x in cc]
    for pi, w in spec['wins']:
        arr = results[w['alias']]
        sel = arr[sel_s]
        slots[pi] = sel.tolist() if hasattr(sel, 'tolist') else list(sel)
    out = list(zip(*slots)) if slots else []
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
