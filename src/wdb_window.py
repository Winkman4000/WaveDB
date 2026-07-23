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
import wdb_kernels as K

_ENABLED = True
_HITS = 0
_MAX_FULL = 1_000_000

_FNS = {'RowNumber': 'row_number', 'Rank': 'rank', 'DenseRank': 'dense_rank',
        'Lag': 'lag', 'Lead': 'lead', 'Sum': 'sum', 'Count': 'count',
        'Avg': 'avg', 'Min': 'min', 'Max': 'max'}


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
    if kind in ('lag', 'lead', 'sum', 'avg', 'min', 'max'):
        a = fn.this
        if not isinstance(a, E.Column):
            return None
        arg_col = col_map.get(a.name, a.name) if col_map else a.name
    if kind == 'count':
        if not isinstance(fn.this, E.Star) and fn.this is not None:
            return None
    part = inner.args.get('partition_by') or []
    if not (1 <= len(part) <= 3) or not all(isinstance(x, E.Column) for x in part):
        return None
    pcols = tuple((col_map.get(x.name, x.name) if col_map else x.name) for x in part)
    order = inner.args.get('order')
    ocol = None; odesc = False
    if order is not None:
        if len(order.expressions) != 1:
            return None
        oe = order.expressions[0]
        odesc = bool(oe.args.get('desc'))
        oc = oe.this
        if not isinstance(oc, E.Column):
            return None
        ocol = col_map.get(oc.name, oc.name) if col_map else oc.name
    frame = None
    fs = inner.args.get('spec')
    if fs is not None:
        if (fs.args.get('kind') == 'ROWS' and fs.args.get('start_side') == 'PRECEDING'
                and str(fs.args.get('end')) == 'CURRENT ROW'
                and str(fs.args.get('start')).isdigit()):
            frame = ('rows', int(str(fs.args.get('start'))))
        else:
            return None                  # other frames: fallback's territory
    return {'kind': kind, 'arg': arg_col, 'pcols': pcols, 'ocol': ocol, 'odesc': odesc,
            'frame': frame, 'alias': wdb_sql._alias(p)}


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
    psets = {w['pcols'] for _pi, w in wins}
    osets = {(w['ocol'], w['odesc']) for _pi, w in wins}
    if len(psets) != 1 or len(osets) != 1:
        return None                      # one shared window spec (frames may differ per fn)
    pcols = next(iter(psets)); ocol, odesc = next(iter(osets))
    span = 1
    for pcn in pcols:
        if not P.columns_exist(seg, pcn) or seg._effective(pcn) is not None:
            return None
        if seg.cols[pcn].get('mode') not in (0, 1, 2):
            return None
        span *= int(seg.cols[pcn]['V'])
        if span > (1 << 62):
            return None
    if ocol is not None:
        if not P.columns_exist(seg, ocol):
            return None
        o_stair = seg.stairs(ocol) is not None
        o_dict = seg.cols[ocol].get('mode') in (0, 1, 2)
        if not o_stair and not o_dict:
            return None                  # order: cluster stair or a dict column
        oV = int(seg.cols[ocol]['V'])
        if span * oV > (1 << 62):
            return None
    for _pi, w in wins:
        if w['arg'] is not None:
            if not P.columns_exist(seg, w['arg']):
                return None
            if w['kind'] in ('sum', 'avg', 'min', 'max') and seg.cols[w['arg']].get('dt') != 0:
                return None
        if w['frame'] is not None and w['kind'] not in ('sum', 'avg', 'count', 'min', 'max'):
            return None
    qual = tree.args.get('qualify')
    qterms = None
    if qual is not None:
        qterms = []
        aliases = {w['alias'] for _pi, w in wins}
        import wdb_wherescan as WS
        _FLIP = {'>': '<', '>=': '<=', '<': '>', '<=': '>=', '=': '=', '<>': '<>'}
        for qc in WS._conjuncts(qual.this):
            tn = type(qc).__name__
            if tn not in WS._SCMP:
                return None
            a, b = qc.this, qc.expression
            if isinstance(a, E.Column) and a.name in aliases and isinstance(b, E.Literal):
                try:
                    qterms.append(('lit', a.name, WS._SCMP[tn], float(str(b.this))))
                except Exception:
                    return None
            elif isinstance(a, E.Column) and isinstance(b, E.Column) \
                    and b.name in aliases and a.name not in aliases:
                nm = col_map.get(a.name, a.name) if col_map else a.name
                if not P.columns_exist(seg, nm) or seg.cols[nm].get('dt') != 0:
                    return None
                qterms.append(('col', b.name, _FLIP[WS._SCMP[tn]], nm))   # alias <flip> col
            elif isinstance(b, E.Column) and isinstance(a, E.Column) \
                    and a.name in aliases and b.name not in aliases:
                nm = col_map.get(b.name, b.name) if col_map else b.name
                if not P.columns_exist(seg, nm) or seg.cols[nm].get('dt') != 0:
                    return None
                qterms.append(('col', a.name, WS._SCMP[tn], nm))          # alias <op> col
            elif isinstance(a, E.Column) and a.name not in aliases and isinstance(b, E.Literal):
                nm = col_map.get(a.name, a.name) if col_map else a.name   # plain col <op> lit
                if not P.columns_exist(seg, nm):
                    return None
                v = str(b.this)
                try:
                    v = float(v) if not b.is_string else v
                except ValueError:
                    pass
                qterms.append(('pcol', nm, WS._SCMP[tn], v))
            else:
                return None
    lim = wdb_sql._limit(tree)
    if qterms is None and (lim is None or lim > _MAX_FULL):
        return None                      # unbounded full-window output: fallback's territory
    return {'cols': cols, 'wins': wins, 'pcols': pcols, 'ocol': ocol, 'odesc': odesc,
            'qterms': qterms, 'lim': lim, 'off': int(wdb_sql._offset(tree) or 0),
            'proj': proj}


def _int_table(seg, col):
    """code -> int64 value, exact. A flat .nline sidecar memmaps directly when present --
    the fixed number line: file layout IS the compute layout, the OS page cache owns the
    bytes, zero per-query rebuild. The map survives drop_derived lawfully (it IS a file,
    same class as seg.buf). Else _dict_ints when the layout has it, typed-dict otherwise.
    Never float64 -- values past 2**53 lose low bits there (caught live: UserID)."""
    c = seg.cols[col]
    if '_nline' not in c:
        import os
        p = os.path.realpath(seg.path) + '.nline.' + col   # segments reach files through
        c['_nline'] = np.memmap(p, dtype='<i8', mode='r').view(np.ndarray) if os.path.exists(p) else None   # symlinks; ndarray view drops subclass gather tax
    if c['_nline'] is not None:
        return c['_nline']
    try:
        return np.asarray(seg._dict_ints(c), dtype=np.int64)
    except Exception:
        return np.array([int(v) for v in seg._typed_dict(col)], dtype=np.int64)


def _numvals(seg, col, exact_int=False):
    c = seg.cols[col]
    if c['mode'] == 4:
        arr = np.asarray(seg._seq_decode(c))
        return arr if exact_int and arr.dtype.kind in 'iu' else arr.astype(np.float64)
    codes = np.asarray(seg._raw_codes(col)).astype(np.int64)
    if exact_int:
        return _int_table(seg, col)[codes]
    import wdb_wherescan as WS
    return WS._num_table(seg, col)[codes]


def _order_codes_file(seg, ocol):
    """Per-row order codes in FILE order: stair columns via searchsorted, dicts via raw codes."""
    if seg.stairs(ocol) is not None:
        return np.searchsorted(seg.stairs(ocol), np.arange(int(seg.N)), side='right')
    return np.asarray(seg._raw_codes(ocol)).astype(np.int64)


def _try_stream_pairs(seg, spec):
    """Jackson's discovery-in-motion: the pair counter reports pairs AS IT FINDS them and
    stops at k -- a LIMIT-only LAG/LEAD dump reads a PREFIX of the disk, not the file.
    Sound because the walk IS the order (stair order column): the first two encounters of a
    partition code are the true first two members, so LAG at the second and LEAD at the
    first are known the instant the pair completes. Anything unknowable (LEAD at the
    second member) is simply never emitted."""
    if spec['qterms'] is not None or spec['lim'] is None or spec['off']:
        return None
    if len(spec['pcols']) != 1 or not spec['wins']:
        return None
    kinds = [w['kind'] for _pi, w in spec['wins']]
    if not all(k in ('lag', 'lead') for k in kinds):
        return None
    if any(w['frame'] is not None for _pi, w in spec['wins']):
        return None
    ocol, odesc = spec['ocol'], spec['odesc']
    if ocol is None or seg.stairs(ocol) is None or odesc:
        return None                              # the walk must BE the order
    pcol = spec['pcols'][0]
    c0 = seg.cols[pcol]
    if c0.get('mode') not in (0, 1, 2):
        return None
    V = int(c0['V'])
    if V > 60_000_000:
        return None
    lim = spec['lim']
    rows_per_pair = 1 if 'lead' in kinds else 2
    need = (lim + rows_per_pair - 1) // rows_per_pair
    N = int(seg.N)
    step = 1 << 18
    cap = min(N, max(step * 16, need * 256))
    first = np.full(V, -1, dtype=np.int64)
    done = np.zeros(V, dtype=bool)
    p1s, p2s = [], []
    lo = 0
    while lo < cap and len(p1s) < need:
        hi = min(lo + step, cap)
        blk = np.asarray(seg._raw_codes_range(pcol, lo, hi)).astype(np.int64)
        for i in range(blk.size):
            code = blk[i]
            if done[code]:
                continue
            pv = first[code]
            if pv < 0:
                first[code] = lo + i
            else:
                p1s.append(int(pv)); p2s.append(lo + i)
                done[code] = True
                if len(p1s) >= need:
                    break
        lo = hi
    if len(p1s) < need:
        return None                              # the prefix ran dry: full path serves
    if rows_per_pair == 2:
        pos = np.empty(2 * len(p1s), np.int64)
        pos[0::2] = p1s; pos[1::2] = p2s
        mate = np.empty(pos.size, np.int64)
        mate[0::2] = p2s; mate[1::2] = p1s
        isf = np.zeros(pos.size, bool); isf[0::2] = True
    else:
        pos = np.asarray(p1s, np.int64)
        mate = np.asarray(p2s, np.int64)
        isf = np.ones(pos.size, bool)
    pos, mate, isf = pos[:lim], mate[:lim], isf[:lim]
    def dec_at(nm, positions):
        cc = np.asarray(seg.codes_at(nm, positions)).astype(np.int64)
        return [wdb_sql._pyval(seg.fetch(nm, int(x))) for x in cc]
    slots = [None] * len(spec['proj'])
    for pi, nm in spec['cols']:
        c = seg.cols[nm]
        if c['mode'] == 4:
            slots[pi] = np.asarray(seg._seq_decode(c))[pos].tolist()
        else:
            slots[pi] = dec_at(nm, pos)
    for pi, w in spec['wins']:
        mv = dec_at(w['arg'], mate)
        if w['kind'] == 'lag':
            slots[pi] = [None if isf[i] else mv[i] for i in range(pos.size)]
        else:
            slots[pi] = [mv[i] if isf[i] else None for i in range(pos.size)]
    out = list(zip(*slots)) if slots else []
    global _HITS
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]


def _rn1_emit_col(seg, nm, cc):
    """Winner-code emission mirroring the generic slot rules: datetime64 C-speed for
    temporal, int-table (nline) for ints, point-fetch for small, sparse decode for big
    string sets. cc is an int64 code array."""
    c = seg.cols[nm]
    if cc.size <= 10000:
        return [wdb_sql._pyval(seg.fetch(nm, int(x))) for x in cc]
    if c.get('dt') == 3:
        vals = np.asarray(seg._dict_ints(c), dtype=np.int64)[cc]
        u = seg.unit(nm)                             # stored unit, not an assumption
        return vals.astype('datetime64[%s]' % u).astype('datetime64[us]').tolist()
    if c.get('dt') == 0 and c.get('mode') == 2:
        return _int_table(seg, nm)[cc].tolist()
    uq, inv = np.unique(cc, return_inverse=True)
    dvals = np.array([wdb_sql._pyval(seg.fetch(nm, int(k))) for k in uq], dtype=object)
    return dvals[inv].tolist()


_PC_MAX = 1 << 26      # composite board cap: 67M cells, a transient the pod yawns at


def _try_partcount(seg, spec):
    """COUNT(*) OVER (PARTITION BY a[, b]) projecting only the partition cols + n, with
    optional QUALIFY on n: the club-size question. One bean board over the composite
    key, one gather so every row looks up its own club, vectorized emission. No sort,
    no parade -- reuses the g-2key composite-board idea with a per-row lookup."""
    wins = spec['wins']
    if len(wins) != 1:
        return None
    w = wins[0][1]
    if w['kind'] != 'count' or w['ocol'] is not None or w['frame'] is not None:
        return None
    pcols = list(w['pcols'])
    if not (1 <= len(pcols) <= 2):
        return None
    alias = w['alias']
    qt = spec['qterms'] or []
    if qt and (len(qt) != 1 or qt[0][0] != 'lit' or qt[0][1] != alias):
        return None
    for nm in pcols:
        c = seg.cols[nm]
        if c.get('mode') not in (0, 1, 2) or c.get('has_null'):
            return None
    emit = []
    for p in spec['proj']:
        nm = wdb_sql._proj_colname(p)
        if nm in pcols:
            emit.append(pcols.index(nm))
        elif (isinstance(p, E.Column) and p.name == alias) or wdb_sql._alias(p) == alias:
            emit.append('n')
        else:
            return None
    Vs = [int(seg.cols[nm]['V']) for nm in pcols]
    tot = Vs[0] * (Vs[1] if len(Vs) == 2 else 1)
    if tot > _PC_MAX:
        return None
    cs = [np.asarray(seg._raw_codes(nm)).astype(np.int64) for nm in pcols]
    key = cs[0] if len(cs) == 1 else cs[0] * Vs[1] + cs[1]
    n = np.bincount(key, minlength=tot)[key]         # every row looks up its club
    if qt:
        op, val = qt[0][2], qt[0][3]
        m = (n > val) if op == '>' else (n >= val) if op == '>=' else \
            (n < val) if op == '<' else (n <= val) if op == '<=' else (n == val)
        sel = np.flatnonzero(m)
    else:
        sel = np.arange(key.size)
    if spec['lim'] is not None:
        sel = sel[spec['off']:spec['off'] + spec['lim']]
    slots = []
    for e in emit:
        if e == 'n':
            slots.append(n[sel].tolist())
        else:
            slots.append(_rn1_emit_col(seg, pcols[e], cs[e][sel]))
    global _HITS
    _HITS += 1
    return list(zip(*slots)), [wdb_sql._alias(p) for p in spec['proj']]


def _try_frame_keyed(seg, spec):
    """SUM/AVG/MIN/MAX OVER (PARTITION p ORDER o ROWS k PRECEDING..CURRENT) where the
    order column's stairs prove walk order == frame order: one compiled pass in RAW
    walk order carrying per-partition ring state. No permutation, no scatter-back,
    no order codes at all -- the stairs gate proves the order, values ride the walk.
    Tie behavior identical to the generic path (both preserve walk order in-lane)."""
    wins = spec['wins']
    if len(wins) != 1:
        return None
    w = wins[0][1]
    if w['kind'] not in ('sum', 'avg', 'min', 'max') or w['frame'] is None:
        return None
    fkind, k = w['frame']
    if fkind != 'rows' or not (0 <= k <= 64):
        return None
    if len(w['pcols']) != 1 or w['ocol'] is None or w['odesc']:
        return None
    pcol, ocol, acol = w['pcols'][0], w['ocol'], w['arg']
    if seg.stairs(ocol) is None:
        return None                                  # walk order must BE frame order
    cp, ca = seg.cols[pcol], seg.cols[acol]
    if cp.get('mode') not in (0, 1, 2) or cp.get('has_null'):
        return None
    if ca.get('dt') != 0 or ca.get('has_null'):
        return None
    if ca.get('mode') not in (0, 1, 2, 4):
        return None
    alias = w['alias']
    qt = spec['qterms'] or []
    if qt and (len(qt) != 1 or qt[0][0] != 'lit' or qt[0][1] != alias):
        return None
    emit = []
    for p in spec['proj']:
        nm = wdb_sql._proj_colname(p)
        if nm == pcol:
            emit.append('p')
        elif (isinstance(p, E.Column) and p.name == alias) or wdb_sql._alias(p) == alias:
            emit.append('w')
        else:
            return None
    import wdb_kernels as K
    pc = np.asarray(seg._raw_codes(pcol)).astype(np.int64)
    if ca['mode'] == 4:
        vals = np.asarray(seg._seq_decode(ca)).astype(np.int64)
    else:
        vals = _int_table(seg, acol)[np.asarray(seg._raw_codes(acol)).astype(np.int64)]
    Vp = int(cp['V'])
    if w['kind'] in ('sum', 'avg'):
        got = K.frame_sum_keyed(pc, vals, int(k), Vp)
        if got is None:
            return None
        fs, fc = got
        out = fs.astype(np.float64) / fc if w['kind'] == 'avg' else fs
    else:
        out = K.frame_ext_keyed(pc, vals, int(k), Vp, w['kind'] == 'max')
        if out is None:
            return None
    if qt:
        op, val = qt[0][2], qt[0][3]
        m = (out > val) if op == '>' else (out >= val) if op == '>=' else \
            (out < val) if op == '<' else (out <= val) if op == '<=' else (out == val)
        sel = np.flatnonzero(m)
    else:
        sel = np.arange(pc.size)
    if spec['lim'] is not None:
        sel = sel[spec['off']:spec['off'] + spec['lim']]
    slots = []
    for e in emit:
        if e == 'p':
            slots.append(_rn1_emit_col(seg, pcol, pc[sel]))
        else:
            slots.append(np.asarray(out)[sel].tolist())
    global _HITS
    _HITS += 1
    return list(zip(*slots)), [wdb_sql._alias(p) for p in spec['proj']]


def _try_rnk(seg, spec):
    """QUALIFY ROW_NUMBER() <= k over (PARTITION p ORDER o ASC) where o's stairs prove
    walk order == order: rn is just ARRIVAL POSITION in the cup, so one compiled pass
    with a counter per cup marks every winner with its rn. No permutation, no lanes,
    no order codes at all -- the searchsorted that computed them is deleted whole."""
    wins = spec['wins']
    if len(wins) != 1 or wins[0][1]['kind'] != 'row_number':
        return None
    w = wins[0][1]
    alias = w['alias']
    qt = spec['qterms']
    if (not qt or len(qt) != 1 or qt[0][0] != 'lit' or qt[0][1] != alias
            or qt[0][2] != '<='):
        return None
    kf = qt[0][3]
    if kf != int(kf) or not (1 <= int(kf) <= 16):
        return None
    k = int(kf)
    if len(w['pcols']) != 1 or w['ocol'] is None or w['odesc'] or w['frame'] is not None:
        return None
    pcol, ocol = w['pcols'][0], w['ocol']
    if seg.stairs(ocol) is None:
        return None                              # walk order must BE the window order
    cp, co = seg.cols[pcol], seg.cols[ocol]
    if cp.get('mode') not in (0, 1, 2) or cp.get('has_null') or co.get('has_null'):
        return None
    emit = []
    for p in spec['proj']:
        nm = wdb_sql._proj_colname(p)
        if nm == pcol:
            emit.append('p')
        elif nm == ocol:
            emit.append('o')
        elif (isinstance(p, E.Column) and p.name == alias) or wdb_sql._alias(p) == alias:
            emit.append('r')
        else:
            return None
    import wdb_kernels as K
    pc = np.asarray(seg._raw_codes(pcol)).astype(np.int64)
    rn = K.rnk_mark(pc, k, int(cp['V']))
    if rn is None:
        return None                              # no numba: fallback's territory
    sel = np.flatnonzero(rn)
    if spec['lim'] is not None:
        sel = sel[spec['off']:spec['off'] + spec['lim']]
    oc = None
    if 'o' in emit:
        oc = np.asarray(seg._raw_codes(ocol)).astype(np.int64)
    slots = []
    for e in emit:
        if e == 'p':
            slots.append(_rn1_emit_col(seg, pcol, pc[sel]))
        elif e == 'o':
            slots.append(_rn1_emit_col(seg, ocol, oc[sel]))
        else:
            slots.append(rn[sel].astype(np.int64).tolist())
    global _HITS
    _HITS += 1
    return list(zip(*slots)), [wdb_sql._alias(p) for p in spec['proj']]


def _try_rn1(seg, spec):
    """QUALIFY ROW_NUMBER()=1 over (PARTITION p ORDER o ASC|DESC), projecting only p, o,
    rn: the answer is the per-partition MINIMUM (or MAXIMUM) order code -- codes are
    ranks, so no sort, no permutation, no row motion. One compiled pass feeds a V-sized
    board of cups; winners emit through the vectorized slot rules. Only decode when
    needed, never more than needed."""
    wins = spec['wins']
    if len(wins) != 1 or wins[0][1]['kind'] != 'row_number':
        return None
    alias = wins[0][1]['alias']
    if spec['qterms'] != [('lit', alias, '=', 1.0)]:
        return None
    if len(spec['pcols']) != 1 or spec['ocol'] is None:
        return None
    pcol, ocol = spec['pcols'][0], spec['ocol']
    cp, co = seg.cols[pcol], seg.cols[ocol]
    if cp.get('mode') not in (0, 1, 2) or co.get('mode') not in (0, 1, 2):
        return None
    if cp.get('has_null') or co.get('has_null'):
        return None
    emit = []                                        # per projection: ('p'|'o'|'rn')
    for p in spec['proj']:
        nm = wdb_sql._proj_colname(p)
        if nm == pcol:
            emit.append('p')
        elif nm == ocol:
            emit.append('o')
        elif isinstance(p, E.Column) and p.name == alias or wdb_sql._alias(p) == alias:
            emit.append('rn')
        else:
            return None
    pc = np.asarray(seg._raw_codes(pcol)).astype(np.int64)
    oc = np.asarray(seg._raw_codes(ocol)).astype(np.int64)
    import wdb_kernels as K
    if spec['odesc']:
        acc = K.group_max(pc, oc, int(cp['V']))
        live = np.flatnonzero(acc >= 0)              # cups start at -1: filled == present
    else:
        acc = K.group_min(pc, oc, int(cp['V']))
        live = np.flatnonzero(acc != np.iinfo(np.int64).max)
    if spec['lim'] is not None:
        live = live[spec['off']:spec['off'] + spec['lim']]
    slots = []
    for kind in emit:
        if kind == 'p':
            slots.append(_rn1_emit_col(seg, pcol, live))
        elif kind == 'o':
            slots.append(_rn1_emit_col(seg, ocol, np.asarray(acc)[live]))
        else:
            slots.append([1] * int(live.size))
    global _HITS
    _HITS += 1
    return list(zip(*slots)), [wdb_sql._alias(p) for p in spec['proj']]


def execute(seg, spec):
    global _HITS
    st = _try_stream_pairs(seg, spec)
    if st is not None:
        return st
    fp = _try_rn1(seg, spec)
    if fp is not None:
        return fp
    fp = _try_rnk(seg, spec)
    if fp is not None:
        return fp
    fp = _try_partcount(seg, spec)
    if fp is not None:
        return fp
    fp = _try_frame_keyed(seg, spec)
    if fp is not None:
        return fp
    N = int(seg.N)
    # composite partition code (the radix fold)
    pc = np.asarray(seg._raw_codes(spec['pcols'][0])).astype(np.int64)
    span = int(seg.cols[spec['pcols'][0]]['V'])
    for pcn in spec['pcols'][1:]:
        v2 = int(seg.cols[pcn]['V'])
        pc = pc * v2 + np.asarray(seg._raw_codes(pcn)).astype(np.int64)
        span *= v2
    sub = None
    if spec['qterms'] is None and spec['lim'] is not None and span <= 60_000_000:
        # LIMIT-only dump: Jackson's pointer-count selection, refined. The pointer counts
        # per partition code are the family sizes (one bincount); the SMALLEST families
        # with pairs (size >= 2) cover the ask with the least motion -- and when the order
        # column is the stair, the walk itself IS the order, so encounter order is birth
        # order. Threshold via a histogram OF the sizes: no argsort over millions of
        # families, O(N) end to end. Complete families by code keep every answer correct.
        need = spec['off'] + spec['lim']
        if need * 4 < N:
            sizes = np.bincount(pc, minlength=span)
            hist = np.bincount(np.minimum(sizes, 1_000_000))
            contrib = np.arange(hist.size, dtype=np.int64) * hist
            contrib[:2] = 0                          # size-1 families: valid but last resort
            cumrows = np.cumsum(contrib)
            t = int(np.searchsorted(cumrows, need))
            chosen, got = [], 0
            for sz in range(2, min(t, hist.size - 1) + 1):
                if got >= need:
                    break
                cs = np.nonzero(sizes == sz)[0]
                take = min(cs.size, (need - got + sz - 1) // sz)
                chosen.append(cs[:take]); got += take * sz
            if got < need:                            # not enough pairs: size-1 fill, then big
                cs = np.nonzero(sizes == 1)[0][:need - got]
                chosen.append(cs); got += cs.size
            if got < need:
                cs = np.argsort(sizes)[::-1][:32]
                chosen.append(cs); got = int(sizes[cs].sum())
            if got >= need and chosen:
                keepmask = np.zeros(sizes.size, bool)
                keepmask[np.concatenate(chosen)] = True
                sub = np.nonzero(keepmask[pc])[0]
                pc = pc[sub]
                N = sub.size
    if sub is None:
        _rows = lambda a: a
        _load = lambda col, exact=False: _numvals(seg, col, exact_int=exact)
        _ocodes = lambda: _order_codes_file(seg, ocol)
    else:
        _rows = lambda a: a[sub]
        def _load(col, exact=False):                 # subset-THEN-gather: the translation
            c = seg.cols[col]                        # table meets only the chosen rows
            if c['mode'] == 4:
                arr = np.asarray(seg._seq_decode(c))[sub]
                return arr if exact and arr.dtype.kind in 'iu' else arr.astype(np.float64)
            codes = np.asarray(seg._raw_codes(col)).astype(np.int64)[sub]
            if exact:
                return _int_table(seg, col)[codes]
            import wdb_wherescan as WS
            return WS._num_table(seg, col)[codes]
        def _ocodes():
            if seg.stairs(ocol) is not None:
                return np.searchsorted(seg.stairs(ocol), sub, side='right')
            return np.asarray(seg._raw_codes(ocol)).astype(np.int64)[sub]
    ocol, odesc = spec['ocol'], spec['odesc']
    o_stair_asc = (ocol is not None and seg.stairs(ocol) is not None and not odesc)
    if ocol is None or o_stair_asc:
        # order inherited from the walk: pure partition scatter (compiled when span is small)
        perm = (K.part_scatter(pc, span) if span <= 50_000_000
                else np.argsort(pc, kind='stable'))
    else:
        # the TWO-AXIS fused motion: one composite integer places both dimensions at once
        ofc = _ocodes()
        oV = int(seg.cols[ocol]['V'])
        okey = (oV - 1 - ofc) if odesc else ofc
        perm = np.argsort(pc * oV + okey, kind='stable')
    ps = pc[perm]
    newlane = np.ones(N, bool); newlane[1:] = ps[1:] != ps[:-1]
    lane_start = np.nonzero(newlane)[0]
    lane_id_sorted = np.cumsum(newlane) - 1
    idx_in_lane = np.arange(N, dtype=np.int64) - lane_start[lane_id_sorted]
    # order-value tie groups (peers): SQL's default frame is RANGE -- peers share the
    # aggregate value of their whole tie group (evaluated at the group's END)
    tie_ends = None; tg = None
    if ocol is not None:
        oc_all = _ocodes()[perm]
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
            vals_s = (_load(w['arg'])[perm] if v is None
                      else _rows(v)[perm])
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
        elif k in ('sum', 'avg', 'count'):
            vv = _load(w['arg'])[perm] if w['arg'] is not None else None
            cs = np.cumsum(vv) if vv is not None else None
            if w['frame'] is not None:               # ROWS k PRECEDING .. CURRENT ROW
                fk = w['frame'][1]
                lo_i = np.maximum(np.arange(N, dtype=np.int64) - fk,
                                  lane_start[lane_id_sorted])
                nrow = (np.arange(N, dtype=np.int64) - lo_i + 1).astype(np.int64)
                if k == 'count':
                    out_s = nrow
                else:
                    cs0 = np.concatenate(([0.0], cs))
                    wsum = cs0[np.arange(1, N + 1)] - cs0[lo_i]
                    out_s = wsum if k == 'sum' else wsum / nrow
            elif spec['ocol'] is None:               # whole-partition aggregate
                if k == 'count':
                    out_s = lane_len[lane_id_sorted].astype(np.int64)
                else:
                    lane_tot = np.add.reduceat(vv, lane_start)
                    out_s = (lane_tot[lane_id_sorted] if k == 'sum'
                             else lane_tot[lane_id_sorted] / lane_len[lane_id_sorted])
            else:                                    # RANGE default: peers share group-end
                if k == 'count':
                    out_s = (tie_ends[tg] - lane_start[lane_id_sorted] + 1).astype(np.int64)
                else:
                    base = np.where(lane_start > 0, cs[lane_start - 1], 0.0)
                    rs = cs[tie_ends[tg]] - base[lane_id_sorted]
                    out_s = rs if k == 'sum' else \
                        rs / (tie_ends[tg] - lane_start[lane_id_sorted] + 1)
        else:                                        # min / max: running or sliding
            vv = _load(w['arg'], exact=True)[perm]
            if w['frame'] is not None:                   # ROWS k PRECEDING: monotonic deque
                fk = w['frame'][1]
                out_s = (K.seg_slidmin if k == 'min' else K.seg_slidmax)(vv, lane_start, fk)
            elif spec['ocol'] is None:
                seg_ext = (np.minimum if k == 'min' else np.maximum).reduceat(vv, lane_start)
                out_s = seg_ext[lane_id_sorted]
            else:
                run = (K.seg_cummin if k == 'min' else K.seg_cummax)(vv, lane_start)
                out_s = run[tie_ends[tg]]                # RANGE: group-end value for peers
        results[w['alias']] = out_s
    # selection: QUALIFY on window results (lane-space), else first off+lim rows
    if spec['qterms'] is not None:
        m = np.ones(N, bool)
        for term in spec['qterms']:
            al, op = term[1], term[2]
            if term[0] != 'pcol':
                arr = results[al]
                if hasattr(arr, 'dtype') and arr.dtype == object:
                    arr = np.array([float(x) if x is not None else np.nan for x in arr])
            if term[0] == 'lit':
                val = term[3]
            elif term[0] == 'pcol':
                nm, v = term[1], term[3]
                if isinstance(v, str):
                    import wdb_wherescan as WS2
                    fl = np.zeros(int(seg.cols[nm]['V']), bool)
                    code = WS2._code_of(seg, nm, v)
                    if code is not None:
                        fl[code] = True
                    cc = _rows(np.asarray(seg._raw_codes(nm)).astype(np.int64))[perm]
                    cmpv = fl[cc]
                    m &= (cmpv if op == '=' else ~cmpv)
                    continue
                arr2 = _load(nm)[perm]
                m &= (arr2 > v if op == '>' else arr2 >= v if op == '>=' else
                      arr2 < v if op == '<' else arr2 <= v if op == '<=' else
                      arr2 == v if op == '=' else arr2 != v)
                continue
            else:
                val = _load(term[3])[perm]   # col compare, lane-space
            m &= (arr > val if op == '>' else arr >= val if op == '>=' else
                  arr < val if op == '<' else arr <= val if op == '<=' else
                  arr == val if op == '=' else arr != val)
        sel_s = np.nonzero(m)[0]
    else:
        sel_s = np.arange(N)
    sel_file = perm[sel_s] if sub is None else sub[perm[sel_s]]   # true file positions
                                                 # lane-order emission: no outer ORDER BY,
                                                 # so no order is promised -- the reorder died
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
            if sel_file.size <= 10000:
                slots[pi] = [wdb_sql._pyval(seg.fetch(nm, int(x))) for x in cc]
            elif c.get('dt') == 3:
                secs = np.asarray(seg._dict_ints(c), dtype=np.int64)[cc]
                slots[pi] = secs.astype('datetime64[s]').tolist()    # C-speed datetime conversion
            elif c.get('dt') == 0:
                slots[pi] = _int_table(seg, nm)[cc].tolist()   # int64 end-to-end: float64
                                                               # mangles ints past 2**53
            elif int(c['V']) > 100_000 and sel_file.size < int(c['V']) // 4:
                # big string dict, sparse selection: decode only the codes that APPEAR
                uq, inv = np.unique(cc, return_inverse=True)
                dvals = np.array([wdb_sql._pyval(seg.fetch(nm, int(k))) for k in uq], dtype=object)
                slots[pi] = dvals[inv].tolist()
            else:
                dv = seg._typed_dict(nm)
                full = np.array([wdb_sql._pyval(x) for x in dv], dtype=object)
                slots[pi] = full[cc].tolist()
    for pi, w in spec['wins']:
        arr = results[w['alias']]
        sel = arr[sel_s]
        slots[pi] = sel.tolist() if hasattr(sel, 'tolist') else list(sel)
    out = list(zip(*slots)) if slots else []
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
