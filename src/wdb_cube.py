#!/usr/bin/env python3
"""
wdb_cube -- materialised low-cardinality GROUP BY cube (a precomputed aggregate).

For a filter-free GROUP BY over low-card dimensions, the entire answer is a handful of rows:
per occurring cell we store [count, Sum(each non-null float measure)]. A matching query then
reads those few numbers instead of scanning the value columns -- so it is parse-bound, not
bandwidth-bound (the only lever that beats the ~50GB/s memory wall is reading fewer bytes, and
this reads almost none). It is a materialised view: a DIFFERENT class than a faster scan, and it
ONLY applies to filter-free low-card group-bys with COUNT/SUM/AVG over stored measures. Anything
else -- a WHERE, a high-card dimension, MIN/MAX, a nullable/unstored measure -- falls back to the
slice2 / grouped_multi scan with identical results.

Worth-it gate (build time, on the columns' own measured cardinality): build only when the cell
count B = prod(dim cardinalities) <= CUBE_MAX_CELLS. Below that the cube is a few KB and the answer
is parse-bound (huge throughput win); far above it the cube costs MB for a query that must emit ~B
rows anyway (output-bound), so it is not worth the permanent storage. 1024 sits safely in the win
zone and captures the genuinely low-card categoricals (and their pairs); above it we decline.
"""
import os, pickle, collections
import numpy as np
import wdb_sql
import workers
import sqlglot.expressions as E
import wdb_policies as P
from wdb_profile import cardinality, segment_cardinalities  # data-measures: cardinality lives in one place
from wdb_measure_runtime import CUBE_MAX_CELLS  # runtime-measures: the cube cell-count cap

# CUBE_MAX_CELLS (the cube cell-count cap) lives in wdb_measure_runtime, imported above.
_CUBE_HITS = 0                 # diagnostic: how many queries were answered from a materialised cube


def _radix(seg, dims, max_cells=CUBE_MAX_CELLS):
    """(per-row composite code, [K per dim], [value array per dim]) or None if prod(K) > cap.

    Dense per-dim codes come from factorising each dim's MEASURED values -- not seg.codes() -- so the
    cube is built off the distinct values actually present and is independent of the column's storage
    mode (dict mode 0 OR positional mode 4). seg.codes() returns row-position codes for mode-4 columns
    (e.g. small-N datetime), which would spuriously look high-card; factorising the values is the honest
    self-measured cardinality. NA (null) factorises to its own code so a null group is preserved."""
    import pandas as pd
    codes = []; Ks = []; vals_list = []
    for d in dims:
        vals = np.asarray(wdb_sql._col(seg, d)[0])
        c, uniq = pd.factorize(vals, sort=True)                 # dense 0..K-1, NA -> -1
        c = c.astype(np.int64, copy=False)
        if (c < 0).any(): c = c + 1; K = len(uniq) + 1          # lift NA into code 0 (its own group)
        else: K = len(uniq)
        codes.append(c); Ks.append(K); vals_list.append(vals)
    prod = 1
    for K in Ks: prod *= K
    if prod == 0 or prod > max_cells: return None
    comp = np.zeros(seg.N, dtype=np.int64)
    for c, K in zip(codes, Ks):
        comp = comp * K + c
    return comp, Ks, vals_list


def build_cube(seg, dims, max_cells=CUBE_MAX_CELLS):
    """Freeze [count, Sum each non-null float measure] per occurring cell of GROUP BY `dims`.
    Returns a cube dict, or None if the grouping is too high-card to be worth materialising."""
    r = _radix(seg, dims, max_cells)
    if r is None: return None
    comp, Ks, dim_arrs = r
    uniq, first_idx, inv = np.unique(comp, return_index=True, return_inverse=True)
    B = int(len(uniq))
    count = np.bincount(inv, minlength=B).astype(np.int64)
    # decode each cell's group-key value from the executor's own per-row decode at a representative
    # row of the cell (dim_arrs come from _radix) -- identical to what a scan would emit.
    # datetime dims are stored as int64 epochs; emit them exactly as the scan path does
    # (epoch -> datetime64[unit] -> _pyval) so cube rows are byte-identical to a full scan.
    dim_units = [seg.unit(d) if seg.cols[d].get('dt') == 3 else None for d in dims]
    def _keyval(i, idx):
        v = dim_arrs[i][idx]
        if dim_units[i] is not None: v = np.int64(v).view(f'datetime64[{dim_units[i]}]')
        return wdb_sql._pyval(v)
    keys = [tuple(_keyval(i, first_idx[b]) for i in range(len(dims))) for b in range(B)]
    sums = {}
    for nm, meta in seg.cols.items():
        if meta.get('dt') == 2 and not meta.get('has_null'):       # non-null float measure only
            w = np.asarray(seg.resident_values(nm)).astype(np.float64, copy=False)
            sums[nm] = np.bincount(inv, weights=w, minlength=B).astype(np.float64)
    return {'dims': list(dims), 'B': B, 'keys': keys, 'count': count, 'sums': sums}


def _derive_cube(parent, sub_dims):
    """Roll a stored cube up to a subset of its dims by summing out the rest. Exact: count and every
    measure-sum are additive, so the result is value-identical to build_cube(seg, sub_dims) (cell order
    aside, which no caller depends on). O(parent cells) -- microseconds for a <=4096-cell parent."""
    pidx = [parent['dims'].index(d) for d in sub_dims]
    pkeys = parent['keys']; pcount = parent['count']; psums = parent['sums']
    pos = {}; order = []; cnt = []; sacc = {m: [] for m in psums}
    for b in range(len(pkeys)):
        k = pkeys[b]; sk = tuple(k[i] for i in pidx)
        j = pos.get(sk)
        if j is None:
            j = len(order); pos[sk] = j; order.append(sk); cnt.append(0)
            for m in sacc: sacc[m].append(0.0)
        cnt[j] += int(pcount[b])
        for m in psums: sacc[m][j] += float(psums[m][b])
    return {'dims': list(sub_dims), 'B': len(order), 'keys': order,
            'count': np.asarray(cnt, dtype=np.int64),
            'sums': {m: np.asarray(sacc[m], dtype=np.float64) for m in psums}}


def _get_cube(cubes, needed):
    """Return the smallest cube that answers GROUP BY `needed`: an exact stored cube if present, else
    the dims rolled down from the smallest stored superset (derived once and cached into `cubes`, so
    later queries of the same shape hit it directly). None if no stored cube covers `needed`.
    This is what lets disk persist only the maximal (non-redundant) cubes while every sub-cube the
    operator asks for still arrives ready-made."""
    needed = set(needed)
    for c in cubes:
        if set(c['dims']) == needed: return c
    cands = [c for c in cubes if needed <= set(c['dims'])]
    if not cands: return None
    parent = min(cands, key=lambda c: c['B'])
    derived = _derive_cube(parent, [d for d in parent['dims'] if d in needed])
    cubes.append(derived)                              # cache for this segment instance (RAM only)
    return derived


def write_cubes(seg_path, cubes):
    pickle.dump(list(cubes), open(seg_path + '.cube', 'wb'), protocol=4)


def load_cubes(seg_path):
    p = seg_path + '.cube'
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


def _build_specs_worker(args):
    """Process-pool worker: reopen the segment (cheap memmap) and build a chunk of specs."""
    path, specs, max_cells = args
    from wdb_engine import Segment
    seg = Segment(path)
    return [c for c in (build_cube(seg, d, max_cells) for d in specs) if c is not None]


def build_and_write(seg, specs, max_cells=CUBE_MAX_CELLS, workers=1):
    """Build every spec that passes the cap and persist the surviving cubes next to the segment.
    workers>1 fans the builds across processes (each reopens the segment) -- the per-cube cost is
    factorize + bincount over N rows, so the exhaustive 'auto' set parallelises well across cores."""
    specs = list(specs)
    if workers and workers > 1 and len(specs) > 1:
        import concurrent.futures as cf
        nw = min(workers, len(specs))
        chunks = [specs[i::nw] for i in range(nw)]          # round-robin: mix cube sizes per worker
        cubes = []
        import multiprocessing as _mp9
        with cf.ProcessPoolExecutor(max_workers=nw, mp_context=_mp9.get_context('spawn')) as ex:
            for part in ex.map(_build_specs_worker, [(seg.path, ch, max_cells) for ch in chunks]):
                cubes.extend(part)
    else:
        cubes = [c for c in (build_cube(seg, d, max_cells) for d in specs) if c is not None]
    if cubes: write_cubes(seg.path, cubes)
    return cubes


# segment_cardinalities / cardinality moved to wdb_profile (the data-measures file); imported above.

def low_card_stamp_cols(parent_seg, cap=CUBE_MAX_CELLS):
    """The general denormalisation rule, self-measured: a parent column is worth stamping onto a child
    iff it is low enough cardinality to live in a cube (2 <= distinct <= cap). High-card parent columns
    (keys, free-text, near-unique numerics) are left alone -- the join gather already wins on those.
    Pure self-measurement, workload-agnostic: no query input. Returns a list of parent column names."""
    cards = segment_cardinalities(parent_seg)
    return [c for c in parent_seg.order if 2 <= cards.get(c, 0) <= cap]


def enumerate_cube_specs(cards, cap=CUBE_MAX_CELLS, max_arity=None):
    """Every non-empty column-subset whose cardinality product <= cap -- the exhaustive set of cubes
    the data allows. `cards` is {col: distinct_count}, taken from each column's stored V (a pure
    self-measurement: no workload input). Branch-and-bound on cards ascending -- once prod*card
    exceeds the cap, every higher-card column in the branch also overflows, so we prune the tail.
    Constant columns (card < 2) are skipped (a 1-value dim never narrows anything). Returns a list
    of dim-lists, smallest-product first."""
    cols = sorted((c for c in cards if cards[c] >= 2), key=lambda c: cards[c])
    n = len(cols); out = []
    lim = n if max_arity is None else max_arity
    def dfs(start, cur, prod):
        if len(cur) >= lim: return
        for i in range(start, n):
            p = prod * cards[cols[i]]
            if p > cap: break                       # ascending: no later (>=) col fits either
            cur.append(cols[i]); out.append((p, list(cur)))
            dfs(i + 1, cur, p)
            cur.pop()
    dfs(0, [], 1)
    out.sort(key=lambda t: t[0])                    # cheapest / most broadly useful cubes first
    return [dims for _p, dims in out]


def maximal_cube_specs(cards, cap=CUBE_MAX_CELLS):
    """The non-redundant cubes to actually persist: every spec to which no further column can be added
    without exceeding the cap. Each spec dropped from the full enumeration is a strict subset of one of
    these, so it is recovered exactly by rolling a maximal cube down (see _get_cube/_derive_cube) --
    full generality, a fraction of the storage and build cost."""
    cset = {c: v for c, v in cards.items() if v >= 2}
    out = []
    for s in enumerate_cube_specs(cards, cap):
        p = 1
        for c in s: p *= cset[c]
        if all((c in s) or (p * v > cap) for c, v in cset.items()):   # cannot extend by any column
            out.append(s)
    return out


def _grouped_cdist_from_cube(tree, col_map, cubes):
    """SELECT col1, COUNT(DISTINCT col2) ... GROUP BY col1  answered from a [col1,col2] cube.
    Every stored cell is one (col1,col2) pair that actually occurred, so COUNT(DISTINCT col2) for a
    given col1 is just the number of stored cells with that col1 (skipping null col2, which SQL's
    COUNT(DISTINCT) ignores; a col1 whose col2 is always null still emits a row with count 0).

    The gate is the cube's existence, which is itself a measurement: a [col1,col2] cube exists only if
    its cell count was under the cap at build time. A high-card col2 (e.g. partkey) blows the cap, no
    cube is built, this returns None, and the query falls back to the scan -- so 'too big' never takes
    the cube path. Returns (rows, colnames) or None."""
    group = tree.args.get('group')
    if group is None or len(group.expressions) != 1: return None
    g0 = wdb_sql._colname(group.expressions[0])
    if g0 is None: return None
    col1 = col_map.get(g0, g0) if col_map else g0
    proj = tree.expressions
    cd_col = None
    for p in proj:
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Count) and isinstance(inner.this, E.Distinct):
            dx = inner.this.expressions
            if len(dx) != 1 or not isinstance(dx[0], E.Column): return None
            nm = wdb_sql._colname(dx[0])
            cd_col = col_map.get(nm, nm) if col_map else nm
        else:                                               # only the bare group key may sit alongside
            nm = wdb_sql._colname(p)
            if nm is None: return None
            if (col_map.get(nm, nm) if col_map else nm) != col1: return None
    if cd_col is None or cd_col == col1: return None
    cube = _get_cube(cubes, {col1, cd_col})                 # exact or derived [col1,col2] cube
    if cube is None: return None
    i1 = cube['dims'].index(col1); i2 = cube['dims'].index(cd_col)
    cnt = collections.Counter(); allv1 = []
    seen = set()
    for k in cube['keys']:
        v1 = k[i1]
        if v1 not in seen: seen.add(v1); allv1.append(v1)
        if k[i2] is not None: cnt[v1] += 1
    rows = []
    for v1 in allv1:
        dc = cnt.get(v1, 0)
        row = []
        for p in proj:
            inner = p.this if isinstance(p, E.Alias) else p
            row.append(int(dc) if (isinstance(inner, E.Count) and isinstance(inner.this, E.Distinct)) else v1)
        rows.append(tuple(row))
    having = tree.args.get('having')
    rows = workers.finalize(rows, proj, tree.args.get('order'), wdb_sql._limit(tree),
                            having=having.this if having is not None else None)
    return rows, [wdb_sql._alias(p) for p in proj]


def _const(x):
    """A literal/constant expr -> python scalar, or None if it is not a constant."""
    if isinstance(x, E.Neg):
        v = _const(x.this)
        return None if v is None else -v
    if isinstance(x, E.Literal):
        if x.is_string: return x.this
        s = x.this
        try: return int(s)
        except ValueError: return float(s)
    if isinstance(x, E.Boolean):
        return bool(x.this)
    return None


def _make_pred(expr):
    """A single WHERE predicate on one column -> (colname, fn(value)->bool), or None.
    fn treats NULL (None) as failing, matching SQL three-valued logic (NULL <op> const is never TRUE)."""
    import operator
    cmp = {E.GT: operator.gt, E.GTE: operator.ge, E.LT: operator.lt,
           E.LTE: operator.le, E.EQ: operator.eq, E.NEQ: operator.ne}
    t = type(expr)
    if t in cmp:
        col, val = expr.this, expr.expression
        if not isinstance(col, E.Column): return None
        c = _const(val)
        if c is None: return None
        f = cmp[t]
        return wdb_sql._colname(col), (lambda v, f=f, c=c: v is not None and f(v, c))
    if isinstance(expr, E.Between):
        col = expr.this
        lo = _const(expr.args.get('low')); hi = _const(expr.args.get('high'))
        if not isinstance(col, E.Column) or lo is None or hi is None: return None
        return wdb_sql._colname(col), (lambda v, lo=lo, hi=hi: v is not None and lo <= v <= hi)
    if isinstance(expr, E.In):
        col = expr.this
        if not isinstance(col, E.Column): return None
        vals = [_const(x) for x in expr.expressions]
        if not vals or any(x is None for x in vals): return None
        s = set(vals)
        return wdb_sql._colname(col), (lambda v, s=s: v is not None and v in s)
    return None


def _range_filter_from_cube(tree, col_map, cubes):
    """SELECT g..., COUNT(*)/SUM/AVG  FROM t  WHERE fcol <op> const  GROUP BY g...
    answered from a cube whose dims are exactly {group cols} + {fcol}.

    The filter column is a stored cube DIMENSION, so every row inside a cell shares fcol's value: the
    predicate selects whole cells exactly, with no straddling residual. Roll the surviving cells up by
    the group sub-key. EXACT (not approximate) precisely because fcol is per-value in the cube.

    Gate = the cube's existence (a self-measurement): a {group, fcol} cube exists only if its cell count
    cleared the cap at build time, so a high-card fcol (no such cube) returns None -> scan fallback.
    Ops: > >= < <= = != BETWEEN IN. Single predicate only (AND/OR -> None). Returns (rows, cols) or None."""
    mp = (lambda nm: col_map.get(nm, nm)) if col_map else (lambda nm: nm)
    group = tree.args.get('group')
    if group is None: return None
    pred = _make_pred(tree.args.get('where').this)
    if pred is None: return None
    fcol_raw, fn = pred
    if fcol_raw is None: return None
    fcol = mp(fcol_raw)
    gcols = []
    for g in group.expressions:
        nm = wdb_sql._colname(g)
        if nm is None: return None
        gcols.append(mp(nm))
    if fcol in gcols: return None                          # filter col must be the non-grouped dim
    cube = _get_cube(cubes, set(gcols) | {fcol})
    if cube is None: return None
    proj = tree.expressions
    getters = []; need = set()
    for p in proj:
        ak = wdb_sql._agg_kind(p)
        if ak is None:                                     # bare group-key column
            nm = wdb_sql._colname(p)
            if nm is None: return None
            pc = mp(nm)
            if pc not in gcols: return None
            getters.append(('key', gcols.index(pc)))
        elif ak[0] == 'COUNT_STAR':
            getters.append(('count',))
        elif ak[0] == 'SUM':
            pc = mp(ak[1])
            if pc not in cube['sums']: return None
            getters.append(('sum', pc)); need.add(pc)
        elif ak[0] == 'AVG':
            pc = mp(ak[1])
            if pc not in cube['sums']: return None
            getters.append(('avg', pc)); need.add(pc)
        else:                                              # COUNT(col) / MIN / MAX -> fall back
            return None
    keys = cube['keys']; count = cube['count']; sums = cube['sums']
    gi = [cube['dims'].index(gc) for gc in gcols]
    fi = cube['dims'].index(fcol)
    acc_c = collections.defaultdict(int)
    acc_s = {m: collections.defaultdict(float) for m in need}
    order = []; seen = set()
    for b, k in enumerate(keys):
        if not fn(k[fi]): continue
        gk = tuple(k[i] for i in gi)
        if gk not in seen: seen.add(gk); order.append(gk)
        acc_c[gk] += int(count[b])
        for m in need: acc_s[m][gk] += float(sums[m][b])
    rows = []
    for gk in order:
        row = []
        for gt in getters:
            if gt[0] == 'key': row.append(gk[gt[1]])
            elif gt[0] == 'count': row.append(int(acc_c[gk]))
            elif gt[0] == 'sum': row.append(float(acc_s[gt[1]][gk]))
            else:
                c = acc_c[gk]; row.append(float(acc_s[gt[1]][gk]) / c if c else 0.0)
        rows.append(tuple(row))
    having = tree.args.get('having')
    rows = workers.finalize(rows, proj, tree.args.get('order'), wdb_sql._limit(tree),
                            having=having.this if having is not None else None)
    return rows, [wdb_sql._alias(p) for p in proj]


def detect(seg, tree, col_map):
    """ACTIVATION (lean): cheap static shape guards + does a cube exist at all. The
    fine-grained 'do the dims/measures match a stored cube' check must inspect the cube,
    so it stays in execute (which declines if nothing matches). Returns a spec or None."""
    # --- shared shape guards (wdb_policies) ---
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.has_group_key(tree):      return None
    cubes = seg.cubes()
    if not cubes: return None
    return {'tree': tree, 'col_map': col_map, 'cubes': cubes}


def execute(seg, spec):
    """THE READ: answer a filter-free GROUP BY from a stored cube -- a range-filter roll-up,
    a grouped COUNT(DISTINCT) from a 2-dim cube, or the main GROUP BY of COUNT(*)/SUM/AVG.
    Declines (None) when the query's dims/measures don't match any stored cube."""
    global _CUBE_HITS
    tree = spec['tree']; col_map = spec['col_map']; cubes = spec['cubes']
    group = tree.args.get('group')
    if tree.args.get('where') is not None:                  # WHERE fcol <op> const on a cube dim -> exact roll-up
        rf = _range_filter_from_cube(tree, col_map, cubes)
        if rf is not None:
            _CUBE_HITS += 1
            return rf
        return None                                         # a WHERE we cannot answer from a cube: scan handles it
    cd = _grouped_cdist_from_cube(tree, col_map, cubes)     # grouped COUNT(DISTINCT) via a [col1,col2] cube
    if cd is not None:
        _CUBE_HITS += 1
        return cd
    gcols = []
    for g in group.expressions:
        nm = wdb_sql._colname(g)
        if nm is None: return None
        gcols.append(col_map.get(nm, nm) if col_map else nm)
    gset = set(gcols)
    cube = _get_cube(cubes, gset)
    if cube is None: return None
    proj = tree.expressions
    getters = []
    for p in proj:
        ak = wdb_sql._agg_kind(p)
        if ak is None:                                      # bare group-key column (unwrap any alias)
            nm = wdb_sql._proj_colname(p)
            if nm is None: return None
            pc = col_map.get(nm, nm) if col_map else nm
            if pc not in cube['dims']: return None
            getters.append(('key', cube['dims'].index(pc)))
        elif ak[0] == 'COUNT_STAR':
            getters.append(('count',))
        elif ak[0] == 'SUM':
            pc = col_map.get(ak[1], ak[1]) if col_map else ak[1]
            if pc not in cube['sums']: return None
            getters.append(('sum', pc))
        elif ak[0] == 'AVG':
            pc = col_map.get(ak[1], ak[1]) if col_map else ak[1]
            if pc not in cube['sums']: return None
            getters.append(('avg', pc))
        else:                                               # COUNT(col) / MIN / MAX -> fall back
            return None
    keys = cube['keys']; count = cube['count']; sums = cube['sums']; B = cube['B']
    col_lists = []
    for gt in getters:
        if gt[0] == 'key':
            di = gt[1]; col_lists.append([k[di] for k in keys])
        elif gt[0] == 'count':
            col_lists.append([int(x) for x in count])
        elif gt[0] == 'sum':
            col_lists.append([float(x) for x in sums[gt[1]]])
        else:
            s = sums[gt[1]]; col_lists.append([float(s[i]) / int(count[i]) for i in range(B)])
    rows = list(zip(*col_lists)) if col_lists else [() for _ in range(B)]
    having = tree.args.get('having')
    rows = workers.finalize(rows, proj, tree.args.get('order'), wdb_sql._limit(tree),
                            having=having.this if having is not None else None)
    _CUBE_HITS += 1
    return rows, [wdb_sql._alias(p) for p in proj]


def try_cube(seg, tree, col_map):
    """Detect + execute, kept as the backward-compatible single-call entry."""
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)
