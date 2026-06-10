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
import os, pickle
import numpy as np
import wdb_sql
import sqlglot.expressions as E

CUBE_MAX_CELLS = 4096          # prod(dim cardinalities) cap; above this storage cost outweighs the win.
                               # Measured: a 2,526-cell l_shipdate cube is ~114 KB and answers GROUP BY date
                               # in 0.37ms vs DuckDB 12.4ms (33x/worker). 4096 keeps that in, stays tiny.
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


def write_cubes(seg_path, cubes):
    pickle.dump(list(cubes), open(seg_path + '.cube', 'wb'), protocol=4)


def load_cubes(seg_path):
    p = seg_path + '.cube'
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


def build_and_write(seg, specs, max_cells=CUBE_MAX_CELLS):
    """Build every spec that passes the cap and persist the surviving cubes next to the segment."""
    cubes = []
    for dims in specs:
        c = build_cube(seg, dims, max_cells)
        if c is not None: cubes.append(c)
    if cubes: write_cubes(seg.path, cubes)
    return cubes


def try_cube(seg, tree, col_map):
    """If `tree` is a filter-free GROUP BY whose dims match a stored cube and whose projections are
    all COUNT(*) / SUM / AVG over stored measures (or group-key columns), answer from the cube and
    return (rows, colnames). Otherwise return None so the caller falls through to the scan paths."""
    if tree.args.get('where') is not None or tree.args.get('joins'): return None
    if tree.args.get('distinct') is not None: return None
    group = tree.args.get('group')
    if group is None: return None
    cubes = seg.cubes()
    if not cubes: return None
    gcols = []
    for g in group.expressions:
        nm = wdb_sql._colname(g)
        if nm is None: return None
        gcols.append(col_map.get(nm, nm) if col_map else nm)
    gset = set(gcols)
    cube = next((c for c in cubes if set(c['dims']) == gset), None)
    if cube is None: return None
    proj = tree.expressions
    getters = []
    for p in proj:
        ak = wdb_sql._agg_kind(p)
        if ak is None:                                      # bare group-key column
            nm = wdb_sql._colname(p)
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
    if having is not None:
        rows = wdb_sql._apply_having(rows, proj, having.this, None)
    rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))
    lim = wdb_sql._limit(tree)
    if lim is not None: rows = rows[:lim]
    global _CUBE_HITS; _CUBE_HITS += 1
    return rows, [wdb_sql._alias(p) for p in proj]
