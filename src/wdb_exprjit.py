#!/usr/bin/env python3
"""
wdb_exprjit -- runtime code generation of fused decode+arithmetic aggregation kernels.

An arithmetic aggregate like SUM(l_extendedprice * (1 - l_discount)) was previously evaluated by
MATERIALISING the whole expression into a 6M-row float64 array (decode every column, run each arithmetic
op full-width, hand the finished array to the group kernel). Shannon says that array is almost all
redundant memory traffic: the discount column is 3.46 bits of real information smeared across 64-bit
slots, and the product adds no information beyond the two codes we already hold.

So instead we never build it. Given an expression as a numba-source body over slot variables v0,v1,...
(each slot = a dict column's (base, codes)), we generate ONE fused kernel that, per row, decodes only
what it needs (v_k = base_k[codes_k[i]]), evaluates the expression into a scalar, and folds it straight
into the grouped accumulator -- a single pass, no intermediate array. The kernel is compiled once per
distinct (body, #slots, gathered?, masked?, minmax?) shape and cached in-process; reuse is free.

This mirrors the hand-written decode-fused kernels in wdb_agg, generalised to arbitrary +,-,*,/ trees.
"""
import os, re
import numpy as np

try:
    from numba import njit as _njit, prange as _prange
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

_NT = min(8, os.cpu_count() or 1)
_PARALLEL_THRESHOLD = 2_000_000
_CACHE = {}                       # (body, slot_gathered, group_gathered, has_mask, need_minmax) -> kernel


def _build(body, slot_gathered, gk_gathered, nkeys, has_mask, need_minmax):
    """Generate + compile (or fetch from cache) the fused kernel for this expression shape.
    slot_gathered : tuple of bools per value slot (True = parent column via per-fact-row pointer).
    gk_gathered   : tuple of bools per GROUP key (True = parent key gathered through a pointer).
    The composite group code is computed inline (g = key0; g = g*r_j + key_j ...), so the mixed-radix
    composite array is never materialised -- one pass does compose-group + decode + expression + accumulate."""
    key = (body, slot_gathered, gk_gathered, nkeys, has_mask, need_minmax)
    fn = _CACHE.get(key)
    if fn is not None:
        return fn

    V = len(slot_gathered)
    params = ['n']
    for j in range(nkeys):
        params.append(f'gk{j}')
        if gk_gathered[j]: params.append(f'gp{j}')
    for j in range(1, nkeys):
        params.append(f'r{j}')                       # radix (K) of key j, applied left-to-right
    for k in range(V):
        params += [f'b{k}', f'c{k}']
        if slot_gathered[k]: params.append(f'p{k}')
    params += ['K', 'NT', 'Kp']
    if has_mask:
        params.append('mask')

    def gkexpr(j): return f'gk{j}[gp{j}[i]]' if gk_gathered[j] else f'gk{j}[i]'

    L = [f"def _k({', '.join(params)}):",
         "    s = np.zeros((NT, Kp)); cnt = np.zeros((NT, Kp), np.int64)"]
    if need_minmax:
        L += ["    mn = np.full((NT, Kp), np.inf); mx = np.full((NT, Kp), -np.inf)"]
    L += ["    chunk = (n + NT - 1) // NT",
          "    for t in _prange(NT):",
          "        lo = t * chunk; hi = min(lo + chunk, n)",
          "        for i in range(lo, hi):"]
    if has_mask:
        L += ["            if not mask[i]: continue"]
    for k in range(V):
        idx = f'c{k}[p{k}[i]]' if slot_gathered[k] else f'c{k}[i]'
        L += [f"            v{k} = b{k}[{idx}]"]
    if nkeys == 0:
        L += ["            g = 0"]
    else:
        L += [f"            g = {gkexpr(0)}"]
        for j in range(1, nkeys):
            L += [f"            g = g * r{j} + {gkexpr(j)}"]
    L += [f"            x = {body}",
          "            s[t, g] += x; cnt[t, g] += 1"]
    if need_minmax:
        L += ["            if x < mn[t, g]: mn[t, g] = x",
              "            if x > mx[t, g]: mx[t, g] = x"]
    L += ["    return (s, cnt, mn, mx)" if need_minmax else "    return (s, cnt)"]

    src = "\n".join(L)
    ns = {'np': np, '_prange': _prange}
    exec(src, ns)
    fn = _njit(parallel=True)(ns['_k'])      # no fastmath: match wdb_agg reduction semantics
    _CACHE[key] = fn
    return fn


def grouped_expr(group_keys, body, inputs, mask, n, need_minmax):
    """Run the fused expression aggregation in a single pass, materialising nothing.
      group_keys : list of (codes, K_i, ptr_or_None). Composed left-to-right into a mixed-radix group
                   code: g = key0; g = g*K1 + key1; ...  (ptr None = fact key codes[i], else codes[ptr[i]]).
      body       : numba-source scalar expression over value slots v0..v{V-1}.
      inputs     : list of (base, codes, ptr_or_None) per value slot (ptr None = direct fact column).
      Returns (count[K], sum[K], min[K] | None, max[K] | None) with K = product of the per-key K_i."""
    K = 1
    for (_c, ki, _p) in group_keys: K *= ki
    Kp = ((K + 7) // 8) * 8 + 8
    nkeys = len(group_keys)
    gk_gathered   = tuple(gk[2] is not None for gk in group_keys)
    slot_gathered = tuple(inp[2] is not None for inp in inputs)
    fn = _build(body, slot_gathered, gk_gathered, nkeys, mask is not None, need_minmax)

    args = [n]
    for (codes, _ki, ptr) in group_keys:
        args.append(np.ascontiguousarray(codes))
        if ptr is not None: args.append(np.ascontiguousarray(ptr))
    for j in range(1, nkeys):
        args.append(group_keys[j][1])                 # radix r_j
    for (base, codes, ptr) in inputs:
        args += [base, codes]
        if ptr is not None: args.append(np.ascontiguousarray(ptr))
    args += [K, _NT, Kp]
    if mask is not None:
        args.append(mask)

    res = fn(*args)
    if need_minmax:
        s, cnt, mn, mx = res
        return cnt[:, :K].sum(0), s[:, :K].sum(0), mn[:, :K].min(0), mx[:, :K].max(0)
    s, cnt = res
    return cnt[:, :K].sum(0), s[:, :K].sum(0), None, None


def _build_multi(bodies, mm_flags, slot_gathered, slot_code, gk_gathered, nkeys, has_mask, pred):
    """Compile a kernel that, in ONE pass, composes the group code inline, decodes the shared value slots
    once, and accumulates EVERY value expression (count + per-expr sum, and min/max where flagged).
    bodies: tuple of numba-source expressions over global slots v0..v{G-1}; mm_flags: per-expr need_minmax.
    pred: a boolean expression over the same slots (WHERE fused inline); '' for none. When present, the
    predicate's slots are decoded first and a failing row is skipped before any other slot is decoded."""
    key = (bodies, mm_flags, slot_gathered, slot_code, gk_gathered, nkeys, has_mask, pred)
    fn = _CACHE.get(key)
    if fn is not None:
        return fn

    G = len(slot_gathered); E = len(bodies)
    params = ['n']
    for j in range(nkeys):
        params.append(f'gk{j}')
        if gk_gathered[j]: params.append(f'gp{j}')
    for j in range(1, nkeys):
        params.append(f'r{j}')
    for k in range(G):
        if not slot_code[k]: params.append(f'b{k}')      # code slots carry no base array (raw codes)
        params.append(f'c{k}')
        if slot_gathered[k]: params.append(f'p{k}')
    params += ['K', 'NT', 'Kp']
    if has_mask:
        params.append('mask')

    def gkexpr(j): return f'gk{j}[gp{j}[i]]' if gk_gathered[j] else f'gk{j}[i]'

    L = [f"def _k({', '.join(params)}):",
         "    cnt = np.zeros((NT, Kp), np.int64)"]
    for e in range(E):
        L += [f"    s{e} = np.zeros((NT, Kp))"]
        if mm_flags[e]:
            L += [f"    mn{e} = np.full((NT, Kp), np.inf)", f"    mx{e} = np.full((NT, Kp), -np.inf)"]
    L += ["    chunk = (n + NT - 1) // NT",
          "    for t in _prange(NT):",
          "        lo = t * chunk; hi = min(lo + chunk, n)",
          "        for i in range(lo, hi):"]
    def _decode(k):
        idx = f'c{k}[p{k}[i]]' if slot_gathered[k] else f'c{k}[i]'
        return f"            v{k} = {idx}" if slot_code[k] else f"            v{k} = b{k}[{idx}]"
    pred_slots = sorted(set(int(x) for x in re.findall(r'v(\d+)', pred))) if pred else []
    done = set()
    if pred:                                    # decode only what the predicate needs, then skip early
        for k in pred_slots:
            L += [_decode(k)]; done.add(k)
        L += [f"            if not ({pred}): continue"]
    elif has_mask:
        L += ["            if not mask[i]: continue"]
    for k in range(G):
        if k not in done: L += [_decode(k)]
    if nkeys == 0:
        L += ["            g = 0"]
    else:
        L += [f"            g = {gkexpr(0)}"]
        for j in range(1, nkeys):
            L += [f"            g = g * r{j} + {gkexpr(j)}"]
    L += ["            cnt[t, g] += 1"]
    for e in range(E):
        L += [f"            x{e} = {bodies[e]}", f"            s{e}[t, g] += x{e}"]
        if mm_flags[e]:
            L += [f"            if x{e} < mn{e}[t, g]: mn{e}[t, g] = x{e}",
                  f"            if x{e} > mx{e}[t, g]: mx{e}[t, g] = x{e}"]
    ret = ["cnt"]
    for e in range(E):
        ret.append(f"s{e}")
        if mm_flags[e]: ret += [f"mn{e}", f"mx{e}"]
    L += [f"    return ({', '.join(ret)},)"]

    src = "\n".join(L)
    ns = {'np': np, '_prange': _prange}
    exec(src, ns)
    fn = _njit(parallel=True)(ns['_k'])
    _CACHE[key] = fn
    return fn


def grouped_multi(group_keys, inputs, exprs, mask, n, pred=None):
    """Single-pass fused aggregation of MANY value expressions sharing one composite group + decoded slots.
      group_keys : list of (codes, K_i, ptr_or_None)  -- composed inline into the group code.
      inputs     : global list of (base, codes, ptr_or_None) distinct value slots.
      exprs      : list of (body_over_global_slots, need_minmax).
      Returns (counts[K], [(sum_e[K], min_e|None, max_e|None) for each expr])."""
    K = 1
    for (_c, ki, _p) in group_keys: K *= ki
    Kp = ((K + 7) // 8) * 8 + 8
    nkeys = len(group_keys)
    gk_gathered   = tuple(gk[2] is not None for gk in group_keys)
    slot_gathered = tuple(inp[2] is not None for inp in inputs)
    slot_code     = tuple(inp[0] is None for inp in inputs)   # base None -> raw code slot (string equality)
    bodies   = tuple(e[0] for e in exprs)
    mm_flags = tuple(bool(e[1]) for e in exprs)
    has_mask = mask is not None and not pred          # fused predicate supersedes a materialised mask
    fn = _build_multi(bodies, mm_flags, slot_gathered, slot_code, gk_gathered, nkeys, has_mask, pred or '')

    args = [n]
    for (codes, _ki, ptr) in group_keys:
        args.append(np.ascontiguousarray(codes))
        if ptr is not None: args.append(np.ascontiguousarray(ptr))
    for j in range(1, nkeys):
        args.append(group_keys[j][1])
    for (base, codes, ptr) in inputs:
        if base is not None: args.append(base)            # code slots pass codes only
        args.append(codes)
        if ptr is not None: args.append(np.ascontiguousarray(ptr))
    args += [K, _NT, Kp]
    if has_mask: args.append(mask)

    res = fn(*args)
    counts = res[0][:, :K].sum(0)
    out = []; idx = 1
    for e in range(len(exprs)):
        s = res[idx][:, :K].sum(0); idx += 1
        if mm_flags[e]:
            mn = res[idx][:, :K].min(0); mx = res[idx + 1][:, :K].max(0); idx += 2
        else:
            mn = mx = None
        out.append((s, mn, mx))
    return counts, out
