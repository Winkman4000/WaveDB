"""
wdb_survgroup — survivor-set group-by for SELECTIVE filtered high-card single-key COUNT(*).

`SELECT key, COUNT(*) FROM t WHERE <pred> GROUP BY key ORDER BY COUNT(*) DESC LIMIT N` on a high-card
key is slow not because of the surviving rows but because the engine scatters them into a K-wide
(millions of entries) accumulator. When the filter is selective, far fewer rows survive than the
accumulator is wide, so it is much cheaper to materialize the survivors and group that small set.

The gate is COMPUTED at runtime, not guessed. The predicate has to be evaluated anyway (mode-4 filter
columns have no value-identity codes, so there is no cheaper way to know which rows pass), so we
popcount the resulting mask for the exact selectivity and take this path only when sel < gate(N, K).
Otherwise we return None and the dense path (which is cheaper once the survivor set approaches N)
handles it. Measured crossover (URL, K=18.3M): survivor-set wins up to ~30% selectivity, and the
crossover rises with K because a wider accumulator costs more to scatter and scan.

Sibling of wdb_gbcount (the filter-free count projection); reuses its by-code decoder and
clean-boundary top-N. Fail-closed on any shape it does not own.
"""
import numpy as np
import wdb_sql
import wdb_gbcount
import wdb_seqpred
import wdb_policies as P
import wdb_measure_runtime as RT
E = wdb_sql.E

_HITS = 0   # telemetry: queries answered from the survivor-set path

# The survivor-vs-dense crossover gate and the structural-pushdown threshold live in
# wdb_measure_runtime (the runtime-measures file): RT.survivor_gate(N, K) and
# RT.structural_pushdown_worth_it(nexc, N).

_CMP_OPS = {E.NEQ: wdb_seqpred.NEQ, E.EQ: wdb_seqpred.EQ, E.LT: wdb_seqpred.LT,
            E.LTE: wdb_seqpred.LTE, E.GT: wdb_seqpred.GT, E.GTE: wdb_seqpred.GTE}
_FLIP = {wdb_seqpred.LT: wdb_seqpred.GT, wdb_seqpred.GT: wdb_seqpred.LT,
         wdb_seqpred.LTE: wdb_seqpred.GTE, wdb_seqpred.GTE: wdb_seqpred.LTE,
         wdb_seqpred.EQ: wdb_seqpred.EQ, wdb_seqpred.NEQ: wdb_seqpred.NEQ}


def _int_lit(node):
    """Extract an int from a Literal / Neg(Literal), or None (strings/floats decline)."""
    if isinstance(node, E.Neg):
        v = _int_lit(node.this)
        return None if v is None else -v
    if isinstance(node, E.Literal) and not node.args.get('is_string'):
        try:
            s = str(node.this)
            return int(s) if s.lstrip('-').isdigit() else None
        except Exception:
            return None
    return None


def _simple_cmp(node, sc):
    """A bare `col <cmp> intconst` (either operand order) -> (phys_col, op, const), else None."""
    op = _CMP_OPS.get(type(node))
    if op is None:
        return None
    a, b = node.this, node.args.get('expression')
    if b is None:
        return None
    if isinstance(a, E.Column):
        lit = _int_lit(b)
        return None if lit is None else (sc(a.name), op, lit)
    if isinstance(b, E.Column):
        lit = _int_lit(a)
        return None if lit is None else (sc(b.name), _FLIP[op], lit)
    return None


def try_survgroup(seg, tree, col_map):
    """Answer a selective filtered high-card `key, COUNT(*) WHERE pred GROUP BY key ORDER BY COUNT(*)
    DESC LIMIT N` by grouping only the survivors, or return None to fall through to the dense path."""
    global _HITS
    # --- shared shape guards (wdb_policies); FILTERED family, so it REQUIRES a WHERE ---
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    if not P.has_where(tree):          return None      # filter-free -> wdb_gbcount / cube
    if not P.single_group_key(tree):   return None
    if not P.has_limit(tree):          return None
    where = tree.args.get('where')
    group = tree.args.get('group')
    lim = wdb_sql._limit(tree)
    proj = tree.expressions
    if len(proj) != 2:
        return None
    ci = wdb_gbcount._count_index(proj)
    if ci is None:
        return None
    ki = 1 - ci
    kp = proj[ki]
    if wdb_sql._agg_kind(kp) is not None:               # the other projection must be the bare key
        return None
    knm = wdb_sql._proj_colname(kp)
    gnm = wdb_sql._colname(group.expressions[0])
    if knm is None or gnm is None or knm != gnm:
        return None
    if not wdb_gbcount._order_is_count_desc(tree, proj, ci):
        return None
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    col = sc(knm)
    # --- shared segment/column guards (wdb_policies) ---
    if not P.columns_exist(seg, col):  return None
    if not P.not_positional(seg, col): return None
    V = wdb_gbcount._code_values(seg, col)              # by-code decoder for the group key
    if V is None:
        return None

    N = seg.N
    K = seg.cols[col].get('V') or (1 << 24)
    codes = seg._raw_codes(col)

    # Structural fast path: push `fcol <op> const` into the filter column's mode-4 exception
    # structure to get survivor row-ranges WITHOUT decoding the filter column, then gather only the
    # key codes there. Skipped when there are deleted rows (ranges don't model presence) or when the
    # column has too many exceptions to be worth it.
    sub = None
    cmp = _simple_cmp(where.this, sc)
    if cmp is not None and seg.presence_mask() is None:
        fcol, op, const = cmp
        nexc = wdb_seqpred.n_exceptions(seg, fcol)
        if RT.structural_pushdown_worth_it(nexc, N):
            r = wdb_seqpred.survivor_ranges(seg, fcol, op, const)
            if r is not None:
                los, his = r
                cnt = wdb_seqpred.survivor_count(los, his)
                if cnt == 0:
                    return None
                if cnt > RT.survivor_gate(N, K) * N:
                    return None                         # dense -> dense path is cheaper
                sub = codes[wdb_seqpred.ranges_to_ids(los, his)]

    if sub is None:
        # General path: evaluate the predicate to a mask (reads the filter column), then gather.
        # The eval is the sunk cost the dense path would pay too, and yields the exact selectivity.
        try:
            mask = wdb_sql._eval_pred(seg, where.this, sc)
        except (NotImplementedError, Exception):
            return None
        pm = seg.presence_mask()
        if pm is not None:
            mask = mask & pm
        cnt = int(mask.sum())
        if cnt == 0:
            return None                                 # empty result -> let the dense path format it
        if cnt > RT.survivor_gate(N, K) * N:
            return None                                 # dense filter -> dense path is cheaper
        sub = codes[mask]

    # Survivor-set group-by: group the small survivor set.
    u, c = np.unique(sub, return_counts=True)
    order = np.argsort(c, kind='stable')[::-1]          # count descending over distinct survivors
    cs = c[order]; us = u[order]
    if lim < cs.size and int(cs[lim - 1]) == int(cs[lim]):
        return None                                     # tie straddles the LIMIT boundary -> defer to
                                                        # the dense path for consistent tie-breaking
    take = min(lim, cs.size)
    rows = []
    for i in range(take):
        row = [None, None]
        row[ki] = wdb_sql._pyval(V[int(us[i])])         # decode only the N emitted keys
        row[ci] = int(cs[i])
        rows.append(tuple(row))
    rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))[:lim]
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in proj]
