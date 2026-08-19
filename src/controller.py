"""
controller.py -- THE CONTROLLER (routing layer).

A query, in the user sense, is a *route*: the controller takes a request and sends
it down the read that fires for it (and, later, a worker). The controller holds no
retrieval logic of its own -- it owns the ORDER of reads and walks it:

    for read in _READ_ORDER:
        spec = read.detect(ctx)        # ACTIVATION -- the auditable, preloadable part
        if spec is None: continue      #   not this read's shape -> next
        rows = read.execute(ctx, spec) # THE READ -- may still decline (measured) -> next
        if rows is not None: return rows
    return general_scan(ctx)           # catch-all: read the columns directly

Every read -- structure reads, clustered slices, the throughput bit-sliced filter,
the fused fast path -- is a uniform Read(detect, execute) in read_methods. The
exception-based reads catch their own "not my shape" signal inside execute; bsi's
throughput-only gate lives in its detect. Byte-identical to the old hand-ordered
try-chain that lived inline in wdb_db.run().

WHERE THIS IS HEADED: bind detect results from system state ONCE when materialization
settles (the preloaded "try state"), and hang a WORKER off each route.
"""
import read_methods as R
import wdb_sql


def _agg_or_group(tree):
    """Precondition for the structure-read chain: the request must aggregate, group,
    or ask for distinct. Plain projections skip straight to the scan."""
    return (tree.args.get('group') is not None
            or tree.args.get('distinct') is not None
            or any(wdb_sql._agg_kind(e) for e in tree.expressions))
def _has_window(tree):
    import sqlglot.expressions as E
    return any(isinstance((p.this if isinstance(p, E.Alias) else p), E.Window)
               for p in tree.expressions)



# The single-segment read order -- the "try state". Each is a read_methods.Read; the
# first whose detect fires AND whose execute returns rows wins.
_READ_ORDER = (
    R.cube,                 # pre-materialized cube
    R.blockstats,           # whole-table aggregates from per-block stats: no row data touched
    R.gbcount,              # full-table group census from the gbc shelf
    R.stair,                # staircase column: single-key GROUP BY from step positions (no decode)
    R.regexgroup,           # GROUP BY regex over the dict: per-code counts, V-level strings
    R.window,               # window fns: stable scatter by partition, lanes inherit cluster order
    R.dict_count,           # dictionary per-code counts (sidecar: answers before any scan)
    R.groupself,            # counting board: GROUP BY K + WHERE on K, bins not rows
    R.coscan,               # fused conjunctive COUNT: zone-map veto, one walk
    R.grid2,               # plain 2-key COUNT grid: one fused pass, narrow detect
    R.firstk,               # staircase early-exit: LIKE + ORDER BY stair LIMIT k, pops only the answer window
    R.funnel,               # selective funnel: plist start, crumb hygiene, code-space group
    R.wherescan,            # conjunctive WHERE: stair spans + blocked-frame predicate scan, disk-only
    R.pairfold,
    R.affinegroup,
    R.affinesum,
    R.mixtop,
    R.pairdistinct,
    R.septop,
    R.sampletop,
    R.tripletop,
    R.pairtop,
    R.gridwalk,
    R.smallk,              # narrow-key pair/triple boards (the small-K weapon)             # two-key COUNT(*) top-K via grid filled-cell + count-ordered head (opt-in)
    R.sumtopk,             # single big-key SUM top-K: one kernel, k label decodes
    R.distinctlim,         # DISTINCT-LIMIT early exit: stops at n pairs, streams never fully read
    R.diskpair,             # disk-only pair GROUP BY: scan-merge floor beneath the structures
    R.countpos,             # two-key COUNT(*) top-K via per-row count-class presence-scan (opt-in)
    R.heavypair,            # two-key COUNT(*) top-N from a count-sorted pair sidecar
    R.scanpair,             # two-key COUNT(*) top-N, high-card non-key filter via code-space scan
    R.survivor_group,       # filtered high-card group via survivor ranges
    R.compound_filter,      # multi-condition AND filter
    R.distinct_sidecar,     # prebuilt group->distinct sidecar
    R.group_distinct,       # one-pass code-hash distinct walk
    R.group_mix,            # group + foldables + one distinct, one walk
    R.cluster_slice,        # contiguous slice of a clustered structure
    R.cluster_group_slice,  # clustered group-runs as slices
    R.bsi_filter,           # bit-sliced index filter (throughput-only; detect gates on esc)
    R.pairagg,              # filtered 2-key top-K by count + COUNT/SUM/AVG via parallel sparse hash-agg
    R.fused_agg,            # single-table fused fast path
)


_PATH_SINK = None
_SERVED = [None]                     # the routing ledger's marker

# THE PLAN CACHE: the routing decision (lane + spec) replays for repeated
# query text -- parse survives, but the detect cascade dies. Keyed to the
# exact sql, segment path, and row count; execution always runs fresh, so
# results are never cached, only the route. LRU-capped; ~KB per entry.
import os as _os
from collections import OrderedDict as _OD
_PLANS = _OD()
_PLANS_CAP = 256
_BYNAME = {}
_PLAN_EPOCH = [0]


def plan_epoch_bump():
    """Any shelf birth or trim changes what routes are best or valid --
    the cache re-keys and re-detects."""
    _PLAN_EPOCH[0] += 1



def _plan_store(ctx, name, spec):
    if _os.environ.get('WDB_PLANCACHE_OFF') or spec is None:
        return
    key = (ctx.sql, ctx.seg.path, ctx.seg.N, _PLAN_EPOCH[0])
    _PLANS[key] = (name, spec)
    _PLANS.move_to_end(key)
    while len(_PLANS) > _PLANS_CAP:
        _PLANS.popitem(last=False)


def _plan_replay(ctx):
    if _os.environ.get('WDB_PLANCACHE_OFF'):
        return None
    key = (ctx.sql, ctx.seg.path, ctx.seg.N, _PLAN_EPOCH[0])
    hit = _PLANS.get(key)
    if hit is None:
        return None
    _PLANS.move_to_end(key)
    name, spec = hit
    rd = _BYNAME.get(name)
    if rd is None:
        for a in dir(R):
            o = getattr(R, a)
            if hasattr(o, 'name') and hasattr(o, 'execute'):
                _BYNAME[o.name] = o
        rd = _BYNAME.get(name)
    if rd is None:
        return None
    rows = rd.execute(ctx, spec)
    if rows is None:
        _PLANS.pop(key, None)                    # the ground shifted: re-detect
        return None
    _SERVED[0] = name
    if _PATH_SINK is not None:
        _PATH_SINK(ctx, name)
    return rows   # None on speed-runs (one None-check per query, zero cost). The path-run
                   # sets this to a callable(ctx, read_name) to record the winning read.


def route_single_segment(ctx):
    """Route a request over a single clean segment to its read, else the general scan.
    Byte-identical to the previous wdb_db try-chain."""
    _rp = _plan_replay(ctx)
    if _rp is not None:
        return _rp
    # OFFSET: the fast structure reads pre-truncate to LIMIT (so they drop the offset
    # window). Any query with OFFSET goes straight to the general scan, which
    # materializes the full ordered result and applies LIMIT/OFFSET together.
    if wdb_sql._offset(ctx.tree):
        # wherescan is the one structure read that applies OFFSET itself (it materializes the
        # full ordered group set and slices [off:off+lim]) -- let it try before the scan.
        for rd in (R.gbcount, R.funnel, R.dict_count, R.wherescan, R.diskpair):   # the reads that apply OFFSET themselves
            spec = rd.detect(ctx)
            if spec is not None:
                rows = rd.execute(ctx, spec)
                if rows is not None:
                    _plan_store(ctx, rd.name, spec); _SERVED[0] = rd.name;  _PATH_SINK(ctx, rd.name) if _PATH_SINK is not None else None
                    return rows
        _SERVED[0] = 'general_scan';  _PATH_SINK(ctx, 'general_scan') if _PATH_SINK is not None else None
        return R.general_scan(ctx)
    if _agg_or_group(ctx.tree) or _has_window(ctx.tree):
        for read in _READ_ORDER:
            spec = read.detect(ctx)
            if spec is None:
                continue
            rows = read.execute(ctx, spec)
            if rows is not None:
                _plan_store(ctx, read.name, spec); _SERVED[0] = read.name;  _PATH_SINK(ctx, read.name) if _PATH_SINK is not None else None
                return rows
    else:
        # non-agg projection: the value-sorted dict read, then the cluster-ordered
        # top-K read, else the scan
        spec = R.sorted_proj.detect(ctx)
        if spec is not None:
            rows = R.sorted_proj.execute(ctx, spec)
            if rows is not None:
                _plan_store(ctx, R.sorted_proj.name, spec); _SERVED[0] = R.sorted_proj.name;  _PATH_SINK(ctx, R.sorted_proj.name) if _PATH_SINK is not None else None
                return rows
        spec = R.cluster_topk.detect(ctx)
        if spec is not None:
            rows = R.cluster_topk.execute(ctx, spec)
            if rows is not None:
                _plan_store(ctx, R.cluster_topk.name, spec); _SERVED[0] = R.cluster_topk.name;  _PATH_SINK(ctx, R.cluster_topk.name) if _PATH_SINK is not None else None
                return rows
        spec = R.value_topk.detect(ctx)      # ordered dump: partition the key, decode k rows
        if spec is not None:
            rows = R.value_topk.execute(ctx, spec)
            if rows is not None:
                _plan_store(ctx, R.value_topk.name, spec); _SERVED[0] = R.value_topk.name;  _PATH_SINK(ctx, R.value_topk.name) if _PATH_SINK is not None else None
                return rows
        spec = R.firstsorted.detect(ctx)     # staircase ORDER BY unprojected time
        if spec is not None:
            rows = R.firstsorted.execute(ctx, spec)
            if rows is not None:
                _plan_store(ctx, R.firstsorted.name, spec); _SERVED[0] = R.firstsorted.name;  _PATH_SINK(ctx, R.firstsorted.name) if _PATH_SINK is not None else None
                return rows
        spec = R.firstk.detect(ctx)         # staircase early-exit: first k matches ARE the answer
        if spec is not None:
            rows = R.firstk.execute(ctx, spec)
            if rows is not None:
                _plan_store(ctx, R.firstk.name, spec); _SERVED[0] = R.firstk.name;  _PATH_SINK(ctx, R.firstk.name) if _PATH_SINK is not None else None
                return rows
        spec = R.wherescan.detect(ctx)       # rows mode: WHERE + ORDER BY cluster col LIMIT k
        if spec is not None:
            rows = R.wherescan.execute(ctx, spec)
            if rows is not None:
                _plan_store(ctx, R.wherescan.name, spec); _SERVED[0] = R.wherescan.name;  _PATH_SINK(ctx, R.wherescan.name) if _PATH_SINK is not None else None
                return rows
    _SERVED[0] = 'general_scan';  _PATH_SINK(ctx, 'general_scan') if _PATH_SINK is not None else None
    return R.general_scan(ctx)
