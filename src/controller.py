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
    R.stair,                # staircase column: single-key GROUP BY from step positions (no decode)
    R.regexgroup,           # GROUP BY regex over the dict: per-code counts, V-level strings
    R.window,               # window fns: stable scatter by partition, lanes inherit cluster order
    R.dict_count,           # dictionary per-code counts (sidecar: answers before any scan)
    R.groupself,            # counting board: GROUP BY K + WHERE on K, bins not rows
    R.coscan,               # fused conjunctive COUNT: zone-map veto, one walk
    R.wherescan,            # conjunctive WHERE: stair spans + blocked-frame predicate scan, disk-only
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


_PATH_SINK = None   # None on speed-runs (one None-check per query, zero cost). The path-run
                   # sets this to a callable(ctx, read_name) to record the winning read.


def route_single_segment(ctx):
    """Route a request over a single clean segment to its read, else the general scan.
    Byte-identical to the previous wdb_db try-chain."""
    # OFFSET: the fast structure reads pre-truncate to LIMIT (so they drop the offset
    # window). Any query with OFFSET goes straight to the general scan, which
    # materializes the full ordered result and applies LIMIT/OFFSET together.
    if wdb_sql._offset(ctx.tree):
        # wherescan is the one structure read that applies OFFSET itself (it materializes the
        # full ordered group set and slices [off:off+lim]) -- let it try before the scan.
        for rd in (R.dict_count, R.wherescan, R.diskpair):   # the reads that apply OFFSET themselves
            spec = rd.detect(ctx)
            if spec is not None:
                rows = rd.execute(ctx, spec)
                if rows is not None:
                    if _PATH_SINK is not None: _PATH_SINK(ctx, rd.name)
                    return rows
        if _PATH_SINK is not None: _PATH_SINK(ctx, 'general_scan')
        return R.general_scan(ctx)
    if _agg_or_group(ctx.tree) or _has_window(ctx.tree):
        for read in _READ_ORDER:
            spec = read.detect(ctx)
            if spec is None:
                continue
            rows = read.execute(ctx, spec)
            if rows is not None:
                if _PATH_SINK is not None: _PATH_SINK(ctx, read.name)
                return rows
    else:
        # non-agg projection: the value-sorted dict read, then the cluster-ordered
        # top-K read, else the scan
        spec = R.sorted_proj.detect(ctx)
        if spec is not None:
            rows = R.sorted_proj.execute(ctx, spec)
            if rows is not None:
                if _PATH_SINK is not None: _PATH_SINK(ctx, R.sorted_proj.name)
                return rows
        spec = R.cluster_topk.detect(ctx)
        if spec is not None:
            rows = R.cluster_topk.execute(ctx, spec)
            if rows is not None:
                if _PATH_SINK is not None: _PATH_SINK(ctx, R.cluster_topk.name)
                return rows
        spec = R.value_topk.detect(ctx)      # ordered dump: partition the key, decode k rows
        if spec is not None:
            rows = R.value_topk.execute(ctx, spec)
            if rows is not None:
                if _PATH_SINK is not None: _PATH_SINK(ctx, R.value_topk.name)
                return rows
        spec = R.wherescan.detect(ctx)       # rows mode: WHERE + ORDER BY cluster col LIMIT k
        if spec is not None:
            rows = R.wherescan.execute(ctx, spec)
            if rows is not None:
                if _PATH_SINK is not None: _PATH_SINK(ctx, R.wherescan.name)
                return rows
    if _PATH_SINK is not None: _PATH_SINK(ctx, 'general_scan')
    return R.general_scan(ctx)
