"""
controller.py -- THE CONTROLLER (routing layer).

A query, in the user sense, is a *route*: the controller takes a request and sends
it down a route that resolves to a read (and, later, a worker). The controller
holds no retrieval logic of its own -- it owns the ORDER of reads and the
activation gating, and wires requests to reads supplied by its imports.

Today this reproduces, byte-identically, the hand-ordered try-chain that used to
live inline in wdb_db.run(): structure reads first (cube -> dict-count -> ... ->
cluster slices), then the throughput/fused reads, then the general scan as the
catch-all. The reads still self-select (return None / raise when a request isn't
their shape); the controller just walks the order and returns the first hit.

WHERE THIS IS HEADED (the point of the layer):
  - bind routes to reads from system state ONCE, when materialization settles
    (the "preloaded try state"), so query time becomes pure dispatch with no
    per-query shape-checking;
  - hang a WORKER off each route, so a route is (read -> worker).
Both are future increments; the seam is here now.
"""
import read_methods as R
import wdb_sql
import wdb_bsi_exec
import wdb_join


def _agg_or_group(tree):
    """The precondition for the structure-read chain: the request must aggregate,
    group, or ask for distinct. Plain projections skip straight to the scan."""
    return (tree.args.get('group') is not None
            or tree.args.get('distinct') is not None
            or any(wdb_sql._agg_kind(e) for e in tree.expressions))


# The single-segment read order -- the "try state". Each entry is a read in
# read_methods; the first to return non-None wins.
_READ_ORDER = (
    R.cube,                 # pre-materialized cube
    R.dict_count,           # dictionary per-code counts
    R.survivor_group,       # filtered high-card group via survivor ranges
    R.compound_filter,      # multi-condition AND filter
    R.distinct_sidecar,     # prebuilt group->distinct sidecar
    R.group_distinct,       # one-pass code-hash distinct walk
    R.group_mix,            # group + foldables + one distinct, one walk
    R.cluster_slice,        # contiguous slice of a clustered structure
    R.cluster_group_slice,  # clustered group-runs as slices
)


def route_single_segment(ctx):
    """Route a request over a single clean segment to its read.

    Byte-identical to the previous wdb_db try-chain: if the request aggregates/
    groups/distincts, walk the structure reads in order; then (throughput only)
    the bit-sliced filter; then the fused aggregate; each falling through on its
    own 'not my shape' signal. Anything unrouted falls to the general scan."""
    if _agg_or_group(ctx.tree):
        for read in _READ_ORDER:
            rows = read(ctx)
            if rows is not None:
                return rows
        if not ctx.esc:
            try:
                return R.bsi_filter(ctx)              # throughput path
            except wdb_bsi_exec._BSIUnsupported:
                pass                                  # shape/selectivity unfit
        try:
            return R.fused_agg(ctx)                   # fused fast path
        except wdb_join._FastUnsupported:
            pass                                      # fall back to the scan
    return R.general_scan(ctx)
