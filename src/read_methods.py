"""
read_methods.py -- THE READS LAYER.

A *read* explores one stored structure and returns the retrieved pattern (rows),
or None to decline. Each read is now a pair:
    detect(ctx)        -> spec | None   ACTIVATION (cheap: query shape + materialization)
    execute(ctx, spec) -> rows | None   THE READ (may still decline on measured conditions)
bundled in a Read object. The spec is OPAQUE to the controller -- detect hands it
straight back to execute, so heterogeneous operator signatures stay hidden in the
adapters here (e.g. group_mix/distinct_sidecar need the db handle + segment path).

This is the seam the controller routes through, and the seam where tier-2
(materialization) decisions will later be PRELOADED: detect is the preloadable part.

ctx (ReadContext) carries everything any read needs: segment, parsed query, column
map, db handle, segment path, raw sql, and the throughput/latency flag.
"""
import wdb_cube
import wdb_gbcount
import wdb_groupself
import wdb_heavypair
import wdb_countpos
import wdb_gridwalk
import wdb_smallk
import wdb_stair
import wdb_blockstats
import wdb_wherescan
import wdb_coscan
import wdb_diskpair
import wdb_distinctlim
import wdb_sumtopk
import wdb_regexgroup
import wdb_window
import wdb_pairagg
import wdb_scanpair
import wdb_survgroup
import wdb_compound
import wdb_gdsidecar
import wdb_groupdistinct
import wdb_groupmix
import wdb_valsort
import wdb_clustertopk
import wdb_bsi_exec
import wdb_join
import wdb_sql


class ReadContext:
    """Everything a read might need to explore the structure. Built once per query
    by the dispatcher and handed to the controller."""
    __slots__ = ('db', 'name', 'seg', 'path', 'tree', 'cmap', 'sql', 'esc')

    def __init__(self, db, name, seg, path, tree, cmap, sql, esc):
        self.db = db          # the WaveDB handle (sidecars / fk pointers)
        self.name = name      # logical table name
        self.seg = seg        # the single materialized segment being read
        self.path = path      # that segment's on-disk path (sidecar lookups)
        self.tree = tree      # parsed sqlglot query
        self.cmap = cmap      # logical -> physical column map
        self.sql = sql        # raw sql (post join-rewrite), for the executor paths
        self.esc = esc        # True = latency/fused preference, False = throughput


class Read:
    """A read = its activation (detect) + its retrieval (execute). The controller calls
    detect(ctx) to route (cheap, preloadable) and execute(ctx, spec) to retrieve. The
    spec is opaque to the controller; only the matching execute interprets it."""
    __slots__ = ('name', 'detect', 'execute', 'note')

    def __init__(self, name, detect, execute, note=''):
        self.name = name
        self.detect = detect
        self.execute = execute
        self.note = note


# --- structure reads (operators with a clean detect/execute split) -----------------

cube = Read('cube',
            lambda c: wdb_cube.detect(c.seg, c.tree, c.cmap),
            lambda c, spec: wdb_cube.execute(c.seg, spec),
            'pre-materialized cube')

blockstats = Read('blockstats',
                  lambda c: wdb_blockstats.detect(c.seg, c.tree, c.cmap),
                  lambda c, spec: wdb_blockstats.execute(c.seg, spec),
                  'whole-table aggregates from per-block statistics (disk-only)')

regexgroup = Read('regexgroup',
                  lambda c: wdb_regexgroup.detect(c.seg, c.tree, c.cmap),
                  lambda c, spec: wdb_regexgroup.execute(c.seg, spec),
                  'GROUP BY REGEXP_REPLACE(dict col): per-code counts + dict-level regex')

window = Read('window',
              lambda c: wdb_window.detect(c.seg, c.tree, c.cmap),
              lambda c, spec: wdb_window.execute(c.seg, spec),
              'window functions on the fused motion: one placement, two coordinates')

groupself = Read('groupself',
                 lambda c: wdb_groupself.detect(c.seg, c.tree, c.cmap),
                 lambda c, spec: wdb_groupself.execute(c.seg, spec),
                 'the counting board: GROUP BY K with WHERE on K -- bins, not rows')

coscan = Read('coscan',
              lambda c: wdb_coscan.detect(c.seg, c.tree, c.cmap),
              lambda c, spec: wdb_coscan.execute(c.seg, spec),
              'fused conjunctive COUNT: blockstats veto + one block walk, all predicates together')

wherescan = Read('wherescan',
                 lambda c: wdb_wherescan.detect(c.seg, c.tree, c.cmap),
                 lambda c, spec: wdb_wherescan.execute(c.seg, spec),
                 'conjunctive WHERE via stair spans + parallel blocked-frame scan (disk-only)')

stair = Read('stair',
             lambda c: wdb_stair.detect(c.seg, c.tree, c.cmap),
             lambda c, spec: wdb_stair.execute(c.seg, spec),
             'staircase-column group-by from step positions')

dict_count = Read('dict_count',
                  lambda c: wdb_gbcount.detect(c.seg, c.tree, c.cmap),
                  lambda c, spec: wdb_gbcount.execute(c.seg, spec),
                  'dictionary per-code counts')

heavypair = Read('heavypair',
                 lambda c: wdb_heavypair.detect(c.seg, c.tree, c.cmap),
                 lambda c, spec: wdb_heavypair.execute(c.seg, spec),
                 'two-key COUNT(*) top-N from a count-sorted pair sidecar')

gridwalk = Read('gridwalk',
                lambda c: wdb_gridwalk.detect(c.seg, c.tree, c.cmap),
                lambda c, spec: wdb_gridwalk.execute(c.seg, spec),
                'two-key COUNT(*) top-K via grid filled-cell + count-ordered head')

smallk = Read('smallk',
              lambda c: wdb_smallk.detect(c.seg, c.tree, c.cmap),
              lambda c, spec: wdb_smallk.execute(c.seg, spec),
              'narrow 2-3 key COUNT(*) via one fused pass onto a composite board')

sumtopk = Read('sumtopk',
               lambda c: wdb_sumtopk.detect(c.seg, c.tree, c.cmap),
               lambda c, spec: wdb_sumtopk.execute(c.seg, spec),
               'single big-key SUM top-K: fused board kernel + argpartition, labels for winners only')

distinctlim = Read('distinctlim',
                   lambda c: wdb_distinctlim.detect(c.seg, c.tree, c.cmap),
                   lambda c, spec: wdb_distinctlim.execute(c.seg, spec),
                   'DISTINCT (hour, key) LIMIT n: stair-ridden block walk, early exit')

diskpair = Read('diskpair',
                lambda c: wdb_diskpair.detect(c.seg, c.tree, c.cmap),
                lambda c, spec: wdb_diskpair.execute(c.seg, spec),
                'disk-only 2-key GROUP BY count top-K: blocked-frame scan + norm split + k-way merge')

pairagg = Read('pairagg',
               lambda c: wdb_pairagg.detect(c.seg, c.tree, c.cmap),
               lambda c, spec: wdb_pairagg.execute(c.seg, spec),
               'filtered 2-key top-K by count with COUNT/SUM/AVG via parallel sparse hash-agg')

countpos = Read('countpos',
                lambda c: wdb_countpos.detect(c.seg, c.tree, c.cmap),
                lambda c, spec: wdb_countpos.execute(c.seg, spec),
                'two-key COUNT(*) top-K via per-row count-class column presence-scan')

scanpair = Read('scanpair',
                lambda c: wdb_scanpair.detect(c.seg, c.tree, c.cmap),
                lambda c, spec: wdb_scanpair.execute(c.seg, spec),
                'two-key COUNT(*) top-N, high-card non-key filter answered by code-space scan')

survivor_group = Read('survivor_group',
                      lambda c: wdb_survgroup.detect(c.seg, c.tree, c.cmap),
                      lambda c, spec: wdb_survgroup.execute(c.seg, spec),
                      'filtered high-card group via survivor ranges')

compound_filter = Read('compound_filter',
                       lambda c: wdb_compound.detect(c.seg, c.tree, c.cmap),
                       lambda c, spec: wdb_compound.execute(c.seg, spec),
                       'multi-condition AND filter')

distinct_sidecar = Read('distinct_sidecar',
                        lambda c: wdb_gdsidecar.detect(c.db, c.name, c.seg, c.path, c.tree, c.cmap),
                        lambda c, spec: wdb_gdsidecar.execute(c.seg, spec),
                        'prebuilt group->distinct sidecar')

group_distinct = Read('group_distinct',
                      lambda c: wdb_groupdistinct.detect(c.seg, c.tree, c.cmap),
                      lambda c, spec: wdb_groupdistinct.execute(c.seg, spec, c.tree),
                      'one-pass code-hash distinct walk')

group_mix = Read('group_mix',
                 lambda c: wdb_groupmix.detect(c.seg, c.tree, c.cmap),
                 lambda c, spec: wdb_groupmix.execute(c.seg, spec, c.tree,
                                                      db=c.db, table=c.name, segment_path=c.path),
                 'group + foldables + one distinct, one walk')


# --- value-sorted projection (non-agg): SELECT col WHERE col<>'' ORDER BY col LIMIT k ---
sorted_proj = Read('sorted_proj',
                   lambda c: wdb_valsort.detect(c.seg, c.tree, c.cmap),
                   lambda c, spec: wdb_valsort.execute(c.seg, spec),
                   'value-sorted dict projection top-K')

cluster_topk = Read('cluster_topk',
                    lambda c: wdb_clustertopk.detect(c.seg, c.tree, c.cmap),
                    lambda c, spec: wdb_clustertopk.execute(c.seg, spec),
                    'cluster-ordered projection top-K')

# --- value top-K dump: ORDER BY numeric/temporal col LIMIT k, no full sort ---
import wdb_valtopk
value_topk = Read('value_topk',
                  lambda c: wdb_valtopk.detect(c.seg, c.tree, c.cmap),
                  lambda c, spec: wdb_valtopk.execute(c.seg, spec),
                  'partition-bounded ordered dump')

# --- clustered-structure slices (detect = the cheap slice/group-slice guard) --------

cluster_slice = Read('cluster_slice',
                     lambda c: True if wdb_sql._cluster_will_slice(c.seg, c.tree, c.cmap) else None,
                     lambda c, spec: wdb_sql.execute(c.seg, c.sql, col_map=c.cmap, tree=c.tree),
                     'contiguous slice of a clustered structure')

cluster_group_slice = Read('cluster_group_slice',
                           lambda c: True if wdb_sql._cluster_will_group_slice(c.seg, c.tree, c.cmap) else None,
                           lambda c, spec: wdb_sql.execute(c.seg, c.sql, col_map=c.cmap, tree=c.tree),
                           'clustered group-runs as slices')


# --- throughput / fused reads: these signal "not my shape" by RAISING, so their
#     execute catches it and returns None (declines) -- folding them into the same
#     uniform detect/execute loop. bsi's detect carries the throughput-only gate.

def _bsi_detect(c):
    return None if c.esc else True          # bsi filter is throughput-mode only

def _bsi_execute(c, spec):
    try:
        return wdb_bsi_exec.execute(c.seg, c.tree, c.cmap)
    except wdb_bsi_exec._BSIUnsupported:
        return None                          # shape/selectivity unfit -> fall through

bsi_filter = Read('bsi_filter', _bsi_detect, _bsi_execute, 'bit-sliced index filter (throughput)')


def _fused_detect(c):
    # fused_agg only groups by bare columns; a computed group key (e.g. EXTRACT(unit FROM col)) is
    # not its shape. Decline at detect (sub-us) instead of attempting it and burning an O(N)-ish
    # table_agg pass before raising _FastUnsupported -- that wasted ~1ms on 100M before the general
    # path's date-coarsening rollup could run. A bare GROUP BY name may be a SELECT alias for a
    # computed expr (GROUP BY g where g AS EXTRACT(...)), so resolve aliases before deciding.
    g = c.tree.args.get('group')
    if g is not None:
        proj = c.tree.expressions
        for ge in g.expressions:
            node = ge.this if isinstance(ge, wdb_sql.E.Alias) else ge
            if isinstance(node, wdb_sql.E.Column):
                nm = node.name
                for p in proj:                       # resolve a bare name to its SELECT-alias expr
                    if isinstance(p, wdb_sql.E.Alias) and p.alias == nm:
                        node = p.this; break
            if not isinstance(node, wdb_sql.E.Column):
                return None
    return True                              # always eligible to try; execute decides

def _fused_execute(c, spec):
    try:
        return wdb_join.table_agg(c.db, c.tree)
    except wdb_join._FastUnsupported:
        return None                          # not the fused shape -> fall through

fused_agg = Read('fused_agg', _fused_detect, _fused_execute, 'single-table fused fast path')


# --- the catch-all: read the columns directly. Not a Read -- it never declines. -----

def general_scan(ctx):
    """The general scan -- read the columns directly. Always returns a result."""
    return wdb_sql.execute(ctx.seg, ctx.sql, col_map=ctx.cmap)
