"""
read_methods.py -- THE READS LAYER.

A *read* explores one stored structure and returns the retrieved pattern (rows),
or None to decline because the structure/query shape isn't its kind.

Uniform contract:
    read(ctx) -> rows | None

`ctx` (ReadContext) carries everything any read might need: the segment, the
parsed query, the column map, the db handle, the segment path, the raw sql, and
the throughput/latency flag. This is the data-access layer -- the menu of ways to
explore the structure. The CONTROLLER (controller.py) decides which read fires.

Increment 1: the read *bodies* still live in the operator modules; the functions
below are the gathered entry points (thin calls) so behavior is byte-identical.
Bodies get relocated here over later increments. Two reads (bsi_filter, fused_agg)
signal "not my shape" by raising their own exception instead of returning None --
left as-is here; the controller catches them.

Read order / activation lives in controller.py, not here.
"""
import wdb_cube
import wdb_gbcount
import wdb_survgroup
import wdb_compound
import wdb_gdsidecar
import wdb_groupdistinct
import wdb_groupmix
import wdb_bsi_exec
import wdb_join
import wdb_sql


class ReadContext:
    """Everything a read might need to explore the structure. Built once per query
    by the dispatcher and handed to the controller."""
    __slots__ = ('db', 'name', 'seg', 'path', 'tree', 'cmap', 'sql', 'esc')

    def __init__(self, db, name, seg, path, tree, cmap, sql, esc):
        self.db = db          # the WaveDB handle (for reads that need sidecars/fk pointers)
        self.name = name      # logical table name
        self.seg = seg        # the single materialized segment being read
        self.path = path      # that segment's on-disk path (for sidecar lookups)
        self.tree = tree      # parsed sqlglot query
        self.cmap = cmap      # logical -> physical column map
        self.sql = sql        # raw sql (post join-rewrite), for the executor paths
        self.esc = esc        # True = latency/fused preference, False = throughput


# --- structure reads: each explores one materialized form, returns rows or None ---

def cube(c):
    """Read a pre-materialized cube: grouped aggregates precomputed at build time."""
    return wdb_cube.try_cube(c.seg, c.tree, c.cmap)


def dict_count(c):
    """Read per-code counts straight from a dictionary: COUNT(*) GROUP BY, no scan."""
    return wdb_gbcount.try_gbcount(c.seg, c.tree, c.cmap)


def survivor_group(c):
    """Read a selectively-filtered high-cardinality group via survivor ranges."""
    return wdb_survgroup.try_survgroup(c.seg, c.tree, c.cmap)


def compound_filter(c):
    """Read grouped counts under a multi-condition AND filter."""
    return wdb_compound.try_compound(c.seg, c.tree, c.cmap)


def distinct_sidecar(c):
    """Read a prebuilt group->distinct sidecar: per-group COUNT(DISTINCT), no walk."""
    return wdb_gdsidecar.try_serve(c.db, c.name, c.seg, c.path, c.tree, c.cmap)


def group_distinct(c):
    """Read per-group COUNT(DISTINCT) via a one-pass code-hash walk."""
    return wdb_groupdistinct.try_groupdistinct(c.seg, c.tree, c.cmap)


def group_mix(c):
    """Read group + foldable aggregates + one distinct together in one walk."""
    return wdb_groupmix.try_groupmix(c.seg, c.tree, c.cmap,
                                     db=c.db, table=c.name, segment_path=c.path)


def cluster_slice(c):
    """Read a contiguous slice of a clustered (sorted) structure, when applicable."""
    if wdb_sql._cluster_will_slice(c.seg, c.tree, c.cmap):
        return wdb_sql.execute(c.seg, c.sql, col_map=c.cmap, tree=c.tree)
    return None


def cluster_group_slice(c):
    """Read clustered group-runs as slices, when applicable."""
    if wdb_sql._cluster_will_group_slice(c.seg, c.tree, c.cmap):
        return wdb_sql.execute(c.seg, c.sql, col_map=c.cmap, tree=c.tree)
    return None


# --- these two raise their own "not my shape" exception instead of returning None ---

def bsi_filter(c):
    """Read a bit-sliced index for throughput filtering. Raises _BSIUnsupported."""
    return wdb_bsi_exec.execute(c.seg, c.tree, c.cmap)


def fused_agg(c):
    """Read a single-table aggregate via the fused fast path. Raises _FastUnsupported."""
    return wdb_join.table_agg(c.db, c.tree)


# --- the catch-all: read the columns directly ---

def general_scan(c):
    """The general scan -- read the columns directly. Always returns a result."""
    return wdb_sql.execute(c.seg, c.sql, col_map=c.cmap)
