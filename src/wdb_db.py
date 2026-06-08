"""WaveDB Database: a directory-backed database you build with SQL.

    db = Database.create('/path/mydb')
    db.run("CREATE TABLE users (id INT, name VARCHAR)")
    db.run("INSERT INTO users VALUES (1,'alice')")
    rows, header = db.run("SELECT * FROM users WHERE id = 1")

Borrowed SQL syntax + sqlglot parser; catalog, storage, and execution are WaveDB's.
Step 3a: single segment per table.
"""
import sqlglot, sqlglot.expressions as E
from wdb_catalog import Catalog
from wdb_engine import Segment
import os
import wdb_ddl, wdb_dml, wdb_sql, wdb_merge, wdb_compact, wdb_join, wdb_fkptr, wdb_bsi_exec
import numpy as np

class Database:
    def __init__(self, catalog):
        self.cat = catalog
        self._seg_cache = {}   # path -> ((mtime_ns,size), Segment): immutable .wdb base, reused across queries
        self._ptr_cache = {}   # (child_seg_path, fk_col) -> ((mtime_ns,size), int64 ptr array)

    @classmethod
    def create(cls, dbdir): return cls(Catalog.create(dbdir))
    @classmethod
    def open(cls, dbdir): return cls(Catalog.open(dbdir))

    def open_segment(self, path, table=None):
        """Cached Segment for an immutable .wdb. Construction reads+parses the whole file (~200ms for a
        large segment), so caching makes repeated analytical queries pay that cost once. Reconstructs only
        if the file changed (mtime/size). Mutable state is refreshed every call: presence (DELETE) and
        override (UPDATE) sidecars are forced to reload, and ADD COLUMN synth columns re-registered."""
        st = os.stat(path); key = (st.st_mtime_ns, st.st_size)
        hit = self._seg_cache.get(path)
        if hit is None or hit[0] != key:
            seg = Segment(path); self._seg_cache[path] = (key, seg)
        else:
            seg = hit[1]
            seg._presence = 0; seg._ov = 0   # reload DELETE/UPDATE sidecars (do not change .wdb mtime)
        if table is not None:
            wdb_dml.register_synth(self.cat, seg, table)
        return seg

    def fk_pointer(self, child_seg_path, fk_col):
        """Cached FK-pointer array (absolute parent row positions). The sidecar is immutable once built
        (only create_fk_pointer rewrites it), so decompress+cumsum is paid once, not per query. None if
        no sidecar exists."""
        sp = wdb_fkptr.path_for(child_seg_path, fk_col)
        if not os.path.exists(sp): return None
        st = os.stat(sp); key = (st.st_mtime_ns, st.st_size)
        ck = (child_seg_path, fk_col); hit = self._ptr_cache.get(ck)
        if hit is None or hit[0] != key:
            arr = wdb_fkptr.load(child_seg_path, fk_col); self._ptr_cache[ck] = (key, arr); return arr
        return hit[1]

    def _table_in(self, tree):
        f = tree.find(E.From)
        if f is None: raise NotImplementedError("SELECT without FROM")
        return f.this.name

    def run(self, sql):
        tree = sqlglot.parse_one(sql, read='duckdb')
        if isinstance(tree, E.Create):
            return wdb_ddl.create_table(self.cat, sql)
        if isinstance(tree, E.Alter):
            return wdb_ddl.alter_table(self.cat, sql, self)
        if isinstance(tree, E.Insert):
            return wdb_dml.insert(self.cat, sql)
        if isinstance(tree, E.Drop):
            self.cat.drop_table(tree.this.this.name); return None
        if isinstance(tree, E.Delete):
            return wdb_dml.delete(self.cat, sql)
        if isinstance(tree, E.Update):
            return wdb_dml.update(self.cat, sql)
        if isinstance(tree, E.Select):
            if tree.args.get('joins'):
                return wdb_join.join_query(self, sql)
            name = self._table_in(tree)
            phys = self.cat.phys_map(name)
            cmap = {c: phys.get(c, c) for c in self.cat.column_names(name)}  # complete logical->physical
            paths = self.cat.segment_paths(name)
            segs = [self.open_segment(p, name) for p in paths]
            hp = wdb_dml.hot_path(self.cat, name)
            hot = hp if os.path.exists(hp) else None
            if hot is None and len(segs) == 1:
                if (tree.args.get('group') is not None or tree.args.get('distinct') is not None
                        or any(wdb_sql._agg_kind(e) for e in tree.expressions)):
                    if wdb_sql._cluster_will_slice(segs[0], tree, cmap):
                        return wdb_sql.execute(segs[0], sql, col_map=cmap, tree=tree)  # clustered slice path
                    if wdb_sql._cluster_will_group_slice(segs[0], tree, cmap):
                        return wdb_sql.execute(segs[0], sql, col_map=cmap, tree=tree)  # clustered group-slice path
                    try:
                        return wdb_bsi_exec.execute(segs[0], tree, cmap)  # BSI filter-aggregate path
                    except wdb_bsi_exec._BSIUnsupported:
                        pass                                          # shape/selectivity unfit -> fused/fallback
                    try:
                        return wdb_join.table_agg(self, tree)        # single-table aggregate -> fused fast path
                    except wdb_join._FastUnsupported:
                        pass                                          # fall back to the single-table executor
                return wdb_sql.execute(segs[0], sql, col_map=cmap)
            if hot is None and not segs:
                raise ValueError(f"table {name!r} has no data yet")
            return wdb_merge.merge_query(segs, hot, sql, col_map=cmap)
        raise NotImplementedError(f"unsupported statement: {type(tree).__name__}")

    def create_fk_pointer(self, child, fk_col, parent, parent_key):
        """Pre-resolve a foreign key into a stored parent-row pointer, turning future joins on
        child.fk_col = parent.parent_key into a gather. Requires single-segment tables and the parent
        stored sorted by a UNIQUE parent_key. Verifies referential integrity. Saves a sidecar."""
        cpaths = self.cat.segment_paths(child); ppaths = self.cat.segment_paths(parent)
        if len(cpaths) != 1 or len(ppaths) != 1:
            raise NotImplementedError("create_fk_pointer: single-segment tables only (step 1)")
        # Reuse already-open (cached) segments instead of opening duplicate file
        # buffers. If a query already touched these tables the segments are resident;
        # if not, this warms the cache the later joins will use anyway.
        cseg = self.open_segment(cpaths[0], child); pseg = self.open_segment(ppaths[0], parent)
        pphys = self.cat.phys_map(parent); cphys = self.cat.phys_map(child)
        pk = pseg.values(pphys.get(parent_key, parent_key))
        if pk.dtype.kind not in 'iufM':
            raise NotImplementedError("create_fk_pointer: numeric/temporal parent key only (step 1)")
        # Sorted + unique in one cheap pass over the (small) parent key. Avoids
        # np.unique's full-array sort+copy (~73 MB on a 1.5M key); views only.
        if (pk[1:] < pk[:-1]).any():
            raise ValueError(f"parent {parent!r} is not stored sorted by {parent_key!r}")
        if (pk[1:] == pk[:-1]).any():
            raise ValueError(f"parent key {parent_key!r} is not unique")
        npk = len(pk); fcol = cphys.get(fk_col, fk_col)
        # Resolve the child->parent pointer block by block so we never hold the whole
        # child key plus searchsorted/where/gather/astype temporaries at once. ptr is
        # the int64 result, written in place (searchsorted already returns int64, so no
        # astype copy). Integrity is verified per block. Peak extra = one block, not 6M.
        ptr = np.empty(cseg.N, dtype=np.int64)
        def _verify_resolve(fkv, out):
            p = np.searchsorted(pk, fkv)
            if (p >= npk).any() or not np.array_equal(pk[p], fkv):
                raise ValueError("referential integrity violation: some child keys are absent in the parent")
            out[:] = p
        if cseg._overrides(fcol) is not None:
            # Pending DML edits on the child key: values_range skips overrides, so fall
            # back to the override-aware whole-column decode (correctness over peak RAM).
            _verify_resolve(cseg.values(fcol), ptr)
        else:
            CH = 1_000_000
            for lo in range(0, cseg.N, CH):
                hi = min(lo + CH, cseg.N)
                _verify_resolve(cseg.values_range(fcol, lo, hi), ptr[lo:hi])
        wdb_fkptr.save(cpaths[0], fk_col, ptr)
        self.cat.add_fk_pointer(child, fk_col, parent, parent_key)
        return int(cseg.N)

    def set_table_mode(self, name, mode):
        """Operator control: 'buffered' = high-traffic, INSERT appends to hot buffer
        without re-encoding; SELECT merges hot+cold; call flush() to fold in."""
        self.cat.set_table_mode(name, mode)

    def flush(self, name):
        return wdb_dml.flush(self.cat, name)

    def compact(self, name, seg_files=None):
        """Merge cold segments into one, verifying FD labels on the union."""
        return wdb_compact.compact(self.cat, name, seg_files)

    def tables(self): return self.cat.list_tables()
