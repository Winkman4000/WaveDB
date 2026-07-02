"""WaveDB Database: a directory-backed database you build with SQL.

    db = Database.create('/path/mydb')
    db.run("CREATE TABLE users (id INT, name VARCHAR)")
    db.run("INSERT INTO users VALUES (1,'alice')")
    rows, header = db.run("SELECT * FROM users WHERE id = 1")

Borrowed SQL syntax + sqlglot parser; catalog, storage, and execution are WaveDB's.
run() is the spine: parse -> commands.route (mutations) -> the read path
(controller.route_single_segment for one clean segment, wdb_merge for many).
"""
import sqlglot, sqlglot.expressions as E
from wdb_catalog import Catalog
from wdb_engine import Segment
import os
import wdb_ddl, wdb_dml, wdb_sql, wdb_merge, wdb_compact, wdb_join, wdb_fkptr, wdb_bsi_exec, wdb_cube, wdb_gbcount, wdb_survgroup, wdb_compound, wdb_groupdistinct, wdb_gdsidecar, wdb_groupmix
import read_methods, controller
import wdb_gridwalk_live
import commands
import numpy as np
import functools

@functools.lru_cache(maxsize=2048)
def _parse_sql_cached(sql):
    """Parse SQL -> AST, memoised per process. Parsing is ~half the wall time of a fast query
    (cProfile: 46% of COUNT(*), and the entire cost of parse-bound shapes), and the SQL->AST map is
    pure, so caching it makes repeated queries (the common case: dashboards, prepared statements,
    throughput workers looping one query) parse exactly once. Returns the SAME tree on a hit -- safe
    because the executor only READS the tree (validated by the full suite). Bounded so a distinct-query
    workload can't grow it without limit."""
    return sqlglot.parse_one(sql, read='duckdb')

class Database:
    def __init__(self, catalog):
        self.cat = catalog
        self._seg_cache = {}   # path -> ((mtime_ns,size), Segment): immutable .wdb base, reused across queries
        self._ptr_cache = {}   # (child_seg_path, fk_col) -> ((mtime_ns,size), int64 ptr array)
        self._gd_cache = {}    # (seg_path, group, target) -> ((mtime_ns,size), sidecar dict): load .npz once
        # Operator-selected execution intent. escalate=False (default) optimizes for THROUGHPUT
        # (shared DB, many concurrent 1-thread workers): engages the BSI filter-index, which reads
        # fewer bytes per query. escalate=True optimizes a single query for LATENCY: it skips BSI
        # and uses the fully parallel fused scan, which wins when one query owns all cores. Both
        # paths return identical results; the choice is the caller's, per-run or as this default.
        self.escalate = False

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

    def prewarm(self, table=None, budget_bytes=4 << 30, verbose=True):
        """Eager launch classification + build: walk EVERY eligible column pair of every table and
        materialize its gridwalk structure into RAM, under a structures budget. Exhaustive by
        construction -- every pair ends the loop CLASSIFIED in the returned manifest (built, or
        deferred with the reason), so no pair-shaped query can fall through unclassified. RAM-only
        by design (disk stays compressed); the one-time build cost moves from first-query to launch,
        which also makes the total footprint VISIBLE here instead of a mid-query surprise.
        Returns (manifest, totals)."""
        import itertools, time
        import wdb_gridwalk as GW
        import wdb_policies as P
        manifest = []
        tot_bytes = 0; t_all = time.perf_counter()
        for t in ([table] if table else self.cat.list_tables()):
            for path in self.cat.segment_paths(t):
                seg = self.open_segment(path, t)
                elig = sorted(c for c in seg.cols
                              if seg.cols[c].get('mode') != 4 and P.not_positional(seg, c))
                for a, b in itertools.combinations(elig, 2):
                    t0 = time.perf_counter()
                    ok = GW.build(seg, [a, b])
                    dt = time.perf_counter() - t0
                    ent = {'table': t, 'pair': (a, b), 'build_s': round(dt, 2)}
                    if not ok:
                        ent.update(status='empty', bytes=0)
                    else:
                        built = GW._load(seg, [a, b])
                        nb = GW.structure_nbytes(built)
                        if tot_bytes + nb > budget_bytes:
                            GW.cache_pop(seg, [a, b])   # classified, deliberately not resident
                            ent.update(status='deferred_budget', bytes=nb, nheavy=int(built[3]))
                        else:
                            tot_bytes += nb
                            ent.update(status='built', bytes=nb, nheavy=int(built[3]),
                                       ones=int(built[5].size))
                    manifest.append(ent)
                    if verbose:
                        print("prewarm %-44s %-16s %10.1f KB  nheavy=%-10s %6.2fs" % (
                            "%s.%s x %s" % (t, a, b), ent['status'], ent['bytes'] / 1024,
                            ent.get('nheavy', '-'), dt), flush=True)
        return manifest, {'pairs': len(manifest), 'built_bytes': tot_bytes,
                          'wall_s': round(time.perf_counter() - t_all, 1)}

    def _table_in(self, tree):
        f = tree.find(E.From)
        if f is None: raise NotImplementedError("SELECT without FROM")
        return f.this.name

    def run_columnar(self, sql):
        """Native columnar result for a join GROUP BY: ({colname: ndarray}, colnames) with no Python
        row-tuple assembly (the per-row cost that dominated db.run at high cardinality). Falls back to
        transposing the row result for shapes the fast path can't emit columnar (non-join, HAVING/
        ORDER/LIMIT)."""
        tree = _parse_sql_cached(sql)
        names = None
        if isinstance(tree, E.Select) and tree.args.get('joins'):
            res, names = wdb_join.join_query(self, sql, columnar=True)
            if isinstance(res, dict):
                return res, names
            rows = res
        else:
            out = self.run(sql)
            rows, names = out if isinstance(out, tuple) else (out, None)
        cols = {n: [r[i] for r in rows] for i, n in enumerate(names)} if names else {}
        return cols, names

    def run(self, sql, escalate=None):
        esc = self.escalate if escalate is None else escalate
        tree = _parse_sql_cached(sql)
        result = commands.route(self, sql, tree)
        if result is not commands._NOT_A_COMMAND:
            return result
        if isinstance(tree, E.Select):
            if tree.args.get('joins'):
                _rw = wdb_join.denorm_rewrite(self, tree)        # join that groups by a denormalised parent
                if _rw is None:                                  # column -> single-table cube read; else gather
                    return wdb_join.join_query(self, sql)
                sql = _rw; tree = _parse_sql_cached(_rw)         # fall through to the single-table path
            name = self._table_in(tree)
            phys = self.cat.phys_map(name)
            cmap = {c: phys.get(c, c) for c in self.cat.column_names(name)}  # complete logical->physical
            paths = self.cat.segment_paths(name)
            segs = [self.open_segment(p, name) for p in paths]
            hp = wdb_dml.hot_path(self.cat, name)
            hot = hp if os.path.exists(hp) else None
            if hot is None and len(segs) == 1:
                ctx = read_methods.ReadContext(self, name, segs[0], paths[0], tree, cmap, sql, esc)
                return controller.route_single_segment(ctx)
            if hot is None and not segs:
                raise ValueError(f"table {name!r} has no data yet")
            if len(segs) == 1 and hot is not None:
                # one cold segment + a hot buffer: try the gridwalk base + maintenance fast path for
                # 2-key COUNT(*) top-K; it declines (None) for any other shape or a new dict value.
                live = wdb_gridwalk_live.try_live(segs[0], hot, tree, cmap)
                if live is not None:
                    return live
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

    def gd_sidecar(self, segment_path, group_col, target_col):
        """Cached group-distinct sidecar (the materialized COUNT(DISTINCT) answer). The .npz is immutable
        once built, so np.load runs once per process, not per query -- this is what makes the served read
        a memory hit rather than a file read. Reloads only if the file changed. None if no sidecar."""
        sp = wdb_gdsidecar.sidecar_path(segment_path, group_col, target_col)
        if not os.path.exists(sp):
            return None
        st = os.stat(sp); key = (st.st_mtime_ns, st.st_size)
        ck = (segment_path, group_col, target_col); hit = self._gd_cache.get(ck)
        if hit is None or hit[0] != key:
            s = wdb_gdsidecar.load(segment_path, group_col, target_col)
            self._gd_cache[ck] = (key, s); return s
        return hit[1]

    def materialize_gd(self, table, group_col, target_col):
        """Build + persist the group-distinct sidecar for table.(group_col, target_col) and register it.
        The default covers ALL groups -- the count comes free from the dictionary. v1: single segment."""
        paths = self.cat.segment_paths(table)
        if len(paths) != 1:
            raise NotImplementedError("materialize_gd: single-segment tables only (v1)")
        phys = self.cat.phys_map(table)
        gcol = phys.get(group_col, group_col); tcol = phys.get(target_col, target_col)
        seg = self.open_segment(paths[0], table)
        s = wdb_gdsidecar.build(seg, gcol, tcol)
        if s is None:
            raise ValueError(f"{table}.{group_col}/{target_col} is not the value-identity "
                             f"COUNT(DISTINCT) shape the sidecar supports")
        wdb_gdsidecar.save(paths[0], s)
        self.cat.set_gd_materialized(table, gcol, tcol, excluded=[])
        return s['meta']

    def gd_trim(self, table, group_col, target_col, exclude_values):
        """Trim the materialized view: leave the given GROUP values to the live walk (pass [] to un-trim
        for a full sidecar serve). Values are mapped to group codes via the segment dictionary."""
        paths = self.cat.segment_paths(table)
        phys = self.cat.phys_map(table)
        gcol = phys.get(group_col, group_col); tcol = phys.get(target_col, target_col)
        seg = self.open_segment(paths[0], table)
        codes = wdb_gdsidecar.codes_for_values(seg, gcol, exclude_values)
        self.cat.gd_set_trim(table, gcol, tcol, codes)
        return codes

    def gd_inspect(self, table, group_col, target_col):
        """The materialized view a human eyeballs to decide what to trim: (group_value, distinct_count,
        is_trimmed) rows, descending by count. Reads the sidecar; no walk."""
        paths = self.cat.segment_paths(table)
        phys = self.cat.phys_map(table)
        gcol = phys.get(group_col, group_col); tcol = phys.get(target_col, target_col)
        seg = self.open_segment(paths[0], table)
        entry = self.cat.gd_entry(table, gcol, tcol)
        excluded = (entry or {}).get('excluded') or []
        return wdb_gdsidecar.inspect(seg, paths[0], gcol, tcol, excluded=excluded)

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
