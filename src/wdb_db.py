import re
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
    """Parse SQL -> AST. Uncached by law (wdb_qmem): the memo was removed with the
    tree-mutation bug, and query-keyed caches grow with history, not with the file."""
    return sqlglot.parse_one(sql, read='duckdb')


_PW_SEGS = {}   # per-WORKER-process: (dbdir, seg_path) -> Segment (opened once, reused across tasks)

def _prewarm_worker(task):
    """Build one pair's structure in a prewarm worker process. The Segment is opened once per
    (worker, segment) and reused; the built tuple pickles back to the parent. Build transients
    (the np.unique sort buffers) die WITH the pool -- so the long-lived parent's heap holds only
    the structures, not ~30 GB of allocator retention (measured on the serial path)."""
    dbdir, table, seg_path, a, b = task
    import time
    import numpy as np
    import wdb_gridwalk as GW
    key = (dbdir, seg_path)
    seg = _PW_SEGS.get(key)
    if seg is None:
        seg = Database.open(dbdir).open_segment(seg_path, table)
        _PW_SEGS[key] = seg
    t0 = time.perf_counter()
    if b is None:                        # column-nd task: the column's true distinct count (FD input)
        return int(np.unique(seg._raw_codes(a)).size), 0, round(time.perf_counter() - t0, 2)
    if b == '':                          # block-stats task: per-block aggregates for the disk-only read
        import wdb_blockstats
        return wdb_blockstats.build(seg, a), 0, round(time.perf_counter() - t0, 2)
    built = GW._build(seg, [a, b])
    nb = GW.structure_nbytes(built) if built is not None else 0
    if sum(v.nbytes for v in seg._codes.values()) > (3 << 29):   # hard 1.5 GB bound per worker:
        seg._codes.clear()          # workers are transient build vessels; unbounded per-process code
    return built, nb, round(time.perf_counter() - t0, 2)          # caches OOM cgroup-limited boxes

def _int_emission(ctx, res):
    """INTEGER EMISSION for the single-table doors (the join engine's law):
    SUM over an integer column emits an integer. Every door that accumulates
    in float64 gets typed here, once, instead of each emitter separately."""
    try:
        import sqlglot.expressions as E
        tree = ctx.tree
        rows = res[0] if isinstance(res, tuple) else res
        if not rows or not isinstance(rows, list): return res
        idx = []
        for i, p in enumerate(tree.expressions):
            nd = p.this if isinstance(p, E.Alias) else p
            if isinstance(nd, E.Sum) and isinstance(nd.this, E.Column):
                pc = (ctx.cmap or {}).get(nd.this.name, nd.this.name)
                if ctx.seg.cols.get(pc, {}).get('dt') == 0 and not ctx.seg.cols[pc].get('has_null'):
                    idx.append(i)
        if not idx or len(rows[0]) != len(tree.expressions): return res
        out = []
        for r in rows:
            r = list(r)
            for i in idx:
                v = r[i]
                if isinstance(v, float) and v == int(v): r[i] = int(v)
            out.append(tuple(r))
        return (out, res[1]) if isinstance(res, tuple) else out
    except Exception:
        return res


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
    def open(cls, dbdir):
        db = cls(Catalog.open(dbdir))
        try:
            import wdb_shelves
            wdb_shelves.replay(db)           # Jackson's eager-shelf law:
        except Exception:                    # every recorded shelf is born
            pass                             # at launch, never on luck
        return db

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

    def _fd_postpass(self, manifest, colnd, segN, GW, verbose=True):
        """Exhaustive FD classification over the completed survey -- FREE, because every pair's full
        distinct count (nd) was computed by its build and every column's nd came back from the pool.
        FD knowledge (parent->child where nd(pair) == nd(parent)) is recorded for every proven pair;
        TWIN deferral additionally requires near-bijection (child/parent nd ratio >= 0.90) AND a fat
        structure -- an FD with heavy collapse (a near-key like WatchID, a hierarchy like
        EventTime->EventDate) does NOT make the parent-side slice substitutable, and deferring tiny
        exception-defined slices saves nothing while costing a first-touch rebuild. Deferred twins
        rebuild-on-touch exactly, so nothing becomes unanswerable. Returns bytes freed."""
        fd = {}; parents = set(); twins = {}
        for ent in manifest:
            ndp = ent.get('nd')
            if not ndp:
                continue
            a, b = ent['pair']; p = ent['path']
            na = colnd.get((p, a)); nb2 = colnd.get((p, b))
            if ndp == na:
                par, ch = a, b          # a determines b (bijection also lands here: lexicographic parent)
            elif ndp == nb2:
                par, ch = b, a
            else:
                continue
            if ch in fd or ch in parents or par in fd:
                continue                # no chains/cycles in v1
            fd[ch] = par; parents.add(par)
            ratio = colnd[(p, ch)] / max(1, colnd[(p, par)])
            if verbose:
                print("prewarm FD PROVEN: %s -> %s (nd(pair)=%d == nd(%s), child/parent nd ratio %.3f)"
                      % (par, ch, ndp, par, ratio), flush=True)
            # TWINS require near-bijection: an FD with heavy collapse (near-key parents like WatchID,
            # hierarchies like EventTime->EventDate) does NOT make (X,parent) substitutable by
            # (X,child) -- those are different matrices and different query shapes. Only a ~1:1
            # relabeling makes the parent-side copy redundant.
            if ratio >= 0.90:
                twins[ch] = par
        freed = 0
        for ent in manifest:
            if ent['status'] != 'built' or ent['bytes'] < (8 << 20):
                continue                # deferring small structures saves nothing, costs first-touch
            a, b = ent['pair']; p = ent['path']
            for ch, par in twins.items():
                if par in (a, b) and ch not in (a, b):
                    GW._CACHE.pop((p, (a, b), segN[p]), None)
                    x = a if b == par else b
                    ent['status'] = 'twin_deferred'
                    ent['via'] = tuple(sorted((x, ch)))
                    freed += ent['bytes']
                    if verbose:
                        print("prewarm twin-deferred %-32s (twin of %s x %s)  -%.1f MB" % (
                            "%s x %s" % (a, b), *ent['via'], ent['bytes'] / 1048576), flush=True)
                    break
        if fd and verbose:
            print("prewarm FD post-pass freed %.1f MB" % (freed / 1048576), flush=True)
        return freed

    def prewarm(self, table=None, budget_bytes=4 << 30, workers=None, fd_derive=True, verbose=True):
        """Eager launch classification + build: walk EVERY eligible column pair of every table and
        materialize its gridwalk structure into RAM, under a structures budget. Exhaustive by
        construction -- every pair ends the loop CLASSIFIED in the returned manifest (built, or
        deferred with the reason), so no pair-shaped query can fall through unclassified. RAM-only
        by design (disk stays compressed); the one-time build cost moves from first-query to launch,
        which also makes the total footprint VISIBLE here instead of a mid-query surprise.

        workers=N builds pairs across N processes (each peaks ~2-3 GB transient on a 100M segment;
        size N to RAM). Budget is applied to results IN TASK ORDER, so built/deferred decisions are
        deterministic and identical to the serial pass. Returns (manifest, totals)."""
        import itertools, time
        import wdb_gridwalk as GW
        import wdb_policies as P
        t_all = time.perf_counter()
        tasks = []; col_tasks = []; stat_tasks = []; segN = {}; manifest = []; tot_bytes = 0; colnd = {}
        for t in ([table] if table else self.cat.list_tables()):
            for path in self.cat.segment_paths(t):
                seg = self.open_segment(path, t)
                segN[path] = int(seg.N)
                elig = sorted(c for c in seg.cols
                              if seg.cols[c].get('mode') != 4 and P.not_positional(seg, c))
                col_tasks.extend((t, path, c) for c in elig)
                import wdb_blockstats
                stat_tasks.extend((t, path, c) for c in sorted(seg.cols)
                                  if wdb_blockstats.eligible(seg, c))
                tasks.extend((t, path, a, b) for a, b in itertools.combinations(elig, 2))

        def _apply(outs_iter):
            nonlocal tot_bytes
            for (tt, p, a, b), (built, nb, dt) in zip(tasks, outs_iter):
                ent = {'table': tt, 'pair': (a, b), 'path': p, 'build_s': dt}
                if built is None:
                    ent.update(status='empty', bytes=0)
                elif tot_bytes + nb > budget_bytes:
                    ent.update(status='deferred_budget', bytes=nb, nheavy=int(built[3]), nd=int(built[7]))
                else:                                   # resident: insert into gridwalk's RAM cache
                    GW._CACHE[(p, (a, b), segN[p])] = built
                    tot_bytes += nb
                    ent.update(status='built', bytes=nb, nheavy=int(built[3]), ones=int(built[5].size), nd=int(built[7]))
                manifest.append(ent)
                if verbose:
                    print("prewarm %-44s %-16s %10.1f KB  nheavy=%-10s %6.2fs" % (
                        "%s.%s x %s" % (tt, a, b), ent['status'], ent['bytes'] / 1024,
                        ent.get('nheavy', '-'), dt), flush=True)

        if workers and int(workers) > 1:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as _mp_pw
            cargs = [(self.cat.dbdir, tt, p, c, None) for (tt, p, c) in col_tasks]
            args = [(self.cat.dbdir, tt, p, a, b) for (tt, p, a, b) in tasks]
            sargs = [(self.cat.dbdir, tt, p, c, '') for (tt, p, c) in stat_tasks]
            with ProcessPoolExecutor(max_workers=int(workers),
                                      mp_context=_mp_pw.get_context('spawn')) as ex:
                for (tt, p, c), (v, _z, _dt) in zip(col_tasks, ex.map(_prewarm_worker, cargs, chunksize=1)):
                    colnd[(p, c)] = int(v)
                import wdb_blockstats
                for (tt, p, c), (st, _z, _dt) in zip(stat_tasks, ex.map(_prewarm_worker, sargs, chunksize=1)):
                    wdb_blockstats.install(p, segN[p], c, st)
                _apply(ex.map(_prewarm_worker, args, chunksize=1))   # yields in task order
        else:
            import numpy as _np
            import wdb_blockstats
            for (tt, p, c) in col_tasks:
                seg = self.open_segment(p, tt)
                colnd[(p, c)] = int(_np.unique(seg._raw_codes(c)).size)
            for (tt, p, c) in stat_tasks:
                wdb_blockstats.build(self.open_segment(p, tt), c)
            def _serial():
                for (tt, p, a, b) in tasks:
                    seg = self.open_segment(p, tt)
                    t0 = time.perf_counter()
                    built = GW._build(seg, [a, b])
                    nb = GW.structure_nbytes(built) if built is not None else 0
                    yield built, nb, round(time.perf_counter() - t0, 2)
            _apply(_serial())
        freed = self._fd_postpass(manifest, colnd, segN, GW, verbose) if fd_derive else 0
        tot_bytes -= freed
        return manifest, {'pairs': len(manifest), 'built_bytes': tot_bytes,
                          'fd_freed_bytes': freed,
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
        """Depth-guarded: recursive runs (subquery rewrites, join sub-queries) share
        memory within one outer query; at depth 0 wdb_qmem.flush forgets everything
        data-derived. A query leaves the engine as if it was never there."""
        import wdb_qmem, os as _os9
        _bill9 = _os9.environ.get('WDB_JOIN_BILL')
        if _bill9:
            import time as _t9
            _r0 = _t9.perf_counter()
        self._qdepth = getattr(self, '_qdepth', 0) + 1
        try:
            r9 = self._run_impl(sql, escalate)
            if _bill9 and self._qdepth == 1:
                print('RUN BILL: impl=%.0fms' % ((_t9.perf_counter() - _r0) * 1000), flush=True)
            return r9
        finally:
            self._qdepth -= 1
            if self._qdepth == 0:
                if _bill9:
                    _f0 = _t9.perf_counter()
                wdb_qmem.flush(self)
                if _bill9:
                    print('RUN BILL: qmem-flush=%.0fms' % ((_t9.perf_counter() - _f0) * 1000), flush=True)

    def _run_impl(self, sql, escalate=None):
        esc = self.escalate if escalate is None else escalate
        tree = _parse_sql_cached(sql)
        result = commands.route(self, sql, tree)
        if result is not commands._NOT_A_COMMAND:
            return result
        if isinstance(tree, E.Select):
            import wdb_cte
            import wdb_literal
            def _is_stored9(nm):
                try:
                    self.cat.get_table(nm); return True
                except Exception:
                    return False
            if wdb_literal.references_only_literals(tree, _is_stored9):
                return wdb_literal.run_literal(tree)       # LITERAL RELATIONS: no stored table touched
            if wdb_cte.has_cte(tree):
                tree = wdb_cte.rewrite(tree)      # flatten views before anything resolves
                sql = tree.sql()
            if tree.args.get('joins'):
                import os as _os9
                if _os9.environ.get('WDB_JOIN_BILL'):
                    import time as _t9
                    _d0 = _t9.perf_counter()
                    _rw = wdb_join.denorm_rewrite(self, tree)
                    print('RUN BILL: denorm-rewrite=%.0fms' % ((_t9.perf_counter() - _d0) * 1000), flush=True)
                else:
                    _rw = wdb_join.denorm_rewrite(self, tree)    # join that groups by a denormalised parent
                if _rw is None:                                  # column -> single-table cube read; else gather
                    return wdb_join.join_query(self, sql)
                sql = _rw; tree = _parse_sql_cached(_rw)         # fall through to the single-table path
            import wdb_subquery
            if tree.find(sqlglot.exp.All) is not None or tree.find(sqlglot.exp.Any) is not None:
                _t9 = wdb_subquery.rewrite_any_all(tree.copy())
                if _t9.sql() != sql:
                    return self._run_impl(_t9.sql(), escalate)     # ANY/ALL -> MIN/MAX scalar or IN
            if any(p.find(sqlglot.exp.Subquery) is not None for p in tree.expressions):
                _t9 = wdb_subquery.substitute_select_scalars(self, tree.copy())
                if _t9.sql() != sql:
                    return self._run_impl(_t9.sql(), escalate)     # SELECT-list scalars -> literals
            _hr9 = wdb_join.hidden_rewrite(tree)
            if _hr9 is not None:
                _sql9, _nh9 = _hr9
                _out9 = self._run_impl(_sql9, escalate)
                if _nh9:
                    _rows9, _hdr9 = _out9 if isinstance(_out9, tuple) else (_out9, None)
                    _rows9 = [r[:len(r) - _nh9] for r in _rows9]
                    if _hdr9: _hdr9 = _hdr9[:len(_hdr9) - _nh9]
                    return (_rows9, _hdr9) if _hdr9 is not None else _rows9
                return _out9
            _rw9 = wdb_join.qualify_rewrite(tree) or wdb_join.distinct_on_rewrite(tree)
            if _rw9 is not None:
                return self._run_impl(_rw9, escalate)      # QUALIFY / DISTINCT ON -> the top-k door's shape
            frm9 = tree.args.get('from') or tree.args.get('from_')
            if frm9 is not None and frm9.this.__class__.__name__ == 'Subquery':
                return wdb_join.join_query(self, sql)     # THE FROM DOOR lives there
            if wdb_join.has_agg_arith(tree):
                return wdb_join.join_query(self, sql)     # AGGREGATE ARITHMETIC lives there
            if tree.find(sqlglot.exp.Window) is not None:
                return wdb_join.join_query(self, sql)     # THE WINDOW DOOR lives there
            if wdb_join.has_expr_group(tree):
                return wdb_join.join_query(self, sql)     # EXPRESSION GROUP KEYS ride the dict there
            if frm9 is not None and frm9.this.__class__.__name__ == 'Values':
                raise NotImplementedError('VALUES as a table source is not supported')
            name = self._table_in(tree)
            phys = self.cat.phys_map(name)
            import wdb_groupsets
            if wdb_groupsets.has_grouping(tree):
                return wdb_groupsets.execute(self, tree)   # each set is a plain fast GROUP BY
            import wdb_subquery
            if wdb_subquery.has_subquery(tree):
                tree = tree.copy()                        # never mutate the parse cache:
                                                          # repeat runs must see pristine trees
                dec = wdb_subquery._try_window_decorrelate(self, tree)
                if dec is not None:                       # self-join becomes one placement
                    t2, drop = dec
                    out = self.run(t2.sql())
                    rows, hdr = out if isinstance(out, tuple) else (out, None)
                    if hdr and drop < len(hdr) and str(hdr[drop]).startswith('__corr'):
                        rows = [r[:drop] + r[drop + 1:] for r in rows]   # legacy paths that
                        hdr = hdr[:drop] + hdr[drop + 1:]                # still emit the helper
                    return rows, hdr
                tree = wdb_subquery.rewrite(self, tree)   # in-tree: no sql-text roundtrip
                sql = tree.sql()                          # for reads that consume raw sql
            cmap = {c: phys.get(c, c) for c in self.cat.column_names(name)}  # complete logical->physical
            paths = self.cat.segment_paths(name)
            segs = [self.open_segment(p, name) for p in paths]
            hp = wdb_dml.hot_path(self.cat, name)
            hot = hp if os.path.exists(hp) else None
            if hot is None and len(segs) == 1:
                ctx = read_methods.ReadContext(self, name, segs[0], paths[0], tree, cmap, sql, esc)
                import time as _time
                import wdb_ledger
                controller._SERVED[0] = None
                wdb_ledger.reset_stages()
                _t0 = _time.perf_counter()
                _res = _int_emission(ctx, controller.route_single_segment(ctx))
                _ms = (_time.perf_counter() - _t0) * 1000
                _rows = _res[0] if isinstance(_res, tuple) else _res
                wdb_ledger.log(ctx.seg, sql, controller._SERVED[0] or '?', _ms,
                               n_rows=len(_rows) if hasattr(_rows, '__len__') else None)
                return _res
            if hot is None and not segs:
                raise ValueError(f"table {name!r} has no data yet")
            if len(segs) == 1 and hot is not None:
                # one cold segment + a hot buffer: try the gridwalk base + maintenance fast path for
                # 2-key COUNT(*) top-K; it declines (None) for any other shape or a new dict value.
                live = wdb_gridwalk_live.try_live(segs[0], hot, tree, cmap)
                if live is not None:
                    return live
            return wdb_merge.merge_query(segs, hot, sql, col_map=cmap)
        import wdb_setops
        if wdb_setops.is_setop(tree):
            return wdb_setops.execute(self, tree, esc)
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
            if s is not None:
                # THE ID-SPACE BIRTHMARK: reject a shelf built in another id space
                try:
                    import wdb_groupdistinct as _gd
                    tbl = next((t for t in self.cat.data['tables']
                                if segment_path in self.cat.segment_paths(t)), None)
                    seg = self.open_segment(segment_path, tbl) if tbl else None
                    want = _gd.idspace_sig(seg, group_col) if seg is not None else None
                    if want is None or s.get('meta', {}).get('idspace') != want:
                        print('GD-SIDECAR: id-space birthmark mismatch (%s vs %s) -- shelf ignored: %s'
                              % (s.get('meta', {}).get('idspace'), want, sp), flush=True)
                        s = None
                except Exception as _e:
                    print('GD-SIDECAR: birthmark check failed (%s) -- shelf ignored' % _e, flush=True)
                    s = None
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
        import controller
        controller.plan_epoch_bump()
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
