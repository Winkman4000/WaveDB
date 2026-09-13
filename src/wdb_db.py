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
        self._cat_stamp = None
        self._seg_cache = {}   # path -> ((mtime_ns,size), Segment): immutable .wdb base, reused across queries
        self._ptr_cache = {}   # (child_seg_path, fk_col) -> ((mtime_ns,size), int64 ptr array)
        self._gd_cache = {}    # (seg_path, group, target) -> ((mtime_ns,size), sidecar dict): load .npz once
        # Operator-selected execution intent. escalate=False (default) optimizes for THROUGHPUT
        # (shared DB, many concurrent 1-thread workers): engages the BSI filter-index, which reads
        # fewer bytes per query. escalate=True optimizes a single query for LATENCY: it skips BSI
        # and uses the fully parallel fused scan, which wins when one query owns all cores. Both
        # paths return identical results; the choice is the caller's, per-run or as this default.
        self.escalate = False
        self._cat_stamp = self._catalog_stamp()

    @classmethod
    def create(cls, dbdir): return cls(Catalog.create(dbdir))

    def _catalog_stamp(self):
        try:
            st = os.stat(os.path.join(self.cat.dbdir, 'catalog.json'))
            return (st.st_mtime_ns, st.st_size)
        except Exception:
            return None

    def refresh(self):
        """THE CATALOG IS THE TRUTH: a long-lived engine re-reads it when another process has
        changed it (a compaction, a flush, a DDL) -- the server kept serving a segment that
        a writer had replaced and answered 500 to 106,245 queries in a row (2026-09-14).
        Cheap: one stat per query; a reload only when the stamp moved."""
        st = self._catalog_stamp()
        if st != self._cat_stamp:
            self.cat = Catalog.open(self.cat.dbdir)
            self._cat_stamp = st
            self._seg_cache.clear(); self._ptr_cache.clear(); self._gd_cache.clear()
            if hasattr(self, '_union_cache'): self._union_cache.clear()
            try:
                import wdb_sqlcache
                wdb_sqlcache.clear() if hasattr(wdb_sqlcache, 'clear') else None
            except Exception:
                pass
            return True
        return False

    @staticmethod
    def recover(dbdir, verbose=False):
        """RECOVERY ON OPEN: the counterpart of the rename law. Every write path writes a
        '.partial' and renames, so a crash leaves at most a partial file (swept here), a
        segment the catalog never learned about (a compaction that died after writing its
        segment: reported, never deleted -- vacuum can), or a catalog that names a segment
        that is gone (refused loudly: that is data loss, and silence would hide it)."""
        report = {'swept': [], 'unknown_segments': [], 'missing_segments': []}
        try:
            names = os.listdir(dbdir)
        except FileNotFoundError:
            return report
        for n in names:
            if n.endswith('.partial') or n.endswith('.tmp') or n.endswith('.tmp.npy') or n.endswith('.tmp.npz'):
                try: os.remove(os.path.join(dbdir, n)); report['swept'].append(n)
                except OSError: pass
        try:
            cat = Catalog.open(dbdir)
            known = set()
            for t in cat.list_tables():
                for sfile, spath in zip(cat.get_table(t)['segments'], cat.segment_paths(t)):
                    known.add(sfile)
                    if not os.path.exists(spath):
                        report['missing_segments'].append((t, sfile))
            for n in names:
                if n.endswith('.wdb') and n not in known and not os.path.islink(os.path.join(dbdir, n)):
                    report['unknown_segments'].append(n)
        except Exception:
            pass
        if verbose or report['swept'] or report['unknown_segments'] or report['missing_segments']:
            if report['swept']: print('wdb recover: swept %d partial file(s): %s' % (len(report['swept']), ', '.join(report['swept'][:4])), flush=True)
            if report['unknown_segments']: print('wdb recover: %d segment file(s) the catalog does not name (a crash before the catalog saved?): %s -- `wdb vacuum` removes them' % (len(report['unknown_segments']), ', '.join(report['unknown_segments'][:4])), flush=True)
        if report['missing_segments']:
            raise RuntimeError('wdb recover: the catalog names segment(s) that are not on disk: %s -- refusing to open silently' % report['missing_segments'][:4])
        return report

    @classmethod
    def open(cls, dbdir):
        cls.recover(dbdir)
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

    _SPECIALISED = ('sorted_proj', 'cluster_topk', 'value_topk', 'firstsorted', 'firstk', 'distinctlim',
                    'affinegroup', 'affinesum', 'window', 'heavypair', 'sumtopk', 'gridwalk', 'smallk')

    def _new_door_shape(self, tree):
        import wdb_join
        E9 = sqlglot.exp
        if tree.args.get('joins'): return False
        if tree.find(E9.Window) is not None: return True
        if wdb_join.has_agg_arith(tree) or wdb_join.has_expr_group(tree): return True
        return (tree.args.get('order') is not None and tree.args.get('limit') is not None
                and tree.args.get('group') is None and tree.find(E9.AggFunc) is None)

    def _specialised_claims(self, ctx):
        """Does one of the controller's SPECIALISED fast doors detect this query?"""
        for nm in self._SPECIALISED:
            rd = getattr(read_methods, nm, None)
            if rd is None: continue
            try:
                if rd.detect(ctx) is not None: return True
            except Exception:
                continue
        return False

    def stats(self):
        """Why is RAM high? What sidecars exist? -- the shelf and the registry, printed."""
        import wdb_shelf, wdb_sidecar
        wdb_shelf.SHELF.stats()
        try:
            wdb_sidecar.stats(self.cat.root if hasattr(self.cat, 'root') else os.path.dirname(self.cat.segment_paths(next(iter(self.cat.data['tables'])))[0]))
        except Exception as e:
            print('SIDECARS: %s' % e)

    def _null_fold(self, tree):
        """THE NULL FOLD: col IS [NOT] NULL on a column WITHOUT nulls is a boolean
        literal (four consumers used to decode the 100M-row column to learn it);
        the WHERE simplifies; a WHERE folded to FALSE answers by the ZERO-SURVIVOR
        law. Returns (tree, 'false'|'changed'|None)."""
        E9 = sqlglot.exp
        w = tree.args.get('where')
        if w is None or tree.args.get('joins') or w.this.find(E9.Is) is None: return tree, None
        try:
            name = self._table_in(tree)
            seg = self.open_segment(self.cat.segment_paths(name)[0], name)
            pm = self.cat.phys_map(name)
        except Exception:
            return tree, None
        t = tree.copy(); changed = False
        for nd in list(t.args['where'].this.find_all(E9.Is)):
            if not (isinstance(nd.this, E9.Column) and isinstance(nd.expression, E9.Null)): continue
            c = seg.cols.get(pm.get(nd.this.name, nd.this.name))
            if c is None or c.get('has_null'): continue
            nd.replace(E9.Boolean(this=False)); changed = True
        if not changed: return tree, None
        def simp(n):
            if isinstance(n, E9.Paren): return simp(n.this)
            if isinstance(n, E9.Not):
                x = simp(n.this)
                return E9.Boolean(this=not x.this) if isinstance(x, E9.Boolean) else E9.Not(this=x)
            if isinstance(n, E9.And):
                a, b = simp(n.this), simp(n.expression)
                if isinstance(a, E9.Boolean): return b if a.this else a
                if isinstance(b, E9.Boolean): return a if b.this else b
                return E9.And(this=a, expression=b)
            if isinstance(n, E9.Or):
                a, b = simp(n.this), simp(n.expression)
                if isinstance(a, E9.Boolean): return a if a.this else b
                if isinstance(b, E9.Boolean): return b if b.this else a
                return E9.Or(this=a, expression=b)
            return n
        cond = simp(t.args['where'].this)
        if isinstance(cond, E9.Boolean):
            if cond.this:
                t.set('where', None); return t, 'changed'
            return t, 'false'
        t.set('where', E9.Where(this=cond)); return t, 'changed'

    def _zero_survivors(self, tree):
        """The ZERO-SURVIVOR law: a scalar aggregate over no rows is ONE row (COUNT 0, else NULL); anything else is empty."""
        E9 = sqlglot.exp
        names = [wdb_sql._alias(p) for p in tree.expressions]
        if tree.args.get('group') is None and tree.expressions and all((p.this if isinstance(p, E9.Alias) else p).find(E9.AggFunc) is not None or isinstance((p.this if isinstance(p, E9.Alias) else p), E9.AggFunc) for p in tree.expressions):
            row = []
            for p in tree.expressions:
                nd = p.this if isinstance(p, E9.Alias) else p
                row.append(0 if isinstance(nd, E9.Count) else None)
            return [tuple(row)], names
        return [], names

    @staticmethod
    def _segments_clean(segs):
        """the union is for CLEAN segments: tombstones, overrides (UPDATE) or synthetic columns on
        any segment keep the table on the merge path that knows them (the suite's UPDATE tests
        ran past V with a union that could not see the overrides)"""
        import wdb_override
        for sg in segs:
            try:
                if sg.presence_mask() is not None: return False
                if wdb_override.load(sg.path): return False
                if any(c.get('mode') == 6 for c in sg.cols.values()): return False
            except Exception:
                return False
        return True

    def _union(self, name, segs, paths):
        """the union view for a multi-segment table, cached per catalog stamp (refresh() clears)"""
        import wdb_union
        cache = getattr(self, '_union_cache', None)
        if cache is None: cache = self._union_cache = {}
        key = (name, tuple(paths))
        u = cache.get(key)
        if u is None:
            u = cache[key] = wdb_union.SegmentUnion(segs, paths)
        return u

    def _new_doors(self, tree, sql):
        """THE PRECEDENCE LAW: the controller's doors serve first; the doors born in
        the scope stage (aggregate arithmetic, expression group keys, windows, top-k
        rows) only answer what the controller declined BY NAME. Returns a result or None."""
        import wdb_join
        E9 = sqlglot.exp
        if tree.args.get('order') is not None and tree.args.get('limit') is not None and tree.args.get('group') is None \
                and not tree.args.get('joins') and tree.find(E9.Window) is None and tree.find(E9.AggFunc) is None:
            try:
                r = wdb_join._topk_rows_door(self, tree)
                if r is not None: return r
            except wdb_join._FastUnsupported:
                pass
        if tree.find(E9.Window) is not None or wdb_join.has_agg_arith(tree) or wdb_join.has_expr_group(tree):
            return wdb_join.join_query(self, sql)
        return None

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

    def explain(self, sql, run=True):
        """EXPLAIN: the plan you can read. Runs the query with the route trace and the
        bills switched on and returns a report -- which door served (the controller's
        _SERVED or the join engine's ROUTE), each stage's bill, rows out, wall time,
        peak RSS, and what the shelf and governor did. run=False reports routing only
        (the shapes the doors would claim) without executing."""
        import os as _os, io, contextlib, time as _t, resource
        import controller, wdb_shelf, wdb_govern
        keys = ('WDB_ROUTE_DEBUG', 'WDB_JOIN_BILL', 'WDB_SEMI_BILL', 'WDB_CASCADE_DEBUG')
        prev = {k: _os.environ.get(k) for k in keys}
        for k in keys: _os.environ[k] = '1'
        buf = io.StringIO()
        controller._SERVED[0] = None
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ev0 = wdb_shelf.SHELF.evictions; rf0 = wdb_shelf.SHELF.refusals
        t0 = _t.perf_counter(); rows = None; err = None
        try:
            with contextlib.redirect_stdout(buf):
                if run:
                    r = self.run(sql)
                    rows = r[0] if isinstance(r, tuple) else r
                else:
                    tree = _parse_sql_cached(sql)
                    print('shape: %s' % ('multi-table' if tree.args.get('joins') else 'single-table'))
                    if not tree.args.get('joins'):
                        try:
                            name = self._table_in(tree)
                            import read_methods
                            phys = self.cat.phys_map(name)
                            cmap = {c: phys.get(c, c) for c in self.cat.column_names(name)}
                            paths = self.cat.segment_paths(name)
                            ctx = read_methods.ReadContext(self, name, self.open_segment(paths[0], name), paths[0], tree, cmap, sql, None)
                            claims = [nm for nm in self._SPECIALISED if getattr(read_methods, nm, None) is not None and (getattr(read_methods, nm).detect(ctx) is not None)]
                            print('specialised doors claiming: %s' % (claims or 'none'))
                            print('new-door shape: %s' % self._new_door_shape(tree))
                        except Exception as e:
                            print('routing probe: %s' % str(e)[:80])
        except Exception as e:
            err = e
        finally:
            for k, v in prev.items():
                if v is None: _os.environ.pop(k, None)
                else: _os.environ[k] = v
        wall = _t.perf_counter() - t0
        lines = [l for l in buf.getvalue().splitlines() if l.strip() and not l.startswith('Numba') and 'warn' not in l.lower()]
        served = controller._SERVED[0]
        route = [l for l in lines if l.startswith('ROUTE:')]
        bills = [l for l in lines if 'BILL' in l]
        other = [l for l in lines if l not in route and l not in bills][:12]
        out = []
        out.append('EXPLAIN %s' % sql.strip().replace(chr(10), ' ')[:160])
        out.append('  served by : %s' % (served or (route[-1].replace('ROUTE: ', 'join engine / ') if route else 'n/a')))
        if route: out.append('  route     : %s' % ' -> '.join(r.replace('ROUTE: ', '') for r in route))
        for b in bills: out.append('  bill      : %s' % b.replace('JOIN BILL: ', '').replace('RUN BILL: ', '').replace('SEMI BILL: ', 'semi: ')[:200])
        for o in other: out.append('  note      : %s' % o[:200])
        if run:
            out.append('  rows out  : %s' % (len(rows) if rows is not None else 'error: %s' % str(err)[:100]))
            out.append('  wall      : %.3fs' % wall)
            out.append('  peak RSS  : %.2f GB (this process)' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
            out.append('  shelf     : %d objects, %.2f GB of %.2f GB; evictions +%d, refusals +%d' % (
                len(wdb_shelf.SHELF._items), wdb_shelf.SHELF._bytes / 1e9, wdb_shelf.ceiling_bytes() / 1e9,
                wdb_shelf.SHELF.evictions - ev0, wdb_shelf.SHELF.refusals - rf0))
            out.append('  governor  : budget %.1f GB (WDB_WORK_MB)' % (wdb_govern.budget_bytes() / 1e9))
        report = chr(10).join(out)
        if err is not None and run:
            raise type(err)(str(err) + chr(10) + report) if isinstance(err, NotImplementedError) else err
        return report

    def stream(self, sql, block_rows=None):
        """STREAM-AND-CLEAN: yield the result in row blocks the caller pulls; each block
        is built from column arrays and freed behind. Uses the join engine's columnar
        path when it serves; otherwise the governed row path (which declines by name
        past the working-set budget)."""
        import wdb_govern, wdb_join, numpy as np
        br = block_rows or wdb_govern.block_rows()
        cols = None
        try:
            r = wdb_join.join_query(self, sql, columnar=True)
            if isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], dict):
                cols, _names9 = r
            elif isinstance(r, dict) and r:
                cols = r
        except Exception:
            cols = None
        if cols is not None:
            names = list(cols.keys()); arrs = [np.asarray(cols[n]) for n in names]
            n = int(arrs[0].shape[0]) if arrs else 0
            for lo in range(0, n, br):
                hi = min(n, lo + br)
                lists = [(a[lo:hi].tolist() if a.dtype != object else list(a[lo:hi])) for a in arrs]
                yield names, list(zip(*lists))
                del lists
            return
        rows, names = self.run(sql)
        for lo in range(0, len(rows), br):
            yield names, rows[lo:lo + br]

    def run(self, sql, escalate=None):
        """Depth-guarded: recursive runs (subquery rewrites, join sub-queries) share
        memory within one outer query; at depth 0 wdb_qmem.flush forgets everything
        data-derived. A query leaves the engine as if it was never there."""
        if getattr(self, '_qdepth', 0) == 0:
            self.refresh()                                   # another process may have rewritten the table
        import wdb_qmem, os as _os9
        _bill9 = _os9.environ.get('WDB_JOIN_BILL')
        if _bill9:
            import time as _t9
            _r0 = _t9.perf_counter()
        self._qdepth = getattr(self, '_qdepth', 0) + 1
        try:
            try:
                r9 = self._run_impl(sql, escalate)
            except (FileNotFoundError, OSError) as _fe9:
                # THE SWAP WINDOW: a query that began under the old catalog reached for a
                # segment a writer just replaced (compaction). The catalog is the truth --
                # refresh it and run once more; a second failure is a real error.
                if self._qdepth == 1 and self.refresh():
                    r9 = self._run_impl(sql, escalate)
                else:
                    raise
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
                import wdb_segparts
                if wdb_segparts.multi_segment_tables(self, tree):
                    _sp9 = wdb_segparts.execute(self, tree, sql)      # SEGMENT PARTIALS: one multi-segment table
                    if _sp9 is not None: return _sp9
                import wdb_semijoin
                if wdb_semijoin.shape_ok(tree) and not wdb_semijoin._needs_weights(tree):
                    try:
                        return wdb_semijoin.execute(self, tree)     # THE SEMI-JOIN FIXPOINT (MIN/MAX-only multi-joins): first
                    except wdb_semijoin._Decline:
                        pass
                if _os9.environ.get('WDB_JOIN_BILL'):
                    import time as _t9
                    _d0 = _t9.perf_counter()
                    _rw = wdb_join.denorm_rewrite(self, tree)
                    print('RUN BILL: denorm-rewrite=%.0fms' % ((_t9.perf_counter() - _d0) * 1000), flush=True)
                else:
                    _rw = wdb_join.denorm_rewrite(self, tree)    # join that groups by a denormalised parent
                if _rw is None:                                  # column -> single-table cube read; else gather
                    try:
                        return wdb_join.join_query(self, sql)
                    except NotImplementedError as _ne9:
                        # THE COUNTING FIXPOINT: last -- only when the road engine declines the join shape
                        if wdb_semijoin.shape_ok(tree) and wdb_semijoin._needs_weights(tree):
                            try:
                                return wdb_semijoin.execute(self, tree)
                            except wdb_semijoin._Decline:
                                pass
                        raise
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
            _tf9, _how9 = self._null_fold(tree)
            if _how9 == 'false':
                return self._zero_survivors(tree)
            if _how9 == 'changed':
                return self._run_impl(_tf9.sql(), escalate)
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
                # THE PRECEDENCE LAW, narrowed: a new-door shape goes to the new doors FIRST
                # unless a SPECIALISED fast door claims it (sorted projection, cluster/value
                # top-k, affine group, the old window door...) -- never yield to the general scan
                if self._new_door_shape(tree) and not self._specialised_claims(ctx):
                    _r9 = self._new_doors(tree, sql)
                    if _r9 is not None: return _r9
                try:
                    _res = _int_emission(ctx, controller.route_single_segment(ctx))
                except NotImplementedError as _ne9:
                    _r9 = self._new_doors(tree, sql)          # and LAST for whatever the controller declined by name
                    if _r9 is not None: return _r9
                    raise
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
            if len(segs) > 1 and hot is None and self._segments_clean(segs):
                # THE MERGED-DICTIONARY VIEW: K segments as one table through the general scan --
                # the path that serves windows, ORDER BY, set ops and every aggregate exactly.
                # The per-segment partial merge stays as the fallback when the union declines.
                try:
                    u = self._union(name, segs, paths)
                    # through the CONTROLLER: the fast doors that need only the union's API (code
                    # sets, dictionary counts, top-k by code...) serve; a door that touches what a
                    # union cannot give declines by name and the chain falls to the general scan
                    ctx = read_methods.ReadContext(self, name, u, u.path, tree, cmap, sql, esc)
                    controller._SERVED[0] = None
                    # THE PRECEDENCE LAW applies to the union exactly as to a single segment: the
                    # scope-stage doors (top-k rows, windows, expression keys, aggregate arithmetic)
                    # first unless a specialised door claims the shape, and last on a decline
                    if self._new_door_shape(tree) and not self._specialised_claims(ctx):
                        _r9 = self._new_doors(tree, sql)
                        if _r9 is not None:
                            controller._SERVED[0] = 'union:' + str(controller._SERVED[0] or 'new-door'); return _r9
                    try:
                        r = controller.route_single_segment(ctx)
                    except NotImplementedError:
                        _r9 = self._new_doors(tree, sql)
                        if _r9 is None: raise
                        controller._SERVED[0] = 'union:' + str(controller._SERVED[0] or 'new-door'); return _r9
                    if controller._SERVED[0] in (None, 'general_scan'): controller._SERVED[0] = 'union_scan'
                    else: controller._SERVED[0] = 'union:' + str(controller._SERVED[0])
                    return r
                except NotImplementedError:
                    pass
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
