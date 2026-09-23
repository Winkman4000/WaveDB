"""THE SEMI-JOIN FIXPOINT (the JOB shape): a multi-way equi-join projecting ONLY
MIN/MAX. Under MIN/MAX row multiplication is irrelevant -- a table's
contribution is the extreme over its rows that PARTICIPATE in the join.
Participation is a fixpoint: local filters seed each table's keep; every
equality edge prunes both sides to the other's surviving keys (dense-id
lookup tables, O(N)); iterate to stability; MIN/MAX per column over its
table's survivors (strings by the extreme present dictionary code)."""
import os
import numpy as np
import wdb_shelf
import wdb_coordroad
import sqlglot
from sqlglot import exp as E


class _Decline(Exception):
    pass


import threading as _threading
_INV_LOCK = _threading.Lock()          # guards _INV_LOCKS
_INV_LOCKS = {}                        # (segment path, key column) -> the lock one road's birth is under


def shape_ok(tree):
    """MIN/MAX-only projections, comma/INNER joins on plain columns, no group/order/limit/subquery/window."""
    if not isinstance(tree, E.Select): return False
    if tree.args.get('having') is not None: return False
    if tree.find(E.Window) is not None or tree.find(E.Subquery) is not None: return False
    if not tree.expressions: return False
    g = tree.args.get('group')
    naggs = 0
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        if isinstance(nd, (E.Min, E.Max)) and isinstance(nd.this, E.Column): naggs += 1; continue
        if isinstance(nd, E.Count) and (nd.this is None or isinstance(nd.this, E.Star)): naggs += 1; continue
        if isinstance(nd, (E.Sum, E.Avg)) and isinstance(nd.this, E.Column): naggs += 1; continue
        if isinstance(nd, E.Column) and g is not None: continue    # a group key (only with GROUP BY)
        return False
    if naggs == 0: return False                                    # a row dump is not this organ
    if g is not None and not all(isinstance(k, E.Column) for k in g.expressions): return False
    joins = tree.args.get('joins') or []
    for jn in joins:
        if (jn.args.get('side') or '') or (jn.args.get('kind') or '').upper() not in ('', 'INNER', 'CROSS'): return False
        if not isinstance(jn.this, E.Table): return False
    frm = tree.args.get('from') or tree.args.get('from_')
    return frm is not None and isinstance(frm.this, E.Table)


def _conjuncts(node):
    if node is None: return []
    if isinstance(node, E.Paren): return _conjuncts(node.this)
    if isinstance(node, E.And): return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


_PLANS = __import__('wdb_qmem').register({})
_KMAX = __import__('wdb_qmem').register({})
_REUSE = None

from numba import njit


@njit(nogil=True, cache=True)
def _postings_kernel(u, offs, order, sk):
    """THE POSTINGS KERNEL: the rows of a key set through the reverse road -- searchsorted,
    the range arithmetic and the order gather as ONE pass (five numpy passes and an mmap gather
    were ~0.4s of a 2s profile)"""
    m = sk.size
    total = 0
    hits = np.empty(m, np.int64); nh = 0
    for i in range(m):
        k = sk[i]
        lo = 0; hi = u.size
        while lo < hi:
            mid = (lo + hi) >> 1
            if u[mid] < k: lo = mid + 1
            else: hi = mid
        if lo < u.size and u[lo] == k:
            hits[nh] = lo; nh += 1; total += offs[lo + 1] - offs[lo]
    out = np.empty(total, np.int64); p = 0
    for j in range(nh):
        h = hits[j]
        for q in range(offs[h], offs[h + 1]):
            out[p] = order[q]; p += 1
    return out


@njit(nogil=True, cache=True)
def _keyset_kernel(col, idx, mx):
    """THE KEY-SET KERNEL: gather the keys at the live rows, mark them in a bitmap, compact to a
    sorted unique array -- one pass in, one pass out (gather / mask / zeros / scatter / nonzero
    were five)"""
    mark = np.zeros(mx + 1, np.bool_)
    touched = np.empty(idx.size, np.int64)
    n = 0
    for i in range(idx.size):
        k = col[idx[i]]
        if k >= 0:
            if not mark[k]:
                mark[k] = True; touched[n] = k; n += 1
    # PROPORTIONAL TO n, NOT TO THE ID SPACE: a table with 300 live rows compacts by sorting its
    # 300 keys, not by sweeping 4M bitmap entries (15 sweeps x 3ms per query)
    if n * 400 < mx:                                   # only when n is tiny: a 4M sweep is ~4ms, a 250K sort ~20ms
        out = touched[:n].copy(); out.sort()
        return out
    out = np.empty(n, np.int64); p = 0
    for k in range(mx + 1):
        if mark[k]:
            out[p] = k; p += 1
    return out


def execute(db, tree, sql=None):
    import wdb_sql, os, time
    _bill = [] if os.environ.get('WDB_SEMI_BILL') else None
    _tk = time.perf_counter; _t0 = _tk()
    from wdb_join import _solo_segment, _FastUnsupported, _bulk_keyvals
    # THE PLAN CACHE (Jackson: known territory first): the query's shape -- aliases, join edges,
    # local conjuncts, column ownership -- is the same every time the SQL is the same; parsed
    # once per process and keyed by the SQL and the catalog stamp
    _pk = (sql if sql is not None else tree.sql(), db._catalog_stamp() if hasattr(db, '_catalog_stamp') else None, id(db.cat))   # keyed by the SQL string: tree.sql() cost ~7ms per query
    plan = _PLANS.get(_pk)
    if plan is None:
        frm = tree.args.get('from') or tree.args.get('from_')
        tabs = [frm.this] + [jn.this for jn in (tree.args.get('joins') or [])]
        alias2t = {}
        for t in tabs:
            alias2t[t.alias or t.name] = t.name
        conds = []
        for c in _conjuncts(tree.args.get('where').this if tree.args.get('where') is not None else None):
            conds.append(c)
        for jn in (tree.args.get('joins') or []):
            if jn.args.get('on') is not None: conds.extend(_conjuncts(jn.args['on']))
        # column ownership
        cols_of = {a: set(db.cat.column_names(t)) for a, t in alias2t.items()}
        def _owner(col):
            if col.table: return col.table
            cands = [a for a, cs in cols_of.items() if col.name in cs]
            if len(cands) != 1: raise _Decline('ambiguous column %s' % col.name)
            return cands[0]
        edges, local = [], {a: [] for a in alias2t}
        for c in conds:
            if isinstance(c, E.EQ) and isinstance(c.this, E.Column) and isinstance(c.expression, E.Column):
                a, b = _owner(c.this), _owner(c.expression)
                if a != b:
                    edges.append((a, c.this.name, b, c.expression.name)); continue
            owners = {_owner(x) for x in c.find_all(E.Column)}
            if len(owners) != 1: raise _Decline('multi-table non-equality conjunct: %s' % c.sql()[:50])
            local[owners.pop()].append(c)
        pms = {a: db.cat.phys_map(t) for a, t in alias2t.items()}
        # the table-stripped conjunct copies belong in the plan too: 2,440 deepcopy calls per query
        # (95ms of a 310ms query) were rebuilding them
        for a in alias2t:
            stripped = []
            for c in local[a]:
                c2 = c.copy()
                for col in c2.find_all(E.Column):
                    col.set('table', None)
                stripped.append(c2)
            local[a] = stripped
        plan = _PLANS[_pk] = (alias2t, cols_of, edges, local, pms)
        if len(_PLANS) > 512: _PLANS.pop(next(iter(_PLANS)))
    alias2t, cols_of, edges, local, pms = plan
    def owner(col):
        if col.table: return col.table
        cands = [a for a, cs in cols_of.items() if col.name in cs]
        if len(cands) != 1: raise _Decline('ambiguous column %s' % col.name)
        return cands[0]
    # segments (a clean segment per table: tombstones/overrides decline by name inside _solo_segment)
    segs, keeps = {}, {}
    for a, t in alias2t.items():
        try:
            seg, _ = _solo_segment(db, t)
        except _FastUnsupported:
            raise _Decline('multi-segment table %s' % t)
        segs[a] = seg
    def _local_keep(a):
        """PHASE 1, ISOLATION: one table's own filters, by itself (runs in the pool)"""
        seg = segs[a]; m = None
        for c2 in local[a]:                            # already table-stripped, in the plan
            # THE PREDICATE SHELF: a local predicate's result on an immutable segment is a fact --
            # kept as an index list on the shelf across queries (JOB's 113 queries reuse the same
            # few filters on the same big tables: mi.info LIKE ... on 14.8M rows was 392ms, every
            # time). A segment with tombstones or overrides never gets here (_solo_segment).
            _pk9 = ('pred', seg.path, c2.sql())
            _hit9 = wdb_shelf.SHELF.get(_pk9)
            if _hit9 is not None:
                mm = np.zeros(int(seg.N), bool); mm[_hit9] = True; il = _hit9
            else:
                mm = np.asarray(wdb_sql._eval_pred(seg, c2, lambda nm, pm=pms[a]: pm.get(nm, nm)), dtype=bool)
                il = _il9 = np.flatnonzero(mm).astype(np.int32 if seg.N < (1 << 31) else np.int64)
                try: wdb_shelf.SHELF.put(_pk9, _il9, int(_il9.nbytes), kind='predicate')
                except Exception: pass
            if m is None:
                m, m_idx = mm, il                          # THE INDEX LIST RIDES ALONG: the shelf already holds it
            else:
                m = m & mm; m_idx = None                   # (a conjunction: derive once at the landing)
        if m is None: return None
        return (m, m_idx)
    from concurrent.futures import ThreadPoolExecutor
    _pool9 = ThreadPoolExecutor(max_workers=min(8, max(1, len(alias2t))))
    futs = {a: _pool9.submit(_local_keep, a) for a in alias2t}     # THE STREAM: isolation runs concurrently;
    def _unpack9(r, a):
        if r is None: return np.ones(int(segs[a].N), bool)
        return r[0]
    for a in alias2t:                                                # a table with no filter is ready at once
        if not local[a]:
            keeps[a] = _unpack9(futs[a].result(), a)
    if os.environ.get('WDB_KEYSPACE', '1') == '0':
        for a in alias2t:
            keeps[a] = _unpack9(futs[a].result(), a)
            if _bill is not None: _bill.append(('local %s(%d) keep=%d' % (a, int(segs[a].N), int(keeps[a].sum())), _tk() - _t0)); _t0 = _tk()
    # key columns as int64 arrays (NULL -> -1)
    keycache = {}
    def keys(a, col):
        k = (a, col)
        if k in keycache: return keycache[k]
        seg = segs[a]; pc = pms[a].get(col, col); cd = seg.cols.get(pc)
        if cd is None: raise _Decline('no such column %s.%s' % (a, col))
        if cd.get('dt') != 0: raise _Decline('non-integer join key %s.%s' % (a, col))
        # THE KEY COLUMN ON THE SHELF: a join key of an immutable segment, decoded once per
        # process as int32 (cast_info's three keys: 435 MB); a key set is then a gather and a
        # bitmap, not a road walk and a binary search per step
        _sk9 = ('keys', seg.path, pc)
        _hit9 = wdb_shelf.SHELF.get(_sk9)
        if _hit9 is not None:
            keycache[k] = _hit9; return _hit9
        raw = wdb_sql.raw_dict_col(seg, pc, want_codes=False)
        codes = np.asarray(seg.codes(pc))
        if raw is not None:
            vals = np.asarray(raw[0]).astype(np.int64)
            if cd.get('has_null'):
                vals = np.append(vals, -1)
            out = vals[codes]
        else:
            arr, nm = wdb_sql._col(seg, pc)
            out = np.asarray(arr).astype(np.int64)
            if nm is not None: out = np.where(nm, -1, out)
        if out.size >= 1_000_000 and out.max() < (1 << 31) and out.min() >= -1:
            out = out.astype(np.int32)
            try: wdb_shelf.SHELF.put(_sk9, out, int(out.nbytes), kind='keys')
            except Exception: pass
        keycache[k] = out
        return out
    def keys_at(a, col, idx):
        """key values at the given rows only (point reads) -- NULL -> -1"""
        seg = segs[a]; pc = pms[a].get(col, col); cd = seg.cols.get(pc)
        if cd is None: raise _Decline('no such column %s.%s' % (a, col))
        if cd.get('dt') != 0: raise _Decline('non-integer join key %s.%s' % (a, col))
        if (a, col) in keycache:
            return keycache[(a, col)][idx]
        inv9 = getattr(seg, '_inv_cache', {}).get(pc)
        if inv9 is not None and not isinstance(inv9, wdb_coordroad.CoordRoad) and inv9[3] is not None:
            # an old flat road that still carries rank: rank[row] -> position -> key. New roads have no rank
            # (its one reader is this; the dictionary below answers the same at 0.2-61 ms measured)
            u9, offs9, _o9, rank9 = inv9
            pos9 = np.asarray(rank9[idx]).astype(np.int64)
            return u9[np.searchsorted(offs9, pos9, side='right') - 1]
        raw = wdb_sql.raw_dict_col(seg, pc, want_codes=False)
        if raw is not None:
            vals = np.asarray(raw[0]).astype(np.int64)
            if cd.get('has_null'): vals = np.append(vals, -1)
            return vals[np.asarray(seg.codes_at(pc, idx))]
        return np.array([(-1 if v is None else int(v)) for v in seg.values_at_rows(pc, idx)], dtype=np.int64)
    # edge merge: several equalities between the same pair -> composite keys
    pair = {}
    for a, ca, b, cb in edges:
        key = (a, b) if a < b else (b, a)
        pair.setdefault(key, []).append((ca, cb) if a < b else (cb, ca))
    def pack(t, cols):
        ks = [keys(t, c) for c in cols]
        if len(ks) == 1: return ks[0]
        x = np.zeros_like(ks[0])
        for v in ks: x = x * (1 << 31) + np.where(v < 0, 0, v)      # ids < 2^31 in IMDB
        return np.where(np.any(np.stack([v < 0 for v in ks]), axis=0), -1, x)
    # THE REVERSE ROAD: an inverted index per key column (sorted unique values,
    # offsets, row order) -- born once per segment file, mmap'd -- so a small
    # surviving key set on one side yields its rows on the other without a
    # pass over N. Used when the source keep is small relative to the target.
    def inverted(a, cols_a):
        """one birth per road: the stream's isolation threads asked for the same road at once and both
        built it (two 36M argsorts, two writes, a .tmp left behind)"""
        if len(cols_a) != 1: return None
        seg = segs[a]; pc = pms[a].get(cols_a[0], cols_a[0])
        with _INV_LOCK:
            lk = _INV_LOCKS.setdefault((getattr(seg, 'path', id(seg)), pc), __import__('threading').Lock())
        with lk:
            return _inverted0(a, cols_a)
    def _inverted0(a, cols_a):
        seg = segs[a]; pc = pms[a].get(cols_a[0], cols_a[0])
        import wdb_shelf
        cache = getattr(seg, '_inv_cache', None)
        if cache is None: cache = seg._inv_cache = {}
        if pc in cache:
            if wdb_shelf.SHELF.get(('inv', getattr(seg, 'path', id(seg)), pc)) is None:
                cache.pop(pc, None)                  # the shelf evicted it: reload from the sidecar
            else:
                return cache[pc]
        path = getattr(seg, 'path', None) or getattr(seg, '_path', None)
        fn = ('%s.%s.inv' % (path, pc)) if path else None       # three mmap'd .npy files: u / offs / order
        import wdb_sidecar, wdb_coordroad
        _os9 = __import__('os')
        try:
            if fn and wdb_sidecar.exists(fn + '.hdr.npy') \
                    and wdb_sidecar.is_fresh(_os9.path.dirname(path), _os9.path.basename(fn + '.hdr.npy')):
                # THE COORDINATE ROAD on disk (Jackson's blocks): headers in RAM, positions mmap'd
                road = wdb_coordroad.load(fn)
                if road is not None:
                    cache[pc] = road
                    try: wdb_shelf.SHELF.put(('inv', getattr(seg, 'path', id(seg)), pc), road, road.resident_bytes, kind='reverse-road')
                    except wdb_shelf.ShelfRefused: pass
                    return road
            if fn and wdb_sidecar.exists(fn + '.order.npy') \
                    and wdb_sidecar.is_fresh(_os9.path.dirname(path), _os9.path.basename(fn + '.order.npy')):
                # THE BIRTHMARK: a sidecar older than its segment is false by construction. rank is optional:
                # retired from births (its one reader, keys_at, has the dictionary), read when an old road has it
                u = np.load(fn + '.u.npy'); offs = np.load(fn + '.offs.npy')
                order = np.load(fn + '.order.npy', mmap_mode='r')
                rank = np.load(fn + '.rank.npy', mmap_mode='r') if wdb_sidecar.exists(fn + '.rank.npy') else None
                cache[pc] = (u, offs, order, rank)
                try: wdb_shelf.SHELF.put(('inv', getattr(seg, 'path', id(seg)), pc), cache[pc], u.nbytes + offs.nbytes, kind='reverse-road')
                except wdb_shelf.ShelfRefused: pass
                return cache[pc]
        except Exception:
            pass
        vals = keys(a, cols_a[0])
        order = np.argsort(vals, kind='stable').astype(np.int32 if vals.size < 2**31 else np.int64)
        sv = vals[order]
        u, starts = np.unique(sv, return_index=True)
        offs = np.append(starts, sv.size).astype(np.int64)
        rank = np.empty_like(order); rank[order] = np.arange(order.size, dtype=order.dtype)   # row -> position (RAM only)
        cache[pc] = (u, offs, order, rank)
        # THE RULE AT BIRTH, BY MEASUREMENT: coordinates when their headers + positions are smaller than
        # the flat rows (keys whose rows share blocks); flat when every row would pay its own header
        _plan9 = wdb_coordroad.plan(order, offs) if (vals.size < (1 << 32) and _os9.environ.get('WDB_COORD_ROAD', '1') != '0') else None
        coord = _plan9 is not None and _plan9[-1] < wdb_coordroad.flat_bytes(order, offs)    # whole file vs whole file (u is shared)
        if coord:
            road = wdb_coordroad.build(u, order, offs)
            cache[pc] = road
        nbytes9 = int(road.disk_bytes) if coord else int(u.nbytes + offs.nbytes + order.nbytes)
        if fn:
            try:
                wdb_sidecar.may_birth(_os9.path.dirname(path), nbytes9,
                                      '%s road %s.%s' % ('coordinate' if coord else 'reverse', _os9.path.basename(path), pc))   # THE DISK GATE
                if coord:
                    wdb_coordroad.save(fn, road)
                    for _s9 in wdb_coordroad.SUFFIXES: wdb_sidecar.born('%s.%s.npy' % (fn, _s9))
                    cache[pc] = wdb_coordroad.load(fn) or road
                    _resident9 = cache[pc].resident_bytes
                else:
                    for nm9, arr9 in (('u', u), ('offs', offs), ('order', order)):
                        np.save(fn + '.%s.tmp.npy' % nm9, arr9)
                        _os9.replace(fn + '.%s.tmp.npy' % nm9, fn + '.%s.npy' % nm9)
                        wdb_sidecar.born(fn + '.%s.npy' % nm9)
                    cache[pc] = (u, offs, np.load(fn + '.order.npy', mmap_mode='r'), rank)
                    _resident9 = u.nbytes + offs.nbytes + rank.nbytes
                # THE BORN ROAD STAYS: on the shelf under its resident bytes, so the next query finds it in the
                # cache instead of asking the filesystem inside the negative-answer window and rebirthing it
                try: wdb_shelf.SHELF.put(('inv', getattr(seg, 'path', id(seg)), pc), cache[pc], int(_resident9), kind='reverse-road')
                except wdb_shelf.ShelfRefused: pass
                if _os9.environ.get('WDB_SEMI_BILL'):
                    print('SEMI: %s road born %s.%s: %.1f MB on disk%s' % ('coordinate' if coord else 'flat', _os9.path.basename(path), pc, nbytes9 / 1e6,
                                                                  (' (flat would be %.1f MB)' % ((u.nbytes + offs.nbytes + order.nbytes) / 1e6)) if coord else ''), flush=True)
            except wdb_sidecar.BirthRefused as _e:
                # THE ROAD ON THE SHELF: refused on disk (the switch off, or the budget) is not refused in
                # RAM -- it lives for the process under the shelf's ceiling, like a loaded one would
                try: wdb_shelf.SHELF.put(('inv', getattr(seg, 'path', id(seg)), pc), cache[pc], nbytes9, kind='reverse-road')
                except wdb_shelf.ShelfRefused:
                    print('SEMI: %s -- serving from RAM this query' % str(_e)[:120], flush=True)
            except Exception as _e:
                if _os9.environ.get('WDB_SEMI_BILL'):
                    print('SEMI: inverted sidecar save failed for %s: %s' % (fn, str(_e)[:80]), flush=True)
        return cache[pc]
    def rows_for_keys(inv, sk):
        """THE ROWS OF A KEY SET: the reverse road's postings for the keys in sk, as row positions --
        cost proportional to the rows matched, no N-scale bitmap"""
        if isinstance(inv, wdb_coordroad.CoordRoad):
            return inv.rows(sk)                                 # THE COORDINATE ROAD: the same walk, block-local positions
        u, offs, order, _rank = inv
        if sk.size and sk.size <= 4_000_000:
            try:
                return _postings_kernel(np.asarray(u, dtype=np.int64), np.asarray(offs, dtype=np.int64), np.asarray(order), np.asarray(sk, dtype=np.int64))
            except Exception:
                pass
        pos = np.searchsorted(u, sk)
        ok = pos < u.size
        pos = pos[ok]; skk = sk[ok]
        hit = pos[u[pos] == skk]
        if hit.size == 0: return np.zeros(0, np.int64)
        st = offs[hit]; ln = offs[hit + 1] - st
        total = int(ln.sum())
        if total == 0: return np.zeros(0, np.int64)
        base = np.repeat(st - np.concatenate(([0], np.cumsum(ln)[:-1])), ln)
        idx = np.arange(total, dtype=np.int64) + base
        return np.asarray(order[idx]).astype(np.int64)
    def prune_inverted(inv, sk, dst_keep, dst_n):
        if isinstance(inv, wdb_coordroad.CoordRoad):
            rows = inv.rows(sk)
            out = np.zeros_like(dst_keep); out[rows] = True
            return out & dst_keep
        u, offs, order, _rank = inv
        pos = np.searchsorted(u, sk)
        ok = pos < u.size
        pos = pos[ok]; skk = sk[ok]
        hit = pos[u[pos] == skk]
        if hit.size == 0: return np.zeros_like(dst_keep)
        st = offs[hit]; ln = offs[hit + 1] - st
        total = int(ln.sum())
        if total == 0: return np.zeros_like(dst_keep)
        # RANGES-CONCAT: every posting list gathered in one vectorised pass
        base = np.repeat(st - np.concatenate(([0], np.cumsum(ln)[:-1])), ln)
        idx = np.arange(total, dtype=np.int64) + base
        rows = np.asarray(order[idx])
        out = np.zeros_like(dst_keep); out[rows] = True
        return out & dst_keep
    # fixpoint
    def prune(src, src_cols, src_keep, dst, dst_cols, dst_keep):
        n_src = int(segs[src].N); n_dst = int(segs[dst].N)
        src_idx = np.flatnonzero(src_keep)
        if src_idx.size == 0:
            return np.zeros_like(dst_keep)
        # SOURCE keys: point reads at the surviving rows when the keep is small
        if len(src_cols) == 1:
            sk = keys_at(src, src_cols[0], src_idx) if src_idx.size * 4 < n_src else keys(src, src_cols[0])[src_idx]
        else:
            sk = pack(src, src_cols)[src_idx]
        sk = sk[sk >= 0]
        if sk.size == 0:
            return np.zeros_like(dst_keep)
        if len(dst_cols) == 1 and n_dst >= 1_000_000 and sk.size * 8 < n_dst and counts.get(dst, n_dst) * 2 > src_idx.size:
            inv = inverted(dst, dst_cols)
            if inv is not None:
                return prune_inverted(inv, np.unique(sk), dst_keep, n_dst)
        idx = np.flatnonzero(dst_keep)
        if idx.size == 0: return dst_keep
        if len(dst_cols) == 1 and idx.size * 4 < n_dst and n_dst >= 1_000_000:
            # SURVIVORS ONLY through the reverse road's RANK: keys at the kept rows,
            # never the whole column (a 2.2M keep on a 36M table walked all 36M)
            dk = keys_at(dst, dst_cols[0], idx)
            mx2 = int(max(sk.max(), dk.max())) if dk.size else int(sk.max())
            if mx2 < 200_000_000:
                lut2 = np.zeros(mx2 + 2, bool); lut2[sk] = True
                hit2 = lut2[np.where(dk < 0, mx2 + 1, dk)]
            else:
                hit2 = np.isin(dk, np.unique(sk))
            out = np.zeros_like(dst_keep); out[idx[hit2]] = True
            return out
        dst_keys = pack(dst, dst_cols)
        mx = int(max(sk.max(), dst_keys.max())) if dst_keys.size else int(sk.max())
        if mx < 200_000_000:
            lut = np.zeros(mx + 2, bool); lut[sk] = True
            if idx.size * 8 < dst_keys.size:
                # SURVIVORS ONLY: once the keep is small, gather only the kept rows' keys
                dk = dst_keys[idx]
                hit_s = lut[np.where(dk < 0, mx + 1, dk)]
                out = np.zeros_like(dst_keep); out[idx[hit_s]] = True
                return out
            hit = lut[np.where(dst_keys < 0, mx + 1, dst_keys)]
        else:
            hit = np.isin(dst_keys, np.unique(sk))
        return dst_keep & hit
    counts = {a: int(np.count_nonzero(k)) for a, k in keeps.items() if a in keeps}
    for a in alias2t:
        if a not in keeps: keeps[a] = np.ones(int(segs[a].N), bool); counts[a] = int(segs[a].N)   # unrestricted until its filter lands
    if os.environ.get('WDB_KEYSPACE', '1') != '0':
        _keyspace_fixpoint(alias2t, segs, edges, keeps, counts, keys, keys_at, inverted, rows_for_keys, pack, _bill, _tk, futs=futs, pool=_pool9, keycache=keycache, local=local)
        _t0 = _tk()
        _pool9.shutdown(wait=False)
    else:
      for _round in range(12):
        changed = False
        for (a, b), pairs in pair.items():
            ca9 = [p[0] for p in pairs]; cb9 = [p[1] for p in pairs]
            # prune the side with MORE survivors from the side with fewer, first
            def _step(x, cx, y, cy):
                nonlocal changed
                ny = prune(x, cx, keeps[x], y, cy, keeps[y])
                cnt = int(np.count_nonzero(ny))
                if cnt != counts[y]:
                    keeps[y] = ny; counts[y] = cnt; changed = True
            if counts[a] >= counts[b]:
                _step(b, cb9, a, ca9); _step(a, ca9, b, cb9)
            else:
                _step(a, ca9, b, cb9); _step(b, cb9, a, ca9)
            if _bill is not None: _bill.append(('r%d %s-%s keep %d/%d' % (_round, a, b, counts[a], counts[b]), _tk() - _t0)); _t0 = _tk()
        if not changed: break
    state = {'alias2t': alias2t, 'segs': segs, 'pms': pms, 'keeps': keeps, 'counts': counts, 'pair': pair,
             'keys_at': keys_at, 'keys': keys, 'owner': owner, 'tree': tree}
    if _needs_weights(tree):
        out9 = _counting_emit(db, state)
        if _bill is not None:
            _bill.append(('weights+emit', _tk() - _t0))
            print('SEMI BILL: ' + ' | '.join('%s=%.0fms' % (n, v * 1000) for n, v in _bill), flush=True)
        return out9
    # any table empty -> every MIN is NULL (a scalar over an empty join)
    empty = any(c == 0 for c in counts.values())
    out = []
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        a = owner(nd.this); seg = segs[a]; pc = pms[a].get(nd.this.name, nd.this.name)
        if empty:
            out.append(None); continue
        rows = np.flatnonzero(keeps[a])
        cd = seg.cols.get(pc, {})
        codes = _codes_shelved(seg, pc)[rows] if cd.get('mode') in (0, 1, 2) else None
        if codes is not None:
            if cd.get('has_null'):
                codes = codes[codes != int(cd['V']) - 1]
            if codes.size == 0:
                out.append(None); continue
            k = int(codes.min()) if isinstance(nd, E.Min) else int(codes.max())     # sorted dictionary
            v = seg.fetch(pc, k)                       # ONE chunk, not the 4.1M-value dictionary decoded for one entry
            out.append(wdb_sql._pyval(v))
        else:
            srank = _string_rank(seg, pc) if (cd.get('dt') == 1 and int(seg.N) >= 200_000) else None
            if srank is not None:
                # THE STRING RANK ROAD: MIN/MAX over an inline (mode-5) column is one argmin over
                # int32 ranks and a single point read -- not 322K Python objects per query
                rr = srank[rows]
                ok = rr >= 0
                if not ok.any():
                    out.append(None); continue
                j = rows[ok][int(rr[ok].argmin()) if isinstance(nd, E.Min) else int(rr[ok].argmax())]
                out.append(wdb_sql._pyval(seg.values_at_rows(pc, np.array([j]))[0]))
                continue
            vals = list(seg.values_at_rows(pc, rows))
            vals = [v for v in vals if v is not None]
            out.append(wdb_sql._pyval(min(vals) if isinstance(nd, E.Min) else max(vals)) if vals else None)
    names = [wdb_sql._alias(p) for p in tree.expressions]
    if _bill is not None:
        _bill.append(('emit', _tk() - _t0))
        print('SEMI BILL: ' + ' | '.join('%s=%.0fms' % (n, v * 1000) for n, v in _bill), flush=True)
    return [tuple(out)], names


def _string_rank(seg, pc):
    """each row's rank in the column's sorted order (NULL -> -1), int32, born once as a sidecar
    (<seg>.<col>.srank.npy) and shelved; a sort of 4.2M names is ~5s, once"""
    import os
    key = ('srank', seg.path, pc)
    hit = wdb_shelf.SHELF.get(key)
    if hit is not None: return hit
    fn = seg.path + '.' + pc + '.srank.npy'
    try:
        import wdb_sidecar
        if wdb_sidecar.exists(fn) and wdb_sidecar.is_fresh(os.path.dirname(seg.path), os.path.basename(fn)):
            r = np.load(fn)
            try: wdb_shelf.SHELF.put(key, r, int(r.nbytes), kind='string-rank')
            except Exception: pass
            return r
    except Exception:
        pass
    vals = np.asarray(seg.values(pc), dtype=object)
    isn = np.array([v is None for v in vals], dtype=bool)
    order = np.argsort(np.where(isn, b'', vals), kind='stable')
    rank = np.empty(vals.size, np.int32); rank[order] = np.arange(vals.size, dtype=np.int32)
    rank[isn] = -1
    try:
        import wdb_sidecar
        if wdb_sidecar.may_birth(os.path.dirname(seg.path), int(rank.nbytes), 'string rank %s' % pc):
            np.save(fn + '.partial.npy', rank); os.replace(fn + '.partial.npy', fn)
    except Exception:
        pass
    try: wdb_shelf.SHELF.put(key, rank, int(rank.nbytes), kind='string-rank')
    except Exception: pass
    return rank


def _codes_shelved(seg, pc):
    """THE CODE COLUMN ON THE SHELF: an emit column's codes on an immutable segment, decoded once
    per process as int32 -- MIN(n.name) over 322K live rows decoded the 4.1M-row name stream
    every query (146ms of a 190ms query) because per-query caches flush"""
    key = ('codes', seg.path, pc)
    hit = wdb_shelf.SHELF.get(key)
    if hit is not None: return hit
    c = np.asarray(seg.codes(pc))
    if c.size >= 200_000 and int(seg.cols[pc].get('V', 0)) < (1 << 31):
        c = c.astype(np.int32)
        try: wdb_shelf.SHELF.put(key, c, int(c.nbytes), kind='codes')
        except Exception: pass
    return c


def _keyspace_fixpoint(alias2t, segs, edges, keeps, counts, keys, keys_at, inverted, rows_for_keys, pack, _bill, _tk, futs=None, pool=None, keycache=None, local=None):
    """THE KEY-SPACE FIXPOINT (Jackson's two phases, 2026-09-14). Phase 1 -- isolation -- is done
    by the caller: every table has applied its own filters. Phase 2 -- the conjoined space:
    the join columns collapse into SHARED VALUE SPACES (one per equivalence class of equal
    columns: 'movie' = t.id = ci.movie_id = mk.movie_id = mc.movie_id); a space's live set is
    the intersection of the KEY SETS of the tables that restrict it (an unfiltered table
    restricts nothing); a table whose spaces shrank re-derives its rows against the live sets --
    smallest table first, the giant last, each pass costing the rows MATCHED, never the table --
    which shrinks its other spaces; until nothing shrinks. The old sweep asked cast_info (36M)
    about name and title before the keyword filter had reached it: 5.5s of a 6.9s query."""
    import time
    t0 = [_tk()]
    def bill(msg):
        if _bill is not None:
            _bill.append((msg, _tk() - t0[0])); t0[0] = _tk()
    # ---- the spaces: union-find over (alias, column) ----
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, ca, b, cb in edges:
        ra, rb = find((a, ca)), find((b, cb))
        if ra != rb: parent[ra] = rb
    members = {}
    for a, ca, b, cb in edges:
        members.setdefault(find((a, ca)), set()).add((a, ca)); members.setdefault(find((b, cb)), set()).add((b, cb))
    spaces = {r: sorted(m) for r, m in members.items()}
    cols_of = {a: sorted({c for (x, c) in parent if x == a}) for a in alias2t}
    n_of = {a: int(segs[a].N) for a in alias2t}
    # ---- key sets: the live keys of a table's column (None = no restriction) ----
    full = {a: counts[a] == n_of[a] for a in alias2t}
    kcache = {}
    import hashlib
    prov = {a: frozenset() for a in alias2t}     # THE PROVENANCE OF A KEEP: local predicates + applied (col, S) cuts -- order-free
    def _sprov(r):
        """the symbolic identity of a space's live set: (segment, column, keep-provenance) of every
        member that restricts it, as a frozenset -- or None when any member's provenance already
        contains a cut. THE CACHE IS BOUNDED TO FIRST HOPS: nested provenances grew exponentially
        (a table's key includes its cutters', which include their cutters' ...) and every shelf
        lookup hashed the whole tree (8.6s -> 20-48s). The census said the repeats ARE first hops."""
        out = []
        for (b, cb) in spaces[r]:
            if full[b]: continue
            if prov[b] is None or _depth(prov[b]) >= 2: return None    # TWO HOPS: a junction's first cut is usually
            out.append((segs[b].path, cb, prov[b]))                    # through a dimension (keyword -> mk -> movie -> ci)
        return frozenset(out)
    _dmemo = {}
    def _depth(p):
        """how many cut-levels a provenance nests: 0 = local predicates only"""
        k = id(p); h = _dmemo.get(k)
        if h is not None: return h
        d = 0
        for e in p:
            if e[0] == 'cut' and e[2] is not None:
                for (_pth, _c, pb) in e[2]:
                    d = max(d, 1 + _depth(pb))
        _dmemo[k] = d
        return d
    kmax = _KMAX                                   # a column's max is a fact about the column: per process, not per query
    keycache = keycache if keycache is not None else {}
    idx_of = {}        # THE LIVE ROW LIST per table, kept beside the bool keep: never re-derived per step
    def live_idx(a):
        if a not in idx_of: idx_of[a] = np.flatnonzero(keeps[a])
        return idx_of[a]
    def keyset(a, c):
        if full[a]: return None
        ck = (a, c, counts[a])                     # cached per (table, column, live count): a key set
        if ck in kcache: return kcache[ck]         # only changes when its table shrinks
        # THE SETTLED-SPACE CACHE (Jackson's relationship chain): a junction's key set after a cut is a
        # fact about (segment, column, provenance of the keep); JOB's families repeat the same first
        # hops -- rt.role = 'actor' reaches cast_info in 13 queries -- and recomputed each one (12.7M
        # live rows -> the movie key set, ~50ms). Shelved across queries; DML moves the stamp.
        _sk9 = ('kset', segs[a].path, c, prov[a]) if (prov[a] is not None and prov[a]) else None
        if _sk9 is not None:
            _h9 = wdb_shelf.SHELF.get(_sk9)
            if _h9 is not None:
                kcache[ck] = _h9; return _h9
        idx = live_idx(a)
        if idx.size == 0: return np.zeros(0, np.int64)
        _kc9 = keycache.get((a, c))
        if _kc9 is None:
            # every table takes the kernel path: the small-table path built a bitmap over the ID SPACE
            # (2.5M entries for title ids) and scanned it for a table with 300 live rows -- 1.5s of
            # nonzero and 1.3s of max per 113 queries
            try: _kc9 = keys(a, c)                      # decodes once per process, shelved when large
            except Exception: _kc9 = None
        if _kc9 is not None:
            _mk9 = (segs[a].path, c)
            mxk = kmax.get(_mk9)
            if mxk is None: mxk = kmax[_mk9] = int(_kc9.max()) if _kc9.size else 0
            if 0 <= mxk < 200_000_000:
                _tq9 = _tk()
                out = _keyset_kernel(np.asarray(_kc9), np.asarray(idx, dtype=np.int64), mxk)     # THE KEY-SET KERNEL
                kcache[ck] = out
                if _sk9 is not None and idx.size >= 100_000:
                    try: wdb_shelf.SHELF.put(_sk9, out, int(out.nbytes), kind='settled-space')
                    except Exception: pass
                if _bill is not None: _bill.append(('      keyset-kernel %s.%s rows=%d keys=%d mx=%d' % (a, c, idx.size, out.size, mxk), _tk() - _tq9))
                return out
        k = np.asarray(_kc9[idx], dtype=np.int64) if _kc9 is not None else (keys_at(a, c, idx) if idx.size * 4 < n_of[a] else keys(a, c)[idx])
        k = k[k >= 0]
        if k.size == 0:
            out = np.zeros(0, np.int64)
        else:
            mx = int(k.max())
            if mx < 200_000_000:
                # A KEY SET IS A BITMAP, NOT A SORT: keys are dense ids; np.unique on 2.7M positions
                # cost 1.27s, twice per key set, three key sets for cast_info (5.1s of a query).
                # (a reused scratch bitmap was measured and rejected: np.zeros is a lazy calloc, nearly
                # free, while flatnonzero over a size-class buffer scanned up to twice the entries)
                lut = np.zeros(mx + 1, bool); lut[k] = True
                out = np.flatnonzero(lut).astype(np.int64)
            else:
                out = np.unique(k)
        kcache[ck] = out
        return out
    live = {}          # space root -> sorted live keys, or None
    scache = {}
    def _isect(cur, ks):
        """A SPACE IS A BITMAP INTERSECTION: keys are dense ids; np.intersect1d sorted the
        concatenation of 1.27M and 1.9M keys, 76 times in one query (1.8s of 2.3s)"""
        if cur.size == 0 or ks.size == 0: return np.zeros(0, np.int64)
        mx = int(max(cur[-1], ks[-1]))
        _tq9 = _tk()
        if mx < 200_000_000:
            m = np.zeros(mx + 1, bool); m[cur] = True
            out = ks[m[ks]]
        else:
            out = np.intersect1d(cur, ks, assume_unique=True)
        if _bill is not None: _bill.append(('      isect %d x %d -> %d mx=%d' % (cur.size, ks.size, out.size, mx), _tk() - _tq9))
        return out
    def space_live(r):
        sk = (r, tuple(counts[a] for (a, c) in spaces[r]))    # cached by the members' live counts
        if sk in scache: return scache[sk]
        cur = None
        for (a, c) in spaces[r]:
            ks = keyset(a, c)
            if ks is None: continue
            cur = ks if cur is None else _isect(cur, ks)
        scache[sk] = cur
        return cur
    # THE STREAM (Jackson): isolation results land while phase 2 runs. A table whose filter has
    # not finished is treated as UNRESTRICTED until it does -- monotone shrinking makes that
    # safe: a late key set is one more shrink, never a correction. Every table starts as full;
    # when its filter lands, it shrinks like any other step.
    arrived = set()
    def _land(a):
        """take a finished isolation result for a (if ready); True if it landed now"""
        if a in arrived or futs is None or a not in futs: return False
        f = futs[a]
        if not f.done(): return False
        r = f.result(); arrived.add(a)
        if r is None: return False
        m, m_idx = r
        if prov[a] is not None: prov[a] = prov[a] | frozenset(('pred', c2.sql()) for c2 in local[a])
        if not full[a]:
            m = m & keeps[a]; m_idx = None        # A LANDING INTERSECTS: cuts the spaces already made are kept
        if m_idx is not None:
            cnt = int(m_idx.size)                 # the shelf's index list: no pass over the mask
        else:
            cnt = int(np.count_nonzero(m))        # (an overwrite here discarded them; the applied-space skip then
        if cnt != counts[a]:                      #  removed the accidental repair -- JOB 3b answered '#1' for '11,830,420')
            keeps[a] = m; counts[a] = cnt; full[a] = False
            idx_of[a] = np.asarray(m_idx, dtype=np.int64) if m_idx is not None else np.flatnonzero(m)
            return True
        return False
    if futs is not None:
        for a in list(alias2t):
            if futs[a].done(): _land(a)
    for r in spaces: live[r] = space_live(r)
    bill('spaces %d (isolation landed: %d/%d)' % (len(spaces), len(arrived), len(alias2t)))
    # ---- the worklist: tables whose spaces restrict them, smallest first ----
    def restrict(a):
        """re-derive a's rows against the live sets of its spaces; True if it shrank"""
        changed = False
        # THE SMALLEST SIGNAL FIRST, within the table too: cast_info's movie space (1.38M keys)
        # gathered 21M postings and sorted them (1.45s) before its person space (2 keys, 486 rows)
        # got its turn; the columns run in order of their live space's size
        order9 = sorted(cols_of[a], key=lambda c: (live[find((a, c))].size if live.get(find((a, c))) is not None else 1 << 62))
        for c in order9:
            r = find((a, c)); S = live.get(r)
            if S is None: continue
            # A SPACE ALREADY APPLIED IS NOT APPLIED AGAIN: a column whose live set has not shrunk
            # since this table last took it cannot cut the table further (cast_info re-gathered
            # the same 7.4M role postings four times in 19d)
            if applied.get((a, c)) == S.size: continue
            applied[(a, c)] = S.size
            # A LIVE SET IS NAMED BY WHERE IT CAME FROM, NOT BY ITS BYTES: the space's identity is the
            # provenances of the members that restricted it (hashing 1.5M keys per cut cost more than
            # the kernel it saved)
            _sp9 = _sprov(r) if prov[a] is not None else None
            prov[a] = (prov[a] | frozenset([('cut', c, _sp9)])) if _sp9 is not None else None   # None: beyond the first hop, uncached
            _rk9 = ('rows', segs[a].path, prov[a]) if (n_of[a] >= 1_000_000 and prov[a] is not None) else None
            if _rk9 is not None:
                _hr9 = wdb_shelf.SHELF.get(_rk9)               # THE CUT ITSELF, SHELVED: the same first hop's rows
                if _hr9 is not None:
                    rows, keep9 = _hr9                         # rows (int32) AND the bool keep: a hit does no N-scale work
                    rows = np.asarray(rows, dtype=np.int64)
                    cnt = int(rows.size)
                    if cnt != counts[a]:
                        keeps[a] = keep9; counts[a] = cnt; full[a] = False; changed = True; idx_of[a] = rows
                    continue
            keep = keeps[a]; n = n_of[a]
            if counts[a] == 0: return changed
            _tq = _tk()
            isfull = full[a] or counts[a] == n
            idx = None if isfull else live_idx(a)            # A FULL TABLE HAS NO LIVE LIST: materialising one is 36M
            live_n = n if isfull else idx.size               # entries of arange for a cut that never reads it
            bill('   %s.%s live %d' % (a, c, live_n)) if _bill is not None else None
            rows = None
            if n >= 1_000_000 and S.size * 8 < n and S.size * 4 < live_n:
                # the giant, cut by a small space: the road's postings, then only those already live
                inv = inverted(a, [c])
                bill('   %s.%s inverted' % (a, c)) if _bill is not None else None
                if inv is not None:
                    if _REUSE is not None:                       # the reuse census: would a cache keyed by (segment, column, S) hit?
                        import hashlib
                        _REUSE.append((segs[a].path, c, S.size, hashlib.blake2b(np.ascontiguousarray(S).tobytes(), digest_size=8).hexdigest(), n))
                    r = rows_for_keys(inv, S)
                    bill('   %s.%s rows_for_keys %d' % (a, c, r.size)) if _bill is not None else None
                    rows = r if isfull else r[keep[r]]           # unsorted is fine: every consumer is order-free
            if rows is None:
                _kc9 = keycache.get((a, c))
                if _kc9 is None and n >= 1_000_000:
                    # THE SHELVED COLUMN, IN THE CUT TOO: keys() decodes a big key column once per process
                    # and shelves it, but the cut only looked in the per-query keycache -- so every cut of
                    # mc.movie_id / mi.movie_id / ci.movie_id took the road walk (8ms binary searches per
                    # step: 1.0s of keys_at across the COUNT board, all from restrict)
                    try: _kc9 = keys(a, c)
                    except Exception: _kc9 = None
                if isfull:
                    idx = np.arange(n, dtype=np.int64) if _kc9 is None else None
                if _kc9 is not None:
                    dk = np.asarray(_kc9 if isfull else _kc9[idx], dtype=np.int64)
                else:
                    dk = keys_at(a, c, idx) if idx.size * 4 < n else keys(a, c)[idx]
                _mk9 = (segs[a].path, c); mxk = kmax.get(_mk9)
                if mxk is None and _kc9 is not None: mxk = kmax[_mk9] = int(_kc9.max()) if _kc9.size else 0
                mx = int(max(int(S[-1]) if S.size else 0, mxk if mxk is not None else (int(dk.max()) if dk.size else 0)))   # S is sorted: its max is free
                if mx < 200_000_000:
                    lut = np.zeros(mx + 2, bool); lut[S] = True
                    hit = lut[np.where(dk < 0, mx + 1, dk)]
                else:
                    hit = np.isin(dk, S)
                rows = np.flatnonzero(hit) if (isfull and idx is None) else idx[hit]
            cnt = int(rows.size)
            if cnt != counts[a]:
                new = np.zeros_like(keep); new[rows] = True       # one N-scale write per shrink, not three
                keeps[a] = new; counts[a] = cnt; full[a] = False; changed = True
                idx_of[a] = rows
                if _rk9 is not None and 0 < cnt <= 16_000_000:
                    # THE BIG FIRST HOP IS THE ONE WORTH SHELVING: role = 'actor' cuts cast_info to 12.7M
                    # rows in 13 queries -- 51 MB of rows plus a 36 MB keep, against a 32 GB shelf
                    try:
                        _r32 = rows.astype(np.int32 if n < (1 << 31) else np.int64)
                        wdb_shelf.SHELF.put(_rk9, (_r32, new), int(_r32.nbytes + new.nbytes), kind='settled-rows')
                    except Exception: pass
        return changed
    pending = set(alias2t)
    applied = {}
    rounds = 0
    def _refresh_spaces_of(a):
        _tq9 = _tk()
        for c in cols_of[a]:
            r = find((a, c)); old = live.get(r); new = space_live(r)
            if new is not None and (old is None or new.size < old.size):
                live[r] = new
                for (b, _c) in spaces[r]:
                    if b != a: pending.add(b)
        if _bill is not None: _bill.append(('      refresh %s' % a, _tk() - _tq9))
    while rounds < 256:
        rounds += 1
        # anything that landed since the last step streams in now
        if futs is not None:
            for a in list(alias2t):
                if a not in arrived and futs[a].done():
                    if _land(a): _refresh_spaces_of(a); pending.add(a)
                    else: arrived.add(a)
        # ISOLATION FIRST, PER TABLE: a table whose own filter is still running is not restricted
        # from the conjoined space yet -- its own cut is usually deeper and always cheaper (cast_info
        # was cut to 12.7M rows by the role space, 787ms, while its note LIKE was about to cut it to 32K)
        _tq9 = _tk()
        cand = [a for a in pending if any(live.get(find((a, c))) is not None for c in cols_of[a])
                and (futs is None or a in arrived or futs[a].done())]
        if _bill is not None: _bill.append(('      pick', _tk() - _tq9))
        if not cand:
            # nothing to do until a slow isolation lands: wait for the next one
            waiting = [a for a in alias2t if futs is not None and a not in arrived]
            if not waiting: break
            import concurrent.futures as _cf
            _cf.wait([futs[a] for a in waiting], return_when=_cf.FIRST_COMPLETED)
            continue
        def _signal(x):
            sz = [live[find((x, c))].size for c in cols_of[x] if live.get(find((x, c))) is not None]
            return (min(sz) if sz else 1 << 62, counts[x])
        # THE SMALLEST SIGNAL FIRST: the table whose restricting space is smallest goes next (a
        # keyword space of size 1 cuts mk to 24K, which hands title a 24K movie set instead of
        # mc's 1.15M) -- not the smallest table
        a = min(cand, key=_signal)
        pending.discard(a)
        before = counts[a]
        if restrict(a):
            bill('%s %d->%d' % (a, before, counts[a]))
            _refresh_spaces_of(a)                    # a shrank: its spaces may shrink; their members are pending again
        else:
            bill('%s %d (no change)' % (a, before))
    if futs is not None:
        for a in alias2t:                          # every isolation result must be in before the emit
            if a not in arrived:
                if _land(a): _refresh_spaces_of(a)
                arrived.add(a)
        while True:                                # and any shrinks it caused must settle
            cand = [a for a in pending if any(live.get(find((a, c))) is not None for c in cols_of[a])]
            if not cand or rounds >= 512: break
            rounds += 1; a = min(cand, key=lambda x: counts[x]); pending.discard(a); before = counts[a]
            if restrict(a): bill('%s %d->%d (settle)' % (a, before, counts[a])); _refresh_spaces_of(a)
    bill('keyspace rounds=%d' % rounds)


def _needs_weights(tree):
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        if isinstance(nd, (E.Count, E.Sum, E.Avg)): return True
    return tree.args.get('group') is not None


def _counting_emit(db, state):
    """THE COUNTING FIXPOINT (Yannakakis): after the semi-join reduction, the join
    hypergraph's VALUE CLASSES (one per equality-connected set of columns) form a
    join tree. Rooted at the aggregate's table, every surviving row's WEIGHT is
    the product, over the value classes it touches (except the one to its
    parent), of the partner counts in the child subtrees; COUNT(*) is the sum of
    root weights, SUM(T.x) the weighted sum, GROUP BY keys aggregate weights per
    key. No pair is ever built. Acyclic hypergraphs only (declines otherwise)."""
    import wdb_sql
    tree = state['tree']; alias2t = state['alias2t']; segs = state['segs']; pms = state['pms']
    keeps = state['keeps']; counts = state['counts']; pair = state['pair']; owner = state['owner']
    _keys = state['keys']; _keys_at0 = state['keys_at']
    def keys_at(t, c, idx):
        """THE SHELVED COLUMN IN THE WALK: the fixpoint's int32 key columns (decoded once per process)
        make this a gather; the road walk + binary search per row cost 0.94s per 113 COUNT queries"""
        try:
            col = _keys(t, c)
            return np.asarray(col[idx], dtype=np.int64)
        except Exception as _e:
            if os.environ.get('WDB_SEMI_BILL'): print('  walk: keys() declined %s.%s (%s) -> road walk' % (t, c, str(_e)[:60]), flush=True)
            return _keys_at0(t, c, idx)
    if any(c == 0 for c in counts.values()):
        return _empty_result(tree)
    # value classes: union-find over (alias, col)
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b): parent[find(a)] = find(b)
    for (a, b), pairs in pair.items():
        for ca, cb in pairs:
            union((a, ca), (b, cb))
    classes = {}
    for (a, b), pairs in pair.items():
        for ca, cb in pairs:
            r = find((a, ca))
            classes.setdefault(r, set()).update({(a, ca), (b, cb)})
    # hypergraph acyclicity by GYO reduction; also builds the join tree (parent pointers)
    tabs = list(alias2t.keys())
    tab_classes = {t: set() for t in tabs}
    for r, cols in classes.items():
        for (a, c) in cols: tab_classes[a].add(r)
    if any(len(cs) > 1 for cs in tab_classes.values()) and len(tabs) > 1:
        pass
    # pick the root: the table owning the aggregates / group keys (all must agree)
    root = None
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        col = nd.this if isinstance(nd, (E.Sum, E.Avg, E.Min, E.Max)) else (nd if isinstance(nd, E.Column) else None)
        if col is not None:
            o = owner(col)
            if root is not None and o != root: raise _Decline('counting fixpoint: aggregates/keys span several tables')
            root = o
    g = tree.args.get('group')
    if g is not None:
        for k in g.expressions:
            o = owner(k)
            if root is not None and o != root: raise _Decline('counting fixpoint: group keys span several tables')
            root = o
    if root is None: root = tabs[0]
    # join tree by BFS over tables sharing a value class; a table reached twice = a cycle -> decline
    tree_parent = {root: None}; order = [root]; seen_cls = set()
    frontier = [root]
    while frontier:
        nxt = []
        for t in frontier:
            for r in tab_classes[t]:
                if r in seen_cls: continue
                seen_cls.add(r)
                for (a, c) in classes[r]:
                    if a == t: continue
                    if a in tree_parent:
                        raise _Decline('counting fixpoint: cyclic join hypergraph (table %s reached twice)' % a)
                    tree_parent[a] = (t, r); order.append(a); nxt.append(a)
        frontier = nxt
    if len(tree_parent) != len(tabs): raise _Decline('counting fixpoint: disconnected join graph')
    # class column per (table, class)
    def class_col(t, r):
        for (a, c) in classes[r]:
            if a == t: return c
        return None
    # bottom-up: weight per surviving row of each table = product over child classes of the
    # child-side partner weight sums keyed by value
    row_weight = {}
    lut = {}                                # (table, class) -> dense value -> summed weight of that table's rows under the value
    def build_lut(t, r):
        idx = np.flatnonzero(keeps[t])
        kv = keys_at(t, class_col(t, r), idx)
        w = row_weight[t]
        ok = kv >= 0
        kv = kv[ok]; w = w[ok]
        mx = int(kv.max()) if kv.size else 0
        if mx > 400_000_000: raise _Decline('counting fixpoint: key space too large for a dense LUT')
        if kv.size * 64 < mx:
            # A LUT SIZED TO THE LIVE KEYS, NOT THE KEY SPACE: a class with a hundred live keys was
            # allocating and zeroing a 2.5M-entry table (20 MB) per edge -- sparse: (sorted keys, sums).
            # SPARSE TO BUILD IS NOT SPARSE TO LOOK UP: the lookup was a binary search per parent row
            # (801 searchsorted calls, 1.31s of a 1.46s walk); sparse only when keys are 64x rarer than
            # the space -- a calloc'd dense table is cheaper than searching 400K rows into it
            u, inv = np.unique(kv, return_inverse=True)
            lut[(t, r)] = ('sparse', u, np.bincount(inv, weights=w, minlength=u.size))
        else:
            lut[(t, r)] = ('dense', np.bincount(kv, weights=w, minlength=mx + 1))   # bincount: np.add.at is ~20x slower
    for t in reversed(order):
        idx = np.flatnonzero(keeps[t])
        w = np.ones(idx.size, np.float64)
        up = tree_parent[t]
        for r in tab_classes[t]:
            if up is not None and r == up[1]: continue           # the class to the parent is applied by the parent
            # multiply by the product over the OTHER tables in this class of their LUT at this row's value
            kv = keys_at(t, class_col(t, r), idx)
            for (a, c) in classes[r]:
                if a == t: continue
                if (a, r) not in lut: raise _Decline('counting fixpoint: child not reduced before parent')
                L = lut[(a, r)]
                if L[0] == 'dense':
                    acc = L[1]
                    safe = np.where((kv >= 0) & (kv < acc.size), kv, 0)
                    f = acc[safe]; f[(kv < 0) | (kv >= acc.size)] = 0.0
                else:
                    u, vals = L[1], L[2]
                    pos = np.searchsorted(u, kv); pos = np.where(pos < u.size, pos, 0)
                    hit = (u[pos] == kv) & (kv >= 0)
                    f = np.where(hit, vals[pos], 0.0)
                w = w * f
        row_weight[t] = w
        if up is not None:
            build_lut(t, up[1])
    idx = np.flatnonzero(keeps[root]); w = row_weight[root]
    # emit
    names = [wdb_sql._alias(p) for p in tree.expressions]
    seg = segs[root]; pm = pms[root]
    def col_vals(col):
        pc = pm.get(col.name, col.name)
        raw = wdb_sql.raw_dict_col(seg, pc, want_codes=False)
        if raw is not None:
            vals = np.asarray(raw[0], dtype=np.float64)
            if seg.cols[pc].get('has_null'): vals = np.append(vals, np.nan)
            return vals[np.asarray(seg.codes_at(pc, idx))]
        return np.array([(np.nan if v is None else float(v)) for v in seg.values_at_rows(pc, idx)], dtype=np.float64)
    def emit_agg(nd, sel):
        if isinstance(nd, E.Count): return int(round(w[sel].sum()))
        if isinstance(nd, (E.Sum, E.Avg)):
            v = col_vals(nd.this)[sel]; ww = w[sel]; ok = ~np.isnan(v)
            tot = float((v[ok] * ww[ok]).sum()); cnt = float(ww[ok].sum())
            if cnt == 0: return None
            if isinstance(nd, E.Avg): return tot / cnt
            isint = seg.cols.get(pm.get(nd.this.name, nd.this.name), {}).get('dt') == 0
            return int(round(tot)) if isint else tot
        if isinstance(nd, (E.Min, E.Max)):
            v = col_vals(nd.this)[sel] if seg.cols.get(pm.get(nd.this.name, nd.this.name), {}).get('dt') != 1 else None
            if v is None:
                pc = pm.get(nd.this.name, nd.this.name); codes = np.asarray(seg.codes_at(pc, idx))[sel]
                if seg.cols[pc].get('has_null'): codes = codes[codes != int(seg.cols[pc]['V']) - 1]
                if codes.size == 0: return None
                return wdb_sql._pyval(seg._typed_dict(pc)[int(codes.min()) if isinstance(nd, E.Min) else int(codes.max())])
            v = v[~np.isnan(v)]
            if v.size == 0: return None
            r = float(v.min()) if isinstance(nd, E.Min) else float(v.max())
            return int(r) if seg.cols.get(pm.get(nd.this.name, nd.this.name), {}).get('dt') == 0 else r
        raise _Decline('counting fixpoint: aggregate %s' % type(nd).__name__)
    if g is None:
        sel = np.ones(idx.size, bool)
        return [tuple(emit_agg(p.this if isinstance(p, E.Alias) else p, sel) for p in tree.expressions)], names
    # GROUP BY keys of the root table: composite codes -> groups
    kcols = [k.name for k in g.expressions]
    comp = np.zeros(idx.size, np.int64); K = 1; dec = []
    for kc in kcols:
        pc = pm.get(kc, kc); V = int(seg.cols[pc]['V']); codes = np.asarray(seg.codes_at(pc, idx)).astype(np.int64)
        comp = comp * V + codes; K *= V; dec.append((pc, V))
    u, inv = np.unique(comp, return_inverse=True)
    out = []
    for gi in range(u.size):
        sel = inv == gi
        row = []
        rem = int(u[gi]); keyvals = {}
        for pc, V in reversed(dec):
            keyvals[pc] = rem % V; rem //= V
        for p in tree.expressions:
            nd = p.this if isinstance(p, E.Alias) else p
            if isinstance(nd, E.Column):
                pc = pm.get(nd.name, nd.name)
                row.append(wdb_sql._pyval(seg._typed_dict(pc)[keyvals[pc]]) if seg.cols[pc].get('mode') != 5 else list(seg.values_at_rows(pc, idx[sel][:1]))[0])
            else:
                row.append(emit_agg(nd, sel))
        out.append(tuple(row))
    return out, names


def _empty_result(tree):
    import wdb_sql
    names = [wdb_sql._alias(p) for p in tree.expressions]
    if tree.args.get('group') is not None: return [], names
    row = []
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        row.append(0 if isinstance(nd, E.Count) else None)
    return [tuple(row)], names
