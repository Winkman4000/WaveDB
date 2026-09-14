"""THE SEMI-JOIN FIXPOINT (the JOB shape): a multi-way equi-join projecting ONLY
MIN/MAX. Under MIN/MAX row multiplication is irrelevant -- a table's
contribution is the extreme over its rows that PARTICIPATE in the join.
Participation is a fixpoint: local filters seed each table's keep; every
equality edge prunes both sides to the other's surviving keys (dense-id
lookup tables, O(N)); iterate to stability; MIN/MAX per column over its
table's survivors (strings by the extreme present dictionary code)."""
import numpy as np
import wdb_shelf
import sqlglot
from sqlglot import exp as E


class _Decline(Exception):
    pass


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


def execute(db, tree):
    import wdb_sql, os, time
    _bill = [] if os.environ.get('WDB_SEMI_BILL') else None
    _tk = time.perf_counter; _t0 = _tk()
    from wdb_join import _solo_segment, _FastUnsupported, _bulk_keyvals
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
    def owner(col):
        if col.table: return col.table
        cands = [a for a, cs in cols_of.items() if col.name in cs]
        if len(cands) != 1: raise _Decline('ambiguous column %s' % col.name)
        return cands[0]
    edges, local = [], {a: [] for a in alias2t}
    for c in conds:
        if isinstance(c, E.EQ) and isinstance(c.this, E.Column) and isinstance(c.expression, E.Column):
            a, b = owner(c.this), owner(c.expression)
            if a != b:
                edges.append((a, c.this.name, b, c.expression.name)); continue
        owners = {owner(x) for x in c.find_all(E.Column)}
        if len(owners) != 1: raise _Decline('multi-table non-equality conjunct: %s' % c.sql()[:50])
        local[owners.pop()].append(c)
    # segments, local keeps
    segs, pms, keeps = {}, {}, {}
    for a, t in alias2t.items():
        try:
            seg, _ = _solo_segment(db, t)
        except _FastUnsupported:
            raise _Decline('multi-segment table %s' % t)
        segs[a] = seg; pms[a] = db.cat.phys_map(t)
        m = None
        for c in local[a]:
            c2 = c.copy()
            for col in c2.find_all(E.Column):
                col.set('table', None)
            # THE PREDICATE SHELF: a local predicate's result on an immutable segment is a fact --
            # kept as an index list on the shelf across queries (JOB's 113 queries reuse the same
            # few filters on the same big tables: mi.info LIKE ... on 14.8M rows was 392ms, every
            # time). A segment with tombstones or overrides never gets here (_solo_segment).
            _pk9 = ('pred', seg.path, c2.sql())
            _hit9 = wdb_shelf.SHELF.get(_pk9)
            if _hit9 is not None:
                mm = np.zeros(int(seg.N), bool); mm[_hit9] = True
            else:
                mm = np.asarray(wdb_sql._eval_pred(seg, c2, lambda nm, pm=pms[a]: pm.get(nm, nm)), dtype=bool)
                _il9 = np.flatnonzero(mm).astype(np.int32 if seg.N < (1 << 31) else np.int64)
                try: wdb_shelf.SHELF.put(_pk9, _il9, int(_il9.nbytes), kind='predicate')
                except Exception: pass
            m = mm if m is None else (m & mm)
        keeps[a] = m if m is not None else np.ones(int(seg.N), bool)
        if _bill is not None: _bill.append(('local %s(%d) keep=%d' % (a, int(seg.N), int(keeps[a].sum())), _tk() - _t0)); _t0 = _tk()
    # key columns as int64 arrays (NULL -> -1)
    keycache = {}
    def keys(a, col):
        k = (a, col)
        if k in keycache: return keycache[k]
        seg = segs[a]; pc = pms[a].get(col, col); cd = seg.cols.get(pc)
        if cd is None: raise _Decline('no such column %s.%s' % (a, col))
        if cd.get('dt') != 0: raise _Decline('non-integer join key %s.%s' % (a, col))
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
        if inv9 is None and int(seg.N) >= 1_000_000:
            inv9 = inverted(a, [col])
        if inv9 is not None:
            u9, offs9, _o9, rank9 = inv9
            pos9 = np.asarray(rank9[idx]).astype(np.int64)
            return u9[np.searchsorted(offs9, pos9, side='right') - 1]    # THE REVERSE ROAD answers point reads
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
        if len(cols_a) != 1: return None
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
        import wdb_sidecar
        try:
            if fn and __import__('os').path.exists(fn + '.rank.npy') \
                    and wdb_sidecar.is_fresh(__import__('os').path.dirname(path), __import__('os').path.basename(fn + '.rank.npy')):
                # THE BIRTHMARK: a sidecar older than its segment is false by construction
                u = np.load(fn + '.u.npy'); offs = np.load(fn + '.offs.npy')
                order = np.load(fn + '.order.npy', mmap_mode='r'); rank = np.load(fn + '.rank.npy', mmap_mode='r')
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
        rank = np.empty_like(order); rank[order] = np.arange(order.size, dtype=order.dtype)   # row -> position
        cache[pc] = (u, offs, order, rank)
        if fn:
            try:
                _os9 = __import__('os')
                wdb_sidecar.may_birth(_os9.path.dirname(path), int(u.nbytes + offs.nbytes + order.nbytes + rank.nbytes),
                                      'reverse road %s.%s' % (_os9.path.basename(path), pc))   # THE DISK GATE
                for nm9, arr9 in (('u', u), ('offs', offs), ('order', order), ('rank', rank)):
                    np.save(fn + '.%s.tmp.npy' % nm9, arr9)
                    _os9.replace(fn + '.%s.tmp.npy' % nm9, fn + '.%s.npy' % nm9)
                cache[pc] = (u, offs, np.load(fn + '.order.npy', mmap_mode='r'), np.load(fn + '.rank.npy', mmap_mode='r'))
            except wdb_sidecar.BirthRefused as _e:
                print('SEMI: %s -- serving from RAM this query' % str(_e)[:120], flush=True)
            except Exception as _e:
                if __import__('os').environ.get('WDB_SEMI_BILL'):
                    print('SEMI: inverted sidecar save failed for %s: %s' % (fn, str(_e)[:80]), flush=True)
        return cache[pc]
    def rows_for_keys(inv, sk):
        """THE ROWS OF A KEY SET: the reverse road's postings for the keys in sk, as row positions --
        cost proportional to the rows matched, no N-scale bitmap"""
        u, offs, order, _rank = inv
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
    counts = {a: int(np.count_nonzero(k)) for a, k in keeps.items()}
    if os.environ.get('WDB_KEYSPACE', '1') != '0':
        _keyspace_fixpoint(alias2t, segs, edges, keeps, counts, keys, keys_at, inverted, rows_for_keys, pack, _bill, _tk)
        _t0 = _tk()
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
        codes = np.asarray(seg.codes(pc))[rows] if cd.get('mode') in (0, 1, 2) else None
        if codes is not None:
            if cd.get('has_null'):
                codes = codes[codes != int(cd['V']) - 1]
            if codes.size == 0:
                out.append(None); continue
            k = int(codes.min()) if isinstance(nd, E.Min) else int(codes.max())     # sorted dictionary
            v = seg._typed_dict(pc)[k]
            out.append(wdb_sql._pyval(v))
        else:
            vals = list(seg.values_at_rows(pc, rows))
            vals = [v for v in vals if v is not None]
            out.append(wdb_sql._pyval(min(vals) if isinstance(nd, E.Min) else max(vals)) if vals else None)
    names = [wdb_sql._alias(p) for p in tree.expressions]
    if _bill is not None:
        _bill.append(('emit', _tk() - _t0))
        print('SEMI BILL: ' + ' | '.join('%s=%.0fms' % (n, v * 1000) for n, v in _bill), flush=True)
    return [tuple(out)], names


def _keyspace_fixpoint(alias2t, segs, edges, keeps, counts, keys, keys_at, inverted, rows_for_keys, pack, _bill, _tk):
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
    idx_of = {}        # THE LIVE ROW LIST per table, kept beside the bool keep: never re-derived per step
    def live_idx(a):
        if a not in idx_of: idx_of[a] = np.flatnonzero(keeps[a])
        return idx_of[a]
    def keyset(a, c):
        if full[a]: return None
        ck = (a, c, counts[a])                     # cached per (table, column, live count): a key set
        if ck in kcache: return kcache[ck]         # only changes when its table shrinks
        idx = live_idx(a)
        if idx.size == 0: return np.zeros(0, np.int64)
        k = keys_at(a, c, idx) if idx.size * 4 < n_of[a] else keys(a, c)[idx]
        k = k[k >= 0]
        if k.size == 0:
            out = np.zeros(0, np.int64)
        else:
            mx = int(k.max())
            if mx < 200_000_000:
                # A KEY SET IS A BITMAP, NOT A SORT: keys are dense ids; np.unique on 2.7M positions
                # cost 1.27s, twice per key set, three key sets for cast_info (5.1s of a query)
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
        if mx < 200_000_000:
            m = np.zeros(mx + 1, bool); m[cur] = True
            return ks[m[ks]]
        return np.intersect1d(cur, ks, assume_unique=True)
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
    for r in spaces: live[r] = space_live(r)
    bill('spaces %d' % len(spaces))
    # ---- the worklist: tables whose spaces restrict them, smallest first ----
    def restrict(a):
        """re-derive a's rows against the live sets of its spaces; True if it shrank"""
        changed = False
        for c in cols_of[a]:
            r = find((a, c)); S = live.get(r)
            if S is None: continue
            keep = keeps[a]; n = n_of[a]
            if counts[a] == 0: return changed
            idx = live_idx(a)
            rows = None
            if n >= 1_000_000 and S.size * 8 < n and S.size * 4 < idx.size:
                # the giant, cut by a small space: the road's postings, then only those already live
                inv = inverted(a, [c])
                if inv is not None:
                    r = rows_for_keys(inv, S)
                    rows = r[keep[r]] if idx.size < n else r
                    rows.sort()
            if rows is None:
                dk = keys_at(a, c, idx) if idx.size * 4 < n else keys(a, c)[idx]
                mx = int(max(S.max() if S.size else 0, dk.max() if dk.size else 0))
                if mx < 200_000_000:
                    lut = np.zeros(mx + 2, bool); lut[S] = True
                    hit = lut[np.where(dk < 0, mx + 1, dk)]
                else:
                    hit = np.isin(dk, S)
                rows = idx[hit]
            cnt = int(rows.size)
            if cnt != counts[a]:
                new = np.zeros_like(keep); new[rows] = True       # one N-scale write per shrink, not three
                keeps[a] = new; counts[a] = cnt; full[a] = False; changed = True
                idx_of[a] = rows
        return changed
    pending = set(alias2t)
    rounds = 0
    while pending and rounds < 64:
        rounds += 1
        # the smallest table among those that could be restricted goes first; the giant waits
        cand = [a for a in pending if any(live.get(find((a, c))) is not None for c in cols_of[a])]
        if not cand: break
        # THE SMALLEST SIGNAL FIRST: the table whose restricting space is smallest goes next (a
        # keyword space of size 1 cuts mk to 24K, which hands title a 24K movie set instead of
        # mc's 1.15M) -- not the smallest table
        def _signal(x):
            sz = [live[find((x, c))].size for c in cols_of[x] if live.get(find((x, c))) is not None]
            return (min(sz) if sz else 1 << 62, counts[x])
        a = min(cand, key=_signal)
        pending.discard(a)
        before = counts[a]
        if restrict(a):
            bill('%s %d->%d' % (a, before, counts[a]))
            # a shrank: its spaces may shrink; every member table of a shrunk space is pending again
            for c in cols_of[a]:
                r = find((a, c)); old = live.get(r); new = space_live(r)
                if new is not None and (old is None or new.size < old.size):
                    live[r] = new
                    for (b, _c) in spaces[r]:
                        if b != a: pending.add(b)
        else:
            bill('%s %d (no change)' % (a, before))
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
    keeps = state['keeps']; counts = state['counts']; pair = state['pair']; keys_at = state['keys_at']; owner = state['owner']
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
        acc = np.zeros(mx + 1, np.float64)
        np.add.at(acc, kv, w) if kv.size < 2_000_000 else None
        if kv.size >= 2_000_000:
            acc = np.bincount(kv, weights=w, minlength=mx + 1).astype(np.float64)
        lut[(t, r)] = acc
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
                acc = lut[(a, r)]
                safe = np.where((kv >= 0) & (kv < acc.size), kv, 0)
                f = acc[safe]; f[(kv < 0) | (kv >= acc.size)] = 0.0
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
