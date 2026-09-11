"""THE SEMI-JOIN FIXPOINT (the JOB shape): a multi-way equi-join projecting ONLY
MIN/MAX. Under MIN/MAX row multiplication is irrelevant -- a table's
contribution is the extreme over its rows that PARTICIPATE in the join.
Participation is a fixpoint: local filters seed each table's keep; every
equality edge prunes both sides to the other's surviving keys (dense-id
lookup tables, O(N)); iterate to stability; MIN/MAX per column over its
table's survivors (strings by the extreme present dictionary code)."""
import numpy as np
import sqlglot
from sqlglot import exp as E


class _Decline(Exception):
    pass


def shape_ok(tree):
    """MIN/MAX-only projections, comma/INNER joins on plain columns, no group/order/limit/subquery/window."""
    if not isinstance(tree, E.Select): return False
    if tree.args.get('group') is not None or tree.args.get('having') is not None: return False
    if tree.args.get('order') is not None or tree.args.get('limit') is not None: return False
    if tree.find(E.Window) is not None or tree.find(E.Subquery) is not None: return False
    if not tree.expressions: return False
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        if not isinstance(nd, (E.Min, E.Max)) or not isinstance(nd.this, E.Column): return False
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
            mm = np.asarray(wdb_sql._eval_pred(seg, c2, lambda nm, pm=pms[a]: pm.get(nm, nm)), dtype=bool)
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
