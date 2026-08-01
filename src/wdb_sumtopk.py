"""
wdb_sumtopk -- single high-card key, SUM top-K: the counting board's SUM sibling.

GROUP BY <big dict key> ORDER BY SUM(<numeric dict col>) DESC LIMIT k paid ~0.9s of
generic gid machinery (factorize-or-dense, casts, radix, prefilter) above a two-stream
floor. Here: both code streams read once, one fused kernel accumulates value-table
gathers into per-thread boards (no casts, no factorize), argpartition finds the k
winners in O(V), and only their labels decode. Small dictionaries stay with fused_agg,
which already wins there.
"""
import numpy as np
import wdb_sql
import wdb_policies as P
import wdb_kernels as WK
import sqlglot.expressions as E

_HITS = 0
_MIN_V = 1_000_000


def detect(seg, tree, col_map):
    if not P.no_joins(tree) or not P.no_having(tree) or not P.no_select_distinct(tree):
        return None
    if tree.args.get('where') is not None or tree.args.get('qualify') is not None:
        return None
    g = tree.args.get('group')
    if g is None or len(g.expressions) != 1 or not isinstance(g.expressions[0], E.Column):
        return None
    lim = wdb_sql._limit(tree)
    if lim is None or lim > 10000:
        return None
    proj = tree.expressions
    if len(proj) != 2:
        return None
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    key = None; sumcol = None; ki = None; si = None
    for pi, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        nm = wdb_sql._colname(inner)
        kd = wdb_sql._agg_kind(p)
        if nm is not None and kd is None:
            key, ki = sc(nm), pi
        elif kd is not None and kd[0] == 'SUM':
            ac = wdb_sql._colname(inner.this)
            if ac is None:
                return None
            sumcol, si = sc(ac), pi
        else:
            return None
    if key is None or sumcol is None:
        return None
    gnm = sc(g.expressions[0].name)
    if gnm != key:
        return None
    order = tree.args.get('order')
    if order is None or len(order.expressions) != 1:
        return None
    o0 = order.expressions[0]
    if not bool(o0.args.get('desc')):
        return None
    onm = wdb_sql._colname(o0.this)
    if onm != wdb_sql._alias(proj[si]) and onm != sumcol:
        return None
    ck, cv = seg.cols.get(key), seg.cols.get(sumcol)
    if ck is None or cv is None:
        return None
    if ck.get('mode') not in (0, 1, 2) or ck.get('has_null'):
        return None
    if cv.get('mode') not in (0, 1, 2) or cv.get('has_null') or cv.get('dt') != 0:
        return None
    if int(ck.get('V') or 0) < _MIN_V:
        return None                                     # small boards: fused_agg already wins
    if not P.no_deleted_rows(seg):
        return None
    return {'key': key, 'val': sumcol, 'lim': lim, 'ki': ki, 'si': si,
            'proj': proj, 'off': wdb_sql._offset(tree) or 0}


def execute(seg, spec):
    global _HITS
    import wdb_window as W
    key, val, lim, off = spec['key'], spec['val'], spec['lim'], spec['off']
    kc = np.asarray(seg._raw_codes(key))
    vc = np.asarray(seg._raw_codes(val))
    vt = np.asarray(W._int_table(seg, val), dtype=np.int64)
    K = int(seg.cols[key]['V'])
    # JACKSON'S FREQUENCY WALK, flag edition: the gbc stores houses count-descending
    # (the itinerary); a CACHE-RESIDENT bool flag (K bytes of query-lifetime scratch,
    # ~9.7MB for ClientIP -- fits the pocket) selects candidate rows; only those pour,
    # compact. Stop line: a skipped house at count c can never beat c * max_value;
    # the shelf's own numbers clear it 9x in round one. One retry, then honest pour.
    fastdone = False
    topvals = None
    if K > 200_000 and P.no_deleted_rows(seg):
        import wdb_gbcount
        loaded = wdb_gbcount._load(seg, key)
        max_v = int(vt.max()) if vt.size else 0
        if loaded is not None and max_v > 0:
            hc = np.asarray(loaded[0]).astype(np.int64)
            hn = np.asarray(loaded[1]).astype(np.int64)
            need0 = lim + off
            B = max(need0 * 64, 8192)
            for _round in (0, 1):
                if B >= hc.size:
                    cand = hc; next_count = 1
                else:
                    cand = hc[:B]; next_count = int(hn[B])
                flag = np.zeros(K, dtype=bool)
                flag[cand] = True
                m = flag[kc]
                kc_sel = kc[m]
                v_sel = vt[np.asarray(vc)[m]]
                cs_sorted = np.sort(cand)
                idx = np.searchsorted(cs_sorted, kc_sel)
                csums = np.bincount(idx, weights=v_sel,
                                    minlength=cs_sorted.size).astype(np.int64)
                if need0 < csums.size:
                    part = np.argpartition(csums, csums.size - need0)[csums.size - need0:]
                else:
                    part = np.arange(csums.size)
                kth = int(csums[part].min()) if part.size else 0
                if kth >= next_count * max_v:
                    top_local = part[np.argsort(csums[part])[::-1]]
                    top_local = top_local[np.lexsort((cs_sorted[top_local],
                                                      -csums[top_local]))][off:off + lim]
                    top = cs_sorted[top_local]
                    topvals = {int(cs_sorted[i]): int(csums[i])
                               for i in top_local.tolist()}
                    fastdone = True
                    break
                B *= 8
    if not fastdone:
        sums = WK.grouped_sum_codes(kc, vc, vt, K)
    need = lim + off
    if fastdone:
        pass
    elif need >= K:
        top = np.argsort(sums)[::-1][:need]
    else:
        part = np.argpartition(sums, K - need)[K - need:]
        top = part[np.argsort(sums[part])[::-1]]
    if not fastdone:
        # ties at the boundary: deterministic: sums desc, then code asc
        top = top[np.lexsort((top, -sums[top]))][off:off + lim]
    rows = []
    ki, si = spec['ki'], spec['si']
    for code in top.tolist():
        v = wdb_sql._pyval(seg.fetch(key, int(code)))
        if isinstance(v, (bytes, bytearray)):
            v = v.decode('utf-8', 'replace')
        row = [None, None]
        row[ki] = v
        row[si] = int(topvals[int(code)]) if topvals is not None else int(sums[code])
        rows.append(tuple(row))
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in spec['proj']]
