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
    # JACKSON'S FREQUENCY WALK (the Fagin bound, reinvented): the gbc stores houses
    # in count-descending order -- the walk's exact itinerary, already on the shelf.
    # Pour ONLY the top-B houses into a compact board (everyone else into one trash
    # cup: the existing kernel, relabeled -- zero new kernels), then check the stop
    # line: a skipped house at count c can never exceed c * max_value, so once the
    # need-th best candidate clears next_count * max_value the answer is PROVEN.
    # Zipf stops it in one round; the loop widens and re-pours if ever not.
    fastdone = False
    if K > 200_000 and P.no_deleted_rows(seg):
        import wdb_gbcount
        loaded = wdb_gbcount._load(seg, key)
        if loaded is not None:
            hc, hn = loaded
            hc = np.asarray(hc).astype(np.int64); hn = np.asarray(hn).astype(np.int64)
            max_v = int(vt.max()) if vt.size else 0
            need0 = lim + off
            B = max(need0 * 64, 8192)
            while B < K and max_v > 0:
                if B >= hc.size:
                    cand = hc; next_count = 1
                else:
                    cand = hc[:B]; next_count = int(hn[B])
                rank = np.full(K, cand.size, dtype=np.int64)
                rank[cand] = np.arange(cand.size)
                cups = WK.grouped_sum_codes(rank[kc], vc, vt, int(cand.size) + 1)
                cs = cups[:cand.size]
                if need0 < cs.size:
                    part = np.argpartition(cs, cs.size - need0)[cs.size - need0:]
                else:
                    part = np.arange(cs.size)
                kth = int(cs[part].min()) if part.size else 0
                if kth >= next_count * max_v:
                    sums_sp = cs
                    top_local = part[np.argsort(cs[part])[::-1]]
                    top_local = top_local[np.lexsort((cand[top_local],
                                                      -cs[top_local]))][off:off + lim]
                    top = cand[top_local]
                    sums = None
                    topvals = {int(cand[i]): int(cs[i]) for i in top_local.tolist()}
                    fastdone = True
                    break
                B *= 8
    if not fastdone:
        sums = WK.grouped_sum_codes(kc, vc, vt, K)
        topvals = None
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
