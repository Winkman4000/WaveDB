"""wdb_gridwalk_live -- buffered-mode 2-key COUNT(*) top-K via the gridwalk base + GwMaint.

When a table has ONE cold segment plus a growing hot buffer (buffered-mode inserts), the generic
read goes through wdb_merge: a full GROUP BY over the whole cold segment + a DuckDB pass on the hot
parquet, merged. For the narrow gridwalk shape we can do far less work: the merge only needs cold
counts for the groups that actually appear in hot, and the gridwalk base provides those by per-pair
point-lookup. So we seed a GwMaint from the cold segment ONCE (cached), apply the hot rows, and read
the top-K straight off the maintained structure.

Contract (conservative, always correct): we accelerate only when every hot key value already exists
in the segment's dictionary (value->code via searchsorted on the sorted dict). If the hot buffer
introduces a NEW dictionary value, or the boundary count is tied (ordered), or the LIMIT would need
count-1 cells, we decline (return None) and the caller falls back to wdb_merge -- which is exact.
The base is cached per (segment, pair); each query rebuilds a fresh GwMaint over a COPY of the base
counts and applies the current hot buffer, so it is robust to any hot mutation (append/delete/update),
not just appends.
"""
import numpy as np
import pandas as pd
import wdb_sql
import workers
import wdb_gridwalk as GW
from wdb_gridwalk_maint import GwMaint

_ENABLED = True
_HITS = 0
_BASE_CACHE = {}          # (seg.path,(a,b),N) -> (base_gid, base_cnt, ones_gid, Vb); RAM-resident


def enable():
    global _ENABLED; _ENABLED = True


def disable():
    global _ENABLED; _ENABLED = False


def is_enabled():
    return _ENABLED


def _base(seg, cols):
    """Immutable launch base for a pair on this segment: heavy (gid,cnt), count-1 gids, Vb. Cached."""
    a, b = sorted(cols)
    ck = (seg.path, (a, b), int(seg.N))
    hit = _BASE_CACHE.get(ck)
    if hit is not None:
        return hit
    ca = seg._raw_codes(a).astype(np.int64); cb = seg._raw_codes(b).astype(np.int64)
    if ca.size == 0:
        return None
    Vb = int(cb.max()) + 1
    gid, cnt = np.unique(ca * Vb + cb, return_counts=True)
    heavy = cnt >= 2
    base = (gid[heavy].copy(), cnt[heavy].astype(np.int64), gid[cnt == 1].copy(), Vb)
    _BASE_CACHE[ck] = base
    return base


def _hot_gids(hot_path, a, b, V, Vb):
    """Grid-ids for the hot rows, mapping each key value to its segment code by searchsorted on the
    sorted dict. Returns None if any value is new to a dict (decline) or the hot is unreadable."""
    try:
        df = pd.read_parquet(hot_path, columns=[a, b])
    except Exception:
        return None
    code = {}
    for col in (a, b):
        Vc = V[col]
        vals = df[col].to_numpy()
        if Vc.dtype.kind == 'S':
            key = np.array([v.encode('utf-8', 'surrogatepass') if isinstance(v, str) else v
                            for v in vals], dtype=Vc.dtype)
        else:
            try:
                key = vals.astype(Vc.dtype, copy=False)
            except Exception:
                return None
        pos = np.searchsorted(Vc, key)
        posc = np.clip(pos, 0, Vc.size - 1)
        if Vc.size == 0 or not bool(((pos < Vc.size) & (Vc[posc] == key)).all()):
            return None                 # a value new to the dictionary -> decline
        code[col] = pos.astype(np.int64)
    return code[a] * Vb + code[b]


def try_live(seg, hot_path, tree, col_map):
    """Accelerated buffered 2-key COUNT(*) top-K, or None to fall back to wdb_merge."""
    global _HITS
    if not _ENABLED:
        return None
    if wdb_sql._offset(tree):
        return None
    spec = GW.detect(seg, tree, col_map)
    if spec is None:
        return None
    cols = spec['cols']; lim = spec['lim']; proj = spec['proj']; ci = spec['ci']
    knames = spec['knames']; V = spec['V']; unordered = spec.get('unordered')
    if lim <= 0:
        return None
    base = _base(seg, cols)
    if base is None:
        return None
    base_gid, base_cnt, ones_gid, Vb = base
    a, b = sorted(cols)
    gids = _hot_gids(hot_path, a, b, V, Vb)
    if gids is None:
        return None
    m = GwMaint(base_gid, base_cnt.copy(), ones_gid)   # gid/ones shared (insert never mutates them)
    m.insert(gids)
    allg, allc = m._merged()
    n_heavy = int(allg.size)
    if lim > n_heavy:                  # top-K would need count-1 cells -> merge_query
        return None
    kk = min(lim + 1, n_heavy)
    part = np.argpartition(allc, -kk)[-kk:]
    order_idx = part[np.argsort(allc[part], kind='stable')[::-1]]
    if not unordered and order_idx.size > lim and int(allc[order_idx[lim - 1]]) == int(allc[order_idx[lim]]):
        return None                    # tied boundary (ordered) -> merge_query for the canonical order
    sel = order_idx[:lim]
    sel_gid = allg[sel]; sel_cnt = allc[sel].astype(np.int64)
    # materialize (mirrors wdb_gridwalk.execute: hoisted plan + vectorized coord/value decode)
    codesA = sel_gid // Vb
    codesB = sel_gid - codesA * Vb
    code_by_col = {a: codesA, b: codesB}
    plan = []; decoded = {}
    for pi, p in enumerate(proj):
        if pi == ci:
            plan.append((pi, None)); continue
        col = cols[knames.index(wdb_sql._proj_colname(p))]
        plan.append((pi, col))
        if col not in decoded:
            decoded[col] = [wdb_sql._pyval(x) for x in V[col][code_by_col[col].astype(np.intp)]]
    cnt_list = sel_cnt.tolist()
    nproj = len(proj)
    rows_out = []
    for i in range(sel_gid.size):
        row = [None] * nproj
        for pi, col in plan:
            row[pi] = cnt_list[i] if col is None else decoded[col][i]
        rows_out.append(tuple(row))
    rows_out = workers.finalize(rows_out, proj, spec['order'], lim)
    _HITS += 1
    return rows_out, [wdb_sql._alias(p) for p in proj]
