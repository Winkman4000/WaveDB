"""
wdb_gdsidecar -- materialized group-wise COUNT(DISTINCT) answer (the "sidecar").

The default shape comes free from the dictionary: encode already enumerated every group, so the
group cardinality V (a ~1us header read) sets the sidecar to cover all groups, one count each. Build
runs the exact single-pass walk ONCE (reusing wdb_groupdistinct), and stores the per-group distinct
counts indexed by group code. Serving a `GROUP BY g, COUNT(DISTINCT t)` query then READS the stored
counts instead of re-walking N rows -- exact, and orders of magnitude faster on repeat.

Group labels are NOT stored: they come back free from the segment's own dictionary at serve time
(codes 0..V-1 -> values). Per-group trim (which groups the sidecar covers) lives in the catalog and
is applied at serve time; trimmed-out groups fall through to the live walk.

Storage = V counts (+ the present-index when the target is nullable). Keyed by (group_col, target_col),
persisted next to the segment as <segment>.gd-<group>-<target>.npz.
"""
import os, json
import numpy as np
import wdb_groupdistinct as gd
import wdb_sql
import wdb_gbcount

E = wdb_sql.E


def sidecar_path(segment_path, group_col, target_col):
    return f"{segment_path}.gd-{group_col}-{target_col}.npz"


def build(seg, group_col, target_col):
    """Run the exact walk once over this segment; return {counts, present, meta} or None.

    Declines (returns None) on exactly the shapes the live walk declines: non-value-identity
    columns (mode-4 positional), nullable group key (v1), or a pack that would overflow int64.
    counts[code] = exact distinct(target) for the group with that code; present = the group codes
    that actually occur (for a nullable target a group can occur with count 0)."""
    if group_col not in seg.cols or target_col not in seg.cols:
        return None
    if seg.cols[group_col].get('has_null'):              # v1: NULL-as-a-group deferred (matches operator)
        return None
    kinfo = gd._ids(seg, group_col)
    tinfo = gd._ids(seg, target_col)
    if kinfo is None or tinfo is None:                   # not value-identity -> walk can't serve it
        return None
    grp, _knull, _kdec = kinfo
    tgt, tnull, _tdec = tinfo
    if grp.shape[0] != tgt.shape[0]:
        return None
    N = int(grp.shape[0])
    gmax = (int(grp.max()) + 1) if N else 0
    k = (int(tgt.max()) + 1) if N else 1
    if N and gmax * k >= (1 << 62):                      # pair-id would overflow int64 -> decline
        return None

    if N == 0:
        counts = np.zeros(gmax, np.int64)
    elif gd._HAVE_NUMBA:
        capbits = max(20, min(28, int(np.ceil(np.log2(max(N, 2)))) + 1))
        counts = gd._walk(grp, tgt, np.int64(k), gmax, np.int64(tnull), capbits)
    else:                                                # numba absent: exact NumPy pack+unique
        keep = (tgt != tnull) if tnull >= 0 else slice(None)
        key = grp[keep].astype(np.int64) * k + tgt[keep].astype(np.int64)
        uq = np.unique(key)
        counts = np.bincount((uq // k).astype(np.int64), minlength=gmax)
    if tnull >= 0:                                        # nullable target: occurs != count>0
        present = np.nonzero(np.bincount(grp, minlength=gmax))[0].astype(np.int64)
    else:
        present = np.nonzero(counts)[0].astype(np.int64)
    meta = {'group_col': group_col, 'target_col': target_col,
            'N': N, 'V': int(gmax), 'target_nullable': bool(tnull >= 0)}
    return {'counts': counts.astype(np.int64, copy=False), 'present': present, 'meta': meta}


def save(segment_path, sc):
    p = sidecar_path(segment_path, sc['meta']['group_col'], sc['meta']['target_col'])
    tmp = p + '.tmp'
    np.savez(tmp, counts=sc['counts'], present=sc['present'],
             meta=np.frombuffer(json.dumps(sc['meta']).encode(), dtype=np.uint8))
    os.replace(tmp + '.npz' if os.path.exists(tmp + '.npz') else tmp, p)  # np.savez appends .npz
    return p


def load(segment_path, group_col, target_col):
    p = sidecar_path(segment_path, group_col, target_col)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    meta = json.loads(bytes(z['meta']).decode())
    return {'counts': z['counts'], 'present': z['present'], 'meta': meta}


def rows_from_sidecar(sc, seg, proj, tree, ci, ki, excluded_codes=None):
    """Assemble result rows from the stored counts -- the read that replaces the walk. `excluded_codes`
    is the per-pair trim (group codes the user left to the walk); they are dropped from the candidate
    set here. Row assembly / ORDER BY / LIMIT reuse the exact same helpers as try_groupdistinct, so the
    served rows are identical to the live operator's."""
    counts = sc['counts']
    present = sc['present']
    if excluded_codes is not None and len(excluded_codes):
        ex = np.asarray(sorted(excluded_codes), dtype=np.int64)
        present = present[~np.isin(present, ex)]
    kdecode = wdb_gbcount._code_values(seg, sc['meta']['group_col'])   # labels: cheap dict read O(V), not O(N)
    sel = present[np.argsort(-counts[present], kind='stable')]
    lim = wdb_sql._limit(tree)
    if lim is not None:
        sel = sel[:lim]
    names = [wdb_sql._alias(p) for p in proj]
    rows = []
    for gid in sel.tolist():
        row = [None, None]
        row[ki] = wdb_sql._pyval(kdecode[gid] if kdecode is not None else np.int64(gid))
        row[ci] = int(counts[gid])
        rows.append(tuple(row))
    rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))
    if lim is not None:
        rows = rows[:lim]
    return rows, names


_NOLIT = object()


def _literal_value(node):
    """Python scalar from a SQL literal node, or _NOLIT if it isn't a plain string/number literal.
    NULL is _NOLIT on purpose: NULL comparisons are three-valued and we won't translate them."""
    if isinstance(node, E.Literal):
        if node.is_string:
            return node.this
        try:
            return int(node.this)
        except ValueError:
            try:
                return float(node.this)
            except ValueError:
                return _NOLIT
    return _NOLIT


def _binary_literal(node):
    """For a comparison whose one side is the group column, return the literal on the other side."""
    l, r = node.this, node.args.get('expression')
    if isinstance(l, E.Column):
        return _literal_value(r)
    if isinstance(r, E.Column):
        return _literal_value(l)
    return _NOLIT


def _excluded_from_group_filter(seg, tree, group_col, col_map, present):
    """Translate a WHERE *confined to the group key* into the set of group codes to DROP from the
    sidecar output. The per-group distinct counts are invariant under a filter that only touches the
    group column -- such a filter only removes whole groups -- so serving the survivors is exact.

    Returns sorted excluded codes, [] when there's no filter, or None to DECLINE (any predicate that
    references another column, or any form we can't translate exactly, falls through to the walk).
    Supported leaves (AND-combined): `col <> v`, `col = v`, `col IN (..)`, `col NOT IN (..)`."""
    where = tree.args.get('where')
    if where is None:
        return []
    pred = where.this
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    for colexpr in pred.find_all(E.Column):
        if sc(colexpr.name) != group_col:
            return None                                  # predicate touches a non-group column -> walk
    leaves = wdb_sql._flatten_and(pred)

    present_set = set(int(x) for x in present.tolist())
    exclude = set()
    keep_only = None                                     # set once any '='/IN narrows to a kept set
    for leaf in leaves:
        if isinstance(leaf, E.NEQ):
            v = _binary_literal(leaf)
            if v is _NOLIT: return None
            exclude |= set(codes_for_values(seg, group_col, [v]))
        elif isinstance(leaf, E.EQ):
            v = _binary_literal(leaf)
            if v is _NOLIT: return None
            ks = set(codes_for_values(seg, group_col, [v]))
            keep_only = ks if keep_only is None else (keep_only & ks)
        elif isinstance(leaf, E.In) and leaf.args.get('query') is None:
            vals = [_literal_value(e) for e in (leaf.expressions or [])]
            if any(v is _NOLIT for v in vals): return None
            ks = set(codes_for_values(seg, group_col, vals))
            keep_only = ks if keep_only is None else (keep_only & ks)
        elif isinstance(leaf, E.Not) and isinstance(leaf.this, E.In) and leaf.this.args.get('query') is None:
            vals = [_literal_value(e) for e in (leaf.this.expressions or [])]
            if any(v is _NOLIT for v in vals): return None
            exclude |= set(codes_for_values(seg, group_col, vals))
        else:
            return None                                  # unhandled predicate form -> walk
    if keep_only is not None:
        kept = (keep_only & present_set) - exclude
        excluded = present_set - kept
    else:
        excluded = exclude & present_set
    return sorted(int(x) for x in excluded)


_SERVE_HITS = 0   # telemetry: queries answered from the materialized sidecar (not the walk)


def detect(db, table, seg, segment_path, tree, col_map):
    """ACTIVATION for the materialized group-distinct sidecar read. Shape (shared gd.detect) plus
    materialization checks: the pair is registered, the sidecar exists on disk, no catalog trim, and
    any WHERE translates to an exact group exclusion. Returns a spec (loaded sidecar + projection info
    + excluded codes) or None. Loading the (cached) sidecar IS the materialization check, so it lives
    here in activation. Fires only when the needed groups are fully covered (v1: every group)."""
    det = gd.detect(seg, tree, col_map, _allow_group_filter=True)
    if det is None:
        return None
    kcol, tcol, ci, ki, proj = det
    entry = db.cat.gd_entry(table, kcol, tcol)
    if entry is None:                                    # pair not materialized -> walk
        return None
    s = db.gd_sidecar(segment_path, kcol, tcol)          # cached load (np.load once, not per query)
    if s is None:                                        # registered but no data on disk -> walk
        return None
    if entry.get('excluded'):                            # catalog trim + query filter don't combine here -> walk
        return None
    excluded = _excluded_from_group_filter(seg, tree, kcol, col_map, s['present'])
    if excluded is None:                                 # a WHERE we can't translate exactly -> walk
        return None
    return {'s': s, 'proj': proj, 'ci': ci, 'ki': ki, 'excluded': excluded, 'tree': tree}


def execute(seg, spec):
    """THE READ: assemble the rows from the materialized sidecar (no decline path -- once detect
    passes, the sidecar fully answers the query)."""
    global _SERVE_HITS
    rows, names = rows_from_sidecar(spec['s'], seg, spec['proj'], spec['tree'],
                                    spec['ci'], spec['ki'], excluded_codes=spec['excluded'])
    _SERVE_HITS += 1
    return rows, names


def try_serve(db, table, seg, segment_path, tree, col_map):
    """Detect + execute, kept as the backward-compatible single-call entry."""
    spec = detect(db, table, seg, segment_path, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)


def codes_for_values(seg, group_col, values):
    """Map group VALUES (what a human sees) to internal group codes, for trimming/filtering. Raw-int
    columns: code == value. Dict columns: resolve only the requested literals in ONE pass over the
    dict (native byte/int compare, no full decode), with early-exit once all are found, and memoize
    results per (segment, column) so repeated filters are O(K). Building the *full* inverse here was
    the bug that made a filtered serve on a 6M-value column take ~5s. Unknown values are skipped."""
    decode = wdb_gbcount._code_values(seg, group_col)
    if decode is None:                                   # raw int: code is the value
        return sorted(int(v) for v in values)
    cache = getattr(seg, '_gd_val2code', None)
    if cache is None:
        cache = {}
        try: seg._gd_val2code = cache
        except Exception: pass
    cc = cache.setdefault(group_col, {})
    string_dict = len(decode) > 0 and isinstance(decode[0], (bytes, bytearray))
    def nk(v):                                           # normalize a filter literal to the dict's key form
        if string_dict:
            return v.encode() if isinstance(v, str) else bytes(v)
        return int(v)
    keys = [nk(v) for v in values]
    out = []; need = set()
    for k in keys:
        if k in cc:
            if cc[k] is not None: out.append(cc[k])
        else:
            need.add(k)
    if need:
        found = {}
        if string_dict:
            for code, val in enumerate(decode):          # single byte-compare pass, early-exit
                vb = bytes(val)
                if vb in need:
                    found[vb] = code
                    if len(found) == len(need): break
        else:
            for code, val in enumerate(decode):
                iv = int(val)
                if iv in need:
                    found[iv] = code
                    if len(found) == len(need): break
        for k in need:
            cc[k] = found.get(k)                          # memoize hits AND misses (None) -> never rescan
            if cc[k] is not None: out.append(cc[k])
    return sorted(int(x) for x in out)


def inspect(seg, segment_path, group_col, target_col, excluded=None):
    """The materialized view a human eyeballs: (group_value, distinct_count) rows, descending by count.
    Reads the stored sidecar -- no walk. `excluded` codes are flagged so you can see what's trimmed."""
    s = load(segment_path, group_col, target_col)
    if s is None:
        return None
    counts = s['counts']; present = s['present']
    decode = wdb_gbcount._code_values(seg, group_col)
    ex = set(excluded or [])
    order = present[np.argsort(-counts[present], kind='stable')]
    rows = []
    for code in order.tolist():
        val = decode[code] if decode is not None else code
        val = val.decode() if isinstance(val, (bytes, bytearray)) else val
        rows.append((val, int(counts[code]), code in ex))
    return rows
