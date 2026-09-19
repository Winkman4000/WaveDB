"""
wdb_gbcount — pre-aggregated count projection for high-cardinality single-key GROUP BY COUNT(*).

A filter-free `SELECT key, COUNT(*) FROM t GROUP BY key ORDER BY COUNT(*) DESC LIMIT N` never needs a
scan: the per-group counts are fixed between writes. We persist them once (sorted by count, heavy
hitters only — singletons are an implicit count of 1), so the query becomes a top-N read instead of a
full-table scan + high-cardinality accumulator scatter.

This is the cube idea extended to high-card keys. try_cube runs first and answers the low-card cases
from a small dense cube; try_gbcount runs right after and answers the high-card single-COUNT(*) case
the cube declines. Same contract as wdb_cube: try_gbcount(seg, tree, col_map) -> (rows, colnames) or
None (caller falls through to the scan paths). Fail-closed on anything outside its exact shape.

Scope (v1): single segment (inherited from the caller's len(segs)==1 gate), one non-mode-4 key whose
codes are value-identity, projections exactly {bare key, COUNT(*)}, no WHERE/HAVING/DISTINCT/JOIN,
ORDER BY the COUNT descending + LIMIT N within the stored heavy-hitter set, no deleted rows. The
sidecar is built lazily on first eligible query and persisted next to the segment as <seg>.<col>.gbc,
keyed by column — general across any table/column. Staleness-guarded by seg.N.
"""
import os, pickle, numpy as np
import wdb_qmem
import wdb_sql
import workers
import wdb_policies as P
E = wdb_sql.E

_HITS = 0   # telemetry: queries answered from a count projection
_CACHE = wdb_qmem.register({})  # (seg.path, col, N) -> (codes, counts), so a repeated query never re-reads the sidecar


def _path(seg, col):
    return f"{seg.path}.{col}.gbc"


def _fetchable(seg, col):
    """Can this column's values be point-fetched by code (seg.fetch)? A pure capability
    check -- touches NO data. The old version materialized the ENTIRE dictionary here
    (dict_vals: 18.3M string reconstructions for URL) so the emit could index ten codes.
    The rule: only decode when needed, never more than needed -- the emit point-fetches."""
    c = seg.cols[col]
    if c['mode'] == 4:
        return False
    if c['dt'] == 1 and c['mode'] in (0, 1):            # string dict: restart-walk fetch
        return True
    if c['dt'] == 0 and c['mode'] == 2:                 # high-card int: nline/dict fetch
        return True
    if c['dt'] == 0 and c['mode'] in (0, 1):            # int-valued byte dict: plain point fetch
        return True                                     # (the j-dim-grp inner: RegionID dt0 mode0)
    return False


class _FetchDecoder:
    """Lazy by-code decoder: indexable like a materialized dictionary, but every [code]
    is a point-fetch (restart-walk for strings, nline/dict for ints). The rule: only
    decode when needed, never more than needed. len() is n_dict -- metadata, no data."""
    __slots__ = ('_seg', '_col', '_n')

    def __init__(self, seg, col):
        self._seg = seg; self._col = col
        c = seg.cols[col]
        self._n = int(c.get('n_dict') or c.get('V') or 0)

    def __len__(self):
        return self._n

    def __getitem__(self, code):
        return self._seg.fetch(self._col, int(code))


def _code_values(seg, col):
    """Shared by gdsidecar/groupdistinct/survgroup: a LAZY by-code decoder (or None).
    Materializes nothing -- the old version decoded ENTIRE dictionaries here."""
    return _FetchDecoder(seg, col) if _fetchable(seg, col) else None


def _build(seg, col):
    """Per-group counts, sorted by count descending, heavy hitters (count>=2) only. Returns
    (codes uint32, counts int64, K, N) or None. The count is already computed when the dictionary is
    built at encode time; here we just (re)materialize and persist it."""
    codes = seg._raw_codes(col)
    if codes.size == 0:
        return None
    K = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=K)
    order = np.argsort(counts, kind='stable')[::-1]      # count descending
    keep = counts[order] >= 2                            # singletons are implicit (count 1), don't store
    hc = np.ascontiguousarray(order[keep], dtype=np.uint32)
    hn = np.ascontiguousarray(counts[order][keep], dtype=np.int64)
    return hc, hn, K, int(seg.N)


def _load(seg, col):
    """Load the persisted sidecar (rebuilding if absent or stale vs seg.N, caching in memory).
    Returns (codes, counts) or None if the column can't be projected."""
    ck = (seg.path, col, int(seg.N))
    hit = _CACHE.get(ck)
    if hit is not None:
        return hit
    p = _path(seg, col)
    if os.path.exists(p):
        try:
            hc, hn, K, n = pickle.load(open(p, 'rb'))
            if n == int(seg.N):
                _CACHE[ck] = (hc, hn)
                return hc, hn
        except Exception:
            pass
    built = _build(seg, col)
    if built is None:
        return None
    hc, hn, K, n = built
    try:
        import wdb_sidecar
        if wdb_sidecar.births_on(os.path.dirname(p)):                       # THE SWITCH
            pickle.dump((hc, hn, K, n), open(p, 'wb'), protocol=4)
    except Exception:
        pass
    _CACHE[ck] = (hc, hn)
    return hc, hn


def _count_index(proj):
    """Index of the single COUNT(*) projection, or None if not exactly one."""
    ci = None
    for i, p in enumerate(proj):
        ak = wdb_sql._agg_kind(p)
        if ak is not None and ak[0] == 'COUNT_STAR':
            if ci is not None:
                return None
            ci = i
    return ci


def _order_is_count_desc(tree, proj, ci):
    """True iff the primary ORDER BY is the COUNT(*) projection, descending — which makes the
    count-descending sidecar prefix contain the answer."""
    order = tree.args.get('order')
    if order is None or not order.expressions:
        return False
    first = order.expressions[0]
    if not isinstance(first, E.Ordered) or not first.args.get('desc'):
        return False
    tgt = first.this
    alias = wdb_sql._alias(proj[ci])
    if isinstance(tgt, E.Column) and tgt.name == alias:
        return True
    ak = wdb_sql._agg_kind(tgt)
    return ak is not None and ak[0] == 'COUNT_STAR'


def _scalar_count_detect(seg, tree, col_map):
    """SELECT COUNT(*) FROM t WHERE col IN/NOT IN (lits) | col = lit | col <> lit --
    pure shelf arithmetic. With no_deleted_rows, a code absent from the gbc but present
    in the dictionary has EXACTLY one row (singletons are trimmed, zeros can't exist),
    so N - sum(excluded) stays exact without touching a single row."""
    if not P.no_joins(tree) or tree.args.get('group') is not None:
        return None
    if tree.args.get('order') is not None or tree.args.get('having') is not None:
        return None
    proj = tree.expressions
    if len(proj) != 1:
        return None
    kd = wdb_sql._agg_kind(proj[0])
    if kd is None or kd[0] != 'COUNT_STAR':
        return None
    w = tree.args.get('where')
    if w is None:
        return None
    node = w.this
    neg = False
    if isinstance(node, E.Not):
        node = node.this
        neg = True
    lits = None; colnode = None
    if isinstance(node, E.In):
        colnode = node.this
        lits = [x.this for x in node.expressions if isinstance(x, E.Literal)]
        if len(lits) != len(node.expressions):
            return None
    elif isinstance(node, (E.EQ, E.NEQ)) and isinstance(node.expression, E.Literal):
        colnode = node.this
        lits = [node.expression.this]
        neg = neg ^ isinstance(node, E.NEQ)
    rng = None
    if lits is None and not neg:
        # value ranges: BETWEEN rewrites to GTE+LTE by sqlglot; single-sided too.
        # The dict is value-sorted, so a range is a contiguous code span.
        lo = hi = None; lo_inc = hi_inc = True
        if isinstance(node, E.Between) and isinstance(node.this, E.Column) \
                and isinstance(node.args.get('low'), E.Literal) \
                and isinstance(node.args.get('high'), E.Literal):
            try:
                node = E.And(this=E.GTE(this=node.this.copy(), expression=node.args['low']),
                             expression=E.LTE(this=node.this.copy(), expression=node.args['high']))
            except Exception:
                pass
        atoms = [node] if not isinstance(node, E.And) else list(node.flatten())
        okr = True; rcol = None
        for a in atoms:
            if isinstance(a, (E.GTE, E.GT, E.LTE, E.LT)) and isinstance(a.expression, E.Literal) \
                    and isinstance(a.this, E.Column):
                if rcol is None:
                    rcol = a.this.name
                elif rcol != a.this.name:
                    okr = False; break
                try:
                    v = float(a.expression.this)
                except (TypeError, ValueError):
                    okr = False; break
                if isinstance(a, (E.GTE, E.GT)):
                    lo, lo_inc = v, isinstance(a, E.GTE)
                else:
                    hi, hi_inc = v, isinstance(a, E.LTE)
            else:
                okr = False; break
        if okr and rcol is not None and (lo is not None or hi is not None):
            colnode = E.Column(this=E.Identifier(this=rcol))
            colnode.set('this', E.to_identifier(rcol))
            rng = (lo, lo_inc, hi, hi_inc)
            lits = []
    lk = None
    if lits is None and rng is None:
        import wdb_wherescan as _WS
        got_lk = _WS._like(node)
        if got_lk is not None:
            lcol, needle, kind, lneg = got_lk
            if kind != 'general':                # dict-testable patterns only
                colnode = E.column(lcol)
                lk = (needle, kind)
                neg = neg ^ lneg
                lits = []
    if (lits is None and rng is None and lk is None) or not isinstance(colnode, E.Column):
        return None
    col = (col_map or {}).get(colnode.name, colnode.name) if col_map else colnode.name
    c = seg.cols.get(col)
    if c is None or c.get('has_null') or c.get('mode') not in (0, 1, 2):
        return None
    if not P.no_deleted_rows(seg):
        return None
    if not _fetchable(seg, col):
        return None
    if rng is not None and seg.cols[col].get('dt') != 0:
        return None                                     # value order is integer business
    return {'scalar': True, 'col': col, 'lits': lits, 'neg': neg,
            'rng': rng, 'like': lk, 'proj': proj}


def _scalar_count_execute(seg, spec):
    global _HITS
    got = _load(seg, spec['col'])
    if got is None:
        return None
    hc, hn = got
    import wdb_wherescan
    if spec.get('like') is not None:
        # Jackson's cut: COUNT + LIKE never needs a row byte. The dict
        # answers WHICH codes match (once, memoized); the gbc shelf answers
        # HOW MANY rows each has (absent code = exactly one, the singleton
        # rule). flag . census -- zero scans, any encoding.
        needle, kind = spec['like']
        flag = wdb_wherescan._like_flags(seg, spec['col'], needle, kind)
        if flag is None:
            return None
        flag = np.asarray(flag, dtype=bool)
        hca = np.asarray(hc)
        m = flag[hca]                            # heavy codes that match
        total = int(np.asarray(hn)[m].sum()) + int(flag.sum()) - int(m.sum())
        ans = (int(seg.N) - total) if spec['neg'] else total
        _HITS += 1
        return [(ans,)], [wdb_sql._alias(spec['proj'][0])]
    import wdb_wherescan
    if spec.get('rng') is not None:
        import wdb_window as _WN
        t = np.asarray(_WN._int_table(seg, spec['col']), dtype=np.int64)
        if t.size < 2 or not bool(np.all(np.diff(t) >= 0)):
            return None                                  # unproven value order: decline
        lo, lo_inc, hi, hi_inc = spec['rng']
        c0 = 0 if lo is None else int(np.searchsorted(t, lo, side='left' if lo_inc else 'right'))
        c1 = t.size if hi is None else int(np.searchsorted(t, hi, side='right' if hi_inc else 'left'))
        if c1 <= c0:
            ans = 0
        else:
            hca = np.asarray(hc)
            m = (hca >= c0) & (hca < c1)
            span = c1 - c0
            on_shelf = int(m.sum())
            ans = int(np.asarray(hn)[m].sum()) + (span - on_shelf)  # trimmed singletons: one each
        _HITS += 1
        return [(ans,)], [wdb_sql._alias(spec['proj'][0])]
    total = 0
    for lit in spec['lits']:
        code = wdb_wherescan._code_of(seg, spec['col'], lit)
        if code is None:
            continue                                     # not in the dictionary: zero rows
        m = np.flatnonzero(np.asarray(hc) == code)
        total += int(np.asarray(hn)[m[0]]) if m.size else 1   # trimmed singleton: exactly one
    ans = (int(seg.N) - total) if spec['neg'] else total
    _HITS += 1
    return [(ans,)], [wdb_sql._alias(spec['proj'][0])]


def _echo_detect(seg, tree, col_map):
    """SELECT col FROM t WHERE col = lit -- the projection IS the filter column, so
    every emitted value IS the literal: count from the shelf, zero row reads."""
    if not P.no_joins(tree) or tree.args.get('group') is not None:
        return None
    if tree.args.get('order') is not None or tree.args.get('having') is not None:
        return None
    if not P.no_select_distinct(tree):
        return None
    proj = tree.expressions
    if len(proj) != 1:
        return None
    pc = wdb_sql._proj_colname(proj[0])
    if pc is None:
        return None
    w = tree.args.get('where')
    if w is None or not isinstance(w.this, E.EQ):
        return None
    node = w.this
    if not isinstance(node.expression, E.Literal):
        return None
    if not isinstance(node.this, E.Column):
        return None
    wc = node.this.name                      # the module's own idiom (scalar detect):
    col = (col_map or {}).get(wc, wc)        # .name + col_map resolution
    col_p = (col_map or {}).get(pc, pc)
    if col != col_p:
        return None
    if not P.columns_exist(seg, col) or not P.no_deleted_rows(seg):
        return None
    lim = None
    lx = tree.args.get('limit')
    if lx is not None:
        try:
            lim = int(lx.expression.this)
        except Exception:
            return None
    return {'echo': True, 'col': col, 'lit': node.expression.this,
            'lim': lim, 'proj': proj}


def detect(seg, tree, col_map):
    ec = _echo_detect(seg, tree, col_map)
    if ec is not None:
        return ec
    sc = _scalar_count_detect(seg, tree, col_map)
    if sc is not None:
        return sc
    """ACTIVATION for the count-projection read. A pure decision over query shape +
    segment metadata -- touches no row data. Returns a spec dict the read needs, or
    None to decline. Self-validating, so it's robust called on its own (the controller
    calls this to route; try_gbcount calls it too)."""
    # --- shared shape guards (wdb_policies); filter-free COUNT(*) top-N ---
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    excl_lit = None
    w = tree.args.get('where')
    if w is not None:
        # admit exactly: key <> literal (one conjunct) -- served by cell exclusion
        cj = w.this
        if not isinstance(cj, E.NEQ) or not isinstance(cj.expression, E.Literal):
            return None
        if wdb_sql._colname(cj.this) is None:
            return None
        excl_lit = cj.expression.this
        excl_key = wdb_sql._colname(cj.this)
    having_min = None
    h = tree.args.get('having')
    if h is not None:
        # admit exactly: COUNT(*) > lit (lit >= 2: the heavy list is complete there)
        hc_ = h.this
        if not isinstance(hc_, E.GT) or wdb_sql._agg_kind(hc_.this) is None:
            return None
        if wdb_sql._agg_kind(hc_.this)[0] != 'COUNT_STAR':
            return None
        if not isinstance(hc_.expression, E.Literal):
            return None
        having_min = float(hc_.expression.this)
        if having_min < 2:
            return None
    if not P.single_group_key(tree):   return None
    # bounded top-N is the home shape; the UNBOUNDED full-counts shape is also
    # servable when the dictionary is small (heavy list + implicit singletons =
    # the complete answer) -- born of join orientation's inner rewrite, which
    # asks for full single-key counts with no limit
    unbounded = not P.has_limit(tree)
    group = tree.args.get('group')
    lim = wdb_sql._limit(tree)
    proj = tree.expressions
    if len(proj) != 2:
        return None
    ci = _count_index(proj)
    if ci is None:
        return None
    ki = 1 - ci
    kp = proj[ki]
    if wdb_sql._agg_kind(kp) is not None:               # the other projection must be the bare key
        return None
    knm = wdb_sql._proj_colname(kp)
    shift = 0
    if knm is None:
        # affine bijection: (key +/- int) groups IDENTICALLY to key -- shifting every
        # sticker's number changes nothing about which stickers tie together. Serve
        # from the key's counts; shift only the emitted winners' labels.
        inner = kp.this if isinstance(kp, E.Alias) else kp
        if isinstance(inner, (E.Add, E.Sub)) and isinstance(inner.this, E.Column) \
                and isinstance(inner.expression, E.Literal):
            try:
                lv = float(inner.expression.this)
            except (TypeError, ValueError):
                lv = None
            if lv is not None and lv.is_integer():
                knm = inner.this.name
                shift = int(lv) if isinstance(inner, E.Add) else -int(lv)
    gnm = wdb_sql._colname(group.expressions[0])
    galias = wdb_sql._alias(kp)
    if knm is None or gnm is None or (gnm != knm and gnm != galias):
        return None
    if excl_lit is not None and (excl_key != knm or shift):
        return None                                     # exclusion must be on the key itself
    col = col_map.get(knm, knm) if col_map else knm
    # --- shared segment/column guards (wdb_policies) ---
    if not P.columns_exist(seg, col):  return None
    if not P.not_positional(seg, col): return None
    if not P.no_deleted_rows(seg):     return None      # deleted rows make stored counts stale
    if shift and seg.cols[col].get('dt') != 0:
        return None                                     # label shifting is integer business
    if unbounded:
        if tree.args.get('order') is not None:
            return None                                 # unbounded serve emits any order
        col_ = col_map.get(knm, knm) if col_map else knm
        c_ = seg.cols.get(col_)
        if c_ is None or int(c_.get('V') or 1 << 30) > 65536:
            return None                                 # big dicts: unbounded stays scan-side
    elif not _order_is_count_desc(tree, proj, ci):
        return None
    if not _fetchable(seg, col):                        # capability only: NO data touched
        return None
    return {'col': col, 'ci': ci, 'ki': ki, 'lim': lim, 'proj': proj,
            'order': tree.args.get('order'), 'excl_lit': excl_lit,
            'having_min': having_min, 'unbounded': unbounded, 'shift': shift,
            'off': wdb_sql._offset(tree)}


def _emit_key(seg, col, code, shift):
    v = wdb_sql._pyval(seg.fetch(col, int(code)))
    if shift:
        return int(v) + shift
    return v


def execute(seg, spec):
    """THE READ: pull the persisted count projection and emit the top-N rows. May still
    decline (return None) on measured boundary conditions that need the loaded sidecar --
    a LIMIT past the stored heavy hitters, or a tie straddling the LIMIT boundary."""
    global _HITS
    if spec.get('echo'):
        import wdb_wherescan as WS
        kc = WS._code_of(seg, spec['col'], spec['lit'])
        if kc is None:
            rows = []
        else:
            cnt = None
            try:
                import wdb_gbshelf
                sh = wdb_gbshelf.open_shelf(seg, spec['col'])
                if sh is None and wdb_gbshelf.birth(seg, spec['col']):
                    sh = wdb_gbshelf.open_shelf(seg, spec['col'])
                if sh is not None:
                    cnt = wdb_gbshelf.point(sh, int(kc))   # tiered-absence shelf:
            except Exception:                              # mmap-open, 3 bit tests
                cnt = None
            if cnt is None:
                loaded = _load(seg, spec['col'])
                if loaded is None:
                    return None
                hc, hn = loaded
                m = hc == int(kc)
                cnt = int(hn[m][0]) if m.any() else 1   # shelf singleton law: in-dict,
            v = wdb_sql._pyval(seg.fetch(spec['col'], int(kc)))   # off-shelf = 1 row
            if isinstance(v, (bytes, bytearray)):
                v = v.decode('utf-8', 'replace')
            n = cnt if spec['lim'] is None else min(cnt, spec['lim'])
            rows = [(v,)] * n
        _HITS += 1
        return rows, [wdb_sql._alias(p) for p in spec['proj']]
    if spec.get('scalar'):
        return _scalar_count_execute(seg, spec)
    col = spec['col']; ci = spec['ci']; ki = spec['ki']; lim = spec['lim']
    proj = spec['proj']
    loaded = _load(seg, col)
    if loaded is None:
        return None
    hc, hn = loaded
    if spec.get('excl_lit') is not None:
        import wdb_wherescan as WS
        kc = WS._code_of(seg, col, spec['excl_lit'])
        if kc is not None:
            keep = hc != int(kc)                        # one cell out; count-desc order kept
            hc, hn = hc[keep], hn[keep]
    if spec.get('unbounded'):
        V = int(seg.cols[col]['V'])
        cnt = np.ones(V, np.int64)                      # dict codes appear >= 1;
        cnt[hc] = hn                                    # absent from heavy == exactly 1
        hm2 = spec.get('having_min')
        rows = []
        for code in range(V):
            n = int(cnt[code])
            if hm2 is not None and not n > hm2:
                continue
            row = [None, None]
            row[ki] = _emit_key(seg, col, code, spec.get('shift', 0))
            row[ci] = n
            rows.append(tuple(row))
        _HITS += 1
        return rows, [wdb_sql._alias(p) for p in proj]
    hm = spec.get('having_min')
    if hm is not None:
        qual = int(np.count_nonzero(hn > hm))           # hn is count-desc: a clean prefix
        hc, hn = hc[:qual], hn[:qual]
        lim = min(lim, qual)                            # fewer qualifiers than LIMIT is a
        if lim == 0:                                    # legitimate short answer, not a decline
            _HITS += 1
            return [], [wdb_sql._alias(p) for p in proj]
    off = int(spec.get('off') or 0)
    need = lim + off
    if hm is None and need > hn.size:                   # would need singletons: fall through
        return None
    if need < hn.size and int(hn[need - 1]) == int(hn[need]):
        return None                                     # a tie straddles the window's END: the emitted
                                                        # SET is ambiguous -> defer to the scan path so
                                                        # tie-breaking stays consistent with the engine
    if 0 < off < hn.size and int(hn[off - 1]) == int(hn[off]):
        return None                                     # a tie straddles the window's START: same law
    rows = []
    for code, n in zip(hc[off:need].tolist(), hn[off:need].tolist()):
        row = [None, None]
        row[ki] = _emit_key(seg, col, code, spec.get('shift', 0))  # decode ONLY the N emitted
        row[ci] = int(n)                                           # keys: point-fetch, never the dict
        rows.append(tuple(row))
    rows = workers.finalize(rows, proj, spec['order'], lim)
    _HITS += 1
    return rows, [wdb_sql._alias(p) for p in proj]


def try_gbcount(seg, tree, col_map):
    """Detect + execute, kept as the backward-compatible single-call entry (read_methods
    and the tests call this). The controller will eventually call detect()/execute() directly."""
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)
