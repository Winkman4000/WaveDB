"""wdb_counted: THE COUNTED DISTINCT (2026-10-24, Jackson's pick).

SELECT DISTINCT k1..kn FROM t [WHERE ...] [ORDER BY ...] [LIMIT ...] returns exactly the rows of
SELECT k1..kn, COUNT(*) FROM t [WHERE ...] GROUP BY k1..kn [ORDER BY ...] [LIMIT ...] with the count dropped
(NULL is one value either way). The counted form reaches the reads built for GROUP BY -- dict_count, stair,
smallk, fused_agg -- where the bare DISTINCT reached diskpair or the general scan's DISTINCT path. Measured, hot,
the same answers every shape (distgb.py, 2026-10-24):
  DISTINCT ResponseEndTiming + 1             diskpair 0.459 s (general scan 65.1)  -> dict_count 0.087-0.093
  DISTINCT minute(EventTime)                 diskpair 0.327 s (general scan 57.9)  -> stair      0.010-0.018
  DISTINCT minute(EventTime) WHERE IsRefresh = 0   general scan 61.0 s             -> wherescan  2.24-2.35
  DISTINCT ResponseEndTiming, Robotness      diskpair 1.262 s (fused_agg 2.44)     -> smallk     0.37-0.41
  DISTINCT ResponseEndTiming, minute         diskpair 1.156 s                      -> fused_agg  0.408
  DISTINCT length(URL)                       diskpair 0.852 s (general scan 63.9)  -> general scan 1.07-1.15
and where fused_agg already took the DISTINCT itself, it stays faster (RegionID, TraficSourceID, IsMobile 0.094 vs
0.219; FetchTiming 0.046 vs 0.081; SearchPhrase WHERE IsRefresh = 0 3.30 vs 102.6), so the counted form goes in
only at two places:
  1. AHEAD OF DISKPAIR, on the shapes diskpair would take: the counted form through the read order (no general
     scan); when no read takes it, diskpair answers as before (length(URL)'s case).
  2. BEFORE THE GENERAL SCAN, for any bare DISTINCT nothing took: the counted form through the read order, the
     general scan included -- its GROUP BY path, not its DISTINCT path (1.1 s against 64).
WDB_COUNTED_DISTINCT=0 turns both off.

THE KEYS-ONLY GROUP BY (2026-10-24, the GROUP BY holes): SELECT k FROM t GROUP BY k with no aggregate is the same
question again -- before the general scan it is asked with a hidden COUNT(*) (fallback), the count dropped."""
import os
import sqlglot.expressions as E

_HITS = [0]          # answers by the counted form, ahead of diskpair
_FALLBACK = [0]      # answers by the counted form, before the general scan
_HIDDEN = '__dn'


def enabled():
    return os.environ.get('WDB_COUNTED_DISTINCT', '1') != '0'


def bare_distinct(tree):
    """SELECT DISTINCT over keys only: no DISTINCT ON, GROUP BY, HAVING, aggregate, window, star or literal key"""
    if not isinstance(tree, E.Select):
        return False
    d = tree.args.get('distinct')
    if d is None or d.args.get('on') is not None:
        return False
    if tree.args.get('group') is not None or tree.args.get('having') is not None:
        return False
    if not tree.expressions:
        return False
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        if isinstance(nd, (E.Star, E.Literal)) or nd.find(E.Star) is not None:
            return False
        if nd.find(E.AggFunc) is not None or nd.find(E.Window) is not None:
            return False
    return True


def keys_only_group(tree):
    """SELECT keys FROM t [WHERE] GROUP BY keys -- no aggregate anywhere, no HAVING, no DISTINCT, no ROLLUP / CUBE /
    GROUPING SETS / GROUP BY ALL, no window, no star"""
    if not isinstance(tree, E.Select) or tree.args.get('distinct') is not None:
        return False
    g = tree.args.get('group')
    if g is None or not g.expressions or tree.args.get('having') is not None:
        return False
    if g.args.get('all') or g.args.get('rollup') or g.args.get('cube') or g.args.get('grouping_sets'):
        return False
    if not tree.expressions:
        return False
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        if isinstance(nd, E.Star) or nd.find(E.Star) is not None:
            return False
        if nd.find(E.AggFunc) is not None or nd.find(E.Window) is not None:
            return False
    o = tree.args.get('order')
    if o is not None and (o.find(E.AggFunc) is not None or o.find(E.Window) is not None):
        return False
    return True


def counted_group_ctx(ctx):
    """a keys-only GROUP BY's counted request: the same tree, COUNT(*) projected last under a hidden name"""
    import read_methods
    t = ctx.tree.copy()
    t.set('expressions', list(t.expressions) +
          [E.Alias(this=E.Count(this=E.Star()), alias=E.Identifier(this=_HIDDEN, quoted=False))])
    return read_methods.ReadContext(ctx.db, ctx.name, ctx.seg, ctx.path, t, ctx.cmap, t.sql(), ctx.esc)


def counted_ctx(ctx):
    """the counted form's request: the same keys grouped, COUNT(*) projected last under a hidden name"""
    import read_methods
    t = ctx.tree.copy()
    t.set('distinct', None)
    cols = set(ctx.cmap or ()) | set(ctx.seg.cols)
    keys = []
    for p in t.expressions:
        if isinstance(p, E.Alias):
            nd = p.this
            same_col = isinstance(nd, E.Column) and nd.name == p.alias
            # group by the alias (the form measured) unless it names a different table column
            keys.append(E.column(p.alias) if (same_col or p.alias not in cols) else nd.copy())
        else:
            keys.append(p.copy())
    t.set('group', E.Group(expressions=keys))
    t.set('expressions', list(t.expressions) +
          [E.Alias(this=E.Count(this=E.Star()), alias=E.Identifier(this=_HIDDEN, quoted=False))])
    return read_methods.ReadContext(ctx.db, ctx.name, ctx.seg, ctx.path, t, ctx.cmap, t.sql(), ctx.esc)


def _drop_count(res):
    rows, hdr = (res if isinstance(res, tuple) else (res, None))
    import wdb_colresult
    if isinstance(rows, wdb_colresult.ColRows) and rows._mat is None:
        rows = wdb_colresult.ColRows(rows._cols[:-1], rows._n)
    else:
        rows = [tuple(r)[:-1] for r in rows]
    if hdr is not None:
        return rows, list(hdr)[:-1]
    return rows


def walk(ctx2, scan):
    """the counted request through the read order (the general scan last when scan) -> (result, read name) or None"""
    import controller, read_methods as R
    rp = controller._plan_replay(ctx2)
    if rp is not None:
        return rp, controller._SERVED[0]
    for read in controller._READ_ORDER:
        spec = read.detect(ctx2)
        if spec is None:
            continue
        rows = read.execute(ctx2, spec)
        if rows is not None:
            controller._plan_store(ctx2, read.name, spec)
            return rows, read.name
    if scan:
        return R.general_scan(ctx2), 'general_scan'
    return None


def detect(ctx):
    """ahead of diskpair: a bare DISTINCT diskpair would take"""
    if not enabled() or not bare_distinct(ctx.tree):
        return None
    import wdb_diskpair
    dspec = wdb_diskpair.detect(ctx.seg, ctx.tree, ctx.cmap)
    if dspec is None or not wdb_diskpair.answers(ctx.seg, dspec):
        return None                                  # not a shape diskpair would answer: the order goes on
    return {'n': len(ctx.tree.expressions)}


def execute(ctx, spec):
    got = walk(counted_ctx(ctx), scan=False)
    if got is None:
        return None                                  # no read takes the counted form: diskpair answers
    res, _name = got
    _HITS[0] += 1
    return _drop_count(res)


def fallback(ctx):
    """before the general scan: a bare DISTINCT, or a keys-only GROUP BY (2026-10-24: the general scan's GROUP BY with no
    aggregate -- the minute 8.8 s, ResponseEndTiming x minute 14.3, SearchPhrase 116.8 -- against COUNT(*)'s stair
    0.016), nothing took -> (result, read name) or None"""
    if not enabled():
        return None
    if bare_distinct(ctx.tree):
        c2 = counted_ctx(ctx)
    elif keys_only_group(ctx.tree):
        c2 = counted_group_ctx(ctx)
    else:
        return None
    got = walk(c2, scan=True)
    if got is None:
        return None
    res, name = got
    _FALLBACK[0] += 1
    return _drop_count(res), name
