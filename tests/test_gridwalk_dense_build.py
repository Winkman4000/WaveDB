"""THE GRID ON EVERY CORE (2026-10-23): wdb_gridwalk._build's dense branch counts the grid with the composed count's
atoms -- the fused per-thread count (grid2's kernel) up to _FUSED_MAX cells, radix_acc + bincount_par past it -- in
place of one-thread numpy glue. Every way must build the identical structure: the head, the counts, Vb, the heavy
count, the bulk, the singletons, the member positions, the distinct count. Oracle: the old numpy build
(_DENSE_PAR off) and DuckDB through SQL."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import wdb_gridwalk as GW
import test_groupself_run as F


def _same(x, y, what):
    if x is None or y is None:
        assert x is None and y is None, what
    elif isinstance(x, (tuple, list)) or isinstance(y, (tuple, list)):
        assert len(x) == len(y), what
        for i, (p, q) in enumerate(zip(x, y)):
            _same(p, q, '%s[%d]' % (what, i))
    elif isinstance(x, dict):
        assert sorted(x) == sorted(y), what
        for kk in x:
            _same(x[kk], y[kk], '%s[%r]' % (what, kk))
    elif isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        assert np.array_equal(np.asarray(x), np.asarray(y)), what
    else:
        assert x == y, (what, x, y)


def _build(cols, dense_par, fused_max):
    old = GW._DENSE_PAR[0], GW._FUSED_MAX[0]
    GW._DENSE_PAR[0], GW._FUSED_MAX[0] = dense_par, fused_max
    try:
        return GW._build(F._fixture()[2], cols)
    finally:
        GW._DENSE_PAR[0], GW._FUSED_MAX[0] = old


def test_every_way_builds_the_same_structure():
    seg = F._fixture()[2]
    pairs = [('u', 'tm'), ('v', 'w'), ('w', 'tm')]
    small = [(a, b) for a in ('u', 'v', 'w', 'tm') for b in ('u', 'v', 'w', 'tm')
             if a < b and int(seg.cols[a]['V']) * int(seg.cols[b]['V']) <= GW._BINC_MAX]
    for cols in pairs + small:
        ref = _build(list(cols), False, 0)
        if ref is None:
            continue
        for fused in (0, 1 << 40):
            got = _build(list(cols), True, fused)
            _same(ref, got, '%s fused=%s' % (cols, fused))


def test_fused_count_against_numpy_on_skewed_small_grids():
    import wdb_kernels
    rng = np.random.default_rng(11)
    for Va, Vb, n in ((2, 3, 1 << 20), (7, 1, 300_000), (19, 250, 2_000_003), (1, 1, 5)):
        a = rng.integers(0, Va, n).astype(np.uint8)
        b = (rng.integers(0, Vb, n) ** 1).astype(np.uint16)
        b[: n // 3] = 0                                     # a heavy value
        want = np.bincount(a.astype(np.int64) * Vb + b, minlength=Va * Vb)
        got = wdb_kernels.grid2_count(a, b, a, np.int64(-1), np.int64(Vb), np.int64(Va * Vb))
        assert np.array_equal(np.asarray(got), want), (Va, Vb, n)


def test_sql_unordered_pairs_still_legal():
    sys.path.insert(0, os.path.dirname(__file__))
    import test_sampletop_census as S
    db, con, _seg = F._fixture()
    for a, b in (('u', 'v'), ('v', 'w'), ('tm', 'nk')):
        for k in (10, 1000):
            q = 'SELECT %s, %s, COUNT(*) FROM t GROUP BY %s, %s LIMIT %d' % (a, b, a, b, k)
            r = db.run(q)
            rows = r[0] if isinstance(r, tuple) else r
            S._legal(a, b, [tuple(S.F._norm(v) for v in x) for x in rows], k)
