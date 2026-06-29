"""wdb_gridwalk_maint: incremental inserts stay EXACTLY equal to a full rebuild.

Covers the three insert paths (bump an existing heavy cell, promote a launch singleton across 1->2,
append a brand-new pair), tie-aware top-K equality vs a numpy group-by rebuild across several LIMITs
and several insert rounds, recompact round-trips (still exact, and exact again after further inserts),
the singleton-promotion exactness that the `ones` set exists for, and edges. Self-contained, CI."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from wdb_gridwalk_maint import GwMaint


def _rebuild_top(ca, cb, Vb, lim):
    # gridwalk serves heavy cells (count>=2); count-1 cells are out of its scope (scan's job),
    # so the ground truth for top-K is the heavy cells only.
    g, c = np.unique(np.asarray(ca) * Vb + np.asarray(cb), return_counts=True)
    heavy = c >= 2
    g, c = g[heavy], c[heavy]
    k = min(lim, g.size)
    if k == 0:
        return g[:0], c[:0]
    idx = np.argpartition(c, -k)[-k:]
    idx = idx[np.argsort(c[idx], kind='stable')[::-1]]
    return g[idx[:lim]], c[idx[:lim]]


def _counts(c):
    return [int(x) for x in c]


def _ident_above(g, c):
    if len(c) == 0:
        return set()
    bnd = int(min(int(x) for x in c))
    return set(int(gi) for gi, ci in zip(g, c) if int(ci) > bnd)


def _match(mg, mc, rg, rc):
    return _counts(mc) == _counts(rc) and _ident_above(mg, mc) == _ident_above(rg, rc)


def test_maint_three_insert_paths_exact():
    Vb = 100
    # base: (1,1)x3 heavy, (2,2)x2 heavy, (3,3)x1 singleton, (4,4)x1 singleton
    baseA = np.array([1, 1, 1, 2, 2, 3, 4]); baseB = np.array([1, 1, 1, 2, 2, 3, 4])
    m, vb = GwMaint.from_codes(baseA, baseB, Vb)
    assert vb == Vb and m.heavy_size() == 2 and m.og.size == 2
    # bump (1,1)->4 ; promote (3,3) 1->2 ; brand-new (5,5)x2 -> heavy 2
    insA = np.array([1, 3, 5, 5]); insB = np.array([1, 3, 5, 5])
    m.insert_codes(insA, insB, Vb)
    mg, mc = m.topk(5)
    rg, rc = _rebuild_top(np.r_[baseA, insA], np.r_[baseB, insB], Vb, 5)
    assert _match(mg, mc, rg, rc), (mc.tolist(), rc.tolist())
    assert _counts(mc) == [4, 2, 2, 2]            # the 4 heavy cells; the count-1 (4,4) is out of scope


def test_maint_singleton_promotion_is_exact():
    """The reason `ones` exists: a singleton inserted once must read as 2, not 1."""
    Vb = 50
    m, _ = GwMaint.from_codes(np.array([7]), np.array([9]), Vb)   # one launch singleton (7,9)
    assert m.heavy_size() == 0 and m.og.size == 1
    m.insert_codes(np.array([7]), np.array([9]), Vb)             # +1 -> true count 2
    mg, mc = m.topk(1)
    assert _counts(mc) == [2], mc.tolist()
    # brand-new pair needs two inserts to become heavy
    m.insert_codes(np.array([3, 3]), np.array([3, 3]), Vb)
    _, mc2 = m.topk(2)
    assert sorted(_counts(mc2), reverse=True) == [2, 2]


def test_maint_random_rounds_and_recompact():
    rng = np.random.default_rng(3); Vb = 500
    baseA = rng.integers(0, 40, 5000); baseB = rng.integers(0, 40, 5000)
    m, _ = GwMaint.from_codes(baseA, baseB, Vb)
    allA = baseA.copy(); allB = baseB.copy()
    for r in range(4):
        insA = rng.integers(0, 45, 2000); insB = rng.integers(0, 45, 2000)   # 45>40 -> some brand-new
        m.insert_codes(insA, insB, Vb)
        allA = np.r_[allA, insA]; allB = np.r_[allB, insB]
        for lim in (3, 10, 25):
            mg, mc = m.topk(lim); rg, rc = _rebuild_top(allA, allB, Vb, lim)
            assert _match(mg, mc, rg, rc), ('live', r, lim, mc.tolist(), rc.tolist())
        before = (m.bg.copy(), m.bc.copy(), m.og.copy())
        m.recompact()
        assert m.tail_size() == 0
        for lim in (3, 10, 25):                                   # still exact after recompact
            mg, mc = m.topk(lim); rg, rc = _rebuild_top(allA, allB, Vb, lim)
            assert _match(mg, mc, rg, rc), ('post-recompact', r, lim)
        # recompact preserved the full cell population (heavy + singletons), no loss
        assert m.bg.size + m.og.size == np.unique(allA * Vb + allB).size


def test_maint_edges():
    Vb = 10
    m, _ = GwMaint.from_codes(np.array([1, 1]), np.array([2, 2]), Vb)   # one heavy cell (1,2)=2
    m.insert(np.empty(0, np.int64))                                     # empty insert is a no-op
    _, mc = m.topk(5); assert _counts(mc) == [2]                        # lim > #cells
    g, c = m.topk(0); assert g.size == 0 and c.size == 0
