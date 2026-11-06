"""THE ROAD AT THE SURVIVORS (2026-11-01): a _LazyRoad answers at(rows) with the composed pointer at those rows
without composing the whole road; np.asarray / slicing / attributes get the full compose, equal to composing
by gather; chained two and three deep. Join answers through grandparent roads equal DuckDB's with WDB_LAZY_ROAD
on and off, and a filtered query asks at the survivors only."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np


def test_road_equals_compose():
    import wdb_join as WJ
    rng = np.random.default_rng(1101)
    N, A, B, C = 50_000, 9_000, 700, 25
    p0 = rng.integers(0, A, N).astype(np.int32)          # fact -> parent
    p1 = rng.integers(0, B, A).astype(np.int32)          # parent -> grandparent
    p2 = rng.integers(0, C, B).astype(np.int64)          # grandparent -> great-grandparent
    r2 = WJ._LazyRoad(p1, p0)
    r3 = WJ._LazyRoad(p2, r2)
    rows = np.sort(rng.choice(N, 3_000, replace=False)).astype(np.int64)
    f0 = WJ._LazyRoad.FULLS[0]
    assert np.array_equal(r2.at(rows), p1[p0[rows]])
    assert np.array_equal(r3.at(rows), p2[p1[p0[rows]]])
    assert np.array_equal(r3[rows], p2[p1[p0[rows]]])               # int index -> at rows
    assert WJ._LazyRoad.FULLS[0] == f0, 'asking at rows must not compose the whole road'
    assert len(r3) == N and r3.shape == (N,) and r3.dtype == p2.dtype and r3.size == N
    assert np.array_equal(np.asarray(r3), p2[p1[p0]])
    assert WJ._LazyRoad.FULLS[0] == f0 + 2                          # r3 composed r2 on the way
    assert np.array_equal(r3[5:9], p2[p1[p0]][5:9])
    assert int(r3.max()) == int(p2[p1[p0]].max())                   # attribute -> the whole road
    assert np.array_equal(r3.at(rows), p2[p1[p0[rows]]])            # after the full: served from it
    m = rng.random(N) < 0.1
    assert np.array_equal(r3[m], p2[p1[p0]][m])
    assert np.array_equal(WJ._at_rows(p0, rows), p0[rows]) and np.array_equal(WJ._at_rows(r2, rows), p1[p0[rows]])


def test_join_answers_both_ways():
    import wdb_join as WJ, controller, test_join_fast as J
    db, con = J._fixture()
    qs = ["SELECT n.n_name, COUNT(*), SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey "
          "WHERE l.l_shipdate >= DATE '1995-01-01' AND l.l_shipdate < DATE '1996-01-01' GROUP BY n.n_name",
          "SELECT r.r_name, COUNT(*), SUM(l.l_quantity) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey "
          "JOIN region r ON n.n_regionkey = r.r_regionkey WHERE l.l_quantity < 10 GROUP BY r.r_name",
          "SELECT n.n_name, COUNT(*) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
          "JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey GROUP BY n.n_name"]
    old = WJ._LAZY_ROAD[0]
    try:
        for q in qs:
            for on in (True, False):
                WJ._LAZY_ROAD[0] = on
                J._match(q, expect_fast=False)
        WJ._LAZY_ROAD[0] = True
        db.run(qs[0])
        f0 = WJ._LazyRoad.FULLS[0]
        db.run(qs[0])
        assert WJ._LazyRoad.FULLS[0] == f0, 'a filtered chain query composes its roads at the survivors only'
    finally:
        WJ._LAZY_ROAD[0] = old
