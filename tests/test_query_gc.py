"""THE QUERY'S CYCLES DIE WITH IT (2026-10-31): a query leaves nothing behind in reference cycles -- with the automatic
collector off, a collection right after a fused join query finds nothing left to free when the end-of-query pass is
on, and finds the cycles (the fused path's closures) when it is switched off; answers equal DuckDB's either way, and
the cycles do not accumulate over repeated runs."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import test_join_fast as J

Q = ("SELECT l_returnflag, COUNT(*), SUM(l_extendedprice) FROM lineitem, orders WHERE l_orderkey = o_orderkey "
     "AND l_shipmode = 'AIR' AND l_quantity > 5 AND o_orderstatus = 'F' GROUP BY l_returnflag ORDER BY l_returnflag")


def _left_behind(db, n=1):
    """runs the query n times with the automatic collector off; the objects a full collection then finds unreachable"""
    was = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for _ in range(n):
            db.run(Q)
        return gc.collect()
    finally:
        if was:
            gc.enable()


def test_the_query_leaves_no_cycles():
    import wdb_db
    db, con = J._fixture()
    exp = [tuple(r) for r in con.execute(Q).fetchall()]
    old = wdb_db._QUERY_GC[0]
    try:
        wdb_db._QUERY_GC[0] = False
        db.run(Q)
        off = _left_behind(db, 5)
        wdb_db._QUERY_GC[0] = True
        on = _left_behind(db, 5)
        got = db.run(Q)[0]
    finally:
        wdb_db._QUERY_GC[0] = old
    assert off > 0, 'the fused path left no cycles -- the test proves nothing'
    assert on == 0, ('cycles left behind with the pass on', on, off)
    assert len(got) == len(exp)
    for a, b in zip(got, exp):
        assert a[0] == b[0] and int(a[1]) == int(b[1]) and abs(float(a[2]) - float(b[2])) <= 1e-6 * max(1.0, abs(float(b[2])))
