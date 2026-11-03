"""THE HAVING IN ARRAY SPACE and THE ONE KEY'S RUNS (probe p118, 2026-11-03, Q11): join GROUP BY queries with a
HAVING over aggregates / COUNT against numeric literals answer like DuckDB with the HAVING tested on the per-group
arrays (AND, OR, nothing passing, AVG, COUNT); a HAVING on a key column keeps the row road; a single fact-direct key
whose codes never decrease takes the monotone accumulator, one that decreases does not; switched off, same answers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

LO = "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
HQ = [
    "SELECT o.o_orderpriority, SUM(l.l_extendedprice * l.l_quantity) AS v " + LO + "WHERE o.o_orderstatus = 'F' "
    "GROUP BY o.o_orderpriority HAVING SUM(l.l_extendedprice * l.l_quantity) > 300000000 ORDER BY v DESC",
    "SELECT l.l_orderkey, COUNT(*) AS c, SUM(l.l_quantity) AS q " + LO + "WHERE o.o_orderstatus = 'O' "
    "GROUP BY l.l_orderkey HAVING COUNT(*) >= 6 AND SUM(l.l_quantity) < 200 ORDER BY q DESC, l.l_orderkey",
    "SELECT l.l_orderkey, SUM(l.l_quantity) AS q " + LO + "WHERE o.o_orderpriority = '1-URGENT' "
    "GROUP BY l.l_orderkey HAVING SUM(l.l_quantity) > 250 OR SUM(l.l_quantity) < 5 ORDER BY l.l_orderkey",
    "SELECT o.o_orderstatus, AVG(l.l_discount) AS d " + LO + "GROUP BY o.o_orderstatus HAVING AVG(l.l_discount) <= 0.05",
    "SELECT l.l_orderkey, SUM(l.l_quantity) AS q " + LO + "WHERE o.o_orderstatus = 'F' "
    "GROUP BY l.l_orderkey HAVING SUM(l.l_quantity) > 100000",
]
KEYQ = ("SELECT l.l_orderkey, SUM(l.l_quantity) AS q " + LO + "WHERE o.o_orderstatus = 'F' "
        "GROUP BY l.l_orderkey HAVING l.l_orderkey > 30000 ORDER BY l.l_orderkey")
RUNQ = ("SELECT l.l_orderkey, SUM(l.l_extendedprice) AS v, COUNT(*) AS c " + LO + "WHERE o.o_orderstatus = 'O' "
        "GROUP BY l.l_orderkey ORDER BY l.l_orderkey")
DOWNQ = ("SELECT l.l_partkey, SUM(l.l_extendedprice) AS v " + LO + "WHERE o.o_orderstatus = 'O' "
         "GROUP BY l.l_partkey ORDER BY l.l_partkey")


def _check(db, con, q):
    import controller, test_join_fast as J
    controller.plan_epoch_bump()
    exp = J._norm([tuple(r) for r in con.execute(q).fetchall()])
    r = db.run(q)
    got = J._norm(r[0] if isinstance(r, tuple) else r)
    k = lambda t: tuple((x is None, str(x)) for x in t)
    assert len(got) == len(exp), (q, len(got), len(exp))
    for g, e in zip(sorted(got, key=k), sorted(exp, key=k)):
        assert J._eq(g, e), (q, g, e)


def test_having_in_array_space():
    import wdb_join as W, test_join_fast as J
    db, con = J._fixture()
    h0 = W._HAVING_ARR_HITS[0]
    for q in HQ:
        _check(db, con, q)
    assert W._HAVING_ARR_HITS[0] >= h0 + len(HQ), 'the array HAVING was not taken'
    h1 = W._HAVING_ARR_HITS[0]
    _check(db, con, KEYQ)                                   # a key column: the row road
    assert W._HAVING_ARR_HITS[0] == h1
    old = W._HAVING_ARR[0]
    try:
        W._HAVING_ARR[0] = False
        for q in HQ:
            _check(db, con, q)
        assert W._HAVING_ARR_HITS[0] == h1
    finally:
        W._HAVING_ARR[0] = old


def test_one_key_runs():
    import wdb_join as W, test_join_fast as J
    db, con = J._fixture()
    oldk = W._ONE_KEY_RUNS_MIN_K
    try:
        W._ONE_KEY_RUNS_MIN_K = 1                           # the fixture's keys are small; the line is the gate
        h0 = W._ONE_KEY_RUNS_HITS[0]
        _check(db, con, RUNQ)                               # lineitem is stored by l_orderkey: never decreases
        assert W._ONE_KEY_RUNS_HITS[0] == h0 + 1
        _check(db, con, DOWNQ)                              # l_partkey steps down: no
        assert W._ONE_KEY_RUNS_HITS[0] == h0 + 1
        old = W._ONE_KEY_RUNS[0]
        try:
            W._ONE_KEY_RUNS[0] = False
            _check(db, con, RUNQ)
            assert W._ONE_KEY_RUNS_HITS[0] == h0 + 1
        finally:
            W._ONE_KEY_RUNS[0] = old
    finally:
        W._ONE_KEY_RUNS_MIN_K = oldk
