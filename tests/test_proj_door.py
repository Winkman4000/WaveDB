"""THE PROJECTION DOOR + THE PER-KEY EXTREME (c121, 2026-11-03, TPC-H Q2): plain-projection pointer-chain joins answer
like DuckDB, in DuckDB's order, through the door -- with LIMIT and without; Q2's per-key MIN / MAX over the same rows
answers like DuckDB, the key in runs (lineitem by order) and not (orders by customer); a subquery over other rows is
not the per-key extreme's (since c128 the value per row answers it, like DuckDB); switched off, the old roads give the
same answers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

PERKEY = [
    "SELECT c_name, n_name, o_orderkey, o_totalprice FROM customer, orders, nation WHERE c_custkey = o_custkey "
    "AND c_nationkey = n_nationkey AND n_name = 'GERMANY' AND o_orderpriority = '1-URGENT' AND o_totalprice = "
    "(SELECT MAX(o_totalprice) FROM orders WHERE o_custkey = c_custkey AND o_orderpriority = '1-URGENT') "
    "ORDER BY o_totalprice DESC, c_name LIMIT 20",
    "SELECT o_orderkey, l_linenumber, l_quantity FROM lineitem, orders WHERE l_orderkey = o_orderkey AND "
    "o_orderstatus = 'F' AND l_quantity = (SELECT MIN(l_quantity) FROM lineitem WHERE l_orderkey = o_orderkey) "
    "ORDER BY o_orderkey, l_linenumber LIMIT 50",
]
PLAIN = [
    "SELECT c_name, o_totalprice, o_orderdate FROM customer, orders WHERE c_custkey = o_custkey AND "
    "o_orderpriority = '1-URGENT' AND c_mktsegment = 'BUILDING' ORDER BY o_totalprice DESC, c_name LIMIT 20",
    "SELECT n_name, c_name FROM customer, nation WHERE c_nationkey = n_nationkey AND n_name = 'PERU' "
    "AND c_acctbal > 9000 ORDER BY c_name",
]
OTHER = ("SELECT c_name, o_orderkey FROM customer, orders WHERE c_custkey = o_custkey AND o_orderpriority = '1-URGENT' "
         "AND o_totalprice = (SELECT MAX(o_totalprice) FROM orders WHERE o_custkey = c_custkey) ORDER BY c_name")


def _ordered(db, con, q):
    import controller, test_join_fast as J
    controller.plan_epoch_bump()
    exp = J._norm([tuple(r) for r in con.execute(q).fetchall()])
    r = db.run(q)
    got = J._norm(r[0] if isinstance(r, tuple) else r)
    assert len(got) == len(exp), (q, len(got), len(exp))
    for g, e in zip(got, exp):                   # every query here orders by a unique tail: the order is the answer
        assert J._eq(g, e), (q, g, e)


def test_per_key_extreme():
    import wdb_projdoor as P, test_join_fast as J
    db, con = J._fixture()
    h0, k0 = P._PROJ_DOOR_HITS[0], P._PER_KEY_HITS[0]
    for q in PERKEY:
        _ordered(db, con, q)
    assert P._PER_KEY_HITS[0] == k0 + len(PERKEY) and P._PROJ_DOOR_HITS[0] == h0 + len(PERKEY)


def test_plain_projection():
    import wdb_projdoor as P, test_join_fast as J
    db, con = J._fixture()
    h0 = P._PROJ_DOOR_HITS[0]
    for q in PLAIN:
        _ordered(db, con, q)
    assert P._PROJ_DOOR_HITS[0] == h0 + len(PLAIN)


def test_other_rows_and_switch():
    import sqlglot, wdb_projdoor as P, test_join_fast as J
    db, con = J._fixture()
    import wdb_corrrow as CR
    k0, c0 = P._PER_KEY_HITS[0], CR._HITS[0]
    exp = sorted(J._norm([tuple(r) for r in con.execute(OTHER).fetchall()]))
    r = db.run(OTHER)
    got = sorted(J._norm(r[0] if isinstance(r, tuple) else r))
    assert len(got) == len(exp) and all(J._eq(g, e) for g, e in zip(got, exp))
                                                 # the subquery's rows are not the outer's: not the per-key extreme,
    assert P._PER_KEY_HITS[0] == k0 and CR._HITS[0] == c0 + 1   # the value per row (c128)
    old = P._PROJ_DOOR[0]
    try:
        P._PROJ_DOOR[0] = False
        h0 = P._PROJ_DOOR_HITS[0]
        # the pandas tail answers PLAIN[1]; PLAIN[0]'s date comes back from it as a day count (10087), a known
        # old-road gap the door does not have
        for q in PLAIN[1:]:
            _ordered(db, con, q)
        assert P._PROJ_DOOR_HITS[0] == h0
    finally:
        P._PROJ_DOOR[0] = old
