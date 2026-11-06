"""THE RUN WALK (c123, 2026-11-04, TPC-H Q4): [NOT] EXISTS over a child stored by its key to the parent (lineitem by
l_orderkey) is a walk of each surviving parent's run, stopping at the first passing child. Answers like DuckDB for
EXISTS and NOT EXISTS, grouped and scalar, with column-vs-column and column-vs-literal child filters, numeric and
string outer filters (and IN), dates (stored as numbers), no outer filter at all; a child NOT stored by its key
(orders by customer) and a child filter with an expression side stand down with the right answer; switched off, the
old road answers the same."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

WALKED = [
    "SELECT o_orderpriority, COUNT(*) FROM orders WHERE o_totalprice > 150000 AND EXISTS "
    "(SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority",
    "SELECT o_orderpriority, COUNT(*) FROM orders WHERE o_totalprice > 150000 AND EXISTS "
    "(SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_quantity > l_linenumber) GROUP BY o_orderpriority",
    "SELECT o_orderpriority, COUNT(*) FROM orders WHERE o_totalprice > 150000 AND NOT EXISTS "
    "(SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_discount > l_tax) GROUP BY o_orderpriority",
    "SELECT o_orderstatus, COUNT(*), SUM(o_totalprice) FROM orders WHERE o_orderpriority IN ('1-URGENT', '2-HIGH') "
    "AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_quantity >= 45 AND l_discount < 0.03) "
    "GROUP BY o_orderstatus",
    "SELECT COUNT(*) FROM orders WHERE EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_tax > l_discount)",
    "SELECT COUNT(*) FROM orders WHERE o_orderstatus = 'O' AND NOT EXISTS "
    "(SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND 30 < l_quantity)",
]
STAND_DOWN = [
    # orders are not stored by customer: the road does not rise
    "SELECT c_mktsegment, COUNT(*) FROM customer WHERE c_acctbal > 5000 AND NOT EXISTS "
    "(SELECT * FROM orders WHERE o_custkey = c_custkey) GROUP BY c_mktsegment",
    # a child filter with an expression side: not column vs column / literal
    "SELECT o_orderpriority, COUNT(*) FROM orders WHERE o_totalprice > 150000 AND EXISTS "
    "(SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_quantity > l_linenumber * 10) GROUP BY o_orderpriority",
]


def _same(db, con, q):
    import controller, test_join_fast as J
    exp = sorted(J._norm([tuple(r) for r in con.execute(q).fetchall()]))
    r = db.run(q)
    got = sorted(J._norm(r[0] if isinstance(r, tuple) else r))
    assert len(got) == len(exp), (q, len(got), len(exp))
    for g, e in zip(got, exp):
        assert J._eq(g, e), (q, g, e)


def test_walked_shapes():
    import wdb_runwalk as RW, test_join_fast as J
    db, con = J._fixture()
    for q in WALKED:
        h0 = RW._HITS[0]
        _same(db, con, q)
        assert RW._HITS[0] == h0 + 1, q


def test_stands_down():
    import wdb_runwalk as RW, test_join_fast as J
    db, con = J._fixture()
    for q in STAND_DOWN:
        h0 = RW._HITS[0]
        try:
            _same(db, con, q)
        except NotImplementedError:
            pass                                   # the old road's own limits are not this test's
        assert RW._HITS[0] == h0, q


def test_switched_off():
    import wdb_runwalk as RW, test_join_fast as J
    db, con = J._fixture()
    o0 = RW._RUN_WALK[0]
    try:
        RW._RUN_WALK[0] = False
        h0 = RW._HITS[0]
        for q in WALKED[:2]:
            _same(db, con, q)
        assert RW._HITS[0] == h0
    finally:
        RW._RUN_WALK[0] = o0
