"""THE VALUE PER ROW (c128, 2026-11-05, TPC-H Q20): a comparison against a correlated scalar subquery -- one table,
one aggregate, equalities to the outer row plus conditions of its own -- judged at the survivors: the survivors' keys,
the inner rows (its own conditions + the narrowest key set), each inner row's key among the survivors' keys, the
aggregate per key. The projection door takes such a comparison (one table or a join) and an IN whose inner query holds
one (Q20's three levels). Answers like DuckDB: Q20 in order, the middle query alone, each aggregate, one equality, the
subquery on the left, a key with no inner rows (NULL: not true; COUNT: 0), a join outer; switched off, it never fires."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

YEAR = "l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'"
SUMQ = ("(SELECT 0.5 * SUM(l_quantity) FROM lineitem WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey AND "
        + YEAR + ")")
MIDDLE = ("SELECT ps_suppkey FROM partsupp WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%') "
          "AND ps_availqty > " + SUMQ)
Q20 = ("SELECT s_name, s_address FROM supplier, nation WHERE s_suppkey IN (" + MIDDLE + ") AND s_nationkey = n_nationkey "
       "AND n_name = 'CANADA' ORDER BY s_name")
PS = "SELECT ps_partkey, ps_suppkey, ps_availqty FROM partsupp WHERE ps_partkey < 300 AND "
ONE_TABLE = [
    PS + "ps_availqty > (SELECT 40 * COUNT(*) FROM lineitem WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey)",
    PS + "ps_availqty < (SELECT 200 * AVG(l_quantity) FROM lineitem WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey "
         "AND l_discount > 0.03)",
    PS + "ps_supplycost >= (SELECT MIN(l_extendedprice) / 10 FROM lineitem WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey)",
    PS + "ps_availqty <= (SELECT MAX(l_quantity) * 100 FROM lineitem WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey)",
    PS + "ps_availqty > (SELECT SUM(l_quantity) FROM lineitem WHERE l_partkey = ps_partkey AND l_quantity > 40)",   # one equality
    PS + "(SELECT 0.5 * SUM(l_quantity) FROM lineitem WHERE l_suppkey = ps_suppkey AND l_partkey = ps_partkey AND "
         + YEAR + ") < ps_availqty",                                                                                # on the left
    PS + "ps_availqty > (SELECT SUM(l_quantity) FROM lineitem WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey AND "
         "l_shipdate > DATE '2030-01-01')",                                                                         # no inner rows
    PS + "ps_availqty > (SELECT COUNT(*) FROM lineitem WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey AND "
         "l_shipdate > DATE '2030-01-01')",                                                                         # COUNT: 0
]
JOIN = ("SELECT s_name, ps_partkey FROM partsupp, supplier WHERE ps_suppkey = s_suppkey AND s_nationkey = 3 AND "
        "ps_availqty < (SELECT 60 * SUM(l_quantity) FROM lineitem WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey) "
        "ORDER BY s_name, ps_partkey")


def _same(db, con, q, ordered=False):
    import controller, test_join_fast as J
    controller.plan_epoch_bump()
    exp = J._norm([tuple(r) for r in con.execute(q).fetchall()])
    r = db.run(q)
    got = J._norm(r[0] if isinstance(r, tuple) else r)
    if not ordered: got, exp = sorted(got), sorted(exp)
    assert len(got) == len(exp), (q, len(got), len(exp))
    for g, e in zip(got, exp):
        assert J._eq(g, e), (q, g, e)


def test_q20_shape():
    import wdb_corrrow as CR, wdb_insub as IS, test_pair_road as PR
    db, con = PR._fixture()
    c0, i0 = CR._HITS[0], IS._HITS[0]
    _same(db, con, Q20, ordered=True)
    assert CR._HITS[0] >= c0 + 1 and IS._HITS[0] >= i0 + 2      # the value per row; both IN levels as code sets


def test_middle_query():
    import wdb_corrrow as CR, test_pair_road as PR
    db, con = PR._fixture()
    c0 = CR._HITS[0]
    _same(db, con, MIDDLE + " ORDER BY ps_suppkey", ordered=True)
    assert CR._HITS[0] == c0 + 1


def test_each_shape_one_table():
    import wdb_corrrow as CR, test_pair_road as PR
    db, con = PR._fixture()
    for q in ONE_TABLE:
        c0 = CR._HITS[0]
        _same(db, con, q)
        assert CR._HITS[0] == c0 + 1, q


def test_join_outer():
    import wdb_corrrow as CR, test_pair_road as PR
    db, con = PR._fixture()
    c0 = CR._HITS[0]
    _same(db, con, JOIN, ordered=True)
    assert CR._HITS[0] == c0 + 1


def test_switched_off():
    import wdb_corrrow as CR, test_pair_road as PR
    db, con = PR._fixture()
    CR._CORR_ROW[0] = False
    try:
        c0 = CR._HITS[0]
        try:
            _same(db, con, ONE_TABLE[0])
        except AssertionError:
            raise
        except Exception:
            pass                                 # the old roads decline it: that is the switch's whole effect
        assert CR._HITS[0] == c0
    finally:
        CR._CORR_ROW[0] = True


if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_'):
            f(); print('PASS', n)
