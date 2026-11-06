"""[NOT] IN (uncorrelated subquery) as a filter in a join, and the distinct count in slices (c127, 2026-11-05, TPC-H
Q16): the inner query runs first, its values become the outer column's codes (NOT IN: the complement); grouped
COUNT(DISTINCT) with group keys from a parent table is counted per group in slices. Answers like DuckDB (Q16's shape
in order, IN, the distinct count alone); a NOT IN whose list holds a NULL stands down; switched off, neither fires."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

W = ("FROM partsupp, part WHERE p_partkey = ps_partkey AND p_brand <> 'Brand#45' AND p_type NOT LIKE 'MEDIUM POLISHED%' "
     "AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)")
NOT_IN = " AND ps_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%')"
Q16 = ("SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey) AS supplier_cnt " + W + NOT_IN +
       " GROUP BY p_brand, p_type, p_size ORDER BY supplier_cnt DESC, p_brand, p_type, p_size")
CD = "SELECT p_brand, p_size, COUNT(DISTINCT ps_suppkey) " + W + " GROUP BY p_brand, p_size"
IN_POS = ("SELECT COUNT(*), SUM(ps_availqty) FROM partsupp, part WHERE p_partkey = ps_partkey AND p_size < 10 "
          "AND ps_suppkey IN (SELECT s_suppkey FROM supplier WHERE s_acctbal > 5000)")
EMPTY_NOT_IN = ("SELECT COUNT(*) " + W + " AND ps_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_acctbal > 1e9)")
EMPTY_IN = ("SELECT COUNT(*) " + W + " AND ps_suppkey IN (SELECT s_suppkey FROM supplier WHERE s_acctbal > 1e9)")
NULL_IN = ("SELECT COUNT(*) FROM partsupp, part WHERE p_partkey = ps_partkey AND p_size < 10 AND ps_suppkey NOT IN "
           "(SELECT CASE WHEN s_acctbal > 5000 THEN s_suppkey ELSE NULL END FROM supplier)")


def _same(db, con, q, ordered=False):
    import controller, test_join_fast as J
    exp = J._norm([tuple(r) for r in con.execute(q).fetchall()])
    r = db.run(q)
    got = J._norm(r[0] if isinstance(r, tuple) else r)
    if not ordered: got, exp = sorted(got), sorted(exp)
    assert len(got) == len(exp), (q, len(got), len(exp))
    for g, e in zip(got, exp):
        assert J._eq(g, e), (q, g, e)


def test_q16_shape():
    import wdb_insub as IS, wdb_join as WJ, test_pair_road as PR
    db, con = PR._fixture()
    i0, s0 = IS._HITS[0], WJ._GCD_SLICES_HITS[0]
    _same(db, con, Q16, ordered=True)
    assert IS._HITS[0] == i0 + 1 and WJ._GCD_SLICES_HITS[0] == s0 + 1


def test_distinct_count_in_slices():
    import wdb_join as WJ, test_pair_road as PR
    db, con = PR._fixture()
    s0 = WJ._GCD_SLICES_HITS[0]
    _same(db, con, CD)
    assert WJ._GCD_SLICES_HITS[0] == s0 + 1


def test_in_positive():
    import wdb_insub as IS, test_pair_road as PR
    db, con = PR._fixture()
    i0 = IS._HITS[0]
    _same(db, con, IN_POS)
    assert IS._HITS[0] == i0 + 1


def test_empty_lists():
    """NOT IN an empty list keeps every row; IN an empty list stands down (not served yet -- never a wrong answer)"""
    import wdb_insub as IS, wdb_join as WJ, test_pair_road as PR
    db, con = PR._fixture()
    i0 = IS._HITS[0]
    _same(db, con, EMPTY_NOT_IN)
    assert IS._HITS[0] == i0 + 1
    i0 = IS._HITS[0]
    try:
        _same(db, con, EMPTY_IN)
    except (NotImplementedError, WJ._FastUnsupported):
        pass
    assert IS._HITS[0] == i0


def test_not_in_with_a_null_stands_down():
    import wdb_insub as IS, test_pair_road as PR
    db, con = PR._fixture()
    import wdb_join as WJ
    i0 = IS._HITS[0]
    try:
        _same(db, con, NULL_IN)                  # SQL: NOT IN against a list with a NULL keeps no row
    except (NotImplementedError, WJ._FastUnsupported):
        pass                                     # the old road's own limits are not this test's
    assert IS._HITS[0] == i0


def test_switched_off():
    import wdb_insub as IS, wdb_join as WJ, test_pair_road as PR
    db, con = PR._fixture()
    o1, o2 = IS._IN_SUB[0], WJ._GCD_SLICES[0]
    try:
        IS._IN_SUB[0] = False; WJ._GCD_SLICES[0] = False
        i0, s0 = IS._HITS[0], WJ._GCD_SLICES_HITS[0]
        for q in (IN_POS, CD):
            try:
                _same(db, con, q)
            except (NotImplementedError, WJ._FastUnsupported):
                pass                             # without them the old road may not answer: not this test's
        assert IS._HITS[0] == i0 and WJ._GCD_SLICES_HITS[0] == s0
    finally:
        IS._IN_SUB[0], WJ._GCD_SLICES[0] = o1, o2
