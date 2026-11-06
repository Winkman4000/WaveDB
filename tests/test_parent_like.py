"""THE PARENT'S LIKE PER ROW (c124, 2026-11-04, TPC-H Q9 / Q20): a LIKE on a parent's big inline string column, in a
join, is judged once per parent row on the stored bytes and carried to the fact rows by the parent's row-number key
column and the road -- in the cascade, and in the fused predicate when the cascade's gate turns the query away
(NOT LIKE, a LIKE beside other parent filters). Answers like DuckDB, grouped and scalar, with '%' and '_' patterns;
switched off, the old road answers the same."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

JOIN = "FROM lineitem, orders WHERE l_orderkey = o_orderkey AND "
Q = [
    "SELECT COUNT(*), SUM(l_quantity) " + JOIN + "o_comment LIKE '%special%'",
    "SELECT l_returnflag, COUNT(*), SUM(l_extendedprice) " + JOIN + "o_comment LIKE '%furiously%' GROUP BY l_returnflag",
    "SELECT COUNT(*) " + JOIN + "o_comment LIKE 'sl_ly%'",
    "SELECT l_linestatus, COUNT(*) " + JOIN + "o_comment LIKE '%deposits%' AND o_orderpriority = '1-URGENT' GROUP BY l_linestatus",
    "SELECT COUNT(*), SUM(l_quantity) " + JOIN + "o_comment NOT LIKE '%the%'",
    "SELECT l_shipmode, COUNT(*) " + JOIN + "o_comment LIKE '%ironic%' AND l_quantity > 25 GROUP BY l_shipmode",
]


def _same(db, con, q):
    import controller, test_join_fast as J
    exp = sorted(J._norm([tuple(r) for r in con.execute(q).fetchall()]))
    r = db.run(q)
    got = sorted(J._norm(r[0] if isinstance(r, tuple) else r))
    assert len(got) == len(exp), (q, len(got), len(exp))
    for g, e in zip(got, exp):
        assert J._eq(g, e), (q, g, e)


class _SmallLut:
    """The fixture's o_comment dictionary (15k) fits the code LUT, so the old road would take it; at SF10 the
    names (2M) do not. Lowering the LUT's ceiling for the test makes the fixture the big-dictionary case."""
    def __enter__(self):
        import wdb_measure_runtime as RT
        self.o = RT.LUT_MAX_CARD; RT.LUT_MAX_CARD = 1000
    def __exit__(self, *a):
        import wdb_measure_runtime as RT
        RT.LUT_MAX_CARD = self.o


def test_the_column_is_the_shape():
    """the fixture's o_comment is a big inline (mode-5) column of a parent with more than 4096 rows"""
    import wdb_join as WJ, test_join_fast as J
    db, con = J._fixture()
    seg, _ = WJ._solo_segment(db, 'orders')
    pm = db.cat.phys_map('orders')
    assert seg.cols[pm.get('o_comment', 'o_comment')].get('mode') == 5 and int(seg.N) > 4096


def test_parent_like_answers():
    import wdb_join as WJ, test_join_fast as J
    db, con = J._fixture()
    with _SmallLut():
        for q in Q:
            h0 = WJ._PARENT_LIKE_HITS[0]
            _same(db, con, q)
            assert WJ._PARENT_LIKE_HITS[0] >= h0 + 1, q


def test_switched_off():
    import wdb_join as WJ, test_join_fast as J
    db, con = J._fixture()
    o0 = WJ._PARENT_LIKE[0]
    try:
        WJ._PARENT_LIKE[0] = False
        h0 = WJ._PARENT_LIKE_HITS[0]
        for q in Q[:3]:
            _same(db, con, q)                       # the old LUT road (the fixture's dictionary fits it)
        assert WJ._PARENT_LIKE_HITS[0] == h0
    finally:
        WJ._PARENT_LIKE[0] = o0
