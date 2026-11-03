"""THE ROWS ASKED FOR (2026-11-02): row projections over few rows -- text, numeric, date and key columns, a constant,
ORDER BY, LIMIT, DISTINCT, and a wide filter (whole columns) -- answer like DuckDB with the read at the rows on and
off, and the read at the rows is taken."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))


def test_row_projection_both_ways():
    import wdb_sql as S, controller, test_join_fast as J
    db, con = J._fixture()
    qs = ["SELECT c_custkey, c_name, c_address, c_phone, c_acctbal FROM customer WHERE c_custkey IN (3, 77, 1201) ORDER BY c_custkey",
          "SELECT c_name, c_mktsegment, 2.5 AS k FROM customer WHERE c_custkey < 9 ORDER BY c_name DESC",
          "SELECT o_orderkey, o_orderdate, o_comment FROM orders WHERE o_orderkey IN (1, 7, 32, 33) ORDER BY o_orderkey",
          "SELECT DISTINCT c_mktsegment FROM customer WHERE c_custkey < 40 ORDER BY c_mktsegment",
          "SELECT c_custkey, c_name FROM customer WHERE c_acctbal > 9000 ORDER BY c_custkey LIMIT 5",
          "SELECT c_custkey, c_phone FROM customer WHERE c_acctbal > -2000 ORDER BY c_custkey LIMIT 3"]
    old = S._ROWS_AT[0]
    try:
        h0 = S._ROWS_AT_HITS[0]
        for q in qs:
            exp = J._norm([tuple(r) for r in con.execute(q).fetchall()])
            for on in (True, False):
                S._ROWS_AT[0] = on
                controller.plan_epoch_bump()
                r = db.run(q)
                got = J._norm(r[0] if isinstance(r, tuple) else r)
                assert len(got) == len(exp) and all(J._eq(g, e) for g, e in zip(got, exp)), (q, on, got[:3], exp[:3])
        assert S._ROWS_AT_HITS[0] > h0, 'the read at the rows was never taken'
    finally:
        S._ROWS_AT[0] = old
        controller.plan_epoch_bump()
