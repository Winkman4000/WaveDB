"""A CONSTANT IN A ROW PROJECTION (2026-11-02): row selects with a literal, a literal expression, a string literal and a
column expression beside columns answer like DuckDB (the literal ones raised IndexError before c114)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))


def test_constant_projection():
    import test_join_fast as J
    db, con = J._fixture()
    for q in ["SELECT c_custkey, 1.5 AS t FROM customer WHERE c_custkey IN (1, 2, 7) ORDER BY c_custkey",
              "SELECT c_custkey, 1.5 * 2 AS t FROM customer WHERE c_custkey IN (1, 2, 7) ORDER BY c_custkey",
              "SELECT c_custkey, 7 AS k, c_name FROM customer WHERE c_custkey < 5 ORDER BY c_custkey",
              "SELECT c_custkey, c_acctbal * 2 AS t FROM customer WHERE c_custkey IN (1, 2, 7) ORDER BY c_custkey"]:
        r = db.run(q)
        got = J._norm(r[0] if isinstance(r, tuple) else r)
        exp = J._norm([tuple(x) for x in con.execute(q).fetchall()])
        assert len(got) == len(exp) and all(J._eq(g, e) for g, e in zip(got, exp)), (q, got, exp)
