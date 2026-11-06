"""THE CENSUS BY CODE (2026-11-01): Q22's shape (a phone-prefix IN, a balance above a scalar AVG of the same shape,
NOT EXISTS orders) answers like DuckDB with the census counted by code and the old way, and the code census is
taken; EXISTS (the positive census) too."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))


def test_answers_both_ways():
    import wdb_join as WJ, controller, test_join_fast as J
    db, con = J._fixture()
    cs = "'13', '31', '23', '29', '30', '18', '17'"
    q22 = ("SELECT cntrycode, COUNT(*) AS numcust, SUM(c_acctbal) AS totacctbal FROM (SELECT SUBSTRING(c_phone, 1, 2) AS cntrycode, "
           "c_acctbal FROM customer WHERE SUBSTRING(c_phone, 1, 2) IN (%s) AND c_acctbal > (SELECT AVG(c_acctbal) FROM customer "
           "WHERE c_acctbal > 0.00 AND SUBSTRING(c_phone, 1, 2) IN (%s)) AND %s (SELECT * FROM orders WHERE o_custkey = c_custkey)) "
           "AS custsale GROUP BY cntrycode ORDER BY cntrycode")
    qs = [q22 % (cs, cs, 'NOT EXISTS'), q22 % (cs, cs, 'EXISTS')]
    old = WJ._CENSUS_CODE[0]
    try:
        h0 = WJ._CENSUS_CODE_HITS[0]
        for q in qs:
            exp = J._norm([tuple(r) for r in con.execute(q).fetchall()])
            assert exp, q
            for on in (True, False):
                WJ._CENSUS_CODE[0] = on
                r = db.run(q)
                got = J._norm(r[0] if isinstance(r, tuple) else r)    # this door returns the rows themselves
                assert len(got) == len(exp), (q, on, got, exp)
                for g, e in zip(got, exp):
                    assert J._eq(g, e), (q, on, g, e)
        assert WJ._CENSUS_CODE_HITS[0] > h0, 'the census by code was never taken'
    finally:
        WJ._CENSUS_CODE[0] = old
