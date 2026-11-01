"""THE KEEP BY CODE (2026-10-31): wdb_kernels.keep_rows_by_code equals numpy's rows[keep[codes]] -- every code
width, signed and unsigned, negative codes (read from the table's end, as numpy does), empty / full / sparse keep
tables, sizes on both sides of the parallel line; a code outside the table raises as numpy does; switched off it is
numpy. Join answers equal DuckDB's with it on and off."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np


def test_equals_numpy():
    import wdb_kernels as WK
    rng = np.random.default_rng(1031)
    checked = 0
    for dt in (np.uint8, np.uint16, np.uint32, np.int32, np.int64):
        for n in (10, 70_000, 400_003):
            V = 200
            rows = np.sort(rng.choice(5 * n + 10, n, replace=False)).astype(np.int64)
            codes = rng.integers(0, V + 1, n).astype(dt)
            if np.dtype(dt).kind == 'i':
                codes[::7] = -1                               # the null sentinel read from the end
            for keep in (rng.random(V + 1) < 0.3, np.zeros(V + 1, bool), np.ones(V + 1, bool), rng.random(V + 1) < 0.01):
                got = WK.keep_rows_by_code(rows, codes, keep)
                assert np.array_equal(got, rows[keep[codes]]), (dt, n)
                checked += 1
    assert checked == 60
    rows = np.arange(100_000, dtype=np.int64); codes = np.full(100_000, 5, np.uint16); codes[77_777] = 999
    try:
        WK.keep_rows_by_code(rows, codes, np.ones(10, bool))
        assert False, 'a code outside the table must raise'
    except IndexError:
        pass
    old = WK._KEEP_ROWS[0]
    try:
        WK._KEEP_ROWS[0] = False
        c2 = np.full(100_000, 3, np.uint16)
        assert np.array_equal(WK.keep_rows_by_code(rows, c2, np.ones(10, bool)), rows)
    finally:
        WK._KEEP_ROWS[0] = old


def test_join_answers_both_ways():
    import wdb_kernels as WK, controller, test_join_fast as J
    db, con = J._fixture()
    qs = ["SELECT l_returnflag, COUNT(*), SUM(l_extendedprice) FROM lineitem, orders WHERE l_orderkey = o_orderkey "
          "AND l_shipmode = 'AIR' AND l_quantity > 5 AND o_orderstatus = 'F' GROUP BY l_returnflag ORDER BY l_returnflag",
          "SELECT SUM(l_extendedprice * l_discount) FROM lineitem WHERE l_discount >= 0.05 AND l_discount <= 0.07 "
          "AND l_quantity < 24 AND l_shipmode IN ('MAIL', 'SHIP')"]
    old = (WK._KEEP_ROWS[0], WK._KEEP_ROWS_MIN)
    try:
        WK._KEEP_ROWS_MIN = 1                                 # the fixture is small: take the parallel path
        for q in qs:
            exp = [tuple(r) for r in con.execute(q).fetchall()]
            for on in (True, False):
                WK._KEEP_ROWS[0] = on
                controller.plan_epoch_bump()
                got = [tuple(r) for r in db.run(q)[0]]
                assert len(got) == len(exp), (q, on)
                for a, b in zip(got, exp):
                    for x, y in zip(a, b):
                        num = not isinstance(y, (str, bytes)) and y is not None
                        assert (abs(float(x) - float(y)) <= 1e-6 * max(1.0, abs(float(y)))) if num else x == y, (q, on, a, b)
    finally:
        WK._KEEP_ROWS[0], WK._KEEP_ROWS_MIN = old
