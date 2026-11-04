"""AGGREGATE ARITHMETIC WITH ORDER BY AND A COMPUTED KEY (c122, 2026-11-04, TPC-H Q8): a ratio of two aggregates
grouped by a computed key, over a derived table and flat, sorted (and cut) after the ratio is evaluated, answers
like DuckDB in DuckDB's order; without ORDER BY the same rows (as a set); switched off, the rewrite stands down."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

JOIN = ("FROM lineitem, orders, customer, nation WHERE l_orderkey = o_orderkey AND o_custkey = c_custkey "
        "AND c_nationkey = n_nationkey AND o_orderstatus = 'F'")
DERIVED = ("SELECT k, SUM(CASE WHEN nation = 'GERMANY' THEN volume ELSE 0 END) / SUM(volume) AS share FROM "
           "(SELECT l_linenumber * 2 AS k, l_extendedprice * (1 - l_discount) AS volume, n_name AS nation " + JOIN + ") AS t "
           "GROUP BY k ORDER BY k")
FLAT_ORDER = ("SELECT l_linenumber * 2 AS k, SUM(CASE WHEN n_name = 'GERMANY' THEN l_extendedprice ELSE 0 END) / SUM(l_extendedprice) "
              "AS share " + JOIN + " GROUP BY k ORDER BY share DESC, k LIMIT 3")
FLAT = ("SELECT l_linenumber * 2 AS k, SUM(CASE WHEN n_name = 'FRANCE' THEN l_quantity ELSE 0 END) / SUM(l_quantity) "
        + JOIN + " GROUP BY k")


def _rows(db, q):
    import controller, test_join_fast as J
    controller.plan_epoch_bump()
    r = db.run(q)
    return J._norm(r[0] if isinstance(r, tuple) else r)


def _check(got, exp, q, ordered=True):
    import test_join_fast as J
    if not ordered:
        got, exp = sorted(got), sorted(exp)
    assert len(got) == len(exp), (q, len(got), len(exp))
    for g, e in zip(got, exp):
        assert J._eq(g, e), (q, g, e)


def test_ratio_over_derived_table_with_order():
    import wdb_join as WJ, test_join_fast as J
    db, con = J._fixture()
    h0 = WJ._AGG_ORDER_HITS[0]
    _check(_rows(db, DERIVED), J._norm(con.execute(DERIVED).fetchall()), DERIVED)
    assert WJ._AGG_ORDER_HITS[0] == h0 + 1


def test_computed_key_with_order_and_limit():
    import wdb_join as WJ, test_join_fast as J
    db, con = J._fixture()
    h0, k0 = WJ._AGG_ORDER_HITS[0], WJ._AGG_KEYCOL_HITS[0]
    _check(_rows(db, FLAT_ORDER), J._norm(con.execute(FLAT_ORDER).fetchall()), FLAT_ORDER)
    assert WJ._AGG_ORDER_HITS[0] == h0 + 1 and WJ._AGG_KEYCOL_HITS[0] == k0 + 1


def test_computed_key_without_order():
    import wdb_join as WJ, test_join_fast as J
    db, con = J._fixture()
    k0 = WJ._AGG_KEYCOL_HITS[0]
    _check(_rows(db, FLAT), J._norm(con.execute(FLAT).fetchall()), FLAT, ordered=False)
    assert WJ._AGG_KEYCOL_HITS[0] == k0 + 1


SINGLE = ("SELECT l_linenumber * 2 AS k, SUM(l_extendedprice * l_discount) / SUM(l_extendedprice) AS d FROM lineitem "
          "GROUP BY k ORDER BY k")


def test_computed_key_on_the_pandas_tail():
    """the pandas tail's grouping itself: a computed key by alias and by expression, a ratio over it, beside DuckDB"""
    import duckdb, numpy as np, pandas as pd, sqlglot, wdb_join as WJ, test_join_fast as J
    db, con = J._fixture()
    _check(_rows(db, SINGLE), J._norm(con.execute(SINGLE).fetchall()), SINGLE)
    rng = np.random.default_rng(7)
    t = pd.DataFrame({'a': rng.integers(0, 40, 5000), 'b': rng.random(5000), 'c': rng.integers(0, 3, 5000)})
    def R(node):                                  # like the tail's own: a name no table owns is not a column
        if node.name not in t.columns: raise WJ._FastUnsupported
        return 't.' + node.name
    for q in ("SELECT a / 7 AS y, SUM(b) / SUM(c) AS r, COUNT(*) FROM t GROUP BY y",
              "SELECT a * 2 + c, SUM(b) FROM t GROUP BY a * 2 + c"):
        tree = sqlglot.parse_one(q, read='duckdb')
        h0 = WJ._PANDAS_KEYEXPR_HITS[0]
        merged = pd.DataFrame({'t.a': t['a'], 't.b': t['b'], 't.c': t['c']})
        got = J._norm(WJ._aggregate(merged, tree.expressions, tree.args.get('group'), R))
        exp = J._norm(duckdb.query_df(t, 't', q).fetchall()) if hasattr(duckdb, 'query_df') else None
        if exp is None:
            c9 = duckdb.connect(); c9.register('t', t); exp = J._norm(c9.execute(q).fetchall())
        _check(got, exp, q, ordered=False)
        assert WJ._PANDAS_KEYEXPR_HITS[0] == h0 + 1


def test_switched_off():
    import sqlglot, wdb_join as WJ, test_join_fast as J
    db, con = J._fixture()
    o0, k0 = WJ._AGG_ORDER[0], WJ._AGG_KEYCOL[0]
    try:
        WJ._AGG_ORDER[0] = False
        assert WJ._agg_expr_rewrite(db, sqlglot.parse_one(FLAT_ORDER, read='duckdb')) is None
        WJ._AGG_ORDER[0] = o0
        WJ._AGG_KEYCOL[0] = False
        h0 = WJ._AGG_KEYCOL_HITS[0]
        try:
            WJ._agg_expr_rewrite(db, sqlglot.parse_one(FLAT, read='duckdb'))
        except Exception:
            pass                                  # the old handling of a computed key is not this test's to judge
        assert WJ._AGG_KEYCOL_HITS[0] == h0
    finally:
        WJ._AGG_ORDER[0], WJ._AGG_KEYCOL[0] = o0, k0
