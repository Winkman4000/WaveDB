"""FK-pointer gather fast path: same join queries as the hash path, but routed through the
pre-resolved-pointer + code-bincount kernel. Verified against DuckDB, and asserted to actually take
the fast path (not silently fall back)."""
import sys, os, tempfile, uuid, math, datetime, re
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, wdb_join
from wdb_db import Database

_DB = None; _CON = None
_WT = {'BIGINT':'int','INTEGER':'int','VARCHAR':'string','DATE':'datetime'}
def _wt(t): return 'float' if t.startswith('DECIMAL') else _WT[t]

def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect(); _CON.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=0.01)")
    d = os.path.join(tempfile.gettempdir(), f'joinfast_{uuid.uuid4().hex[:8]}')
    _DB = Database.create(d)
    order = {'region':'r_regionkey','nation':'n_nationkey','customer':'c_custkey','orders':'o_orderkey','lineitem':'l_orderkey'}
    for tbl in ('region','nation','customer','orders','lineitem'):
        desc = _CON.execute(f"DESCRIBE {tbl}").fetchall()
        sel = ", ".join((f"CAST({c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else c[0]) for c in desc)
        pq = os.path.join(d, f'{tbl}.parquet')
        _CON.execute(f"COPY (SELECT {sel} FROM {tbl} ORDER BY {order[tbl]}) TO '{pq}' (FORMAT parquet)")
        _DB.cat.add_table(tbl, [[c[0], _wt(c[1])] for c in desc])
        seg = f'{tbl}_0.wdb'; wdb_encode.encode(pq, os.path.join(d, seg)); _DB.cat.add_segment(tbl, seg)
    _DB.create_fk_pointer('orders', 'o_custkey', 'customer', 'c_custkey')
    _DB.create_fk_pointer('nation', 'n_regionkey', 'region', 'r_regionkey')
    _DB.create_fk_pointer('customer', 'c_nationkey', 'nation', 'n_nationkey')
    _DB.create_fk_pointer('lineitem', 'l_orderkey', 'orders', 'o_orderkey')
    return _DB, _CON

_TS = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(\.\d+)?$')
def _cell(c):
    if isinstance(c, Decimal): c = float(c)
    if isinstance(c, datetime.datetime): c = c.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(c, datetime.date): return c.strftime('%Y-%m-%d')
    if isinstance(c, str):
        m = _TS.match(c)
        if m: c = m.group(1)
        if len(c) == 19 and c.endswith(' 00:00:00'): c = c[:10]
    return c
def _norm(rows): return [tuple(_cell(c) for c in r) for r in rows]
def _eq(g, e):
    if len(g) != len(e): return False
    for a, b in zip(g, e):
        if isinstance(a, float) or isinstance(b, float):
            if a is None or b is None: return a is b
            if not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6): return False
        elif a != b: return False
    return True

def _match(q, ordered=False, expect_fast=True):
    db, con = _fixture()
    before = wdb_join._FAST_HITS
    g = _norm(db.run(q)[0]); e = _norm([tuple(r) for r in con.execute(q).fetchall()])
    if expect_fast:
        assert wdb_join._FAST_HITS == before + 1, f"fast path NOT taken: {q}"
    if not ordered:
        k = lambda t: tuple((x is None, str(x)) for x in t); g = sorted(g, key=k); e = sorted(e, key=k)
    assert len(g) == len(e), f"{q}\n nG={len(g)} nE={len(e)}"
    for gr, er in zip(g, e):
        assert _eq(gr, er), f"{q}\n got={gr}\n exp={er}"

def test_fast_group_parent_key():
    _match("SELECT c.c_mktsegment, COUNT(*), SUM(o.o_totalprice) FROM orders o "
           "JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment")

def test_fast_group_child_key():
    _match("SELECT o.o_orderpriority, COUNT(*), AVG(o.o_totalprice) FROM orders o "
           "JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY o.o_orderpriority")

def test_fast_all_aggs():
    _match("SELECT c.c_mktsegment, MIN(o.o_totalprice), MAX(o.o_totalprice), AVG(o.o_totalprice), COUNT(o.o_orderkey) "
           "FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment")

def test_fast_minmax_date():
    _match("SELECT c.c_mktsegment, MIN(o.o_orderdate), MAX(o.o_orderdate) FROM orders o "
           "JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment")

def test_fast_minmax_string():
    _match("SELECT c.c_mktsegment, MIN(o.o_orderpriority), MAX(o.o_orderpriority) FROM orders o "
           "JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment")

def test_fast_where_and():
    _match("SELECT c.c_mktsegment, COUNT(*) FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey "
           "WHERE o.o_totalprice > 100000 AND c.c_mktsegment <> 'BUILDING' GROUP BY c.c_mktsegment")

def test_fast_where_between_in():
    _match("SELECT c.c_mktsegment, SUM(o.o_totalprice) FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey "
           "WHERE o.o_totalprice BETWEEN 50000 AND 250000 AND o.o_orderpriority IN ('1-URGENT','2-HIGH') "
           "GROUP BY c.c_mktsegment")

def test_fast_whole_table_aggregate():
    _match("SELECT COUNT(*), SUM(o.o_totalprice) FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey "
           "WHERE c.c_mktsegment = 'AUTOMOBILE'")

def test_fast_reversed_on():
    _match("SELECT c.c_mktsegment, COUNT(*) FROM orders o JOIN customer c ON c.c_custkey=o.o_custkey "
           "GROUP BY c.c_mktsegment")

def test_fast_nation_region():
    _match("SELECT r.r_name, COUNT(*) FROM nation n JOIN region r ON n.n_regionkey=r.r_regionkey GROUP BY r.r_name")

def test_fast_customer_nation():
    _match("SELECT n.n_name, COUNT(*), AVG(c.c_acctbal) FROM customer c JOIN nation n "
           "ON c.c_nationkey=n.n_nationkey GROUP BY n.n_name")

def test_fast_order_limit():
    _match("SELECT c.c_mktsegment, SUM(o.o_totalprice) AS rev FROM orders o JOIN customer c "
           "ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment ORDER BY rev DESC LIMIT 3", ordered=True)

def test_threaded_path_matches_duckdb():
    # force the threaded fused kernel on the small fixture (threshold normally 2M rows)
    import wdb_agg
    saved = wdb_agg.PARALLEL_THRESHOLD; wdb_agg.PARALLEL_THRESHOLD = 0
    try:
        _match("SELECT c.c_mktsegment, COUNT(*), SUM(o.o_totalprice), AVG(o.o_totalprice), COUNT(o.o_orderkey) "
               "FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment")
        _match("SELECT c.c_mktsegment, SUM(o.o_totalprice) FROM orders o JOIN customer c "
               "ON o.o_custkey=c.c_custkey WHERE o.o_totalprice > 100000 GROUP BY c.c_mktsegment")
    finally:
        wdb_agg.PARALLEL_THRESHOLD = saved

def test_plain_projection_falls_back():
    # no aggregate -> not eligible -> must fall back to the hash path, still correct
    _match("SELECT o.o_orderkey, c.c_name FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey "
           "WHERE o.o_totalprice > 400000 ORDER BY o.o_orderkey LIMIT 5", ordered=True, expect_fast=False)


# ── multi-table FK-pointer chains: lineitem -> orders -> customer -> nation -> region ──
_J3 = ("FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey "
       "JOIN customer c ON o.o_custkey=c.c_custkey JOIN nation n ON c.c_nationkey=n.n_nationkey ")
_J5 = _J3 + "JOIN region r ON n.n_regionkey=r.r_regionkey "

def test_chain_3table_count_by_nation():
    _match("SELECT n.n_name, COUNT(*) " + _J3 + "GROUP BY n.n_name")
def test_chain_3table_sum_by_nation():
    _match("SELECT n.n_name, SUM(l.l_quantity) " + _J3 + "GROUP BY n.n_name")
def test_chain_3table_minmax_by_nation():
    _match("SELECT n.n_name, MIN(l.l_quantity), MAX(l.l_quantity) " + _J3 + "GROUP BY n.n_name")
def test_chain_5table_count_by_region():
    _match("SELECT r.r_name, COUNT(*) " + _J5 + "GROUP BY r.r_name")
def test_chain_5table_avg_by_region():
    _match("SELECT r.r_name, AVG(l.l_extendedprice) " + _J5 + "GROUP BY r.r_name")
def test_chain_5table_where_intermediate():
    _match("SELECT r.r_name, COUNT(*) " + _J5 + "WHERE o.o_orderpriority='1-URGENT' GROUP BY r.r_name")
def test_chain_whole_no_group():
    _match("SELECT COUNT(*), SUM(l.l_quantity) " + _J5)


# ── arithmetic in aggregate expressions (TPC-H Q5/Q1 revenue/charge) over the chain ──
def test_chain_revenue_by_region():
    _match("SELECT r.r_name, SUM(l.l_extendedprice*(1-l.l_discount)) " + _J5 + "GROUP BY r.r_name")
def test_chain_charge_by_nation():
    _match("SELECT n.n_name, SUM(l.l_extendedprice*(1-l.l_discount)*(1+l.l_tax)) " + _J3 + "GROUP BY n.n_name")
def test_chain_avg_expr_by_nation():
    _match("SELECT n.n_name, AVG(l.l_extendedprice*(1-l.l_discount)) " + _J3 + "GROUP BY n.n_name")
def test_chain_cross_table_arith():
    _match("SELECT c.c_mktsegment, SUM(l.l_quantity*o.o_totalprice) FROM lineitem l "
           "JOIN orders o ON l.l_orderkey=o.o_orderkey JOIN customer c ON o.o_custkey=c.c_custkey "
           "GROUP BY c.c_mktsegment")
def test_chain_revenue_where():
    _match("SELECT r.r_name, SUM(l.l_extendedprice*(1-l.l_discount)) " + _J5 +
           "WHERE o.o_orderpriority='1-URGENT' GROUP BY r.r_name")
def test_chain_minmax_expr():
    _match("SELECT n.n_name, MIN(l.l_extendedprice*(1-l.l_discount)), "
           "MAX(l.l_extendedprice*(1-l.l_discount)) " + _J3 + "GROUP BY n.n_name")


# ── multi-column GROUP BY over the chain (mixed-radix composite group codes) ──
def test_chain_group_two_fact_keys():
    _match("SELECT l.l_returnflag, l.l_linestatus, COUNT(*), SUM(l.l_extendedprice) " + _J3 +
           "GROUP BY l.l_returnflag, l.l_linestatus")
def test_chain_group_fact_parent_keys():
    _match("SELECT l.l_returnflag, o.o_orderpriority, SUM(l.l_extendedprice) " + _J3 +
           "GROUP BY l.l_returnflag, o.o_orderpriority")
def test_chain_group_three_keys():
    _match("SELECT c.c_mktsegment, o.o_orderpriority, l.l_returnflag, SUM(l.l_extendedprice) " + _J3 +
           "GROUP BY c.c_mktsegment, o.o_orderpriority, l.l_returnflag")
def test_chain_group_multi_proj_swapped():
    _match("SELECT o.o_orderpriority, l.l_returnflag, COUNT(*) " + _J3 +
           "GROUP BY l.l_returnflag, o.o_orderpriority")
def test_chain_group_multi_where():
    _match("SELECT l.l_returnflag, l.l_linestatus, SUM(l.l_extendedprice) " + _J3 +
           "WHERE l.l_quantity > 25 GROUP BY l.l_returnflag, l.l_linestatus")
def test_chain_group_multi_minmax():
    _match("SELECT c.c_mktsegment, l.l_returnflag, MIN(l.l_extendedprice), MAX(l.l_extendedprice), "
           "AVG(l.l_extendedprice) " + _J3 + "GROUP BY c.c_mktsegment, l.l_returnflag")
def test_chain_group_multi_order_limit():
    _match("SELECT l.l_returnflag, l.l_linestatus, SUM(l.l_extendedprice) AS rev " + _J3 +
           "GROUP BY l.l_returnflag, l.l_linestatus ORDER BY rev DESC LIMIT 3", ordered=True)


# ── codegen-fused arithmetic aggregates (wdb_exprjit): never materialise the expression array ──
def test_chain_arith_revenue():
    _match("SELECT n.n_name, SUM(l.l_extendedprice*(1-l.l_discount)) AS rev " + _J3 + "GROUP BY n.n_name")
def test_chain_arith_q1_form():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice*(1-l.l_discount)*(1+l.l_tax)) AS rev " + _J3 +
           "GROUP BY l.l_returnflag")
def test_chain_arith_multikey():
    _match("SELECT c.c_mktsegment, l.l_returnflag, SUM(l.l_extendedprice*(1-l.l_discount)) AS rev " + _J3 +
           "GROUP BY c.c_mktsegment, l.l_returnflag")
def test_chain_arith_where():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice*(1-l.l_discount)) AS rev " + _J3 +
           "WHERE l.l_quantity > 25 GROUP BY l.l_returnflag")
def test_chain_arith_avg():
    _match("SELECT l.l_returnflag, AVG(l.l_extendedprice*(1-l.l_discount)) AS a " + _J3 +
           "GROUP BY l.l_returnflag")
def test_chain_arith_minmax():
    _match("SELECT l.l_returnflag, MIN(l.l_extendedprice*(1-l.l_discount)), "
           "MAX(l.l_extendedprice*(1-l.l_discount)) " + _J3 + "GROUP BY l.l_returnflag")
def test_chain_arith_mixed_plain():
    _match("SELECT l.l_returnflag, COUNT(*), SUM(l.l_extendedprice), "
           "SUM(l.l_extendedprice*(1-l.l_discount)) AS rev " + _J3 + "GROUP BY l.l_returnflag")
def test_chain_arith_parent_group():
    _match("SELECT o.o_orderpriority, SUM(l.l_extendedprice*(1-l.l_discount)) AS rev " + _J3 +
           "GROUP BY o.o_orderpriority")
def test_chain_arith_order_limit():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice*(1-l.l_discount)) AS rev " + _J3 +
           "GROUP BY l.l_returnflag ORDER BY rev DESC LIMIT 2", ordered=True)


# ── gathered (parent) value columns inside fused arithmetic expressions ──
def test_chain_arith_parent_value():
    _match("SELECT l.l_returnflag, SUM(o.o_totalprice*(1-l.l_discount)) AS x " + _J3 +
           "GROUP BY l.l_returnflag")
def test_chain_arith_parent_value_scaled():
    _match("SELECT l.l_returnflag, SUM(o.o_totalprice*0.5) AS x " + _J3 + "GROUP BY l.l_returnflag")
def test_chain_arith_two_parents():
    _match("SELECT l.l_returnflag, SUM(c.c_acctbal + o.o_totalprice) AS x " + _J3 +
           "GROUP BY l.l_returnflag")
def test_chain_arith_parent_value_parent_group():
    _match("SELECT c.c_mktsegment, SUM(o.o_totalprice*(1-l.l_discount)) AS x " + _J3 +
           "GROUP BY c.c_mktsegment")
def test_chain_arith_parent_value_where():
    _match("SELECT l.l_returnflag, SUM(o.o_totalprice*(1-l.l_discount)) AS x " + _J3 +
           "WHERE l.l_quantity > 25 GROUP BY l.l_returnflag")
def test_chain_arith_parent_value_minmax():
    _match("SELECT l.l_returnflag, MIN(o.o_totalprice*(1-l.l_discount)), "
           "MAX(o.o_totalprice*(1-l.l_discount)) " + _J3 + "GROUP BY l.l_returnflag")


# ── composite GROUP BY fused inline (no comp array) under codegen arithmetic ──
def test_chain_arith_q1_multikey():
    _match("SELECT l.l_returnflag, l.l_linestatus, "
           "SUM(l.l_extendedprice*(1-l.l_discount)*(1+l.l_tax)) AS rev " + _J3 +
           "GROUP BY l.l_returnflag, l.l_linestatus")
def test_chain_arith_two_parent_keys():
    _match("SELECT c.c_mktsegment, o.o_orderpriority, "
           "SUM(l.l_extendedprice*(1-l.l_discount)) AS rev " + _J3 +
           "GROUP BY c.c_mktsegment, o.o_orderpriority")
def test_chain_arith_multikey_avg_order():
    _match("SELECT c.c_mktsegment, l.l_returnflag, AVG(l.l_extendedprice*(1-l.l_discount)) AS a " + _J3 +
           "GROUP BY c.c_mktsegment, l.l_returnflag ORDER BY a DESC LIMIT 4", ordered=True)


# ── unified single-pass: multiple plain value aggregates over a multi-key group (no value matrix) ──
def test_chain_plain_multikey_multivalue():
    _match("SELECT l.l_returnflag, l.l_linestatus, SUM(l.l_extendedprice), SUM(l.l_quantity), COUNT(*) "
           + _J3 + "GROUP BY l.l_returnflag, l.l_linestatus")
def test_chain_plain_multikey_parent_multivalue():
    _match("SELECT c.c_mktsegment, o.o_orderpriority, SUM(l.l_extendedprice), COUNT(*), "
           "AVG(l.l_quantity) " + _J3 + "GROUP BY c.c_mktsegment, o.o_orderpriority")
def test_chain_mixed_plain_arith_onepass():
    _match("SELECT l.l_returnflag, COUNT(*), SUM(l.l_extendedprice), SUM(l.l_quantity), "
           "SUM(l.l_extendedprice*(1-l.l_discount)) AS rev, MIN(l.l_extendedprice), MAX(l.l_discount) "
           + _J3 + "GROUP BY l.l_returnflag")
def test_chain_singlekey_multivalue_distinct():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice), SUM(l.l_quantity), AVG(l.l_discount) "
           + _J3 + "GROUP BY l.l_returnflag")


# ── WHERE-predicate fusion: numeric / datetime comparisons evaluated inline in the kernel ──
def test_chain_where_fused_numeric():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice*(1-l.l_discount)) "
           + _J3 + "WHERE l.l_quantity > 25 GROUP BY l.l_returnflag")
def test_chain_where_fused_between_and():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice) " + _J3 +
           "WHERE l.l_quantity BETWEEN 10 AND 40 AND l.l_discount > 0.02 GROUP BY l.l_returnflag")
def test_chain_where_fused_date_range_parent():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice) " + _J3 +
           "WHERE o.o_orderdate >= DATE '1994-01-01' AND o.o_orderdate < DATE '1995-01-01' "
           "GROUP BY l.l_returnflag")
def test_chain_where_fused_or():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_quantity < 5 OR l.l_quantity > 45 GROUP BY l.l_returnflag")
def test_chain_where_string_fallback():            # string predicate -> materialised mask, still correct
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice) " + _J3 +
           "WHERE c.c_mktsegment = 'BUILDING' GROUP BY l.l_returnflag")
def test_chain_where_mixed_string_numeric():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice*(1-l.l_discount)) " + _J3 +
           "WHERE c.c_mktsegment = 'BUILDING' AND l.l_quantity > 20 GROUP BY l.l_returnflag")


# ── string-equality WHERE fusion (inline dictionary-code comparison, no materialised mask) ──
def test_chain_where_fused_string_fact():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice) " + _J3 +
           "WHERE l.l_shipmode = 'AIR' GROUP BY l.l_returnflag")
def test_chain_where_fused_string_neq():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_shipmode != 'AIR' GROUP BY l.l_returnflag")
def test_chain_where_fused_string_parent():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice) " + _J3 +
           "WHERE c.c_mktsegment = 'BUILDING' GROUP BY l.l_returnflag")
def test_chain_where_fused_string_or():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_shipmode = 'AIR' OR l.l_shipmode = 'RAIL' GROUP BY l.l_returnflag")
def test_chain_where_fused_string_and_numeric():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice*(1-l.l_discount)) " + _J3 +
           "WHERE c.c_mktsegment = 'BUILDING' AND l.l_quantity > 20 GROUP BY l.l_returnflag")
def test_chain_where_fused_string_missing_value():     # literal not in dict -> matches nothing
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_shipmode = 'NOSUCHMODE' GROUP BY l.l_returnflag")


# ── IN (string OR-of-codes / numeric OR-of-values) and column-vs-column comparison fusion ──
def test_chain_where_fused_string_in():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_shipmode IN ('MAIL','SHIP','AIR') GROUP BY l.l_returnflag")
def test_chain_where_fused_numeric_in():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice) " + _J3 +
           "WHERE l.l_quantity IN (10,20,30,40) GROUP BY l.l_returnflag")
def test_chain_where_fused_col_vs_col():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_commitdate < l.l_receiptdate GROUP BY l.l_returnflag")
def test_chain_where_fused_q12_clause():
    _match("SELECT l.l_shipmode, COUNT(*) " + _J3 +
           "WHERE l.l_shipmode IN ('MAIL','SHIP') AND l.l_commitdate < l.l_receiptdate "
           "AND l.l_shipdate < l.l_commitdate AND l.l_receiptdate >= DATE '1994-01-01' "
           "AND l.l_receiptdate < DATE '1995-01-01' GROUP BY l.l_shipmode")
def test_chain_where_fused_in_none_present():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_shipmode IN ('NOSUCH','ZZZ') GROUP BY l.l_returnflag")


# ── code-LUT predicate fusion: LIKE / string-ordering / IS NULL / computed-vs-literal ──
def test_chain_where_fused_like_prefix():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_shipmode LIKE 'A%' GROUP BY l.l_returnflag")
def test_chain_where_fused_like_contains():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE o.o_orderpriority LIKE '%URGENT%' GROUP BY l.l_returnflag")
def test_chain_where_fused_like_underscore():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_shipmode LIKE 'R_IL' GROUP BY l.l_returnflag")
def test_chain_where_fused_not_like():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_shipmode NOT LIKE 'A%' GROUP BY l.l_returnflag")
def test_chain_where_fused_string_ordering():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE o.o_orderpriority > '3-MEDIUM' GROUP BY l.l_returnflag")
def test_chain_where_fused_like_parent():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE c.c_mktsegment LIKE 'B%' GROUP BY l.l_returnflag")
def test_chain_where_fused_is_not_null():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_shipmode IS NOT NULL GROUP BY l.l_returnflag")
def test_chain_where_fused_computed_vs_literal():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_extendedprice * l.l_discount > 5000 GROUP BY l.l_returnflag")
def test_chain_where_fused_like_and_numeric():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice) " + _J3 +
           "WHERE l.l_shipmode LIKE '%AIL' AND l.l_quantity > 30 GROUP BY l.l_returnflag")


# ── IS NULL / IS NOT NULL / high-card LIKE resolve ON the fast path via a chain-gathered mask (mask_eval),
# even on a multi-table chain -- previously these bailed and the single-join fallback failed the chain. ──
def test_chain_where_numeric_is_null_fast():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_quantity IS NULL GROUP BY l.l_returnflag")
def test_chain_where_numeric_is_not_null_fast():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE l.l_quantity IS NOT NULL GROUP BY l.l_returnflag")
def test_chain_where_is_not_null_and_numeric_fast():
    _match("SELECT l.l_returnflag, SUM(l.l_extendedprice) " + _J3 +
           "WHERE l.l_quantity IS NOT NULL AND l.l_quantity > 30 GROUP BY l.l_returnflag")
def test_chain_where_string_is_null_fast():
    _match("SELECT l.l_returnflag, COUNT(*) " + _J3 +
           "WHERE c.c_mktsegment IS NOT NULL GROUP BY l.l_returnflag")


# ── FK-chain queries that can't take the fused fast path now resolve via the SHARED chain (gather + pandas
# tail) instead of hitting the old single-join cliff. Plain projection across a 3-table chain is the clearest
# case: the fast path is agg-only, so this always falls through -- and used to raise 'multi-join requires FK'. ──
def test_chain_plain_projection_three_table():
    _match("SELECT l.l_returnflag, o.o_orderpriority, c.c_mktsegment " + _J3 +
           "WHERE l.l_quantity > 49 ORDER BY l.l_returnflag, o.o_orderpriority, c.c_mktsegment",
           ordered=True, expect_fast=False)
def test_chain_plain_projection_distinct_cols():
    _match("SELECT l.l_orderkey, c.c_mktsegment " + _J3 +
           "WHERE l.l_quantity > 49 ORDER BY l.l_orderkey, c.c_mktsegment LIMIT 12",
           ordered=True, expect_fast=False)


# ── high-card multi-group: when the dense composite code space exceeds MULTI_GROUP_CEIL, the group key is
# hash-factorised to dense ids and the SAME fused kernel runs over them (previously this bailed to the pandas
# tail). Force the branch on the small fixture by lowering the ceiling. ──
def test_chain_highcard_multigroup_factorize():
    save = wdb_join.MULTI_GROUP_CEIL
    try:
        wdb_join.MULTI_GROUP_CEIL = 4                 # force factorise for any 2+ col composite
        _match("SELECT l.l_returnflag, l.l_shipmode, SUM(l.l_extendedprice) " + _J3 +
               "GROUP BY l.l_returnflag, l.l_shipmode")
        _match("SELECT l.l_returnflag, l.l_shipmode, o.o_orderpriority, COUNT(*) " + _J3 +
               "GROUP BY l.l_returnflag, l.l_shipmode, o.o_orderpriority")
    finally:
        wdb_join.MULTI_GROUP_CEIL = save
