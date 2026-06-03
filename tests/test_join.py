"""Two-table INNER equi-join, verified against DuckDB on real TPC-H data (sf=0.01, generated
in-process). A module-level fixture loads a few TPC-H tables into WaveDB once (read-only) and a
parallel DuckDB connection holds the same tables as the oracle."""
import sys, os, tempfile, uuid, math, datetime, re
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode
from wdb_db import Database

_DB = None; _CON = None; _DIR = None
_WT = {'BIGINT': 'int', 'INTEGER': 'int', 'VARCHAR': 'string', 'DATE': 'datetime'}

def _wtype(t): return 'float' if t.startswith('DECIMAL') else _WT[t]

def _fixture():
    global _DB, _CON, _DIR
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect(); _CON.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=0.01)")
    _DIR = os.path.join(tempfile.gettempdir(), f'jointest_{uuid.uuid4().hex[:8]}')
    _DB = Database.create(_DIR)
    for tbl in ('region', 'nation', 'customer', 'orders', 'supplier'):
        desc = _CON.execute(f"DESCRIBE {tbl}").fetchall()
        sel = ", ".join((f"CAST({c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else c[0]) for c in desc)
        pq = os.path.join(_DIR, f'{tbl}.parquet')
        _CON.execute(f"COPY (SELECT {sel} FROM {tbl}) TO '{pq}' (FORMAT parquet)")
        _DB.cat.add_table(tbl, [[c[0], _wtype(c[1])] for c in desc])
        seg = f'{tbl}_0.wdb'; wdb_encode.encode(pq, os.path.join(_DIR, seg)); _DB.cat.add_segment(tbl, seg)
    return _DB, _CON

_TS = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(\.\d+)?$')
def _cell(c):
    if isinstance(c, Decimal): c = float(c)
    if isinstance(c, (datetime.datetime,)): c = c.strftime('%Y-%m-%d %H:%M:%S')
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
        elif a != b:
            return False
    return True

def _match(q, ordered=False):
    db, con = _fixture()
    g = _norm(db.run(q)[0]); e = _norm([tuple(r) for r in con.execute(q).fetchall()])
    if not ordered:
        k = lambda t: tuple((x is None, str(x)) for x in t); g = sorted(g, key=k); e = sorted(e, key=k)
    assert len(g) == len(e), f"{q}\n nG={len(g)} nE={len(e)}\n {g[:3]}\n {e[:3]}"
    for gr, er in zip(g, e):
        assert _eq(gr, er), f"{q}\n got={gr}\n exp={er}"

# ── the orders ⋈ customer atom ───────────────────────────────────────────────

def test_join_count():
    _match("SELECT COUNT(*) FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey")

def test_join_projection_filter_order_limit():
    _match("SELECT o.o_orderkey, c.c_name FROM orders o JOIN customer c "
           "ON o.o_custkey = c.c_custkey WHERE o.o_totalprice > 400000 "
           "ORDER BY o.o_orderkey LIMIT 10", ordered=True)

def test_join_group_sum_count():
    _match("SELECT c.c_mktsegment, COUNT(*), SUM(o.o_totalprice) "
           "FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey GROUP BY c.c_mktsegment")

def test_join_group_all_aggs():
    _match("SELECT c.c_mktsegment, MIN(o.o_totalprice), MAX(o.o_totalprice), AVG(o.o_totalprice), COUNT(o.o_orderkey) "
           "FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey GROUP BY c.c_mktsegment")

def test_join_minmax_date():
    _match("SELECT c.c_mktsegment, MIN(o.o_orderdate), MAX(o.o_orderdate) "
           "FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey GROUP BY c.c_mktsegment")

def test_join_minmax_string():
    _match("SELECT o.o_orderpriority, MIN(c.c_name), MAX(c.c_name) "
           "FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey GROUP BY o.o_orderpriority")

def test_join_where_and_or():
    _match("SELECT COUNT(*) FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey "
           "WHERE (o.o_totalprice > 200000 AND c.c_mktsegment = 'BUILDING') OR o.o_orderstatus = 'F'")

def test_join_where_between_in():
    _match("SELECT c.c_mktsegment, COUNT(*) FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey "
           "WHERE o.o_totalprice BETWEEN 100000 AND 200000 AND o.o_orderpriority IN ('1-URGENT','2-HIGH') "
           "GROUP BY c.c_mktsegment")

def test_join_filter_on_both_sides():
    _match("SELECT o.o_orderkey, o.o_totalprice, c.c_mktsegment FROM orders o JOIN customer c "
           "ON o.o_custkey = c.c_custkey WHERE c.c_mktsegment = 'AUTOMOBILE' AND o.o_totalprice > 300000 "
           "ORDER BY o.o_totalprice DESC LIMIT 8", ordered=True)

# ── other relationships (small dimension joins) ──────────────────────────────

def test_join_nation_region():
    _match("SELECT r.r_name, COUNT(*) FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey GROUP BY r.r_name")

def test_join_customer_nation():
    _match("SELECT n.n_name, COUNT(*), AVG(c.c_acctbal) FROM customer c JOIN nation n "
           "ON c.c_nationkey = n.n_nationkey GROUP BY n.n_name")

def test_join_reversed_on_order():
    # ON written parent.key = child.key (reversed) must resolve the same
    _match("SELECT COUNT(*) FROM orders o JOIN customer c ON c.c_custkey = o.o_custkey")

def test_join_unqualified_columns():
    # columns without table prefix resolve by membership
    _match("SELECT c_mktsegment, COUNT(*) FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey "
           "GROUP BY c_mktsegment")
