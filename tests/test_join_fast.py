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
    order = {'region':'r_regionkey','nation':'n_nationkey','customer':'c_custkey','orders':'o_orderkey'}
    for tbl in ('region','nation','customer','orders'):
        desc = _CON.execute(f"DESCRIBE {tbl}").fetchall()
        sel = ", ".join((f"CAST({c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else c[0]) for c in desc)
        pq = os.path.join(d, f'{tbl}.parquet')
        _CON.execute(f"COPY (SELECT {sel} FROM {tbl} ORDER BY {order[tbl]}) TO '{pq}' (FORMAT parquet)")
        _DB.cat.add_table(tbl, [[c[0], _wt(c[1])] for c in desc])
        seg = f'{tbl}_0.wdb'; wdb_encode.encode(pq, os.path.join(d, seg)); _DB.cat.add_segment(tbl, seg)
    _DB.create_fk_pointer('orders', 'o_custkey', 'customer', 'c_custkey')
    _DB.create_fk_pointer('nation', 'n_regionkey', 'region', 'r_regionkey')
    _DB.create_fk_pointer('customer', 'c_nationkey', 'nation', 'n_nationkey')
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
