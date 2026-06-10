"""General auto-denormalization: a build-time stamp of a low-card parent column onto the child, plus
the planner rewrite that turns a join grouping by that stamped column into a single-table cube read.
Locks: (1) the rewrite fires and is answered from the cube, penny-exact vs DuckDB; (2) the rewrite
correctly BAILS (stays on the gather) on every shape where the single-table rewrite would not be
equivalent -- WHERE present, a non-stamped parent group key, a grain-trap measure over a stamped
column, a child-key join with no stamps, and a two-join query."""
import sys, os, tempfile, uuid, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, wdb_cube, wdb_join
from wdb_db import Database, _parse_sql_cached

_DB = None; _CON = None
_WT = {'BIGINT':'int','INTEGER':'int','VARCHAR':'string','DATE':'datetime'}
def _wt(t): return 'float' if t.startswith('DECIMAL') else _WT[t]

def _fixture():
    """customer + orders at sf=0.01; low-card customer cols stamped onto orders (cubes='auto') and the
    stamp registered, exactly as the build does. lineitem too, for the child-key / two-join bail cases."""
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect(); _CON.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=0.01)")
    d = os.path.join(tempfile.gettempdir(), f'denorm_{uuid.uuid4().hex[:8]}'); _DB = Database.create(d)
    def load(tbl, order_by, cubes=None):
        desc = _CON.execute(f"DESCRIBE {tbl}").fetchall()
        sel = ", ".join((f"CAST({c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else c[0]) for c in desc)
        pq = os.path.join(d, f'{tbl}.parquet')
        _CON.execute(f"COPY (SELECT {sel} FROM {tbl} ORDER BY {order_by}) TO '{pq}' (FORMAT parquet)")
        _DB.cat.add_table(tbl, [[c[0], _wt(c[1])] for c in desc])
        seg = f'{tbl}_0.wdb'; wdb_encode.encode(pq, os.path.join(d, seg), cubes=cubes); _DB.cat.add_segment(tbl, seg)
    load('customer', 'c_custkey')
    # stamp every low-card customer col onto orders (the general, self-measured selection), skipping any
    # whose target name already exists on the child (the FK key c_custkey -> o_custkey is already present)
    cust = _DB.open_segment(os.path.join(d, 'customer_0.wdb'), 'customer')
    odesc = _CON.execute("DESCRIBE orders").fetchall(); ocolset = {c[0] for c in odesc}
    stamp = [c for c in wdb_cube.low_card_stamp_cols(cust) if f'o_{c[2:]}' not in ocolset]
    assert 'c_mktsegment' in stamp and 'c_nationkey' in stamp, f"low-card dims must be stamped, got {stamp}"
    assert 'c_custkey' not in stamp, f"FK key must be skipped (o_custkey collision): {stamp}"
    ctypes = {x[0]: x[1] for x in _CON.execute("DESCRIBE customer").fetchall()}
    ocols = ", ".join((f"CAST(o.{c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else f"o.{c[0]}") for c in odesc)
    ssel = ", ".join(f"c.{c} AS o_{c[2:]}" for c in stamp)
    osch = [[c[0], _wt(c[1])] for c in odesc] + [[f"o_{c[2:]}", _wt(ctypes[c])] for c in stamp]
    opq = os.path.join(d, 'orders.parquet')
    _CON.execute(f"COPY (SELECT {ocols}, {ssel} FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey ORDER BY o.o_orderkey) TO '{opq}' (FORMAT parquet)")
    _DB.cat.add_table('orders', osch); wdb_encode.encode(opq, os.path.join(d, 'orders_0.wdb'), cubes='auto'); _DB.cat.add_segment('orders', 'orders_0.wdb')
    unmatched = _CON.execute("SELECT COUNT(*) FROM orders o LEFT JOIN customer c ON o.o_custkey=c.c_custkey WHERE c.c_custkey IS NULL").fetchone()[0]
    for c in stamp: _DB.cat.add_stamp('orders', f'o_{c[2:]}', 'customer', c, 'o_custkey', total=(unmatched == 0))
    load('lineitem', 'l_orderkey')
    _DB.create_fk_pointer('orders', 'o_custkey', 'customer', 'c_custkey')
    _DB.create_fk_pointer('lineitem', 'l_orderkey', 'orders', 'o_orderkey')
    return _DB, _CON

def _rows(r): return r[0] if isinstance(r, tuple) else r
def _key(rows): return sorted((str(x[0]), int(x[1]), round(float(x[2]), 2)) for x in rows)
def _bails(sql):
    db, _ = _fixture(); return wdb_join.denorm_rewrite(db, _parse_sql_cached(sql)) is None

def test_denorm_join_rewrites_to_cube_and_matches_duck():
    db, con = _fixture()
    q = ("SELECT c.c_mktsegment,COUNT(*),SUM(o.o_totalprice) FROM orders o "
         "JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment")
    assert wdb_join.denorm_rewrite(db, _parse_sql_cached(q)) is not None, "rewrite should fire"
    h = wdb_cube._CUBE_HITS; got = _key(_rows(db.run(q)))
    assert wdb_cube._CUBE_HITS == h + 1, "answer must come from the cube"
    exp = _key(con.execute(q).fetchall())
    assert got == exp, f"cube answer != duck join\n got={got}\n exp={exp}"

def test_denorm_bails_on_where():
    assert _bails("SELECT c.c_mktsegment,COUNT(*) FROM orders o JOIN customer c "
                  "ON o.o_custkey=c.c_custkey WHERE o.o_totalprice>100000 GROUP BY c.c_mktsegment")

def test_denorm_bails_on_nonstamped_parent_group():
    # the FK key is never stamped (high-card at scale / collision-skipped) -> grouping by it stays on the gather
    assert _bails("SELECT c.c_custkey,COUNT(*) FROM orders o JOIN customer c "
                  "ON o.o_custkey=c.c_custkey GROUP BY c.c_custkey")

def test_denorm_bails_on_grain_trap_measure():
    # SUM over a STAMPED parent column would sum the parent attribute once per child row -> must bail
    assert _bails("SELECT c.c_mktsegment,SUM(c.c_nationkey) FROM orders o JOIN customer c "
                  "ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment")

def test_denorm_bails_on_child_key_join_without_stamps():
    # lineitem<->orders grouping by a child-native col: lineitem carries no stamps -> gather, not rewrite
    assert _bails("SELECT l.l_returnflag,SUM(l.l_extendedprice) FROM lineitem l "
                  "JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY l.l_returnflag")

def test_denorm_bails_on_two_joins():
    assert _bails("SELECT c.c_mktsegment,SUM(l.l_extendedprice) FROM lineitem l "
                  "JOIN orders o ON l.l_orderkey=o.o_orderkey JOIN customer c "
                  "ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment")
