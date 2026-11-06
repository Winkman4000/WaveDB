"""THE PAIR ROAD (c126, 2026-11-05, TPC-H Q9): lineitem -> partsupp through (l_partkey, l_suppkey) = (ps_partkey,
ps_suppkey) -- unique only together, partsupp stored by ps_partkey -- is one road: binary search for the run of the
first key, a scan of the run for the second, built at the rows asked. Answers like DuckDB: the pair alone, the Q9 shape
(a parent LIKE, a computed year key, nation through supplier), a filter on partsupp itself; a pair that is not unique
stands down; switched off, the pair is not a road."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

_DB = None; _CON = None
ORDER = {'region': 'r_regionkey', 'nation': 'n_nationkey', 'supplier': 's_suppkey', 'part': 'p_partkey',
         'partsupp': 'ps_partkey, ps_suppkey', 'orders': 'o_orderkey', 'lineitem': 'l_orderkey, l_linenumber'}


def _fixture():
    """sf=0.01 with part, supplier and partsupp (the join tests' fixture has none of them)"""
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    import duckdb, wdb_encode, helpers as _H, test_join_fast as J
    from wdb_db import Database
    _CON = duckdb.connect(); _CON.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=0.01)")
    d = os.path.join(tempfile.gettempdir(), f'pairroad_{uuid.uuid4().hex[:8]}')
    _DB = Database.create(d); _H.register_dir(d)
    for tbl, by in ORDER.items():
        desc = _CON.execute(f"DESCRIBE {tbl}").fetchall()
        sel = ", ".join((f"CAST({c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else c[0]) for c in desc)
        pq = os.path.join(d, f'{tbl}.parquet')
        _CON.execute(f"COPY (SELECT {sel} FROM {tbl} ORDER BY {by}) TO '{pq}' (FORMAT parquet)")
        _DB.cat.add_table(tbl, [[c[0], J._wt(c[1])] for c in desc])
        seg = f'{tbl}_0.wdb'; wdb_encode.encode(pq, os.path.join(d, seg)); _DB.cat.add_segment(tbl, seg)
    for c, ck, p, pk in (('nation', 'n_regionkey', 'region', 'r_regionkey'), ('supplier', 's_nationkey', 'nation', 'n_nationkey'),
                         ('lineitem', 'l_orderkey', 'orders', 'o_orderkey'), ('lineitem', 'l_partkey', 'part', 'p_partkey'),
                         ('lineitem', 'l_suppkey', 'supplier', 's_suppkey')):
        _DB.create_fk_pointer(c, ck, p, pk)
    return _DB, _CON


PAIR = "FROM lineitem, partsupp WHERE ps_partkey = l_partkey AND ps_suppkey = l_suppkey"
WALKED = [
    "SELECT SUM(ps_supplycost * l_quantity), COUNT(*) " + PAIR,
    "SELECT l_returnflag, SUM(l_extendedprice - ps_supplycost * l_quantity) " + PAIR + " AND l_quantity > 30 GROUP BY l_returnflag",
    "SELECT COUNT(*), SUM(ps_availqty) " + PAIR + " AND ps_availqty < 2000 AND l_discount > 0.05",
    # the Q9 shape
    "SELECT nation, o_year, SUM(amount) AS sum_profit FROM (SELECT n_name AS nation, YEAR(o_orderdate) AS o_year, "
    "l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount FROM part, supplier, lineitem, partsupp, "
    "orders, nation WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey AND "
    "p_partkey = l_partkey AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey AND p_name LIKE '%green%') AS profit "
    "GROUP BY nation, o_year ORDER BY nation, o_year DESC",
]
NOT_A_PAIR = "SELECT COUNT(*) FROM lineitem, partsupp WHERE ps_suppkey = l_suppkey AND ps_availqty = l_quantity"


def _same(db, con, q, ordered=False):
    import controller, test_join_fast as J
    exp = J._norm([tuple(r) for r in con.execute(q).fetchall()])
    r = db.run(q)
    got = J._norm(r[0] if isinstance(r, tuple) else r)
    if not ordered: got, exp = sorted(got), sorted(exp)
    assert len(got) == len(exp), (q, len(got), len(exp))
    for g, e in zip(got, exp):
        assert J._eq(g, e), (q, g, e)


def test_pair_road_answers():
    import wdb_join as WJ
    db, con = _fixture()
    for i, q in enumerate(WALKED):
        h0 = WJ._PAIR_ROAD_HITS[0]
        _same(db, con, q, ordered=(i == len(WALKED) - 1))
        assert WJ._PAIR_ROAD_HITS[0] >= h0 + 1, q


def test_not_a_pair_stands_down():
    import wdb_join as WJ
    db, con = _fixture()
    h0 = WJ._PAIR_ROAD_HITS[0]
    try:
        _same(db, con, NOT_A_PAIR)
    except NotImplementedError:
        pass                                         # the old road's own limits are not this test's
    assert WJ._PAIR_ROAD_HITS[0] == h0


def test_switched_off():
    import wdb_join as WJ
    db, con = _fixture()
    o0 = WJ._PAIR_ROAD[0]
    try:
        WJ._PAIR_ROAD[0] = False
        h0 = WJ._PAIR_ROAD_HITS[0]
        try:
            _same(db, con, WALKED[0])
        except NotImplementedError:
            pass                                     # without the pair road the old road cannot join them
        assert WJ._PAIR_ROAD_HITS[0] == h0
    finally:
        WJ._PAIR_ROAD[0] = o0
