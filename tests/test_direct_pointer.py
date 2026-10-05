"""THE KEY AS AN ADDRESS (2026-10-04): a child->parent pointer over integer keys in a narrow range is a table indexed
by key -- equal to pandas' hash (Index.get_indexer) everywhere it answers: shuffled parents, offset ranges, int32 and
int64 keys; a repeated parent key or (unless allowed) a child key with no parent is not a pointer; keys too sparse
for a table are left to the hash. Through SQL, joins that build the pointer (with the decoded keys kept on the shelf)
and a top-k grouped on a unique key plus unique text columns answer as DuckDB does."""
import sys, os, uuid, tempfile, shutil, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd

TMP = tempfile.gettempdir()
WDB = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
FLOOR = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_QMEM_STRICT='1')


def _hash_ref(ck, pk):
    return pd.Index(pk).get_indexer(ck)


def test_direct_pointer_equals_the_hash():
    import wdb_join as J
    rng = np.random.default_rng(101)
    for n, lo, gap in ((1, 0, 1), (1000, 7, 3), (200_003, 1_000_000, 4), (50_000, 0, 1)):
        pk = (lo + np.arange(n, dtype=np.int64) * gap)[rng.permutation(n)]
        ck = pk[rng.integers(0, n, 3 * n + 1)]
        for dt in (np.int64, np.int32):
            p, c = pk.astype(dt), ck.astype(dt)
            got = J._direct_pointer(c, p)
            assert isinstance(got, np.ndarray) and np.array_equal(got, _hash_ref(c, p)), (n, dt)
        miss = np.concatenate([ck, [lo - 1, int(pk.max()) + 1, lo + 1 if gap > 1 else int(pk.max()) + 5]])
        assert J._direct_pointer(miss, pk) is False                          # a child with no parent: not a pointer
        got = J._direct_pointer(miss, pk, allow_miss=True)
        assert np.array_equal(got, _hash_ref(miss, pk)) and (got[-3:] == -1).all()
        if n > 1:
            dup = pk.copy(); dup[-1] = dup[0]
            assert J._direct_pointer(ck, dup) is False                       # a repeated parent key: not a pointer
    wide = np.array([0, 10 ** 9, 5], np.int64)
    assert J._direct_pointer(np.array([5, 0], np.int64), wide) is None       # too sparse for a table: the hash's
    assert J._direct_pointer(np.array([1.0]), np.array([1.0])) is None       # not integers: the hash's


def test_pointer_joins_through_sql_equal_duck():
    import duckdb
    from wdb_db import Database
    rng = np.random.default_rng(103)
    nc, no, nl = 20_000, 300_000, 1_200_000
    cust = pd.DataFrame({'c_custkey': np.arange(1, nc + 1, dtype=np.int64),
                         'c_name': ['Customer#%09d' % i for i in range(1, nc + 1)],
                         'c_phone': ['%02d-%07d' % (i % 25 + 10, (i * 7919) % 10 ** 7) for i in range(nc)],
                         'c_seg': np.array(['BUILDING', 'AUTO', 'MACHINERY'])[rng.integers(0, 3, nc)]})
    okeys = (np.arange(no, dtype=np.int64) // 8) * 32 + (np.arange(no) % 8) + 1   # TPC-H's sparse order keys
    orders = pd.DataFrame({'o_orderkey': okeys, 'o_custkey': rng.integers(1, nc + 1, no),
                           'o_date': rng.integers(8000, 10000, no)})
    lineitem = pd.DataFrame({'l_orderkey': np.sort(okeys[rng.integers(0, no, nl)]),
                             'l_price': np.round(rng.random(nl) * 1000, 2), 'l_ship': rng.integers(8000, 10500, nl)})
    d = os.path.join(TMP, 'dp_' + uuid.uuid4().hex[:8]); db_dir = d + '_db'; os.makedirs(d)
    old = {k: os.environ.get(k) for k in FLOOR}
    try:
        con = duckdb.connect()
        for name, df in (('customer', cust), ('orders', orders), ('lineitem', lineitem)):
            pq = os.path.join(d, name + '.parquet'); df.to_parquet(pq, index=False)
            subprocess.run([sys.executable, WDB, 'load', db_dir, name, pq], check=True, capture_output=True,
                           env=dict(os.environ, **FLOOR))
            con.execute("CREATE VIEW %s AS SELECT * FROM read_parquet('%s')" % (name, pq))
        sqls = [
            "SELECT l_orderkey, SUM(l_price) AS revenue, o_date FROM customer, orders, lineitem WHERE c_seg = 'BUILDING' "
            "AND c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_date < 9000 AND l_ship > 9000 "
            "GROUP BY l_orderkey, o_date ORDER BY revenue DESC, l_orderkey LIMIT 10",
            "SELECT c_custkey, c_name, c_phone, SUM(l_price) AS revenue FROM customer, orders, lineitem "
            "WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_date >= 9000 AND o_date < 9100 "
            "GROUP BY c_custkey, c_name, c_phone ORDER BY revenue DESC, c_custkey LIMIT 20",
            "SELECT COUNT(*), SUM(l_price) FROM orders, lineitem WHERE l_orderkey = o_orderkey AND o_date < 8500",
        ]
        os.environ.update(FLOOR)
        db = Database.open(db_dir)
        for rnd in range(2):                                     # the second round reads the shelved keys
            for sql in sqls:
                r = db.run(sql); got = [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)]
                want = [tuple(x) for x in con.execute(sql).fetchall()]
                assert len(got) == len(want), (rnd, sql, got[:3], want[:3])
                for g, w in zip(got, want):
                    for a, b in zip(g, w):
                        assert (abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))) if isinstance(b, float) \
                            else str(a) == str(b), (rnd, sql, g, w)
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
