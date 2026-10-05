"""THE COUNT INSTEAD OF THE SEARCH (2026-10-04, TPC-H Q21): [NOT] EXISTS (another row of my group, different from me
on x, meeting P) is (rows of my group meeting P) minus (rows of my group meeting P with my own x). The run kernels
equal a brute force (short runs, runs past 64 rows, with and without P, both polarities); the census equals it too;
through SQL the Q21 pair, a lone EXISTS, a lone NOT EXISTS without P, a text x, big groups, a different inner table
and an unsorted self-table answer as DuckDB does -- and every one of them is served by the rewrite."""
import sys, os, uuid, tempfile, shutil, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd

TMP = tempfile.gettempdir()
WDB = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
FLOOR = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_QMEM_STRICT='1')


def _brute(ki, xi, pm, ko, xo, neg):
    out = np.empty(ko.shape[0], bool)
    for r in range(ko.shape[0]):
        sel = (ki == ko[r]) & (xi != xo[r])
        if pm is not None: sel &= pm
        hit = bool(sel.any())
        out[r] = (not hit) if neg else hit
    return out


def test_run_bounds():
    import wdb_kernels as K
    rng = np.random.default_rng(7)
    for n in (0, 1, 2, 63, 64, 65, 1000, 100_001):
        k = np.sort(rng.integers(0, max(1, n // 3), n)).astype(np.int32)
        st = K.run_bounds(k)
        want = np.concatenate([[0], np.flatnonzero(np.diff(k) != 0) + 1, [n]]) if n else np.array([0])
        assert np.array_equal(st, want), n
        if n > 2:
            k2 = k.copy(); k2[n // 2], k2[-1] = k2[-1] + 1, k2[0] - 1
            assert K.run_bounds(k2).size == 0, n                             # a decrease: not runs


def test_runs_and_census_equal_the_brute_force():
    import wdb_kernels as K, wdb_groupexists as G
    rng = np.random.default_rng(11)
    for sizes in ((1, 2, 3, 7), (65, 130, 1, 200, 2)):                       # short runs; runs past the pairwise cut
        k = np.repeat(np.arange(len(sizes) * 40), np.tile(sizes, 40)).astype(np.int64)
        n = k.size
        x = rng.integers(0, 4, n).astype(np.int64)
        for pm in (None, rng.random(n) < 0.4):
            for neg in (False, True):
                want = _brute(k, x, pm, k, x, neg)
                st = K.run_bounds(k)
                out = np.empty(n, np.bool_)
                pu = pm.view(np.uint8) if pm is not None else np.ones(1, np.uint8)
                K.pruns_others(st, x, np.ascontiguousarray(pu), pm is not None, neg, out)
                assert np.array_equal(out, want), (sizes, neg, pm is None)
                perm = rng.permutation(n)                                    # the census: any order, other outers
                ko = np.concatenate([k[perm][:500], [10 ** 6]]); xo = np.concatenate([x[perm][:500], [1]])
                assert np.array_equal(G._census(k, x, pm, ko, xo, neg), _brute(k, x, pm, ko, xo, neg))


def test_counted_exists_through_sql_equal_duck():
    import duckdb
    import wdb_groupexists as G
    from wdb_db import Database
    rng = np.random.default_rng(13)
    ns, no = 400, 60_000
    nation = pd.DataFrame({'n_nationkey': np.arange(5, dtype=np.int64), 'n_name': ['A', 'B', 'C', 'D', 'E']})
    supplier = pd.DataFrame({'s_suppkey': np.arange(1, ns + 1, dtype=np.int64),
                             's_name': ['Supplier#%06d' % i for i in range(1, ns + 1)],
                             's_nationkey': rng.integers(0, 5, ns)})
    orders = pd.DataFrame({'o_orderkey': np.arange(1, no + 1, dtype=np.int64) * 4,
                           'o_status': np.array(['F', 'O', 'P'])[rng.integers(0, 3, no)]})
    per = rng.integers(1, 8, no); per[rng.integers(0, no, 30)] = rng.integers(65, 300, 30)   # a few big groups
    lk = np.repeat(orders.o_orderkey.values, per)
    nl = lk.size
    commit = rng.integers(9000, 9100, nl)
    lineitem = pd.DataFrame({'l_orderkey': lk, 'l_suppkey': rng.integers(1, ns + 1, nl),
                             'l_commit': commit, 'l_receipt': commit + rng.integers(-30, 31, nl),
                             'l_mode': np.array(['AIR', 'RAIL', 'SHIP', 'MAIL'])[rng.integers(0, 4, nl)]})
    nsh = 90_000                                                             # unsorted, another table
    ship = pd.DataFrame({'sh_orderkey': orders.o_orderkey.values[rng.integers(0, no, nsh)],
                         'sh_suppkey': rng.integers(1, ns + 1, nsh), 'sh_flag': rng.integers(0, 2, nsh)})
    late = "l3.l_receipt > l3.l_commit"
    sqls = [
        # TPC-H Q21's shape: the pair, P on the NOT EXISTS, grouped on text, top-k
        "SELECT s_name, COUNT(*) AS numwait FROM supplier, lineitem l1, orders, nation WHERE s_suppkey = l1.l_suppkey "
        "AND o_orderkey = l1.l_orderkey AND o_status = 'F' AND l1.l_receipt > l1.l_commit AND EXISTS (SELECT * FROM "
        "lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey) AND NOT EXISTS (SELECT * "
        "FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey AND " + late + ") "
        "AND s_nationkey = n_nationkey AND n_name = 'B' GROUP BY s_name ORDER BY numwait DESC, s_name LIMIT 50",
        # a lone EXISTS with P, scalar
        "SELECT COUNT(*), SUM(l1.l_commit) FROM lineitem l1, orders WHERE l1.l_orderkey = o_orderkey AND "
        "o_status <> 'P' AND EXISTS (SELECT * FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey AND "
        "l3.l_suppkey <> l1.l_suppkey AND " + late + ")",
        # a lone NOT EXISTS without P, grouped
        "SELECT o_status, COUNT(*) FROM lineitem l1, orders WHERE l1.l_orderkey = o_orderkey AND NOT EXISTS "
        "(SELECT * FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey) "
        "GROUP BY o_status ORDER BY o_status",
        # x is text (codes compare within the column), the <> written outer-first
        "SELECT o_status, COUNT(*) FROM lineitem l1, orders WHERE l1.l_orderkey = o_orderkey AND EXISTS "
        "(SELECT * FROM lineitem l2 WHERE l1.l_orderkey = l2.l_orderkey AND l1.l_mode <> l2.l_mode AND "
        "l2.l_suppkey < 100) GROUP BY o_status ORDER BY o_status",
        # another inner table, unsorted: the census
        "SELECT o_status, COUNT(*) FROM lineitem l1, orders WHERE l1.l_orderkey = o_orderkey AND EXISTS "
        "(SELECT * FROM ship s WHERE s.sh_orderkey = l1.l_orderkey AND s.sh_suppkey <> l1.l_suppkey AND "
        "s.sh_flag = 1) GROUP BY o_status ORDER BY o_status",
        # the unsorted table against itself: the census over one column's identities
        "SELECT o_status, COUNT(*) FROM ship s1, orders WHERE s1.sh_orderkey = o_orderkey AND NOT EXISTS "
        "(SELECT * FROM ship s2 WHERE s2.sh_orderkey = s1.sh_orderkey AND s2.sh_suppkey <> s1.sh_suppkey) "
        "GROUP BY o_status ORDER BY o_status",
    ]
    d = os.path.join(TMP, 'ge_' + uuid.uuid4().hex[:8]); db_dir = d + '_db'; os.makedirs(d)
    old = {k: os.environ.get(k) for k in FLOOR}
    served = []
    orig = G.rewrite
    G.rewrite = lambda db, tree: served.append(orig(db, tree)) or served[-1]
    try:
        con = duckdb.connect()
        for name, df in (('nation', nation), ('supplier', supplier), ('orders', orders), ('lineitem', lineitem),
                         ('ship', ship)):
            pq = os.path.join(d, name + '.parquet'); df.to_parquet(pq, index=False)
            subprocess.run([sys.executable, WDB, 'load', db_dir, name, pq], check=True, capture_output=True,
                           env=dict(os.environ, **FLOOR))
            con.execute("CREATE VIEW %s AS SELECT * FROM read_parquet('%s')" % (name, pq))
        os.environ.update(FLOOR)
        db = Database.open(db_dir)
        for sql in sqls:
            del served[:]
            r = db.run(sql); got = [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)]
            want = [tuple(x) for x in con.execute(sql).fetchall()]
            assert sum(served) >= 1, ('not served by the count', sql)
            assert len(got) == len(want) and len(want) > 0, (sql, got[:3], want[:3])
            for g, w in zip(got, want):
                for a, b in zip(g, w):
                    assert (abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))) if isinstance(b, float) \
                        else str(a) == str(b), (sql, g, w)
    finally:
        G.rewrite = orig
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
