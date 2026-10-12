"""THE PARALLEL FACTORISE (wdb_kernels.parallel_factorize): dense group ids for one or two int64 key
words per row, on every core -- the atom under fused_agg's high-cardinality multi-key GROUP BY,
replacing pandas' single-threaded factorize and the serial hcomposite2 (2026-10-12).

Kernel: the ids must split rows into exactly the groups pandas finds, give back every row's key
from its group id, be dense, and not depend on the thread count. Through SQL: both fused roads
that now use it (the mixed-radix composite, and the two-word hashed composite past 2^62) must
answer as DuckDB does, and must actually have called it."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_kernels as K, wdb_encode, wdb_join
from wdb_db import Database


def _same_groups(gid, ref):
    pairs = np.unique(np.stack([gid, ref]), axis=1)          # a bijection between the two labelings
    return pairs.shape[1] == np.unique(gid).size == np.unique(ref).size


def _check(n, card, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(-card, card, n).astype(np.int64) * 7919
    b = rng.integers(0, 3, n).astype(np.int64) - (1 << 62)
    g1, k1, none = K.parallel_factorize(a)
    assert none is None
    g2, k20, k21 = K.parallel_factorize(a, b)
    if n == 0:
        assert g1.size == 0 and k1.size == 0 and g2.size == 0
        return
    assert _same_groups(g1, pd.factorize(a, sort=False)[0])
    assert np.array_equal(k1[g1], a) and g1.min() == 0 and g1.max() == k1.size - 1
    assert _same_groups(g2, pd.factorize(pd.MultiIndex.from_arrays([a, b]))[0])
    assert np.array_equal(k20[g2], a) and np.array_equal(k21[g2], b) and g2.max() == k20.size - 1


def test_empty():               _check(0, 1, 1)
def test_one_row():             _check(1, 1, 2)
def test_small():               _check(17, 5, 3)
def test_few_groups():          _check(100_000, 10, 4)
def test_near_unique():         _check(1_000_000, 900_000, 5)
def test_mid_cardinality():     _check(3_000_000, 50_000, 6)


def test_thread_count_independent():
    import numba
    rng = np.random.default_rng(7)
    a = rng.integers(0, 40_000, 2_000_000).astype(np.int64)
    b = rng.integers(0, 9, 2_000_000).astype(np.int64)
    g, _, _ = K.parallel_factorize(a, b)
    nt = numba.get_num_threads()
    try:
        numba.set_num_threads(max(1, nt // 3))
        g_few, _, _ = K.parallel_factorize(a, b)
    finally:
        numba.set_num_threads(nt)
    assert np.array_equal(g, g_few)


# ---------------------------------------------------------------- through SQL, both fused roads

_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None:
        return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'pfz_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    # a x b: ~5000 x ~5000 keys (composite far past the dense ceiling, inside one word);
    # u, v, w: three near-unique keys of ~1.8M values each (composite past 2^62: two words)
    _CON.execute("""CREATE TABLE t AS SELECT i AS id,
        (i * 7919) % 4999 AS a, (i * 104729) % 5003 AS b,
        (i * 2654435761) % 1800017 AS u, (i * 40503) % 1800011 AS v, (i * 12289) % 1799999 AS w,
        i % 13 AS x FROM range(2000000) t(i)""")
    _DB = Database.create(d)
    pq = os.path.join(d, 't.parquet')
    _CON.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    _DB.cat.add_table('t', [[c, 'int'] for c in ('id', 'a', 'b', 'u', 'v', 'w', 'x')])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON


def _through_sql(q):
    db, con = _fixture()
    calls = []
    orig = K.parallel_factorize
    def counted(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    K.parallel_factorize = counted
    try:
        f0 = wdb_join._FAST_HITS
        got = sorted(tuple(r) for r in db.run(q)[0])
        took_fused = wdb_join._FAST_HITS > f0
    finally:
        K.parallel_factorize = orig
    want = sorted(tuple(r) for r in con.execute(q).fetchall())
    assert got == want, (q, got[:3], want[:3])
    assert took_fused and calls, 'fused road with the parallel factorise not taken: %s' % q


def test_sql_two_keys_one_word():
    _through_sql("SELECT a, b, COUNT(*), SUM(x), MAX(x) FROM t GROUP BY a, b HAVING COUNT(*) > 1")

def test_sql_two_keys_filtered():
    _through_sql("SELECT a, b, COUNT(*), MIN(x) FROM t WHERE x <> 3 GROUP BY a, b HAVING COUNT(*) > 1")

def test_sql_three_keys_two_words():
    _through_sql("SELECT u, v, w, COUNT(*), SUM(x) FROM t GROUP BY u, v, w HAVING SUM(x) > 23")
