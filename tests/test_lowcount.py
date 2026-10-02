"""THE LOW COUNT (2026-10-02, Q25): ORDER BY a value-sorted dictionary column LIMIT k needs exact counts of
its first k + 2 codes only. The counting kernels equal a full count (the sparse lane at every width
1..32, ragged tails; decoded codes); through SQL, SELECT c FROM t WHERE c <> '' ORDER BY c LIMIT k
equals DuckDB for duplicated and single first values, LIMITs past the distinct count, a sparse (tag 8)
column and a dense one; and the fallback (a walk past the bound) gives the same rows."""
import sys, os, uuid, tempfile, shutil, subprocess, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_kernels as K

TMP = tempfile.gettempdir()


def _pack_msb(vals, bits):
    m = ((vals[:, None].astype(np.uint64) >> np.arange(bits - 1, -1, -1, dtype=np.uint64)) & 1).astype(np.uint8)
    return np.packbits(m.reshape(-1))


def test_kernels_equal_a_full_count():
    rng = np.random.default_rng(71)
    for bits in range(1, 33):
        n = int(rng.integers(0, 5000))
        vals = rng.integers(0, 1 << bits, n, dtype=np.uint64)
        lane = _pack_msb(vals, bits)
        for T in (1, 3, 12, 1 << min(bits, 10)):
            for L in (1, 7, 16):
                got = K.e8_lowcount(lane, np.int64(n), np.int64(bits), np.int64(T), np.int64(L))
                ref = np.bincount(vals[vals < T].astype(np.int64), minlength=T)[:T]
                assert np.array_equal(got, ref), (bits, n, T, L)
    codes = rng.integers(0, 40, 100001).astype(np.uint32)
    for T in (1, 5, 40, 60):
        got = K.low_count(codes, np.int64(T), np.int64(16))
        assert np.array_equal(got, np.bincount(codes[codes < T].astype(np.int64), minlength=T)[:T]), T


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update(kv)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def _frame():
    rng = np.random.default_rng(72)
    n = 300_000
    pool = np.array(['ph%05d' % i for i in range(20000)], dtype=object)
    sparse = np.where(rng.random(n) < 0.87, '', pool[rng.integers(0, 20000, n)])
    sparse[[5, 900, 77777]] = ' a first'                      # the smallest value, three times
    sparse[[12, 250000]] = ' b second'                        # twice
    sparse[[40]] = ' c third'                                  # once
    dense = pool[rng.integers(0, 20000, n)]
    dense[rng.random(n) < 0.05] = ''
    dense[[3, 4, 5, 6, 7, 8, 9]] = ' aa'                     # seven of the smallest
    return pd.DataFrame({'sp': sparse, 'dn': dense})


def test_order_by_limit_equals_duck_and_the_fallback():
    import duckdb
    import wdb_valsort as VS
    import wdb_engine
    from wdb_db import Database
    df = _frame()
    d = os.path.join(TMP, 'lc_' + uuid.uuid4().hex[:8]); pq = d + '.parquet'; db_dir = d + '_db'
    os.makedirs(d)
    floor = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0')
    try:
        df.to_parquet(pq, index=False)
        wdb = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
        subprocess.run([sys.executable, wdb, 'load', db_dir, 't', pq], check=True, capture_output=True,
                       env=dict(os.environ, **floor))
        with _env(**floor):
            db = Database.open(db_dir)
            seg = next(iter(db._seg_cache.values()))[1] if db._seg_cache else None
            if seg is None:
                db.run('SELECT COUNT(*) FROM t'); seg = next(iter(db._seg_cache.values()))[1]
            assert seg.cols['sp'].get('code_enc') == 8, seg.cols['sp'].get('code_enc')   # the sparse lane ran
            assert seg.cols['dn'].get('code_enc') != 8
            sqls = ["SELECT %s FROM t WHERE %s <> '' ORDER BY %s LIMIT %d" % (c, c, c, k)
                    for c in ('sp', 'dn') for k in (1, 2, 3, 4, 6, 10, 25, 40000)]
            got = {}
            for sql in sqls:
                h0 = VS._HITS
                r = db.run(sql); got[sql] = [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)]
                assert VS._HITS == h0 + 1, ('valsort did not serve', sql)
            old = wdb_engine.Segment.low_counts
            try:                                       # a walk past the bound: the census path answers
                wdb_engine.Segment.low_counts = lambda self, nm, T: np.zeros(int(T), np.int64)
                for sql in sqls:
                    if sql.endswith('LIMIT 40000'):    # V < k + 2: the walk ends inside the bound,
                        continue                       # so all-zero counts never reach the fallback
                    r = db.run(sql)
                    assert [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)] == got[sql], ('fallback', sql)
            finally:
                wdb_engine.Segment.low_counts = old
        for sql in sqls:
            ref = duckdb.sql(sql.replace('FROM t', "FROM read_parquet('%s')" % pq)).fetchall()
            assert got[sql] == [tuple(x) for x in ref], (sql, got[sql][:5], ref[:5])
    finally:
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
        try: os.remove(pq)
        except OSError: pass
