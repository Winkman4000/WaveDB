"""THE CENSUS AND THE FOLD IN ONE PASS (2026-10-30): Segment.e19_census_fold of a shelved enc-19 column over a
chunked integer dictionary equals i2_fold_sum over the full census and the exact Python sum of the values -- with
chunks that do not line up with the shelves (1,000-value chunks against ~11K-code shelves), shelf counts 4 and
16 -- and SUM / AVG through SQL equal DuckDB's with the fold on and off."""
import sys, os, uuid, tempfile, shutil, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import duckdb, wdb_encode, wdb_engine
from wdb_engine import Segment
from wdb_db import Database

TMP = tempfile.gettempdir()


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def _frame():
    rng = np.random.default_rng(2030)
    n = 400_003
    pool = np.unique(rng.integers(-(1 << 62), 1 << 62, 120_000, dtype=np.int64))   # past NUM_THRESHOLD: mode 2
    walk = np.clip(np.cumsum(rng.integers(-40, 41, n)) + pool.size // 2, 0, pool.size - 1)
    pick = np.where(rng.random(n) < 0.5, walk, rng.integers(0, pool.size, n))      # half local, half spread:
    big = pool[pick].astype(np.int64)                                                # uneven counts
    nul = pd.array([None if m else int(v) for m, v in zip(rng.random(n) < 0.04, big[::-1])], dtype='Int64')
    return pd.DataFrame({'big': big, 'nul': nul})


def test_census_fold_equals_the_census_and_the_sum():
    if wdb_engine._libzstd() is None:
        return
    df = _frame()
    want = {'big': (sum(int(v) for v in df['big']), len(df)),
            'nul': (sum(int(v) for v in df['nul'].dropna()), int(df['nul'].notna().sum()))}
    ran = 0
    for shelves in ('4', '16'):
        d = os.path.join(TMP, 'ecf_' + uuid.uuid4().hex[:8]); pq = d + '.parquet'
        os.makedirs(d)
        try:
            df.to_parquet(pq, index=False)
            with _env(WDB_E19_FORCE='1', WDB_E19_SHELVES=shelves, WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0',
                      WDB_I2CHUNK='1000', WDB_I2CHUNK_MIN='1000'):
                wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'), stream=True)
            P = os.path.join(d, 't_0.wdb')
            for nm in ('big', 'nul'):
                s = Segment(P); c = s.cols[nm]
                if c.get('code_enc') != 19 or 'e19R' not in c or not s.i2_foldable(nm):
                    continue
                got = s.e19_census_fold(nm)
                assert got is not None, nm
                nd = int(c.get('n_dict', c['V']))
                cn = Segment(P).raw_census(nm, int(c['V']))[:nd]
                ref = Segment(P).i2_fold_sum(nm, cn)
                assert got == (ref, int(cn.sum())), (nm, got, ref)
                assert got == want[nm], (nm, got, want[nm])
                assert nm not in s._codes                       # no decode built
                ran += 1
        finally:
            shutil.rmtree(d, ignore_errors=True)
            try: os.remove(pq)
            except OSError: pass
    assert ran >= 2, ran                                        # the fold ran, not a vacuous pass


def test_sql_sum_avg_equal_duck_fold_on_and_off():
    if wdb_engine._libzstd() is None:
        return
    df = _frame()
    d = os.path.join(TMP, 'ecq_' + uuid.uuid4().hex[:8]); pq = d + '.parquet'
    os.makedirs(d)
    try:
        df.to_parquet(pq, index=False)
        db = Database.create(d)
        db.cat.add_table('t', [['big', 'int'], ['nul', 'int']])
        with _env(WDB_E19_FORCE='1', WDB_E19_SHELVES='4', WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0',
                  WDB_I2CHUNK='1000', WDB_I2CHUNK_MIN='1000'):
            wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'), stream=True)
        db.cat.add_segment('t', 't_0.wdb')
        con = duckdb.connect(); con.register('df', df)
        for nm in ('big', 'nul'):
            sql_s = 'SELECT SUM(%s) FROM t' % nm
            sql_a = 'SELECT AVG(%s) FROM t' % nm
            ws = con.execute('SELECT SUM(%s)::HUGEINT, AVG(%s) FROM df' % (nm, nm)).fetchone()
            for on in (True, False):
                wdb_engine._E19FOLD[0] = on
                try:
                    db2 = Database.open(d)
                    assert int(db2.run(sql_s)[0][0][0]) == int(ws[0]), (nm, on)
                    a = float(db2.run(sql_a)[0][0][0])
                    assert abs(a - float(ws[1])) <= 1e-9 * max(1.0, abs(float(ws[1]))), (nm, on, a, ws[1])
                finally:
                    wdb_engine._E19FOLD[0] = True
    finally:
        shutil.rmtree(d, ignore_errors=True)
        try: os.remove(pq)
        except OSError: pass
