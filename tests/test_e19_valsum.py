"""THE SUM FROM THE BLOCK DICTIONARIES (2026-10-01): Segment.e19_value_sum (per block: pointer counts on a
board the size of the block's own dictionary, each entry weighed once, two 32-bit halves) must equal the
old exact fold -- fold_counts(np.bincount(codes), tab) -- for enc 19 with its labels by block AND on
shelves, with values near +-2^62 (where a one-limb int64 sum wraps), negative values and NULLs; it must
not build the decode; and SQL SUM/AVG through it must equal DuckDB."""
import sys, os, uuid, tempfile, shutil, contextlib, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode
import wdb_engine
import wdb_exactint as XI
import wdb_window as WN
from wdb_engine import Segment

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
    rng = np.random.default_rng(51)
    n = 300_001
    pool = np.concatenate([rng.integers(-(1 << 62), 1 << 62, 40_000), rng.integers(-5000, 5000, 2_000)])
    walk = np.clip(np.cumsum(rng.integers(-40, 41, n)) + 20_000, 0, pool.size - 1)   # time locality
    big = pool[walk].astype(np.int64)
    nul = pd.array(np.where(rng.random(n) < 0.03, np.nan, big[::-1].astype(np.float64)), dtype='Float64')
    nul = pd.array([None if pd.isna(x) else int(y) for x, y in zip(nul, big[::-1])], dtype='Int64')
    return pd.DataFrame({'big': big, 'nul': nul})


def test_e19_value_sum_equals_the_fold_both_layouts():
    df = _frame()
    seen = set()
    for shelves in ('0', '4'):
        d = os.path.join(TMP, 'vs_' + uuid.uuid4().hex[:8]); pq = d + '.parquet'
        os.makedirs(d)
        try:
            df.to_parquet(pq, index=False)
            with _env(WDB_E19_FORCE='1', WDB_E19_SHELVES=shelves, WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0'):
                wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'), stream=True)
            P = os.path.join(d, 't_0.wdb')
            for nm in ('big', 'nul'):
                s = Segment(P); c = s.cols[nm]
                if c.get('code_enc') != 19:
                    continue
                seen.add(('e19R' in c, nm))
                tab = np.asarray(WN._int_table(s, nm), np.int64)
                got = s.e19_value_sum(nm, tab)
                assert got is not None and nm not in s._codes, nm      # no decode was built
                cn = np.bincount(np.asarray(Segment(P)._raw_codes(nm)), minlength=tab.size)[:tab.size]
                assert got == (XI.fold_counts(cn, tab), int(cn.sum())), (nm, shelves, got)
                vals = df[nm].dropna().astype('int64').map(int)
                assert got[0] == sum(vals) and got[1] == len(vals), nm   # the exact Python sum
        finally:
            shutil.rmtree(d, ignore_errors=True)
            try: os.remove(pq)
            except OSError: pass
    assert any(sh for sh, _ in seen) and any(not sh for sh, _ in seen), seen   # both layouts ran


def test_sum_avg_through_sql_equal_duck():
    import duckdb
    from wdb_db import Database
    df = _frame()
    d = os.path.join(TMP, 'vq_' + uuid.uuid4().hex[:8]); pq = d + '.parquet'; db_dir = d + '_db'
    os.makedirs(d)
    try:
        df.to_parquet(pq, index=False)
        wdb = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
        # THE FLOOR (the board's configuration): no load answers, no sidecars -- run.py turns both on, and
        # then the load's block statistics answer SUM/AVG in float64 (off by 140 at this magnitude: a
        # separate matter, not this path's)
        floor = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0')
        env = dict(os.environ, WDB_E19_FORCE='1', **floor)
        subprocess.run([sys.executable, wdb, 'load', db_dir, 't', pq], check=True, capture_output=True, env=env)
        import wdb_blockstats as BS                   # the switch is read at import: set it for this test
        old_ans = BS._ANSWERS[0]; BS._ANSWERS[0] = False
        try:
            with _env(**floor):
                db = Database.open(db_dir)
                rs = [(sql, db.run(sql)) for sql in ('SELECT SUM(big) FROM t', 'SELECT AVG(big) FROM t',
                                                     'SELECT SUM(nul), AVG(nul) FROM t')]
        finally:
            BS._ANSWERS[0] = old_ans
        for sql, r in rs:
            got = (r[0] if isinstance(r, tuple) else r)[0]
            ref = duckdb.sql(sql.replace('FROM t', "FROM read_parquet('%s')" % pq)).fetchall()[0]
            for g, x in zip(got, ref):
                if isinstance(x, float):
                    assert abs(float(g) - x) <= abs(x) * 1e-15, (sql, g, x)
                else:
                    assert int(g) == int(x), (sql, g, x)
    finally:
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
        try: os.remove(pq)
        except OSError: pass
