"""THE PERSON COUNT FROM THE SHELVES (2026-10-02): Segment.raw_census of a shelved enc-19 column (pass A per
block on block-sized boards, pass B per shelf into its own slice) equals np.bincount of the decoded codes --
with NULLs, with values spread over many shelves -- without building the decode; the unshelved layout keeps
the decode + boards and equals too; and GROUP BY c ORDER BY COUNT(*) DESC LIMIT k through it equals DuckDB."""
import sys, os, uuid, tempfile, shutil, contextlib, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode
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
    rng = np.random.default_rng(131)
    n = 400_003
    pool = rng.integers(-(1 << 40), 1 << 40, 45_000)
    walk = np.clip(np.cumsum(rng.integers(-40, 41, n)) + 22_000, 0, pool.size - 1)   # time locality:
    big = pool[walk].astype(np.int64)                                                # uneven counts
    nul = pd.array([None if m else int(v) for m, v in zip(rng.random(n) < 0.04, big[::-1])], dtype='Int64')
    return pd.DataFrame({'big': big, 'nul': nul})


def test_census_from_the_shelves_equals_the_decode_both_layouts():
    df = _frame()
    seen = set()
    for shelves in ('0', '4'):
        d = os.path.join(TMP, 'ec_' + uuid.uuid4().hex[:8]); pq = d + '.parquet'
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
                shelved = 'e19R' in c
                seen.add(shelved)
                got = s.raw_census(nm, int(c['V']) + 2)
                if shelved:
                    assert nm not in s._codes, nm                  # counted from the dress: no decode built
                ref = np.bincount(np.asarray(Segment(P)._raw_codes(nm)), minlength=int(c['V']) + 2)
                assert got.size == ref.size and np.array_equal(got, ref), (nm, shelved)
                assert int(got.sum()) == int(s.N)
        finally:
            shutil.rmtree(d, ignore_errors=True)
            try: os.remove(pq)
            except OSError: pass
    assert True in seen and False in seen, seen                     # both layouts ran


def test_group_count_top_through_sql_equals_duck():
    import duckdb
    from wdb_db import Database
    df = _frame()
    d = os.path.join(TMP, 'ecq_' + uuid.uuid4().hex[:8]); pq = d + '.parquet'; db_dir = d + '_db'
    os.makedirs(d)
    floor = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0')
    try:
        df.to_parquet(pq, index=False)
        wdb = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
        subprocess.run([sys.executable, wdb, 'load', db_dir, 't', pq], check=True, capture_output=True,
                       env=dict(os.environ, WDB_E19_FORCE='1', WDB_E19_SHELVES='4', **floor))
        with _env(**floor):
            db = Database.open(db_dir)
            db.run('SELECT COUNT(*) FROM t'); seg = next(iter(db._seg_cache.values()))[1]
            assert seg.cols['big'].get('code_enc') == 19 and 'e19R' in seg.cols['big'], 'not shelved enc 19'
            for k in (1, 10, 50):
                sql = 'SELECT big, COUNT(*) FROM t GROUP BY big ORDER BY COUNT(*) DESC LIMIT %d' % k
                got = [tuple(x) for x in db.run(sql)[0]]
                truth = dict((int(a), int(b)) for a, b in duckdb.sql(
                    "SELECT big, COUNT(*) FROM read_parquet('%s') GROUP BY big" % pq).fetchall())
                for a, b in got:
                    assert truth[int(a)] == int(b), (sql, a, b)
                assert [int(b) for _, b in got] == sorted(truth.values(), reverse=True)[:k], sql
    finally:
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
        try: os.remove(pq)
        except OSError: pass
