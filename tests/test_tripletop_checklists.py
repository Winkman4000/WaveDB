"""THE CHECKLIST WALK for Q18's shape (2026-10-02): SELECT u, extract(minute FROM t) AS m, p, COUNT(*) FROM tab
GROUP BY u, m, p ORDER BY COUNT(*) DESC LIMIT k. The walk by 1-bit checklists, the minute from the time
column's staircase steps, the empty-phrase tally -- equal counts to the old walk (tt_survivors) and to
DuckDB: every returned triple's count is DuckDB's, and the top-k counts are DuckDB's."""
import sys, os, uuid, tempfile, shutil, subprocess, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd

TMP = tempfile.gettempdir()


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
    rng = np.random.default_rng(121)
    n = 1_500_000
    user = (rng.zipf(1.15, n) % 200_000).astype(np.int64) * 104729 + 7          # a few very heavy users
    phrases = np.array(['q%04d' % i for i in range(3000)], dtype=object)
    ph = np.where(rng.random(n) < 0.86, '', phrases[np.minimum(rng.zipf(1.3, n), 3000) - 1])
    t0 = np.datetime64('2013-07-01T00:00:00')
    secs = np.sort(rng.integers(0, 3 * 86400, n))                                # time-ordered: a staircase
    return pd.DataFrame({'uid': user, 'ph': ph, 'ts': t0 + secs.astype('timedelta64[s]')})


def test_checklist_walk_equals_old_walk_and_duck():
    import duckdb
    import wdb_tripletop as TT
    from wdb_db import Database
    df = _frame()
    d = os.path.join(TMP, 'ttc_' + uuid.uuid4().hex[:8]); pq = d + '.parquet'; db_dir = d + '_db'
    os.makedirs(d)
    floor = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0')
    try:
        df.to_parquet(pq, index=False)
        wdb = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
        subprocess.run([sys.executable, wdb, 'load', db_dir, 't', pq], check=True, capture_output=True,
                       env=dict(os.environ, WDB_E8_FORCE='1', **floor))      # the phrase column as the sparse dress
        calls = []
        real = TT._walk_checklists
        TT._walk_checklists = lambda *a9: (lambda r: calls.append(r is not None) or r)(real(*a9))
        try:
            with _env(**floor):
                db = Database.open(db_dir)
                db.run('SELECT COUNT(*) FROM t'); seg = next(iter(db._seg_cache.values()))[1]
                assert seg.cols['ph'].get('code_enc') in (8, 9), seg.cols['ph'].get('code_enc')
                assert seg.stairs('ts') is not None
                truth = {}
                for r in duckdb.sql("SELECT uid, extract(minute FROM ts), ph, COUNT(*) FROM read_parquet('%s') "
                                    "GROUP BY 1, 2, 3" % pq).fetchall():
                    truth[(int(r[0]), int(r[1]), str(r[2]))] = int(r[3])
                top = sorted(truth.values(), reverse=True)
                for k in (1, 3, 10, 25):
                    sql = ("SELECT uid, extract(minute FROM ts) AS m, ph, COUNT(*) FROM t "
                           "GROUP BY uid, m, ph ORDER BY COUNT(*) DESC LIMIT %d" % k)
                    h0 = TT._HITS; n0 = len(calls)
                    new = [tuple(x) for x in db.run(sql)[0]]
                    assert TT._HITS == h0 + 1 and len(calls) == n0 + 1 and calls[-1], ('the checklist walk did not serve', sql)
                    TT._TTBITS[0] = False
                    try:
                        old = [tuple(x) for x in db.run(sql)[0]]
                    finally:
                        TT._TTBITS[0] = True
                    assert [x[3] for x in new] == [x[3] for x in old], (sql, new[:3], old[:3])
                    for u, m, p, c in new:
                        assert truth[(int(u), int(m), str(p))] == int(c), (sql, u, m, p, c)
                    assert [int(x[3]) for x in new] == top[:k], (sql, [x[3] for x in new], top[:k])
        finally:
            TT._walk_checklists = real
    finally:
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
        try: os.remove(pq)
        except OSError: pass
