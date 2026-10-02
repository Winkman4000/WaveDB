"""THE NARROW READ for Q11's shape (2026-10-02): SELECT a, b, COUNT(DISTINCT u) AS n FROM t WHERE b <> ''
GROUP BY a, b ORDER BY n DESC LIMIT k -- b's planes are the filter, a's codes by the zipper (or codes_at
when a is not sparse/tiered), u's codes at those rows only, the lane hunt 32 pairs a round. Equal to the
old path, and every returned count equal to DuckDB's for that pair, the top-k counts equal to DuckDB's."""
import sys, os, uuid, tempfile, shutil, subprocess, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_kernels as K

TMP = tempfile.gettempdir()


def test_zipper_and_lane_hunt_kernels():
    rng = np.random.default_rng(101)
    n = 200_000
    typed = np.sort(rng.choice(5 * n, n, replace=False)).astype(np.int64)
    posA = np.sort(rng.choice(5 * n, 2 * n, replace=False)).astype(np.int64)
    litA = rng.integers(1, 40, posA.size).astype(np.uint16)
    out = np.empty(n, np.uint16)
    K.pd_at_planes(typed, posA, litA, np.uint16(0), out, np.int64(7))
    dense = np.zeros(5 * n, np.uint16); dense[posA] = litA
    assert np.array_equal(out, dense[typed])
    key = rng.integers(0, 300, n).astype(np.int32)
    ut = rng.integers(0, 70_000, n).astype(np.uint32)
    lut = np.full(300, -1, np.int8); pick = rng.choice(300, 32, replace=False); lut[pick] = np.arange(32)
    for SH in (1, 6, 10, 17):
        NL = ((70_000 - 1) >> SH) + 1
        got = K.pd_hunt_lanes(key, ut, lut, np.int64(32), np.int64(SH), np.int64(NL), np.int64(5))
        for s, pk in enumerate(pick):
            assert int(got[s]) == np.unique(ut[key == pk]).size, (SH, s)
    at = rng.integers(0, 40, n).astype(np.uint16); bt = rng.integers(0, 90, n).astype(np.uint16)
    k2, cnt = K.pd_pair_count(at, bt, np.int64(90), np.int64(40 * 90), np.int64(6))
    assert np.array_equal(k2, at.astype(np.int32) * 90 + bt) and np.array_equal(cnt, np.bincount(k2, minlength=3600))


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
    rng = np.random.default_rng(102)
    n = 1_200_000
    models = np.array([''] + ['m%03d' % i for i in range(120)], dtype=object)
    mi = np.where(rng.random(n) < 0.9, 0, np.minimum(rng.zipf(1.4, n), 120))
    phone = np.where(rng.random(n) < 0.8, 0, rng.integers(1, 40, n))           # tiered: mostly 0
    plain = rng.integers(0, 30, n)                                               # not sparse
    user = (rng.zipf(1.2, n) % 400_000).astype(np.int64) * 7919 + 13
    return pd.DataFrame({'phone': phone.astype(np.int64), 'plain': plain.astype(np.int64),
                         'model': models[mi], 'uid': user})


def test_through_sql_equal_old_path_and_duck():
    import duckdb
    import wdb_pairdistinct as PD
    from wdb_db import Database
    df = _frame()
    d = os.path.join(TMP, 'pdn_' + uuid.uuid4().hex[:8]); pq = d + '.parquet'; db_dir = d + '_db'
    os.makedirs(d)
    floor = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0')
    try:
        df.to_parquet(pq, index=False)
        wdb = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
        subprocess.run([sys.executable, wdb, 'load', db_dir, 't', pq], check=True, capture_output=True,
                       env=dict(os.environ, **floor))
        calls = []
        real = PD._narrow
        PD._narrow = lambda *a9: calls.append(a9[1]) or real(*a9)
        try:
            with _env(**floor):
                db = Database.open(db_dir)
                db.run('SELECT COUNT(*) FROM t'); seg = next(iter(db._seg_cache.values()))[1]
                assert seg.cols['model'].get('code_enc') in (8, 9), seg.cols['model'].get('code_enc')
                for a in ('phone', 'plain'):
                    for k in (1, 5, 10, 40):
                        sql = ("SELECT %s, model, COUNT(DISTINCT uid) AS n FROM t WHERE model <> '' "
                               "GROUP BY %s, model ORDER BY n DESC LIMIT %d" % (a, a, k))
                        h0 = PD._HITS; n0 = len(calls)
                        new = [tuple(r) for r in db.run(sql)[0]]
                        assert PD._HITS == h0 + 1 and len(calls) == n0 + 1, ('pairdistinct did not serve', sql)
                        PD._NARROW[0] = False
                        try:
                            old = [tuple(r) for r in db.run(sql)[0]]
                        finally:
                            PD._NARROW[0] = True
                        assert new == old, (sql, new[:3], old[:3])
                        full = duckdb.sql("SELECT %s, model, COUNT(DISTINCT uid) FROM read_parquet('%s') "
                                          "WHERE model <> '' GROUP BY 1, 2" % (a, pq)).fetchall()
                        truth = {(int(x), str(y)): int(c) for x, y, c in full}
                        for x, y, c in new:
                            assert truth[(int(x), str(y))] == int(c), (sql, x, y, c)
                        top = sorted(truth.values(), reverse=True)[:k]
                        assert [int(c) for _, _, c in new] == top, (sql, [c for _, _, c in new], top)
            assert seg.cols['phone'].get('code_enc') in (8, 9), seg.cols['phone'].get('code_enc')   # the zipper ran
            assert seg.cols['plain'].get('code_enc') not in (8, 9)                                 # codes_at ran
        finally:
            PD._narrow = real
    finally:
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
        try: os.remove(pq)
        except OSError: pass
