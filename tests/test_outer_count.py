"""THE OUTER JOIN AS COUNTS OVER THE POINTER and PARTITION TOTALS WITHOUT A SORT (2026-10-05).
RIGHT / FULL / parent-side LEFT joins with aggregates answer from per-parent counts (no joined rows), with NULL
keys, NULL values, NULL group values, and extra ON conditions on either side; AGG(x) OVER (PARTITION BY g)
answers from one counting pass; COUNT(col) OVER counts non-NULL rows (it summed the values before); the
correlated MAX subquery skips rows whose correlation key is NULL. Every answer is DuckDB's."""
import sys, os, uuid, tempfile, shutil, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd

TMP = tempfile.gettempdir()
WDB = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
FLOOR = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_QMEM_STRICT='1')


def _norm(r):
    return tuple('NULL' if v is None else (('%.6g' % v) if isinstance(v, float) else str(v)) for v in r)


def test_outer_count_and_partition_totals_equal_duck():
    import duckdb
    from wdb_db import Database
    rng = np.random.default_rng(7)
    n, nd = 200_000, 300
    k = pd.array(rng.integers(0, nd + 40, n), dtype='Int64'); k[rng.random(n) < 0.02] = pd.NA
    v = pd.array(rng.integers(0, 1000, n), dtype='Int64'); v[rng.random(n) < 0.05] = pd.NA
    w = rng.random(n) * 100; w[rng.random(n) < 0.05] = np.nan
    f = pd.DataFrame({'k': k, 'v': v, 'w': w, 'o': np.arange(n, dtype=np.int64),
                      's': np.array(['a', 'bb', 'ccc'])[rng.integers(0, 3, n)]})
    zone = np.array(['z%d' % (i % 23) for i in range(nd)], dtype=object); zone[rng.random(nd) < 0.1] = None
    d = pd.DataFrame({'id': np.arange(nd, dtype=np.int64), 'zone': zone,
                      'tier': np.array(['gold', 'silver', 'tin'])[rng.integers(0, 3, nd)]})
    outer = [
        "SELECT d.zone, COUNT(f.v) FROM f RIGHT JOIN d ON f.k = d.id GROUP BY d.zone",
        "SELECT COUNT(*) FROM f FULL OUTER JOIN d ON f.k = d.id",
        "SELECT d.zone, COUNT(f.v), COUNT(*) FROM f FULL OUTER JOIN d ON f.k = d.id AND f.k < 30 GROUP BY d.zone",
        "SELECT d.tier, SUM(f.v), AVG(f.w), MIN(f.v), MAX(f.w), COUNT(d.zone) FROM f FULL JOIN d "
        "ON f.k = d.id AND d.tier = 'gold' GROUP BY d.tier",
        "SELECT d.zone, d.tier, COUNT(*) AS n FROM d LEFT JOIN f ON f.k = d.id GROUP BY d.zone, d.tier HAVING COUNT(*) > 600",
        "SELECT SUM(f.v), MAX(f.v), MIN(f.w) FROM f RIGHT JOIN d ON f.k = d.id AND f.v > 900",
        "SELECT d.zone, SUM(f.w) FROM f RIGHT JOIN d ON d.id = f.k AND f.s = 'bb' GROUP BY d.zone",
        "SELECT d.tier, COUNT(f.k) AS c FROM f RIGHT JOIN d ON f.k = d.id AND f.k > 10000 GROUP BY d.tier ORDER BY c",
    ]
    win = [
        "SELECT k, v, MAX(v) OVER (PARTITION BY k) AS m FROM f",
        "SELECT o, COUNT(v) OVER (PARTITION BY k) AS c, SUM(w) OVER (PARTITION BY k) AS s, MIN(w) OVER (PARTITION BY s) AS mi FROM f",
        "SELECT o, COUNT(w) OVER (PARTITION BY k ORDER BY o) AS c FROM f",
        "SELECT o, AVG(v) OVER (PARTITION BY k, s) AS a, COUNT(*) OVER (PARTITION BY s) AS n FROM f",
        "SELECT k, v FROM f WHERE v > (SELECT MAX(v) - 3 FROM f f2 WHERE f2.k = f.k)",
        "SELECT k, w FROM f WHERE w >= (SELECT MAX(w) - 0.5 FROM f f2 WHERE f2.k = f.k)",
    ]
    dd = os.path.join(TMP, 'oc_' + uuid.uuid4().hex[:8]); db_dir = dd + '_db'; os.makedirs(dd)
    old = {kk: os.environ.get(kk) for kk in FLOOR}
    try:
        con = duckdb.connect()
        for name, df in (('f', f), ('d', d)):
            pq = os.path.join(dd, name + '.parquet'); df.to_parquet(pq, index=False)
            subprocess.run([sys.executable, WDB, 'load', db_dir, name, pq], check=True, capture_output=True,
                           env=dict(os.environ, **FLOOR))
            con.execute("CREATE VIEW %s AS SELECT * FROM read_parquet('%s')" % (name, pq))
        os.environ.update(FLOOR)
        db = Database.open(db_dir)
        import wdb_join
        s0 = wdb_join._OUTER_COUNT_SERVED[0]
        for sql in outer + win:
            r = db.run(sql); got = sorted(_norm(x) for x in (r[0] if isinstance(r, tuple) else r))
            want = sorted(_norm(x) for x in con.execute(sql).fetchall())
            assert len(got) == len(want), (sql, len(got), len(want))
            bad = [(g, w_) for g, w_ in zip(got, want) if g != w_]
            assert not bad, (sql, bad[:5])
        assert wdb_join._OUTER_COUNT_SERVED[0] - s0 == len(outer), 'the outer count door did not answer every outer shape'
    finally:
        for kk, vv in old.items():
            if vv is None: os.environ.pop(kk, None)
            else: os.environ[kk] = vv
        shutil.rmtree(dd, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
