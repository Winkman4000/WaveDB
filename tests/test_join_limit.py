"""THE LIMIT FIRST (2026-10-05): a row-emitting join with LIMIT and no ORDER BY reads only the first OFFSET+LIMIT
joined rows. Every row it returns is a row of the full join (INNER, LEFT, RIGHT, FULL, with WHERE on either side),
the count is DuckDB's, an ORDER BY ... LIMIT ... OFFSET answer is DuckDB's exactly (OFFSET was ignored on this road
before), and SELECT DISTINCT over the join answers as DuckDB does."""
import sys, os, uuid, tempfile, shutil, subprocess
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd

TMP = tempfile.gettempdir()
WDB = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
FLOOR = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_QMEM_STRICT='1')


def _norm(r):
    return tuple('NULL' if v is None else (('%.6g' % v) if isinstance(v, float) else str(v)) for v in r)


def test_join_limit_equals_duck():
    import duckdb
    from wdb_db import Database
    rng = np.random.default_rng(21)
    n, nd = 300_000, 1000
    f = pd.DataFrame({'k': rng.integers(0, nd + 200, n), 'v': rng.integers(0, 1000, n),
                      's': np.array(['a', 'bb', 'ccc', ''])[rng.integers(0, 4, n)]})
    d = pd.DataFrame({'id': np.arange(nd, dtype=np.int64), 'tier': np.array(['gold', 'silver'])[rng.integers(0, 2, nd)],
                      'zone': ['z%d' % (i % 37) for i in range(nd)]})
    sub = [   # (sql without LIMIT, LIMIT clause) -- every limited row must be a row of the unlimited answer
        ("SELECT f.s, d.zone FROM f JOIN d ON f.k = d.id WHERE d.tier = 'gold' AND f.s <> ''", 'LIMIT 1000'),
        ("SELECT f.v, d.zone FROM f LEFT JOIN d ON f.k = d.id WHERE f.v < 100", 'LIMIT 500'),
        ("SELECT f.v, d.zone FROM f RIGHT JOIN d ON f.k = d.id", 'LIMIT 700'),
        ("SELECT f.v, d.zone FROM f FULL OUTER JOIN d ON f.k = d.id WHERE f.v > 990", 'LIMIT 400'),
        ("SELECT f.v, d.id FROM f JOIN d ON f.k = d.id", 'LIMIT 10 OFFSET 5'),
        ("SELECT f.v, d.zone FROM f JOIN d ON f.k = d.id WHERE f.v = 3", 'LIMIT 100000'),   # fewer rows than the limit
        ("SELECT f.v, d.zone FROM f LEFT JOIN d ON f.k = d.id WHERE d.tier = 'gold' AND f.s = 'bb'", 'LIMIT 300'),
        ("SELECT f.v, d.zone FROM f LEFT JOIN d ON f.k = d.id WHERE d.id IS NULL", 'LIMIT 50'),   # the unmatched rows
        ("SELECT f.s, d.tier FROM f JOIN d ON f.k = d.id WHERE f.s <> 'a' AND f.v BETWEEN 10 AND 20", 'LIMIT 2000'),
    ]
    exact = [
        "SELECT f.v, d.zone FROM f JOIN d ON f.k = d.id ORDER BY f.v DESC, d.zone LIMIT 20 OFFSET 7",
        "SELECT DISTINCT d.zone FROM f JOIN d ON f.k = d.id WHERE f.v < 50",
    ]
    dd = os.path.join(TMP, 'jl_' + uuid.uuid4().hex[:8]); db_dir = dd + '_db'; os.makedirs(dd)
    old = {k: os.environ.get(k) for k in FLOOR}
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
        p0 = wdb_join._PIECES_SERVED[0]
        for base, lim in sub:
            r = db.run(base + ' ' + lim); got = [_norm(x) for x in (r[0] if isinstance(r, tuple) else r)]
            full = Counter(_norm(x) for x in con.execute(base).fetchall())
            want_n = len(con.execute(base + ' ' + lim).fetchall())
            assert len(got) == want_n, (base, lim, len(got), want_n)
            assert not (Counter(got) - full), (base, lim, 'rows outside the join')
        assert wdb_join._PIECES_SERVED[0] - p0 >= 3, 'the piecewise road never answered'   # INNER / LEFT shapes
        for sql in exact:
            r = db.run(sql); got = [_norm(x) for x in (r[0] if isinstance(r, tuple) else r)]
            want = [_norm(x) for x in con.execute(sql).fetchall()]
            if 'DISTINCT' in sql: got, want = sorted(got), sorted(want)
            assert got == want, (sql, got[:5], want[:5])
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        shutil.rmtree(dd, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
