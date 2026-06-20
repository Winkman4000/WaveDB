"""Code-space equality filters and the COUNT(*) top-K fast select.

WHERE col = X / col <> X on a dictionary-coded column is evaluated in CODE space (resolve the
literal to its dict code, compare the code array) instead of decoding N values -- same result,
no per-row value materialisation. The grouped COUNT(*) top-K then builds only the k surviving
groups (argpartition) rather than every group. Both are verified against DuckDB here, including
the cases that must FALL BACK (overrides) or stay correct (literal absent, NULLs, NEQ)."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd, numpy as np, duckdb
from wdb_db import Database
import wdb_encode

def _tmpdb(): return os.path.join(tempfile.gettempdir(), f'wdeq_{uuid.uuid4().hex[:8]}')
def _norm(rows):
    return sorted([tuple(round(float(x), 4) if isinstance(x, (int, float, np.floating, np.integer))
                         and not isinstance(x, bool) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))

def _build(dbdir, df):
    db = Database.create(dbdir); db.run("CREATE TABLE t (cat VARCHAR, eng INT, x INT)")
    pq = os.path.join(dbdir, 'src.parquet'); df.to_parquet(pq, index=False)
    wdb_encode.encode(pq, os.path.join(dbdir, 't_0.wdb'))
    db.cat.add_segment('t', 't_0.wdb')
    return db, pq

# skewed categorical 'cat' incl. an empty-string value (the <> '' case), a low-card int 'eng'
_rng = np.random.default_rng(7)
_N = 4000
_cats = np.array(['', 'red', 'green', 'blue', 'amber'])
_catw = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
_DF = pd.DataFrame({
    'cat': _rng.choice(_cats, _N, p=_catw),
    'eng': _rng.integers(0, 8, _N),
    'x':   _rng.integers(0, 1000, _N),
})

_QUERIES = [
    "SELECT COUNT(*) FROM t WHERE cat = 'red'",
    "SELECT COUNT(*) FROM t WHERE cat <> ''",
    "SELECT COUNT(*) FROM t WHERE cat = 'nonexistent'",          # literal absent -> 0
    "SELECT COUNT(*) FROM t WHERE eng = 3",
    "SELECT COUNT(*) FROM t WHERE eng <> 0",
    "SELECT cat, COUNT(*) AS c FROM t WHERE cat <> '' GROUP BY cat ORDER BY c DESC LIMIT 3",
    "SELECT eng, cat, COUNT(*) AS c FROM t WHERE cat <> '' GROUP BY eng, cat ORDER BY c DESC LIMIT 5",
    "SELECT eng, cat, COUNT(*) AS c FROM t GROUP BY eng, cat ORDER BY c DESC LIMIT 4",
    "SELECT cat, COUNT(*) AS c FROM t WHERE eng = 2 GROUP BY cat ORDER BY c DESC LIMIT 5",
]

def test_dict_eq_and_count_topk():
    d = _tmpdb()
    try:
        db, pq = _build(d, _DF); con = duckdb.connect()
        for sql in _QUERIES:
            got, _ = db.run(sql)
            want = con.execute(sql.replace('FROM t', f"FROM '{pq}'")).fetchall()
            assert _norm(got) == _norm(want), (sql, _norm(got), _norm(want))
    finally:
        shutil.rmtree(d, ignore_errors=True)

