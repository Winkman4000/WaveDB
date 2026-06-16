"""LIMIT n OFFSET k must return the ordered rows [k : k+n]. The fast structure reads
pre-truncate to LIMIT, so the controller routes any OFFSET query to the general scan
(single segment) and wdb_merge applies LIMIT/OFFSET post-merge (multi segment). Verified
against DuckDB. Sort keys are unique so the offset window is unambiguous."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd, numpy as np, duckdb
from wdb_db import Database
import wdb_encode

def _tmpdb(): return os.path.join(tempfile.gettempdir(), f'woff_{uuid.uuid4().hex[:8]}')
def _norm(rows):
    return sorted([tuple(round(float(x), 4) if isinstance(x, (int, float, np.floating, np.integer))
                         and not isinstance(x, bool) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))

def _build(dbdir, frames):
    db = Database.create(dbdir); db.run("CREATE TABLE t (g VARCHAR, x INT)")
    allrows = pd.concat(frames, ignore_index=True)
    for i, fr in enumerate(frames):
        pq = os.path.join(dbdir, f'src{i}.parquet'); fr.to_parquet(pq, index=False)
        wdb_encode.encode(pq, os.path.join(dbdir, f't_{i}.wdb'))
        db.cat.add_segment('t', f't_{i}.wdb')
    allpq = os.path.join(dbdir, '_all.parquet'); allrows.to_parquet(allpq, index=False)
    return db, allpq

# 15 rows: unique x (total order), distinct group counts (a=5,b=4,c=3,d=2,e=1)
_G = ['a']*5 + ['b']*4 + ['c']*3 + ['d']*2 + ['e']*1
_X = list(range(15))
def _one(): return [pd.DataFrame({'g': _G, 'x': _X})]
def _three():
    df = pd.DataFrame({'g': _G, 'x': _X})
    return [df.iloc[0:6], df.iloc[6:11], df.iloc[11:15]]

_QUERIES = [
    "SELECT x FROM t ORDER BY x LIMIT 3 OFFSET 5",
    "SELECT x FROM t ORDER BY x DESC LIMIT 2 OFFSET 3",
    "SELECT g, x FROM t ORDER BY x LIMIT 4 OFFSET 8",
    "SELECT g, COUNT(*) AS c FROM t GROUP BY g ORDER BY c DESC LIMIT 2 OFFSET 1",
    "SELECT x FROM t ORDER BY x LIMIT 100 OFFSET 12",   # window runs past the end
    "SELECT x FROM t ORDER BY x OFFSET 13",             # OFFSET with no LIMIT
    "SELECT x FROM t ORDER BY x LIMIT 3 OFFSET 0",      # OFFSET 0 == plain LIMIT
]

def _check(frames):
    d = _tmpdb()
    try:
        db, allpq = _build(d, frames); con = duckdb.connect()
        for sql in _QUERIES:
            got, _ = db.run(sql)
            want = con.execute(sql.replace('FROM t', f"FROM '{allpq}'")).fetchall()
            assert _norm(got) == _norm(want), (sql, _norm(got), _norm(want))
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_offset_single_segment():
    _check(_one())

def test_offset_multi_segment():
    _check(_three())
