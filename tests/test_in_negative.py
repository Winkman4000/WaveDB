"""IN (-1, 6): negative members parse as Neg(Literal), not bare Literal. The general-scan
predicate evaluator must unwrap Neg for numeric IN lists. Verified vs DuckDB."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd, numpy as np, duckdb
from wdb_db import Database
import wdb_encode

def _norm(rows):
    return sorted([tuple(int(x) if isinstance(x, (np.integer,)) else x for x in r) for r in rows], key=repr)

def test_in_with_negative_literals():
    d = os.path.join(tempfile.gettempdir(), f'wneg_{uuid.uuid4().hex[:8]}')
    try:
        db = Database.create(d); db.run("CREATE TABLE t (ts INT, x INT)")
        df = pd.DataFrame({'ts': [-1, 0, 6, 6, -1, 3, 9], 'x': list(range(7))})
        pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb')
        allpq = os.path.join(d, '_all.parquet'); df.to_parquet(allpq, index=False)
        con = duckdb.connect()
        for sql in ["SELECT x FROM t WHERE ts IN (-1, 6)",
                    "SELECT COUNT(*) FROM t WHERE ts IN (-1, 6, 9)",
                    "SELECT x FROM t WHERE ts IN (3)"]:
            got, _ = db.run(sql)
            want = con.execute(sql.replace('FROM t', f"FROM '{allpq}'")).fetchall()
            assert _norm(got) == _norm(want), (sql, _norm(got), _norm(want))
    finally:
        shutil.rmtree(d, ignore_errors=True)
