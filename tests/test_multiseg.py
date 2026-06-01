"""Multi-segment reads: a table's answer is the union across all its cold segments.
Part 1 builds segments manually (disjoint row subsets) to test the union in isolation,
before append-flush exists to produce them naturally."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd, numpy as np, duckdb
from wdb_db import Database
import wdb_encode

def _tmpdb(): return os.path.join(tempfile.gettempdir(), f'wmseg_{uuid.uuid4().hex[:8]}')
def _norm(rows):
    return sorted([tuple(round(float(x),4) if isinstance(x,(int,float,np.floating,np.integer)) and not isinstance(x,bool) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))

def _build_multiseg(dbdir, tname, schema_sql, frames):
    """Create a table, then encode each DataFrame into its own segment and register it."""
    db = Database.create(dbdir)
    db.run(schema_sql)
    allrows = pd.concat(frames, ignore_index=True)
    for i, fr in enumerate(frames):
        pq = os.path.join(dbdir, f'{tname}_src{i}.parquet'); fr.to_parquet(pq, index=False)
        seg_file = f'{tname}_{i}.wdb'
        wdb_encode.encode(pq, os.path.join(dbdir, seg_file))
        db.cat.add_segment(tname, seg_file)
    # ground-truth parquet of the union
    allpq = os.path.join(dbdir, '_all.parquet'); allrows.to_parquet(allpq, index=False)
    return db, allpq

def _frames():
    return [
        pd.DataFrame({'g':['a','b'],     'x':[10,20]}),
        pd.DataFrame({'g':['a','c','b'], 'x':[30,40,50]}),
        pd.DataFrame({'g':['c','a'],     'x':[60,70]}),
    ]

def test_multiseg_row_projection():
    d=_tmpdb(); db,allpq=_build_multiseg(d,'t',"CREATE TABLE t (g VARCHAR, x INT)",_frames())
    con=duckdb.connect()
    for sql in ["SELECT g,x FROM t", "SELECT x FROM t WHERE x > 35", "SELECT g FROM t WHERE g = 'a'"]:
        got,_=db.run(sql); want=con.execute(sql.replace('t ',f"'{allpq}' ",1) if False else sql.replace('FROM t',f"FROM '{allpq}'")).fetchall()
        assert _norm(got)==_norm(want), (sql,_norm(got),_norm(want))
    shutil.rmtree(d)

def test_multiseg_aggregates():
    d=_tmpdb(); db,allpq=_build_multiseg(d,'t',"CREATE TABLE t (g VARCHAR, x INT)",_frames())
    con=duckdb.connect()
    for sql in ["SELECT COUNT(*) FROM t","SELECT SUM(x) FROM t","SELECT AVG(x) FROM t",
                "SELECT MIN(x), MAX(x) FROM t"]:
        got,_=db.run(sql); want=con.execute(sql.replace('FROM t',f"FROM '{allpq}'")).fetchall()
        assert _norm(got)==_norm(want), (sql,_norm(got),_norm(want))
    shutil.rmtree(d)

def test_multiseg_groupby():
    d=_tmpdb(); db,allpq=_build_multiseg(d,'t',"CREATE TABLE t (g VARCHAR, x INT)",_frames())
    con=duckdb.connect()
    for sql in ["SELECT g, COUNT(*) FROM t GROUP BY g",
                "SELECT g, SUM(x) FROM t GROUP BY g",
                "SELECT g, AVG(x) FROM t GROUP BY g",
                "SELECT g, COUNT(*), SUM(x), AVG(x), MIN(x), MAX(x) FROM t GROUP BY g"]:
        got,_=db.run(sql); want=con.execute(sql.replace('FROM t',f"FROM '{allpq}'")).fetchall()
        assert _norm(got)==_norm(want), (sql,_norm(got),_norm(want))
    shutil.rmtree(d)

def test_multiseg_where_groupby_having_order_limit():
    d=_tmpdb(); db,allpq=_build_multiseg(d,'t',"CREATE TABLE t (g VARCHAR, x INT)",_frames())
    con=duckdb.connect()
    for sql in ["SELECT g, SUM(x) FROM t WHERE x > 15 GROUP BY g",
                "SELECT g, COUNT(*) FROM t GROUP BY g HAVING COUNT(*) >= 3",
                "SELECT g, SUM(x) FROM t GROUP BY g ORDER BY g LIMIT 2"]:
        got,_=db.run(sql); want=con.execute(sql.replace('FROM t',f"FROM '{allpq}'")).fetchall()
        assert _norm(got)==_norm(want), (sql,_norm(got),_norm(want))
    shutil.rmtree(d)

def test_multiseg_plus_hot_buffer():
    # N cold segments AND a hot buffer all merged together
    d=_tmpdb(); db,allpq=_build_multiseg(d,'t',"CREATE TABLE t (g VARCHAR, x INT)",_frames())
    db.set_table_mode('t','buffered')
    db.run("INSERT INTO t VALUES ('b',100),('a',200)")   # hot rows on top of 3 cold segments
    con=duckdb.connect()
    allrows = pd.concat(_frames()+[pd.DataFrame({'g':['b','a'],'x':[100,200]})], ignore_index=True)
    allpq2 = os.path.join(d,'_all2.parquet'); allrows.to_parquet(allpq2, index=False)
    for sql in ["SELECT g, SUM(x) FROM t GROUP BY g", "SELECT COUNT(*) FROM t",
                "SELECT g, AVG(x) FROM t GROUP BY g"]:
        got,_=db.run(sql); want=con.execute(sql.replace('FROM t',f"FROM '{allpq2}'")).fetchall()
        assert _norm(got)==_norm(want), (sql,_norm(got),_norm(want))
    shutil.rmtree(d)
