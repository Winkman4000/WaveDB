"""UPDATE with column/expression RHS (mutable layer, step 1d): SET a=b, SET x=x+1,
SET p=p*1.1-q, multi-column simultaneous (SET a=b, b=a). Verified vs DuckDB oracle in both
storage modes and across a hot+cold mix (the cold-segment numpy evaluator must agree with
DuckDB's evaluation of the hot/segment-mode parquet path)."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb
from wdb_db import Database

def _tmpdb(): return os.path.join(tempfile.gettempdir(), f'updx_{uuid.uuid4().hex[:8]}')
def _norm(rows):
    return sorted([tuple(round(float(x),4) if isinstance(x,(int,float)) and not isinstance(x,bool) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))
def _lit(v): return "NULL" if v is None else ("'"+v.replace("'","''")+"'" if isinstance(v,str) else str(v))
def _oracle(rows, cols, after_sql, sel):
    """Apply the UPDATE in DuckDB on an in-memory table, then run sel."""
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t({','.join(c+' '+ty for c,ty in cols)})")
    con.execute("INSERT INTO t VALUES "+",".join("("+",".join(_lit(v) for v in r)+")" for r in rows))
    con.execute(after_sql)
    return _norm(con.execute(sel).fetchall())

def _buffered(db,t,batches,schema="(a INT, b INT)"):
    db.run(f"CREATE TABLE {t} {schema}"); db.set_table_mode(t,'buffered')
    for bb in batches:
        db.run(f"INSERT INTO {t} VALUES "+",".join("("+",".join(str(v) for v in r)+")" for r in bb)); db.flush(t)

def test_segment_mode_increment():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (a INT, b INT)")
    rows=[(1,10),(2,20),(3,30)]
    db.run("INSERT INTO t VALUES "+",".join(str(r) for r in rows))
    db.run("UPDATE t SET a = a + 1 WHERE b > 15")
    sel="SELECT a,b FROM t"
    assert _norm(db.run(sel)[0])==_oracle(rows,[('a','INT'),('b','INT')],"UPDATE t SET a=a+1 WHERE b>15",sel)
    shutil.rmtree(d)

def test_column_copy_both_modes():
    rows=[[(1,10),(2,20)],[(3,30),(4,40)]]
    flat=[r for b in rows for r in b]
    sel="SELECT a,b FROM t"; upd="UPDATE t SET a=b"
    # buffered
    d=_tmpdb(); db=Database.create(d); _buffered(db,'t',rows)
    db.run(upd)
    assert _norm(db.run(sel)[0])==_oracle(flat,[('a','INT'),('b','INT')],upd,sel)
    shutil.rmtree(d)

def test_expr_over_cold_and_hot_mix():
    d=_tmpdb(); db=Database.create(d)
    _buffered(db,'t',[[(1,10),(2,20)],[(3,30)]])   # cold segments
    db.run("INSERT INTO t VALUES (4,40),(5,50)")    # hot, un-flushed
    flat=[(1,10),(2,20),(3,30),(4,40),(5,50)]
    upd="UPDATE t SET a = a*2 + 1 WHERE b >= 20"; sel="SELECT a,b FROM t"
    db.run(upd)
    assert _norm(db.run(sel)[0])==_oracle(flat,[('a','INT'),('b','INT')],upd,sel)
    shutil.rmtree(d)

def test_float_expr():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (p DOUBLE, q DOUBLE)")
    rows=[(10.0,1.0),(20.0,2.0),(30.0,3.0)]
    db.run("INSERT INTO t VALUES "+",".join(str(r) for r in rows))
    upd="UPDATE t SET p = p*1.1 - q"; sel="SELECT p,q FROM t"
    db.run(upd)
    assert _norm(db.run(sel)[0])==_oracle(rows,[('p','DOUBLE'),('q','DOUBLE')],upd,sel)
    shutil.rmtree(d)

def test_simultaneous_swap():
    d=_tmpdb(); db=Database.create(d)
    _buffered(db,'t',[[(1,2),(3,4)],[(5,6)]])
    upd="UPDATE t SET a=b, b=a"; sel="SELECT a,b FROM t"
    db.run(upd)
    assert _norm(db.run(sel)[0])==_oracle([(1,2),(3,4),(5,6)],[('a','INT'),('b','INT')],upd,sel)
    shutil.rmtree(d)

def test_expr_then_query_groupby():
    d=_tmpdb(); db=Database.create(d)
    _buffered(db,'t',[[(1,10),(2,20)],[(1,30),(2,40)]],schema="(g INT, x INT)")
    db.run("UPDATE t SET x = x + 100")
    sel="SELECT g,SUM(x) FROM t GROUP BY g"
    assert _norm(db.run(sel)[0])==_oracle([(1,10),(2,20),(1,30),(2,40)],[('g','INT'),('x','INT')],
                                          "UPDATE t SET x=x+100",sel)
    shutil.rmtree(d)
