"""UPDATE statement (mutable layer, step 1c), literal assignment, both storage modes.
Proof: after UPDATE, every query matches the same query on the data with those values
changed (DuckDB oracle). Buffered: cold segments stay byte-immutable (override sidecars
carry the change). Segment: canonical buffer updated + re-encoded. Column-expression RHS
is deferred to 1d (must raise)."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb
from wdb_db import Database
import wdb_dml, wdb_override

def _tmpdb(): return os.path.join(tempfile.gettempdir(), f'upd_{uuid.uuid4().hex[:8]}')
def _norm(rows):
    return sorted([tuple(round(float(x),4) if isinstance(x,(int,float)) and not isinstance(x,bool) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))
def _lit(v): return "NULL" if v is None else ("'"+v.replace("'","''")+"'" if isinstance(v,str) else str(v))
def _oracle(rows, cols, sql, table='t'):
    con = duckdb.connect()
    vals = ",".join("("+",".join(_lit(v) for v in r)+")" for r in rows)
    return _norm(con.execute(sql.replace(f"FROM {table}", f"FROM (VALUES {vals}) AS {table}({','.join(cols)})")).fetchall())

def test_segment_mode_update_matches_oracle():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (g VARCHAR, x INT)")
    rows=[('a',10),('b',20),('a',30),('c',40)]
    db.run("INSERT INTO t VALUES "+",".join(f"('{g}',{x})" for g,x in rows))
    n=db.run("UPDATE t SET x=99 WHERE g='a'")
    assert n==2, n
    upd=[('a',99) if g=='a' else (g,x) for g,x in rows]
    for sql in ["SELECT g,x FROM t","SELECT g,SUM(x) FROM t GROUP BY g","SELECT COUNT(*) FROM t WHERE x=99"]:
        assert _norm(db.run(sql)[0])==_oracle(upd,['g','x'],sql), sql
    shutil.rmtree(d)

def _buffered(db,t,batches):
    db.run(f"CREATE TABLE {t} (g VARCHAR, x INT)"); db.set_table_mode(t,'buffered')
    for b in batches:
        db.run(f"INSERT INTO {t} VALUES "+",".join(f"('{g}',{x})" for g,x in b)); db.flush(t)

def test_buffered_update_matches_oracle_and_segments_immutable():
    d=_tmpdb(); db=Database.create(d)
    batches=[[('a',10),('b',20)],[('a',30),('c',40)],[('b',50),('a',60)]]
    _buffered(db,'t',batches); allrows=[r for b in batches for r in b]
    paths=db.cat.segment_paths('t'); before={p:open(p,'rb').read() for p in paths}
    n=db.run("UPDATE t SET x=7 WHERE g='a'")
    assert n==3, n
    for p in paths:
        assert open(p,'rb').read()==before[p], f"segment {p} rewritten!"
    assert any(os.path.exists(wdb_override.path_for(p)) for p in paths)
    upd=[('a',7) if g=='a' else (g,x) for g,x in allrows]
    for sql in ["SELECT g,x FROM t","SELECT g,SUM(x) FROM t GROUP BY g","SELECT g,AVG(x) FROM t GROUP BY g"]:
        assert _norm(db.run(sql)[0])==_oracle(upd,['g','x'],sql), (sql,_norm(db.run(sql)[0]))
    shutil.rmtree(d)

def test_update_to_brand_new_value_queryable():
    d=_tmpdb(); db=Database.create(d)
    _buffered(db,'t',[[('a',1)],[('b',2)],[('c',3)]])
    db.run("UPDATE t SET g='ZZZ' WHERE x=2")     # 'ZZZ' never in any dict
    assert _norm(db.run("SELECT g,x FROM t")[0])==_norm([('a',1),('ZZZ',2),('c',3)])
    assert _norm(db.run("SELECT g,COUNT(*) FROM t GROUP BY g")[0])==_norm([('a',1),('ZZZ',1),('c',1)])
    assert _norm(db.run("SELECT x FROM t WHERE g='ZZZ'")[0])==_norm([(2,)])
    shutil.rmtree(d)

def test_update_hits_hot_buffer():
    d=_tmpdb(); db=Database.create(d)
    _buffered(db,'t',[[('a',1)],[('b',2)]])
    db.run("INSERT INTO t VALUES ('a',3)")        # hot, un-flushed
    n=db.run("UPDATE t SET x=100 WHERE g='a'")     # cold a(1) AND hot a(3)
    assert n==2, n
    assert _norm(db.run("SELECT g,x FROM t")[0])==_norm([('a',100),('b',2),('a',100)])
    shutil.rmtree(d)

def test_update_all_no_where_and_multicol():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (g VARCHAR, x INT)")
    db.run("INSERT INTO t VALUES ('a',1),('b',2)")
    n=db.run("UPDATE t SET g='z', x=0")           # all rows, two columns
    assert n==2
    assert _norm(db.run("SELECT g,x FROM t")[0])==_norm([('z',0),('z',0)])
    shutil.rmtree(d)

def test_update_persists_across_reopen():
    d=_tmpdb(); db=Database.create(d)
    _buffered(db,'t',[[('a',1)],[('b',2)],[('c',3)]])
    db.run("UPDATE t SET x=50 WHERE g='b'")
    db2=Database.open(d)
    assert _norm(db2.run("SELECT g,x FROM t")[0])==_norm([('a',1),('b',50),('c',3)])
    shutil.rmtree(d)

def test_update_then_delete_consistent():
    d=_tmpdb(); db=Database.create(d)
    _buffered(db,'t',[[('a',1),('b',2)],[('c',3),('a',4)]])
    db.run("UPDATE t SET x=99 WHERE g='a'")        # a(1)->99, a(4)->99
    db.run("DELETE FROM t WHERE x=99")             # delete the two updated rows
    assert _norm(db.run("SELECT g,x FROM t")[0])==_norm([('b',2),('c',3)])
    shutil.rmtree(d)

def test_column_expression_rhs_deferred():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (a INT, b INT)")
    db.run("INSERT INTO t VALUES (1,2)")
    for sql in ["UPDATE t SET a = b", "UPDATE t SET a = a + 1"]:
        try:
            db.run(sql); assert False, f"{sql} should defer to 1d"
        except NotImplementedError:
            pass
    shutil.rmtree(d)
