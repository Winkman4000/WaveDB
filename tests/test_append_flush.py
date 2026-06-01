"""Append-style flush (step 3c, part 2): each flush encodes ONLY the hot rows into a
NEW segment and appends it - existing segments untouched (O(hot)). This is what
produces multi-segment tables naturally. End-to-end answers must match DuckDB."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd, numpy as np, duckdb
from wdb_db import Database

def _tmpdb(): return os.path.join(tempfile.gettempdir(), f'waf_{uuid.uuid4().hex[:8]}')
def _norm(rows):
    return sorted([tuple(round(float(x),4) if isinstance(x,(int,float,np.floating,np.integer)) and not isinstance(x,bool) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))

def test_repeated_flush_creates_multiple_segments():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (x INT)"); db.set_table_mode('t','buffered')
    db.run("INSERT INTO t VALUES (1)"); db.flush('t')
    db.run("INSERT INTO t VALUES (2)"); db.flush('t')
    db.run("INSERT INTO t VALUES (3)"); db.flush('t')
    segs = db.cat.get_table('t')['segments']
    assert segs == ['t_0.wdb','t_1.wdb','t_2.wdb'], segs
    rows,_=db.run("SELECT x FROM t")
    assert _norm(rows)==_norm([(1,),(2,),(3,)]), _norm(rows)
    shutil.rmtree(d)

def test_flush_is_append_only_existing_untouched():
    # after first flush, t_0 exists; second flush must NOT modify t_0's mtime/content
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (x INT)"); db.set_table_mode('t','buffered')
    db.run("INSERT INTO t VALUES (1),(2)"); db.flush('t')
    seg0=os.path.join(d,'t_0.wdb'); m0=os.path.getmtime(seg0); sz0=os.path.getsize(seg0)
    import time; time.sleep(0.05)
    db.run("INSERT INTO t VALUES (3),(4),(5)"); db.flush('t')
    assert os.path.getmtime(seg0)==m0 and os.path.getsize(seg0)==sz0, "t_0 must be untouched by later flush"
    assert os.path.exists(os.path.join(d,'t_1.wdb'))
    shutil.rmtree(d)

def test_end_to_end_multiflush_vs_duckdb():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE s (g VARCHAR, x INT)"); db.set_table_mode('s','buffered')
    batches = [[('a',10),('b',20)], [('a',30),('c',40),('b',50)], [('c',60),('a',70)]]
    allrows=[]
    for b in batches:
        vals=','.join(f"('{g}',{x})" for g,x in b)
        db.run(f"INSERT INTO s VALUES {vals}"); db.flush('s'); allrows+=b
    # plus un-flushed hot rows on top
    db.run("INSERT INTO s VALUES ('b',5),('a',1)"); allrows+=[('b',5),('a',1)]
    con=duckdb.connect()
    vals=','.join(f"('{g}',{x})" for g,x in allrows)
    for sql in ["SELECT g,COUNT(*),SUM(x),AVG(x),MIN(x),MAX(x) FROM s GROUP BY g",
                "SELECT SUM(x) FROM s","SELECT g,x FROM s WHERE x>25",
                "SELECT g,SUM(x) FROM s GROUP BY g HAVING SUM(x)>60 ORDER BY g"]:
        got,_=db.run(sql)
        want=con.execute(sql.replace('FROM s',f"FROM (VALUES {vals}) AS s(g,x)")).fetchall()
        assert _norm(got)==_norm(want), (sql,_norm(got),_norm(want))
    shutil.rmtree(d)

def test_flush_empty_hot_is_noop():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (x INT)"); db.set_table_mode('t','buffered')
    assert db.flush('t')==0                       # nothing buffered yet
    db.run("INSERT INTO t VALUES (1)"); db.flush('t')
    assert db.flush('t')==0                        # already flushed, hot gone
    assert db.cat.get_table('t')['segments']==['t_0.wdb']
    shutil.rmtree(d)

def test_multiflush_persists_across_reopen():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (x INT)"); db.set_table_mode('t','buffered')
    db.run("INSERT INTO t VALUES (1),(2)"); db.flush('t')
    db.run("INSERT INTO t VALUES (3)"); db.flush('t')
    db2=Database.open(d)
    rows,_=db2.run("SELECT x FROM t")
    assert _norm(rows)==_norm([(1,),(2,),(3,)])
    assert db2.cat.get_table('t')['segments']==['t_0.wdb','t_1.wdb']
    shutil.rmtree(d)
