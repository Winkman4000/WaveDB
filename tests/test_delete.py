"""DELETE statement (mutable layer, step 1b), both storage modes.
Proof obligation: after a DELETE, every query returns exactly what the same query returns
against the data with those rows physically removed (DuckDB oracle).
Buffered: cold segments stay byte-immutable (only sidecars appear). Segment: buffer re-encoded."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb
from wdb_db import Database
import wdb_dml, wdb_presence

def _tmpdb(): return os.path.join(tempfile.gettempdir(), f'del_{uuid.uuid4().hex[:8]}')
def _norm(rows):
    out=[]
    for r in rows:
        out.append(tuple(round(float(x),4) if isinstance(x,(int,float)) and not isinstance(x,bool) else x for x in r))
    return sorted(out, key=lambda t: tuple(str(x) for x in t))
def _oracle(remaining_rows, cols, sql, table):
    if not remaining_rows:
        # build an empty typed table by selecting from an empty VALUES isn't trivial; use a filter
        con=duckdb.connect()
        vals=",".join("("+",".join("NULL" for _ in cols)+")")
        q=sql.replace(f"FROM {table}", f"FROM (SELECT {','.join(cols)} FROM (VALUES {vals}) AS _t({','.join(cols)}) WHERE FALSE) AS {table}")
        return _norm(con.execute(q).fetchall())
    con=duckdb.connect()
    vals=",".join("("+",".join(_lit(v) for v in row)+")" for row in remaining_rows)
    q=sql.replace(f"FROM {table}", f"FROM (VALUES {vals}) AS {table}({','.join(cols)})")
    return _norm(con.execute(q).fetchall())
def _lit(v):
    if v is None: return "NULL"
    if isinstance(v,str): return "'"+v.replace("'","''")+"'"
    return str(v)

def test_segment_mode_delete_matches_oracle():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (g VARCHAR, x INT)")
    rows=[('a',10),('b',20),('a',30),('c',40),('b',50),('a',60)]
    db.run("INSERT INTO t VALUES "+",".join(f"('{g}',{x})" for g,x in rows))
    n=db.run("DELETE FROM t WHERE x > 35")
    assert n==3, n
    remaining=[r for r in rows if not r[1]>35]
    for sql in ["SELECT g,x FROM t","SELECT g,SUM(x) FROM t GROUP BY g","SELECT COUNT(*) FROM t"]:
        assert _norm(db.run(sql)[0])==_oracle(remaining,['g','x'],sql,'t'), sql
    assert len(db.cat.get_table('t')['segments'])==1
    shutil.rmtree(d)

def test_segment_mode_no_resurrection_on_reinsert():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (x INT)")
    db.run("INSERT INTO t VALUES (1),(2),(3)")
    db.run("DELETE FROM t WHERE x=2")
    db.run("INSERT INTO t VALUES (4)")           # must NOT bring back x=2
    assert _norm(db.run("SELECT x FROM t")[0])==_norm([(1,),(3,),(4,)])
    shutil.rmtree(d)

def _build_buffered(db, t, batches):
    db.run(f"CREATE TABLE {t} (g VARCHAR, x INT)"); db.set_table_mode(t,'buffered')
    for b in batches:
        db.run(f"INSERT INTO {t} VALUES "+",".join(f"('{g}',{x})" for g,x in b)); db.flush(t)

def test_buffered_delete_matches_oracle_and_keeps_segments_immutable():
    d=_tmpdb(); db=Database.create(d)
    batches=[[('a',10),('b',20)],[('a',30),('c',40),('b',50)],[('c',60),('a',70)]]
    _build_buffered(db,'t',batches)
    allrows=[r for b in batches for r in b]
    # capture cold segment bytes before delete (must stay identical)
    paths=db.cat.segment_paths('t')
    before_bytes={p:open(p,'rb').read() for p in paths}
    n=db.run("DELETE FROM t WHERE x >= 50")
    assert n==3, n   # 50,60,70
    # cold .wdb bytes unchanged (immutable); sidecars appear only where rows were deleted
    for p in paths:
        assert open(p,'rb').read()==before_bytes[p], f"segment {p} was rewritten!"
    assert any(os.path.exists(wdb_presence.path_for(p)) for p in paths), "some sidecar should exist"
    remaining=[r for r in allrows if not r[1]>=50]
    for sql in ["SELECT g,x FROM t","SELECT g,SUM(x) FROM t GROUP BY g","SELECT COUNT(*) FROM t",
                "SELECT g,AVG(x) FROM t GROUP BY g"]:
        assert _norm(db.run(sql)[0])==_oracle(remaining,['g','x'],sql,'t'), (sql,_norm(db.run(sql)[0]))
    shutil.rmtree(d)

def test_buffered_delete_hits_hot_buffer_too():
    d=_tmpdb(); db=Database.create(d)
    _build_buffered(db,'t',[[('a',1)],[('b',2)]])
    db.run("INSERT INTO t VALUES ('a',3)")        # hot, un-flushed
    n=db.run("DELETE FROM t WHERE g='a'")          # must delete cold a(1) AND hot a(3)
    assert n==2, n
    assert _norm(db.run("SELECT g,x FROM t")[0])==_norm([('b',2)])
    shutil.rmtree(d)

def test_delete_all_no_where():
    for mode in ('segment','buffered'):
        d=_tmpdb(); db=Database.create(d)
        if mode=='buffered': _build_buffered(db,'t',[[('a',1)],[('b',2)]])
        else:
            db.run("CREATE TABLE t (g VARCHAR, x INT)")
            db.run("INSERT INTO t VALUES ('a',1),('b',2)")
        n=db.run("DELETE FROM t")
        assert n==2, (mode,n)
        assert db.run("SELECT COUNT(*) FROM t")[0][0][0]==0, mode
        shutil.rmtree(d)

def test_delete_matching_nothing_is_noop():
    d=_tmpdb(); db=Database.create(d)
    _build_buffered(db,'t',[[('a',1)],[('b',2)]])
    n=db.run("DELETE FROM t WHERE x=999")
    assert n==0
    assert _norm(db.run("SELECT g,x FROM t")[0])==_norm([('a',1),('b',2)])
    shutil.rmtree(d)

def test_delete_persists_across_reopen():
    d=_tmpdb(); db=Database.create(d)
    _build_buffered(db,'t',[[('a',1)],[('b',2)],[('c',3)]])
    db.run("DELETE FROM t WHERE g='b'")
    db2=Database.open(d)
    assert _norm(db2.run("SELECT g,x FROM t")[0])==_norm([('a',1),('c',3)])
    shutil.rmtree(d)

def test_delete_does_not_remove_null_rows():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (g VARCHAR, x INT)")
    db.run("INSERT INTO t VALUES ('a',10),('b',NULL),('c',40)")
    n=db.run("DELETE FROM t WHERE x > 5")    # NULL row must survive (pred is NULL, not TRUE)
    assert n==2, n
    assert _norm(db.run("SELECT g FROM t")[0])==_norm([('b',)])
    shutil.rmtree(d)
