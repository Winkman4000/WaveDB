"""Compaction (stage 2a): merge cold segments into one, verify FD labels on the union.
Query results must be identical before and after; true FDs survive, casualties dropped."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb
from wdb_db import Database

def _tmpdb(): return os.path.join(tempfile.gettempdir(), f'wcmp_{uuid.uuid4().hex[:8]}')
def _norm(rows):
    return sorted([tuple(round(float(x),4) if isinstance(x,(int,float)) and not isinstance(x,bool) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))

def _build_multiseg(db, t, batches):
    db.run(f"CREATE TABLE {t} (g VARCHAR, x INT)"); db.set_table_mode(t,'buffered')
    for b in batches:
        vals=",".join(f"('{g}',{x})" for g,x in b)
        db.run(f"INSERT INTO {t} VALUES {vals}"); db.flush(t)

def test_compaction_preserves_query_results():
    d=_tmpdb(); db=Database.create(d)
    batches=[[('a',10),('b',20)],[('a',30),('c',40),('b',50)],[('c',60),('a',70)]]
    _build_multiseg(db,'t',batches)
    queries=["SELECT g,x FROM t","SELECT g,SUM(x) FROM t GROUP BY g","SELECT COUNT(*) FROM t",
             "SELECT g,AVG(x) FROM t GROUP BY g","SELECT x FROM t WHERE x>35"]
    before={q: _norm(db.run(q)[0]) for q in queries}
    assert len(db.cat.get_table('t')['segments'])==3
    res=db.compact('t')
    assert res['new_segment'] is not None and len(db.cat.get_table('t')['segments'])==1, db.cat.get_table('t')['segments']
    for q in queries:
        assert _norm(db.run(q)[0])==before[q], (q, _norm(db.run(q)[0]), before[q])
    shutil.rmtree(d)

def test_compaction_deletes_old_segment_files():
    d=_tmpdb(); db=Database.create(d)
    _build_multiseg(db,'t',[[('a',1)],[('b',2)],[('c',3)]])
    olds=list(db.cat.get_table('t')['segments'])
    db.compact('t')
    for s in olds:
        assert not os.path.exists(os.path.join(d,s)), f"{s} should be deleted"
    shutil.rmtree(d)

def test_verified_label_survives():
    # part->brand is a true FD across all segments -> must survive compaction
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (part INT, brand INT)"); db.set_table_mode('t','buffered')
    for _ in range(2):
        rows=",".join(f"({p},{p%4})" for p in list(range(40))*3)   # part repeats, brand=part%4
        db.run(f"INSERT INTO t VALUES {rows}"); db.flush('t')
    res=db.compact('t')
    kept={(l['det'],l['dep']) for l in db.cat.segment_labels('t', res['new_segment'])}
    assert ('part','brand') in kept, f"true FD must survive, got {kept}"
    shutil.rmtree(d)

def test_accidental_label_dropped_on_merge():
    # seg0: g='a' always maps to 1 (FD holds in seg0). seg1: g='a' maps to 2.
    # union: a->{1,2} breaks the FD -> label must be DROPPED by verification.
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (g VARCHAR, v INT)"); db.set_table_mode('t','buffered')
    db.run("INSERT INTO t VALUES ('a',1),('a',1),('a',1),('b',9),('b',9)"); db.flush('t')  # g->v holds
    s0=db.cat.get_table('t')['segments'][0]
    assert ('g','v') in {(l['det'],l['dep']) for l in db.cat.segment_labels('t',s0)}, "g->v should be labeled in seg0"
    db.run("INSERT INTO t VALUES ('a',2),('a',2),('a',2),('c',7),('c',7)"); db.flush('t')  # a now ->2
    res=db.compact('t')
    kept={(l['det'],l['dep']) for l in db.cat.segment_labels('t',res['new_segment'])}
    assert ('g','v') not in kept, f"g->v breaks on union and must be dropped, got {kept}"
    assert res['labels_kept'] < res['labels_in'] or res['labels_in']==0
    shutil.rmtree(d)

def test_compact_leaves_hot_buffer_untouched():
    d=_tmpdb(); db=Database.create(d)
    _build_multiseg(db,'t',[[('a',1)],[('b',2)]])
    db.run("INSERT INTO t VALUES ('c',3)")   # hot, un-flushed
    import wdb_dml
    assert os.path.exists(wdb_dml.hot_path(db.cat,'t'))
    db.compact('t')
    assert os.path.exists(wdb_dml.hot_path(db.cat,'t')), "hot buffer must survive compaction"
    rows,_=db.run("SELECT g,x FROM t")
    assert _norm(rows)==_norm([('a',1),('b',2),('c',3)]), _norm(rows)
    shutil.rmtree(d)

def test_compaction_persists_across_reopen():
    d=_tmpdb(); db=Database.create(d)
    _build_multiseg(db,'t',[[('a',1)],[('b',2)],[('c',3)]])
    db.compact('t')
    db2=Database.open(d)
    rows,_=db2.run("SELECT g,x FROM t")
    assert _norm(rows)==_norm([('a',1),('b',2),('c',3)])
    assert len(db2.cat.get_table('t')['segments'])==1
    shutil.rmtree(d)

def test_compact_noop_on_single_segment():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (x INT)")
    db.run("INSERT INTO t VALUES (1),(2)")    # default mode -> single segment
    res=db.compact('t')
    assert res['new_segment'] is None and res['merged']==[]
    shutil.rmtree(d)

def test_compaction_vs_duckdb_aggregates():
    d=_tmpdb(); db=Database.create(d)
    batches=[[('n',10),('s',20)],[('n',30),('s',40),('e',5)],[('n',1),('e',2)]]
    _build_multiseg(db,'sales',batches)
    db.compact('sales')
    allrows=[r for b in batches for r in b]
    vals=",".join(f"('{g}',{x})" for g,x in allrows)
    con=duckdb.connect()
    for sql in ["SELECT g,COUNT(*),SUM(x),AVG(x) FROM sales GROUP BY g",
                "SELECT SUM(x) FROM sales"]:
        got,_=db.run(sql.replace('sales','sales'))
        want=con.execute(sql.replace('FROM sales',f"FROM (VALUES {vals}) AS sales(g,x)")).fetchall()
        assert _norm(got)==_norm(want), (sql,_norm(got),_norm(want))
    shutil.rmtree(d)
