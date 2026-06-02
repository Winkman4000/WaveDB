"""Compaction honors presence (mutable layer, step 1c): tombstoned rows are physically
dropped when segments merge -- this is where deleted space is reclaimed. Proof: delete then
compact -> (a) query results unchanged by the compaction, (b) tombstoned rows physically gone
(new segment has fewer rows, no sidecar), (c) no resurrection, (d) old sidecars cleaned up."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from wdb_db import Database
from wdb_engine import Segment
import wdb_presence

def _tmpdb(): return os.path.join(tempfile.gettempdir(), f'cp_{uuid.uuid4().hex[:8]}')
def _norm(rows):
    return sorted([tuple(round(float(x),4) if isinstance(x,(int,float)) and not isinstance(x,bool) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))

def _build(db, t, batches):
    db.run(f"CREATE TABLE {t} (g VARCHAR, x INT)"); db.set_table_mode(t,'buffered')
    for b in batches:
        db.run(f"INSERT INTO {t} VALUES "+",".join(f"('{g}',{x})" for g,x in b)); db.flush(t)

def test_compaction_drops_tombstoned_rows():
    d=_tmpdb(); db=Database.create(d)
    batches=[[('a',10),('b',20)],[('a',30),('c',40),('b',50)],[('c',60),('a',70)]]
    _build(db,'t',batches)
    db.run("DELETE FROM t WHERE x >= 50")          # tombstone 50,60,70 (logical)
    before=_norm(db.run("SELECT g,x FROM t")[0])
    assert len(before)==4
    res=db.compact('t')
    # query results identical after compaction
    assert _norm(db.run("SELECT g,x FROM t")[0])==before
    # physically reclaimed: new single segment has exactly the 4 live rows
    seg=Segment(os.path.join(d,res['new_segment']))
    assert seg.N==4, seg.N
    # no sidecar on the fresh segment (nothing tombstoned in it)
    assert not os.path.exists(wdb_presence.path_for(os.path.join(d,res['new_segment'])))
    shutil.rmtree(d)

def test_compaction_no_resurrection_then_more_queries():
    d=_tmpdb(); db=Database.create(d)
    _build(db,'t',[[('a',1),('b',2)],[('c',3),('a',4)],[('b',5),('c',6)]])
    db.run("DELETE FROM t WHERE g='a'")            # remove a(1),a(4)
    db.compact('t')
    assert _norm(db.run("SELECT g,x FROM t")[0])==_norm([('b',2),('c',3),('b',5),('c',6)])
    assert db.run("SELECT COUNT(*) FROM t")[0][0][0]==4
    assert _norm(db.run("SELECT g,SUM(x) FROM t GROUP BY g")[0])==_norm([('b',7),('c',9)])
    shutil.rmtree(d)

def test_compaction_cleans_up_old_sidecars():
    d=_tmpdb(); db=Database.create(d)
    _build(db,'t',[[('a',1)],[('b',2)],[('c',3)]])
    db.run("DELETE FROM t WHERE g='b'")
    olds=db.cat.segment_paths('t')
    sidecars_before=[wdb_presence.path_for(p) for p in olds]
    assert any(os.path.exists(s) for s in sidecars_before)
    db.compact('t')
    for s in sidecars_before:
        assert not os.path.exists(s), f"stale sidecar {s} not cleaned up"
    shutil.rmtree(d)

def test_compaction_smaller_after_heavy_delete():
    d=_tmpdb(); db=Database.create(d)
    rows=[[ (f'k{i%5}', i) for i in range(r*200,(r+1)*200) ] for r in range(3)]
    _build(db,'t',rows)
    sizes_before=sum(os.path.getsize(p) for p in db.cat.segment_paths('t'))
    db.run("DELETE FROM t WHERE x < 300")           # delete exactly half the rows (0..299)
    db.compact('t')
    seg=Segment(os.path.join(d, db.cat.get_table('t')['segments'][0]))
    assert seg.N==300, seg.N                         # 600 rows, half deleted
    size_after=os.path.getsize(os.path.join(d, db.cat.get_table('t')['segments'][0]))
    assert size_after < sizes_before, (size_after, sizes_before)
    shutil.rmtree(d)
