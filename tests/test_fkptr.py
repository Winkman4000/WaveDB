"""FK-pointer builder: pointer correctness, gather reconstruction, and the guard rails."""
import sys, os, tempfile, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from wdb_db import Database
from wdb_engine import Segment
import wdb_fkptr

def _db():
    d = tempfile.mkdtemp(); return Database.create(os.path.join(d, 'db'))

def test_create_and_gather():
    db = _db()
    db.run("CREATE TABLE parent (id INT, name VARCHAR)")
    for i, nm in [(0,'ann'),(1,'bob'),(2,'cy'),(3,'dee')]:
        db.run(f"INSERT INTO parent VALUES ({i},'{nm}')")
    db.run("CREATE TABLE child (cid INT, pid INT)")
    pids = [3,0,2,2,1,0]
    for i, pid in enumerate(pids): db.run(f"INSERT INTO child VALUES ({i},{pid})")
    n = db.create_fk_pointer('child', 'pid', 'parent', 'id')
    assert n == len(pids)
    ptr = wdb_fkptr.load(db.cat.segment_paths('child')[0], 'pid')
    assert list(ptr) == pids                                   # parent sorted by id -> position == id
    pseg = Segment(db.cat.segment_paths('parent')[0])
    got = [x.decode() if isinstance(x,(bytes,bytearray)) else x for x in pseg.values('name')[ptr]]
    assert got == ['dee','ann','cy','cy','bob','ann']          # join resolved by gather
    assert db.cat.fk_pointers('child')['pid'] == {'parent':'parent','parent_key':'id'}

def test_ri_violation():
    db = _db()
    db.run("CREATE TABLE p (id INT)")
    for v in (0,1,2): db.run(f"INSERT INTO p VALUES ({v})")
    db.run("CREATE TABLE c (pid INT)")
    for v in (0,5): db.run(f"INSERT INTO c VALUES ({v})")
    try: db.create_fk_pointer('c','pid','p','id'); assert False, "expected RI error"
    except ValueError as e: assert 'referential' in str(e)

def test_nonunique_parent_key():
    db = _db()
    db.run("CREATE TABLE p (id INT)")
    for v in (0,1,1): db.run(f"INSERT INTO p VALUES ({v})")
    db.run("CREATE TABLE c (pid INT)"); db.run("INSERT INTO c VALUES (1)")
    try: db.create_fk_pointer('c','pid','p','id'); assert False, "expected unique error"
    except ValueError as e: assert 'unique' in str(e)

def test_unsorted_parent():
    db = _db()
    db.run("CREATE TABLE p (id INT)")
    for v in (2,0,1): db.run(f"INSERT INTO p VALUES ({v})")
    db.run("CREATE TABLE c (pid INT)"); db.run("INSERT INTO c VALUES (1)")
    try: db.create_fk_pointer('c','pid','p','id'); assert False, "expected sorted error"
    except ValueError as e: assert 'sorted' in str(e)
