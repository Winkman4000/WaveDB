"""ALTER TABLE ... RENAME TO ... end to end. The table's logical identity moves in the catalog;
the canonical buffer file (whose name derives from the table) moves with it; segment files keep
their tracked filenames (so presence/override sidecars are never orphaned); any hot buffer is
folded in by the pre-ALTER flush."""
import sys, os, tempfile, uuid, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from wdb_db import Database

def _db():
    d = os.path.join(tempfile.gettempdir(), f'altrt_{uuid.uuid4().hex[:8]}')
    return Database.create(d), d

def test_rename_table_segment_mode():
    db, d = _db()
    db.run("CREATE TABLE t (id INT, v INT)")
    for i in range(5): db.run(f"INSERT INTO t VALUES ({i}, {i*10})")
    db.run("ALTER TABLE t RENAME TO events")
    rows, _ = db.run("SELECT id, v FROM events WHERE id >= 2")
    assert sorted(rows) == [(2,20),(3,30),(4,40)], rows
    try: db.run("SELECT id FROM t"); assert False, "old table name should be gone"
    except KeyError: pass
    db.run("INSERT INTO events VALUES (5, 50)")           # writes through the moved buffer
    assert db.run("SELECT v FROM events WHERE id = 5")[0] == [(50,)]
    shutil.rmtree(d)

def test_rename_table_buffered_mode_folds_hot():
    db, d = _db()
    db.run("CREATE TABLE t (id INT, v INT)")
    db.set_table_mode('t', 'buffered')
    for i in range(4): db.run(f"INSERT INTO t VALUES ({i}, {i})")
    db.flush('t')                                          # 4 rows -> one cold segment
    for i in range(4, 8): db.run(f"INSERT INTO t VALUES ({i}, {i})")  # 4 rows in hot buffer
    db.run("ALTER TABLE t RENAME TO logs")                 # flush folds hot, then rename
    assert db.run("SELECT COUNT(*) FROM logs")[0] == [(8,)]
    db.run("INSERT INTO logs VALUES (8, 8)")               # buffered mode preserved across rename
    assert db.run("SELECT COUNT(*) FROM logs")[0] == [(9,)]
    assert db.run("SELECT SUM(v) FROM logs")[0] == [(sum(range(9)),)]
    shutil.rmtree(d)

def test_rename_table_preserves_deletes():
    db, d = _db()
    db.run("CREATE TABLE t (id INT, v INT)")
    db.set_table_mode('t', 'buffered')
    for i in range(6): db.run(f"INSERT INTO t VALUES ({i}, {i})")
    db.flush('t')
    db.run("DELETE FROM t WHERE id = 2")                   # presence sidecar on the cold segment
    db.run("ALTER TABLE t RENAME TO t2")
    rows, _ = db.run("SELECT id FROM t2")
    assert sorted(x[0] for x in rows) == [0,1,3,4,5], rows  # delete survived the rename
    shutil.rmtree(d)

def test_rename_table_roundtrip_and_dupe():
    db, d = _db()
    db.run("CREATE TABLE a (id INT)"); db.run("CREATE TABLE b (id INT)")
    db.run("INSERT INTO a VALUES (1)")
    db.run("ALTER TABLE a RENAME TO c"); db.run("ALTER TABLE c RENAME TO a")
    assert db.run("SELECT id FROM a")[0] == [(1,)]
    try: db.run("ALTER TABLE a RENAME TO b"); assert False
    except ValueError: pass
    shutil.rmtree(d)
