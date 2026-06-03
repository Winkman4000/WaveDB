"""ALTER TABLE ... RENAME COLUMN end to end. Logical name lives in the catalog; storage keeps the
original PHYSICAL name everywhere (segments, buffer, hot parquet). Reads translate logical->physical
via col_map (our engine) and via _to_physical (DuckDB-side parquet queries). Inserts write physical
names so the canonical/hot buffer stays aligned with what's already on disk."""
import sys, os, tempfile, uuid, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from wdb_db import Database

def _db():
    d = os.path.join(tempfile.gettempdir(), f'altrc_{uuid.uuid4().hex[:8]}')
    return Database.create(d), d

def test_rename_column_segment_mode():
    db, d = _db()
    db.run("CREATE TABLE t (id INT, name VARCHAR, amt INT)")
    for i in range(5): db.run(f"INSERT INTO t VALUES ({i}, 'n{i}', {i*10})")
    db.run("ALTER TABLE t RENAME COLUMN name TO label")
    assert db.run("SELECT id, label FROM t WHERE id = 2")[0] == [(2,'n2')]
    assert db.run("SELECT id FROM t WHERE label = 'n3'")[0] == [(3,)]
    try: db.run("SELECT name FROM t"); assert False, "old column name should error"
    except NotImplementedError: pass
    db.run("INSERT INTO t VALUES (5, 'n5', 50)")                 # positional, must align to buffer
    assert db.run("SELECT label FROM t WHERE id = 5")[0] == [('n5',)]
    db.run("INSERT INTO t (id, label, amt) VALUES (6, 'n6', 60)")# explicit new name
    assert db.run("SELECT label FROM t WHERE id = 6")[0] == [('n6',)]
    shutil.rmtree(d)

def test_rename_column_buffered_cold_and_hot():
    db, d = _db()
    db.run("CREATE TABLE t (id INT, v INT)")
    db.set_table_mode('t','buffered')
    for i in range(4): db.run(f"INSERT INTO t VALUES ({i}, {i})")
    db.flush('t')                                                # cold segment, physical 'v'
    db.run("ALTER TABLE t RENAME COLUMN v TO val")
    for i in range(4,8): db.run(f"INSERT INTO t VALUES ({i}, {i})")  # hot, stored as physical 'v'
    assert sorted(db.run("SELECT id, val FROM t WHERE val >= 3")[0]) == [(3,3),(4,4),(5,5),(6,6),(7,7)]
    assert db.run("SELECT SUM(val) FROM t")[0] == [(sum(range(8)),)]   # spans cold + hot tiers
    shutil.rmtree(d)

def test_rename_column_then_update_and_delete_cold():
    db, d = _db()
    db.run("CREATE TABLE t (id INT, score INT)")
    db.set_table_mode('t','buffered')
    for i in range(6): db.run(f"INSERT INTO t VALUES ({i}, {i*100})")
    db.flush('t')
    db.run("ALTER TABLE t RENAME COLUMN score TO pts")
    db.run("UPDATE t SET pts = 999 WHERE id = 2")                # override keyed by physical 'score'
    assert db.run("SELECT pts FROM t WHERE id = 2")[0] == [(999,)]
    db.run("DELETE FROM t WHERE pts = 999")                      # predicate on new name, cold segment
    assert db.run("SELECT id FROM t WHERE id = 2")[0] == []
    assert db.run("SELECT SUM(pts) FROM t")[0] == [(0+100+300+400+500,)]
    shutil.rmtree(d)

def test_rename_column_arithmetic_update_mixes_names():
    db, d = _db()
    db.run("CREATE TABLE t (id INT, price INT, qty INT)")
    db.set_table_mode('t','buffered')
    for i in range(1,5): db.run(f"INSERT INTO t VALUES ({i}, {i*10}, {i})")
    db.flush('t')
    db.run("ALTER TABLE t RENAME COLUMN price TO cost")
    db.run("UPDATE t SET cost = cost * qty WHERE id >= 2")       # renamed * non-renamed, cold path
    assert db.run("SELECT id, cost FROM t ORDER BY id")[0] == [(1,10),(2,40),(3,90),(4,160)]
    shutil.rmtree(d)

def test_rename_column_chain_vs_duckdb():
    import duckdb
    db, d = _db()
    db.run("CREATE TABLE t (id INT, a INT, b INT)")
    data = [(i, i*2, i*3) for i in range(60)]
    for r in data: db.run(f"INSERT INTO t VALUES {r}")
    db.run("ALTER TABLE t RENAME COLUMN a TO x")
    db.run("ALTER TABLE t RENAME COLUMN x TO y")                 # chain a->x->y; physical stays 'a'
    con = duckdb.connect(); con.execute("CREATE TABLE t (id INT, y INT, b INT)")
    con.executemany("INSERT INTO t VALUES (?,?,?)", data)
    for q in ["SELECT id, y, b FROM t WHERE y >= 20 ORDER BY id",
              "SELECT SUM(b), COUNT(*), MAX(y), MIN(y) FROM t WHERE y < 30",
              "SELECT id FROM t WHERE y = 40"]:
        got,_ = db.run(q); exp = con.execute(q).fetchall()
        assert got == [tuple(r) for r in exp], (q, got, exp)
    shutil.rmtree(d)

def test_rename_column_groupby_dict_mode():
    # GROUP BY on a renamed plain-dict (mode-0) column resolves through col_map correctly.
    import duckdb, random
    db, d = _db()
    db.run("CREATE TABLE t (id INT, cat INT)")
    random.seed(3); data = [(i, random.randint(0, 3)) for i in range(40)]
    for r in data: db.run(f"INSERT INTO t VALUES {r}")
    db.run("ALTER TABLE t RENAME COLUMN cat TO bucket")
    got,_ = db.run("SELECT bucket, COUNT(*) FROM t GROUP BY bucket ORDER BY bucket")
    con = duckdb.connect(); con.execute("CREATE TABLE t (id INT, bucket INT)")
    con.executemany("INSERT INTO t VALUES (?,?)", data)
    exp = con.execute("SELECT bucket, COUNT(*) FROM t GROUP BY bucket ORDER BY bucket").fetchall()
    assert got == [tuple(r) for r in exp], (got, exp)
    shutil.rmtree(d)
