"""End-to-end: build a database from scratch with SQL. CREATE -> INSERT -> SELECT.
Step 3a: single segment per table."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb
from wdb_db import Database

def _tmpdb(): return os.path.join(tempfile.gettempdir(), f'wdb_{uuid.uuid4().hex[:8]}')

def _norm(rows):
    out = []
    for r in rows:
        out.append(tuple(round(float(x),4) if isinstance(x,(int,float)) and not isinstance(x,bool) else x for x in r))
    return sorted(out, key=lambda t: tuple(str(x) for x in t))

def test_create_insert_select_all():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE users (id INT, name VARCHAR)")
    db.run("INSERT INTO users VALUES (1,'alice'),(2,'bob'),(3,'carol')")
    rows, hdr = db.run("SELECT id, name FROM users")
    assert _norm(rows) == _norm([(1,'alice'),(2,'bob'),(3,'carol')]), rows
    shutil.rmtree(d)

def test_insert_accumulates_across_statements():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (x INT)")
    db.run("INSERT INTO t VALUES (10)")
    db.run("INSERT INTO t VALUES (20),(30)")
    rows, _ = db.run("SELECT x FROM t")
    assert _norm(rows) == _norm([(10,),(20,),(30,)]), rows
    shutil.rmtree(d)

def test_select_where():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (x INT, g VARCHAR)")
    db.run("INSERT INTO t VALUES (5,'a'),(15,'b'),(25,'a'),(35,'b')")
    rows, _ = db.run("SELECT x FROM t WHERE x > 20")
    assert _norm(rows) == _norm([(25,),(35,)]), rows
    shutil.rmtree(d)

def test_groupby_matches_duckdb():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE sales (region VARCHAR, amt INT)")
    vals = "(  'north',10),('south',20),('north',30),('south',40),('east',5)"
    db.run(f"INSERT INTO sales VALUES {vals}")
    rows, _ = db.run("SELECT region, SUM(amt) FROM sales GROUP BY region")
    con = duckdb.connect()
    duck = con.execute(f"SELECT region, SUM(amt) FROM (VALUES {vals}) AS t(region,amt) GROUP BY region").fetchall()
    assert _norm(rows) == _norm(duck), (rows, duck)
    shutil.rmtree(d)

def test_explicit_columns():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (a INT, b VARCHAR, c INT)")
    db.run("INSERT INTO t (c, a, b) VALUES (3, 1, 'x')")  # out-of-order column list
    rows, _ = db.run("SELECT a, b, c FROM t")
    assert _norm(rows) == _norm([(1,'x',3)]), rows
    shutil.rmtree(d)

def test_float_and_datetime():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (price DOUBLE, day DATE)")
    db.run("INSERT INTO t VALUES (1.5, '2020-01-15'),(2.25,'2021-06-30')")
    rows, _ = db.run("SELECT price FROM t WHERE price > 2.0")
    assert _norm(rows) == _norm([(2.25,)]), rows
    shutil.rmtree(d)

def test_reopen_database():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (x INT)")
    db.run("INSERT INTO t VALUES (42),(99)")
    db2 = Database.open(d)                       # reopen from disk
    rows, _ = db2.run("SELECT x FROM t")
    assert _norm(rows) == _norm([(42,),(99,)]), rows
    shutil.rmtree(d)

def test_single_segment_invariant():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (x INT)")
    db.run("INSERT INTO t VALUES (1)")
    db.run("INSERT INTO t VALUES (2)")
    db.run("INSERT INTO t VALUES (3)")
    assert db.cat.get_table('t')['segments'] == ['t_0.wdb'], "must stay single-segment"
    shutil.rmtree(d)
