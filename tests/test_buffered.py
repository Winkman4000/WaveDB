"""Two-tier buffered tables: hot/cold merge-read must equal a plain table's answers.
The operator opts a table into 'buffered' mode (INSERT skips re-encode); SELECT merges
the hot buffer with the cold segment; FLUSH folds hot into cold."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb
from wdb_db import Database
import wdb_dml

def _tmpdb(): return os.path.join(tempfile.gettempdir(), f'wbuf_{uuid.uuid4().hex[:8]}')
def _norm(rows):
    return sorted([tuple(round(float(x),4) if isinstance(x,(int,float)) and not isinstance(x,bool) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))

def _two_dbs(schema_sql, insert_batches):
    """Build one default-mode DB and one buffered-mode DB with the SAME data."""
    da, db_ = _tmpdb(), _tmpdb()
    A = Database.create(da); B = Database.create(db_)
    A.run(schema_sql); B.run(schema_sql)
    tname = schema_sql.split()[2]
    B.set_table_mode(tname, 'buffered')
    for batch in insert_batches:
        A.run(batch); B.run(batch)
    return A, B, da, db_, tname

def test_buffered_matches_default_select():
    A,B,da,db_,t = _two_dbs("CREATE TABLE t (g VARCHAR, x INT)",
        ["INSERT INTO t VALUES ('a',10),('b',20)", "INSERT INTO t VALUES ('a',30),('c',40)"])
    for sql in ["SELECT g,x FROM t WHERE x>15", "SELECT g,SUM(x) FROM t GROUP BY g",
                "SELECT COUNT(*) FROM t", "SELECT g,AVG(x) FROM t GROUP BY g",
                "SELECT g,MIN(x),MAX(x) FROM t GROUP BY g"]:
        ra,_ = A.run(sql); rb,_ = B.run(sql)
        assert _norm(ra)==_norm(rb), (sql, _norm(ra), _norm(rb))
    shutil.rmtree(da); shutil.rmtree(db_)

def test_buffered_skips_encode_on_insert():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (x INT)"); db.set_table_mode('t','buffered')
    db.run("INSERT INTO t VALUES (1),(2),(3)")
    # no segment yet (never encoded), but a hot buffer exists
    assert db.cat.get_table('t')['segments'] == [], "buffered insert must not encode a segment"
    assert os.path.exists(wdb_dml.hot_path(db.cat,'t')), "hot buffer should exist"
    rows,_ = db.run("SELECT x FROM t")
    assert _norm(rows)==_norm([(1,),(2,),(3,)])
    shutil.rmtree(d)

def test_flush_folds_hot_into_cold():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (x INT)"); db.set_table_mode('t','buffered')
    db.run("INSERT INTO t VALUES (1),(2)")
    db.run("INSERT INTO t VALUES (3)")
    n = db.flush('t')
    assert n == 3, f"flush should fold 3 rows, got {n}"
    assert db.cat.get_table('t')['segments'] == ['t_0.wdb'], "flush must create the segment"
    assert not os.path.exists(wdb_dml.hot_path(db.cat,'t')), "hot buffer must be cleared"
    rows,_ = db.run("SELECT x FROM t")
    assert _norm(rows)==_norm([(1,),(2,),(3,)])
    shutil.rmtree(d)

def test_insert_after_flush_merges_again():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (g VARCHAR, x INT)"); db.set_table_mode('t','buffered')
    db.run("INSERT INTO t VALUES ('a',10)")
    db.flush('t')                                  # now cold segment has a/10
    db.run("INSERT INTO t VALUES ('a',20),('b',5)")  # new hot rows
    rows,_ = db.run("SELECT g,SUM(x) FROM t GROUP BY g")  # must merge cold(a:10)+hot(a:20,b:5)
    assert _norm(rows)==_norm([('a',30),('b',5)]), _norm(rows)
    shutil.rmtree(d)

def test_buffered_vs_duckdb_groupby():
    A,B,da,db_,t = _two_dbs("CREATE TABLE sales (region VARCHAR, amt INT)",
        ["INSERT INTO sales VALUES ('n',10),('s',20)",
         "INSERT INTO sales VALUES ('n',30),('s',40),('e',5)"])
    sql = "SELECT region, COUNT(*), SUM(amt), AVG(amt) FROM sales GROUP BY region"
    rb,_ = B.run(sql)
    con = duckdb.connect()
    vals = "('n',10),('s',20),('n',30),('s',40),('e',5)"
    duck = con.execute(f"SELECT region,COUNT(*),SUM(amt),AVG(amt) FROM (VALUES {vals}) t(region,amt) GROUP BY region").fetchall()
    assert _norm(rb)==_norm(duck), (_norm(rb), _norm(duck))
    shutil.rmtree(da); shutil.rmtree(db_)

def test_default_table_has_no_hot_file():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (x INT)")              # default mode
    db.run("INSERT INTO t VALUES (1),(2)")
    assert not os.path.exists(wdb_dml.hot_path(db.cat,'t')), "default tables must never create a hot file"
    assert db.cat.get_table('t')['segments'] == ['t_0.wdb']
    shutil.rmtree(d)

def test_flush_persists_across_reopen():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (x INT)"); db.set_table_mode('t','buffered')
    db.run("INSERT INTO t VALUES (7),(8),(9)")
    db.flush('t')
    db2 = Database.open(d)
    rows,_ = db2.run("SELECT x FROM t")
    assert _norm(rows)==_norm([(7,),(8,),(9,)])
    shutil.rmtree(d)
