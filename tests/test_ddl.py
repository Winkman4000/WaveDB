"""CREATE TABLE: borrowed SQL syntax -> catalog schema. Types map correctly."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from wdb_catalog import Catalog
import wdb_ddl

def _tmpdb():
    return os.path.join(tempfile.gettempdir(), f'wddl_{uuid.uuid4().hex[:8]}')

def test_create_table_basic_types():
    d = _tmpdb(); cat = Catalog.create(d)
    wdb_ddl.create_table(cat, "CREATE TABLE users (id INT, name VARCHAR, price DOUBLE, created TIMESTAMP)")
    sch = Catalog.open(d).get_table('users')['schema']
    assert sch == [['id','int'],['name','string'],['price','float'],['created','datetime']], sch
    shutil.rmtree(d)

def test_create_table_int_variants():
    d = _tmpdb(); cat = Catalog.create(d)
    wdb_ddl.create_table(cat, "CREATE TABLE t (a BIGINT, b SMALLINT, c TINYINT, d INTEGER)")
    sch = cat.get_table('t')['schema']
    assert [c[1] for c in sch] == ['int','int','int','int'], sch
    shutil.rmtree(d)

def test_create_table_type_families():
    d = _tmpdb(); cat = Catalog.create(d)
    wdb_ddl.create_table(cat, "CREATE TABLE t (f REAL, dec DECIMAL(10,2), s TEXT, dt DATE, b BOOLEAN)")
    sch = {c[0]: c[1] for c in cat.get_table('t')['schema']}
    assert sch == {'f':'float','dec':'float','s':'string','dt':'datetime','b':'int'}, sch
    shutil.rmtree(d)

def test_create_table_persists_and_registers():
    d = _tmpdb(); cat = Catalog.create(d)
    name, schema = wdb_ddl.create_table(cat, "CREATE TABLE orders (oid INT, total DOUBLE)")
    assert name == 'orders'
    assert Catalog.open(d).list_tables() == ['orders']  # persisted to disk
    shutil.rmtree(d)

def test_create_table_duplicate_errors():
    d = _tmpdb(); cat = Catalog.create(d)
    wdb_ddl.create_table(cat, "CREATE TABLE t (a INT)")
    try:
        wdb_ddl.create_table(cat, "CREATE TABLE t (a INT)")
        assert False, "expected duplicate error"
    except ValueError:
        pass
    shutil.rmtree(d)

def test_create_table_no_columns_errors():
    d = _tmpdb(); cat = Catalog.create(d)
    try:
        wdb_ddl.create_table(cat, "CREATE TABLE t AS SELECT 1")
        assert False, "expected error (no column list)"
    except (ValueError, NotImplementedError):
        pass
    shutil.rmtree(d)

def test_non_create_rejected():
    d = _tmpdb(); cat = Catalog.create(d)
    try:
        wdb_ddl.parse_create_table("SELECT * FROM t")
        assert False, "expected rejection of non-CREATE"
    except NotImplementedError:
        pass
    shutil.rmtree(d)
