"""Catalog: persistent table registry. Create -> register -> reopen -> still there."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from wdb_catalog import Catalog

def _tmpdb():
    return os.path.join(tempfile.gettempdir(), f'wdbdb_{uuid.uuid4().hex[:8]}')

def test_create_and_reopen_empty():
    d = _tmpdb()
    Catalog.create(d)
    cat = Catalog.open(d)
    assert cat.list_tables() == []
    shutil.rmtree(d)

def test_add_table_persists():
    d = _tmpdb()
    cat = Catalog.create(d)
    cat.add_table('users', [['id','int'],['name','string']])
    # reopen from disk - must survive
    cat2 = Catalog.open(d)
    assert cat2.list_tables() == ['users']
    assert cat2.get_table('users')['schema'] == [['id','int'],['name','string']]
    shutil.rmtree(d)

def test_duplicate_table_errors():
    d = _tmpdb()
    cat = Catalog.create(d)
    cat.add_table('t', [['a','int']])
    try:
        cat.add_table('t', [['a','int']])
        assert False, "expected error on duplicate table"
    except ValueError:
        pass
    shutil.rmtree(d)

def test_unknown_table_errors():
    d = _tmpdb()
    cat = Catalog.create(d)
    try:
        cat.get_table('nope'); assert False, "expected KeyError"
    except KeyError:
        pass
    shutil.rmtree(d)

def test_create_existing_errors():
    d = _tmpdb()
    Catalog.create(d)
    try:
        Catalog.create(d); assert False, "expected FileExistsError"
    except FileExistsError:
        pass
    shutil.rmtree(d)

def test_drop_table():
    d = _tmpdb()
    cat = Catalog.create(d)
    cat.add_table('t', [['a','int']])
    cat.drop_table('t')
    assert Catalog.open(d).list_tables() == []
    shutil.rmtree(d)

def test_segments_register_and_persist():
    d = _tmpdb()
    cat = Catalog.create(d)
    cat.add_table('t', [['a','int']])
    cat.add_segment('t', 't_0.wdb')
    cat.add_segment('t', 't_1.wdb')
    cat2 = Catalog.open(d)
    assert cat2.get_table('t')['segments'] == ['t_0.wdb', 't_1.wdb']
    assert all(p.endswith('.wdb') for p in cat2.segment_paths('t'))
    shutil.rmtree(d)
