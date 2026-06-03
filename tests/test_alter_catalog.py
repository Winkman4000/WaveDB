"""ALTER foundation: catalog schema-evolution methods (no SQL yet). The logical schema lives in
the catalog; 'phys' tracks the fixed physical name across renames; 'defaults' holds ADD COLUMN
fill values. These are the primitives that ALTER TABLE statements drive in later steps."""
import sys, os, tempfile, uuid, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from wdb_catalog import Catalog

def _cat():
    d = os.path.join(tempfile.gettempdir(), f'altcat_{uuid.uuid4().hex[:8]}')
    c = Catalog.create(d); c.add_table('t', [['id','int'],['name','string'],['amt','float']])
    return c, d

def test_identity_when_unaltered():
    c, d = _cat()
    assert c.column_names('t') == ['id','name','amt']
    assert c.phys_map('t') == {}                       # identity -> no translation
    shutil.rmtree(d)

def test_rename_column_keeps_physical():
    c, d = _cat()
    c.rename_column('t', 'name', 'label')
    assert c.column_names('t') == ['id','label','amt']
    assert c.phys_map('t') == {'label': 'name'}         # physical stays 'name' in segments
    # reopen -> persisted
    assert Catalog.open(d).phys_map('t') == {'label': 'name'}
    shutil.rmtree(d)

def test_rename_column_chain_preserves_origin():
    c, d = _cat()
    c.rename_column('t', 'name', 'label')
    c.rename_column('t', 'label', 'title')
    assert c.column_names('t') == ['id','title','amt']
    assert c.phys_map('t') == {'title': 'name'}         # chained a->b->c still points at original
    shutil.rmtree(d)

def test_rename_back_to_physical_clears_map():
    c, d = _cat()
    c.rename_column('t', 'name', 'label'); c.rename_column('t', 'label', 'name')
    assert c.phys_map('t') == {}                         # renamed back -> identity again
    shutil.rmtree(d)

def test_rename_errors():
    c, d = _cat()
    try: c.rename_column('t','nope','x'); assert False
    except KeyError: pass
    try: c.rename_column('t','id','amt'); assert False   # target exists
    except ValueError: pass
    shutil.rmtree(d)

def test_add_column_records_default():
    c, d = _cat()
    c.add_column('t', 'active', 'int', default=1)
    assert c.column_names('t') == ['id','name','amt','active']
    assert c.column_default('t','active') == 1
    assert c.column_default('t','id') is None
    try: c.add_column('t','id','int'); assert False
    except ValueError: pass
    shutil.rmtree(d)

def test_drop_column():
    c, d = _cat()
    c.rename_column('t','name','label')                  # give it a phys entry first
    c.drop_column('t', 'label')
    assert c.column_names('t') == ['id','amt']
    assert c.phys_map('t') == {}                          # phys entry cleaned up
    try: c.drop_column('t','nope'); assert False
    except KeyError: pass
    shutil.rmtree(d)

def test_drop_last_column_refused():
    d = os.path.join(tempfile.gettempdir(), f'altcat_{uuid.uuid4().hex[:8]}')
    c = Catalog.create(d); c.add_table('s', [['only','int']])
    try: c.drop_column('s','only'); assert False
    except ValueError: pass
    shutil.rmtree(d)

def test_rename_table():
    c, d = _cat()
    c.rename_table('t', 'events')
    assert 'events' in c.list_tables() and 't' not in c.list_tables()
    assert c.column_names('events') == ['id','name','amt']
    try: c.rename_table('nope','x'); assert False
    except KeyError: pass
    shutil.rmtree(d)
