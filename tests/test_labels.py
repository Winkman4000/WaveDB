"""FD labeling: discover single-column functional dependencies at flush, store as
segment hints for the compactor to later verify."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd, numpy as np
import wdb_labels
from wdb_db import Database

def _tmpdb(): return os.path.join(tempfile.gettempdir(), f'wlab_{uuid.uuid4().hex[:8]}')

def _fd_set(labels): return {(l['det'], l['dep']) for l in labels}

def test_discover_finds_true_fds():
    # key -> attribute FDs, with repeats so there's evidence
    n = 3000
    rng = np.random.default_rng(0)
    part = rng.integers(0, 200, n)            # ~15 repeats each -> low uniqueness
    df = pd.DataFrame({
        'part': part,
        'brand': part % 10,                   # brand is an exact function of part
        'mfgr':  part % 3,                     # mfgr too
        'noise': rng.integers(0, 1000, n),    # unrelated
    })
    fds = _fd_set(wdb_labels.discover_fds(df))
    assert ('part','brand') in fds and ('part','mfgr') in fds
    assert ('brand','noise') not in fds and ('noise','part') not in fds

def test_records_det_uniqueness():
    n=2000; part=np.arange(n)%100        # uniqueness ~ 100/2000 = 0.05
    df=pd.DataFrame({'part':part,'brand':part%10})
    labs=[l for l in wdb_labels.discover_fds(df) if (l['det'],l['dep'])==('part','brand')]
    assert labs and labs[0]['det_uniqueness'] < 0.1

def test_skips_zero_evidence_determinant():
    # fully-unique determinant -> no repeats -> no evidence -> skipped
    n=1000
    df=pd.DataFrame({'uid':np.arange(n), 'x':np.arange(n)%5})
    fds=_fd_set(wdb_labels.discover_fds(df))
    assert ('uid','x') not in fds, "fully-unique determinant must be skipped (zero evidence)"

def test_flush_stores_labels():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (part INT, brand INT)"); db.set_table_mode('t','buffered')
    # part repeats, brand = part%? exact function: brand determined by part
    rows=",".join(f"({p},{p%4})" for p in list(range(50))*4)   # each part repeats 4x
    db.run(f"INSERT INTO t VALUES {rows}")
    db.flush('t')
    labs = db.cat.segment_labels('t','t_0.wdb')
    fds = _fd_set(labs)
    assert ('part','brand') in fds, f"expected part->brand label, got {fds}"
    shutil.rmtree(d)

def test_labels_persist_across_reopen():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (part INT, brand INT)"); db.set_table_mode('t','buffered')
    rows=",".join(f"({p},{p%4})" for p in list(range(50))*4)
    db.run(f"INSERT INTO t VALUES {rows}"); db.flush('t')
    db2=Database.open(d)
    assert ('part','brand') in _fd_set(db2.cat.segment_labels('t','t_0.wdb'))
    shutil.rmtree(d)

def test_default_table_gets_no_labels():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (part INT, brand INT)")     # default mode, no flush path
    rows=",".join(f"({p},{p%4})" for p in list(range(50))*4)
    db.run(f"INSERT INTO t VALUES {rows}")
    assert db.cat.all_labels('t') == {}, "default-mode tables should carry no labels"
    shutil.rmtree(d)

def test_multiple_flushes_label_each_segment():
    d=_tmpdb(); db=Database.create(d)
    db.run("CREATE TABLE t (part INT, brand INT)"); db.set_table_mode('t','buffered')
    db.run(f"INSERT INTO t VALUES {','.join(f'({p},{p%4})' for p in list(range(30))*3)}"); db.flush('t')
    db.run(f"INSERT INTO t VALUES {','.join(f'({p},{p%4})' for p in list(range(30))*3)}"); db.flush('t')
    lab = db.cat.all_labels('t')
    assert set(lab.keys()) == {'t_0.wdb','t_1.wdb'}
    for seg in lab: assert ('part','brand') in _fd_set(lab[seg])
    shutil.rmtree(d)
