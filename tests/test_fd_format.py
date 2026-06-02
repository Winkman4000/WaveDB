"""FD-reference format integration (stage 3b): a verified FD stores the dependent column
as mode-3 references on disk. Must decode IDENTICALLY to normal encoding, and be smaller.
End-to-end: flush->compact verifies the FD and encodes it; queries match before/after."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode
from wdb_engine import Segment
from wdb_db import Database

TMP = tempfile.gettempdir()
def _pq(df):
    p = os.path.join(TMP, f'fdf_{uuid.uuid4().hex[:8]}.parquet'); df.to_parquet(p, index=False); return p
def _norm(rows):
    return sorted([tuple(round(float(x),4) if isinstance(x,(int,float)) and not isinstance(x,bool) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))

def _equiv(df, fd_specs):
    src = _pq(df); a = src+'.n.wdb'; b = src+'.f.wdb'
    wdb_encode.encode(src, a)
    wdb_encode.encode(src, b, fd_specs=fd_specs)
    sn, sf = Segment(a), Segment(b)
    for col in df.columns:
        assert np.array_equal(np.asarray(sn.values(col)), np.asarray(sf.values(col))), f"{col} differs"
    for dep in fd_specs:
        assert sf.cols[dep]['mode'] == 3, f"{dep} should be mode 3"
    return os.path.getsize(a), os.path.getsize(b)

def test_equivalence_and_smaller():
    rng = np.random.default_rng(0); N = 100000
    part = rng.integers(0, 2000, N)
    df = pd.DataFrame({'part': part.astype(np.int64),
                       'brand': (part % 25).astype(np.int64),
                       'noise': rng.integers(0, 1000, N).astype(np.int64)})
    bn, bf = _equiv(df, {'brand': 'part'})
    assert bf < bn, (bn, bf)

def test_string_dependent():
    rng = np.random.default_rng(1); N = 60000
    part = rng.integers(0, 500, N)
    typ = np.array([f'type_{p%30}' for p in part], dtype=object)
    df = pd.DataFrame({'part': part.astype(np.int64), 'typ': typ})
    _equiv(df, {'typ': 'part'})

def test_highcard_mode2_determinant():
    # determinant with many distinct ints (-> mode 2 delta dict); gather must still work
    rng = np.random.default_rng(2); N = 120000
    part = rng.integers(0, 90000, N)            # high-card determinant
    df = pd.DataFrame({'part': part.astype(np.int64), 'brand': (part % 7).astype(np.int64)})
    src = _pq(df); b = src + '.f.wdb'
    wdb_encode.encode(src, b, fd_specs={'brand': 'part'})
    s = Segment(b)
    assert s.cols['part']['mode'] == 2 and s.cols['brand']['mode'] == 3
    # lossless against a normal encode
    a = src + '.n.wdb'; wdb_encode.encode(src, a); sn = Segment(a)
    assert np.array_equal(np.asarray(sn.values('brand')), np.asarray(s.values('brand')))

def test_nulls_in_x_and_y():
    # determinant and dependent both have nulls; FD still exact (null-part -> null-brand)
    parts = [1, 2, None, 1, 2, None, 3, 3]
    brands = [10, 20, None, 10, 20, None, 30, 30]
    df = pd.DataFrame({'part': pd.array(parts*200, dtype='Int64'),
                       'brand': pd.array(brands*200, dtype='Int64')})
    _equiv(df, {'brand': 'part'})

def _tmpdb(): return os.path.join(TMP, f'fdfdb_{uuid.uuid4().hex[:8]}')

def test_end_to_end_compact_uses_fd_and_is_lossless():
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (part INT, brand INT, qty INT)"); db.set_table_mode('t','buffered')
    rng = np.random.default_rng(3)
    for _ in range(3):
        rows = ",".join(f"({int(p)},{int(p)%12},{int(q)})"
                        for p,q in zip(rng.integers(0,300,500), rng.integers(1,99,500)))
        db.run(f"INSERT INTO t VALUES {rows}"); db.flush('t')
    before = {q: _norm(db.run(q)[0]) for q in
              ["SELECT part,brand FROM t","SELECT brand,COUNT(*) FROM t GROUP BY brand",
               "SELECT SUM(qty) FROM t WHERE brand=5","SELECT brand,SUM(qty) FROM t GROUP BY brand"]}
    res = db.compact('t')
    assert res['fd_encoded'] >= 1, res
    seg = Segment(os.path.join(d, res['new_segment']))
    assert seg.cols['brand']['mode'] == 3, "brand should be FD-encoded after compaction"
    for q, want in before.items():
        assert _norm(db.run(q)[0]) == want, (q, _norm(db.run(q)[0]), want)
    # persists across reopen
    db2 = Database.open(d)
    assert _norm(db2.run("SELECT part,brand FROM t")[0]) == before["SELECT part,brand FROM t"]
    shutil.rmtree(d)

def test_fd_compaction_smaller_than_normal():
    # same union encoded by compaction (with FD) vs a normal encode of the same rows
    d = _tmpdb(); db = Database.create(d)
    db.run("CREATE TABLE t (part INT, brand INT)"); db.set_table_mode('t','buffered')
    rng = np.random.default_rng(4); allrows = []
    for _ in range(3):
        ps = rng.integers(0, 400, 1000)
        rows = ",".join(f"({int(p)},{int(p)%20})" for p in ps)
        allrows += [(int(p), int(p)%20) for p in ps]
        db.run(f"INSERT INTO t VALUES {rows}"); db.flush('t')
    res = db.compact('t')
    fd_size = os.path.getsize(os.path.join(d, res['new_segment']))
    df = pd.DataFrame(allrows, columns=['part','brand'])
    src = _pq(df); norm = src + '.n.wdb'; wdb_encode.encode(src, norm)
    assert fd_size < os.path.getsize(norm), (fd_size, os.path.getsize(norm))
    shutil.rmtree(d)
