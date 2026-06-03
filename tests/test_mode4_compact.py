"""Mode-4 x compaction (step 4c): re-detection on the merged union, fold-in of deletes &
overrides with sidecar cleanup, and FD + mode-4 coexistence. Compaction reconstructs each
segment via values() (mode-4 / override / presence aware) and re-encodes, re-running the
detector -- so a contiguous (affine) union stays mode 4, a broken union degrades losslessly,
and folded mutations are baked into the new segment. Differential vs DuckDB throughout."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_override, wdb_presence
from wdb_engine import Segment
from wdb_db import Database

TMP = tempfile.gettempdir()
def _dir(): return os.path.join(TMP, f'm4c_{uuid.uuid4().hex[:8]}')
def _norm(rows):
    return sorted([tuple(int(x) if isinstance(x, np.integer) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))
def _cold_segment(db, table):
    paths = db.cat.segment_paths(table)
    assert len(paths) == 1, f"expected 1 cold segment after compact, got {len(paths)}"
    return Segment(paths[0]), paths[0]

def test_compact_affine_union_stays_mode4():
    d = _dir(); db = Database.create(d); con = duckdb.connect()
    db.run("CREATE TABLE t (id INT, g VARCHAR)"); db.set_table_mode('t', 'buffered')
    con.execute("CREATE TABLE t (id INTEGER, g VARCHAR)")
    k = 0
    for _ in range(3):                                      # 3 contiguous flushes
        rows = ",".join(f"({1000000+k+i},'g{(k+i)%3}')" for i in range(50)); k += 50
        db.run(f"INSERT INTO t VALUES {rows}"); db.flush('t'); con.execute(f"INSERT INTO t VALUES {rows}")
    assert len(db.cat.get_table('t')['segments']) == 3
    db.compact('t')
    seg, _ = _cold_segment(db, 't')
    assert seg.cols['id']['mode'] == 4, f"contiguous union should stay mode 4, got {seg.cols['id']['mode']}"
    assert _norm(db.run("SELECT id,g FROM t")[0]) == _norm(con.execute("SELECT id,g FROM t").fetchall())
    shutil.rmtree(d)

def test_compact_nonaffine_union_lossless():
    # flush batches OUT of order so the concatenated union is not monotonic
    d = _dir(); db = Database.create(d); con = duckdb.connect()
    db.run("CREATE TABLE t (id INT)"); db.set_table_mode('t', 'buffered')
    con.execute("CREATE TABLE t (id INTEGER)")
    for lo in (1000200, 1000000, 1000100):                 # discontinuous order
        rows = ",".join(f"({lo+i})" for i in range(100))
        db.run(f"INSERT INTO t VALUES {rows}"); db.flush('t'); con.execute(f"INSERT INTO t VALUES {rows}")
    db.compact('t')
    seg, _ = _cold_segment(db, 't')
    # lossless regardless of which mode the detector chose for the broken union
    assert _norm(db.run("SELECT id FROM t")[0]) == _norm(con.execute("SELECT id FROM t").fetchall())
    assert _norm(db.run("SELECT id FROM t WHERE id BETWEEN 1000050 AND 1000150")[0]) == \
           _norm(con.execute("SELECT id FROM t WHERE id BETWEEN 1000050 AND 1000150").fetchall())
    shutil.rmtree(d)

def test_compact_folds_deletes_overrides_and_cleans_sidecars():
    d = _dir(); db = Database.create(d); con = duckdb.connect()
    db.run("CREATE TABLE t (id INT, v INT)"); db.set_table_mode('t', 'buffered')
    con.execute("CREATE TABLE t (id INTEGER, v INTEGER)")
    k = 0
    for _ in range(3):
        rows = ",".join(f"({1000000+k+i},{(k+i)%7})" for i in range(40)); k += 40
        db.run(f"INSERT INTO t VALUES {rows}"); db.flush('t'); con.execute(f"INSERT INTO t VALUES {rows}")
    for sql in ["DELETE FROM t WHERE v = 0",
                "UPDATE t SET v = 99 WHERE v = 1",
                "UPDATE t SET id = id + 5000000 WHERE v = 2"]:   # update the key (mode-4 override)
        db.run(sql); con.execute(sql)
    seg_paths_before = list(db.cat.segment_paths('t'))
    db.compact('t')
    seg, _ = _cold_segment(db, 't')
    assert _norm(db.run("SELECT id,v FROM t")[0]) == _norm(con.execute("SELECT id,v FROM t").fetchall())
    # the merged segments' override/presence sidecars must be gone (no orphans)
    for p in seg_paths_before:
        assert not os.path.exists(wdb_override.path_for(p)), "orphaned override sidecar"
        assert not os.path.exists(wdb_presence.path_for(p)), "orphaned presence sidecar"
        assert not os.path.exists(p), "merged segment file not removed"
    shutil.rmtree(d)

def test_fd_and_mode4_coexist_in_one_segment():
    # encoder-level: a segment with a sequential id (mode 4) AND an exact FD region->code
    n = 4000
    region = np.array((['north','south','east','west'] * (n // 4)))
    rmap = {'north':0,'south':1,'east':2,'west':3}
    df = pd.DataFrame({'id': 1_000_000 + np.arange(n, dtype=np.int64),
                       'region': region,
                       'rcode': np.array([rmap[r] for r in region], dtype=np.int64)})
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/m4fd_{t}.parquet'; wdb = f'{TMP}/m4fd_{t}.wdb'
    df.to_parquet(pq, index=False)
    wdb_encode.encode(pq, wdb, fd_specs={'rcode': 'region'})    # rcode as FD-ref into region
    seg = Segment(wdb)
    assert seg.cols['id']['mode'] == 4, f"id should be mode 4, got {seg.cols['id']['mode']}"
    assert seg.cols['rcode']['mode'] == 3, f"rcode should be mode 3 (FD), got {seg.cols['rcode']['mode']}"
    assert seg.cols['region']['mode'] == 0
    assert np.array_equal(seg.values('id'), df['id'].values)
    assert np.array_equal(seg.values('rcode'), df['rcode'].values)
    assert [x.decode() for x in seg.values('region')] == list(region)
    for f in (pq, wdb):
        if os.path.exists(f): os.remove(f)
