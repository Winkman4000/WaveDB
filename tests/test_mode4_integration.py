"""Mode-4 (affine/sequence) END-TO-END integration through the real encoder + engine:
the column type chosen, the size win locked, lossless round-trip, query correctness vs the
DuckDB oracle, correct decline, and the override (UPDATE-the-key) interaction. The codec
itself is covered in isolation by test_seqcodec; this proves it wired in correctly."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_sql
from wdb_engine import Segment
from wdb_db import Database
from helpers import roundtrip, assert_lossless

TMP = tempfile.gettempdir()
def _enc(df):
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/mi_{t}.parquet'; wdb = f'{TMP}/mi_{t}.wdb'
    df.to_parquet(pq, index=False); wdb_encode.encode(pq, wdb)
    return Segment(wdb), wdb, pq
def _norm(rows):
    return sorted([tuple(round(float(x),4) if isinstance(x,(int,float,np.integer,np.floating)) and not isinstance(x,bool) else x for x in r) for r in rows], key=lambda t: tuple(str(x) for x in t))

def test_sequential_int_is_mode4_and_tiny():
    seg, wdb, _ = _enc(pd.DataFrame({'id': 1_000_000 + np.arange(1_000_000, dtype=np.int64)}))
    assert seg.cols['id']['mode'] == 4
    assert os.path.getsize(wdb) < 100, f"1M sequential column must be tiny, got {os.path.getsize(wdb)} bytes"

def test_mode4_massively_smaller_than_mode2():
    n = 500_000
    seg_s, wdb_s, _ = _enc(pd.DataFrame({'id': np.arange(n, dtype=np.int64)}))               # mode 4
    shuf = np.random.default_rng(0).permutation(n).astype(np.int64)
    seg_r, wdb_r, _ = _enc(pd.DataFrame({'id': shuf}))                                        # mode 2
    assert seg_s.cols['id']['mode'] == 4 and seg_r.cols['id']['mode'] == 2
    assert os.path.getsize(wdb_s) * 1000 < os.path.getsize(wdb_r), "mode-4 should be >1000x smaller"

def test_mode4_roundtrip_lossless_int_dt_gaps():
    df = pd.DataFrame({'x': 1_000_000 + np.arange(60000, dtype=np.int64)})
    seg, pq = roundtrip(df); assert assert_lossless(seg, pq, 'x') == 4
    base = np.datetime64('2010-06-01T00:00:00')
    df = pd.DataFrame({'x': base + np.arange(60000, dtype='timedelta64[s]')})
    seg, pq = roundtrip(df); assert assert_lossless(seg, pq, 'x') == 4
    keep = np.random.default_rng(1).random(70000) > 0.02
    df = pd.DataFrame({'x': np.flatnonzero(keep)[:60000].astype(np.int64)})
    seg, pq = roundtrip(df); assert assert_lossless(seg, pq, 'x') == 4

def test_mode4_queries_match_oracle():
    n = 4000
    df = pd.DataFrame({'id': 1_000_000 + np.arange(n, dtype=np.int64),
                       'grp': (['a','b','c','d'] * (n // 4))})
    seg, wdb, pq = _enc(df)
    assert seg.cols['id']['mode'] == 4
    con = duckdb.connect()
    for sql in ["SELECT grp FROM TBL WHERE id = 1001500",
                "SELECT COUNT(*) FROM TBL WHERE id BETWEEN 1000100 AND 1000200",
                "SELECT MIN(id), MAX(id), SUM(id) FROM TBL",
                "SELECT grp, COUNT(*) FROM TBL GROUP BY grp",
                "SELECT id FROM TBL WHERE id > 1003990 ORDER BY id ASC"]:
        duck = _norm(con.execute(sql.replace('TBL', f"'{pq}'")).fetchall())
        rows, _ = wdb_sql.execute(seg, sql.replace('TBL', 'tbl'))
        assert _norm(rows) == duck, f"{sql}\n wdb={_norm(rows)[:4]}\n duck={duck[:4]}"

def test_mode4_declines_on_unsuitable():
    for col, want_not in [
        (np.random.default_rng(2).integers(0, 2**60, 80000).astype(np.int64), 4),  # random
        (np.array([1,2,3,2,1]*1000, dtype=np.int64), 4),                            # lowcard
    ]:
        seg, _, _ = _enc(pd.DataFrame({'x': col}))
        assert seg.cols['x']['mode'] != want_not
    s = pd.array(list(range(2000)), dtype='Int64'); s[3] = pd.NA                     # seq + null
    seg, _, _ = _enc(pd.DataFrame({'x': s}))
    assert seg.cols['x']['mode'] != 4

def test_mode4_update_key_then_query():
    d = f'{TMP}/mi_db_{uuid.uuid4().hex[:8]}'; db = Database.create(d)
    db.run("CREATE TABLE t (id INT, grp VARCHAR)"); db.set_table_mode('t','buffered')
    db.run("INSERT INTO t VALUES " + ",".join(f"({1000000+i},'g{i%3}')" for i in range(300))); db.flush('t')
    con = duckdb.connect(); con.execute("CREATE TABLE t (id INTEGER, grp VARCHAR)")
    con.execute("INSERT INTO t VALUES " + ",".join(f"({1000000+i},'g{i%3}')" for i in range(300)))
    for sql in ["UPDATE t SET id = id + 9000000 WHERE grp = 'g1'", "UPDATE t SET id = id + 1 WHERE id = 1000000"]:
        db.run(sql); con.execute(sql)
    w = _norm(db.run("SELECT id, grp FROM t")[0]); dk = _norm(con.execute("SELECT id, grp FROM t").fetchall())
    assert w == dk, f"update-on-mode4-key mismatch: {w[:4]} vs {dk[:4]}"
    shutil.rmtree(d)
