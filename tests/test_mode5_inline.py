"""Mode-5 inline string columns: high-cardinality non-null string columns are stored row-inline
(no dictionary, no per-row codes) when that beats dict+codes -- dictionary pointers are dead
weight when values rarely repeat AND the strings are high-entropy (no shared prefixes for front-
coding to exploit). Provides values() directly (like mode 4); GROUP BY/fetch factorize on demand.
Gated -- repetitive OR prefix-compressible columns keep the dict. Verified lossless, query-correct
vs DuckDB, and correct through overrides and compaction."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_sql, wdb_override
from wdb_engine import Segment
from wdb_db import Database
from helpers import roundtrip

TMP = tempfile.gettempdir()
def _uniq(n, seed):                       # high-entropy ~unique strings -> reliably mode 5
    rng = np.random.default_rng(seed)
    return np.array([f"{v:016x}" for v in rng.integers(0, 2**63, n, dtype=np.int64)])
def _enc(df):
    t=uuid.uuid4().hex[:8]; pq=f'{TMP}/m5_{t}.parquet'; w=f'{TMP}/m5_{t}.wdb'
    df.to_parquet(pq, index=False); wdb_encode.encode(pq, w); return Segment(w), w, pq
def _norm(rows):
    return sorted([tuple(x.decode() if isinstance(x,(bytes,bytearray)) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))

def test_unique_string_is_mode5():
    seg, w, _ = _enc(pd.DataFrame({'s': _uniq(80000, 0)}))
    assert seg.cols['s']['mode'] == 5
    assert os.path.getsize(w) < 80000 * 18, "should beat storing dict + pointers"

def test_repetitive_string_stays_dict():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({'s': rng.choice([f'v{i}' for i in range(40000)], 400000)})
    seg, _, _ = _enc(df)
    assert seg.cols['s']['mode'] in (0, 1)

def test_mode5_lossless_including_unicode_and_varlen():
    rng = np.random.default_rng(2); pre = ['', 'a', 'café', '日本語', 'emoji😀', 'x'*200]
    vals = [f'{rng.choice(pre)}_{v:016x}' for v in rng.integers(0, 2**63, 60000, dtype=np.int64)]
    seg, w, pq = _enc(pd.DataFrame({'s': vals}))
    assert seg.cols['s']['mode'] == 5
    assert [x.decode('utf-8','surrogatepass') for x in seg.values('s')] == vals

def test_mode5_queries_match_oracle():
    n = 150000; rng = np.random.default_rng(3)
    df = pd.DataFrame({'sid': _uniq(n, 3), 'g': rng.integers(0, 50, n).astype(np.int64)})
    seg, w, pq = _enc(df); assert seg.cols['sid']['mode'] == 5
    con = duckdb.connect(); X = df['sid'].iloc[777]
    for sql in [f"SELECT g FROM TBL WHERE sid = '{X}'",
                f"SELECT COUNT(*) FROM TBL WHERE sid != '{df['sid'].iloc[0]}'",
                "SELECT COUNT(*) FROM TBL WHERE g = 7",
                "SELECT g, COUNT(*) FROM TBL GROUP BY g"]:
        duck = _norm(con.execute(sql.replace('TBL', f"'{pq}'")).fetchall())
        rows, _ = wdb_sql.execute(seg, sql.replace('TBL', 'tbl'))
        assert _norm(rows) == duck, sql

def test_mode5_where_in_on_inline_column():
    sid = _uniq(60000, 4); df = pd.DataFrame({'sid': sid})
    seg, w, pq = _enc(df); assert seg.cols['sid']['mode'] == 5
    a, b = sid[10], sid[20]
    con = duckdb.connect()
    duck = _norm(con.execute(f"SELECT sid FROM '{pq}' WHERE sid IN ('{a}','{b}')").fetchall())
    rows, _ = wdb_sql.execute(seg, f"SELECT sid FROM tbl WHERE sid IN ('{a}','{b}')")
    assert _norm(rows) == duck and len(rows) == 2

def test_mode5_override_rides_on_top():
    sid = _uniq(40000, 5); seg, w, pq = _enc(pd.DataFrame({'sid': sid}))
    assert seg.cols['sid']['mode'] == 5
    wdb_override.set_override(w, 'sid', [10, 20], np.array(['NEW10','NEW20'], dtype=object))
    v = Segment(w).values('sid')
    assert v[10] == 'NEW10' and v[20] == 'NEW20'
    assert v[11] == sid[11].encode() and v[0] == sid[0].encode()

def test_mode5_survives_compaction():
    d = f'{TMP}/m5db_{uuid.uuid4().hex[:8]}'; db = Database.create(d); con = duckdb.connect()
    db.run("CREATE TABLE t (id INT, sid VARCHAR)"); db.set_table_mode('t', 'buffered')
    con.execute("CREATE TABLE t (id INTEGER, sid VARCHAR)")
    rng = np.random.default_rng(9); k = 0
    for _ in range(3):
        ss = [f"{v:016x}" for v in rng.integers(0, 2**63, 100, dtype=np.int64)]
        vals = ",".join(f"({k+i},'{ss[i]}')" for i in range(100)); k += 100
        db.run(f"INSERT INTO t VALUES {vals}"); db.flush('t'); con.execute(f"INSERT INTO t VALUES {vals}")
    db.compact('t')
    assert _norm(db.run("SELECT id, sid FROM t")[0]) == _norm(con.execute("SELECT id, sid FROM t").fetchall())
    shutil.rmtree(d)
