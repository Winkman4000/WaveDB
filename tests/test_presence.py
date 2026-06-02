"""Presence sidecar (mutable layer, step 1a): a per-segment bitset tombstones rows without
rewriting the immutable .wdb. Reads must skip tombstoned rows across every query shape;
a missing sidecar must behave exactly as before."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode, wdb_presence, wdb_sql
from wdb_engine import Segment

TMP = tempfile.gettempdir()
def _seg(df):
    base = os.path.join(TMP, f'pres_{uuid.uuid4().hex[:8]}')
    pq = base + '.parquet'; df.to_parquet(pq, index=False)
    sp = base + '.wdb'; wdb_encode.encode(pq, sp)
    return sp
def _rows(sp, sql):
    return sorted(map(tuple, wdb_sql.execute(Segment(sp), sql)[0]), key=lambda t: str(t))
def _scalar(sp, sql):
    return wdb_sql.execute(Segment(sp), sql)[0][0][0]

def test_no_sidecar_unchanged():
    sp = _seg(pd.DataFrame({'g':['a','b','a'], 'x':[1,2,3]}))
    assert _scalar(sp, "SELECT COUNT(*) FROM t") == 3
    assert _scalar(sp, "SELECT SUM(x) FROM t") == 6
    assert seg_presence_is_none(sp)

def seg_presence_is_none(sp):
    return Segment(sp).presence_mask() is None

def test_sidecar_roundtrip_format():
    sp = _seg(pd.DataFrame({'x': list(range(100))}))
    pres = np.ones(100, dtype=bool); pres[[3, 7, 50, 99]] = False
    wdb_presence.save(sp, pres)
    back = wdb_presence.load(sp, 100)
    assert back is not None and np.array_equal(back, pres)
    os.remove(wdb_presence.path_for(sp))

def test_count_and_sum_skip_tombstoned():
    sp = _seg(pd.DataFrame({'g':['a','b','a','c','b','a'], 'x':[10,20,30,40,50,60]}))
    pres = np.ones(6, dtype=bool); pres[[0,4]] = False   # drop x=10, x=50
    wdb_presence.save(sp, pres)
    assert _scalar(sp, "SELECT COUNT(*) FROM t") == 4
    assert _scalar(sp, "SELECT SUM(x) FROM t") == 150
    os.remove(wdb_presence.path_for(sp))

def test_projection_and_where_skip_tombstoned():
    sp = _seg(pd.DataFrame({'x':[10,20,30,40,50,60]}))
    pres = np.ones(6, dtype=bool); pres[[0,4]] = False
    wdb_presence.save(sp, pres)
    assert _rows(sp, "SELECT x FROM t") == [(20,),(30,),(40,),(60,)]
    assert _rows(sp, "SELECT x FROM t WHERE x>25") == [(30,),(40,),(60,)]
    os.remove(wdb_presence.path_for(sp))

def test_group_by_skips_tombstoned():
    sp = _seg(pd.DataFrame({'g':['a','b','a','c','b','a'], 'x':[10,20,30,40,50,60]}))
    pres = np.ones(6, dtype=bool); pres[[0,4]] = False
    wdb_presence.save(sp, pres)
    assert _rows(sp, "SELECT g,COUNT(*) FROM t GROUP BY g") == [('a',2),('b',1),('c',1)]
    assert _rows(sp, "SELECT g,SUM(x) FROM t GROUP BY g") == [('a',90),('b',20),('c',40)]
    os.remove(wdb_presence.path_for(sp))

def test_mark_deleted_helper():
    sp = _seg(pd.DataFrame({'x': list(range(10))}))
    n1 = wdb_presence.mark_deleted(sp, 10, [1, 2, 3])
    assert n1 == 3 and wdb_presence.live_count(sp, 10) == 7
    n2 = wdb_presence.mark_deleted(sp, 10, [3, 4])   # 3 already gone, only 4 is new
    assert n2 == 1 and wdb_presence.live_count(sp, 10) == 6
    os.remove(wdb_presence.path_for(sp))

def test_all_rows_tombstoned():
    sp = _seg(pd.DataFrame({'x':[1,2,3]}))
    wdb_presence.save(sp, np.zeros(3, dtype=bool))
    assert _scalar(sp, "SELECT COUNT(*) FROM t") == 0
    assert _rows(sp, "SELECT x FROM t") == []
    os.remove(wdb_presence.path_for(sp))
