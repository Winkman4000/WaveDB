"""Cluster / cube / grouped_multi ROUTING coverage on a clustered + cubed synthetic dataset (separate
from test_path_coverage.py's unclustered star schema). This locks the post-slice2 routing contract:

  - low-card filter-free GROUP BY            -> materialised CUBE              (wdb_cube._CUBE_HITS)
  - high-card GROUP BY (cube over its cap)   -> grouped_multi single-pass     (wdb_join._FAST_HITS, no cube)
  - clustered single-key GROUP BY + WHERE    -> per-slice scalar agg          (wdb_join._SLICE_SCALAR_HITS)
  - filtered 2-key GROUP BY (cube gated off) -> grouped_multi fallback        (wdb_join._FAST_HITS, no cube)

slice2 was removed: a high-card 2-key group MUST land on grouped_multi (which beats DuckDB at high card),
never a per-range Python loop. Each case is checked against DuckDB too -- a wrong fast answer is the worst
outcome. All synthetic + tiny; no bench DB needed. Run `python3 tests/run.py cluster_paths`."""
import sys, os, tempfile, uuid, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from decimal import Decimal
import duckdb, wdb_encode, wdb_cube, wdb_join
from wdb_db import Database
from wdb_engine import Segment

_DB = None; _CON = None
def _fixture():
    """60k rows clustered by `a`. a x b = 32 cells (cube fires); a x c = 8000 cells (> 4096 cap -> the
    ['a','c'] cube is declined at build, so that group must fall to grouped_multi). String keys are
    dict-encoded (mode 0) and measures are real doubles (dt 2) so the fused path actually engages."""
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    _CON.execute("CREATE TABLE t AS SELECT i AS id, 'A'||(i%8) AS a, 'B'||(i%4) AS b, "
                 "'C'||(i%1000) AS c, CAST(((i%50)+1)*1.5 AS DOUBLE) AS m1, "
                 "CAST(((i%97)+1)*2.0 AS DOUBLE) AS m2 FROM range(60000) t(i)")
    d = os.path.join(tempfile.gettempdir(), f'clpath_{uuid.uuid4().hex[:8]}'); os.makedirs(d, exist_ok=True)
    _DB = Database.create(d)
    wt = {'BIGINT': 'int', 'INTEGER': 'int', 'VARCHAR': 'string', 'DOUBLE': 'float'}
    desc = _CON.execute("DESCRIBE t").fetchall()
    pq = os.path.join(d, 't.parquet'); _CON.execute(f"COPY (SELECT * FROM t ORDER BY id) TO '{pq}' (FORMAT parquet)")
    _DB.cat.add_table('t', [[c[0], wt[c[1]]] for c in desc])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'),
                      cluster_by='a', cubes=[['a'], ['a', 'b'], ['a', 'c']])   # ['a','c'] declined by cap
    _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON

def _norm(rows):
    return sorted(tuple(round(float(x), 3) if isinstance(x, (float, Decimal)) else
                  (None if x is None else (int(x) if isinstance(x, int) and not isinstance(x, bool)
                   else str(x))) for x in r) for r in rows)

def _run(q, expect):
    """expect in {'cube','slice','grouped_multi'} -- assert the matching diagnostic counter advanced
    (and the others that should NOT) and that the answer equals DuckDB."""
    db, con = _fixture()
    c0, s0, f0 = wdb_cube._CUBE_HITS, wdb_join._SLICE_SCALAR_HITS, wdb_join._FAST_HITS
    g = _norm(db.run(q)[0]); e = _norm([tuple(r) for r in con.execute(q).fetchall()])
    dc, ds, df = wdb_cube._CUBE_HITS - c0, wdb_join._SLICE_SCALAR_HITS - s0, wdb_join._FAST_HITS - f0
    assert g == e, f"answer != DuckDB for {q}\n got {g[:4]}\n exp {e[:4]}"
    if expect == 'cube':
        assert dc == 1 and ds == 0, f"expected CUBE, got cube={dc} slice={ds} fast={df}: {q}"
    elif expect == 'slice':
        assert ds == 1 and dc == 0, f"expected SLICE, got cube={dc} slice={ds} fast={df}: {q}"
    elif expect == 'grouped_multi':
        assert dc == 0 and ds == 0 and df == 1, \
            f"expected grouped_multi (fast, no cube/slice), got cube={dc} slice={ds} fast={df}: {q}"
    else:
        raise ValueError(expect)

# ---- CUBE: low-card, filter-free ----
def test_cube_single_key():   _run("SELECT a, COUNT(*), SUM(m1) FROM t GROUP BY a", 'cube')
def test_cube_two_key():      _run("SELECT a, b, COUNT(*), SUM(m1), AVG(m2) FROM t GROUP BY a, b", 'cube')
def test_cube_grouped_distinct(): _run("SELECT a, COUNT(DISTINCT b) FROM t GROUP BY a", 'cube')  # [a,b] cube

# ---- grouped_multi: high-card 2-key (cube over cap -> single-pass scatter, NOT a per-range loop) ----
def test_highcard_two_key_grouped_multi():
    _run("SELECT a, c, COUNT(*), SUM(m1) FROM t GROUP BY a, c", 'grouped_multi')
def test_highcard_two_key_avg():
    _run("SELECT a, c, AVG(m2) FROM t GROUP BY a, c", 'grouped_multi')

# ---- per-slice scalar: clustered single-key GROUP BY + WHERE (cube gated off by the predicate) ----
def test_slice_scalar_filtered_clusterkey():
    _run("SELECT a, COUNT(*), SUM(m1) FROM t WHERE m2 > 50 GROUP BY a", 'slice')

# ---- grouped_multi fallback: filtered 2-key (cube gated off by WHERE, single-key slice gate fails) ----
def test_filtered_two_key_grouped_multi():
    _run("SELECT a, b, COUNT(*), SUM(m1) FROM t WHERE m2 > 50 GROUP BY a, b", 'grouped_multi')

# ---- build-time gate: the over-cap cube is declined, low-card ones kept ----
def test_cube_cap_gate():
    db, _ = _fixture()
    dims = sorted(c['dims'] for c in Segment(db.cat.segment_paths('t')[0]).cubes())
    assert ['a'] in dims and ['a', 'b'] in dims and ['a', 'c'] not in dims

# ---- slice2 is gone: no two-key per-range kernel may exist to regress back to ----
def test_no_slice2_kernel():
    assert not hasattr(wdb_join, '_slice2_scalar_agg') and not hasattr(wdb_join, '_SLICE2_SCALAR_HITS')
