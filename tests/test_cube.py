"""Tests for the materialised low-card GROUP BY cube (wdb_cube).
The cube is a precomputed aggregate: a filter-free low-card GROUP BY with COUNT/SUM/AVG is answered
from a few stored numbers instead of a scan. It must (a) fire and match DuckDB, (b) decline above the
cell cap, (c) gate off for WHERE / MIN-MAX / COUNT(DISTINCT) and fall back with identical results.
All synthetic + tiny: runs in the suite with no bench DB."""
import sys, os, tempfile, uuid, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from decimal import Decimal
import duckdb, wdb_encode, wdb_cube
from wdb_db import Database
from wdb_engine import Segment

_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), f'cube_{uuid.uuid4().hex[:8]}'); os.makedirs(d, exist_ok=True)
    # g: 6-value low card, s: 4-value; k: high-card (near unique); amt: float measure
    _CON.execute("CREATE TABLE t AS SELECT i AS id, 'G'||(i%6) AS g, 'S'||(i%4) AS s, i AS k, "
                 "CAST(i%8 AS INTEGER) AS q, "
                 "(DATE '1992-01-01' + CAST(i%40 AS INTEGER)) AS d, "
                 "((i%50)+1)*1.5 AS amt FROM range(30000) t(i)")
    _DB = Database.create(d)
    wt = {'BIGINT':'int','INTEGER':'int','VARCHAR':'string','DOUBLE':'float','DATE':'datetime'}
    desc = _CON.execute("DESCRIBE t").fetchall()
    sel = ", ".join((f"CAST({c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else c[0]) for c in desc)
    pq = os.path.join(d, 't.parquet'); _CON.execute(f"COPY (SELECT {sel} FROM t ORDER BY id) TO '{pq}' (FORMAT parquet)")
    _DB.cat.add_table('t', [[c[0], ('float' if c[1].startswith('DECIMAL') else wt[c[1]])] for c in desc])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'),
                      cubes=[['g'], ['g','s'], ['k'], ['d'], ['g','q']])  # ['k'] declined; ['d'] datetime(40); ['g','q'] locks range-filter
    _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON

def _norm(rows):
    return sorted(tuple(round(float(c),3) if isinstance(c,(float,Decimal)) else (None if c is None else
                  (int(c) if isinstance(c,int) and not isinstance(c,bool) else str(c))) for c in r) for r in rows)

def _run(q, expect_cube):
    db, con = _fixture()
    before = wdb_cube._CUBE_HITS
    g = _norm(db.run(q)[0]); e = _norm([tuple(r) for r in con.execute(q).fetchall()])
    fired = wdb_cube._CUBE_HITS == before + 1
    assert fired == expect_cube, f"cube fired={fired}, expected {expect_cube}: {q}"
    assert g == e, f"mismatch {q}\n got {g[:4]}\n exp {e[:4]}"

def test_cube_single_dim_sum():   _run("SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g", True)
def test_cube_datetime_dim():     _run("SELECT d, COUNT(*) FROM t GROUP BY d", True)        # dt=3 key path
def test_cube_datetime_sum():     _run("SELECT d, SUM(amt) FROM t GROUP BY d", True)        # date key + measure
def test_cube_single_dim_avg():   _run("SELECT g, AVG(amt) FROM t GROUP BY g", True)
def test_cube_two_dim_q1():       _run("SELECT g, s, COUNT(*), SUM(amt), AVG(amt) FROM t GROUP BY g, s", True)
def test_cube_order_limit():      _run("SELECT g, SUM(amt) z FROM t GROUP BY g ORDER BY z DESC, g LIMIT 2", True)
def test_cube_having():           _run("SELECT g, COUNT(*) c FROM t GROUP BY g HAVING COUNT(*) > 4000", True)
# gated off -> must fall back (cube must NOT fire), still correct
def test_cube_where_falls_back(): _run("SELECT g, SUM(amt) FROM t WHERE k > 100 GROUP BY g", False)
def test_cube_minmax_falls_back():_run("SELECT g, MIN(amt), MAX(amt) FROM t GROUP BY g", False)
# grouped COUNT(DISTINCT col2) GROUP BY col1 -> answered from the [col1,col2] cube (set-matched)
def test_cube_grouped_distinct():     _run("SELECT g, COUNT(DISTINCT s) FROM t GROUP BY g", True)
def test_cube_grouped_distinct_rev(): _run("SELECT s, COUNT(DISTINCT g) FROM t GROUP BY s", True)
# COUNT(DISTINCT col2) with NO [col1,col2] cube (high-card col2, never built) -> must fall back
def test_cube_grouped_distinct_no_cube(): _run("SELECT g, COUNT(DISTINCT k) FROM t GROUP BY g", False)
def test_cube_highcard_no_match(): _run("SELECT k, COUNT(*) FROM t GROUP BY k", False)  # ['k'] cube declined by cap

def test_cube_cap_declines_highcard():
    db, _ = _fixture()
    seg = Segment(os.path.join(db.cat.dir if hasattr(db.cat,'dir') else os.path.dirname(
        db.cat.segment_paths('t')[0]), 't_0.wdb')) if False else Segment(db.cat.segment_paths('t')[0])
    dims = sorted(c['dims'] for c in seg.cubes())
    assert ['g'] in dims and ['g','s'] in dims and ['k'] not in dims  # cap kept low-card, dropped k

def test_cube_build_returns_none_over_cap():
    db, _ = _fixture()
    seg = Segment(db.cat.segment_paths('t')[0])
    assert wdb_cube.build_cube(seg, ['k']) is None              # 30000 cells >> 1024 cap
    assert wdb_cube.build_cube(seg, ['g','s']) is not None      # 24 cells, fine


# --- range filter on a cube dimension: WHERE fcol <op> const GROUP BY g -> exact roll-up from [g,fcol] cube ---
# fcol is per-value in the cube, so the predicate selects whole cells (no straddling residual): EXACT, not approximate.
def test_cube_range_gt():          _run("SELECT g, SUM(amt) FROM t WHERE q > 5 GROUP BY g", True)
def test_cube_range_gte_count():   _run("SELECT g, COUNT(*) FROM t WHERE q >= 5 GROUP BY g", True)
def test_cube_range_lt_sum():      _run("SELECT g, SUM(amt) FROM t WHERE q < 3 GROUP BY g", True)
def test_cube_range_between():     _run("SELECT g, SUM(amt), COUNT(*) FROM t WHERE q BETWEEN 2 AND 5 GROUP BY g", True)
def test_cube_range_in():          _run("SELECT g, SUM(amt) FROM t WHERE q IN (1,3,5) GROUP BY g", True)
def test_cube_range_neq_avg():     _run("SELECT g, AVG(amt) FROM t WHERE q != 0 GROUP BY g", True)
def test_cube_range_string_dim():  _run("SELECT g, SUM(amt) FROM t WHERE s >= 'S2' GROUP BY g", True)   # predicate on string dim
def test_cube_range_order_limit(): _run("SELECT g, SUM(amt) z FROM t WHERE q > 3 GROUP BY g ORDER BY z DESC, g LIMIT 2", True)
def test_cube_range_having():      _run("SELECT g, COUNT(*) c FROM t WHERE q > 1 GROUP BY g HAVING COUNT(*) > 2000", True)
# multi-predicate AND -> _make_pred returns None -> cube must NOT fire, scan answers (still correct)
def test_cube_range_and_fallback(): _run("SELECT g, SUM(amt) FROM t WHERE q > 2 AND g = 'G1' GROUP BY g", False)


# --- cubes='auto': exhaustive enumeration from TRUE per-column cardinality ---
def test_cube_auto_true_cardinality():
    # auto must use TRUE distinct counts, not header V: mode-4 affine columns (small synthetic int/date)
    # store N, not cardinality. If auto trusted V it would miss the (g,q) cube. Lock the fix.
    import tempfile, uuid
    con = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), f'cubeauto_{uuid.uuid4().hex[:8]}'); os.makedirs(d, exist_ok=True)
    con.execute("CREATE TABLE t AS SELECT i AS id, 'G'||(i%5) AS g, CAST(i%8 AS INTEGER) AS q, "
                "CAST(((i%30)+1)*1.0 AS DOUBLE) AS amt FROM range(20000) t(i)")
    db = Database.create(d)
    wt = {'BIGINT':'int','INTEGER':'int','VARCHAR':'string','DOUBLE':'float'}
    desc = con.execute("DESCRIBE t").fetchall()
    pq = os.path.join(d,'t.parquet'); con.execute(f"COPY (SELECT * FROM t ORDER BY id) TO '{pq}' (FORMAT parquet)")
    db.cat.add_table('t', [[c[0], wt[c[1]]] for c in desc])
    wdb_encode.encode(pq, os.path.join(d,'t_0.wdb'), cubes='auto')
    db.cat.add_segment('t','t_0.wdb')
    seg = Segment(db.cat.segment_paths('t')[0])
    cards = wdb_cube.segment_cardinalities(seg)
    assert cards['q'] == 8 and cards['g'] == 5, f"true card wrong (mode-4 V leak?): {cards}"   # not 20000
    built = set(frozenset(c['dims']) for c in seg.cubes())
    assert built == set(frozenset(s) for s in wdb_cube.enumerate_cube_specs(cards)), "auto set != enumerator"
    assert frozenset(['g','q']) in built, f"expected (g,q) cube, got {built}"
    before = wdb_cube._CUBE_HITS
    qy = "SELECT g, SUM(amt) FROM t WHERE q > 4 GROUP BY g"
    got = _norm(db.run(qy)[0]); exp = _norm([tuple(r) for r in con.execute(qy).fetchall()])
    assert wdb_cube._CUBE_HITS == before + 1, "auto (g,q) cube did not fire on range filter"
    assert got == exp, f"auto cube range-filter mismatch\n got {got[:4]}\n exp {exp[:4]}"
