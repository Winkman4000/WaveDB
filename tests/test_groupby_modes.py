"""GROUP BY correctness across storage modes, with a DuckDB oracle.

Regression net for the mode-4 (affine/sequence) GROUP BY bug: the engine used to group on
seg.codes(), but mode-4 codes are arange(N) (identity-per-row), so every row became its own group.
The fix factorizes mode-4 *values* for the key while keeping codes for dict/inline modes (which are
already value-identity). These tests exercise affine, constant, wrapping, sorted-repeat, datetime,
multi-column, override, delete, HAVING, ORDER/LIMIT, rename, and multi-segment paths, and assert the
intended storage mode so we know the bug path is actually covered."""
import sys, os, tempfile, uuid, shutil, math, datetime, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from wdb_db import Database
from wdb_engine import Segment
import duckdb

def _lit(v):
    if v is None: return "NULL"
    if isinstance(v, str): return "'" + v.replace("'", "''") + "'"
    return str(v)

def _build(create_sql, rows, buffered=False, flush_every=None):
    """Build a WaveDB table 't' and a DuckDB oracle 't' from identical data. flush_every>0 (with
    buffered) produces MULTIPLE cold segments (exercises the merge path)."""
    d = os.path.join(tempfile.gettempdir(), f'gb_{uuid.uuid4().hex[:8]}')
    db = Database.create(d); db.run(create_sql)
    if buffered: db.set_table_mode('t', 'buffered')
    for i, r in enumerate(rows):
        db.run(f"INSERT INTO t VALUES ({', '.join(_lit(v) for v in r)})")
        if buffered and flush_every and (i + 1) % flush_every == 0: db.flush('t')
    if buffered: db.flush('t')
    con = duckdb.connect(); con.execute(create_sql)
    con.executemany(f"INSERT INTO t VALUES ({', '.join(['?']*len(rows[0]))})", [list(r) for r in rows])
    return db, con, d

_TS_RE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(\.\d+)?$')
def _cell(c):
    if isinstance(c, datetime.datetime): return c.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(c, datetime.date):     return c.strftime('%Y-%m-%d')
    if isinstance(c, str):
        m = _TS_RE.match(c)
        if m: return m.group(1)            # strip fractional seconds for cross-engine compare
    return c

def _norm(rows):
    return [tuple(_cell(c) for c in r) for r in rows]

def _eq_row(g, e):
    if len(g) != len(e): return False
    for a, b in zip(g, e):
        if isinstance(a, float) or isinstance(b, float):
            if a is None or b is None: return a is b
            if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9): return False
        elif a != b:
            return False
    return True

def _match(db, con, q, ordered=False):
    got, _ = db.run(q)
    exp = con.execute(q).fetchall()
    g = _norm(got); e = _norm([tuple(r) for r in exp])
    if not ordered:
        key = lambda t: tuple((x is None, str(x)) for x in t)
        g = sorted(g, key=key); e = sorted(e, key=key)
    assert len(g) == len(e), f"{q}\n got({len(g)})={got}\n exp({len(e)})={exp}"
    for gr, er in zip(g, e):
        assert _eq_row(gr, er), f"{q}\n got={gr}\n exp={er}"

def _mode(db, col, seg=0):
    return Segment(db.cat.segment_paths('t')[seg]).cols[col]['mode']

# ── mode-4 (affine) group keys ───────────────────────────────────────────────

def test_mode4_wrapping_all_aggs():
    rows = [(i, i % 7, i * 3) for i in range(40)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, b INT)", rows)
    assert _mode(db, 'a') == 4 and _mode(db, 'id') == 4 and _mode(db, 'b') == 4
    for q in ["SELECT a, COUNT(*) FROM t GROUP BY a",
              "SELECT a, COUNT(b), SUM(b), MIN(b), MAX(b) FROM t GROUP BY a",
              "SELECT a, AVG(b) FROM t GROUP BY a",
              "SELECT a, COUNT(*) FROM t WHERE b > 30 GROUP BY a"]:
        _match(db, con, q)
    shutil.rmtree(d)

def test_mode4_clean_affine_no_collapse():
    # a clean affine column: every value distinct -> N singleton groups (must NOT be merged,
    # the inverse failure mode of the bug).
    rows = [(i, i * 2) for i in range(50)]
    db, con, d = _build("CREATE TABLE t (id INT, v INT)", rows)
    assert _mode(db, 'id') == 4
    got, _ = db.run("SELECT id, COUNT(*) FROM t GROUP BY id")
    assert len(got) == 50 and all(c == 1 for _, c in got)
    _match(db, con, "SELECT id, SUM(v) FROM t GROUP BY id")
    shutil.rmtree(d)

def test_mode4_constant_one_group():
    rows = [(i, 5) for i in range(30)]
    db, con, d = _build("CREATE TABLE t (id INT, k INT)", rows)
    assert _mode(db, 'k') == 4                      # constant -> mode-4 stride 0
    _match(db, con, "SELECT k, COUNT(*), SUM(id) FROM t GROUP BY k")
    assert db.run("SELECT k, COUNT(*) FROM t GROUP BY k")[0] == [(5, 30)]
    shutil.rmtree(d)

def test_mode4_sorted_repeats():
    rows = [(i, v) for i, v in enumerate(sorted([i % 4 for i in range(60)]))]
    db, con, d = _build("CREATE TABLE t (id INT, g INT)", rows)
    assert _mode(db, 'g') == 4
    _match(db, con, "SELECT g, COUNT(*), SUM(id), AVG(id) FROM t GROUP BY g")
    shutil.rmtree(d)

# ── multi-column GROUP BY ────────────────────────────────────────────────────

def test_multicol_mode4_plus_dict():
    import random; random.seed(5)
    rows = [(i, i % 5, random.choice(['x', 'y', 'z'])) for i in range(80)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, s VARCHAR)", rows)
    assert _mode(db, 'a') == 4 and _mode(db, 's') == 0
    _match(db, con, "SELECT a, s, COUNT(*) FROM t GROUP BY a, s")
    _match(db, con, "SELECT s, a, COUNT(*), SUM(id) FROM t GROUP BY s, a")
    shutil.rmtree(d)

def test_multicol_mode4_plus_mode4():
    rows = [(i, i % 6, i % 7) for i in range(84)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, b INT)", rows)
    assert _mode(db, 'a') == 4 and _mode(db, 'b') == 4
    _match(db, con, "SELECT a, b, COUNT(*) FROM t GROUP BY a, b")
    _match(db, con, "SELECT a, b, SUM(id) FROM t GROUP BY a, b")
    shutil.rmtree(d)

# ── overrides (UPDATE) and deletes (presence) under GROUP BY ─────────────────

def test_mode4_groupby_after_update():
    rows = [(i, i % 6) for i in range(60)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT)", rows, buffered=True)
    assert _mode(db, 'a') == 4
    for stmt in ["UPDATE t SET a = 0 WHERE id < 10", "UPDATE t SET a = 99 WHERE id = 55"]:
        db.run(stmt); con.execute(stmt)
    _match(db, con, "SELECT a, COUNT(*), SUM(id) FROM t GROUP BY a")
    shutil.rmtree(d)

def test_mode4_groupby_after_delete():
    rows = [(i, i % 5) for i in range(50)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT)", rows, buffered=True)
    assert _mode(db, 'a') == 4
    db.run("DELETE FROM t WHERE a = 2"); con.execute("DELETE FROM t WHERE a = 2")
    db.run("DELETE FROM t WHERE id = 7"); con.execute("DELETE FROM t WHERE id = 7")
    _match(db, con, "SELECT a, COUNT(*) FROM t GROUP BY a")
    shutil.rmtree(d)

# ── HAVING, ORDER BY, LIMIT on a mode-4 grouping ─────────────────────────────

def test_mode4_having():
    rows = [(i, i % 8) for i in range(64)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT)", rows)
    _match(db, con, "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 7")
    shutil.rmtree(d)

def test_mode4_order_limit():
    rows = [(i, i % 9, i) for i in range(90)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, v INT)", rows)
    _match(db, con, "SELECT a, SUM(v) AS sv FROM t GROUP BY a ORDER BY sv DESC LIMIT 3", ordered=True)
    _match(db, con, "SELECT a, COUNT(*) FROM t GROUP BY a ORDER BY a DESC", ordered=True)
    shutil.rmtree(d)

# ── datetime mode-4 keys (non-midnight so rendering matches the oracle) ──────

def test_mode4_datetime_keys():
    base = datetime.datetime(2024, 1, 1, 0, 0, 1)
    rows = [(i, (base + datetime.timedelta(seconds=i)).strftime('%Y-%m-%d %H:%M:%S')) for i in range(20)]
    db, con, d = _build("CREATE TABLE t (id INT, ts TIMESTAMP)", rows)
    assert _mode(db, 'ts') == 4                     # sequential timestamps -> affine
    got, _ = db.run("SELECT ts, COUNT(*) FROM t GROUP BY ts")
    assert len(got) == 20 and all(c == 1 for _, c in got)   # distinct -> no collapse
    _match(db, con, "SELECT ts, COUNT(*) FROM t GROUP BY ts")
    shutil.rmtree(d)

# ── mode-5 inline strings already group correctly (codes are factorized) ─────

def test_mode5_string_groupby():
    rng = __import__('random').Random(9)
    cats = ['alpha', 'beta', 'gamma', 'delta']
    rows = [(i, rng.choice(cats), i) for i in range(120)]
    db, con, d = _build("CREATE TABLE t (id INT, c VARCHAR, v INT)", rows)
    # force mode-5? low-card strings usually stay dict; just assert correctness regardless of mode.
    _match(db, con, "SELECT c, COUNT(*), SUM(v) FROM t GROUP BY c")
    shutil.rmtree(d)

# ── dict-mode regression (the path that always worked) ───────────────────────

def test_dict_mode0_regression():
    rng = __import__('random').Random(2)
    rows = [(i, rng.randint(0, 3)) for i in range(60)]
    db, con, d = _build("CREATE TABLE t (id INT, g INT)", rows)
    assert _mode(db, 'g') == 0
    _match(db, con, "SELECT g, COUNT(*), SUM(id), AVG(id), MIN(id), MAX(id) FROM t GROUP BY g")
    shutil.rmtree(d)

# ── aggregating a NULLable column grouped by mode-4 (nulls ignored in aggs) ──

def test_mode4_group_agg_skips_nulls():
    rows = [(i, i % 6, (None if i % 3 == 0 else i)) for i in range(48)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, v INT)", rows)
    assert _mode(db, 'a') == 4
    _match(db, con, "SELECT a, COUNT(*), COUNT(v), SUM(v), AVG(v) FROM t GROUP BY a")
    shutil.rmtree(d)

# ── ties to ALTER: GROUP BY a renamed mode-4 column ──────────────────────────

def test_mode4_groupby_renamed_column():
    rows = [(i, i % 6, i) for i in range(60)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, v INT)", rows)
    assert _mode(db, 'a') == 4
    db.run("ALTER TABLE t RENAME COLUMN a TO bucket")
    con.execute("ALTER TABLE t RENAME COLUMN a TO bucket")
    _match(db, con, "SELECT bucket, COUNT(*), SUM(v) FROM t GROUP BY bucket")
    shutil.rmtree(d)

# ── multi-segment (merge path): each cold segment partial-aggregates via our engine ──

def test_mode4_groupby_multisegment():
    rows = [(i, i % 5, i) for i in range(100)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, v INT)", rows, buffered=True, flush_every=25)
    assert len(db.cat.segment_paths('t')) >= 3      # several cold segments
    assert _mode(db, 'a', seg=0) == 4
    _match(db, con, "SELECT a, COUNT(*), SUM(v), MIN(v), MAX(v) FROM t GROUP BY a")
    _match(db, con, "SELECT a, AVG(v) FROM t GROUP BY a")
    shutil.rmtree(d)

# ── mixed: hot buffer + cold segments under GROUP BY on mode-4 ───────────────

def test_mode4_groupby_hot_plus_cold():
    rows = [(i, i % 4, i) for i in range(40)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, v INT)", rows, buffered=True)
    # add more rows that stay in the hot buffer (no flush)
    for i in range(40, 60):
        db.run(f"INSERT INTO t VALUES ({i}, {i % 4}, {i})")
        con.execute(f"INSERT INTO t VALUES ({i}, {i % 4}, {i})")
    _match(db, con, "SELECT a, COUNT(*), SUM(v) FROM t GROUP BY a")
    shutil.rmtree(d)
