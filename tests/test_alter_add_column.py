"""ALTER TABLE ... ADD COLUMN end to end, against a DuckDB oracle (DuckDB backfills existing rows
with the DEFAULT, matching our synth-at-read for segments that predate the column).

Storage model: ADD COLUMN is metadata-only for buffered tables -- old cold segments synthesize the
default at read (engine mode-6 constant column), new inserts/flushes carry it for real, and
compaction materializes it. Segment-mode tables materialize eagerly into their canonical buffer."""
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
    d = os.path.join(tempfile.gettempdir(), f'addc_{uuid.uuid4().hex[:8]}')
    db = Database.create(d); db.run(create_sql)
    if buffered: db.set_table_mode('t', 'buffered')
    for i, r in enumerate(rows):
        db.run(f"INSERT INTO t VALUES ({', '.join(_lit(v) for v in r)})")
        if buffered and flush_every and (i + 1) % flush_every == 0: db.flush('t')
    if buffered: db.flush('t')
    con = duckdb.connect(); con.execute(create_sql)
    con.executemany(f"INSERT INTO t VALUES ({', '.join(['?']*len(rows[0]))})", [list(r) for r in rows])
    return db, con, d

def _both(db, con, stmt):
    db.run(stmt); con.execute(stmt)

_TS = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(\.\d+)?$')
def _cell(c):
    if isinstance(c, datetime.datetime): return c.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(c, datetime.date):     return c.strftime('%Y-%m-%d')
    if isinstance(c, str):
        m = _TS.match(c)
        if m: return m.group(1)
    return c

def _norm(rows): return [tuple(_cell(c) for c in r) for r in rows]

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
    got, _ = db.run(q); exp = con.execute(q).fetchall()
    g = _norm(got); e = _norm([tuple(r) for r in exp])
    if not ordered:
        key = lambda t: tuple((x is None, str(x)) for x in t)
        g = sorted(g, key=key); e = sorted(e, key=key)
    assert len(g) == len(e), f"{q}\n got({len(g)})={got}\n exp({len(e)})={exp}"
    for gr, er in zip(g, e):
        assert _eq_row(gr, er), f"{q}\n got={gr}\n exp={er}"

# ── basics: default backfill on old rows, both storage modes ─────────────────

def test_add_int_default_segment_mode():
    rows = [(i, i * 10) for i in range(6)]
    db, con, d = _build("CREATE TABLE t (id INT, v INT)", rows)
    _both(db, con, "ALTER TABLE t ADD COLUMN flag INT DEFAULT 7")
    _both(db, con, "INSERT INTO t (id, v, flag) VALUES (6, 60, 99)")
    _both(db, con, "INSERT INTO t (id, v) VALUES (7, 70)")          # omitted -> default
    for q in ["SELECT id, flag FROM t", "SELECT flag, COUNT(*) FROM t GROUP BY flag",
              "SELECT SUM(flag), AVG(flag), MIN(flag), MAX(flag) FROM t",
              "SELECT id FROM t WHERE flag = 7", "SELECT COUNT(*) FROM t WHERE flag <> 7"]:
        _match(db, con, q)
    shutil.rmtree(d)

def test_add_int_default_buffered_cold_synth():
    rows = [(i, i) for i in range(8)]
    db, con, d = _build("CREATE TABLE t (id INT, v INT)", rows, buffered=True)  # one cold segment
    seg = Segment(db.cat.segment_paths('t')[0])
    assert 'flag' not in seg.cols                                   # predates the column
    _both(db, con, "ALTER TABLE t ADD COLUMN flag INT DEFAULT 3")
    for q in ["SELECT id, flag FROM t", "SELECT flag, COUNT(*), SUM(v) FROM t GROUP BY flag",
              "SELECT COUNT(*) FROM t WHERE flag = 3"]:
        _match(db, con, q)
    # new rows in the hot buffer carry the column for real
    for i in range(8, 12): _both(db, con, f"INSERT INTO t (id, v, flag) VALUES ({i}, {i}, {i})")
    _match(db, con, "SELECT flag, COUNT(*) FROM t GROUP BY flag")
    shutil.rmtree(d)

def test_add_no_default_is_null():
    rows = [(i,) for i in range(5)]
    db, con, d = _build("CREATE TABLE t (id INT)", rows)
    _both(db, con, "ALTER TABLE t ADD COLUMN x INT")
    for q in ["SELECT id, x FROM t", "SELECT COUNT(*) FROM t WHERE x IS NULL",
              "SELECT COUNT(x), SUM(x) FROM t"]:                    # aggregates ignore NULLs
        _match(db, con, q)
    shutil.rmtree(d)

def test_add_string_and_float_defaults():
    rows = [(i,) for i in range(5)]
    db, con, d = _build("CREATE TABLE t (id INT)", rows)
    _both(db, con, "ALTER TABLE t ADD COLUMN name VARCHAR DEFAULT 'anon'")
    _both(db, con, "ALTER TABLE t ADD COLUMN ratio DOUBLE DEFAULT 1.5")
    _match(db, con, "SELECT id, name, ratio FROM t")
    _match(db, con, "SELECT name, COUNT(*) FROM t GROUP BY name")
    _match(db, con, "SELECT SUM(ratio) FROM t")
    shutil.rmtree(d)

# ── DELETE / UPDATE referencing the new column ───────────────────────────────

def test_delete_on_added_column_buffered():
    rows = [(i, i) for i in range(10)]
    db, con, d = _build("CREATE TABLE t (id INT, v INT)", rows, buffered=True)
    _both(db, con, "ALTER TABLE t ADD COLUMN g INT DEFAULT 0")
    _both(db, con, "INSERT INTO t (id, v, g) VALUES (10, 10, 1)")
    _both(db, con, "DELETE FROM t WHERE g = 0")                     # removes all old (synth) rows
    _match(db, con, "SELECT id, g FROM t")
    _match(db, con, "SELECT COUNT(*) FROM t")
    shutil.rmtree(d)

def test_update_added_column_on_cold_segment():
    rows = [(i, i) for i in range(8)]
    db, con, d = _build("CREATE TABLE t (id INT, v INT)", rows, buffered=True)
    _both(db, con, "ALTER TABLE t ADD COLUMN score INT DEFAULT 100")
    _both(db, con, "UPDATE t SET score = 5 WHERE id < 3")          # override on synth column
    for q in ["SELECT id, score FROM t", "SELECT score, COUNT(*) FROM t GROUP BY score",
              "SELECT SUM(score) FROM t"]:
        _match(db, con, q)
    shutil.rmtree(d)

def test_update_added_column_arithmetic():
    rows = [(i, i) for i in range(6)]
    db, con, d = _build("CREATE TABLE t (id INT, v INT)", rows, buffered=True)
    _both(db, con, "ALTER TABLE t ADD COLUMN base INT DEFAULT 10")
    _both(db, con, "UPDATE t SET base = base + v WHERE id >= 2")    # synth col in arithmetic RHS
    _match(db, con, "SELECT id, base FROM t ORDER BY id", ordered=True)
    shutil.rmtree(d)

# ── compaction materializes the default ──────────────────────────────────────

def test_compaction_materializes_default():
    rows = [(i, i) for i in range(40)]
    db, con, d = _build("CREATE TABLE t (id INT, v INT)", rows, buffered=True, flush_every=10)
    assert len(db.cat.segment_paths('t')) >= 3
    _both(db, con, "ALTER TABLE t ADD COLUMN tag INT DEFAULT 8")
    _both(db, con, "UPDATE t SET tag = 1 WHERE id = 5")            # override on a synth column
    for i in range(40, 45): _both(db, con, f"INSERT INTO t (id, v, tag) VALUES ({i}, {i}, 2)")
    db.flush('t')
    db.compact('t')                                                # merge all cold segments
    paths = db.cat.segment_paths('t'); assert len(paths) == 1
    merged = Segment(paths[0])
    assert 'tag' in merged.cols and merged.cols['tag']['mode'] != 6  # now physically present
    for q in ["SELECT tag, COUNT(*) FROM t GROUP BY tag", "SELECT id, tag FROM t WHERE id IN (5, 42)",
              "SELECT SUM(tag) FROM t"]:
        _match(db, con, q)
    shutil.rmtree(d)

# ── composition: ADD then RENAME, two ADDs, ADD on multi-segment ─────────────

def test_add_then_rename_added_column():
    rows = [(i,) for i in range(5)]
    db, con, d = _build("CREATE TABLE t (id INT)", rows)
    _both(db, con, "ALTER TABLE t ADD COLUMN temp INT DEFAULT 4")
    _both(db, con, "ALTER TABLE t RENAME COLUMN temp TO permanent")
    _match(db, con, "SELECT id, permanent FROM t")
    try: db.run("SELECT temp FROM t"); assert False
    except NotImplementedError: pass
    shutil.rmtree(d)

def test_two_added_columns():
    rows = [(i,) for i in range(4)]
    db, con, d = _build("CREATE TABLE t (id INT)", rows)
    _both(db, con, "ALTER TABLE t ADD COLUMN a INT DEFAULT 1")
    _both(db, con, "ALTER TABLE t ADD COLUMN b INT DEFAULT 2")
    _match(db, con, "SELECT id, a, b FROM t")
    _match(db, con, "SELECT a, b, COUNT(*) FROM t GROUP BY a, b")
    shutil.rmtree(d)

def test_add_column_multisegment_groupby():
    rows = [(i, i % 5) for i in range(60)]
    db, con, d = _build("CREATE TABLE t (id INT, k INT)", rows, buffered=True, flush_every=20)
    assert len(db.cat.segment_paths('t')) >= 3
    _both(db, con, "ALTER TABLE t ADD COLUMN w INT DEFAULT 1")
    for i in range(60, 70): _both(db, con, f"INSERT INTO t (id, k, w) VALUES ({i}, {i%5}, 2)")
    _match(db, con, "SELECT w, COUNT(*), SUM(k) FROM t GROUP BY w")
    _match(db, con, "SELECT k, w, COUNT(*) FROM t GROUP BY k, w")
    shutil.rmtree(d)

# ── default value used in WHERE and combined predicates ──────────────────────

def test_added_column_in_compound_where():
    rows = [(i, i) for i in range(20)]
    db, con, d = _build("CREATE TABLE t (id INT, v INT)", rows, buffered=True)
    _both(db, con, "ALTER TABLE t ADD COLUMN active INT DEFAULT 1")
    _both(db, con, "INSERT INTO t (id, v, active) VALUES (20, 20, 0)")
    _match(db, con, "SELECT id FROM t WHERE active = 1 AND v > 15")
    _match(db, con, "SELECT COUNT(*) FROM t WHERE active = 0 OR v < 3")
    shutil.rmtree(d)
