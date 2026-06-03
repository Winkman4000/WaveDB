"""ALTER TABLE ... DROP COLUMN, against a DuckDB oracle (DuckDB supports DROP COLUMN and ADD COLUMN
... DEFAULT, so it mirrors our semantics including drop-then-re-add backfill).

Storage model: DROP COLUMN removes the column from the LOGICAL schema. Strict col_map then makes it
unqueryable (referencing it errors). Segment-mode tables shed it eagerly from the canonical buffer;
buffered tables leave the bytes dead in cold segments until compaction reclaims them (compaction
rebuilds from the logical schema). Re-adding a dropped name gets a fresh physical name so it reads
its DEFAULT, never the dropped column's stale bytes."""
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
    d = os.path.join(tempfile.gettempdir(), f'drpc_{uuid.uuid4().hex[:8]}')
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

def _has_col(db, pcol, seg=0):
    return pcol in Segment(db.cat.segment_paths('t')[seg]).cols

# ── basic drop, both storage modes ───────────────────────────────────────────

def test_drop_segment_mode_sheds_column():
    rows = [(i, i * 2, i * 3) for i in range(6)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, b INT)", rows)
    assert _has_col(db, 'b')
    _both(db, con, "ALTER TABLE t DROP COLUMN b")
    assert not _has_col(db, 'b')                       # eagerly shed from the single segment
    _match(db, con, "SELECT id, a FROM t")
    _match(db, con, "SELECT a, COUNT(*), SUM(id) FROM t GROUP BY a")
    shutil.rmtree(d)

def test_drop_buffered_metadata_only_then_compaction_reclaims():
    rows = [(i, i, i * 5) for i in range(40)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, b INT)", rows, buffered=True, flush_every=10)
    assert len(db.cat.segment_paths('t')) >= 3 and _has_col(db, 'b')
    _both(db, con, "ALTER TABLE t DROP COLUMN b")
    assert _has_col(db, 'b')                           # buffered: bytes still present (dead)
    _match(db, con, "SELECT id, a FROM t")
    db.compact('t')
    assert len(db.cat.segment_paths('t')) == 1 and not _has_col(db, 'b')   # reclaimed
    _match(db, con, "SELECT a, COUNT(*) FROM t GROUP BY a")
    shutil.rmtree(d)

# ── the dropped column becomes unqueryable ───────────────────────────────────

def test_dropped_column_errors():
    rows = [(i, i, i) for i in range(4)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, b INT)", rows)
    db.run("ALTER TABLE t DROP COLUMN b")
    for q in ["SELECT b FROM t", "SELECT id FROM t WHERE b = 1", "SELECT b, COUNT(*) FROM t GROUP BY b"]:
        try: db.run(q); assert False, f"expected error for {q}"
        except NotImplementedError: pass
    shutil.rmtree(d)

# ── INSERT after drop (must not supply the dropped column) ────────────────────

def test_insert_after_drop():
    rows = [(i, i, i) for i in range(5)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, b INT)", rows)
    _both(db, con, "ALTER TABLE t DROP COLUMN b")
    _both(db, con, "INSERT INTO t (id, a) VALUES (5, 50)")
    _both(db, con, "INSERT INTO t VALUES (6, 60)")     # positional now matches the 2-col schema
    _match(db, con, "SELECT id, a FROM t")
    shutil.rmtree(d)

# ── refuse dropping the last column ──────────────────────────────────────────

def test_cannot_drop_last_column():
    rows = [(i,) for i in range(3)]
    db, con, d = _build("CREATE TABLE t (solo INT)", rows)
    try: db.run("ALTER TABLE t DROP COLUMN solo"); assert False
    except ValueError: pass
    _match(db, con, "SELECT solo FROM t")              # table intact
    shutil.rmtree(d)

# ── dropping different positions, several columns ────────────────────────────

def test_drop_first_middle_last_of_many():
    rows = [(i, i + 1, i + 2, i + 3) for i in range(6)]
    db, con, d = _build("CREATE TABLE t (a INT, b INT, c INT, e INT)", rows)
    _both(db, con, "ALTER TABLE t DROP COLUMN a")      # first
    _both(db, con, "ALTER TABLE t DROP COLUMN c")      # middle (of remaining)
    _match(db, con, "SELECT b, e FROM t")
    _both(db, con, "ALTER TABLE t DROP COLUMN e")      # last (of remaining)
    _match(db, con, "SELECT b FROM t")
    _match(db, con, "SELECT b, COUNT(*) FROM t GROUP BY b")
    shutil.rmtree(d)

# ── drop then re-add the same name: must read DEFAULT, not stale bytes ────────

def test_drop_then_readd_same_name_buffered():
    rows = [(i, i, i * 100) for i in range(8)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, b INT)", rows, buffered=True)
    _both(db, con, "ALTER TABLE t DROP COLUMN b")
    _both(db, con, "ALTER TABLE t ADD COLUMN b INT DEFAULT 0")
    assert db.cat.phys_map('t').get('b') != 'b'        # aliased to dodge stale bytes
    _match(db, con, "SELECT id, b FROM t")             # all 0, not i*100
    _both(db, con, "INSERT INTO t (id, a, b) VALUES (8, 8, 77)")
    _match(db, con, "SELECT b, COUNT(*) FROM t GROUP BY b")
    # and it survives compaction
    db.compact('t'); _match(db, con, "SELECT id, b FROM t")
    shutil.rmtree(d)

def test_drop_then_readd_same_name_segment():
    rows = [(i, i * 100) for i in range(5)]
    db, con, d = _build("CREATE TABLE t (id INT, b INT)", rows)
    _both(db, con, "ALTER TABLE t DROP COLUMN b")
    _both(db, con, "ALTER TABLE t ADD COLUMN b INT DEFAULT 9")
    _match(db, con, "SELECT id, b FROM t")             # all 9 (eager buffer clean removed old bytes)
    shutil.rmtree(d)

# ── compose with ADD and RENAME ──────────────────────────────────────────────

def test_add_then_drop_then_compaction():
    rows = [(i, i) for i in range(30)]
    db, con, d = _build("CREATE TABLE t (id INT, v INT)", rows, buffered=True, flush_every=10)
    _both(db, con, "ALTER TABLE t ADD COLUMN tmp INT DEFAULT 5")
    _both(db, con, "ALTER TABLE t DROP COLUMN tmp")    # added then removed -> never materialized
    db.compact('t')
    assert not _has_col(db, 'tmp')
    _match(db, con, "SELECT id, v FROM t")
    try: db.run("SELECT tmp FROM t"); assert False
    except NotImplementedError: pass
    shutil.rmtree(d)

def test_drop_a_renamed_column():
    rows = [(i, i, i) for i in range(6)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, b INT)", rows, buffered=True)
    _both(db, con, "ALTER TABLE t RENAME COLUMN a TO alpha")
    _both(db, con, "ALTER TABLE t DROP COLUMN alpha")
    _match(db, con, "SELECT id, b FROM t")
    for q in ["SELECT alpha FROM t", "SELECT a FROM t"]:
        try: db.run(q); assert False
        except NotImplementedError: pass
    shutil.rmtree(d)

# ── remaining columns keep working under mutation after a drop ───────────────

def test_delete_update_after_drop():
    rows = [(i, i, i) for i in range(10)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, b INT)", rows, buffered=True)
    _both(db, con, "ALTER TABLE t DROP COLUMN b")
    _both(db, con, "DELETE FROM t WHERE a < 3")
    _both(db, con, "UPDATE t SET a = a * 10 WHERE id >= 7")
    _match(db, con, "SELECT id, a FROM t ORDER BY id", ordered=True)
    _match(db, con, "SELECT COUNT(*), SUM(a) FROM t")
    shutil.rmtree(d)

# ── multi-segment GROUP BY/aggregate after drop ──────────────────────────────

def test_drop_multisegment_group_aggs():
    rows = [(i, i % 5, i * 2) for i in range(60)]
    db, con, d = _build("CREATE TABLE t (id INT, k INT, junk INT)", rows, buffered=True, flush_every=20)
    _both(db, con, "ALTER TABLE t DROP COLUMN junk")
    _match(db, con, "SELECT k, COUNT(*), SUM(id), AVG(id), MIN(id), MAX(id) FROM t GROUP BY k")
    db.compact('t')
    _match(db, con, "SELECT k, COUNT(*) FROM t GROUP BY k")
    assert not _has_col(db, 'junk')
    shutil.rmtree(d)

# ── drop a column that carried a default (defaults map cleaned) ───────────────

def test_drop_column_with_default():
    rows = [(i,) for i in range(5)]
    db, con, d = _build("CREATE TABLE t (id INT)", rows)
    _both(db, con, "ALTER TABLE t ADD COLUMN flag INT DEFAULT 1")
    _both(db, con, "ALTER TABLE t DROP COLUMN flag")
    assert 'flag' not in db.cat.get_table('t').get('defaults', {})
    _match(db, con, "SELECT id FROM t")
    shutil.rmtree(d)
