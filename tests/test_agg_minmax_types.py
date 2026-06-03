"""MIN/MAX (and SUM/AVG) return values of the correct TYPE, against a DuckDB oracle.

Regression net for a silent wrong-answer: the aggregate path used to force every column to float64
before MIN/MAX, so MIN/MAX over a datetime column came back as a float epoch (and MIN/MAX over a
string column crashed). The fix keeps the native dtype for MIN/MAX (datetime stays a timestamp,
strings compare lexicographically) and only casts to float for SUM/AVG. The two-tier merge path
normalizes datetime partials so the cold tier's rendered string and DuckDB's datetime object compare
like-with-like."""
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
    d = os.path.join(tempfile.gettempdir(), f'agg_{uuid.uuid4().hex[:8]}')
    db = Database.create(d); db.run(create_sql)
    if buffered: db.set_table_mode('t', 'buffered')
    for i, r in enumerate(rows):
        db.run(f"INSERT INTO t VALUES ({', '.join(_lit(v) for v in r)})")
        if buffered and flush_every and (i + 1) % flush_every == 0: db.flush('t')
    if buffered: db.flush('t')
    con = duckdb.connect(); con.execute(create_sql)
    con.executemany(f"INSERT INTO t VALUES ({', '.join(['?']*len(rows[0]))})", [list(r) for r in rows])
    return db, con, d

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
    assert len(g) == len(e), f"{q}\n got={got}\n exp={exp}"
    for gr, er in zip(g, e):
        assert _eq_row(gr, er), f"{q}\n got={gr}\n exp={er}"

def _ts(i):
    base = datetime.datetime(2024, 3, 1, 8, 15, 30)
    return (base + datetime.timedelta(hours=7 * i + (i % 3))).strftime('%Y-%m-%d %H:%M:%S')

# ── datetime MIN/MAX (the flagged bug): scalar + grouped ─────────────────────

def test_minmax_datetime_scalar():
    rows = [(i, _ts(i)) for i in range(15)]
    db, con, d = _build("CREATE TABLE t (id INT, ts TIMESTAMP)", rows)
    _match(db, con, "SELECT MIN(ts), MAX(ts) FROM t")
    _match(db, con, "SELECT MIN(ts), MAX(ts) FROM t WHERE id >= 5")
    shutil.rmtree(d)

def test_minmax_datetime_grouped():
    rows = [(i, i % 4, _ts(i)) for i in range(24)]
    db, con, d = _build("CREATE TABLE t (id INT, g INT, ts TIMESTAMP)", rows)
    _match(db, con, "SELECT g, MIN(ts), MAX(ts), COUNT(*) FROM t GROUP BY g")
    shutil.rmtree(d)

def test_minmax_datetime_with_nulls():
    rows = [(i, (None if i % 4 == 0 else _ts(i))) for i in range(16)]
    db, con, d = _build("CREATE TABLE t (id INT, ts TIMESTAMP)", rows)
    _match(db, con, "SELECT MIN(ts), MAX(ts), COUNT(ts) FROM t")
    shutil.rmtree(d)

def test_minmax_datetime_affine_mode4():
    # sequential timestamps -> mode-4 affine column
    base = datetime.datetime(2024, 1, 1, 0, 0, 1)
    rows = [(i, (base + datetime.timedelta(seconds=i)).strftime('%Y-%m-%d %H:%M:%S')) for i in range(20)]
    db, con, d = _build("CREATE TABLE t (id INT, ts TIMESTAMP)", rows)
    assert Segment(db.cat.segment_paths('t')[0]).cols['ts']['mode'] == 4
    _match(db, con, "SELECT MIN(ts), MAX(ts) FROM t")
    shutil.rmtree(d)

# ── string MIN/MAX (used to crash on the float cast) ─────────────────────────

def test_minmax_string_scalar_and_grouped():
    names = ['mango', 'apple', 'pear', 'kiwi', 'fig', 'cherry']
    rows = [(i, names[i % len(names)], i % 3) for i in range(18)]
    db, con, d = _build("CREATE TABLE t (id INT, name VARCHAR, g INT)", rows)
    _match(db, con, "SELECT MIN(name), MAX(name) FROM t")
    _match(db, con, "SELECT g, MIN(name), MAX(name) FROM t GROUP BY g")
    shutil.rmtree(d)

def test_minmax_string_with_nulls():
    names = ['delta', 'alpha', 'charlie', 'bravo']
    rows = [(i, (None if i % 5 == 0 else names[i % 4])) for i in range(20)]
    db, con, d = _build("CREATE TABLE t (id INT, name VARCHAR)", rows)
    _match(db, con, "SELECT MIN(name), MAX(name), COUNT(name) FROM t")
    shutil.rmtree(d)

# ── numeric MIN/MAX/SUM/AVG regression (must stay correct) ───────────────────

def test_numeric_aggs_regression():
    rows = [(i, i * 3, i / 2.0, i % 5) for i in range(40)]
    db, con, d = _build("CREATE TABLE t (id INT, a INT, b DOUBLE, g INT)", rows)
    _match(db, con, "SELECT MIN(a), MAX(a), SUM(a), AVG(a) FROM t")
    _match(db, con, "SELECT MIN(b), MAX(b), SUM(b), AVG(b) FROM t")
    _match(db, con, "SELECT g, MIN(a), MAX(a), SUM(a), AVG(b), COUNT(*) FROM t GROUP BY g")
    shutil.rmtree(d)

# ── mixed aggregate types in one projection ──────────────────────────────────

def test_mixed_aggregate_types():
    rows = [(i, _ts(i), f"u{i % 3}", i * 2) for i in range(18)]
    db, con, d = _build("CREATE TABLE t (id INT, ts TIMESTAMP, name VARCHAR, v INT)", rows)
    _match(db, con, "SELECT MIN(ts), MAX(name), SUM(v), AVG(v), COUNT(*) FROM t")
    _match(db, con, "SELECT name, MIN(ts), MAX(v) FROM t GROUP BY name")
    shutil.rmtree(d)

# ── two-tier merge path: datetime / string MIN/MAX across cold + hot ─────────

def test_minmax_datetime_merge_hot_cold():
    rows = [(i, _ts(i)) for i in range(12)]
    db, con, d = _build("CREATE TABLE t (id INT, ts TIMESTAMP)", rows, buffered=True)  # 1 cold seg
    for i in range(12, 20):                                  # these stay in the hot buffer
        db.run(f"INSERT INTO t VALUES ({i}, '{_ts(i)}')"); con.execute(f"INSERT INTO t VALUES ({i}, '{_ts(i)}')")
    _match(db, con, "SELECT MIN(ts), MAX(ts) FROM t")
    shutil.rmtree(d)

def test_minmax_grouped_merge_multisegment():
    rows = [(i, i % 4, _ts(i)) for i in range(40)]
    db, con, d = _build("CREATE TABLE t (id INT, g INT, ts TIMESTAMP)", rows, buffered=True, flush_every=10)
    assert len(db.cat.segment_paths('t')) >= 3
    for i in range(40, 46):                                  # hot tier on top of several cold tiers
        db.run(f"INSERT INTO t VALUES ({i}, {i % 4}, '{_ts(i)}')")
        con.execute(f"INSERT INTO t VALUES ({i}, {i % 4}, '{_ts(i)}')")
    _match(db, con, "SELECT g, MIN(ts), MAX(ts), COUNT(*) FROM t GROUP BY g")
    shutil.rmtree(d)

def test_minmax_string_merge():
    names = ['xi', 'mu', 'beta', 'omega', 'pi']
    rows = [(i, names[i % len(names)]) for i in range(15)]
    db, con, d = _build("CREATE TABLE t (id INT, name VARCHAR)", rows, buffered=True, flush_every=8)
    for i in range(15, 20):
        db.run(f"INSERT INTO t VALUES ({i}, '{names[i % len(names)]}')")
        con.execute(f"INSERT INTO t VALUES ({i}, '{names[i % len(names)]}')")
    _match(db, con, "SELECT MIN(name), MAX(name) FROM t")
    shutil.rmtree(d)
