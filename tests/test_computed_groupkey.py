"""Computed GROUP BY keys: EXTRACT(unit FROM datecol) grouped as a coarsening of the sorted date
dictionary -- evaluated over V dict values, never per-row, no new storage. Verifies the engine
answers year/month/quarter group-bys (alias and explicit forms, with COUNT/SUM, single and multi-key)
identically to a direct computation. Self-contained small segment."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from wdb_db import Database
from wdb_engine import Segment
import wdb_sql, wdb_encode


def _build(d):
    db = Database.create(d); db.run("CREATE TABLE t (mode VARCHAR, d DATE, q INTEGER)")
    dates = pd.to_datetime('1992-01-01') + pd.to_timedelta(np.arange(2000) * 7, unit='D')  # spans years
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'mode': rng.choice(['AIR', 'SHIP', 'RAIL'], 2000),
                       'd': rng.choice(dates, 2000),
                       'q': rng.integers(1, 50, 2000)})
    pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb')
    return db, Segment(os.path.join(d, 't_0.wdb')), df


def _truth_year(df):
    g = df.assign(y=df['d'].dt.year).groupby('y').size()
    return sorted((int(y), int(c)) for y, c in g.items())


def test_computed_groupkey():
    d = os.path.join(tempfile.gettempdir(), f'cgk_{uuid.uuid4().hex[:8]}')
    try:
        db, seg, df = _build(d)
        # alias form: GROUP BY y where y is a SELECT alias for EXTRACT(year FROM d)
        rows = wdb_sql.execute(seg, "SELECT EXTRACT(year FROM d) y, COUNT(*) FROM t GROUP BY y ORDER BY y")[0]
        assert sorted((int(r[0]), int(r[1])) for r in rows) == _truth_year(df), rows
        # explicit form: GROUP BY EXTRACT(...) -- same answer
        rows2 = wdb_sql.execute(seg, "SELECT EXTRACT(year FROM d) y, COUNT(*) FROM t GROUP BY EXTRACT(year FROM d) ORDER BY y")[0]
        assert sorted((int(r[0]), int(r[1])) for r in rows2) == _truth_year(df)
        # month + quarter produce the right number of groups
        rmo = wdb_sql.execute(seg, "SELECT EXTRACT(month FROM d) m, COUNT(*) FROM t GROUP BY m")[0]
        assert {int(r[0]) for r in rmo} <= set(range(1, 13)) and len(rmo) == df['d'].dt.month.nunique()
        rq = wdb_sql.execute(seg, "SELECT EXTRACT(quarter FROM d) q, COUNT(*) FROM t GROUP BY q")[0]
        assert {int(r[0]) for r in rq} <= {1, 2, 3, 4}
        # SUM over a computed key matches pandas
        rs = wdb_sql.execute(seg, "SELECT EXTRACT(year FROM d) y, SUM(q) s FROM t GROUP BY y ORDER BY y")[0]
        truth_sum = df.assign(y=df['d'].dt.year).groupby('y')['q'].sum()
        assert {int(r[0]): int(r[1]) for r in rs} == {int(k): int(v) for k, v in truth_sum.items()}
        # two-key: bare column + computed key
        r2 = wdb_sql.execute(seg, "SELECT mode, EXTRACT(year FROM d) y, COUNT(*) c FROM t GROUP BY mode, y")[0]
        truth2 = df.assign(y=df['d'].dt.year).groupby(['mode', 'y']).size()
        assert {(r[0], int(r[1])): int(r[2]) for r in r2} == {(m, int(y)): int(c) for (m, y), c in truth2.items()}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_datecount_rollup_fastpath():
    """Single-key date-coarsening COUNT(*) is answered via the V->G per-code rollup (no N pass),
    fires only for that shape, and matches the general path. SUM / two-key / WHERE decline it."""
    import wdb_sql
    d = os.path.join(tempfile.gettempdir(), f'dcr_{uuid.uuid4().hex[:8]}')
    try:
        db, seg, df = _build(d)
        def fired(sql):
            h0 = wdb_sql._DATECOUNT_HITS
            rows = wdb_sql.execute(seg, sql)[0]
            return rows, (wdb_sql._DATECOUNT_HITS - h0 == 1)
        # fires + correct vs pandas truth
        rows, f = fired("SELECT EXTRACT(year FROM d) y, COUNT(*) FROM t GROUP BY y ORDER BY y")
        assert f, "single-key date COUNT(*) should use the rollup fast path"
        assert sorted((int(r[0]), int(r[1])) for r in rows) == _truth_year(df)
        # ORDER BY count DESC + LIMIT still uses it
        _, f2 = fired("SELECT EXTRACT(year FROM d) y, COUNT(*) c FROM t GROUP BY y ORDER BY c DESC LIMIT 2")
        assert f2
        # must NOT fire for SUM, two-key, or WHERE (but those still answer via the general path)
        _, f3 = fired("SELECT EXTRACT(year FROM d) y, SUM(q) s FROM t GROUP BY y")
        assert not f3
        _, f4 = fired("SELECT mode, EXTRACT(year FROM d) y, COUNT(*) c FROM t GROUP BY mode, y")
        assert not f4
        _, f5 = fired("SELECT EXTRACT(year FROM d) y, COUNT(*) c FROM t WHERE q > 10 GROUP BY y")
        assert not f5
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_time_coarsening_units():
    """EXTRACT(minute/hour) and DATE_TRUNC over a timestamp column: classifier recognizes them,
    the coarsening is evaluated over the dictionary, and DATE_TRUNC renders as a datetime."""
    import wdb_sql, sqlglot
    # classifier recognizes the new shapes (incl. SELECT-alias resolution)
    def keys(q):
        t = sqlglot.parse_one(q, read='duckdb')
        return [wdb_sql._group_key(g, t.expressions) for g in t.args['group'].expressions]
    assert keys("SELECT extract(minute FROM t) m, COUNT(*) FROM x GROUP BY m") == [('fn', 't', 'MINUTE')]
    assert keys("SELECT extract(hour FROM t) h, COUNT(*) FROM x GROUP BY h") == [('fn', 't', 'HOUR')]
    assert keys("SELECT DATE_TRUNC('minute', t) m, COUNT(*) FROM x GROUP BY m") == [('fn', 't', 'TRUNC:MINUTE')]
    assert keys("SELECT a, extract(minute FROM t) m, b, COUNT(*) FROM x GROUP BY a, m, b") == [
        ('col', 'a'), ('fn', 't', 'MINUTE'), ('col', 'b')]
    # _date_unit math over a tiny second-resolution dict (epoch seconds)
    import numpy as np
    secs = np.array([1372708800, 1372708860, 1372712400], dtype=np.int64)  # 20:00:00, 20:01:00, 21:00:00 UTC
    assert wdb_sql._date_unit(secs, 'MINUTE', 's').tolist() == [0, 1, 0]
    assert wdb_sql._date_unit(secs, 'HOUR', 's').tolist() == [20, 20, 21]
    tr = wdb_sql._date_unit(secs, 'TRUNC:MINUTE', 's')   # truncates to the minute (epoch ticks)
    assert tr.tolist() == [1372708800, 1372708860, 1372712400]
    assert wdb_sql._is_trunc('TRUNC:MINUTE') and not wdb_sql._is_trunc('MINUTE')
