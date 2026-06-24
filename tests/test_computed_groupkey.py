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


def test_positional_and_const_groupkeys():
    """Positional GROUP BY (GROUP BY n -> Nth SELECT item), constant-literal group keys
    (GROUP BY 1 where SELECT 1), and projection<->group-key matching by expression identity
    (SELECT/GROUP BY orderings may differ). Verified vs DuckDB on a small segment."""
    import duckdb, sqlglot
    d = os.path.join(tempfile.gettempdir(), f'pos_{uuid.uuid4().hex[:8]}')
    try:
        os.makedirs(d, exist_ok=True)
        rng = np.random.default_rng(1)
        urls = np.array(['', 'http://a.com', 'http://b.org/p', 'https://c.net/x'], dtype=object)
        df = pd.DataFrame({'URL': urls[rng.integers(0, len(urls), 4000)],
                           'RegionID': rng.integers(0, 9, 4000).astype(np.int64)})
        pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
        seg = Segment(os.path.join(d, 't_0.wdb')); con = duckdb.connect()

        def norm(rs):
            return sorted([tuple(round(float(x), 4) if isinstance(x, float) else x for x in r) for r in rs],
                          key=lambda t: tuple(str(x) for x in t))
        def chk(q):
            wave = wdb_sql.execute(seg, q.replace('FROM t', 'FROM hits'), col_map={c: c for c in seg.cols})[0]
            duck = con.execute(q.replace('FROM t', f"FROM '{pq}'")).fetchall()
            assert norm(wave) == norm(duck), (q, norm(wave)[:4], norm(duck)[:4])

        # classifier: positional + const tuples
        def keys(q):
            tt = sqlglot.parse_one(q, read='duckdb')
            return [wdb_sql._group_key(g, tt.expressions) for g in tt.args['group'].expressions]
        assert keys("SELECT 1, URL, COUNT(*) FROM t GROUP BY 1, URL") == [('const', 1), ('col', 'URL')]
        assert keys("SELECT URL, RegionID, COUNT(*) FROM t GROUP BY 2, 1") == [('col', 'RegionID'), ('col', 'URL')]

        chk("SELECT 1, URL, COUNT(*) AS c FROM t GROUP BY 1, URL")            # Q34 shape
        chk("SELECT URL, COUNT(*) c FROM t GROUP BY 1")                        # positional -> real column
        chk("SELECT URL, RegionID, COUNT(*) c FROM t GROUP BY 2, 1")           # SELECT order != GROUP BY order
        chk("SELECT RegionID, 1, URL, COUNT(*) c FROM t GROUP BY 3, 1, 2")     # 3-key mixed order
        chk("SELECT 1, COUNT(*) c FROM t GROUP BY 1")                          # const-only
        chk("SELECT 'x', RegionID, COUNT(*) c FROM t GROUP BY 1, RegionID")    # string const + real key
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_affine_groupkeys():
    """Injective single-column arithmetic group keys (col +/- const, *const, nested) reduce to their
    base column; repeated base columns dedup so GROUP BY ClientIP, ClientIP-1, ClientIP-2, ClientIP-3
    groups by ClientIP ONCE (no radix-combo explosion). Independent derived keys (RegionID, UserID-1)
    add their base column to the group. Values computed per pile at emit. Verified vs DuckDB."""
    import duckdb, sqlglot
    d = os.path.join(tempfile.gettempdir(), f'aff_{uuid.uuid4().hex[:8]}')
    try:
        os.makedirs(d, exist_ok=True)
        rng = np.random.default_rng(3); N = 8000
        df = pd.DataFrame({'ClientIP': rng.integers(1000, 5000, N).astype(np.int64),
                           'UserID': rng.integers(0, 300, N).astype(np.int64),
                           'RegionID': rng.integers(0, 8, N).astype(np.int64)})
        pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
        seg = Segment(os.path.join(d, 't_0.wdb')); con = duckdb.connect()

        def norm(rs): return sorted([tuple(int(x) for x in r) for r in rs])
        def chk(q):
            wave = wdb_sql.execute(seg, q.replace('FROM t', 'FROM hits'), col_map={c: c for c in seg.cols})[0]
            duck = con.execute(q.replace('FROM t', f"FROM '{pq}'")).fetchall()
            assert norm(wave) == norm(duck), (q, norm(wave)[:3], norm(duck)[:3])

        # classifier: affine tuple + dedup to a single effective base column
        def keys(q):
            tt = sqlglot.parse_one(q, read='duckdb')
            return [wdb_sql._group_key(g, tt.expressions) for g in tt.args['group'].expressions]
        assert keys("SELECT ClientIP-1 k FROM t GROUP BY ClientIP-1") == [('affine', 'ClientIP', 'ClientIP - 1')]
        assert wdb_sql._affine_key(sqlglot.parse_one("length(URL)", read='duckdb')) is None  # not injective

        chk("SELECT ClientIP, ClientIP-1, ClientIP-2, ClientIP-3, COUNT(*) c FROM t "
            "GROUP BY ClientIP, ClientIP-1, ClientIP-2, ClientIP-3")                 # Q35 (full set)
        chk("SELECT ClientIP-5 k, COUNT(*) c FROM t GROUP BY ClientIP-5")            # single affine
        chk("SELECT ClientIP*2 k, COUNT(*) c FROM t GROUP BY ClientIP*2")            # mul by const
        chk("SELECT (ClientIP-1)*2+3 k, COUNT(*) c FROM t GROUP BY (ClientIP-1)*2+3")# nested affine
        chk("SELECT RegionID, UserID-1, COUNT(*) c FROM t GROUP BY RegionID, UserID-1")  # independent
        chk("SELECT RegionID, ClientIP, ClientIP-1, COUNT(*) c FROM t "
            "GROUP BY RegionID, ClientIP, ClientIP-1")                               # affine + plain mix
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_case_groupkeys():
    """CASE WHEN ... THEN ... ELSE ... END as a group key (Q39). Not injective / multi-column, so it
    is a genuine derived key: evaluated per row (vectorized) then factorized. Verified vs DuckDB,
    including the alias-referenced CASE (GROUP BY Src where Src is the SELECT alias)."""
    import duckdb, sqlglot
    d = os.path.join(tempfile.gettempdir(), f'case_{uuid.uuid4().hex[:8]}')
    try:
        os.makedirs(d, exist_ok=True)
        rng = np.random.default_rng(7); N = 12000
        refs = np.array(['', 'http://r1.com', 'http://r2.org', 'http://r3.net'], dtype=object)
        urls = np.array(['', 'http://u1.com', 'http://u2.org'], dtype=object)
        df = pd.DataFrame({
            'TraficSourceID': rng.integers(-1, 4, N).astype(np.int64),
            'SearchEngineID': rng.integers(0, 3, N).astype(np.int64),
            'AdvEngineID':    rng.integers(0, 3, N).astype(np.int64),
            'Referer': refs[rng.integers(0, len(refs), N)],
            'URL':     urls[rng.integers(0, len(urls), N)],
            'CounterID': rng.integers(60, 64, N).astype(np.int64),
            'IsRefresh': rng.integers(0, 2, N).astype(np.int64)})
        pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
        seg = Segment(os.path.join(d, 't_0.wdb')); con = duckdb.connect()

        def norm(rs): return sorted([tuple(x for x in r) for r in rs], key=lambda t: tuple(str(x) for x in t))
        def chk(q):
            wave = wdb_sql.execute(seg, q.replace('FROM t', 'FROM hits'), col_map={c: c for c in seg.cols})[0]
            duck = con.execute(q.replace('FROM t', f"FROM '{pq}'")).fetchall()
            assert norm(wave) == norm(duck), (q, norm(wave)[:4], norm(duck)[:4])

        k = wdb_sql._group_key(sqlglot.parse_one(
            "SELECT CASE WHEN AdvEngineID=0 THEN Referer ELSE '' END AS S FROM t", read='duckdb').expressions[0],
            None)
        assert k[0] == 'rowexpr'

        chk("SELECT TraficSourceID, SearchEngineID, AdvEngineID, "
            "CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, "
            "URL AS Dst, COUNT(*) AS PageViews FROM t WHERE CounterID = 62 AND IsRefresh = 0 "
            "GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst")          # Q39 shape
        chk("SELECT CASE WHEN SearchEngineID=0 THEN 'a' ELSE 'b' END AS k, COUNT(*) c FROM t GROUP BY k")
        chk("SELECT CASE WHEN AdvEngineID>0 THEN 1 ELSE 0 END AS k, COUNT(*) c FROM t GROUP BY k")
        # plain column + CASE together, grouping by the explicit CASE expression (DuckDB-accepted form)
        chk("SELECT SearchEngineID, CASE WHEN AdvEngineID=0 THEN Referer ELSE '' END AS S, COUNT(*) c "
            "FROM t GROUP BY SearchEngineID, CASE WHEN AdvEngineID=0 THEN Referer ELSE '' END")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_regexp_replace_groupkey():
    """REGEXP_REPLACE(col, pattern, repl) as a group key (Q28) -- a string-valued scalar function over
    a single column (non-injective). Computed per distinct dict value, factorized, grouped. Verified
    vs DuckDB incl. AVG(length(col)), COUNT(*), MIN(col), WHERE, HAVING, ORDER BY."""
    import duckdb, sqlglot
    d = os.path.join(tempfile.gettempdir(), f'rx_{uuid.uuid4().hex[:8]}')
    try:
        os.makedirs(d, exist_ok=True)
        rng = np.random.default_rng(11); N = 9000
        refs = np.array(['', 'http://www.example.com/page/1', 'https://sub.foo.org/x', 'http://a.b.c/',
                         'https://www.example.com/q?z=1', 'http://other.net/path', 'notaurl',
                         'http://example.com/y', 'https://deep.site.co.uk/a/b/c'], dtype=object)
        df = pd.DataFrame({'Referer': refs[rng.integers(0, len(refs), N)]})
        pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
        seg = Segment(os.path.join(d, 't_0.wdb')); con = duckdb.connect()
        RX = r"REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\1')"

        def norm(rs): return sorted([tuple(round(float(x),4) if isinstance(x,float) else x for x in r)
                                     for r in rs], key=lambda t: tuple(str(x) for x in t))
        def chk(q):
            wave = wdb_sql.execute(seg, q.replace('FROM t', 'FROM hits'), col_map={c: c for c in seg.cols})[0]
            duck = con.execute(q.replace('FROM t', f"FROM '{pq}'")).fetchall()
            assert norm(wave) == norm(duck), (q, norm(wave)[:4], norm(duck)[:4])

        k = wdb_sql._group_key(sqlglot.parse_one(f"SELECT {RX} AS k FROM t", read='duckdb').expressions[0], None)
        assert k[0] == 'sfn' and k[2] == 'REGEXP_REPLACE'

        chk(f"SELECT {RX} AS k, COUNT(*) c FROM t WHERE Referer<>'' GROUP BY k")
        chk(f"SELECT {RX} AS k, AVG(length(Referer)) AS l, COUNT(*) AS c, MIN(Referer) "
            f"FROM t WHERE Referer<>'' GROUP BY k HAVING COUNT(*)>500 ORDER BY l DESC, k LIMIT 25")
    finally:
        shutil.rmtree(d, ignore_errors=True)
