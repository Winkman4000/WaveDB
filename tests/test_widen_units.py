"""THE STORED UNIT (2026-11-07): a time column holds integers in ITS OWN unit (seg.cols[c]['aux'] -> s, ms, us, D),
and every date part -- EXTRACT, DATE_TRUNC -- must read that unit. Reads that took the integers for seconds answered
EXTRACT(minute) of a millisecond column with 18 groups instead of 60. Every part on a column in s, ms and us, a date
column, and a time column with NULLs (crossing the epoch, so floors are floors): GROUP BY, DISTINCT, WHERE, DATE_TRUNC
-- every answer DuckDB's."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

N = 200000
_FX = None
PARTS = ('second', 'minute', 'hour', 'day', 'month', 'year', 'dow')
COLS = ('ts_s', 'ts_ms', 'ts_us', 'tn', 'd')


def _fixture():
    """N rows in time order, each time held by 2 rows (a staircase): ts_s (epoch seconds, declared a timestamp in s),
    ts_ms (pa.timestamp('ms')), ts_us (pa.timestamp('us'), sub-second parts), over ~3 years from 2012; tn (ms, NULL
    on 1 row in 7, from 1969-12-20 -- across the epoch); d (date32); k (3 ints, a second key)"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    from wdb_db import Database
    i = np.arange(N, dtype=np.int64)
    b = i // 2
    rng = np.random.default_rng(11)
    step = rng.integers(1, 900, N // 2)                     # seconds between ticks: ~ 1.4 years over the table
    sec = np.int64(1325376000) + np.concatenate(([0], np.cumsum(step)[:-1]))     # 2012-01-01
    ts_s = sec[b]
    ts_ms = (sec * 1000 + rng.integers(0, 1000, N // 2))[b]
    ts_us = (sec * 1000000 + rng.integers(0, 1000000, N // 2))[b]
    tn_v = (np.int64(-1036800000) + np.concatenate(([0], np.cumsum(step * 1000 + 7)[:-1])))[b]   # 1969-12-20
    tn = pa.array(tn_v, pa.timestamp('ms'), mask=(i % 7 == 3))
    d = (ts_s // 86400 * 3 - 30000).astype('datetime64[D]')   # ~ 4 years of days
    k = ((i * 7919) % 3).astype(np.int32)
    tbl = pa.table({'id': i, 'k': k, 'ts_s': ts_s, 'ts_ms': pa.array(ts_ms, pa.timestamp('ms')),
                    'ts_us': pa.array(ts_us, pa.timestamp('us')), 'tn': tn, 'd': pa.array(d, pa.date32())})
    dd = os.path.join(tempfile.gettempdir(), 'wun_%s' % uuid.uuid4().hex[:8]); os.makedirs(dd, exist_ok=True)
    try:
        import helpers as _H; _H.register_dir(dd)
    except Exception:
        pass
    p = os.path.join(dd, 't.parquet'); pq.write_table(tbl, p)
    con = duckdb.connect()
    con.execute("CREATE VIEW t AS SELECT * REPLACE (make_timestamp(ts_s * 1000000) AS ts_s) FROM read_parquet('%s')" % p)
    db = Database.create(dd)
    db.cat.add_table('t', [['id', 'int'], ['k', 'int'], ['ts_s', 'datetime'], ['ts_ms', 'datetime'], ['ts_us', 'datetime'],
                           ['tn', 'datetime'], ['d', 'date']])
    saved = {k9: os.environ.pop(k9, None) for k9 in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    import wdb_read
    rc0 = wdb_read.read_columns

    def rc(*a, **k):                             # the declared cast 'timestamp_s' (an epoch-second int64), in process
        coldata, n9, cols9 = rc0(*a, **k)
        coldata['ts_s'] = wdb_encode._CASTS['timestamp_s'](coldata['ts_s'])
        return coldata, n9, cols9
    wdb_read.read_columns = rc
    try:                                         # the runner's sequence levers would store the times raw, not stairs
        wdb_encode.encode(p, os.path.join(dd, 't_0.wdb'))
    finally:
        wdb_read.read_columns = rc0
        for k9, v9 in saved.items():
            if v9 is not None:
                os.environ[k9] = v9
    db.cat.add_segment('t', 't_0.wdb')
    _FX = (db, con)
    return _FX


def _norm(v):
    if hasattr(v, 'isoformat'):
        s = str(v)
        return s + ' 00:00:00' if len(s) == 10 else s[:26]
    if isinstance(v, float) and v != v:
        return "NaN"
    if isinstance(v, float) and v == int(v):
        return int(v)
    if hasattr(v, 'item'):
        return _norm(v.item())
    return v


def _rows(r):
    return [tuple(_norm(v) for v in x) for x in (r[0] if isinstance(r, tuple) else r)]


def _key(x):
    return tuple((v is None, str(v)) for v in x)


def _check(db, con, q, ordered=False, bad=None):
    import controller
    got = _rows(db.run(q)); want = _rows(con.execute(q).fetchall())
    served = controller._SERVED[0]
    ok = got == want if ordered else sorted(got, key=_key) == sorted(want, key=_key)
    if not ok and ' DESC LIMIT' in q:            # ties at the cut: the counts exact, every row a true group
        full = set(_rows(con.execute(re.sub(r' ORDER BY .*$', '', q)).fetchall()))
        ok = [x[-1] for x in got] == [x[-1] for x in want] and all(x in full for x in got)
    if not ok:
        msg = '%s [%s]\n  got  %s\n  want %s' % (q, served, got[:6], want[:6])
        if bad is None:
            raise AssertionError(msg)
        bad.append(msg)
    return served


def _shapes(col, part):
    e = 'EXTRACT(%s FROM %s)' % (part, col)
    return [("SELECT %s AS e, COUNT(*) AS c FROM t GROUP BY e" % e, False),
            ("SELECT %s AS e, COUNT(*) AS c FROM t GROUP BY e ORDER BY c DESC LIMIT 5" % e, True),
            ("SELECT %s, COUNT(*) AS c FROM t GROUP BY %s" % (e, e), False),
            ("SELECT DISTINCT %s AS e FROM t" % e, False),
            ("SELECT COUNT(DISTINCT %s) FROM t" % e, False),
            ("SELECT COUNT(*) FROM t WHERE %s = 3" % e, False),
            ("SELECT %s AS e, MIN(id) FROM t GROUP BY e" % e, False)]


def _key_shapes(col, part):
    """the part beside a second key: two-key DISTINCT, GROUP BY, top-K, a filter on both"""
    e = 'EXTRACT(%s FROM %s)' % (part, col)
    return [("SELECT DISTINCT %s AS e, k FROM t" % e, False),
            ("SELECT %s AS e, k, COUNT(*) AS c FROM t GROUP BY e, k" % e, False),
            ("SELECT k, %s AS e, COUNT(*) AS c FROM t GROUP BY k, e ORDER BY c DESC LIMIT 7" % e, True),
            ("SELECT COUNT(*) FROM t WHERE %s = 3 AND k = 1" % e, False)]


def _trunc_shapes(col, u):
    e = "DATE_TRUNC('%s', %s)" % (u, col)
    return [("SELECT %s AS m, COUNT(*) AS c FROM t GROUP BY m ORDER BY c DESC, m LIMIT 10" % e, True),
            ("SELECT %s AS m, COUNT(*) AS c FROM t GROUP BY m ORDER BY m LIMIT 10" % e, True),
            ("SELECT COUNT(DISTINCT %s) FROM t" % e, False)]


def _run_all(db, con, cols, parts=PARTS, trunc=('minute', 'hour', 'day'), keys=True):
    bad, served = [], {}
    for col in cols:
        for part in parts:
            for q, o in _shapes(col, part) + (_key_shapes(col, part) if keys else []):
                served.setdefault(_check(db, con, q, o, bad), []).append(q)
        for u in trunc:
            for q, o in _trunc_shapes(col, u):
                served.setdefault(_check(db, con, q, o, bad), []).append(q)
    return bad, served


def test_units_s_ms_us_null_date():
    db, con = _fixture()
    bad, served = _run_all(db, con, COLS)
    assert not bad, '%d wrong:\n' % len(bad) + '\n'.join(bad[:20])


def test_stair_serves_minute_on_ms():
    """the stair-walk serves a millisecond staircase, exact"""
    db, con = _fixture()
    for col in ('ts_s', 'ts_ms', 'ts_us'):
        for part in ('minute', 'hour'):
            q = "SELECT EXTRACT(%s FROM %s) AS m, COUNT(*) AS c FROM t GROUP BY m" % (part, col)
            assert _check(db, con, q) == 'stair', q


def test_stair_serves_every_part():
    """the stair-walk serves every integer part of a time staircase, in any unit, and its ties at a cut"""
    db, con = _fixture()
    for col in ('ts_s', 'ts_ms', 'ts_us', 'd'):
        for part in PARTS + ('quarter', 'doy', 'isodow'):
            q = "SELECT EXTRACT(%s FROM %s) AS m, COUNT(*) AS c FROM t GROUP BY m" % (part, col)
            assert _check(db, con, q) == 'stair', q
    # a GROUP BY of another part than the one projected is not the stair-walk's
    q = "SELECT EXTRACT(hour FROM ts_ms), COUNT(*) AS c FROM t GROUP BY EXTRACT(minute FROM ts_ms)"
    import controller
    try:
        db.run(q)
    except Exception:
        pass
    assert controller._SERVED[0] != 'stair', q
    # the bare staircase's top-K: every time ties at 2 rows, so the cut is the lowest times (code ASC)
    for col in ('ts_s', 'ts_ms', 'ts_us', 'd'):
        _check(db, con, "SELECT %s, COUNT(*) AS c FROM t GROUP BY %s ORDER BY c DESC, %s LIMIT 5" % (col, col, col), True)


def test_null_parts():
    """a time column with NULLs: NULL's part is NULL -- grouped as NULL, never equal, unequal, above or below
    anything, skipped by aggregates, emitted as NULL"""
    db, con = _fixture()
    for q in ("SELECT COUNT(*) FROM t WHERE EXTRACT(minute FROM tn) <> 30",
              "SELECT COUNT(*) FROM t WHERE EXTRACT(minute FROM tn) > 30",
              "SELECT COUNT(*) FROM t WHERE EXTRACT(dow FROM tn) <= 2",
              "SELECT COUNT(*) FROM t WHERE EXTRACT(hour FROM tn) = 0 AND k = 2",
              "SELECT MAX(EXTRACT(hour FROM tn)), MIN(DATE_TRUNC('day', tn)), COUNT(DATE_TRUNC('day', tn)) FROM t",
              "SELECT COUNT(EXTRACT(hour FROM tn)), COUNT(DISTINCT EXTRACT(second FROM tn)) FROM t",
              "SELECT EXTRACT(minute FROM tn) AS m, k, COUNT(*) AS c FROM t GROUP BY m, k ORDER BY c DESC, m, k LIMIT 5"):
        _check(db, con, q)
    for q in ("SELECT id, EXTRACT(minute FROM tn) AS m FROM t ORDER BY id LIMIT 20",
              "SELECT id, EXTRACT(year FROM tn) AS m, EXTRACT(second FROM ts_us) AS s FROM t WHERE k = 1 ORDER BY id LIMIT 20",
              "SELECT id, DATE_TRUNC('minute', tn) AS m FROM t ORDER BY id LIMIT 20"):
        _check(db, con, q, True)


def test_distinct_hour_limit():
    """DISTINCT (hour, key) LIMIT n: any n of the true pairs"""
    db, con = _fixture()
    for col in ('ts_s', 'ts_ms', 'ts_us', 'd'):
        q = "SELECT DISTINCT EXTRACT(hour FROM %s) AS h, k FROM t LIMIT 50" % col
        got = _rows(db.run(q))
        full = set(_rows(con.execute(q.replace(' LIMIT 50', '')).fetchall()))
        assert len(set(got)) == min(50, len(full)) and set(got) <= full, (q, got[:5])


def test_sparse_default_tm():
    """the fixture the bug was found on: tm built from pa.timestamp('s'), stored in ms"""
    import test_sparse_default as SD
    db, con = SD._fixture()
    bad, _ = _run_all(db, con, ('tm', 'd'))
    assert not bad, '%d wrong:\n' % len(bad) + '\n'.join(bad[:20])


if __name__ == '__main__':
    fails = 0
    for nm, fn in sorted(globals().items()):
        if nm.startswith('test_') and callable(fn):
            try:
                fn(); print('ok  ', nm)
            except AssertionError as ex:
                fails += 1; print('FAIL', nm, ex)
    sys.exit(1 if fails else 0)
