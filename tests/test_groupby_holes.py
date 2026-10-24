"""THE GROUP BY HOLES (2026-10-24): a keys-only GROUP BY (no aggregate) nothing took is asked with a hidden COUNT(*),
the count dropped (wdb_counted.fallback); wherescan's GROUP BY emits big answers by column (_emit_columns) -- the
same values, the same types, the same order as its per-group loop. Every answer is DuckDB's."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import sqlglot
import wdb_counted as WC

_FX = None


def _fixture():
    """300K rows: a (3,000 values), b (41), c (701), s (strings, 20,000 values), u (strings, 300)"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        CAST((i * 2654435761) % 3000 AS INTEGER) AS a,
        CAST((i * 48271) % 41 AS INTEGER) AS b,
        CAST((i * 69621) % 701 AS INTEGER) AS c,
        's' || CAST((i * 40507) % 20000 AS VARCHAR) AS s,
        'u' || CAST((i * 7919) % 300 AS VARCHAR) AS u,
        DATE '1990-01-01' + CAST((i * 31) % 12000 AS INTEGER) AS d,
        TIMESTAMP '2013-07-01 00:00:00' + to_seconds(CAST((i * 97) % 20000 AS BIGINT)) AS tm
        FROM range(300000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'gbh_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['a', 'int'], ['b', 'int'], ['c', 'int'], ['s', 'str'], ['u', 'str'],
                           ['d', 'date'], ['tm', 'datetime']])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    db.cat.add_segment('t', 't_0.wdb')
    _FX = (db, con)
    return _FX


def _p(q):
    return sqlglot.parse_one(q, read='duckdb')


def _rows(r):
    return r[0] if isinstance(r, tuple) else r


def test_which_group_bys_are_keys_only():
    yes = ('SELECT a FROM t GROUP BY a', 'SELECT a + 1 AS x FROM t WHERE b = 3 GROUP BY x',
           'SELECT a FROM t GROUP BY a, b', 'SELECT a, b FROM t GROUP BY a, b ORDER BY a LIMIT 5')
    no = ('SELECT a, COUNT(*) FROM t GROUP BY a', 'SELECT a FROM t', 'SELECT DISTINCT a FROM t',
          'SELECT a FROM t GROUP BY a HAVING a > 3', 'SELECT * FROM t GROUP BY a', 'SELECT a FROM t GROUP BY ALL',
          'SELECT a FROM t GROUP BY ROLLUP (a)', 'SELECT a, SUM(b) OVER () FROM t GROUP BY a, b')
    for q in yes:
        assert WC.keys_only_group(_p(q)), q
    for q in no:
        assert not WC.keys_only_group(_p(q)), q


def test_keys_only_group_by_legal_with_and_without_the_counted_form():
    db, con = _fixture()
    import controller
    qs = ['SELECT a + 1 AS x FROM t GROUP BY x',
          'SELECT a FROM t GROUP BY a',
          'SELECT a, c FROM t GROUP BY a, c',
          'SELECT a FROM t GROUP BY a, b',                           # a key not projected: a repeats
          'SELECT b * 2 AS x FROM t WHERE c < 300 GROUP BY x',
          'SELECT length(s) AS n, b FROM t GROUP BY n, b',
          'SELECT lower(u) AS x FROM t WHERE b <> 3 GROUP BY x']
    saved = os.environ.get('WDB_COUNTED_DISTINCT')
    try:
        for q in qs:
            want = sorted(con.execute(q).fetchall())
            got = {}
            for on in ('1', '0'):
                os.environ['WDB_COUNTED_DISTINCT'] = on
                controller.plan_epoch_bump()
                rows = sorted(tuple(x) for x in _rows(db.run(q)))
                assert rows == [tuple(x) for x in want], (q, on, rows[:3], want[:3])
                got[on] = rows
            assert got['1'] == got['0'], q
        os.environ['WDB_COUNTED_DISTINCT'] = '1'
        controller.plan_epoch_bump()
        q = 'SELECT a + 1 AS x FROM t GROUP BY x ORDER BY x DESC LIMIT 7'
        assert [tuple(r) for r in _rows(db.run(q))] == con.execute(q).fetchall()
    finally:
        if saved is None:
            os.environ.pop('WDB_COUNTED_DISTINCT', None)
        else:
            os.environ['WDB_COUNTED_DISTINCT'] = saved
        controller.plan_epoch_bump()


def test_keys_only_group_by_reaches_the_counted_form():
    db, _con = _fixture()
    import controller
    controller.plan_epoch_bump()
    f0 = WC._FALLBACK[0]
    db.run('SELECT a + 1 AS x FROM t GROUP BY x')
    assert WC._FALLBACK[0] == f0 + 1 and str(controller._SERVED[0]).startswith('counted:'), controller._SERVED[0]


def test_wherescan_columnar_emission_is_the_loop_exactly():
    db, con = _fixture()
    import controller, wdb_wherescan as WS
    qs = ['SELECT s, COUNT(*) FROM t WHERE b = 5 GROUP BY s',
          'SELECT s, a, COUNT(*) AS n FROM t WHERE b = 5 GROUP BY s, a',
          'SELECT s, MIN(u), MAX(u), COUNT(DISTINCT a), SUM(c), AVG(c) FROM t WHERE b = 5 GROUP BY s',
          'SELECT s, MIN(id), MAX(id), COUNT(u) FROM t WHERE b = 7 GROUP BY s',
          'SELECT a + 1 AS x, COUNT(*) FROM t WHERE b = 5 GROUP BY x',
          'SELECT s, COUNT(*) AS n FROM t WHERE b = 5 GROUP BY s ORDER BY n DESC, s LIMIT 50',
          'SELECT u, COUNT(*) FROM t WHERE c = 11 GROUP BY u',
          "SELECT s, COUNT(*) FROM t WHERE u = 'u7' GROUP BY s",
          "SELECT a, COUNT(*) FROM t WHERE s LIKE '%1%' GROUP BY a",
          'SELECT d, COUNT(*) FROM t WHERE b = 5 GROUP BY d',                 # DATE keys emit dates, not datetimes
          'SELECT tm, COUNT(*) FROM t WHERE b = 5 GROUP BY tm',
          'SELECT s, MIN(d), MAX(tm) FROM t WHERE b = 5 GROUP BY s']
    saved = WS._COL_EMIT_MIN
    served = 0
    try:
        for q in qs:
            outs = {}
            for gate in (0, 1 << 40):
                WS._COL_EMIT_MIN = gate
                controller.plan_epoch_bump()
                h0 = WS._HITS
                r = db.run(q)
                if WS._HITS > h0:
                    served += 1
                outs[gate] = [tuple(x) for x in _rows(r)]
            a0, a1 = outs[0], outs[1 << 40]
            assert [tuple(map(repr, x)) for x in a0] == [tuple(map(repr, x)) for x in a1], (q, a0[:2], a1[:2])
            want = con.execute(q).fetchall()
            ncols = len(want[0])
            keyish = [i for i in range(ncols) if not isinstance(want[0][i], float)]
            proj = lambda rows: sorted(tuple(x[i] for i in keyish) for x in rows)
            if 'ORDER BY' in q:
                assert [tuple(x[i] for i in keyish) for x in a0] == [tuple(x[i] for i in keyish) for x in want], q
            else:
                assert proj(a0) == proj(want), q
            for i in keyish:                                     # the same Python types as DuckDB's (date stays date)
                assert type(a0[0][i]) is type(want[0][i]) or isinstance(want[0][i], int), (q, i, a0[0][i], want[0][i])
    finally:
        WS._COL_EMIT_MIN = saved
        controller.plan_epoch_bump()
    assert served >= 12, served      # wherescan answered both gates of most queries (which keys the encoder frames,
                                     # and so which filters wherescan takes, varies with the run's settings)
