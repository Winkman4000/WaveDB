"""THE DASHBOARD (2026-10-24): funnel takes text and date literals in = / <> filters (codes through wherescan's
_code_of) and DATE_TRUNC by second, minute, hour or day on a staircase datetime or date column (bucket = epoch
seconds over the unit; a DATE column's days scaled to seconds). Every answer DuckDB's; funnel answers them."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

_FX = None


def _fixture():
    """300K rows in time order (so tm and d are staircases): tm one tick per 4 rows, 7 s apart, from 2013-07-01 (six
    days; unique ticks would be stored raw), d its date, c (300 values: the wide equality), f (flag), s (strings,
    2,000), u (strings, 50,000)"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        TIMESTAMP '2013-07-01 00:00:00' + to_seconds(CAST((i // 4) * 7 AS BIGINT)) AS tm,
        CAST(TIMESTAMP '2013-07-01 00:00:00' + to_seconds(CAST((i // 4) * 7 AS BIGINT)) AS DATE) AS d,
        CAST((i * 2654435761) % 300 AS INTEGER) AS c,
        CAST(CASE WHEN (i * 7919) % 10 < 8 THEN 0 ELSE 1 END AS INTEGER) AS f,
        's' || CAST((i * 40507) % 2000 AS VARCHAR) AS s,
        'u' || CAST((i * 48271) % 50000 AS VARCHAR) AS u
        FROM range(300000) r(i)""")
    dd = os.path.join(tempfile.gettempdir(), 'dsh_%s' % uuid.uuid4().hex[:8]); os.makedirs(dd, exist_ok=True)
    pq = os.path.join(dd, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(dd)
    db.cat.add_table('t', [['id', 'int'], ['tm', 'datetime'], ['d', 'date'], ['c', 'int'], ['f', 'int'],
                           ['s', 'str'], ['u', 'str']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:                                         # the runner's sequence levers would store tm raw (mode 4), not a staircase
        wdb_encode.encode(pq, os.path.join(dd, 't_0.wdb'))
    finally:
        for k9, v9 in saved.items():
            if v9 is not None:
                os.environ[k9] = v9
    db.cat.add_segment('t', 't_0.wdb')
    _FX = (db, con)
    return _FX


W = "c = 62 AND d >= '2013-07-02' AND d <= '2013-07-05' AND f = 0"
QS = ["SELECT s, COUNT(*) AS PageViews FROM t WHERE %s AND u <> 'u77' GROUP BY s ORDER BY PageViews DESC LIMIT 10" % W,
      "SELECT s, COUNT(*) AS PageViews FROM t WHERE %s AND u <> 'zz-absent' GROUP BY s ORDER BY PageViews DESC LIMIT 10" % W,
      "SELECT d, COUNT(*) AS PageViews FROM t WHERE %s AND d <> '2013-07-04' GROUP BY d ORDER BY PageViews DESC LIMIT 10" % W,
      "SELECT s, COUNT(*) AS PageViews FROM t WHERE %s AND s = 's7' GROUP BY s ORDER BY PageViews DESC LIMIT 10" % W]
for unit in ('second', 'minute', 'hour', 'day'):
    for col in ('tm', 'd'):
        for off in (0, 3):
            QS.append("SELECT DATE_TRUNC('%s', %s) AS M, COUNT(*) AS PageViews FROM t WHERE %s GROUP BY DATE_TRUNC('%s', "
                      "%s) ORDER BY DATE_TRUNC('%s', %s) LIMIT 10 OFFSET %d" % (unit, col, W, unit, col, unit, col, off))


def _legal(q, got, con):
    want = con.execute(q).fetchall()
    if 'ORDER BY PageViews DESC' not in q:
        return [tuple(x) for x in got] == [tuple(x) for x in want]
    if [x[-1] for x in got] != [x[-1] for x in want]:      # ties at the cut: the counts exact, each group's true
        return False
    full = dict((tuple(x[:-1]), x[-1]) for x in con.execute(re.sub(r' ORDER BY .*$', '', q)).fetchall())
    return all(full.get(tuple(x[:-1])) == x[-1] for x in got)


def test_dashboard_shapes_legal_and_funnel_answers():
    db, con = _fixture()
    import controller
    served = 0
    for q in QS:
        controller.plan_epoch_bump()
        r = db.run(q)
        got = [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)]
        assert _legal(q, got, con), (q, got[:3], con.execute(q).fetchall()[:3])
        served += controller._SERVED[0] == 'funnel'
    assert served == len(QS), (served, len(QS))
