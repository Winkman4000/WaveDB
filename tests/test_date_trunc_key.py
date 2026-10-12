"""DATE_TRUNC(unit, col) as a GROUP BY key rides the dictionary in fused_agg: the V distinct times are truncated
(wdb_sql.date_trunc, the same definition the row evaluator uses), never the N rows. Before, a computed key sent
the query past fused_agg -- the funnel family's Q42 variants scanned the whole table (2026-10-12). Checked
against DuckDB; a midnight timestamp and a date compare equal (the board's own normalisation)."""
import sys, os, tempfile, uuid, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, controller
from wdb_db import Database

_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'dtk_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    _CON.execute("""CREATE TABLE t AS SELECT i AS id, i % 40 AS site, i % 2 AS r,
        TIMESTAMP '2013-01-01 00:00:00' + INTERVAL ((i * 7919) % 31536000) SECOND AS ts,
        DATE '2013-01-01' + ((i * 13) % 365)::INT AS d FROM range(300000) t(i)""")
    _DB = Database.create(d)
    pq = os.path.join(d, 't.parquet')
    _CON.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    _DB.cat.add_table('t', [['id', 'int'], ['site', 'int'], ['r', 'int'], ['ts', 'datetime'], ['d', 'date']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None: os.environ[k] = v
    _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON


def _n(v):
    if isinstance(v, datetime.datetime) and v.time() == datetime.time(0):
        return v.date()
    return round(v, 6) if isinstance(v, float) else v


def _rows(res):
    return [tuple(_n(c) for c in r) for r in (res[0] if isinstance(res, tuple) else res)]


def _exact(q):
    db, con = _fixture()
    got = _rows(db.run(q)); served = controller._SERVED[0]
    want = [tuple(_n(c) for c in r) for r in con.execute(q).fetchall()]
    assert got == want, (q, served, got[:3], want[:3])
    assert served == 'fused_agg', (q, served)


def test_minute_with_where_order_limit_offset():
    _exact("SELECT DATE_TRUNC('minute', ts) AS m, COUNT(*) AS c FROM t WHERE site + r = 7 "
           "GROUP BY DATE_TRUNC('minute', ts) ORDER BY DATE_TRUNC('minute', ts) LIMIT 10 OFFSET 100")

def test_every_unit_on_a_timestamp():
    for u in ('second', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'):
        _exact("SELECT DATE_TRUNC('%s', ts) AS k, COUNT(*) AS c, SUM(site) FROM t WHERE r = 1 "
               "GROUP BY DATE_TRUNC('%s', ts) ORDER BY c DESC, k LIMIT 15" % (u, u))

def test_calendar_units_on_a_date():
    for u in ('day', 'week', 'month', 'quarter', 'year'):
        _exact("SELECT DATE_TRUNC('%s', d) AS k, COUNT(*) AS c FROM t GROUP BY DATE_TRUNC('%s', d) "
               "ORDER BY k LIMIT 20" % (u, u))

def test_with_a_second_key():
    _exact("SELECT DATE_TRUNC('hour', ts) AS h, site, COUNT(*) AS c FROM t WHERE r = 0 "
           "GROUP BY DATE_TRUNC('hour', ts), site ORDER BY c DESC, h, site LIMIT 12")
