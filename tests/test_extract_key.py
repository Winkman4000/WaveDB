"""EXTRACT(field FROM col) -- and its named forms, MINUTE(col), YEAR(col) -- as a GROUP BY key rides the dictionary
in fused_agg: the field of the V distinct times (wdb_sql.extract_field, the row evaluator's own definition),
never the N rows; a GROUP BY that names the key's SELECT alias groups by the expression. Before, the Q18 shape
with any key other than tripletop's own sent every fill-in to the general scan (JavaEnable x minute x Age:
10.8 s, 2026-10-13). Checked against DuckDB."""
import sys, os, tempfile, uuid, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, controller
from wdb_db import Database

_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'exk_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    _CON.execute("""CREATE TABLE t AS SELECT i AS id, i % 40 AS site, i % 2 AS r, (i * 7919) % 299993 AS u,
        TIMESTAMP '2013-01-01 00:00:00' + INTERVAL ((i * 7919) % 31536000) SECOND AS ts,
        DATE '2013-01-01' + ((i * 13) % 365)::INT AS d FROM range(300000) t(i)""")
    _DB = Database.create(d)
    pq = os.path.join(d, 't.parquet')
    _CON.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    _DB.cat.add_table('t', [['id', 'int'], ['site', 'int'], ['r', 'int'], ['u', 'int'], ['ts', 'datetime'], ['d', 'date']])
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


def test_q18_shape_by_alias():
    _exact("SELECT site, extract(minute FROM ts) AS m, r, COUNT(*) FROM t GROUP BY site, m, r "
           "ORDER BY COUNT(*) DESC, site, m, r LIMIT 10")

def test_every_field_on_a_timestamp():
    for f in ('second', 'minute', 'hour', 'day', 'month', 'quarter', 'year', 'dow', 'doy'):
        _exact("SELECT extract(%s FROM ts) AS k, COUNT(*) AS c, SUM(site) FROM t WHERE r = 1 "
               "GROUP BY extract(%s FROM ts) ORDER BY c DESC, k LIMIT 15" % (f, f))

def test_fields_on_a_date():
    for f in ('day', 'month', 'year', 'dow'):
        _exact("SELECT extract(%s FROM d) AS k, COUNT(*) AS c FROM t GROUP BY k ORDER BY k LIMIT 20" % f)

def test_named_forms():
    _exact("SELECT minute(ts) AS k, site, COUNT(*) AS c FROM t WHERE site < 5 GROUP BY minute(ts), site "
           "ORDER BY c DESC, k, site LIMIT 12")
    _exact("SELECT year(d) AS y, COUNT(*) AS c FROM t GROUP BY year(d) ORDER BY y")

def test_singleton_peel_with_an_expression_key():
    """a near-unique key beside a minute key: the peel groups the repeaters and reads the minute through the
    dictionary map (WatchID x minute x K on the Q18 board)"""
    import wdb_peel
    h0 = wdb_peel._HITS
    _exact("SELECT u, extract(minute FROM ts) AS m, site, COUNT(*) AS c FROM t GROUP BY u, m, site "
           "ORDER BY c DESC, u, m, site LIMIT 10")
    _exact("SELECT u, minute(ts) AS m, COUNT(*) AS c, SUM(site) FROM t WHERE r = 1 GROUP BY u, minute(ts) "
           "ORDER BY m DESC, u LIMIT 7")
    assert wdb_peel._HITS == h0 + 2
