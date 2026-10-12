"""DECIDE BEFORE PAYING: a grouped single-table WHERE holding a LIKE on a string dictionary too big for the
code LUT always ends in fused_agg declining -- so it declines first, before the cascade's censuses and the
group keys are paid for (a DATE_TRUNC key paid 0.7 s warm, 7 s cold, to reach the refusal; 2026-10-12).
The singleton peel, which reads the WHERE as one mask, keeps its chance. The LUT limit is shrunk here so a
small dictionary counts as too big; every answer against DuckDB."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, wdb_join, wdb_peel, controller
import wdb_measure_runtime as RT
from wdb_db import Database

_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'lkd_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    _CON.execute("""CREATE TABLE t AS SELECT i AS id, i % 40 AS site, (i * 7) % 300 AS a,
        'phrase ' || CAST((i * 13) % 2000 AS VARCHAR) AS s,
        CASE WHEN i % 97 = 0 AND i > 0 THEN (i - 1) * 5 ELSE i * 5 END AS u FROM range(200000) t(i)""")
    _DB = Database.create(d)
    pq = os.path.join(d, 't.parquet')
    _CON.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    _DB.cat.add_table('t', [['id', 'int'], ['site', 'int'], ['a', 'int'], ['s', 'string'], ['u', 'int']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None: os.environ[k] = v
    _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON


def _run(q):
    db, con = _fixture()
    orig = RT.code_lut_fits
    RT.code_lut_fits = lambda n: n <= 100            # every string dictionary here is 'too big'
    try:
        res = db.run(q)
    finally:
        RT.code_lut_fits = orig
    got = sorted(tuple(r) for r in (res[0] if isinstance(res, tuple) else res))
    want = sorted(tuple(r) for r in con.execute(q).fetchall())
    assert got == want, (q, got[:3], want[:3])
    return controller._SERVED[0]


def test_like_on_a_too_big_dictionary_answers_elsewhere():
    s = _run("SELECT a, site, COUNT(*) AS c FROM t WHERE site < 30 AND s LIKE '%99%' GROUP BY a, site ORDER BY c DESC, a, site LIMIT 10")
    assert s != 'fused_agg', s

def test_the_peel_keeps_its_chance():
    h0 = wdb_peel._HITS
    _run("SELECT u, COUNT(*) AS c FROM t WHERE s LIKE '%99%' GROUP BY u ORDER BY c DESC, u LIMIT 10")
    assert wdb_peel._HITS == h0 + 1 or controller._SERVED[0] != 'fused_agg'

def test_a_small_dictionary_still_fuses():
    db, con = _fixture()
    q = "SELECT a, COUNT(*) AS c FROM t WHERE site + 1 < 30 AND s LIKE '%99%' GROUP BY a ORDER BY c DESC, a LIMIT 10"
    got = sorted(tuple(r) for r in (lambda r: r[0] if isinstance(r, tuple) else r)(db.run(q)))
    assert got == sorted(tuple(r) for r in con.execute(q).fetchall())
