"""THE SORTED-PROJECTION FAMILY, GENERAL (2026-10-21, the Q24-Q26 board): oracle DuckDB.
  - sorted_proj (wdb_valsort): SELECT c WHERE <predicate on c> ORDER BY c [DESC] LIMIT k [OFFSET o] for any
    dictionary column -- numbers, a big (mode-2) dictionary, a nullable column (NULLS LAST), text, time -- by a
    walk of the value-sorted dictionary (it took only text with `c <> ''`; numbers fell to the general top-k
    door: Age 7.0 s against DuckDB's 0.036 s).
  - firstk (wdb_firstk): SELECT cols WHERE <predicate on one column> ORDER BY staircase [, more keys] LIMIT k
    [OFFSET o] by the walk in file order, the run certificate settling a second key (it took only SELECT * with a
    LIKE)."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import datetime
import numpy as np
import duckdb, wdb_encode, controller
from wdb_db import Database

N = 200_000
_FIX = None


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        CASE WHEN i %% 3 = 0 THEN 31 ELSE (i * 7) %% 60 END AS a,
        (i * 48271) %% 100003 AS w,
        CASE WHEN i %% 11 = 0 THEN NULL ELSE (i * 13) %% 97 END AS n,
        ['abc', 'xabcz', 'qyz', 'ab', 'жук', 'a1c', 'zzz', '', 'yy5', 'abyz'][1 + ((i * 7919) %% 10)] AS s,
        TIMESTAMP '2013-07-01 00:00:00' + INTERVAL (i) SECOND AS ts,
        TIMESTAMP '2013-07-01 00:00:00' + INTERVAL (i // 60) MINUTE AS tm
        FROM range(%d) r(i)""" % N)
    d = os.path.join(tempfile.gettempdir(), 'srtg_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['a', 'int'], ['w', 'int'], ['n', 'int'], ['s', 'str'],
                           ['ts', 'datetime'], ['tm', 'datetime']])
    # the ClickBench storage is dictionaries: the suite runner's toy sequences (WDB_SEQ_*) are off while encoding
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    db.cat.add_segment('t', 't_0.wdb')
    _FIX = (db, con)
    return _FIX


def _norm(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, datetime.datetime):
        return int((v - datetime.datetime(1970, 1, 1)) / datetime.timedelta(microseconds=1))
    if isinstance(v, np.datetime64):
        return int(v.astype('datetime64[us]').astype(np.int64))
    return v


def test_fixture_is_dictionaries():
    # the reads serve dictionary columns (modes 0/1/2); a sequence column (mode 4: id, ts) is declined
    db, _con = _fixture()
    import wdb_engine
    seg = wdb_engine.Segment(db.cat.segment_paths('t')[0])
    assert seg.cols['w']['mode'] == 2 and seg.cols['n']['has_null'] and seg.cols['tm']['dt'] == 3, \
        {c: (v.get('mode'), v.get('dt')) for c, v in seg.cols.items()}


def _same(sql, read):
    db, con = _fixture()
    controller._SERVED[0] = None
    res = db.run(sql)
    w = [tuple(_norm(v) for v in r) for r in (res[0] if isinstance(res, tuple) else res)]
    served = controller._SERVED[0]
    e = [tuple(_norm(v) for v in r) for r in con.execute(sql).fetchall()]
    assert served == read, (sql, served)
    assert w == e, (sql, w[:5], e[:5])


def test_sorted_proj_numbers_big_dictionary_and_offset():
    for q in ("SELECT a FROM t WHERE a <> 31 ORDER BY a LIMIT 10",
              "SELECT a FROM t WHERE a <> 31 ORDER BY a DESC LIMIT 7 OFFSET 3",
              "SELECT a FROM t ORDER BY a LIMIT 5",
              "SELECT w FROM t WHERE w > 50000 ORDER BY w LIMIT 10 OFFSET 2",
              "SELECT w FROM t WHERE w BETWEEN 1000 AND 2000 OR w IN (5, 7) ORDER BY w DESC LIMIT 12",
              "SELECT w FROM t WHERE w <> 0 ORDER BY w DESC LIMIT 4"):
        _same(q, 'sorted_proj')


def test_sorted_proj_nulls_text_time():
    _db, con = _fixture()
    nn = con.execute("SELECT COUNT(n) FROM t").fetchone()[0]
    for q in ("SELECT n FROM t WHERE n <> 3 ORDER BY n LIMIT 10",
              "SELECT n FROM t ORDER BY n DESC LIMIT 5 OFFSET %d" % (nn - 2),     # runs into the NULLs (last)
              "SELECT n FROM t ORDER BY n LIMIT 6 OFFSET %d" % (nn - 3),
              "SELECT s FROM t WHERE s LIKE 'ab%' ORDER BY s LIMIT 10",
              "SELECT s FROM t WHERE s NOT LIKE '%a%' ORDER BY s DESC LIMIT 10",
              "SELECT s FROM t WHERE s <> '' ORDER BY s LIMIT 10",
              "SELECT tm FROM t WHERE tm >= '2013-07-01 12:00:00' ORDER BY tm LIMIT 3",
              "SELECT tm FROM t WHERE tm < '2013-07-02 00:00:00' ORDER BY tm DESC LIMIT 70"):
        _same(q, 'sorted_proj')


def test_firstk_any_filter_projection_offset():
    for q in ("SELECT a FROM t WHERE a <> 31 ORDER BY ts LIMIT 10",
              "SELECT id, s FROM t WHERE w BETWEEN 1000 AND 300000 ORDER BY ts LIMIT 10 OFFSET 5",
              "SELECT id FROM t WHERE n IS NULL ORDER BY ts LIMIT 7",
              "SELECT * FROM t WHERE s LIKE '%yz' ORDER BY ts LIMIT 5"):
        _same(q, 'firstk')


def test_firstk_run_certificate_settles_the_second_key():
    # tm repeats for 60 rows: the k-th row's run is read to its end and ordered by (tm, the second key)
    for q in ("SELECT tm, s FROM t WHERE s <> '' ORDER BY tm, s LIMIT 10",
              "SELECT tm, a FROM t WHERE a IN (3, 4, 5) ORDER BY tm, a DESC LIMIT 25 OFFSET 7",
              "SELECT tm, w FROM t WHERE n IS NULL ORDER BY tm, w LIMIT 10",
              "SELECT tm, w FROM t WHERE a <> 31 ORDER BY tm, w DESC LIMIT 61"):
        _same(q, 'firstk')
