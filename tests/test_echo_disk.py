"""THE ECHO, DISK-ONLY (2026-10-21, Q19's board): oracle DuckDB. SELECT c FROM t WHERE c = lit [LIMIT k] [OFFSET o]
is the literal's value repeated by its code's count (wdb_echo) -- counted from the stored codes, so it serves with
sidecars off (the shelf-counted echo inside the general scan never did). A large dictionary that is no staircase is
declined (wherescan's equality scan keeps it)."""
import sys, os, uuid, tempfile, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
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
        TIMESTAMP '2013-07-01 00:00:00' + INTERVAL (i // 60) MINUTE AS tm
        FROM range(%d) r(i)""" % N)
    d = os.path.join(tempfile.gettempdir(), 'echo_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['a', 'int'], ['w', 'int'], ['n', 'int'], ['s', 'str'], ['tm', 'datetime']])
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
    if isinstance(v, np.generic):
        return v.item()
    return v


def _run(sql):
    db, con = _fixture()
    controller._SERVED[0] = None
    res = db.run(sql)
    w = [tuple(_norm(v) for v in r) for r in (res[0] if isinstance(res, tuple) else res)]
    e = [tuple(_norm(v) for v in r) for r in con.execute(sql).fetchall()]
    return controller._SERVED[0], sorted(w), sorted(e)


def test_echo_counts_and_values():
    for q in ("SELECT a FROM t WHERE a = 31",                     # a third of the rows
              "SELECT a FROM t WHERE a = 7",
              "SELECT a FROM t WHERE a = 9999",                   # absent: no rows
              "SELECT n FROM t WHERE n = 3",                      # NULL rows are another code
              "SELECT s FROM t WHERE s = 'жук'",
              "SELECT s FROM t WHERE s = ''",
              "SELECT tm FROM t WHERE tm = '2013-07-01 01:05:00'"):
        served, w, e = _run(q)
        assert served == 'echo', (q, served)
        assert w == e, (q, len(w), len(e), w[:2], e[:2])


def test_echo_limit_and_offset():
    _db, con = _fixture()
    c31 = con.execute("SELECT COUNT(*) FROM t WHERE a = 31").fetchone()[0]
    for q, n in (("SELECT a FROM t WHERE a = 31 LIMIT 5", 5),
                 ("SELECT a FROM t WHERE a = 31 LIMIT 5 OFFSET 3", 5),
                 ("SELECT a FROM t WHERE a = 31 LIMIT 10 OFFSET %d" % (c31 - 4), 4),
                 ("SELECT a FROM t WHERE a = 31 OFFSET %d" % (c31 + 1), 0),
                 ("SELECT a FROM t WHERE a = 31 LIMIT 0", 0)):
        served, w, e = _run(q)
        # with OFFSET the reads that slice it themselves go first (dict_count, when its sidecar is there)
        assert served == 'echo' or 'OFFSET' in q, (q, served)
        assert w == e and len(w) == n, (q, served, len(w), len(e))


def test_echo_declines_a_large_dictionary():
    served, w, e = _run("SELECT w FROM t WHERE w = 777")
    assert served != 'echo' and w == e, (served, w, e)
