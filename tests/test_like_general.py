"""THE LIKE FAMILY, GENERAL (2026-10-19, the LIKE shape board): every pattern form through the three LIKE shapes,
oracle DuckDB.
  - kind 'all' ('%', '%%'): every non-NULL value (was refused as too short; Q21/Q23 fell to the general path).
  - COUNT(*) WHERE s LIKE p from the census on a small dictionary; a needle no value holds counts 0 with no scan.
  - the general flag scan (encodings with no fused scanner) in parallel.
  - GROUP BY under a LIKE: one key factorized densely, the candidates only, the late aggregates (MIN, COUNT
    DISTINCT, SUM, AVG over the returned groups' rows) -- forced on with the thresholds lowered, and the same
    answers with them off. A tied LIMIT is judged legal: the counts are DuckDB's top-k in order, and every row is
    DuckDB's row for that key.
  - SELECT * WHERE s LIKE p ORDER BY ts LIMIT k (firstk) for prefix / suffix / '_' / two-part / '%' / no match."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bench'))
import numpy as np
import duckdb, wdb_encode, controller
import wdb_wherescan as WS
from wdb_db import Database

N = 200_000
_FIX = None
PATS = ['%', '%%', '%a%', 'ab%', '%yz', '%a_c%', '%a%c%', '%zzzq%', '%ж%', '%мо%', 'x%', '%5']


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        ['abc', 'xabcz', 'qyz', 'ab', 'москва', 'жук', 'a1c', 'zzz', 'x', 'yy5', 'abyz', '']
            [1 + ((i * 7919) %% 12)] || CASE WHEN i %% 9 = 0 THEN 'aqc' ELSE '' END AS s,
        'v' || ((i * 104729) %% 30011)::VARCHAR || CASE WHEN i %% 5 = 0 THEN 'abc' ELSE 'xy' END AS b,
        CASE WHEN i %% 13 = 0 THEN '' ELSE 'g' || (CAST(floor(3000 * pow(((i * 6007) %% 99989) / 99989.0, 3)) AS BIGINT))::VARCHAR END AS g,
        (i * 31) %% 7001 AS u,
        TIMESTAMP '2013-07-01 00:00:00' + INTERVAL (i) SECOND AS ts,
        CASE WHEN (i * 7) %% 10 = 0 THEN 'p' || ((i * 13) %% 500)::VARCHAR ELSE '' END AS p
        FROM range(%d) r(i)""" % N)
    d = os.path.join(tempfile.gettempdir(), 'lkg_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['s', 'str'], ['b', 'str'], ['g', 'str'], ['u', 'int'], ['ts', 'datetime'],
                           ['p', 'str']])
    # the suite runner lets toys encode as sequences (WDB_SEQ_*): this fixture is the ClickBench storage, dictionaries
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    db.cat.add_segment('t', 't_0.wdb')
    _FIX = (db, con, d)
    return _FIX


def _rows(res):
    return [tuple(v.decode() if isinstance(v, bytes) else v for v in r) for r in (res[0] if isinstance(res, tuple) else res)]


def _norm(v):
    """cells compared as DuckDB would print them: floats rounded, timestamps as microseconds since 1970 (firstk
    returns a datetime column's stored microseconds)"""
    import datetime
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, datetime.datetime):
        return int((v - datetime.datetime(1970, 1, 1)) / datetime.timedelta(microseconds=1))
    if isinstance(v, np.datetime64):
        return int(v.astype('datetime64[us]').astype(np.int64))
    return v


def _legal(sql, w, con, ci):
    """the tied-LIMIT legality: counts are DuckDB's top-k in order, every row is DuckDB's row for its key"""
    d = con.execute(sql).fetchall()
    assert [int(x[ci]) for x in w] == [int(x[ci]) for x in d], (sql, w[:3], d[:3])
    inner = sql[:sql.upper().index(' ORDER BY')]
    truth = {x[0]: tuple(_norm(v) for v in x) for x in con.execute(inner).fetchall()}
    for x in w:
        assert tuple(_norm(v) for v in x) == truth.get(x[0]), (sql, x, truth.get(x[0]))
    assert len(set(x[0] for x in w)) == len(w), sql


def _set(dense, late):
    WS._DENSE_MIN_ROWS = 1 if dense else (1 << 40)
    WS._LATE_MIN_ROWS = 1 if late else (1 << 40)


def test_count_every_form():
    db, con, _d = _fixture()
    for p in PATS:
        for col in ('s', 'b'):
            q = "SELECT COUNT(*) FROM t WHERE %s LIKE '%s'" % (col, p)
            assert _rows(db.run(q)) == [tuple(x) for x in con.execute(q).fetchall()], q
    q = "SELECT COUNT(*) FROM t WHERE s NOT LIKE '%'"
    assert _rows(db.run(q)) == [(0,)], q


def test_grouped_like_every_form_new_and_old_roads():
    db, con, _d = _fixture()
    old = (WS._DENSE_MIN_ROWS, WS._LATE_MIN_ROWS)
    try:
        for p in PATS:
            q21 = ("SELECT g, MIN(b), COUNT(*) AS c FROM t WHERE b LIKE '%s' AND g <> '' GROUP BY g "
                   "ORDER BY c DESC LIMIT 10" % p)
            q22 = ("SELECT g, MIN(b), MIN(s), COUNT(*) AS c, COUNT(DISTINCT u), SUM(u), AVG(u) FROM t "
                   "WHERE s LIKE '%s' AND b NOT LIKE '%%abc%%' AND g <> '' GROUP BY g ORDER BY c DESC LIMIT 10" % p)
            for q, ci in ((q21, 2), (q22, 3)):
                got = {}
                for dense, late in ((True, True), (False, False), (True, False)):
                    _set(dense, late)
                    w = _rows(db.run(q))
                    assert controller._SERVED[0] == 'wherescan', (q, controller._SERVED[0])
                    _legal(q, w, con, ci)
                    got[(dense, late)] = w
                assert got[(True, True)] == got[(False, False)] == got[(True, False)], q
    finally:
        WS._DENSE_MIN_ROWS, WS._LATE_MIN_ROWS = old


def test_firstk_every_form():
    db, con, _d = _fixture()
    for p in PATS:
        q = "SELECT * FROM t WHERE s LIKE '%s' ORDER BY ts LIMIT 5" % p
        w = _rows(db.run(q))
        assert controller._SERVED[0] == 'firstk', (q, controller._SERVED[0])
        d = con.execute(q).fetchall()
        assert [tuple(_norm(v) for v in x) for x in w] == [tuple(_norm(v) for v in x) for x in d], (q, w[:2], d[:2])


def test_potency_picks_the_driver():
    """THE POTENCY PICK: a LIKE keeping ~45% of the rows and p <> '' keeping 10% -- the <> drives (the fixed ladder
    drove on the LIKE); a LIKE keeping few rows still drives over a <> keeping most. Same answers with the pick off."""
    db, con, _d = _fixture()
    q1 = ("SELECT g, MIN(s), COUNT(*) AS c FROM t WHERE s LIKE '%a%' AND p <> '' AND g <> '' GROUP BY g "
          "ORDER BY c DESC LIMIT 10")
    q2 = ("SELECT g, MIN(b), COUNT(*) AS c FROM t WHERE b LIKE '%v1000%' AND g <> '' GROUP BY g "
          "ORDER BY c DESC LIMIT 10")
    for q, want in ((q1, 'p'), (q2, 'b')):
        WS._POT_LAST[0] = None
        w = _rows(db.run(q))
        assert controller._SERVED[0] == 'wherescan', (q, controller._SERVED[0])
        best = WS._POT_LAST[0][0] if WS._POT_LAST[0] else None
        sp = WS.detect(db.open_segment(db.cat.segment_paths('t')[0], 't'), __import__('sqlglot').parse_one(q), None)
        kind, i = best
        col = sp['likes'][i][0] if kind == 'like' else sp['eqs'][i][0]
        assert col == want, (q, best, WS._POT_LAST[0])
        _legal(q, w, con, 2)
        WS._POTENCY[0] = False
        try:
            assert _rows(db.run(q)) == w, q
        finally:
            WS._POTENCY[0] = True


if __name__ == '__main__':
    try:
        test_count_every_form()
        test_grouped_like_every_form_new_and_old_roads()
        test_firstk_every_form()
        test_potency_picks_the_driver()
        print('ok')
    finally:
        if _FIX is not None:
            shutil.rmtree(_FIX[2], ignore_errors=True)
