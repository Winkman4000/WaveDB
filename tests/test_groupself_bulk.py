"""THE COUNTING PATH WITHOUT A CAP (2026-10-22, Q7's board): GROUP BY K + WHERE on K, ORDER BY COUNT(*) alone, no LIMIT,
past 4,096 groups -- wdb_groupself orders the board by count in two linear passes (count desc, value asc) and decodes the
values in bulk. Driven through groupself's own detect + execute (on this small fixture earlier reads -- gbcount,
dict_count -- answer some keys first through SQL); oracle: DuckDB in the total order ORDER BY COUNT(*) DESC, K."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import duckdb, sqlglot, wdb_encode, wdb_groupself as GS
from wdb_db import Database

N = 300_000
_FIX = None


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        ((i * 2654435761) %% 20011) * 1000003 + ((i * 7) %% 3) AS big,
        'v' || CAST(((i * 48271) %% 9001) AS VARCHAR) || CASE WHEN i %% 5 = 0 THEN 'x' ELSE '' END AS s,
        TIMESTAMP '2013-07-01 00:00:00' + INTERVAL ((i * 37) %% 15000) MINUTE AS tm,
        DATE '2000-01-01' + CAST(((i * 13) %% 6000) AS INTEGER) AS d
        FROM range(%d) r(i)""" % N)
    d = os.path.join(tempfile.gettempdir(), 'gsb_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['big', 'int'], ['s', 'str'], ['tm', 'datetime'], ['d', 'date']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    db.cat.add_segment('t', 't_0.wdb')
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    _FIX = (db, con, seg)
    return _FIX


def _norm(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _rows(rs):
    return [tuple(_norm(v) for v in r) for r in rs]


def _gs(q):
    """groupself's own answer: (spec, rows) -- spec None when it does not take the shape"""
    _db, _con, seg = _fixture()
    spec = GS.detect(seg, sqlglot.parse_one(q, read='duckdb'), {})
    if spec is None:
        return None, None
    h0 = GS._HITS
    out = GS.execute(seg, spec)
    if out is None:
        return spec, None
    assert GS._HITS == h0 + 1
    return spec, _rows(out[0])


def test_bulk_path_every_key_type():
    _db, con, _seg = _fixture()
    lit = {'big': '7', 's': "'v17'", 'tm': "'2013-07-01 00:37:00'", 'd': "'2000-01-14'"}
    for k in ('big', 's', 'tm', 'd'):
        q = "SELECT %s, COUNT(*) FROM t WHERE %s <> %s GROUP BY %s ORDER BY COUNT(*) DESC" % (k, k, lit[k], k)
        spec, got = _gs(q)
        assert got is not None and len(got) > GS._NOLIM_MAX, (q, spec is None)
        assert got == _rows(con.execute(q + ", %s" % k).fetchall()), (q, got[:3])


def test_swapped_projection_having_not_in():
    _db, con, _seg = _fixture()
    for q, tot in (("SELECT COUNT(*) AS c, s FROM t WHERE s <> 'v3' GROUP BY s ORDER BY c DESC", ", s"),
                   ("SELECT big, COUNT(*) AS c FROM t WHERE big <> 7 GROUP BY big HAVING COUNT(*) > 3 "
                    "ORDER BY COUNT(*) DESC", ", big"),
                   ("SELECT COUNT(*), big FROM t WHERE big NOT IN (7, 8) GROUP BY big ORDER BY COUNT(*) DESC", ", big")):
        spec, got = _gs(q)
        assert got is not None and len(got) > GS._NOLIM_MAX, q
        assert got == _rows(con.execute(q + tot).fetchall()), (q, got[:3])


def test_two_order_terms_and_switch_off_decline():
    q = "SELECT s, COUNT(*) FROM t WHERE s <> 'v3' GROUP BY s ORDER BY COUNT(*) DESC, s DESC"
    spec, got = _gs(q)
    assert spec is not None and got is None                    # more order terms: not the bulk path's order
    old = GS._BULK[0]
    try:
        GS._BULK[0] = False
        spec, got = _gs("SELECT big, COUNT(*) FROM t WHERE big <> 7 GROUP BY big ORDER BY COUNT(*) DESC")
        assert spec is not None and got is None
    finally:
        GS._BULK[0] = old


def test_sql_answers_whatever_read_serves():
    db, con, _seg = _fixture()
    for k, lit in (('big', '7'), ('s', "'v17'"), ('tm', "'2013-07-01 00:37:00'")):
        q = "SELECT %s, COUNT(*) FROM t WHERE %s <> %s GROUP BY %s ORDER BY COUNT(*) DESC" % (k, k, lit, k)
        res = db.run(q)
        w = _rows(res[0] if isinstance(res, tuple) else res)
        e = _rows(con.execute(q).fetchall())
        assert sorted(w) == sorted(e), q                      # ties are SQL's to order: compare as multisets
        assert [r[1] for r in w] == sorted((r[1] for r in w), reverse=True), q
