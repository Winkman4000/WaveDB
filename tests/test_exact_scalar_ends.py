"""THE EXACT SCALAR, WIDER (2026-10-21, the Q6 / Q29 boards): oracle DuckDB. wdb_join._exact_scalar answers
no-WHERE scalar aggregates from the dictionary; it now also takes
  - text MIN / MAX: the value-ordered dictionary's two ends (code 0, the last non-NULL code) -- two point reads
    (text fell to the general path: MIN(OriginalURL), MAX(OriginalURL) 8.3 s against DuckDB's 0.5 s);
  - the affine sum SUM(col + k) = SUM(col) + k * COUNT(col), exact at any magnitude, for a dictionary of any size
    (wdb_affinesum serves only dictionaries up to 1M values and only when every select item is affine; past
    that the shape fell to the general path: Q29 over WatchID / UserID / RefererHash timed out at 10 s)."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
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
        ((i * 48271) %% 100003) * 90000000000000 + 7 AS big,
        (i * 7) %% 1000 AS r,
        CASE WHEN i %% 11 = 0 THEN NULL ELSE (i * 13) %% 97 - 40 END AS n,
        ['abc', 'xabcz', 'qyz', 'ab', 'жук', 'a1c', 'zzz', '', 'yy5', 'abyz'][1 + ((i * 7919) %% 10)] AS s,
        CASE WHEN i %% 7 = 0 THEN NULL ELSE 'u' || CAST((i * 31) %% 5003 AS VARCHAR) END AS sn
        FROM range(%d) r(i)""" % N)
    d = os.path.join(tempfile.gettempdir(), 'exsc_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['big', 'int'], ['r', 'int'], ['n', 'int'], ['s', 'str'], ['sn', 'str']])
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
    return v.decode() if isinstance(v, bytes) else v


def _same(sql, read):
    db, con = _fixture()
    controller._SERVED[0] = None
    res = db.run(sql)
    w = [tuple(_norm(v) for v in r) for r in (res[0] if isinstance(res, tuple) else res)]
    served = controller._SERVED[0]
    e = [tuple(_norm(v) for v in r) for r in con.execute(sql).fetchall()]
    assert served == read, (sql, served)
    assert w == e, (sql, w, e)


def test_fixture_is_dictionaries():
    db, _con = _fixture()
    import wdb_engine
    seg = wdb_engine.Segment(db.cat.segment_paths('t')[0])
    m = {c: seg.cols[c].get('mode') for c in ('big', 'n', 's', 'sn')}
    assert m['big'] == 2 and m['n'] in (0, 1) and m['s'] in (0, 1) and m['sn'] in (0, 1), m
    assert seg.cols['sn']['has_null'] and seg.cols['n']['has_null']


def test_text_min_max_from_the_dictionary_ends():
    for q in ("SELECT MIN(s), MAX(s) FROM t",
              "SELECT MAX(sn), MIN(sn) FROM t",                      # NULL is the last code: MAX skips it
              "SELECT MIN(s), COUNT(*), MAX(sn), SUM(n) FROM t"):
        _same(q, 'fused_agg')


def test_affine_sum_exact_past_int64():
    for q in ("SELECT SUM(big), SUM(big + 1), SUM(big + 2), COUNT(*) FROM t",     # past 2^63: exact
              "SELECT SUM(5 + big), MIN(s), SUM(big) FROM t",
              "SELECT SUM(n + 3), SUM(n), COUNT(n), MAX(s) FROM t"):             # NULLs: k * COUNT(n), not N
        _same(q, 'fused_agg')


def test_affinesum_keeps_its_own_shape():
    _same("SELECT SUM(r), SUM(r + 1), SUM(r + 2) FROM t", 'affinesum')     # small, non-null: its own read
