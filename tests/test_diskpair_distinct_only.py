"""THE COUNT LANE RETIRED (2026-10-23): diskpair declines every aggregate -- the count top-K queries it used to claim
go to the general reads (fused_agg -> wdb_bound), which beat it on every measured pair -- and keeps answering bare
SELECT DISTINCT over its blocked keys. Every answer, either way, is DuckDB's."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import sqlglot
import wdb_diskpair as DP

_FX = None


def _fixture():
    """400K rows: a (3,000 values), b (40 values), c (700 values) -- spread values, so the encoder frames them"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        CAST((i * 2654435761) % 3000 AS INTEGER) AS a,
        CAST((i * 48271) % 41 AS INTEGER) AS b,
        CAST((i * 69621) % 701 AS INTEGER) AS c
        FROM range(400000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'dpd_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['a', 'int'], ['b', 'int'], ['c', 'int']])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    db.cat.add_segment('t', 't_0.wdb')
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    _FX = (db, con, seg)
    return _FX


class _Stub:
    cols = {}
    N = 0

    def stairs(self, _c):
        return None


def test_count_lane_declines_before_any_column():
    for q in ('SELECT a, b, COUNT(*) FROM t GROUP BY a, b ORDER BY COUNT(*) DESC LIMIT 10',
              'SELECT a, COUNT(*) FROM t GROUP BY a ORDER BY COUNT(*) DESC LIMIT 10',
              'SELECT a, b, c, COUNT(*) FROM t GROUP BY a, b, c ORDER BY COUNT(*) DESC LIMIT 10 OFFSET 3'):
        assert DP.detect(_Stub(), sqlglot.parse_one(q, read='duckdb'), {}) is None, q


def test_counts_and_distincts_legal_against_duckdb():
    db, con, seg = _fixture()
    framed = [c for c in ('a', 'b', 'c') if seg.cols[c].get('code_enc') == 3]
    for q in ('SELECT a, b, COUNT(*) AS n FROM t GROUP BY a, b ORDER BY n DESC LIMIT 10',
              'SELECT a, b, c, COUNT(*) AS n FROM t GROUP BY a, b, c ORDER BY n DESC LIMIT 10'):
        assert DP.detect(seg, sqlglot.parse_one(q, read='duckdb'), {}) is None, q
        r = db.run(q)
        rows = r[0] if isinstance(r, tuple) else r
        want = sorted((int(x[-1]) for x in con.execute(q).fetchall()), reverse=True)
        assert sorted((int(x[-1]) for x in rows), reverse=True) == want, q
    assert framed, 'the fixture frames no key'          # which keys the encoder frames varies by run settings
    f0 = framed[0]
    for q in ('SELECT DISTINCT a, b FROM t', 'SELECT DISTINCT b, c FROM t', 'SELECT DISTINCT a, b, c FROM t',
              'SELECT DISTINCT a, c FROM t', 'SELECT DISTINCT b FROM t', 'SELECT DISTINCT %s FROM t' % f0):
        h0 = DP._HITS
        r = db.run(q)
        rows = r[0] if isinstance(r, tuple) else r
        assert sorted(map(tuple, rows)) == sorted(con.execute(q).fetchall()), q
        if q == 'SELECT DISTINCT %s FROM t' % f0:
            assert DP._HITS > h0, q                      # the DISTINCT lane still serves its framed keys
    for q, full in (('SELECT DISTINCT %s FROM t LIMIT 5' % f0, 'SELECT DISTINCT %s FROM t' % f0),
                    ('SELECT DISTINCT a, b FROM t LIMIT 10', 'SELECT DISTINCT a, b FROM t')):
        h0 = DP._HITS
        r = db.run(q)
        if q.startswith('SELECT DISTINCT %s FROM' % f0):
            assert DP._HITS > h0, q
        rows = r[0] if isinstance(r, tuple) else r
        n = int(q.rsplit(' ', 1)[1])
        allp = set(con.execute(full).fetchall())
        assert len(rows) == n and len(set(map(tuple, rows))) == n and all(tuple(x) in allp for x in rows), q
