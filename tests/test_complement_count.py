"""THE COMPLEMENT COUNT (2026-10-24, Q1): COUNT(*) WHERE k <> v is N - rows(k = v) - NULLs, the middle term from
k's census (V <= 2^20) or v's equality scan (past it). Both sides forced, every answer DuckDB's: a common value, a
rare one, one no row holds, a column with NULLs, a string column, = as well as <>, two COUNT(*)s."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

_FX = None


def _fixture():
    """300K rows: a (3,000 values, one value on ~60%), s (strings, 400), n (NULL on ~8%), b (2 values)"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        CAST(CASE WHEN (i * 7919) % 10 < 6 THEN 17 ELSE (i * 2654435761) % 3000 END AS INTEGER) AS a,
        's' || CAST((i * 40507) % 400 AS VARCHAR) AS s,
        CAST(CASE WHEN (i * 104729) % 100 < 8 THEN NULL ELSE (i * 15485863) % 50 END AS INTEGER) AS n,
        CAST((i // 3000) % 2 AS INTEGER) AS b
        FROM range(300000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'cmp_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['a', 'int'], ['s', 'str'], ['n', 'int'], ['b', 'int']])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    db.cat.add_segment('t', 't_0.wdb')
    _FX = (db, con)
    return _FX


QS = ['SELECT COUNT(*) FROM t WHERE a <> 17',           # the common value
      'SELECT COUNT(*) FROM t WHERE a <> 2999',
      'SELECT COUNT(*) FROM t WHERE a <> 123456',       # no row holds it
      'SELECT COUNT(*) FROM t WHERE a = 17',
      'SELECT COUNT(*) FROM t WHERE a = 123456',
      'SELECT COUNT(*) FROM t WHERE n <> 3',            # NULL <> 3 is not TRUE
      'SELECT COUNT(*) FROM t WHERE n = 3',
      "SELECT COUNT(*) FROM t WHERE s <> 's7'",
      "SELECT COUNT(*) FROM t WHERE s = 's399'",
      'SELECT COUNT(*) FROM t WHERE b <> 1',
      'SELECT COUNT(*) AS x, COUNT(*) AS y FROM t WHERE a <> 17']


def test_complement_both_sides_legal():
    """the read itself (an earlier read, dict_count, may answer the routed query first on a small table), census
    side and scan side forced; and the routed answer, whichever read gives it"""
    db, con = _fixture()
    import sqlglot, controller, wdb_complement as WC
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    cmap = {c: c for c in ('id', 'a', 's', 'n', 'b')}
    saved = WC._CENSUS_V
    ran = 0
    try:
        for cut in (1 << 30, 0):                        # every column by its census, then every one by its scan
            WC._CENSUS_V = cut
            for q in QS:
                want = [tuple(x) for x in con.execute(q).fetchall()]
                col = q.split('WHERE ')[1].split()[0]
                if seg.cols[col].get('mode') not in (0, 1, 2):
                    continue                            # the runner may store a narrow int as a sequence (mode 4)
                spec = WC.detect(seg, sqlglot.parse_one(q, read='duckdb'), cmap)
                assert spec is not None, q
                rows, _hdr = WC.execute(seg, spec)
                assert [tuple(x) for x in rows] == want, (q, cut, rows)
                assert all(type(v) is int for v in rows[0]), rows
                controller.plan_epoch_bump()
                r = db.run(q)
                assert [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)] == want, (q, cut)
                ran += 1
    finally:
        WC._CENSUS_V = saved
        controller.plan_epoch_bump()
    assert ran >= 16, ran                               # most queries ran, on both sides


def test_complement_declines_other_shapes():
    db, _con = _fixture()
    import sqlglot, wdb_complement as WC
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    cmap = {c: c for c in ('id', 'a', 's', 'n', 'b')}
    for q in ('SELECT COUNT(*) FROM t WHERE a <> 17 AND b = 1', 'SELECT COUNT(*) FROM t WHERE a > 17',
              'SELECT SUM(a) FROM t WHERE a <> 17', 'SELECT a, COUNT(*) FROM t WHERE a <> 17 GROUP BY a',
              'SELECT COUNT(*) FROM t', 'SELECT COUNT(a) FROM t WHERE a <> 17', 'SELECT COUNT(*) FROM t WHERE id <> 5'):
        assert WC.detect(seg, sqlglot.parse_one(q, read='duckdb'), cmap) is None, q
