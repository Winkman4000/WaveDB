"""THE COUNTED DISTINCT (2026-10-24): a bare SELECT DISTINCT is asked as GROUP BY + COUNT(*) with the count dropped --
ahead of diskpair on the shapes diskpair would answer, and before the general scan for any bare DISTINCT nothing
took. Every answer, the counted form on or off, is DuckDB's; the hidden count never shows."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import sqlglot
import wdb_counted as WC

_FX = None


def _fixture():
    """300K rows: a (3,000 values), b (41 values), c (701 values), s (strings, 500 values) -- spread values"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        CAST((i * 2654435761) % 3000 AS INTEGER) AS a,
        CAST((i * 48271) % 41 AS INTEGER) AS b,
        CAST((i * 69621) % 701 AS INTEGER) AS c,
        'v' || CAST((i * 40507) % 500 AS VARCHAR) AS s
        FROM range(300000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'cdt_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['a', 'int'], ['b', 'int'], ['c', 'int'], ['s', 'str']])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    db.cat.add_segment('t', 't_0.wdb')
    _FX = (db, con)
    return _FX


def _p(q):
    return sqlglot.parse_one(q, read='duckdb')


def test_which_distincts_are_bare():
    yes = ('SELECT DISTINCT a FROM t', 'SELECT DISTINCT a + 1 AS x FROM t WHERE b = 3',
           'SELECT DISTINCT a, length(s) FROM t ORDER BY a LIMIT 5')
    no = ('SELECT a FROM t', 'SELECT DISTINCT ON (a) a, b FROM t', 'SELECT DISTINCT * FROM t',
          'SELECT DISTINCT COUNT(*) FROM t', 'SELECT DISTINCT a FROM t GROUP BY a', 'SELECT DISTINCT 1 FROM t',
          'SELECT DISTINCT a, SUM(b) OVER () FROM t')
    for q in yes:
        assert WC.bare_distinct(_p(q)), q
    for q in no:
        assert not WC.bare_distinct(_p(q)), q


def _rows(r):
    rows, hdr = r if isinstance(r, tuple) else (r, None)
    return sorted(tuple(x) for x in rows), hdr


def test_answers_legal_with_and_without_the_counted_form():
    db, con = _fixture()
    import controller
    qs = ['SELECT DISTINCT a + 1 AS x FROM t',
          'SELECT DISTINCT a, c FROM t',
          'SELECT DISTINCT b * 2 AS x FROM t WHERE c < 300',
          'SELECT DISTINCT length(s) AS n, b FROM t',
          'SELECT DISTINCT a + 1 AS b FROM t',                      # the alias names another column
          'SELECT DISTINCT lower(s) AS x FROM t WHERE b <> 3']
    saved = os.environ.get('WDB_COUNTED_DISTINCT')
    try:
        for q in qs:
            want = sorted(con.execute(q).fetchall())
            got = {}
            for on in ('1', '0'):
                os.environ['WDB_COUNTED_DISTINCT'] = on
                controller.plan_epoch_bump()
                rows, hdr = _rows(db.run(q))
                assert [tuple(x) for x in rows] == [tuple(x) for x in want], (q, on, rows[:3], want[:3])
                if hdr is not None:
                    assert WC._HIDDEN not in [str(h) for h in hdr], (q, hdr)
                got[on] = rows
            assert got['1'] == got['0'], q
    finally:
        if saved is None:
            os.environ.pop('WDB_COUNTED_DISTINCT', None)
        else:
            os.environ['WDB_COUNTED_DISTINCT'] = saved
        controller.plan_epoch_bump()


def test_ordered_and_limited_distinct_legal():
    db, con = _fixture()
    import controller
    controller.plan_epoch_bump()
    q = 'SELECT DISTINCT a + 1 AS x FROM t ORDER BY x DESC LIMIT 7'
    rows = db.run(q)
    rows = rows[0] if isinstance(rows, tuple) else rows
    assert [tuple(r) for r in rows] == con.execute(q).fetchall()
    q = 'SELECT DISTINCT b * 3 AS x FROM t LIMIT 5'
    rows = db.run(q)
    rows = rows[0] if isinstance(rows, tuple) else rows
    allv = set(con.execute('SELECT DISTINCT b * 3 AS x FROM t').fetchall())
    assert len(rows) == 5 and len({tuple(r) for r in rows}) == 5 and all(tuple(r) in allv for r in rows)


def test_counted_form_serves():
    """a derived DISTINCT reaches the counted form -- ahead of diskpair or before the general scan"""
    db, _con = _fixture()
    import controller
    controller.plan_epoch_bump()
    h0, f0 = WC._HITS[0], WC._FALLBACK[0]
    db.run('SELECT DISTINCT a + 1 AS x FROM t WHERE b = 3')     # nothing takes it: before the general scan
    assert WC._FALLBACK[0] == f0 + 1 and str(controller._SERVED[0]).startswith('counted:'), controller._SERVED[0]
    controller.plan_epoch_bump()
    db.run('SELECT DISTINCT a + 1 AS x FROM t')                 # ahead of diskpair (or, unframed, before the scan)
    assert WC._HITS[0] + WC._FALLBACK[0] >= h0 + f0 + 2, controller._SERVED[0]


def test_counted_request_groups_by_the_alias_unless_it_names_another_column():
    db, _con = _fixture()
    import read_methods as R
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    cmap = {c: c for c in ('id', 'a', 'b', 'c', 's')}
    for q, want in (('SELECT DISTINCT a + 1 AS x FROM t', 'GROUP BY x'),
                    ('SELECT DISTINCT a + 1 AS b FROM t', 'GROUP BY a + 1'),
                    ('SELECT DISTINCT a AS a FROM t', 'GROUP BY a'),
                    ('SELECT DISTINCT a, c FROM t', 'GROUP BY a, c')):
        ctx = R.ReadContext(db, 't', seg, seg.path, _p(q), cmap, q, True)
        sql = WC.counted_ctx(ctx).sql
        assert want in sql and 'COUNT(*) AS %s' % WC._HIDDEN in sql and 'DISTINCT' not in sql, (q, sql)
