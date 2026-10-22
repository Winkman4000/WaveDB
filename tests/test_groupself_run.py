"""THE RUN ORDER (2026-10-22, WatchID): groupself's bulk path, asked for columns (db.execute), leaves the biggest count
run unwritten -- its rows are the census board read left to right, walked per fetched batch (wdb_result.RunOrder) --
and sorts only the squares outside it. Keys here: u (count 1 nearly everywhere, a few 2s and 3s above it), v (count 2
nearly everywhere, heavy squares above, singletons below), w (text over u, past the lazy-text line), tm (times over u).
Oracle: DuckDB in the total order ORDER BY COUNT(*) DESC, key."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import duckdb, sqlglot, wdb_encode, wdb_kernels, wdb_result as WR, wdb_groupself as GS
from wdb_db import Database

N = 300_000
_FIX = None


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id, p,
        CASE WHEN p %% 1000 = 1 THEN p - 1 WHEN p %% 97 = 0 AND p > 0 THEN p - 1 ELSE p END AS u
        FROM (SELECT i, (i * 7919) %% %d AS p FROM range(%d) r(i))""" % (N, N))
    con.execute("""CREATE TABLE t2 AS SELECT id, u,
        CASE WHEN p % 5003 = 0 THEN 7 WHEN p % 101 = 0 THEN 1000000 + p ELSE p // 2 END AS v,
        'w' || LPAD(CAST(CASE WHEN p % 5003 = 0 THEN 7 WHEN p % 101 = 0 THEN 1000000 + p ELSE p // 3 END
                         AS VARCHAR), 7, '0') AS w,
        TIMESTAMP '2013-07-01 00:00:00' + INTERVAL (u) SECOND AS tm,
        CASE WHEN p % 7 = 0 THEN NULL ELSE p // 2 END AS nk
        FROM t""")
    d = os.path.join(tempfile.gettempdir(), 'gsr_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT id, u, v, w, tm, nk FROM t2 ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['u', 'int'], ['v', 'int'], ['w', 'str'], ['tm', 'datetime'],
                           ['nk', 'int']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    db.cat.add_segment('t', 't_0.wdb')
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    con.execute("CREATE VIEW tt AS SELECT id, u, v, w, tm, nk FROM t2")
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


def _want(q, k):
    _db, con, _seg = _fixture()
    return _rows(con.execute(q.replace(' FROM t ', ' FROM tt ') + ", %s" % k).fetchall())


def _cursor(q):
    """groupself's columnar answer for q, as db.execute wraps it"""
    db, _con, seg = _fixture()
    spec = GS.detect(seg, sqlglot.parse_one(q, read='duckdb'), {})
    assert spec is not None, q
    WR.WANT[0] = True
    lz = GS._LAZY_TEXT_V
    GS._LAZY_TEXT_V = 1000           # the fixture's text dictionary (~100K values) counts as a big one here
    try:
        out = GS.execute(seg, spec)
    finally:
        WR.WANT[0] = False
        GS._LAZY_TEXT_V = lz
    assert out is not None and isinstance(out[0], WR.Columns), q
    return WR.Cursor(out[0], out[1], db=db), out[0]


_QS = (('u', '5'), ('v', '9'), ('w', "'w0000009'"), ('tm', "'2013-07-01 00:00:05'"))


def _q(k, lit):
    return "SELECT %s, COUNT(*) FROM t WHERE %s <> %s GROUP BY %s ORDER BY COUNT(*) DESC" % (k, k, lit, k)


def test_run_order_serves_every_key_and_fetch_form():
    for k, lit in _QS:
        q = _q(k, lit)
        want = _want(q, k)
        cur, cols = _cursor(q)
        assert any(isinstance(cc, WR.OrderedColumn) for cc in cols.cols), k     # the run order answered
        o = cols.cols[0].o
        assert o.R > 65536 and o.A > 0, (k, o.A, o.R)
        if k in ('v', 'w'):
            assert o.tail.size > 0 and int(o.c) == (2 if k == 'v' else 3)   # smaller counts below the run
        assert cur.rowcount == len(want)
        got = [cur.fetchone()] + cur.fetchmany(37) + cur.fetchmany(2048) + cur.fetchmany(100000)
        got += [r for r in cur]                        # iteration, 2048 a batch
        assert _rows(got) == want, k
        cur, _c = _cursor(q)                           # columnar forms after a partial fetch
        cur.fetchmany(10)
        npd = cur.fetchnumpy()
        names = list(npd)
        assert [_norm(v) for v in npd[names[1]].tolist()] == [r[1] for r in want[10:]], k
        tbl = _cursor(q)[0].arrow()
        assert tbl.num_rows == len(want) and tbl.column(1).to_pylist() == [r[1] for r in want], k


def test_run_order_any_range_any_direction():
    rng = np.random.default_rng(7)
    for pick in ((1, 1, 1, 1, 2, 3, 9), (2, 2, 2, 2, 1, 1, 5, 40, 3), (1, 0, 0, 1, 1)):
        board = rng.choice(np.array(pick, np.int64), size=50_000)
        h = wdb_kernels.count_hist(board, 64)
        cs = int(np.argmax(h[1:65])) + 1
        rest = wdb_kernels.off_run(board, cs)
        o = wdb_kernels.count_desc_order(board[rest])
        rest = rest[o]
        A = int(np.count_nonzero(board[rest] > cs))
        ro = WR.RunOrder(board, cs, rest[:A], int(h[cs]), rest[A:])
        full = wdb_kernels.count_desc_order(board)
        assert ro.n == full.size == int(np.count_nonzero(board))
        for _ in range(200):
            lo = int(rng.integers(0, ro.n)); hi = int(rng.integers(lo, ro.n + 5))
            assert np.array_equal(ro.codes(lo, hi), full[lo:hi]), (pick, lo, hi)
        assert np.array_equal(ro.codes(0, ro.n), full)


def test_switch_line_declines_to_the_full_order():
    q = _q('u', '5')
    want = _want(q, 'u')
    old = GS._RUN_MIN[0]
    try:
        GS._RUN_MIN[0] = 1 << 40                       # no run is that big: count_desc_order, as before
        cur, cols = _cursor(q)
        assert not any(isinstance(cc, WR.OrderedColumn) for cc in cols.cols)
        assert _rows(cur.fetchall()) == want
    finally:
        GS._RUN_MIN[0] = old


def test_copy_live_count_through_terms_nulls_and_having():
    """the census copy counted in the same parallel pass (the size line lowered to reach this fixture): the live
    count kept exact through a NULL square, <> and NOT IN sweeps (a value named twice), IN, and HAVING"""
    sys.path.insert(0, os.path.dirname(__file__))
    import test_groupself_bulk as B
    rng = np.random.default_rng(3)
    src = rng.integers(-2, 4, size=100_003).astype(np.int64)
    out = np.empty(100_000, np.int64)
    assert int(wdb_kernels.copy_live(src, out)) == int(np.count_nonzero(src[:100_000] > 0))
    assert np.array_equal(out, src[:100_000])
    old = GS._COPY_LIVE_MIN[0]
    try:
        GS._COPY_LIVE_MIN[0] = 0
        for q, k in ((_q('nk', '9'), 'nk'),
                     ("SELECT nk, COUNT(*) FROM t WHERE nk NOT IN (9, 9, 11) GROUP BY nk ORDER BY COUNT(*) DESC", 'nk'),
                     ("SELECT v, COUNT(*) FROM t WHERE v <> 7 GROUP BY v HAVING COUNT(*) > 1 ORDER BY COUNT(*) DESC",
                      'v'),
                     (_q('u', '5'), 'u'), (_q('w', "'w0000009'"), 'w')):
            want = _want(q, k)
            cur, _cols = _cursor(q)
            assert cur.rowcount == len(want) and _rows(cur.fetchall()) == want, q
        test_run_order_serves_every_key_and_fetch_form()
        for f in (B.test_bulk_path_every_key_type, B.test_swapped_projection_having_not_in,
                  B.test_key_ascending_tie_break_serves, B.test_sql_answers_whatever_read_serves):
            f()
    finally:
        GS._COPY_LIVE_MIN[0] = old
