"""THE HEAVY SPLIT IN THE COMPOSITION (2026-10-23): a count-ordered pair past the dense board whose key has one value
holding most rows -- the heavy rows' cells from the other key's census less its light rows, the few light rows
(gathered in one parallel pass, wdb_kernels.off_value_gather) grouped exactly. Every way (split on / off; no filter,
<> and = on either key, a filter removing the heavy value itself) gives the same canonical top as numpy's."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import wdb_gridwalk as GW, wdb_kernels

_FX = None


def _fixture():
    """300K rows: h (5,000 values, 'h0' on ~92% of rows), o (6,000 values), p (4,500 values) -- every pair 22-30M
    cells, past the dense board"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        CASE WHEN (i * 7919) % 100 < 92 THEN 'h0' ELSE 'h' || CAST(((i * 48271) % 4999) + 1 AS VARCHAR) END AS h,
        CAST((i * 2654435761) % 6000 AS INTEGER) AS o,
        CAST((i * 40507) % 4500 AS INTEGER) AS p
        FROM range(300000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'ght_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['h', 'str'], ['o', 'int'], ['p', 'int']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    db.cat.add_segment('t', 't_0.wdb')
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    _FX = (db, con, seg)
    return _FX


def test_off_value_gather_against_numpy():
    rng = np.random.default_rng(2)
    for n, V, top in ((1_000_003, 50, 7), (17, 3, 0), (300_000, 1, 0)):
        ch = rng.integers(0, V, n).astype(np.uint16)
        ch[: n // 2] = top
        co = rng.integers(0, 1000, n).astype(np.uint32)
        lh, lo = wdb_kernels.off_value_gather(ch, co, top)
        m = ch != top
        assert np.array_equal(lh, ch[m]) and np.array_equal(lo, co[m]), (n, V)


def _want(seg, a, b, afilter, lim):
    ca = np.asarray(seg._raw_codes(a)).astype(np.int64); cb = np.asarray(seg._raw_codes(b)).astype(np.int64)
    Vb = int(seg.cols[b]['V'])
    m = np.ones(ca.size, bool)
    if afilter is not None:
        fcol, fneg, fcode = afilter
        fc = ca if fcol == a else cb
        m = (fc != fcode) if fneg else (fc == fcode)
    gid, cnt = np.unique(ca[m] * Vb + cb[m], return_counts=True)
    o = np.lexsort((gid, -cnt))[:lim]
    return gid[o].tolist(), cnt[o].tolist()


def test_heavy_split_every_way_the_same():
    _db, _con, seg = _fixture()
    import wdb_bound
    hcode = int(np.argmax(np.asarray(seg.raw_census('h', int(seg.cols['h']['V'])))))
    used = 0
    for cols in (('h', 'o'), ('h', 'p'), ('o', 'p')):
        a, b = sorted(cols)
        assert int(seg.cols[a]['V']) * int(seg.cols[b]['V']) > wdb_bound.DENSE_MAX, cols
        filters = [None, (b, True, 3), (a, True, 5), (a, False, 5)]
        if 'h' in cols:
            filters += [('h', True, hcode), ('h', False, hcode), ('h', True, (hcode + 1) % 100)]
        for afilter in filters:
            for lim in (10, 37):
                want = _want(seg, a, b, afilter, lim)
                for on in (True, False):
                    old = GW._HEAVY_TOP[0]
                    GW._HEAVY_TOP[0] = on
                    h0 = GW._HEAVY_HITS
                    try:
                        got = GW._compose_top(seg, [a, b], lim, afilter=afilter)
                    finally:
                        GW._HEAVY_TOP[0] = old
                    if on and GW._HEAVY_HITS > h0:
                        used += 1
                    assert got is not None, (cols, afilter, on)
                    assert (np.asarray(got[0]).tolist(), np.asarray(got[1]).tolist()) == want, (cols, afilter, lim, on)
    assert used >= 10                                   # the split served the heavy pairs (o x p has no heavy value)


def test_sql_tops_legal_against_duckdb():
    db, con, _seg = _fixture()
    for q in ("SELECT h, o, COUNT(*) AS c FROM t GROUP BY h, o ORDER BY c DESC LIMIT 10",
              "SELECT o, h, COUNT(*) AS c FROM t WHERE h <> 'h0' GROUP BY o, h ORDER BY c DESC LIMIT 10",
              "SELECT h, p, COUNT(*) AS c FROM t WHERE p <> 3 GROUP BY h, p ORDER BY c DESC LIMIT 10"):
        r = db.run(q)
        rows = r[0] if isinstance(r, tuple) else r
        want = [int(x[-1]) for x in con.execute(q).fetchall()]
        assert sorted((int(x[-1]) for x in rows), reverse=True) == sorted(want, reverse=True), q
        where = q[q.index(' WHERE '):q.index(' GROUP BY ')] if ' WHERE ' in q else ''
        a, b = q.split('SELECT ')[1].split(',')[0], q.split(', ')[1]
        for x in rows:
            v0 = x[0].decode() if isinstance(x[0], bytes) else x[0]
            v1 = x[1].decode() if isinstance(x[1], bytes) else x[1]
            n = con.execute('SELECT COUNT(*) FROM t %s %s %s = ? AND %s = ?' % (where, 'AND' if where else 'WHERE',
                                                                                 a, b), [v0, v1]).fetchone()[0]
            assert n == int(x[2]), (q, x, n)
