"""THE COUNT FUSED, ITS SIBLINGS (2026-10-23): gridwalk's count-ordered top (_compose_top) and the grab count the dense
grid through _dense_cells (the fused count to _FUSED_MAX cells, radix_acc + bincount_par past it); and the grab
declines before any work when it would decode the other key whole (_grab_at), so the remove lane's fused count
answers. Every way must give the same canonical top; oracle: the radix path and DuckDB (the top-k counts)."""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import wdb_gridwalk as GW, wdb_kernels
import test_groupself_run as F


def _with(fused_max=None, dense_par=None, grab_max=None):
    old = GW._FUSED_MAX[0], GW._DENSE_PAR[0], GW._GRAB_MAX[0]
    if fused_max is not None:
        GW._FUSED_MAX[0] = fused_max
    if dense_par is not None:
        GW._DENSE_PAR[0] = dense_par
    if grab_max is not None:
        GW._GRAB_MAX[0] = grab_max
    return old


def _restore(old):
    GW._FUSED_MAX[0], GW._DENSE_PAR[0], GW._GRAB_MAX[0] = old


def test_dense_cells_fused_equals_radix():
    rng = np.random.default_rng(5)
    for Va, Vb, n in ((2, 2, 1 << 20), (59, 44, 3_000_001), (250, 7, 777), (1, 1, 9)):
        ca = rng.integers(0, Va, n).astype(np.uint8)
        cb = rng.integers(0, Vb, n).astype(np.uint16)
        old = _with(fused_max=1 << 40)
        try:
            f = GW._dense_cells(ca, cb, Va, Vb)
            GW._FUSED_MAX[0] = 0
            r = GW._dense_cells(ca, cb, Va, Vb)
        finally:
            _restore(old)
        want = np.bincount(ca.astype(np.int64) * Vb + cb, minlength=Va * Vb)
        assert np.array_equal(f, want) and np.array_equal(r, want), (Va, Vb, n)


_SM = None


def _small():
    """a table of small keys (2 / 7 / 50 / 300 values, skewed, one NULL-free each) -- dense grids"""
    global _SM
    if _SM is not None:
        return _SM
    import uuid, tempfile, duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE s AS SELECT i AS id,
        CAST((i * 7919) % 2 AS INTEGER) AS x2,
        CAST(CASE WHEN i % 3 = 0 THEN 0 ELSE (i * 31) % 7 END AS INTEGER) AS x7,
        CAST((i * 2654435761) % 50 AS INTEGER) AS x50,
        'k' || CAST(((i * 48271) % 300) AS VARCHAR) AS x300
        FROM range(200000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'gfs_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 's.parquet')
    con.execute("COPY (SELECT * FROM s ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('s', [['id', 'int'], ['x2', 'int'], ['x7', 'int'], ['x50', 'int'], ['x300', 'str']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 's_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    db.cat.add_segment('s', 's_0.wdb')
    seg = db.open_segment(db.cat.segment_paths('s')[0], 's')
    _SM = (db, con, seg)
    return _SM


def _top(cols, lim, afilter, **kw):
    seg = _small()[2]
    old = _with(**kw)
    try:
        got = GW._compose_top(seg, list(cols), lim, afilter=afilter)
    finally:
        _restore(old)
    return None if got is None else (np.asarray(got[0]).tolist(), np.asarray(got[1]).tolist(), got[2])


def test_compose_top_every_way_the_same():
    seg = _small()[2]
    K = ('x2', 'x7', 'x50', 'x300')
    pairs = [(a, b) for a in K for b in K if a < b]
    ran = 0
    for cols in pairs:
        a, b = sorted(cols)
        if any(seg.cols[c].get('has_null') for c in cols):
            continue
        ran += 1
        for afilter in (None, (b, True, 0), (a, True, 1), (a, False, 1)):
            ref = _top(cols, 10, afilter, fused_max=0, grab_max=0)          # radix count, the remove lane
            for kw in ({'fused_max': 1 << 40, 'grab_max': 0}, {'fused_max': 1 << 40, 'grab_max': -1},
                       {'fused_max': 0, 'grab_max': -1}):
                got = _top(cols, 10, afilter, **kw)
                assert got == ref, (cols, afilter, kw)
            if afilter is None:                     # the top itself against numpy: count desc, cell asc
                ca = np.asarray(seg._raw_codes(a)).astype(np.int64); cb = np.asarray(seg._raw_codes(b)).astype(np.int64)
                Vb = int(seg.cols[b]['V'])
                full = np.bincount(ca * Vb + cb)
                gid = np.flatnonzero(full)
                o = np.lexsort((gid, -full[gid]))[:10]
                assert ref[0] == gid[o].tolist() and ref[1] == full[gid[o]].tolist(), cols
    assert ran == 6                                 # every pair of the small keys was a dense grid


def test_grab_at_rule():
    fake = types.SimpleNamespace(N=100_000_000, cols={
        'z': {'code_enc': 1, 'bits': 4, 'V': 10},                           # a zstd stream: decoded whole
        'f': {'code_enc': 3, 'BR': 65536, 'cwidth': 1, 'bits': 8, 'V': 200},
        'p': {'code_enc': 10, 'bits': 8, 'V': 200}})
    assert not GW._grab_at(fake, 'z', 10, fake.N)
    assert GW._grab_at(fake, 'f', 20, fake.N)                               # 20 frames: under the decode
    assert not GW._grab_at(fake, 'f', 2263, fake.N)                         # ~1,180 frames: over it
    assert GW._grab_at(fake, 'p', fake.N // 300, fake.N) and not GW._grab_at(fake, 'p', fake.N // 100, fake.N)


def test_sql_filtered_tops_legal():
    import test_sampletop_census as S
    db, con, _seg = F._fixture()
    for q in ("SELECT u, v, COUNT(*) AS c FROM t WHERE v <> 9 GROUP BY u, v ORDER BY c DESC LIMIT 10",
              "SELECT v, w, COUNT(*) AS c FROM t WHERE w <> 'w0000009' GROUP BY v, w ORDER BY c DESC LIMIT 10",
              "SELECT tm, v, COUNT(*) AS c FROM t GROUP BY tm, v ORDER BY c DESC LIMIT 10"):
        r = db.run(q)
        rows = [tuple(S.F._norm(x) for x in row) for row in (r[0] if isinstance(r, tuple) else r)]
        want = [int(x[-1]) for x in con.execute(q.replace(' FROM t ', ' FROM tt ')).fetchall()]
        assert sorted((x[-1] for x in rows), reverse=True) == sorted(want, reverse=True), q
        a, b = q.split('SELECT ')[1].split(',')[0], q.split(', ')[1]
        where = q[q.index(' WHERE '):q.index(' GROUP BY ')] if ' WHERE ' in q else ''
        for x in rows:
            n = con.execute('SELECT COUNT(*) FROM tt %s %s %s IS NOT DISTINCT FROM ? AND %s IS NOT DISTINCT FROM ?' % (
                where, 'AND' if where else 'WHERE', a, b), [x[0], x[1]]).fetchone()[0]
            assert n == x[2], (q, x, n)
