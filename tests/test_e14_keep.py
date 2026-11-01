"""THE DATE PLANES STAY (2026-10-31): an enc-14 column's planes from the shelf equal a fresh build, byte for byte;
they are read-only; the second ask is served from the shelf (no rebuild); WDB_E14_KEEP=0 and WDB_HOT_KEEP=0 do not
keep them; and date-filtered answers equal DuckDB's with the planes kept and rebuilt."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np

_FX = None


def _fixture():
    """300K rows: two DATE columns in TPC-H's range (types nominate the field planes: dt 3), q a small int"""
    global _FX
    if _FX is not None:
        return _FX
    import duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        DATE '1992-01-01' + CAST(hash(i // 64, 1) % 2526 AS INTEGER) AS d1,
        DATE '1992-01-01' + CAST(hash(i // 64, 1) % 2526 + hash(i, 2) % 60 AS INTEGER) AS d2,
        CAST(hash(i, 3) % 50 AS INTEGER) AS q
        FROM range(300000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'e14k_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['d1', 'datetime'], ['d2', 'datetime'], ['q', 'int']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k9, v9 in saved.items():
            if v9 is not None:
                os.environ[k9] = v9
    db.cat.add_segment('t', 't_0.wdb')
    _FX = (db, con, db.open_segment(db.cat.segment_paths('t')[0], 't'))
    return _FX


def _e14_cols(seg):
    return [c for c in ('d1', 'd2') if seg.cols[c].get('code_enc') == 14]


def test_kept_planes_equal_a_fresh_build():
    import wdb_engine, wdb_shelf
    db, con, seg = _fixture()
    cols = _e14_cols(seg)
    assert cols, {c: seg.cols[c].get('code_enc') for c in ('d1', 'd2')}   # the fixture must wear the field planes
    for nm in cols:
        wdb_shelf.SHELF.keep_only(frozenset())   # nothing kept yet
        a = seg._e14_planes(nm)
        b = seg._e14_planes(nm)                   # the second ask
        fresh = seg._e14_planes_build(nm)
        for x, y, z in zip(a, b, fresh):
            assert x is y, 'the second ask rebuilt the planes'
            assert np.array_equal(x, z)
            assert not x.flags.writeable
        old = wdb_engine._E14KEEP[0]
        try:
            wdb_engine._E14KEEP[0] = False
            wdb_shelf.SHELF.keep_only(frozenset())
            c = seg._e14_planes(nm); d = seg._e14_planes(nm)
            assert c[0] is not d[0] and np.array_equal(c[0], fresh[0])
        finally:
            wdb_engine._E14KEEP[0] = old


def test_answers_both_ways():
    import wdb_engine, controller
    db, con, seg = _fixture()
    qs = ["SELECT COUNT(*), SUM(q) FROM t WHERE d1 >= DATE '1994-01-01' AND d1 < DATE '1995-01-01'",
          "SELECT q, COUNT(*) FROM t WHERE d1 < d2 AND d2 >= DATE '1994-08-01' GROUP BY q ORDER BY q",
          "SELECT COUNT(*) FROM t WHERE q < 10 AND d2 < DATE '1993-03-01'"]
    old = wdb_engine._E14KEEP[0]
    try:
        for qy in qs:
            exp = [tuple(int(v) for v in r) for r in con.execute(qy).fetchall()]
            for on in (True, False):
                wdb_engine._E14KEEP[0] = on
                controller.plan_epoch_bump()
                for _ in range(2):                # the second run reads the kept planes
                    got = [tuple(int(v) for v in r) for r in db.run(qy)[0]]
                    assert got == exp, (qy, on, got[:3], exp[:3])
    finally:
        wdb_engine._E14KEEP[0] = old
