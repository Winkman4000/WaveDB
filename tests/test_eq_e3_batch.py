"""THE ONE-CODE FRAME SCAN (2026-10-24, the dashboard's fixed cost): an enc-3 equality scan decompresses the span's
frames in one call on zstd's threads and compares each with the code -- the same positions as the flag frame scan,
for a common code, a rare one, one past the dictionary, whole and ragged spans; routed answers DuckDB's."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np

_FX = None


def _fixture():
    """400K rows the encoder writes in its 65,536-row frames (seven, the last one short): short runs and frame-local
    repeats, which zstd wins frame by frame (scattered values go to bitpack; a whole-table period seals as one
    frame). a (17 on ~60%, else runs of 8 over 3,000 values), c (runs of 4, 701 values), s (strings, 5,000), b (41)"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        CAST(CASE WHEN hash(i, 1) % 10 < 6 THEN 17 ELSE hash(i // 8, 2) % 3000 END AS INTEGER) AS a,
        CAST(hash(i // 4, 3) % 701 AS INTEGER) AS c,
        's' || CAST(hash((i % 4093) + 4093 * (i // 65536), 4) % 5000 AS VARCHAR) AS s,
        CAST((i * 48271) % 41 AS INTEGER) AS b
        FROM range(400000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'e3b_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['a', 'int'], ['c', 'int'], ['s', 'str'], ['b', 'int']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k9, v9 in saved.items():
            if v9 is not None:
                os.environ[k9] = v9
    db.cat.add_segment('t', 't_0.wdb')
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    _FX = (db, con, seg)
    return _FX


def test_batch_scan_equals_the_frame_scan():
    _db, _con, seg = _fixture()
    import wdb_wherescan as WS
    N = int(seg.N)
    cols = [c for c in ('a', 'c', 's', 'b') if seg.cols[c].get('code_enc') == 3]
    assert cols, 'the fixture frames no column'
    checked = 0
    for col in cols:
        c = seg.cols[col]
        V = int(c['V']); BR = int(c['BR'])
        assert len(c['boffs']) - 1 >= 3, (col, BR, N)         # the batch path's many frames, not one
        raw = np.asarray(seg._raw_codes_range(col, 0, N))
        cen = np.bincount(raw, minlength=V)
        live = np.flatnonzero(cen)
        for code in sorted({int(live[np.argmax(cen[live])]), int(live[np.argmin(cen[live])]), V + 3}):
            for lo, hi in ((0, N), (777, N - 1234), (BR, 2 * BR), (BR + 5, BR + 9), (BR - 3, 3 * BR + 11),
                           (N - 10, N)):
                lo, hi = max(0, lo), min(N, hi)
                seg.__dict__.pop('_scan_codes', None)
                got = WS._eq_e3_batch(seg, col, c, code, lo, hi)
                part = raw[lo:hi]
                want = np.flatnonzero(part == code) + lo
                assert np.array_equal(got, want), (col, code, lo, hi, got.size, want.size)
                flag = np.zeros(max(V, code + 1), np.bool_); flag[code] = True
                old = WS._scan_flag(seg, col, flag, lo, hi)
                assert np.array_equal(got, old), (col, code, lo, hi)
                checked += 1
    assert checked >= 12, checked


def test_routed_answers_legal_both_ways():
    db, con, seg = _fixture()
    import controller, wdb_wherescan as WS
    qs = ['SELECT COUNT(*) FROM t WHERE a = 17 AND b = 3', 'SELECT c, COUNT(*) FROM t WHERE a = 2999 GROUP BY c',
          "SELECT b, COUNT(*) AS n FROM t WHERE s = 's42' GROUP BY b ORDER BY n DESC, b LIMIT 5",
          'SELECT id FROM t WHERE c = 700 AND a = 17 ORDER BY id LIMIT 20']
    saved = WS._E3_BATCH[0]
    try:
        for on in (True, False):
            WS._E3_BATCH[0] = on
            for q in qs:
                controller.plan_epoch_bump()
                r = db.run(q)
                got = [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)]
                want = con.execute(q).fetchall()
                if 'ORDER BY' not in q:
                    got, want = sorted(got), sorted(want)
                assert got == want, (q, on, got[:3], want[:3])
    finally:
        WS._E3_BATCH[0] = saved
        controller.plan_epoch_bump()
