"""THE FRAME SCAN IN COMPILED LANES (2026-10-30, E): an enc-3 flag scan run by wdb_kernels.z3_flag_hits equals the
thread-pool scan and the decoded column -- positions, and the codes kept for codes_at -- for dense, sparse, empty
and full flags, over whole, ragged, one-frame and tail spans, on one- and two-byte codes; LIKE through SQL answers
DuckDB's with the lanes on and off."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np

_FX = None


def _fixture():
    """400K rows in the encoder's 65,536-row frames: c (runs of 4 over 701 values: two-byte codes), u (runs of 4 over
    200 values: one-byte codes), s (strings, 5,000), b (41) -- the frame-local repeats zstd wins frame by frame"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        CAST(hash(i // 4, 3) % 701 AS INTEGER) AS c,
        CAST(hash(i // 4, 5) % 200 AS INTEGER) AS u,
        's' || CAST(hash((i % 4093) + 4093 * (i // 65536), 4) % 5000 AS VARCHAR) AS s,
        CAST((i * 48271) % 41 AS INTEGER) AS b
        FROM range(400000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'fl3_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['c', 'int'], ['u', 'int'], ['s', 'str'], ['b', 'int']])
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


def _scan(WS, seg, col, flag, lo, hi, lanes):
    old = WS._FLAG_LANES[0]
    WS._FLAG_LANES[0] = lanes
    try:
        seg.__dict__.pop('_scan_codes', None)
        pos = np.asarray(WS._scan_flag(seg, col, flag, lo, hi))
        sc = seg.__dict__.get('_scan_codes', {}).get(col)
        return pos, sc
    finally:
        WS._FLAG_LANES[0] = old


def test_lanes_equal_the_pool_and_the_column():
    _db, _con, seg = _fixture()
    import wdb_wherescan as WS, wdb_engine
    if wdb_engine._libzstd() is None:
        return
    N = int(seg.N)
    cols = [c for c in ('c', 'u', 's', 'b') if seg.cols[c].get('code_enc') == 3]
    assert cols, 'the fixture frames no column'
    widths = set()
    rng = np.random.default_rng(1030)
    checked = 0
    for col in cols:
        c = seg.cols[col]
        V = int(c['V']); BR = int(c['BR'])
        widths.add(int(c['cwidth']))
        raw = np.asarray(seg._raw_codes_range(col, 0, N)).astype(np.int64)
        cen = np.bincount(raw, minlength=V)
        live = np.flatnonzero(cen)
        rare = int(live[np.argmin(cen[live])])
        flags = {'dense': rng.random(V) < 0.3, 'none': np.zeros(V, bool), 'all': np.ones(V, bool)}
        f1 = np.zeros(V, bool); f1[rare] = True; flags['rare'] = f1
        for name, flag in flags.items():
            for lo, hi in ((0, N), (777, N - 1234), (BR, 2 * BR), (BR + 5, BR + 9), (BR - 3, 3 * BR + 11),
                           (N - 10, N), (0, 1)):
                lo, hi = max(0, lo), min(N, hi)
                want = np.flatnonzero(flag[raw[lo:hi]]) + lo
                got, sc = _scan(WS, seg, col, flag, lo, hi, True)
                old, sco = _scan(WS, seg, col, flag, lo, hi, False)
                assert np.array_equal(got, want), (col, name, lo, hi, got.size, want.size)
                assert np.array_equal(old, want), (col, name, lo, hi)
                if want.size:
                    assert sc is not None and np.array_equal(sc[0], want) and np.array_equal(sc[1], raw[want]), (col, name, lo, hi)
                    assert sco is not None and np.array_equal(sco[1], sc[1])
                checked += 1
    assert checked >= 80, checked
    assert widths >= {1, 2}, widths
    c = seg.cols[cols[0]]; BR = int(c['BR'])
    flag = np.ones(int(c['V']), bool)
    got = WS._scan_flag3_lanes(seg, c, WS._flag_bits(seg, flag), 3, 2 * BR + 1, 0, 3)   # the lanes ran, not the pool
    assert got is not None and np.array_equal(got[0], np.arange(3, 2 * BR + 1)), 'the lanes fell back'


def test_slots_that_overflow_and_slots_that_fit():
    """pass 2 copies a frame's hits from its slot when they fit and inflates the frame again when they do not: one-hit
    slots (nearly every frame overflows), whole-frame slots (none does) and the quarter frame give the same answer"""
    _db, _con, seg = _fixture()
    import wdb_wherescan as WS, wdb_engine
    if wdb_engine._libzstd() is None:
        return
    N = int(seg.N)
    cols = [c for c in ('c', 'u', 's', 'b') if seg.cols[c].get('code_enc') == 3]
    rng = np.random.default_rng(1031)
    old = WS._SLOT_DIV[0]
    over = fit = 0
    try:
        for col in cols:
            c = seg.cols[col]
            V = int(c['V']); BR = int(c['BR'])
            raw = np.asarray(seg._raw_codes_range(col, 0, N)).astype(np.int64)
            for share in (0.02, 0.4, 0.9):
                flag = rng.random(V) < share
                for lo, hi in ((0, N), (BR - 3, 3 * BR + 11)):
                    want = np.flatnonzero(flag[raw[lo:hi]]) + lo
                    per = np.bincount(want // BR, minlength=(N + BR - 1) // BR)
                    over += int((per > BR // 4).sum()); fit += int(((per > 0) & (per <= BR // 4)).sum())
                    for div in (BR, 4, 1):
                        WS._SLOT_DIV[0] = div
                        got, sc = _scan(WS, seg, col, flag, lo, hi, True)
                        assert np.array_equal(got, want), (col, share, lo, hi, div, got.size, want.size)
                        if want.size:
                            assert sc is not None and np.array_equal(sc[1], raw[want]), (col, share, lo, hi, div)
    finally:
        WS._SLOT_DIV[0] = old
    assert over > 0 and fit > 0, (over, fit)


def test_like_answers_duckdb_both_ways():
    db, con, seg = _fixture()
    import controller, wdb_wherescan as WS
    qs = ["SELECT u, COUNT(*) AS n FROM t WHERE s LIKE '%42%' GROUP BY u ORDER BY n DESC, u LIMIT 7",
          "SELECT COUNT(*) FROM t WHERE s LIKE '%9'",
          "SELECT b, MIN(s) AS m, COUNT(*) AS n FROM t WHERE s LIKE 's1%' AND b <> 3 GROUP BY b ORDER BY n DESC, b LIMIT 5",
          "SELECT COUNT(*) FROM t WHERE s LIKE '%zz%'"]
    old = WS._FLAG_LANES[0]
    try:
        for q in qs:
            want = [tuple(r) for r in con.execute(q).fetchall()]
            for lanes in (True, False):
                WS._FLAG_LANES[0] = lanes
                controller.plan_epoch_bump()
                got = [tuple(r) for r in db.run(q)[0]]
                assert got == want, (q, lanes, got[:3], want[:3])
    finally:
        WS._FLAG_LANES[0] = old
