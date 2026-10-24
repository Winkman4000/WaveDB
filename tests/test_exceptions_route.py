"""THE EXCEPTIONS ROAD (2026-10-24, Jackson): a COUNT-only GROUP BY under = / <> filters lists whichever side is
smaller -- the rows some filter fails (counts = the key's census less theirs) or the survivors (counted on a board by
code, no factorise) -- and enc 10 lists its rows straight from its blocks (wdb_kernels.bp10_eq_pos). Every answer,
each road on or off, is DuckDB's."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np

_FX = None


def _fixture():
    """400K rows: k (900 values), s (strings, 5,000); flag columns in runs, as the real flags lie: f (0 on ~91%),
    g (0 on ~85%, 3 values), h (0 on 10%), n (0 on ~73%, NULL on ~7%)"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        CAST((i * 2654435761) % 900 AS INTEGER) AS k,
        's' || CAST((i * 40507) % 5000 AS VARCHAR) AS s,
        CAST(CASE WHEN (i // 3000) % 12 = 5 OR (i * 7919) % 100 = 3 THEN 1 ELSE 0 END AS INTEGER) AS f,
        CAST(CASE WHEN (i // 2500) % 7 = 2 THEN 1 + ((i // 2500) % 2) WHEN (i * 104729) % 100 = 7 THEN 2
             ELSE 0 END AS INTEGER) AS g,
        CAST(CASE WHEN (i // 4000) % 10 = 3 THEN 0 ELSE 1 END AS INTEGER) AS h,
        CAST(CASE WHEN (i // 3500) % 14 = 1 THEN NULL WHEN (i // 3500) % 5 = 2 THEN 1 ELSE 0 END
             AS INTEGER) AS n
        FROM range(400000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'exr_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['k', 'int'], ['s', 'str'], ['f', 'int'], ['g', 'int'], ['h', 'int'],
                           ['n', 'int']])
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


def _pack(codes, bits):
    b = ((codes[:, None] >> np.arange(bits - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    return np.packbits(b).tobytes()


def _dress(codes, bits, run_every):
    """enc 10's own layout (test_bp10_counts_widths): a directory of (offset << 1 | is_run), a run block as u16
    count of runs then (u16 length, u16 value) pairs, a bitpack block as rows x bits MSB-first"""
    B = 4096
    n = codes.size
    nb = (n + B - 1) // B
    pay = bytearray(); dirX = np.zeros(nb, np.int64)
    for k in range(nb):
        blk = codes[k * B:(k + 1) * B]
        if k % run_every == 0:
            starts = np.flatnonzero(np.r_[True, blk[1:] != blk[:-1]])
            ends = np.r_[starts[1:], blk.size]
            dirX[k] = (len(pay) << 1) | 1
            pay += np.uint16(starts.size).tobytes()
            for s0, e0 in zip(starts, ends):
                pay += np.uint16(e0 - s0).tobytes() + np.uint16(blk[s0]).tobytes()
        else:
            dirX[k] = len(pay) << 1
            pay += _pack(blk, bits)
    return np.frombuffer(bytes(16) + bytes(pay) + bytes(16), np.uint8), dirX, 16


def test_bp10_eq_pos_every_width():
    """the scan equals decode-and-compare at every width 1-16: run and bitpack blocks mixed, a partial last block,
    a code no row holds, = and <>, whole and ragged spans (a span starting and ending inside blocks)"""
    import wdb_kernels as WK
    rng = np.random.default_rng(24)
    for bits in range(1, 17):
        V = 1 << bits
        n = 4096 * 23 + 777
        codes = rng.integers(0, min(V, 5), n)
        codes[:6000] = np.repeat(rng.integers(0, V, 60), 100)
        codes[20] = V - 1
        if bits == 1:
            codes[4096 * 3:4096 * 4] = 1                 # a whole 1-bit block of ones, and one of zeros
            codes[4096 * 5:4096 * 6] = 0
        buf, dirX, pay = _dress(codes.astype(np.int64), bits, 3)
        for code in sorted({0, 1, int(V - 1), int(V), int(codes[5000])}):
            for neg in (False, True):
                for lo, hi in ((0, n), (4095, n - 4097), (8192, 8192 + 4096), (100, 101)):
                    got = WK.bp10_eq_pos(buf, dirX, np.int64(pay), np.int64(bits), np.int64(n), np.int64(lo),
                                         np.int64(hi), np.int64(code), bool(neg))
                    part = codes[lo:hi]
                    want = (np.flatnonzero(part != code) if neg else np.flatnonzero(part == code)) + lo
                    assert np.array_equal(got, want), (bits, code, neg, lo, hi)


def test_union_sorted():
    import wdb_kernels as WK
    rng = np.random.default_rng(25)
    for na, nb in ((0, 0), (0, 5), (7, 0), (1000, 3000), (50000, 20000)):
        a = np.unique(rng.integers(0, 100000, na)).astype(np.int64)
        b = np.unique(rng.integers(0, 100000, nb)).astype(np.int64)
        assert np.array_equal(WK.union_sorted(a, b), np.union1d(a, b).astype(np.int64)), (na, nb)


QS = ['SELECT k, COUNT(*) FROM t WHERE f = 0 GROUP BY k',                       # 92% kept: the exceptions
      'SELECT k FROM t WHERE f = 0 GROUP BY k',
      'SELECT k, COUNT(*) AS c FROM t WHERE f = 0 AND g = 0 GROUP BY k ORDER BY c DESC LIMIT 7',
      'SELECT k, COUNT(*) FROM t WHERE g <> 1 GROUP BY k',
      'SELECT k, COUNT(*) FROM t WHERE n <> 1 GROUP BY k',                       # NULL <> 1 is not true
      'SELECT k, COUNT(*) FROM t WHERE n = 0 GROUP BY k HAVING COUNT(*) > 350',
      'SELECT s, COUNT(*) AS c FROM t WHERE f = 0 GROUP BY s ORDER BY c DESC, s LIMIT 20',
      'SELECT k + 1 AS x, COUNT(*) FROM t WHERE f = 0 GROUP BY x',
      'SELECT k, COUNT(*) FROM t WHERE h = 0 GROUP BY k',                       # 10% kept: the survivors
      'SELECT s, COUNT(*) AS c FROM t WHERE f <> 0 GROUP BY s ORDER BY c DESC, s LIMIT 15',
      'SELECT k, COUNT(*) FROM t WHERE f = 7 GROUP BY k',                        # a value no row holds
      'SELECT k, COUNT(*) FROM t WHERE f <> 7 GROUP BY k',
      'SELECT k, SUM(g), COUNT(*) FROM t WHERE f = 0 GROUP BY k']                # SUM: neither road


def _canon(rows, ordered):
    t = [tuple(x) for x in rows]
    return t if ordered else sorted(t)


def _legal(q, got, con):
    """exact for unordered answers and fully tie-broken orders; for ORDER BY count LIMIT without a tie-break, the
    count sequence exact and every returned group's count its true one (SQL leaves ties at the cut in any order)"""
    import re
    want = con.execute(q).fetchall()
    if 'ORDER BY' not in q:
        return sorted(tuple(x) for x in got) == sorted(want)
    if re.search(r'ORDER BY c DESC, \w+ LIMIT', q):
        return [tuple(x) for x in got] == want
    if [x[-1] for x in got] != [x[-1] for x in want]:
        return False
    full = dict((tuple(x[:-1]), x[-1]) for x in con.execute(re.sub(r' ORDER BY .*$', '', q)).fetchall())
    return all(full.get(tuple(x[:-1])) == x[-1] for x in got)


def _wherescan(seg, q):
    import sqlglot, wdb_wherescan as WS
    cmap = {c: c for c in seg.cols}
    return WS.detect(seg, sqlglot.parse_one(q, read='duckdb'), cmap)


def test_every_road_legal_and_the_same():
    """through the controller (whichever read answers) and through wherescan itself, each road on and off"""
    db, con, seg = _fixture()
    import controller, wdb_wherescan as WS
    saved = (WS._EXC_ROUTE[0], WS._COUNT_BOARD[0])
    direct = 0
    try:
        for q in QS:
            spec = _wherescan(seg, q)
            for exc in (True, False):
                for board in (True, False):
                    WS._EXC_ROUTE[0], WS._COUNT_BOARD[0] = exc, board
                    controller.plan_epoch_bump()
                    outs = [db.run(q)]
                    if spec is not None:
                        outs.append(WS.execute(seg, dict(spec)))
                    for r in outs:
                        got = r[0] if isinstance(r, tuple) else r
                        got = [tuple(x) for x in got]
                        assert _legal(q, got, con), (q, exc, board, got[:3])
                        # Python values, never numpy scalars (a scalar key once came back as np.int64)
                        assert all(not isinstance(v, np.generic) for row in got for v in row), (q, got[:2])
            direct += spec is not None
    finally:
        WS._EXC_ROUTE[0], WS._COUNT_BOARD[0] = saved
        controller.plan_epoch_bump()
    assert direct >= 8, direct                      # wherescan itself answered most shapes


def test_the_smaller_side_is_listed():
    _db, _con, seg = _fixture()
    import wdb_wherescan as WS
    took = {}
    for q in ('SELECT k, COUNT(*) FROM t WHERE f = 0 GROUP BY k',          # 92% kept
              'SELECT k, COUNT(*) FROM t WHERE f = 0 AND g = 0 GROUP BY k',
              'SELECT k, COUNT(*) FROM t WHERE h = 0 GROUP BY k'):          # 10% kept
        spec = _wherescan(seg, q)
        assert spec is not None, q
        e0, b0 = WS._EXC_HITS, WS._BOARD_HITS
        WS.execute(seg, dict(spec))
        took[q] = 'exceptions' if WS._EXC_HITS > e0 else 'board' if WS._BOARD_HITS > b0 else 'other'
    assert took == {'SELECT k, COUNT(*) FROM t WHERE f = 0 GROUP BY k': 'exceptions',
                    'SELECT k, COUNT(*) FROM t WHERE f = 0 AND g = 0 GROUP BY k': 'exceptions',
                    'SELECT k, COUNT(*) FROM t WHERE h = 0 GROUP BY k': 'board'}, took
