"""Two exactness holes the shifting board found (2026-10-12), each checked where it lived.

1. RAW PACKED CODES (tag 17) had no branch in Segment._raw_codes_range and fell to the bit-packed
   reader, which reads another layout: every window of an enc-17 column came back scrambled, and
   wherescan's WHERE col = v counted the wrong rows (WindowClientWidth = 1863: 123,662 of 1,985,884).
   The window must equal the full decode, small windows and wide, and the equality road must count
   exactly what the decoded column holds.
2. MIN/MAX over an integer column past 2^53 rode the fused engine's float64 accumulators and came back
   rounded (MIN(URLHash) WHERE ... lost its last three digits). They must equal DuckDB exactly."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import helpers, wdb_encode
import wdb_wherescan as WS
from wdb_db import Database

_E17 = None
def _e17():
    global _E17
    if _E17 is None:
        rng = np.random.default_rng(3)
        n = (1 << 22) + 1000                       # tag 17 is elected only at >= 2^22 rows
        w = np.where(rng.random(n) < 0.5, rng.integers(0, 3000, n), rng.integers(0, 30, n)).astype(np.int64)
        seg, _ = helpers.roundtrip(pd.DataFrame({'w': w}))
        _E17 = (seg, w)
    return _E17


def test_enc17_is_elected():
    seg, _ = _e17()
    assert seg.cols['w'].get('code_enc') == 17, 'fixture no longer elects tag 17: %r' % seg.cols['w'].get('code_enc')


def test_enc17_windows_equal_full_decode():
    seg, _ = _e17()
    full = np.asarray(seg._raw_codes('w')).astype(np.int64)
    seg._codes.pop('w', None)                       # the cold road: no cached decode
    N = seg.N
    for lo, hi in ((0, 1), (0, 1000), (7, 8), (12345, 80000), (N // 2, N // 2 + 70001), (N - 5000, N), (0, N)):
        seg._codes.pop('w', None)
        part = np.asarray(seg._raw_codes_range('w', lo, hi)).astype(np.int64)
        assert np.array_equal(part, full[lo:hi]), (lo, hi)


def test_enc17_equality_counts_the_decoded_rows():
    seg, w = _e17()
    for v in (0, 7, 29, 1863, 2999):
        code = WS._code_of(seg, 'w', v)
        seg._codes.pop('w', None)
        got = 0 if code is None else len(WS._scan_eq(seg, 'w', int(code), 0, seg.N))
        assert got == int((w == v).sum()), (v, got, int((w == v).sum()))


_DB = None; _CON = None
def _big():
    global _DB, _CON
    if _DB is not None:
        return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'exr_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    # h spans the int64 extremes: odd low digits that float64 cannot hold
    _CON.execute("CREATE TABLE t AS SELECT i AS id, i % 7 AS k, "
                 "CAST(CASE WHEN i % 2 = 0 THEN -9223370484722263399 + i * 1000003 "
                 "ELSE 9223371440932022236 - i * 999983 END AS BIGINT) AS h FROM range(40000) t(i)")
    _DB = Database.create(d)
    pq = os.path.join(d, 't.parquet')
    _CON.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    _DB.cat.add_table('t', [['id', 'int'], ['k', 'int'], ['h', 'int']])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON


def _same(q):
    db, con = _big()
    got = sorted(tuple(r) for r in db.run(q)[0])
    want = sorted(tuple(r) for r in con.execute(q).fetchall())
    assert got == want, (q, got[:3], want[:3])


def test_min_big_int_filtered():  _same("SELECT MIN(h) FROM t WHERE k <> 3")
def test_max_big_int_filtered():  _same("SELECT MAX(h) FROM t WHERE k <> 3")
def test_min_max_big_int_grouped(): _same("SELECT k, MIN(h), MAX(h) FROM t GROUP BY k")
def test_min_big_int_grouped_filtered(): _same("SELECT k, MIN(h) FROM t WHERE k > 1 GROUP BY k")
