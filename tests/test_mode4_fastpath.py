"""Mode-4 equality fast-path (step 4d): O(1)-compute '= X' / '!= X' masks for clean-affine
integer columns (n_exc==0, no overrides) via the affine inverse, instead of decoding+comparing
all N values. SAFETY NET: the fast-path mask must EXACTLY equal the brute-force comparison for
every (column, value) -- verified over a fuzz of bases/strides (incl negative & descending &
stride-0) and value classes (hit / miss / non-divisible / out-of-range / boundary). Columns
with exceptions or overrides must fall back and stay correct (vs DuckDB)."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb, sqlglot
import wdb_encode, wdb_override, wdb_sql
from wdb_engine import Segment

TMP = tempfile.gettempdir()
def _seg(col):
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/fp_{t}.parquet'; wdb = f'{TMP}/fp_{t}.wdb'
    pd.DataFrame({'x': np.asarray(col, dtype=np.int64)}).to_parquet(pq, index=False)
    wdb_encode.encode(pq, wdb); return Segment(wdb), wdb, pq
def _mask(seg, where):
    node = sqlglot.parse_one(f"SELECT x FROM t WHERE {where}").args['where'].this
    return wdb_sql._eval_pred(seg, node, lambda c: c)

def test_eq_fastpath_exactly_matches_bruteforce_fuzz():
    rng = np.random.default_rng(2026)
    for _ in range(150):
        N = int(rng.integers(20, 3000))
        base = int(rng.integers(-10**11, 10**11))
        stride = int(rng.choice([1, 1, 1, 2, 7, 13, 1000, -1, -3]))   # incl descending
        col = base + stride * np.arange(N, dtype=np.int64)
        seg, wdb, pq = _seg(col)
        assert seg.cols['x']['mode'] == 4, "fuzz column should be mode 4 (clean affine)"
        vals = seg.values('x')
        # value classes: exact hits, just-off (non-divisible), out of range, boundaries
        Xs = set()
        for p in rng.integers(0, N, size=6): Xs.add(int(base + stride * int(p)))
        if abs(stride) > 1:
            Xs.add(int(base + 1)); Xs.add(int(col[N//2]) + 1)
        Xs.update({int(base - stride), int(base + stride * N), int(col[0]), int(col[-1]), 0, -1})
        for X in Xs:
            assert np.array_equal(_mask(seg, f"x = {X}"), vals == X), f"EQ {X} base={base} stride={stride} N={N}"
            assert np.array_equal(_mask(seg, f"x != {X}"), vals != X), f"NEQ {X} base={base} stride={stride} N={N}"
        for f in (wdb, pq):
            if os.path.exists(f): os.remove(f)

def test_constant_column_stride0_fastpath():
    seg, wdb, pq = _seg(np.full(800, 42, dtype=np.int64))     # stride 0 -> mode 4
    assert seg.cols['x']['mode'] == 4
    assert _mask(seg, "x = 42").all() and not _mask(seg, "x = 43").any()
    assert not _mask(seg, "x != 42").any() and _mask(seg, "x != 43").all()

def test_eq_query_matches_oracle():
    n = 5000
    df = pd.DataFrame({'id': 1_000_000 + np.arange(n, dtype=np.int64),
                       'g': (['a','b','c','d','e'] * (n // 5))})
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/fpq_{t}.parquet'; wdb = f'{TMP}/fpq_{t}.wdb'
    df.to_parquet(pq, index=False); wdb_encode.encode(pq, wdb); seg = Segment(wdb)
    assert seg.cols['id']['mode'] == 4
    con = duckdb.connect()
    for X in [1_000_000, 1_002_500, 1_004_999, 999_999, 2_000_000]:     # hits + misses
        for sql in [f"SELECT g FROM TBL WHERE id = {X}", f"SELECT COUNT(*) FROM TBL WHERE id != {X}"]:
            duck = sorted(con.execute(sql.replace('TBL', f"'{pq}'")).fetchall())
            rows, _ = wdb_sql.execute(seg, sql.replace('TBL', 'tbl'))
            assert sorted(rows) == duck, sql

def test_fallback_with_exceptions_still_correct():
    # gaps column (n_exc>0) -> fast-path declines, general path must be correct vs brute
    keep = np.random.default_rng(7).random(6000) > 0.02
    col = np.flatnonzero(keep)[:5000].astype(np.int64)
    seg, wdb, pq = _seg(col)
    assert seg.cols['x']['mode'] == 4
    vals = seg.values('x')
    for X in [int(col[100]), int(col[2500]), int(col[-1]), int(col[0]) - 1, 999_999]:
        assert np.array_equal(_mask(seg, f"x = {X}"), vals == X)
        assert np.array_equal(_mask(seg, f"x != {X}"), vals != X)

def test_fallback_with_override_still_correct():
    seg, wdb, pq = _seg(1_000_000 + np.arange(3000, dtype=np.int64))
    assert seg.cols['x']['mode'] == 4
    wdb_override.set_override(wdb, 'x', [10], np.array([7_777_777], dtype=np.int64))
    seg2 = Segment(wdb)                                   # now has an override -> fast-path declines
    vals = seg2.values('x')
    for X in [7_777_777, 1_000_010, 1_000_000, 1_002_999]:
        assert np.array_equal(_mask(seg2, f"x = {X}"), vals == X), X
        assert np.array_equal(_mask(seg2, f"x != {X}"), vals != X), X
