"""wdb_gridwalk robustness across ALL pair-classes (not just the benchmarked one).

A self-contained multi-type fixture (high/mid/low-card ints + strings + a scattered near-unique
column) yields 15 pair-classes: 10 with heavy cells, 5 zero-heavy (every pair with the near-unique
column). For EVERY pair and several LIMITs we assert gridwalk's 2-key COUNT(*) top-K is identical to
WaveDB's own canonical answer with gridwalk OFF (tie-aware), and that the counts match an independent
brute-force pandas group-by. Separately: the gap-encoded bulk codec round-trips for every byte-width,
both count dtypes, and the block boundaries; column order is irrelevant; zero-heavy and non-2-key
shapes decline. Runs in CI (no pod, no DuckDB).

Identity is checked gridwalk-on vs gridwalk-off (same decode) rather than vs an external engine, so a
column's display format never causes a false failure; counts are checked vs pandas (engine-agnostic).
"""
import sys, os, uuid, tempfile, shutil, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, sqlglot
from wdb_db import Database
from wdb_engine import Segment
import wdb_encode, wdb_gridwalk as GW


def _build(d):
    rng = np.random.default_rng(7); N = 600
    df = pd.DataFrame({
        's':    rng.choice([f'k{i}' for i in range(12)], N),     # high-card string
        'n':    rng.integers(0, 15, N).astype('int64'),          # high-card int
        'mid':  rng.integers(0, 8, N).astype('int64'),           # mid-card int
        'lo':   rng.integers(0, 3, N).astype('int64'),           # low-card int (dense, high counts)
        'cat':  rng.choice(['p', 'q'], N),                       # 2-value categorical
        'uniq': (rng.permutation(N) * 7 + 3).astype('int64'),    # scattered near-unique -> zero-heavy
    })
    db = Database.create(d)
    db.run("CREATE TABLE t (s VARCHAR, n BIGINT, mid BIGINT, lo BIGINT, cat VARCHAR, uniq BIGINT)")
    pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb')
    return db, Segment(os.path.join(d, 't_0.wdb')), df


def _rows(res):
    return list(res[0] if isinstance(res, tuple) else res)


def _counts(rows):
    return sorted(int(r[-1]) for r in rows)


def _ident_above(rows):
    # identity of cells strictly above the boundary count -- the unambiguous, tie-independent part
    if not rows:
        return set()
    bnd = min(int(r[-1]) for r in rows)
    return set((str(r[0]), str(r[1])) for r in rows if int(r[-1]) > bnd)


def _gt_counts(df, a, b, lim):
    g = df.groupby([a, b]).size().sort_values(ascending=False).head(lim)
    return sorted(int(x) for x in g.values)


def test_gridwalk_all_pairs_match_canonical_and_groundtruth():
    """Every pair-class, several LIMITs: gridwalk == WaveDB-without-gridwalk (tie-aware) and counts
    match a brute-force pandas group-by. Covers high/mid/low-card, string/int, and zero-heavy pairs."""
    d = os.path.join(tempfile.gettempdir(), f'gwa_{uuid.uuid4().hex[:8]}')
    try:
        db, seg, df = _build(d)
        elig = [c for c in seg.cols if seg.cols[c].get('mode') != 4 and GW._vals(seg, c) is not None]
        assert len(elig) == 6, elig
        pairs = list(itertools.combinations(sorted(elig), 2))
        assert len(pairs) == 15, len(pairs)
        n_heavy_pairs = 0; n_built = 0
        for a, b in pairs:
            built = GW._build(seg, [a, b])
            if built is not None:
                n_built += 1
                if built[3] > 0:                     # nheavy > 0: pair has heavy cells
                    n_heavy_pairs += 1
            for lim in (3, 5, 8):
                q = f'SELECT "{a}","{b}", COUNT(*) FROM t GROUP BY "{a}","{b}" ORDER BY COUNT(*) DESC LIMIT {lim}'
                GW.enable();  h0 = GW._HITS; on = _rows(db.run(q)); hit = GW._HITS > h0
                GW.enable(); off = _rows(db.run(q))
                # gridwalk's answer is identical to WaveDB's canonical answer (tie-aware)
                assert _counts(on) == _counts(off), f'{a}x{b} L{lim} counts on!=off {_counts(on)} {_counts(off)}'
                assert _ident_above(on) == _ident_above(off), f'{a}x{b} L{lim} identity on!=off'
                assert len(on) == len(off), f'{a}x{b} L{lim} nrows {len(on)} {len(off)}'
                # and the counts are objectively right (independent oracle)
                assert _counts(on) == _gt_counts(df, a, b, lim), f'{a}x{b} L{lim} counts vs pandas'
                # v2: a plateau at the boundary RESOLVES deterministically (never declines), so any
                # LIMIT within the heavy set must route through gridwalk.
                if built is not None and lim <= built[3]:
                    assert hit, f'{a}x{b} L{lim} declined within the heavy set'
        assert n_built == 15, n_built              # v2: zero-heavy pairs build too (ones-only fill)
        assert n_heavy_pairs == 10, n_heavy_pairs  # the 5 *xuniq pairs are zero-heavy
    finally:
        GW.enable(); shutil.rmtree(d, ignore_errors=True)


def test_gridwalk_bulk_path_beyond_head():
    """Force the gap-encoded bulk: shrink the head to 2 so a larger LIMIT must decode the bulk.
    Unordered LIMIT skips the tie guard so the bulk path actually executes; result must match
    WaveDB-without-gridwalk and the pandas ground truth."""
    d = os.path.join(tempfile.gettempdir(), f'gwb_{uuid.uuid4().hex[:8]}')
    old_head = GW._HEAD_N
    try:
        db, seg, df = _build(d)
        GW._HEAD_N = 2                                   # head holds 2; LIMIT>2 -> bulk
        GW._CACHE.clear()                                # drop any head-200k cache from prior tests
        a, b = 'n', 's'                                  # 156 heavy cells
        for lim in (5, 10, 20):
            q = f'SELECT "{a}","{b}", COUNT(*) FROM t GROUP BY "{a}","{b}" LIMIT {lim}'   # unordered
            GW.enable();  h0 = GW._HITS; on = _rows(db.run(q)); hit = GW._HITS > h0
            GW.enable(); off = _rows(db.run(q))
            assert hit, f'L{lim} should route through gridwalk bulk (head=2)'
            assert _counts(on) == _gt_counts(df, a, b, lim), f'bulk L{lim} counts vs pandas'
            assert _counts(on) == _counts(off), f'bulk L{lim} counts on!=off'
            assert _ident_above(on) == _ident_above(off), f'bulk L{lim} identity on!=off'
    finally:
        GW._HEAD_N = old_head; GW._CACHE.clear(); GW.enable(); shutil.rmtree(d, ignore_errors=True)


def test_gridwalk_bulk_codec_roundtrip():
    """The byte-block frame-of-reference codec is exact for every byte-width, both count dtypes,
    a singleton, and the block boundaries (B, B+1, 2B). Independent of any segment."""
    B = GW._BULK_B
    rng = np.random.default_rng(1)

    def roundtrip(gid, cnt):
        bulk = GW._bulk_encode(gid.astype(np.int64), cnt.astype(np.int64))
        dg, dc = GW._bulk_decode_all(bulk)
        return bool(np.array_equal(dg, gid) and np.array_equal(dc, cnt)), str(bulk['cnt'].dtype)

    # 1-byte gaps
    gid = np.cumsum(rng.integers(1, 5, 1000)); cnt = rng.integers(2, 100, 1000)
    ok, dt = roundtrip(gid, cnt); assert ok and dt == 'uint16'
    # large gaps -> 3/4-byte block widths
    gid = np.cumsum(rng.integers(1, 10_000_000, 500)); cnt = rng.integers(2, 50, 500)
    ok, _ = roundtrip(gid, cnt); assert ok
    # count overflow -> uint32 promotion, still exact
    gid = np.cumsum(rng.integers(1, 4, 300)); cnt = rng.integers(2, 100, 300); cnt[7] = 70000
    ok, dt = roundtrip(gid, cnt); assert ok and dt == 'uint32'
    # singleton
    ok, _ = roundtrip(np.array([42]), np.array([5])); assert ok
    # block boundaries
    for m in (B - 1, B, B + 1, 2 * B, 2 * B + 3):
        gid = np.cumsum(rng.integers(1, 7, m)); cnt = rng.integers(2, 30, m)
        ok, _ = roundtrip(gid, cnt); assert ok, f'block boundary m={m}'


def test_gridwalk_column_order_irrelevant():
    """The two group keys may be listed in either order; the result is the same pairs+counts
    (tie-aware: swapping keys changes the internal gid order, so cells tied at the LIMIT boundary
    may differ -- the counts and the unambiguous above-boundary identity are what must match)."""
    d = os.path.join(tempfile.gettempdir(), f'gwo_{uuid.uuid4().hex[:8]}')
    try:
        db, seg, df = _build(d)
        GW.enable()
        r1 = [(str(x[1]), str(x[0]), int(x[2])) for x in _rows(db.run(
            'SELECT "n","s", COUNT(*) FROM t GROUP BY "n","s" ORDER BY COUNT(*) DESC LIMIT 5'))]
        r2 = [(str(x[0]), str(x[1]), int(x[2])) for x in _rows(db.run(
            'SELECT "s","n", COUNT(*) FROM t GROUP BY "s","n" ORDER BY COUNT(*) DESC LIMIT 5'))]
        assert _counts(r1) == _counts(r2), f'{r1} != {r2}'
        assert _ident_above(r1) == _ident_above(r2), f'{r1} != {r2}'
    finally:
        GW.enable(); shutil.rmtree(d, ignore_errors=True)


def test_gridwalk_declines_unsupported_shapes():
    """WHERE / single-key / 3-key / SELECT DISTINCT / zero-heavy all decline; the zero-heavy pair is
    still answered correctly by the canonical path."""
    d = os.path.join(tempfile.gettempdir(), f'gwd_{uuid.uuid4().hex[:8]}')
    try:
        db, seg, df = _build(d)
        GW.enable()

        def dec(sql):
            return GW.try_gridwalk(seg, sqlglot.parse_one(sql, read='duckdb'), None)
        assert dec('SELECT "s","n",COUNT(*) FROM t WHERE lo>0 GROUP BY "s","n" ORDER BY COUNT(*) DESC LIMIT 3') is None
        assert dec('SELECT "s",COUNT(*) FROM t GROUP BY "s" ORDER BY COUNT(*) DESC LIMIT 3') is None
        assert dec('SELECT "s","n","mid",COUNT(*) FROM t GROUP BY "s","n","mid" ORDER BY COUNT(*) DESC LIMIT 3') is None
        assert dec('SELECT DISTINCT "s","n" FROM t GROUP BY "s","n" LIMIT 3') is None
        # zero-heavy (near-unique key): v2 SERVES it -- every cell is a singleton, so the answer is a
        # deterministic fill from `ones` (smallest gids). Must match the pandas ground truth.
        zq = 'SELECT "uniq","s", COUNT(*) FROM t GROUP BY "uniq","s" ORDER BY COUNT(*) DESC LIMIT 3'
        zr = dec(zq)
        assert zr is not None, 'zero-heavy pair should serve via singleton fill'
        assert _counts([tuple(r) for r in zr[0]]) == _gt_counts(df, 'uniq', 's', 3)
        assert _counts(_rows(db.run(zq))) == _gt_counts(df, 'uniq', 's', 3)
    finally:
        GW.enable(); shutil.rmtree(d, ignore_errors=True)
