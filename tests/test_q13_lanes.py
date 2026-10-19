"""THE Q13 BOARD'S LANES (wdb_groupdistinct, 2026-10-13): GROUP BY k, COUNT(DISTINCT d) with
WHERE k <> <any literal> -- a number, a negative number, a string other than '', a timestamp, a value
no row holds -- and the walk chosen by the two columns' profiles: the DENSE lane (small group x target
table), the KEYMARK lane (a target space small enough for marker tables), the KEYSORT lane (anything else). Checked against DuckDB on the full answer (no LIMIT), and on
the top-10 with the key as tiebreak."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, duckdb, sqlglot
import helpers
import wdb_groupdistinct as GD

N = 300_000
_SEG = None


def _fixture():
    global _SEG
    if _SEG is not None:
        return _SEG
    rng = np.random.default_rng(13)
    df = pd.DataFrame({
        'small': rng.integers(0, 40, N),                          # dense with any small target
        'neg': rng.integers(-3, 5, N),                             # negative values
        'big': rng.integers(0, 200_000, N),                        # many groups: the key lanes
        'wide': rng.integers(0, 250_000, N),                       # big target
        'tiny': rng.integers(0, 6, N),
        'one': np.full(N, 4),                                      # a key with one value
        'dom': np.where(rng.random(N) < 0.99, 3, rng.integers(0, 500, N)),   # 99% one value
        's': np.array(['s%03d' % i for i in range(300)], dtype=object)[rng.integers(0, 300, N)],
        'ts': (pd.Timestamp('2013-07-01') + pd.to_timedelta(rng.integers(0, 50_000, N), unit='s')),
    })
    df.loc[rng.random(N) < 0.4, 'small'] = 7                       # one heavy value, like Q13's ''
    # born as a real database is: the suite's toy switches would admit repeating sequences (mode 4),
    # which no database is born with
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        _SEG = helpers.roundtrip(df)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    return _SEG


def _run(sql, lane=None):
    seg, pq = _fixture()
    tree = sqlglot.parse_one(sql)
    det = GD.detect(seg, tree, None)
    assert det is not None, 'the shape must reach the read: ' + sql
    before = dict(GD.LANES)
    got = GD.execute(seg, det, tree)
    assert got is not None, sql
    if lane is not None:
        assert GD.LANES.get(lane, 0) == before.get(lane, 0) + 1, (lane, GD.LANES, before)
    want = duckdb.sql(sql.replace('FROM t', "FROM '%s'" % pq)).fetchall()
    norm = lambda rows: [(str(a), int(b)) for a, b in rows]
    return norm(got[0]), norm(want)


def _full(k, d, lit, lane=None):
    g, w = _run("SELECT %s, COUNT(DISTINCT %s) AS u FROM t WHERE %s <> %s GROUP BY %s" % (k, d, k, lit, k), lane)
    assert sorted(g) == sorted(w), (k, d, lit, sorted(g)[:5], sorted(w)[:5])


def _top(k, d, lit, lane=None):
    g, w = _run("SELECT %s, COUNT(DISTINCT %s) AS u FROM t WHERE %s <> %s GROUP BY %s ORDER BY u DESC, %s LIMIT 10"
                % (k, d, k, lit, k, k), lane)
    assert g == w, (k, d, lit, g, w)


def test_dense_int_literal():
    _full('small', 'tiny', '7', 'dense')
    _top('small', 'tiny', '7', 'dense')


def test_dense_negative_literal():
    _full('neg', 'small', '-2', 'dense')


def test_string_literal_not_empty():
    _full('s', 'tiny', "'s017'", 'dense')


def test_literal_no_row_holds():
    _full('small', 'tiny', '12345', 'dense')
    _full('s', 'tiny', "'zzz'", 'dense')


def test_keymark_many_groups():
    _full('big', 'small', '5', 'keymark')
    _top('big', 'small', '5', 'keymark')


def test_keymark_big_target():
    _full('small', 'wide', '7', 'keymark')


def test_keymark_one_group_per_bucket():
    old = GD.DENSE_CELLS
    GD.DENSE_CELLS = 0                                  # small key space: SH = 0, every group its own bucket
    try:
        _full('small', 'tiny', '7', 'keymark')
        _full('s', 'neg', "'s001'", 'keymark')
    finally:
        GD.DENSE_CELLS = old


def test_keysort_past_the_marker_gate():
    old = (GD.DENSE_CELLS, GD.MARK_MAX)
    GD.DENSE_CELLS = 0; GD.MARK_MAX = 0
    try:
        _full('big', 'small', '5', 'keysort')
        _full('small', 'wide', '7', 'keysort')
        _top('big', 'wide', '5', 'keysort')
    finally:
        GD.DENSE_CELLS, GD.MARK_MAX = old


def test_timestamp_key_literal():
    seg, pq = _fixture()
    v = duckdb.sql("SELECT ts FROM '%s' GROUP BY ts ORDER BY COUNT(*) DESC LIMIT 1" % pq).fetchone()[0]
    _full('ts', 'tiny', "'%s'" % v)


def test_unfiltered_shape():
    g, w = _run("SELECT big, COUNT(DISTINCT wide) AS u FROM t GROUP BY big", "keymark")
    assert sorted(g) == sorted(w)


def test_filter_removes_the_only_value():
    _full('one', 'tiny', '4')
    _full('one', 'tiny', '5')                                     # a value no row holds: all rows stay


def test_pick_past_the_group_count_and_on_the_excluded_top():
    """the census pick: a LIMIT past the group count, and the filter removing the busiest group"""
    old = GD.DENSE_CELLS
    GD.DENSE_CELLS = 0
    try:
        g, w = _run("SELECT small, COUNT(DISTINCT big) AS u FROM t WHERE small <> 7 GROUP BY small "
                    "ORDER BY u DESC, small LIMIT 100", 'keymark')
        assert g == w and len(g) == 39, (len(g), g[:3], w[:3])
        _top('neg', 'big', '-3', 'keymark')
    finally:
        GD.DENSE_CELLS = old


def test_nullable_target_on_every_key_lane():
    """a NULL target marks its group present and is never counted -- on the marker and the sort walks"""
    import test_groupdistinct_filter as F
    old = (GD.DENSE_CELLS, GD.MARK_MAX)
    try:
        GD.DENSE_CELLS = 0
        F._case(200_000, 0.3, 50_000, null_share=0.1)             # keymark
        GD.MARK_MAX = 0
        F._case(200_000, 0.3, 50_000, null_share=0.1)             # keysort
        F._case(200_000, 0.3, 50, null_share=0.5)                 # keysort, long runs per group
    finally:
        GD.DENSE_CELLS, GD.MARK_MAX = old


def test_heavy_groups_counted_apart():
    """THE HEAVY GROUPS (2026-10-19): a key value holding most rows (no WHERE: the open form) is counted on
    every core apart from the key lanes' buckets -- on the marker and the sort walks, with and without a
    filter removing another value, with NULL targets"""
    old = (GD.DENSE_CELLS, GD.MARK_MAX, GD.HEAVY_MIN_N, GD.PRESENCE_CELLS, GD.PRESENCE_GROUPS)
    try:
        GD.DENSE_CELLS = 0; GD.HEAVY_MIN_N = 0; GD.PRESENCE_CELLS = 0; GD.PRESENCE_GROUPS = 0
        for mark in (True, False):
            if not mark:
                GD.MARK_MAX = 0
            lane = 'keymark' if mark else 'keysort'
            h0 = GD.HEAVY_HITS
            g, w = _run("SELECT dom, COUNT(DISTINCT big) AS u FROM t GROUP BY dom", lane)
            assert sorted(g) == sorted(w) and GD.HEAVY_HITS == h0 + 1, (lane, sorted(g)[:3], sorted(w)[:3])
            g, w = _run("SELECT small, COUNT(DISTINCT wide) AS u FROM t GROUP BY small ORDER BY u DESC, small LIMIT 10", lane)
            assert g == w, (lane, g, w)
            _full('small', 'wide', '3', lane)                        # the filter takes another value; 7 stays heavy
            _full('dom', 'big', '5', lane)
        import test_groupdistinct_filter as F
        GD.MARK_MAX = old[1]
        F._case(200_000, 0.3, 50_000, null_share=0.1)
    finally:
        GD.DENSE_CELLS, GD.MARK_MAX, GD.HEAVY_MIN_N, GD.PRESENCE_CELLS, GD.PRESENCE_GROUPS = old


def test_presence_lane_few_groups():
    """a handful of groups over a wide target, or any groups whose presence rows fit one shared table: each
    group a presence row (2026-10-19)"""
    old = (GD.DENSE_CELLS, GD.HEAVY_MIN_N)
    try:
        GD.DENSE_CELLS = 0; GD.HEAVY_MIN_N = 0
        _full('tiny', 'wide', '2', 'presence')
        _full('small', 'wide', '7', 'presence')                       # 40 groups x 250K targets: one table
        _top('big', 'small', '5', 'presence')                         # 200K groups x 40 targets
        g, w = _run("SELECT tiny, COUNT(DISTINCT big) AS u FROM t GROUP BY tiny", 'presence')
        assert sorted(g) == sorted(w)
        g, w = _run("SELECT one, COUNT(DISTINCT wide) AS u FROM t GROUP BY one", 'presence')
        assert sorted(g) == sorted(w)
        rng = np.random.default_rng(3)                                 # a NULL-able target; one group all NULL
        n = 120_000
        k = rng.integers(0, 4, n)
        u = pd.array(rng.integers(0, 60_000, n), dtype='Int64')
        u[rng.random(n) < 0.2] = pd.NA
        u[k == 3] = pd.NA
        seg, pq = helpers.roundtrip(pd.DataFrame({'k': k, 'u': u}))
        sql = "SELECT k, COUNT(DISTINCT u) AS c FROM t GROUP BY k"
        tree = sqlglot.parse_one(sql)
        b0 = GD.LANES.get('presence', 0)
        got = GD.execute(seg, GD.detect(seg, tree, None), tree)
        assert GD.LANES.get('presence', 0) == b0 + 1
        want = duckdb.sql(sql.replace('FROM t', "FROM '%s'" % pq)).fetchall()
        assert sorted((int(a), int(b)) for a, b in got[0]) == sorted((int(a), int(b)) for a, b in want)
    finally:
        GD.DENSE_CELLS, GD.HEAVY_MIN_N = old


def test_packed_scatter_equals_two_array_scatter():
    """the scatter lane's one-u32-per-row form counts what the (target, group) two-array form counts"""
    import wdb_kernels as WK
    rng = np.random.default_rng(8)
    for VR, VT, n in ((9040, 1 << 20, 400_000), (3, 70_000, 200_000), (60_000, 300_000, 500_000)):
        grp = rng.integers(0, VR, n).astype(np.uint16 if VR < 65536 else np.uint32)
        tgt = (rng.integers(0, VT, n) ** 1).astype(np.uint32)
        tgt[: n // 3] = rng.integers(0, 50, n // 3)                       # a few targets shared by many rows
        SH = max(1, int(VT - 1).bit_length() - 12)
        GB = max(1, int(VR - 1).bit_length())
        assert SH + GB <= 32
        ku, kr, offs = WK.gd_pass1(tgt, grp, np.int64(SH), np.int64(8))
        a = WK.gd_pass2_count(ku, kr, offs, np.int64(SH), np.int64(VR))
        pk, offs2 = WK.gd_pass1_packed(tgt, grp, np.int64(SH), np.int64(GB), np.int64(8))
        b = WK.gd_pass2_packed(pk, offs2, np.int64(SH), np.int64(GB), np.int64(VR))
        want = np.bincount(np.unique(grp.astype(np.int64) * VT + tgt) // VT, minlength=VR)
        assert np.array_equal(a, want) and np.array_equal(b, want), (VR, VT)


def test_topk_capped_census():
    """gd_topk with counts past its census width: the tall counts are picked apart, exactly the full
    sort's first k (count descending, code ascending), with the filtered group left out"""
    import wdb_kernels as WK
    rng = np.random.default_rng(5)
    old = WK.GD_TOPK_CAP
    try:
        WK.GD_TOPK_CAP = 50
        for _ in range(20):
            runs = rng.integers(-1, 120, 3000).astype(np.int32)
            ec = int(rng.integers(-1, 3000))
            for k in (1, 3, 10, 40, 200):
                got = list(WK.gd_topk(runs, ec, k))
                idx = [i for i in range(runs.size) if runs[i] >= 0 and i != ec]
                want = sorted(idx, key=lambda i: (-int(runs[i]), i))[:k]
                assert got == want, (k, ec, got[:5], want[:5])
    finally:
        WK.GD_TOPK_CAP = old


def test_survivor_road_when_the_removed_group_dominates():
    """the filter keeps ~1%: the counted column is read at the survivors only"""
    h0 = GD.SURVIVOR_HITS
    _full('dom', 'big', '3')
    assert GD.SURVIVOR_HITS == h0 + 1
    _full('small', 'big', '7')                                    # keeps 60%: the whole column
    assert GD.SURVIVOR_HITS == h0 + 1
    _top('dom', 'wide', '3')
    _full('dom', 's', '3')
    old = GD.DENSE_CELLS
    GD.DENSE_CELLS = 0
    try:
        _full('dom', 'tiny', '3', 'keymark')
    finally:
        GD.DENSE_CELLS = old
