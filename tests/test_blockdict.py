"""enc 19 = THE BLOCK DICTIONARIES (Jackson, 2026-09-23: pointer compression one level down).
Per block of BR rows, the sorted distinct codes present are stored once (first code + gaps) and each
row keeps a pointer into its block's list at the block's own width. Decode is one jump per row.
Elected over an inflating dress (zstd 1 / blocked 3 / packed frames 18) within 5% of its bytes.
Lossless vs the original, exact vs the full decode, and query-correct vs DuckDB."""
import sys, os, uuid, tempfile, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_sql
from wdb_engine import Segment

TMP = tempfile.gettempdir()


@contextlib.contextmanager
def _force19():
    os.environ['WDB_E19_FORCE'] = '1'
    try: yield
    finally: os.environ.pop('WDB_E19_FORCE', None)


def _enc(df):
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/bd19_{t}.parquet'; w = f'{TMP}/bd19_{t}.wdb'
    df.to_parquet(pq, index=False); wdb_encode.encode(pq, w); return Segment(w), w, pq


_FIX = None
def _fixture():
    """users in sessions: each 16K-row stretch sees a few thousand of 120K users (UserID's shape)"""
    global _FIX
    if _FIX is not None: return _FIX
    rng = np.random.default_rng(19)
    n = 300_000
    pool = rng.integers(0, 120_000, n // 16384 + 1)
    uid = np.empty(n, np.int64)
    for b in range(0, n, 16384):
        live = rng.integers(0, 120_000, 3000)
        uid[b:b + 16384] = live[rng.integers(0, 3000, min(16384, n - b))]
    user = np.array([f'user-{u:06d}' for u in uid])
    k = rng.integers(0, 40, n).astype(np.int64)
    df = pd.DataFrame({'user': user, 'uid': uid * 7919 + 13, 'k': k})
    with _force19():
        seg, w, pq = _enc(df)
    _FIX = (seg, w, pq, df)
    return _FIX


def _kernel_roundtrip(codes, bits, BR):
    import wdb_kernels as K
    a = np.ascontiguousarray(codes, dtype=np.int64); N = a.size; nb = (N + BR - 1) // BR
    lb = np.empty(nb, np.uint8); gw = np.empty(nb, np.uint8); dc = np.empty(nb, np.uint32)
    pwn = np.empty(nb, np.int64); dwn = np.empty(nb, np.int64)
    K.e19_plan(a, np.int64(BR), np.int64(bits), lb, gw, dc, pwn, dwn)
    poff = np.zeros(nb + 1, np.int64); np.cumsum(pwn, out=poff[1:])
    doff = np.zeros(nb + 1, np.int64); np.cumsum(dwn, out=doff[1:])
    pw = np.zeros(int(poff[-1]) + 1, np.uint64); dw = np.zeros(int(doff[-1]) + 1, np.uint64)
    K.e19_write(a, np.int64(BR), np.int64(bits), lb, gw, poff, doff, pw, dw)
    out = np.empty(N, np.int64)
    K.e19_decode(pw, dw, np.int64(BR), np.int64(N), np.int64(bits), lb, gw, dc, poff, doff, out)
    assert np.array_equal(out, a), (bits, BR)
    rows = np.sort(np.random.default_rng(bits).choice(N, min(N, 777), replace=False)).astype(np.int64)
    blk = rows // BR
    starts = np.concatenate(([0], np.flatnonzero(blk[1:] != blk[:-1]) + 1, [rows.size])).astype(np.int64)
    g = np.empty(rows.size, np.int64)
    K.e19_gather(pw, dw, np.int64(BR), np.int64(bits), lb, gw, dc, poff, doff, rows, starts, g)
    assert np.array_equal(g, a[rows]), (bits, BR)


def test_kernels_roundtrip_every_width():
    rng = np.random.default_rng(1)
    for bits in (1, 5, 9, 14, 17, 25, 31, 32):
        top = (1 << bits) - 1
        for BR in (1024, 4096):
            n = 3 * BR + 17
            _kernel_roundtrip(rng.integers(0, top + 1, n, dtype=np.int64), bits, BR)       # dense
            c = rng.integers(0, top + 1, n, dtype=np.int64); c[BR:2 * BR] = top               # a one-code block (width 0)
            c[0] = 0; c[1] = top                                                              # the widest gap
            _kernel_roundtrip(c, bits, BR)


def test_elected_and_lossless():
    seg, w, pq, df = _fixture()
    c = seg.cols['user']                                   # (uid: enc 5's later election may take it -- lawful)
    assert c['code_enc'] == 19, c.get('code_enc')
    assert 'boffs' not in c and 'cwidth' not in c and 'BR' not in c   # the enc-3 readers must not see it
    got = np.array([x.decode() for x in seg.values('user')])
    assert np.array_equal(got, df['user'].to_numpy())
    assert np.array_equal(np.asarray(seg.values('uid')).astype(np.int64), df['uid'].to_numpy())


def test_point_reads_and_ranges_match_full_decode():
    seg, w, pq, df = _fixture()
    full = np.asarray(seg._raw_codes('user')).astype(np.int64)
    BR = int(seg.cols['user']['e19BR'])
    rng = np.random.default_rng(2)
    for rows in (np.sort(rng.choice(seg.N, 3000, replace=False)), np.arange(BR * 2, BR * 2 + 50),
                 np.array([5, 3, BR + 4, 1, 200_000, BR + 1, seg.N - 1]), rng.choice(seg.N, 20_000),
                 np.array([], dtype=np.int64)):
        fresh = Segment(w)                                   # never the cached full decode
        assert np.array_equal(np.asarray(fresh.codes_at('user', rows)).astype(np.int64), full[rows]), rows[:5]
    for lo, hi in ((0, 10), (BR - 6, BR + 6), (100_000, 180_000), (0, seg.N), (seg.N - 10, seg.N)):
        fresh = Segment(w)
        assert np.array_equal(np.asarray(fresh._raw_codes_range('user', lo, hi)).astype(np.int64), full[lo:hi]), (lo, hi)
        fresh = Segment(w)
        assert np.array_equal(np.asarray(fresh.codes_band('user', lo, hi)).astype(np.int64), full[lo:hi]), (lo, hi)


def test_signposts_and_label_scan_match_full_decode():
    """THE BOX LABELS + SIGNPOSTS (2026-09-27): every 128th label entry kept whole; 'rows with code k'
    through the labels -- with and without the signposts -- equals the full decode, for present and
    absent codes, the first and last codes, and partial row ranges crossing block edges"""
    import wdb_blockstats, wdb_wherescan, wdb_kernels as K
    seg, w, pq, df = _fixture()
    c = seg.cols['user']
    full = np.asarray(seg._raw_codes('user')).astype(np.int64)
    sp, spo = wdb_blockstats.e19_signposts(seg, 'user')
    sp = np.asarray(sp, np.int64); BR = int(c['e19BR']); S = wdb_blockstats.SIGNPOST_EVERY
    for b in range(c['e19dc'].size):                      # each signpost is that label entry
        lab = np.unique(full[b * BR:min(seg.N, (b + 1) * BR)])
        assert np.array_equal(sp[spo[b]:spo[b + 1]], lab[::S]), b
    V = int(c['V']); rng = np.random.default_rng(27)
    present = full[rng.integers(0, seg.N, 20)]
    codes = sorted(set(present.tolist()) | {0, V - 1, int(full[0]), int(full[-1]), int(full[BR]), int(full[BR - 1])})
    absent = sorted(set(range(V)) - set(np.unique(full).tolist()))[:5]
    orig = wdb_blockstats.signposts_from_load
    try:
        for mode in ('signposts', 'walk'):
            wdb_blockstats.signposts_from_load = (lambda s, cc: (sp, spo, S)) if mode == 'signposts' else (lambda s, cc: None)
            sg = Segment(w)                               # a fresh segment: never the cached full decode
            for k in codes + absent:
                for lo, hi in ((0, seg.N), (BR - 5, 2 * BR + 5), (123, 124), (seg.N - 3, seg.N)):
                    got = wdb_wherescan._scan_eq19(sg, 'user', int(k), lo, hi)
                    want = lo + np.flatnonzero(full[lo:hi] == k)
                    assert got is not None and np.array_equal(got, want), (mode, k, lo, hi)
    finally:
        wdb_blockstats.signposts_from_load = orig


def test_label_scan_queries_match_the_old_scan():
    """the same answers with the label scan switched off (WDB_E19_LABELS=0: the full decode)"""
    seg, w, pq, df = _fixture()
    lit = df['user'].iloc[99_999]
    sqls = [f"SELECT COUNT(*) FROM tbl WHERE user = '{lit}'", f"SELECT k, user FROM tbl WHERE user = '{lit}' ORDER BY k LIMIT 7",
            "SELECT COUNT(*) FROM tbl WHERE user = 'no-such-user'"]
    a = [sorted(wdb_sql.execute(Segment(w), q)[0]) for q in sqls]
    os.environ['WDB_E19_LABELS'] = '0'
    try:
        b = [sorted(wdb_sql.execute(Segment(w), q)[0]) for q in sqls]
    finally:
        os.environ.pop('WDB_E19_LABELS', None)
    assert a == b


def test_queries_match_duck():
    seg, w, pq, df = _fixture()
    con = duckdb.connect()
    lit = df['user'].iloc[123_456]; lit2 = df['user'].iloc[7]; v = int(df['uid'].iloc[250_001])
    for sql in [f"SELECT COUNT(*) FROM TBL WHERE user = '{lit}'",
                f"SELECT k, COUNT(*) FROM TBL WHERE user = '{lit}' GROUP BY k ORDER BY k",
                f"SELECT COUNT(*) FROM TBL WHERE user <> '{lit}'",
                f"SELECT COUNT(*) FROM TBL WHERE user IN ('{lit}', '{lit2}')",
                f"SELECT COUNT(*) FROM TBL WHERE uid = {v}",
                "SELECT COUNT(*) FROM TBL WHERE user LIKE '%-0001%'",
                "SELECT user, COUNT(*) FROM TBL GROUP BY user ORDER BY 2 DESC, 1 LIMIT 10",
                "SELECT COUNT(DISTINCT user) FROM TBL",
                "SELECT COUNT(DISTINCT uid) FROM TBL WHERE k < 20",
                "SELECT k, COUNT(DISTINCT user) FROM TBL GROUP BY k ORDER BY 2 DESC, 1 LIMIT 5",
                "SELECT k, COUNT(DISTINCT uid) FROM TBL WHERE k < 5 GROUP BY k ORDER BY k",
                "SELECT MIN(user), MAX(user), SUM(uid) FROM TBL WHERE k = 3",
                "SELECT uid, user, COUNT(*) FROM TBL GROUP BY uid, user ORDER BY 3 DESC, 1 LIMIT 5"]:
        duck = sorted(con.execute(sql.replace('TBL', f"'{pq}'")).fetchall())
        rows, _ = wdb_sql.execute(Segment(w), sql.replace('TBL', 'tbl'))
        assert sorted(tuple(x.decode() if isinstance(x, bytes) else x for x in r) for r in rows) == duck, sql


def test_election_takes_it_within_slack_and_not_beyond():
    """JACKSON'S RULE: enc 19 replaces an inflating dress within E19_SLACK of its bytes. Sessions of
    users (few distinct per block, no runs for zstd) elect it; a skewed mid-V code with no runs
    (zstd's entropy coding is its home ground, RegionID's shape) does not."""
    rng = np.random.default_rng(7)
    n = 1 << 21
    sess = np.empty(n, np.int64)
    for b in range(0, n, 65536):
        live = rng.integers(0, 1 << 22, 12000)
        sess[b:b + 65536] = live[rng.integers(0, 12000, 65536)]
    sec = wdb_encode._code_section(sess, 22)
    assert sec[0] == 19, sec[0]
    skew = np.minimum(rng.zipf(1.4, n), 5000) - 1         # skewed mid-V, no runs: zstd's entropy wins
    sec = wdb_encode._code_section(skew, 13)
    assert sec[0] in (1, 3, 18), sec[0]                   # an inflating dress, and enc 19 did not take it
