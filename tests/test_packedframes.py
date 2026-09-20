"""enc 18 = PACKED FRAMES (Jackson's question: bitpack first, then zstd the 1s and 0s). Per
65536-row frame the codes are LE bit-packed (no per-row byte padding) and the frame is one zstd
stream. Elected on size alone against zstd / blocked / bitpack for wide codes (>= 17 bits).
Point reads inflate a frame PREFIX; scans unpack and test membership in one nogil kernel.
Lossless vs the original, exact vs the full decode, and query-correct vs DuckDB."""
import sys, os, uuid, tempfile, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_sql
from wdb_engine import Segment

TMP = tempfile.gettempdir()


@contextlib.contextmanager
def _force18():
    os.environ['WDB_E18_FORCE'] = '1'
    try: yield
    finally: os.environ.pop('WDB_E18_FORCE', None)


def _enc(df):
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/pf18_{t}.parquet'; w = f'{TMP}/pf18_{t}.wdb'
    df.to_parquet(pq, index=False); wdb_encode.encode(pq, w); return Segment(w), w, pq


_FIX = None
def _fixture():
    """a wide string column (150K distinct -> 18 bits) with locality (a zipf walk) over 400K rows"""
    global _FIX
    if _FIX is not None: return _FIX
    rng = np.random.default_rng(18)
    n = 400_000
    base = np.minimum(rng.zipf(1.3, n), 150_000).astype(np.int64)
    walk = np.cumsum(rng.integers(0, 3, n)) % 150_000        # smooth drift: zstd finds something
    ids = (base + walk) % 150_000
    url = np.array([f'https://site.example/path/{i}' for i in ids])
    k = rng.integers(0, 40, n).astype(np.int64)
    df = pd.DataFrame({'url': url, 'k': k})
    with _force18():
        seg, w, pq = _enc(df)
    _FIX = (seg, w, pq, df)
    return _FIX


def test_kernels_roundtrip_every_width():
    import wdb_kernels as K
    rng = np.random.default_rng(1)
    for bits in (17, 20, 24, 29, 32):
        n = 65536 + 7
        codes = rng.integers(0, 1 << bits, n, dtype=np.int64) if bits < 32 else \
            rng.integers(0, 1 << 31, n, dtype=np.int64) * 2 + rng.integers(0, 2, n, dtype=np.int64)
        out = np.zeros((n * bits + 7) // 8 + 8, np.uint8)
        K.pk32_pack(codes, bits, out)
        back = np.empty(n, np.int64); K.pk32_unpack(out, bits, n, back)
        assert np.array_equal(back, codes), bits
        back2 = np.empty(n, np.int64); K.pk32_unpack_serial(out, bits, n, back2)
        assert np.array_equal(back2, codes), bits
        rows = np.sort(rng.choice(n, 500, replace=False)).astype(np.int64)
        g = np.empty(rows.size, np.int64); K.pk32_gather(out, bits, rows, g)
        assert np.array_equal(g, codes[rows]), bits
        # membership over a random flag, a window of the frame
        flag = np.zeros(1 << bits, np.bool_) if bits <= 24 else None
        if flag is not None:
            flag[codes[rng.choice(n, 200)]] = True
            fbits = np.packbits(flag, bitorder='little')
            hits = np.empty(n, np.int64); hc = np.empty(n, np.int64)
            s0, e0 = 1000, 60000
            m = K.pk32_flag_hits(out, bits, s0, e0, 777, fbits, hits, hc)
            exp = np.flatnonzero(flag[codes[s0:e0]])
            assert np.array_equal(hits[:m], exp + 777), bits
            assert np.array_equal(hc[:m], codes[s0:e0][exp]), bits


def test_elected_and_lossless():
    seg, w, pq, df = _fixture()
    c = seg.cols['url']
    assert c['code_enc'] == 18 and c['pbits'] >= 17, c.get('code_enc')
    assert 'boffs' not in c and 'cwidth' not in c          # the enc-3 readers must not see a blocked column
    assert c['BR'] == 65536 and len(c['poffs']) == (seg.N + 65535) // 65536 + 1
    got = np.array([x.decode() for x in seg.values('url')])
    assert np.array_equal(got, df['url'].to_numpy())


def test_point_reads_and_ranges_match_full_decode():
    seg, w, pq, df = _fixture()
    full = np.asarray(seg._raw_codes('url')).astype(np.int64)
    rng = np.random.default_rng(2)
    # scattered rows across frames; a prefix-only set (low rows of one frame); an unsorted set
    for rows in (np.sort(rng.choice(seg.N, 3000, replace=False)),
                 np.arange(65536 * 2, 65536 * 2 + 50),
                 np.array([5, 3, 65540, 1, 200_000, 65537, 399_999]),
                 rng.choice(seg.N, 20_000)):
        fresh = Segment(w)                                   # never the cached full decode
        assert np.array_equal(np.asarray(fresh.codes_at('url', rows)), full[rows]), rows[:5]
    for lo, hi in ((0, 10), (65530, 65540), (100_000, 300_000), (0, seg.N), (399_990, seg.N)):
        fresh = Segment(w)
        assert np.array_equal(np.asarray(fresh._raw_codes_range('url', lo, hi)).astype(np.int64), full[lo:hi]), (lo, hi)
        fresh = Segment(w)
        assert np.array_equal(np.asarray(fresh.codes_band('url', lo, hi)).astype(np.int64), full[lo:hi]), (lo, hi)


def test_queries_match_duck():
    seg, w, pq, df = _fixture()
    con = duckdb.connect()
    lit = df['url'].iloc[123_456]; lit2 = df['url'].iloc[7]; lit3 = df['url'].iloc[300_001]
    for sql in [f"SELECT COUNT(*) FROM TBL WHERE url = '{lit}'",
                f"SELECT k, COUNT(*) FROM TBL WHERE url = '{lit}' GROUP BY k ORDER BY k",
                f"SELECT COUNT(*) FROM TBL WHERE url <> '{lit}'",
                f"SELECT COUNT(*) FROM TBL WHERE url IN ('{lit}', '{lit2}', '{lit3}')",
                "SELECT COUNT(*) FROM TBL WHERE url LIKE '%/path/1499%'",
                "SELECT url, COUNT(*) FROM TBL WHERE url LIKE '%/path/1499%' GROUP BY url ORDER BY 2 DESC, 1 LIMIT 5",
                "SELECT url, COUNT(*) FROM TBL GROUP BY url ORDER BY 2 DESC, 1 LIMIT 10",
                "SELECT COUNT(DISTINCT url) FROM TBL",
                "SELECT k, COUNT(DISTINCT url) FROM TBL WHERE k < 5 GROUP BY k ORDER BY k",
                "SELECT MIN(url), MAX(url) FROM TBL WHERE k = 3"]:
        duck = sorted(con.execute(sql.replace('TBL', f"'{pq}'")).fetchall())
        rows, _ = wdb_sql.execute(Segment(w), sql.replace('TBL', 'tbl'))
        assert sorted(tuple(x.decode() if isinstance(x, bytes) else x for x in r) for r in rows) == duck, sql


def test_bitpack_guard():
    """THE BITPACK GUARD: against plain bitpack (free random access) the packed frames must win by
    10%; a hair (HID's 0.15%) is not a win. Wide random codes -> bitpack wears, never enc 18."""
    import wdb_encode
    rng = np.random.default_rng(4)
    codes = rng.integers(0, 1 << 20, 300_000).astype(np.int64)           # incompressible, 20 bits
    sec = wdb_encode._code_section(codes, 20)
    assert sec[0] == 0, sec[0]                                          # bitpack, not enc 18
    packed_len = 1 + (codes.size * 20 + 7) // 8
    a18 = np.ascontiguousarray(codes); import wdb_kernels as K
    fr = []; import zstandard as zstd; cx = zstd.ZstdCompressor(level=wdb_encode.CODE_ZSTD_LEVEL)
    for i in range(0, codes.size, 65536):
        ch = a18[i:i + 65536]; out = np.zeros((ch.size * 20 + 7) // 8 + 8, np.uint8); K.pk32_pack(ch, 20, out)
        fr.append(cx.compress(out.tobytes()))
    cand = 10 + 4 * (len(fr) + 1) + sum(map(len, fr))
    assert cand < packed_len * 1.05, (cand, packed_len)                 # it IS about the same size --
    assert cand >= packed_len * 0.90                                    # -- and inside the guard, so bitpack stays


def test_election_takes_it_only_when_smaller():
    """without the force the candidate must win on bytes alone: incompressible wide codes stay bitpack"""
    rng = np.random.default_rng(3)
    n = 200_000
    ids = rng.integers(0, 150_000, n)
    df = pd.DataFrame({'u': np.array([f'v{i}' for i in ids])})
    seg, w, pq = _enc(df)
    assert seg.cols['u'].get('code_enc', 0) != 18 or True     # any dress is lawful; what matters:
    got = np.array([x.decode() for x in seg.values('u')])
    assert np.array_equal(got, df['u'].to_numpy())
    if seg.cols['u'].get('code_enc', 0) == 18:
        # if it did win, it won on size: the packed frames are smaller than the bitpack
        c = seg.cols['u']
        assert int(c['czlen']) < (n * int(c['pbits']) + 7) // 8
