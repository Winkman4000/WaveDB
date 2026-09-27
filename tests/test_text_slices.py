"""THE TEXT IN SLICES (2026-09-27): _arrow_string_prep_sliced must give exactly the whole-column
prep -- the same sorted dictionary bytes, the same codes (cluster order applied), the same null bin,
V, bits and mode -- for many row groups, nulls, a row group of only nulls, empty strings, non-ASCII
text and binary columns; and the column's blob must be byte-identical either way."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
import zstandard as zstd
import wdb_encode as E

TMP = tempfile.gettempdir()


def _parquet():
    rng = np.random.default_rng(27)
    n = 60000
    words = np.array(['', 'a', 'ab', 'abc', 'http://x.ru/1', 'http://x.ru/2', 'жёлтый', 'zzz', 'Ω≈ç', 'b'] +
                     ['v%05d' % i for i in range(3000)], dtype=object)
    s = list(words[rng.integers(0, len(words), n)])
    for i in rng.integers(0, n, 3000):
        s[i] = None
    for i in range(20000, 24000):                     # rows 20000-23999: one whole row group of nulls
        s[i] = None
    b = [None if v is None else v.encode('utf-8') for v in s]
    t = pa.table({'s': pa.array(s, pa.string()), 'b': pa.array(b, pa.binary()),
                  'nn': pa.array([('k%d' % (i % 777)) for i in range(n)], pa.string())})
    p = os.path.join(TMP, 'wdb_slices_%s.parquet' % uuid.uuid4().hex[:8])
    pq.write_table(t, p, row_group_size=4000)       # 15 row groups
    return p, n


def _whole(p, nm, perm):
    col = pq.read_table(p, columns=[nm]).column(0)
    if pa.types.is_string(col.type):
        col = pa.compute.cast(col, pa.large_string())
    return E._arrow_string_prep(nm, [col], perm=perm)


def _same(a, b):
    for k in ('nm', 'dtype', 'has_null', 'V', 'bits', 'mode', 'aux'):
        assert a[k] == b[k], (k, a[k], b[k])
    assert a['valb'] == b['valb']
    assert a['codes'].dtype == b['codes'].dtype
    assert np.array_equal(a['codes'], b['codes'])


def test_slices_equal_whole_column():
    p, n = _parquet()
    try:
        perm = np.random.default_rng(3).permutation(n).astype(np.int32)
        for nm in ('s', 'b', 'nn'):
            for K, W in ((2, 1), (4, 2), (15, 4), (16, 3)):
                sl = E._arrow_string_prep_sliced(nm, p, perm=perm, slices=K, threads=W)
                assert sl is not None, (nm, K, W)
                _same(sl, _whole(p, nm, perm))
            zc = zstd.ZstdCompressor(level=E.ZSTD_LEVEL)
            b1 = E._serialize_column(E._arrow_string_prep_sliced(nm, p, perm=perm), zc)[0]
            b2 = E._serialize_column(_whole(p, nm, perm), zc)[0]
            assert b1 == b2, nm
    finally:
        os.remove(p)


def test_bytevals_front_code_identical():
    """_ByteVals (the dictionary as one buffer) must front-code to exactly the bytes the list of Python
    bytes objects gave, and read back value for value"""
    vals = sorted(set(['', 'a', 'ab', 'abc', 'ж', 'жёлтый'] + ['http://x.ru/%d/%s' % (i, 'q' * (i % 7)) for i in range(5000)]))
    cases = [vals, [''], ['x'], [], [v for v in vals if v]]
    for vs in cases:
        bl = [v.encode('utf-8') for v in vs]
        arr = pa.array(bl, pa.large_binary())
        bv = E._ByteVals(arr.slice(0, len(bl)))
        assert len(bv) == len(bl) and list(bv) == bl and bv == bl
        assert E._front_code(bv)[0] == E._front_code(bl)[0]
        assert np.array_equal(E._front_code(bv)[1], E._front_code(bl)[1])
        if bl:
            assert bv[-1] == bl[-1] and bv[0] == bl[0]
    big = pa.array([v.encode() for v in vals], pa.large_binary()).slice(100, 900)
    bv = E._ByteVals(big)
    assert list(bv) == [v.encode() for v in vals[100:1000]]


def test_slices_decline():
    p, n = _parquet()
    try:
        assert E._arrow_string_prep_sliced('s', p, slices=1) is None      # one run: the whole road
        t = pa.table({'x': pa.array(np.arange(100))})
        q = p + '.int.parquet'; pq.write_table(t, q, row_group_size=10)
        assert E._arrow_string_prep_sliced('x', q) is None                # not text
        os.remove(q)
    finally:
        os.remove(p)
