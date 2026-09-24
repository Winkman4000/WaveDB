"""THE READ MATCHED TO THE QUESTION (2026-09-24): a chunked integer dictionary (mode 2) read by
runs -- neighbouring chunks share one pread, each run inflates frame by frame and sums every chunk
with one reshaped cumsum. Point reads (fetch, _dict_ints_at) and the full read must give the same
values at any chunk size, across chunk edges, with a partial last chunk, and however the chunks
fall into runs."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode
from wdb_engine import Segment

TMP = tempfile.gettempdir()
_FIX = None


def _encode(pq, out, ch):
    old = {k: os.environ.get(k) for k in ('WDB_I2CHUNK', 'WDB_I2CHUNK_MIN')}
    os.environ['WDB_I2CHUNK'] = str(ch); os.environ['WDB_I2CHUNK_MIN'] = '1000'
    try:
        wdb_encode.encode(pq, out)
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    rng = np.random.default_rng(7)
    uniq = np.unique(rng.integers(-(1 << 62), 1 << 62, 90721, dtype=np.int64))   # past NUM_THRESHOLD (50000): mode 2; a partial last chunk at every size
    n = 200000
    x = uniq[rng.integers(0, uniq.size, n)]
    df = pd.DataFrame({'x': x, 'g': rng.integers(0, 7, n).astype(np.int64)})
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/i2_{t}.parquet'
    df.to_parquet(pq, index=False)
    files = {}
    for ch in (1000, 8192, 1 << 19):
        files[ch] = f'{TMP}/i2_{ch}_{t}.wdb'
        _encode(pq, files[ch], ch)
    _FIX = (files, np.unique(x), df)
    return _FIX


def _col(seg):
    c = seg.cols['x']
    assert c['mode'] == 2 and c.get('i2ch') is not None, ('fixture is not a chunked integer dictionary', c['mode'])
    return c


def test_full_read_every_chunk_size():
    files, uniq, _ = _fixture()
    for ch, f in files.items():
        seg = Segment(f); c = _col(seg)
        assert int(c['i2ch']) == ch
        v = np.asarray(seg._dict_ints(c))
        assert np.array_equal(v[:uniq.size], uniq), ch


def test_point_reads_across_edges():
    files, uniq, _ = _fixture()
    rng = np.random.default_rng(3)
    for ch, f in files.items():
        seg = Segment(f); c = _col(seg)
        V0 = uniq.size
        edges = np.array(sorted({0, V0 - 1} | {min(V0 - 1, k * ch + d) for k in range(V0 // ch + 1) for d in (-1, 0, 1) if 0 <= k * ch + d}))
        for idx in (np.array([5]), np.array([V0 - 1]), edges, rng.integers(0, V0, 3000)):
            got = seg._dict_ints_at(c, idx)
            assert np.array_equal(got, uniq[idx]), (ch, idx[:5])
        for code in rng.integers(0, V0, 50):
            assert seg.fetch('x', int(code)) == int(uniq[code]), (ch, code)


def test_runs_however_they_fall():
    """a tiny run cap forces many runs (the pool); a huge cap forces one run of every chunk"""
    files, uniq, _ = _fixture()
    f = files[1000]
    for cap in (1, 5000, 1 << 30):
        prev = Segment._I2_RUN; Segment._I2_RUN = cap
        try:
            seg = Segment(f); c = _col(seg)
            assert np.array_equal(np.asarray(seg._dict_ints(c))[:uniq.size], uniq), cap
            seg2 = Segment(f); c2 = _col(seg2)
            idx = np.arange(0, uniq.size, 7)
            assert np.array_equal(seg2._dict_ints_at(c2, idx), uniq[idx]), cap
        finally:
            Segment._I2_RUN = prev


def test_point_then_full_then_point():
    """cached chunks from point reads never leak into the full read, and the spine answers after"""
    files, uniq, _ = _fixture()
    seg = Segment(files[1000]); c = _col(seg)
    a = seg._dict_ints_at(c, np.array([3, 2500, 17000]))
    assert np.array_equal(a, uniq[[3, 2500, 17000]])
    assert len(c['i2chunks']) == 3
    v = np.asarray(seg._dict_ints(c))
    assert np.array_equal(v[:uniq.size], uniq) and len(c['i2chunks']) == 0
    assert np.array_equal(seg._dict_ints_at(c, np.array([40000, 1])), uniq[[40000, 1]])
