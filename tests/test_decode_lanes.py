"""THE DECODES WITHOUT PYTHON PER FRAME (2026-10-21, the Q24-Q26 board's last 2x):
  - enc 3 / enc 18 full decodes in compiled lanes calling libzstd (wdb_kernels.z3_frames / z18_frames) equal the
    thread-pool decodes code for code (WDB_ZLANES / wdb_engine._ZLANES = the switch);
  - an enc-1 stream read into its reserved array equals the one-shot decompress, and its prefix (codes_head, read
    on window after window) equals the whole decode's first rows;
  - the enc-3 and enc-20 low counts through each lane's board (wdb_kernels.z3_low_count / e20_low_count) equal a
    count of the decoded codes."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import wdb_encode, wdb_engine
from wdb_engine import Segment

TMP = tempfile.gettempdir()


def _enc(df, **env):
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        t = uuid.uuid4().hex[:8]; pq = f'{TMP}/dl_{t}.parquet'; w = f'{TMP}/dl_{t}.wdb'
        df.to_parquet(pq, index=False); wdb_encode.encode(pq, w)
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return Segment(w)


def _both(seg, col):
    old = wdb_engine._ZLANES[0]
    try:
        wdb_engine._ZLANES[0] = True; seg._codes.pop(col, None)
        a = np.asarray(seg._raw_codes(col)).copy()
        wdb_engine._ZLANES[0] = False; seg._codes.pop(col, None)
        b = np.asarray(seg._raw_codes(col)).copy()
    finally:
        wdb_engine._ZLANES[0] = old; seg._codes.pop(col, None)
    return a, b


def test_enc18_lanes_equal_the_pool():
    import test_packedframes as PF
    seg = PF._fixture()[0]
    assert seg.cols['url']['code_enc'] == 18
    a, b = _both(seg, 'url')
    assert a.dtype == b.dtype and np.array_equal(a, b)


def test_enc3_lanes_equal_the_pool():
    rng = np.random.default_rng(3)
    n = 1_200_000                                   # three 512K-row frames, the last one short
    x = (np.cumsum(rng.integers(0, 2, n)) // 50 + rng.integers(0, 4, n)).astype(np.int64)
    seg = _enc(pd.DataFrame({'x': x}), WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0')
    assert seg.cols['x'].get('code_enc') == 3, seg.cols['x'].get('code_enc')
    a, b = _both(seg, 'x')
    assert a.dtype == b.dtype and np.array_equal(a, b)
    assert np.array_equal(seg.values('x'), x)
    for T in (1, 5, 40):                            # the low count through each lane's board (z3_low_count)
        seg.drop_derived()
        lc = np.asarray(seg.low_counts('x', T))
        aa = a.astype(np.int64)
        assert np.array_equal(lc, np.bincount(aa[aa < T], minlength=T)), T


def test_enc1_stream_and_prefix():
    n = 1_000_000
    x = ((np.arange(n) * 5) % 13).astype(np.int64)  # a period zstd crushes and runs cannot: the stream dress
    seg = _enc(pd.DataFrame({'x': x}), WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0')
    assert seg.cols['x'].get('code_enc') == 1, seg.cols['x'].get('code_enc')
    whole = np.asarray(seg._raw_codes('x')).copy(); seg.drop_derived()
    for hi in (1000, 262_144, 400_000):             # the walk's windows: read on, never re-read (to half)
        pre = np.asarray(seg.codes_head('x', hi))
        assert np.array_equal(pre[:hi], whole[:hi]), hi
    rest = np.asarray(seg._raw_codes('x'))          # the whole decode reads on from the open prefix
    assert np.array_equal(rest, whole)
    seg.drop_derived()
    assert np.array_equal(seg.values('x'), x)


def test_enc20_low_count_equals_the_decoded_count():
    import test_hash_enc as HE
    seg = HE._seg()
    assert seg.cols['h']['code_enc'] == 20
    for T in (1, 7, 300):
        seg.drop_derived()
        lc = np.asarray(seg.low_counts('h', T))
        cc = np.asarray(seg._raw_codes('h')).astype(np.int64)
        assert np.array_equal(lc, np.bincount(cc[cc < T], minlength=T)), T
