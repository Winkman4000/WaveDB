"""THE ONE-CODE SCANS (2026-10-21, Q19's board): wherescan's equality scan for a single code on the encodings that had
none -- plain bitpack (enc 0: wdb_kernels.bp0_eq_count / _fill), the back-reference (enc 20: e20_eq_count / _fill)
and the packed frames in compiled lanes (enc 18: z18_eq_count / _fill) -- give exactly the rows the decode-and-compare
gives, over the whole column and over spans that cut blocks, for = and <>, for a common code, a rare one and one no
row holds. WDB_BP0_EQ / wdb_wherescan._BP0EQ is the switch."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import wdb_encode, wdb_wherescan as WS
from wdb_engine import Segment

TMP = tempfile.gettempdir()


def _enc0():
    rng = np.random.default_rng(7)
    n = 1_200_000
    x = rng.integers(0, 300_000, n).astype(np.int64)           # no order, no repeats to crush: plain bitpack
    keep = {k: os.environ.get(k) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    os.environ.update(WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0')
    try:
        t = uuid.uuid4().hex[:8]; pq = f'{TMP}/eq0_{t}.parquet'; w = f'{TMP}/eq0_{t}.wdb'
        pd.DataFrame({'x': x}).to_parquet(pq, index=False); wdb_encode.encode(pq, w)
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return Segment(w), 'x'


def _check(seg, col):
    seg.drop_derived(); seg._codes.pop(col, None)
    cc = np.asarray(seg._raw_codes(col)).astype(np.int64).copy()
    seg.drop_derived(); seg._codes.pop(col, None)
    N = cc.size
    vals, cnt = np.unique(cc, return_counts=True)
    codes = [int(vals[np.argmax(cnt)]), int(vals[np.argmin(cnt)]), int(vals.max()) + 1]
    spans = [(0, N), (N // 3 + 5, N - 1000), (1, 1 + (1 << 16) + 77)]
    for code in codes:
        for lo, hi in spans:
            for neg in (False, True):
                seg._codes.pop(col, None)
                got = np.asarray(WS._scan_eq(seg, col, code, lo, hi, negate=neg))
                assert col not in seg._codes, 'the scan decoded the column'
                part = cc[lo:hi]
                want = np.nonzero(part != code if neg else part == code)[0] + lo
                assert np.array_equal(got, want), (col, code, lo, hi, neg, got.size, want.size)
                if not neg and want.size:
                    kp, kc = seg._scan_codes[col]
                    assert np.array_equal(kp, want) and (kc == code).all()


def test_enc0_one_code_scan():
    seg, col = _enc0()
    assert seg.cols[col].get('code_enc', 0) == 0 and 'boffs' not in seg.cols[col], seg.cols[col].get('code_enc')
    _check(seg, col)


def test_enc20_one_code_scan():
    import test_hash_enc as HE
    seg = HE._seg()
    assert seg.cols['h']['code_enc'] == 20
    _check(seg, 'h')


def test_enc18_one_code_scan_in_lanes():
    import test_packedframes as PF
    seg = PF._fixture()[0]
    assert seg.cols['url']['code_enc'] == 18
    c = seg.cols['url']
    assert WS._scan_eq18_lanes(seg, c, 3, 0, int(seg.N), False) is not None, 'the compiled lanes declined'
    _check(seg, 'url')


def test_switch_off_is_the_decode():
    seg, col = _enc0()
    old = WS._BP0EQ[0]
    try:
        WS._BP0EQ[0] = False
        cc = np.asarray(seg._raw_codes(col)).astype(np.int64)
        got = np.asarray(WS._scan_eq(seg, col, int(cc[17]), 0, cc.size))
        assert np.array_equal(got, np.nonzero(cc == int(cc[17]))[0])
    finally:
        WS._BP0EQ[0] = old
