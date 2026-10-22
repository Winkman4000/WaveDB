"""THE DIFFERENTIATION PATH on the back-reference encoding (Jackson, 2026-10-21): '= k' on an enc-20 column at
least 23 bits wide searches the raw bytes for k's literal pattern and walks only the flagged blocks
(wdb_wherescan._eq_e20_pattern, the first entry of enc 20 in the equality jump table). Its rows must be exactly the
decode-and-compare's: rare codes, spans that cut blocks, a hot value (most blocks flagged -> the full walk), <>
(not this path), and the switch off."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode, wdb_wherescan as WS
from wdb_engine import Segment

TMP = tempfile.gettempdir()
_SEG = None


def _seg():
    global _SEG
    if _SEG is not None:
        return _SEG
    rng = np.random.default_rng(23)
    V = 6_000_000                                       # codes need 23 bits
    n = 6_600_000
    pool = rng.integers(-(1 << 62), 1 << 62, V, dtype=np.int64)
    x = np.concatenate([pool, pool[rng.integers(0, V, n - V)]])
    x = x[rng.permutation(n)]
    near = rng.random(n) < 0.3                          # local repeats: back-references
    back = np.maximum(0, np.arange(n) - rng.integers(1, 40, n))
    x[near] = x[back[near]]
    x[rng.random(n) < 0.01] = pool[7]                   # a hot value: in every block
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/eqp_{t}.parquet'; w = f'{TMP}/eqp_{t}.wdb'
    pd.DataFrame({'h': x}).to_parquet(pq, index=False)
    keep = {k: os.environ.get(k) for k in ('WDB_HASH_COLS', 'WDB_E20_BR', 'WDB_SEQ_NARROW_OK')}
    os.environ.update(WDB_HASH_COLS='h', WDB_E20_BR='4096', WDB_SEQ_NARROW_OK='0')
    try:
        wdb_encode.encode(pq, w)
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    _SEG = Segment(w)
    return _SEG


def _codes(seg):
    seg._codes.pop('h', None)
    cc = np.asarray(seg._raw_codes('h')).astype(np.int64).copy()
    seg.drop_derived(); seg._codes.pop('h', None)
    return cc


def test_rare_codes_take_the_path_and_match_the_decode():
    seg = _seg(); c = seg.cols['h']
    assert c['code_enc'] == 20 and int(c['e20bits']) >= 23, (c.get('code_enc'), c.get('e20bits'))
    cc = _codes(seg); N = cc.size
    cnt = np.bincount(cc)
    rare = [int(k) for k in np.flatnonzero(cnt == 1)[:3]] + [int(k) for k in np.flatnonzero(cnt == 3)[:2]]
    for code in rare:
        for lo, hi in ((0, N), (N // 3 + 5, N - 1000), (4096 * 7 + 11, 4096 * 7 + 11 + (1 << 16) + 77)):
            h0 = WS._E20PAT_HITS
            got = np.asarray(WS._scan_eq(seg, 'h', code, lo, hi))
            assert WS._E20PAT_HITS == h0 + 1, ('not the pattern path', code, lo, hi)
            want = np.nonzero(cc[lo:hi] == code)[0] + lo
            assert np.array_equal(got, want), (code, lo, hi, got[:5], want[:5])
            assert 'h' not in seg._codes


def test_hot_value_negate_and_switch_fall_to_the_walk():
    seg = _seg()
    cc = _codes(seg); N = cc.size
    hot = int(np.argmax(np.bincount(cc)))
    one = int(np.flatnonzero(np.bincount(cc) == 1)[0])
    for code, neg in ((hot, False), (one, True)):
        h0 = WS._E20PAT_HITS
        got = np.asarray(WS._scan_eq(seg, 'h', code, 0, N, negate=neg))
        assert WS._E20PAT_HITS == h0, (code, neg)       # declined: the full walk answered
        want = np.nonzero((cc != code) if neg else (cc == code))[0]
        assert np.array_equal(got, want), (code, neg)
    old = WS._E20PAT[0]
    try:
        WS._E20PAT[0] = False
        h0 = WS._E20PAT_HITS
        got = np.asarray(WS._scan_eq(seg, 'h', one, 0, N))
        assert WS._E20PAT_HITS == h0 and np.array_equal(got, np.nonzero(cc == one)[0])
    finally:
        WS._E20PAT[0] = old
