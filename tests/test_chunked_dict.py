"""Block-segmented (chunked) front-coded dict: the dict is written as independent zstd
frames of CHUNK_DICT_VALS values, so a value decode only decompresses its own frame
instead of the whole 6M-value blob. Lossless and byte-identical answers vs the classic
monolithic dict; the only difference is per-frame decompression. Flag-gated (default off)."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode
from helpers import roundtrip, assert_lossless

def _frame():
    # > FC_THRESHOLD (50k) distinct so the column is front-coded (mode 1), with '' + dups
    distinct = [''] + [f"tok_{i:06d}_{i%9}" for i in range(70000)]
    rng = np.random.default_rng(1)
    extra = [distinct[i] for i in rng.integers(1, len(distinct), size=60000)]
    rows = distinct + extra + [''] * 40000
    rng.shuffle(rows)
    return pd.DataFrame({'s': rows})

def test_chunked_dict_lossless_and_multichunk():
    df = _frame()
    wdb_encode.CHUNK_DICT = True
    try:
        seg, pq = roundtrip(df, ['s'])
    finally:
        wdb_encode.CHUNK_DICT = False
    c = seg.cols['s']
    assert c['mode'] == 1 and c.get('chunked') is True
    assert len(c['chunk_czlen']) >= 4          # multi-chunk (V>50k, 16384/chunk)
    assert_lossless(seg, pq, 's')

def test_chunked_matches_monolithic():
    df = _frame()
    wdb_encode.CHUNK_DICT = True
    try:
        seg_c, _ = roundtrip(df, ['s'])
    finally:
        wdb_encode.CHUNK_DICT = False
    seg_m, _ = roundtrip(df, ['s'])
    assert seg_m.cols['s'].get('chunked') in (False, None)
    assert seg_c.dict_vals('s') == seg_m.dict_vals('s')
    V = seg_c.cols['s']['V']
    rng = np.random.default_rng(2)
    codes = list(range(0, 300)) + [int(x) for x in rng.integers(0, V, 120)]
    assert all(seg_c.fetch('s', cd) == seg_m.fetch('s', cd) for cd in codes)
