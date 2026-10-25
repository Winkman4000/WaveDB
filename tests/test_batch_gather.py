"""THE BATCH GATHER (2026-10-25): codes at rows of an enc-3 column touching many frames inflate in one zstd call on
its own threads, then the compiled per-frame gather. Forced on (the line at one frame) and off (the lanes), sorted
and unsorted rows, every column of the fixture's seven frames: the same codes, the same dtype."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np


def test_batch_gather_equals_the_lanes():
    import test_eq_e3_batch as F, wdb_engine as EN
    _db, _con, seg = F._fixture()
    N = int(seg.N)
    cols = [c for c in ('a', 'c', 's') if seg.cols[c].get('code_enc') == 3]
    assert cols, 'the fixture frames no column'
    rng = np.random.default_rng(7)
    sets = [np.arange(0, N, 97, dtype=np.int64), np.sort(rng.choice(N, 5000, replace=False)).astype(np.int64),
            np.array([N - 1], np.int64), rng.permutation(N)[:3000].astype(np.int64)]       # the last one unsorted
    line0, sg0 = EN._E3_BATCH_FRAMES, EN._SORTGATHER[0]
    ran = 0
    try:
        for col in cols:
            raw = np.asarray(seg._raw_codes_range(col, 0, N))
            for rows in sets:
                EN._E3_BATCH_FRAMES = 0
                seg._codes.pop(col, None); seg.__dict__.pop('_scan_codes', None)
                lanes = np.asarray(seg.codes_at(col, rows))
                EN._E3_BATCH_FRAMES = 1
                seg._codes.pop(col, None); seg.__dict__.pop('_scan_codes', None)
                batch = np.asarray(seg.codes_at(col, rows))
                assert batch.dtype == lanes.dtype, (col, batch.dtype, lanes.dtype)
                assert np.array_equal(batch, lanes), col
                assert np.array_equal(batch.astype(np.int64), raw[rows].astype(np.int64)), col
                ran += 1
    finally:
        EN._E3_BATCH_FRAMES, EN._SORTGATHER[0] = line0, sg0
    assert ran >= 8, ran
