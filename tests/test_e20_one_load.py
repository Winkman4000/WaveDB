"""THE ONE LOAD (2026-10-27): enc 20's row read (wdb_kernels._e20_load) is one unaligned 8-byte load. It must equal
the 8 single-byte loads glued little-endian at every byte offset (aligned or not, up to the buffer's last 8 bytes), on
a buffer whose data starts off an 8-byte boundary too. The enc-20 kernels' own answers are covered by test_eq_scans,
test_eq_pattern, test_e20_read_sum and test_decode_lanes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import wdb_kernels as WK
from wdb_kernels import njit


@njit(cache=False)
def _both(buf, idx, got, ref):
    for j in range(idx.size):
        b = idx[j]
        got[j] = WK._e20_load(buf, b)
        w = np.uint64(0)
        for k in range(8):
            w |= np.uint64(buf[b + k]) << np.uint64(8 * k)
        ref[j] = w


def test_e20_one_load_equals_the_byte_glue():
    rng = np.random.default_rng(20)
    raw = rng.integers(0, 256, 1 << 16).astype(np.uint8)
    for off in range(8):                          # the array's data starting on every byte of a word
        buf = raw[off:]
        idx = np.r_[np.arange(0, 64), rng.integers(0, buf.size - 8, 20000), np.arange(buf.size - 64, buf.size - 7)]
        idx = idx.astype(np.int64)
        got = np.empty(idx.size, np.uint64); ref = np.empty(idx.size, np.uint64)
        _both(buf, idx, got, ref)
        assert np.array_equal(got, ref), (off, int(np.count_nonzero(got != ref)))
