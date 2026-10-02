"""RUN_WIDE (2026-10-02): a latency-bound kernel runs on every vCPU for one call -- the kernel sees the
wide thread count, the caller's count (the engine's cap) is restored after, also when the kernel raises;
pass 2 of the distinct-count scatter gives the same counts at any width, equal to a plain count of
distinct (key, target) pairs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import numba
from numba import njit
import wdb_kernels as K


@njit
def _seen_threads(x):
    return numba.get_num_threads() + 0 * x


def _boom(x):
    raise ValueError('kernel failed')


def test_threads_raised_for_the_call_and_restored():
    top = int(numba.config.NUMBA_NUM_THREADS)
    capped = max(1, top // 2)                                  # the caller's cap, below every vCPU
    old = numba.get_num_threads()
    try:
        numba.set_num_threads(capped)
        assert int(K.run_wide(_seen_threads, 0)) == K.wide_threads() == top
        assert numba.get_num_threads() == capped
        try:
            K.run_wide(_boom, 0)
        except ValueError:
            pass
        assert numba.get_num_threads() == capped              # restored even when the kernel raises
        K._WIDE[0] = 1
        try:
            assert int(K.run_wide(_seen_threads, 0)) == 1      # WDB_WIDE_THREADS pins it
        finally:
            K._WIDE[0] = 0
        assert numba.get_num_threads() == capped
    finally:
        numba.set_num_threads(old)


def test_pass2_counts_equal_at_any_width():
    rng = np.random.default_rng(91)
    n = 2_000_000
    tgt = rng.integers(0, 300_000, n).astype(np.uint32)
    grp = rng.integers(0, 500, n).astype(np.uint16)
    VT = int(tgt.max()) + 1; SH = np.int64(max(1, VT.bit_length() - 12)); VR = np.int64(int(grp.max()) + 1)
    ku, kr, offs = K.gd_pass1(tgt, grp, SH, np.int64(8))
    pair = np.unique(grp.astype(np.int64) * VT + tgt.astype(np.int64))
    ref = np.bincount(pair // VT, minlength=int(VR))
    outs = []
    for w in (1, 3, 0):
        K._WIDE[0] = w
        try:
            outs.append(np.asarray(K.gd_pass2_count(ku, kr, offs, SH, VR)))
        finally:
            K._WIDE[0] = 0
    for o in outs:
        assert np.array_equal(o, ref)
