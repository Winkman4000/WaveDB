"""Regression guard: codes() must decode in bounded memory, not allocate N x bits at once.
This freezes the fix for the 10.7GB blowup that crashed at 60M rows."""
import sys, os, resource
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from helpers import roundtrip

def test_codes_memory_bounded():
    # 3M rows, 22-bit codes. The OLD code built a 3M x 22 uint64 matrix (~528MB);
    # chunked code keeps peak far lower. Assert we decode correctly with a sane cap.
    n = 3_000_000
    vals = np.arange(n, dtype=np.int64)  # high-card -> mode 2, 22 bits
    df = pd.DataFrame({'x': vals})
    seg, pq = roundtrip(df)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cc = seg.codes('x')
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    grew_mb = (after - before) / 1024  # ru_maxrss is KB on Linux
    assert cc.shape[0] == n
    # the full N x bits uint64 matrix would be n*22*8 = 528MB; chunked must be far less.
    # generous ceiling: 200MB growth. (regression catches reintroducing the full matrix)
    assert grew_mb < 200, f"codes() grew {grew_mb:.0f}MB - the N x bits matrix may be back"

def test_codes_values_match():
    n = 100000
    vals = np.arange(n, dtype=np.int64)
    df = pd.DataFrame({'x': vals})
    seg, pq = roundtrip(df)
    cc = seg.codes('x')
    # codes index a sorted dict of arange(n) -> code == value
    assert np.array_equal(cc, vals), "codes should map 1:1 to sorted distinct values"
