"""Standalone tests for the mode-4 sequence/affine codec (wdb_seqcodec), isolated from the
engine. Two separable properties:
  (1) MECHANISM is lossless on ANY int64 input (build->decode == input), incl. inputs the
      gate would decline (shuffled/random) and overflow boundaries.
  (2) GATE (encode) fires only when beneficial: clean/strided/gapped sequences -> blob;
      shuffled/random/tiny -> None."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import wdb_seqcodec as sc

IMIN, IMAX = np.iinfo(np.int64).min, np.iinfo(np.int64).max
rng = np.random.default_rng(4)

def _mech(col):
    """build->decode must equal col exactly."""
    col = np.asarray(col, dtype=np.int64)
    p = sc.params(col)
    if p is None:                      # n<2: nothing to encode, trivially fine
        return
    assert np.array_equal(sc.decode(sc.build(p)), col), f"mechanism lossy on n={len(col)}"
    assert sc.is_lossless(col)

# ---------- (1) mechanism losslessness across the input space ----------
def test_mech_clean():        _mech(1_000_000 + np.arange(100000))
def test_mech_strided():      _mech(500 + np.arange(80000) * 7)
def test_mech_negative_base(): _mech(-10**9 + np.arange(50000))
def test_mech_gaps():
    keep = rng.random(120000) > 0.01
    _mech(np.flatnonzero(keep)[:100000])
def test_mech_shuffled():      a = np.arange(50000); rng.shuffle(a); _mech(a)        # gate declines, mech lossless
def test_mech_random():        _mech(rng.integers(-10**15, 10**15, size=50000))      # gate declines, mech lossless
def test_mech_extremes():      _mech(np.array([IMIN, IMAX, 0, -1, 1, IMIN+1, IMAX-1]))
def test_mech_huge_gap_overflow():
    a = np.arange(40000, dtype=np.int64); a[20000:] += (IMAX - 50000)               # gap delta overflows int64
    _mech(np.unique(a))
def test_mech_two_rows():      _mech(np.array([IMIN, IMAX], dtype=np.int64))
def test_mech_three_rows():    _mech(np.array([10, 11, 12], dtype=np.int64))
def test_mech_all_same():      _mech(np.full(1000, 42, dtype=np.int64))              # stride 0
def test_mech_datetime_epochs():
    base = np.datetime64('2020-01-01T00:00:00','s').astype('int64')
    _mech(base + np.arange(60000, dtype=np.int64))                                   # monotonic timestamps

def test_mech_fuzz_200():
    r = np.random.default_rng(99)
    for _ in range(200):
        n = int(r.integers(2, 6000))
        shape = r.integers(0, 5)
        if shape == 0:   a = int(r.integers(-10**12,10**12)) + np.arange(n)
        elif shape == 1: a = int(r.integers(-10**6,10**6)) + np.arange(n)*int(r.integers(1,1000))
        elif shape == 2: a = np.cumsum(r.integers(1,5,size=n)).astype(np.int64)
        elif shape == 3: a = r.integers(-10**14,10**14,size=n).astype(np.int64)
        else:            a = np.arange(n); r.shuffle(a)
        _mech(a.astype(np.int64))

# ---------- (2) gate: fire when beneficial, decline otherwise ----------
def test_gate_fires_clean():
    blob = sc.encode(1_000_000 + np.arange(1_000_000))
    assert blob is not None and len(blob) <= 64, f"clean should be tiny, got {None if blob is None else len(blob)}"
    assert np.array_equal(sc.decode(blob), 1_000_000 + np.arange(1_000_000))

def test_gate_fires_strided():
    col = 42 + np.arange(200000) * 13
    blob = sc.encode(col)
    assert blob is not None and len(blob) <= 64
    assert np.array_equal(sc.decode(blob), col)

def test_gate_fires_gaps_smaller_than_raw():
    keep = rng.random(1_020_000) > 0.01
    col = np.flatnonzero(keep)[:1_000_000].astype(np.int64)
    blob = sc.encode(col)
    assert blob is not None
    assert len(blob) < col.size * 8 * 0.1, f"gaps blob should be <10% of raw, got {len(blob)} vs {col.size*8}"
    assert np.array_equal(sc.decode(blob), col)

def test_gate_declines_shuffled():
    a = np.arange(100000); rng.shuffle(a)
    assert sc.encode(a) is None

def test_gate_declines_random():
    assert sc.encode(rng.integers(0, 2**62, size=100000)) is None

def test_gate_declines_tiny():
    assert sc.encode(np.array([5, 6])) is None          # n < MIN_ROWS
    assert sc.encode(np.array([5])) is None

def test_clean_blob_size_matches_measurement():
    blob = sc.encode(1_000_000 + np.arange(1_000_000))
    assert len(blob) == 32, f"clean affine blob expected 32 bytes (header only), got {len(blob)}"

def test_encode_deterministic():
    col = np.flatnonzero(rng.random(120000) > 0.02)[:100000].astype(np.int64)
    assert sc.encode(col) == sc.encode(col), "encode not byte-deterministic"
