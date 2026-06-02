"""FD-reference codec (stage 3, standalone): Y stored as a reference into X's pool.
Lossless IFF X->Y is exact. Pure arrays — no .wdb format involved yet."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import wdb_fdcodec as fc

def _roundtrip_ok(x_codes, y_values):
    enc = fc.fd_encode(x_codes, y_values)
    dec = fc.fd_decode(x_codes, enc)
    return fc._equal(dec, np.asarray(y_values)), enc

def test_int_dependent_true_fd():
    # x in 0..99, y = x % 4  -> exact FD
    x = np.random.default_rng(0).integers(0, 100, 5000)
    y = (x % 4).astype(np.int64)
    ok, enc = _roundtrip_ok(x, y)
    assert ok and len(enc) == 100          # map has one entry per distinct x

def test_string_dependent_true_fd():
    x = np.random.default_rng(1).integers(0, 50, 4000)
    brands = np.array([f'brand_{i%7}'.encode() for i in range(50)], dtype=object)
    y = brands[x]                           # y determined by x
    ok, enc = _roundtrip_ok(x, y)
    assert ok

def test_float_dependent_true_fd():
    x = np.random.default_rng(2).integers(0, 30, 3000)
    table = (np.arange(30) * 1.5).astype(np.float64)
    y = table[x]
    ok, _ = _roundtrip_ok(x, y)
    assert ok

def test_datetime_dependent_true_fd():
    x = np.random.default_rng(3).integers(0, 20, 2000)
    base = np.datetime64('2020-01-01')
    table = base + np.arange(20).astype('timedelta64[D]')
    y = table[x]
    ok, _ = _roundtrip_ok(x, y)
    assert ok

def test_null_in_x_code():
    # X has a reserved null code (highest). null maps to one y consistently -> still exact.
    rng = np.random.default_rng(4)
    x = rng.integers(0, 10, 3000)           # codes 0..9; let 9 be the "null" code
    y = (x % 3).astype(np.int64)            # x=9 -> y=0 consistently
    ok, _ = _roundtrip_ok(x, y)
    assert ok

def test_null_in_y_values():
    # Y has nulls (object array with None), still functionally determined by X
    x = np.random.default_rng(5).integers(0, 8, 2000)
    lut = np.array([1, None, 3, None, 5, 6, None, 8], dtype=object)
    y = lut[x]
    ok, _ = _roundtrip_ok(x, y)
    assert ok

def test_non_fd_is_detected_not_lossless():
    # same x maps to DIFFERENT y -> not an FD -> codec must report not lossless
    x = np.array([0,0,1,1,2,2])
    y = np.array([10,99,20,20,30,30])       # x=0 -> {10,99}: violates FD
    assert fc.is_lossless(x, y) is False

def test_is_lossless_true_for_real_fd():
    x = np.random.default_rng(6).integers(0, 40, 5000)
    y = (x * 7 % 11).astype(np.int64)
    assert fc.is_lossless(x, y) is True

def test_storage_is_smaller():
    # the whole point: map has Vx entries, not N
    x = np.random.default_rng(7).integers(0, 200, 100000)
    y = (x % 5).astype(np.int64)
    enc = fc.fd_encode(x, y)
    assert len(enc) == 200 and len(enc) < len(x) / 100   # 200 vs 100000
