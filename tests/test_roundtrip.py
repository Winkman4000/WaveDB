"""Lossless round-trip across every dtype, mode, and edge case."""
import numpy as np, pandas as pd
from helpers import roundtrip, assert_lossless

def test_int_lowcard_mode0():
    df = pd.DataFrame({'x': np.array([1,2,3,2,1,3,3,2]*100, dtype=np.int64)})
    seg, pq = roundtrip(df)
    assert assert_lossless(seg, pq, 'x') == 0

def test_int_sequential_mode4():
    # clean sequential int -> mode 4 (affine codec); lossless
    df = pd.DataFrame({'x': np.arange(60000, dtype=np.int64)})
    seg, pq = roundtrip(df)
    assert assert_lossless(seg, pq, 'x') == 4, "clean sequence should be mode 4"

def test_int_highcard_nonsequential_mode2():
    # sparse high-card ints (the orderkey-like case): mode 2, base value != 0
    rng = np.random.default_rng(0)
    vals = np.unique(rng.integers(1, 10_000_000, size=80000)).astype(np.int64)
    df = pd.DataFrame({'x': vals})
    seg, pq = roundtrip(df)
    m = assert_lossless(seg, pq, 'x')
    assert m == 2 and int(seg.values('x')[0]) == int(vals.min())

def test_int_negative():
    df = pd.DataFrame({'x': np.array([-5,-1,0,3,-5,7,-100,3]*50, dtype=np.int64)})
    seg, pq = roundtrip(df)
    assert assert_lossless(seg, pq, 'x') == 0

def test_int_nulls_mode0():
    s = pd.array([1,2,None,4,None,2,1], dtype='Int64')
    df = pd.DataFrame({'x': pd.concat([pd.Series(s)]*100, ignore_index=True)})
    seg, pq = roundtrip(df)
    assert assert_lossless(seg, pq, 'x') == 0  # nulls force mode 0

def test_float_basic():
    df = pd.DataFrame({'x': np.array([1.5,2.25,3.125,1.5,0.0,-7.5]*100, dtype=np.float64)})
    seg, pq = roundtrip(df)
    assert assert_lossless(seg, pq, 'x') == 0

def test_float_nulls():
    df = pd.DataFrame({'x': np.array([1.5,np.nan,3.0,np.nan,2.0]*100, dtype=np.float64)})
    seg, pq = roundtrip(df)
    assert_lossless(seg, pq, 'x')

def test_string_lowcard_mode0():
    df = pd.DataFrame({'x': (['apple','banana','cherry']*200)})
    seg, pq = roundtrip(df)
    assert assert_lossless(seg, pq, 'x') == 0

def test_string_highcard_mode1():
    # >50k distinct strings WITH repetition -> front-coded mode 1 (dict beats inline)
    df = pd.DataFrame({'x': [f'item_{i:08d}' for i in range(60000)] * 4})
    seg, pq = roundtrip(df)
    assert assert_lossless(seg, pq, 'x') == 1, "high-card repetitive string should be mode 1"

def test_string_unique_mode5():
    # near-unique high-card strings -> inline mode 5 (dictionary pointers are dead weight)
    df = pd.DataFrame({'x': [f'item_{i:08d}' for i in range(60000)]})
    seg, pq = roundtrip(df)
    assert assert_lossless(seg, pq, 'x') == 5, "unique string column should be mode 5"

def test_string_unicode():
    df = pd.DataFrame({'x': (['café','日本語','emoji😀','naïve','']*100)})
    seg, pq = roundtrip(df)
    assert_lossless(seg, pq, 'x')

def test_string_nulls():
    df = pd.DataFrame({'x': (['a','b',None,'d',None]*100)})
    seg, pq = roundtrip(df)
    assert_lossless(seg, pq, 'x')

def test_datetime_lowcard():
    base = np.datetime64('2020-01-01')
    df = pd.DataFrame({'x': (base + np.array([0,1,2,0,5]*100, dtype='timedelta64[D]'))})
    seg, pq = roundtrip(df)
    assert_lossless(seg, pq, 'x')

def test_datetime_sequential_mode4():
    base = np.datetime64('2000-01-01T00:00:00')
    df = pd.DataFrame({'x': base + np.arange(60000, dtype='timedelta64[s]')})
    seg, pq = roundtrip(df)
    assert assert_lossless(seg, pq, 'x') == 4   # monotonic timestamps -> affine codec

def test_datetime_highcard_nonseq_mode2():
    # high-card NON-monotonic timestamps (random gaps) -> mode 2; preserves datetime delta-dict coverage
    rng = np.random.default_rng(5)
    base = np.datetime64('2000-01-01T00:00:00')
    secs = np.unique(rng.integers(0, 50_000_000, 70000))
    df = pd.DataFrame({'x': (base + secs.astype('timedelta64[s]'))})
    seg, pq = roundtrip(df)
    assert assert_lossless(seg, pq, 'x') == 2

def test_single_distinct_value():
    df = pd.DataFrame({'x': np.full(500, 42, dtype=np.int64)})
    seg, pq = roundtrip(df)
    assert_lossless(seg, pq, 'x')

def test_all_null():
    df = pd.DataFrame({'x': pd.array([None]*200, dtype='Int64')})
    seg, pq = roundtrip(df)
    assert_lossless(seg, pq, 'x')
