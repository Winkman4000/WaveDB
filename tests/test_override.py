"""Override sidecar (mutable layer, step 1a): per-segment row->new-value map for UPDATE
without rewriting the immutable .wdb. This step covers MATERIALIZATION (values()): the
override is scattered in after decode, dtype-preserving, and can carry a value the segment's
dictionary never had. Code-based query paths (GROUP BY / WHERE) come in step 1b.
A column with no override must behave exactly as before."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode, wdb_override
from wdb_engine import Segment

TMP = tempfile.gettempdir()
def _seg(df):
    base = os.path.join(TMP, f'ov_{uuid.uuid4().hex[:8]}')
    df.to_parquet(base + '.parquet', index=False)
    sp = base + '.wdb'; wdb_encode.encode(base + '.parquet', sp)
    return sp
def _dec(arr):
    return [b.decode() if isinstance(b,(bytes,bytearray)) else b for b in arr]

def test_no_override_unchanged():
    sp = _seg(pd.DataFrame({'x':[1,2,3]}))
    assert list(Segment(sp).values('x')) == [1,2,3]
    assert Segment(sp)._overrides('x') is None

def test_override_new_int_value_dtype_preserved():
    sp = _seg(pd.DataFrame({'x':[10,20,30,40,50]}))
    wdb_override.set_override(sp, 'x', [1,3], np.array([999, 40], dtype=np.int64))
    v = Segment(sp).values('x')
    assert list(v) == [10,999,30,40,50]
    assert v.dtype == np.int64                # not promoted to object

def test_override_new_string_value():
    sp = _seg(pd.DataFrame({'s':['a','b','c']}))
    wdb_override.set_override(sp, 's', [0], np.array(['ZZZ'.encode()], dtype=object))
    assert _dec(Segment(sp).values('s')) == ['ZZZ','b','c']

def test_override_float_value():
    sp = _seg(pd.DataFrame({'f':[1.5,2.5,3.5]}))
    wdb_override.set_override(sp, 'f', [2], np.array([9.25], dtype=np.float64))
    v = Segment(sp).values('f')
    assert list(v) == [1.5,2.5,9.25] and v.dtype == np.float64

def test_override_multiple_columns_independent():
    sp = _seg(pd.DataFrame({'x':[1,2,3], 's':['a','b','c']}))
    wdb_override.set_override(sp, 'x', [0], np.array([100], dtype=np.int64))
    wdb_override.set_override(sp, 's', [2], np.array(['Q'.encode()], dtype=object))
    seg = Segment(sp)
    assert list(seg.values('x')) == [100,2,3]
    assert _dec(seg.values('s')) == ['a','b','Q']

def test_override_latest_write_wins():
    sp = _seg(pd.DataFrame({'x':[1,2,3]}))
    wdb_override.set_override(sp, 'x', [1], np.array([20], dtype=np.int64))
    wdb_override.set_override(sp, 'x', [1], np.array([200], dtype=np.int64))  # overwrite same row
    assert list(Segment(sp).values('x')) == [1,200,3]
    assert wdb_override.override_count(sp) == 1

def test_override_count_helper():
    sp = _seg(pd.DataFrame({'x':list(range(10))}))
    wdb_override.set_override(sp, 'x', [1,2,3], np.array([10,20,30], dtype=np.int64))
    assert wdb_override.override_count(sp) == 3
