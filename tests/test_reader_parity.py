"""Arrow vs DuckDB reader must produce BYTE-IDENTICAL encoder output, across all
dtypes including nulls. This guards the Arrow migration: the two backends are
interchangeable on the read path."""
import sys, os, uuid, tempfile, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode

TMP = tempfile.gettempdir()

def _mk(df):
    pq = os.path.join(TMP, f'rp_{uuid.uuid4().hex[:8]}.parquet'); df.to_parquet(pq, index=False)
    return pq

def _md5(path):
    a = path + '.arrow.wdb'; d = path + '.duck.wdb'
    wdb_encode.encode(path, a, reader='arrow')
    wdb_encode.encode(path, d, reader='duckdb')
    return hashlib.md5(open(a,'rb').read()).hexdigest(), hashlib.md5(open(d,'rb').read()).hexdigest()

def test_parity_all_dtypes_with_nulls():
    df = pd.DataFrame({
        'int_nn':   pd.array([1,2,3,2,1]*40, dtype='int64'),
        'int_null': pd.array([1,None,3,None,5]*40, dtype='Int64'),
        'flt_null': np.array([1.5,np.nan,3.5,np.nan,2.0]*40),
        'str_null': (['a',None,'c',None,'café']*40),
        'dt_null':  pd.to_datetime(['2020-01-01',None,'2020-01-03',None,'2021-06-30']*40),
    })
    ha, hd = _md5(_mk(df))
    assert ha == hd, f"arrow {ha} != duckdb {hd}"

def test_parity_highcard_modes():
    # exercise mode 1 (front-coded strings) and mode 2 (delta ints): >50k distinct
    df = pd.DataFrame({
        'bigint': np.arange(60000, dtype=np.int64),
        'bigstr': [f'item_{i:08d}' for i in range(60000)],
    })
    ha, hd = _md5(_mk(df))
    assert ha == hd, f"arrow {ha} != duckdb {hd}"

def test_parity_negatives_and_single_value():
    df = pd.DataFrame({
        'neg': np.array([-5,-1,0,3,-100]*100, dtype=np.int64),
        'one': np.full(500, 42, dtype=np.int64),
    })
    ha, hd = _md5(_mk(df))
    assert ha == hd, f"arrow {ha} != duckdb {hd}"
