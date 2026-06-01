"""Test helpers: synthetic data -> encode -> decode round-trip + comparison."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode
from wdb_engine import Segment

TMP = tempfile.gettempdir()
_files = []

def roundtrip(df, columns=None):
    """Write df->parquet, encode->.wdb, load Segment. Returns (seg, parquet_path)."""
    tag = uuid.uuid4().hex[:8]
    pq  = os.path.join(TMP, f'wt_{tag}.parquet')
    wdb = os.path.join(TMP, f'wt_{tag}.wdb')
    df.to_parquet(pq, index=False)
    wdb_encode.encode(pq, wdb, columns)
    _files.extend([pq, wdb])
    return Segment(wdb), pq

def orig_list(pq, nm):
    """Ground-truth column as Python objects (None for nulls)."""
    con = duckdb.connect()
    return [r[0] for r in con.execute(f'SELECT "{nm}" FROM \'{pq}\'').fetchall()]

def recon_list(seg, nm):
    """Reconstructed column as comparable Python objects."""
    c = seg.cols[nm]; vals = seg.values(nm)
    out = []
    for v in vals:
        if c['dt'] == 1:                       # bytes/string
            out.append(None if v is None else (v.decode() if isinstance(v,(bytes,bytearray)) else v))
        elif c['dt'] == 2:                     # float
            out.append(None if (v is None or (isinstance(v,float) and np.isnan(v))) else float(v))
        elif c['dt'] == 3:                      # datetime
            out.append(None if v is None else np.datetime64(v))
        else:                                   # int
            out.append(None if v is None else int(v))
    return out

def assert_lossless(seg, pq, nm):
    o = orig_list(pq, nm); r = recon_list(seg, nm)
    assert len(o) == len(r), f"{nm}: length {len(o)} != {len(r)}"
    c = seg.cols[nm]
    for i,(a,b) in enumerate(zip(o,r)):
        if a is None or (isinstance(a,float) and np.isnan(a)):
            assert b is None, f"{nm}[{i}]: expected null, got {b!r}"
            continue
        if c['dt'] == 3:
            assert np.datetime64(a) == b, f"{nm}[{i}]: {a!r} != {b!r}"
        elif c['dt'] == 2:
            assert abs(float(a) - b) < 1e-9 or float(a)==b, f"{nm}[{i}]: {a!r} != {b!r}"
        else:
            aa = a.decode() if isinstance(a,(bytes,bytearray)) else a
            assert aa == b, f"{nm}[{i}]: {aa!r} != {b!r}"
    return c['mode']
