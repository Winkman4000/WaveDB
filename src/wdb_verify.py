#!/usr/bin/env python3
"""Verify a WVDB3 segment is byte-exact lossless vs its source file. Null- and float-aware."""
import duckdb, numpy as np, numpy.ma as ma, sys
sys.path.insert(0, '/home/jack/WaveDB/src')
from wdb_engine import Segment

def _expected(col):
    """Build an object array of expected values from the source column, with None for nulls."""
    if isinstance(col, ma.MaskedArray):
        m = ma.getmaskarray(col); data = np.asarray(col.data)
    else:
        m = np.zeros(len(col), dtype=bool); data = np.asarray(col)
    k = data.dtype.kind
    if k == 'M':
        iv = data.view('int64')
        return np.array([None if m[i] else int(iv[i]) for i in range(len(data))], dtype=object)
    out = np.empty(len(data), dtype=object)
    for i in range(len(data)):
        if m[i]: out[i] = None
        elif k in 'iu': out[i] = int(data[i])
        elif k == 'f': out[i] = float(data[i])
        else:
            x = data[i]
            out[i] = bytes(x) if isinstance(x,(bytes,bytearray)) else (x.encode('utf-8','surrogatepass') if isinstance(x,str) else str(x).encode('utf-8','surrogatepass'))
    return out

def _recon_obj(seg, nm):
    """Reconstructed values as an object array (None for null), comparable to _expected."""
    c = seg.cols[nm]
    if c['dt'] == 3:
        dv = seg._typed_dict(nm); codes = seg.codes(nm)
        if c['has_null']:
            nc = c['V']-1
            return np.array([None if cd==nc else dv[cd] for cd in codes], dtype=object)
        return np.array(dv, dtype=np.int64)[codes]
    vals = seg.values(nm)
    if c['has_null'] or c['dt'] == 2 or vals.dtype == object:
        # already object (nullable) or float/bytes -> normalize to python scalars
        out = np.empty(len(vals), dtype=object)
        for i in range(len(vals)):
            v = vals[i]
            if v is None: out[i] = None
            elif c['dt'] == 0: out[i] = int(v)
            elif c['dt'] == 2: out[i] = float(v)
            else: out[i] = bytes(v) if isinstance(v,(bytes,bytearray)) else v
        return out
    if c['dt'] == 0:
        return np.array([int(v) for v in vals], dtype=object)
    return np.array([bytes(v) for v in vals], dtype=object)

def verify(segment_path, source_path):
    seg = Segment(segment_path); con = duckdb.connect(); con.execute("PRAGMA threads=8")
    ok = bad = 0; bad_cols = []
    for nm in seg.order:
        orig = con.execute(f'SELECT "{nm}" FROM \'{source_path}\'').fetchnumpy()[nm]
        c = seg.cols[nm]; masked = isinstance(orig, ma.MaskedArray)
        if not c['has_null'] and not masked and c['dt'] == 0:
            recon = np.array([int(v) for v in c['vals']], dtype=np.int64)[seg.codes(nm)]
            match = np.array_equal(recon, orig.astype(np.int64))
        elif not c['has_null'] and not masked and c['dt'] == 3:
            dv = seg._typed_dict(nm); recon = np.array(dv, dtype=np.int64)[seg.codes(nm)]
            match = np.array_equal(recon, orig.view('int64'))
        elif not c['has_null'] and not masked and c['dt'] == 1:
            recon = seg.values(nm)
            ob = np.array([bytes(x) if isinstance(x,(bytes,bytearray)) else (x.encode('utf-8','surrogatepass') if isinstance(x,str) else str(x).encode('utf-8','surrogatepass')) for x in orig], dtype=object)
            match = np.array_equal(recon, ob)
        else:
            exp = _expected(orig); rec = _recon_obj(seg, nm)
            match = len(exp) == len(rec) and all(a == b for a, b in zip(exp, rec))
        if match: ok += 1
        else: bad += 1; bad_cols.append(nm)
    return ok, bad, bad_cols

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: wdb_verify.py <segment.wdb> <source.parquet>"); sys.exit(1)
    ok, bad, bad_cols = verify(sys.argv[1], sys.argv[2])
    print(f"LOSSLESS: {ok}/{ok+bad} columns byte-perfect" + ("" if bad==0 else f"  MISMATCH: {bad_cols}"))
    sys.exit(0 if bad==0 else 1)
