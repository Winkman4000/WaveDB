#!/usr/bin/env python3
"""
wdb_profile -- per-column statistics that gate the physical-design levers.

Computed from a Segment (piggybacks the dict the encoder already built):

    V          cardinality (distinct values)
    H_bits     Shannon entropy of the column (bits/row) -- compression target
    uniq       V/N -- near-1 => column can serve as a content-address / free key
    sorted     dict monotonic? -> range filter can be a contiguous slice
    code_bits  ceil(log2 V) -- entropy-minimal in-memory code width
    mode, dt, has_null

Persisted next to the segment as <segment>.stats.json. The planner reads it to
decide: code-LUT (V<=lut_cliff), content-address (uniq>=thresh), range-as-slice
(sorted), narrow codes (code_bits), cluster candidacy.
"""
import os, json, math
import numpy as np


def _entropy_bits(codes, V):
    if codes.size == 0:
        return 0.0
    cnt = np.bincount(codes, minlength=max(V, 1))
    p = cnt[cnt > 0] / codes.size
    return float(-(p * np.log2(p)).sum())


def _dict_sorted(seg, nm):
    """Is the value dictionary monotonic non-decreasing? Numeric/datetime via the
    typed dict; strings via byte comparison. None if not determinable (mode-4
    affine codes are positional, not a value dict)."""
    c = seg.cols[nm]
    if c['mode'] == 4:
        return None
    try:
        if c['dt'] == 1:                       # string dict -> compare as bytes
            dv = seg.dict_vals(nm)
            if not dv:
                return None
            bs = [v if isinstance(v, (bytes, bytearray)) else str(v).encode() for v in dv]
            return all(bs[i] <= bs[i + 1] for i in range(len(bs) - 1))
        td = np.asarray(seg._typed_dict(nm))   # numeric / datetime epochs
        if td.dtype.kind in 'iuf' and td.size:
            return bool(np.all(np.diff(td) >= 0))
    except Exception:
        return None
    return None


def cardinality(seg, col):
    """True distinct-value count for one column -- the canonical data-measurement of cardinality.
    Dict/inline modes (0,1,2,3,5) store it directly as V (verified on real-scale segments); mode 6 is a
    synthetic constant (1); mode 4 (affine/positional) stores N rather than the cardinality, so it is
    measured from the values. Centralized here so every consumer (cube auto-enumeration, the physical-
    design planner, profiling) reads ONE definition and they cannot drift."""
    m = seg.cols[col]['mode']
    if m in (0, 1, 2, 3, 5): return int(seg.cols[col]['V'])
    if m == 6:               return 1
    import pandas as pd
    return int(len(pd.unique(np.asarray(seg.resident_values(col)))))


def segment_cardinalities(seg):
    """True distinct count for every column, in column order -- the input to cube auto-enumeration."""
    return {nm: cardinality(seg, nm) for nm in seg.order}


def profile_col(seg, nm):
    c = seg.cols[nm]; N = int(seg.N)
    if c['mode'] == 4:                          # affine/seq: stored V is positional -> decode values for true stats
        vals = seg.values(nm); u, cnt = np.unique(vals, return_counts=True)
        V = int(u.size); p = cnt / cnt.sum()
        H = float(-(p * np.log2(p)).sum()) if p.size else 0.0
        srt = bool(np.all(np.diff(u) >= 0)) if (getattr(u, 'dtype', None) is not None and u.dtype.kind in 'iuf') else None
    else:
        V = int(c['V']); H = _entropy_bits(seg.codes(nm), V); srt = _dict_sorted(seg, nm)
    return {
        'V': V,
        'H_bits': round(max(H, 0.0), 3),
        'uniq': round(V / N, 6) if N else 0.0,
        'sorted': srt,
        'code_bits': max(1, math.ceil(math.log2(max(V, 2)))),
        'mode': int(c['mode']),
        'dt': int(c['dt']),
        'has_null': bool(c['has_null']),
    }


def profile_segment(seg):
    return {nm: profile_col(seg, nm) for nm in seg.cols}


def stats_path(segment_path):
    return segment_path + '.stats.json'


def save_profile(segment_path, stats):
    p = stats_path(segment_path)
    with open(p, 'w') as f:
        json.dump(stats, f, indent=2)
    return p


def load_profile(segment_path):
    p = stats_path(segment_path)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def profile_and_save(seg, segment_path):
    st = profile_segment(seg)
    save_profile(segment_path, st)
    return st


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wdb_db import Database
    db = Database.open(sys.argv[1]); table = sys.argv[2]
    for sp in db.cat.segment_paths(table):
        seg = db.open_segment(sp, table)
        st = profile_and_save(seg, sp)
        print(f"\n{sp}  ->  {stats_path(sp)}")
        print(f"{'column':18s} {'V':>10s} {'H':>7s} {'uniq':>8s} {'cbits':>5s} {'sorted':>6s}")
        for nm, s in st.items():
            print(f"{nm:18s} {s['V']:>10,d} {s['H_bits']:>7.2f} "
                  f"{s['uniq']:>8.4f} {s['code_bits']:>5d} {str(s['sorted']):>6s}")
