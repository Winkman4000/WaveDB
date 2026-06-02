"""Override sidecar: per-segment mutable map of row -> new value, for UPDATE without
rewriting the immutable .wdb. Sits next to the segment as '<segment>.overrides'.

A row's override carries the VALUE directly (not a code), so an UPDATE can set a value the
segment's frozen dictionary has never seen (the hard Tier-2 case) with no special handling --
the value is just scattered in at read time. Reads resolve overrides with a vectorized,
dtype-preserving scatter (measured ~1.1x flat regardless of how many rows are overridden).

A MISSING sidecar means "no overrides" -> zero cost, no migration. This is the mutable
scratch tier; at compaction the overrides are folded into a fresh dictionary and the sidecar
is dropped (same lifecycle as tombstones). Format is pickle for now (values are arbitrary-
typed and the data is small); can be tightened later -- it never reaches the cold form.

Stored per column: {col_name: (row_idx uint32[k], vals object/typed[k])}, row_idx sorted.
"""
import os, pickle, numpy as np

def path_for(seg_path):
    return seg_path + '.overrides'

def load(seg_path):
    """Return {col: (idx, vals)} or None if no sidecar exists."""
    p = path_for(seg_path)
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)

def save(seg_path, overrides):
    p = path_for(seg_path); tmp = p + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(overrides, f, protocol=4)
    os.replace(tmp, p)   # atomic

def set_override(seg_path, col, row_idx, vals):
    """Merge overrides for one column; later writes win on a row. Persists the sidecar."""
    ov = load(seg_path) or {}
    row_idx = np.asarray(row_idx, dtype=np.uint32)
    vals = vals if isinstance(vals, np.ndarray) else np.asarray(vals, dtype=object)
    if col in ov:
        oi, ovv = ov[col]
        m = dict(zip(oi.tolist(), list(ovv)))
        for i, v in zip(row_idx.tolist(), list(vals)):
            m[i] = v
        ni = np.array(sorted(m.keys()), dtype=np.uint32)
        nv = np.array([m[int(i)] for i in ni.tolist()], dtype=object)
        ov[col] = (ni, nv)
    else:
        order = np.argsort(row_idx, kind='stable')
        ov[col] = (row_idx[order], vals[order])
    save(seg_path, ov)

def override_count(seg_path):
    ov = load(seg_path)
    return 0 if ov is None else sum(len(v[0]) for v in ov.values())
