#!/usr/bin/env python3
"""
wdb_profile -- DATA measurement: true per-column cardinality.

The .stats.json sidecar machinery (profile_col/profile_segment/save/load) and its
sole consumer (wdb_plan) were parked 2026-06-14 -- they fed only the retired
physical-design advisor. What remains is the one live data-measurement: the
canonical distinct-value count read by cube auto-enumeration. See
docs/auto_physical_design.md for the parked subsystem.
"""
import numpy as np


def cardinality(seg, col):
    """True distinct-value count for one column -- the canonical data-measurement of cardinality.
    Dict/inline modes (0,1,2,3,5) store it directly as V (verified on real-scale segments); mode 6 is a
    synthetic constant (1); mode 4 (affine/positional) stores N rather than the cardinality, so it is
    measured from the values. Centralized here so every consumer reads ONE definition and cannot drift."""
    m = seg.cols[col]['mode']
    if m in (0, 1, 2, 3, 5): return int(seg.cols[col]['V'])
    if m == 6:               return 1
    import pandas as pd
    return int(len(pd.unique(np.asarray(seg.resident_values(col)))))


def segment_cardinalities(seg):
    """True distinct count for every column, in column order -- the input to cube auto-enumeration."""
    return {nm: cardinality(seg, nm) for nm in seg.order}
