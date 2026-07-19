"""wdb_exactint: exact integer aggregation at any magnitude.

SUM over an int64 column can exceed int64 (SUM(UserID) at 100M reaches ~2.5e26):
float64 accumulation drifts (~1e13 absolute at that scale) and int64 accumulation
silently wraps. Duck promotes to HUGEINT; we get exactness from dictionary
arithmetic instead: SUM = sum_c counts[c] * vals[c], computed in two 32-bit limbs
so every int64 partial stays within bounds (safe to N ~ 4e9 rows), recombined in
python ints -- arbitrary precision, exact to the last digit.
"""
import numpy as np

_MASK = np.int64(0xFFFFFFFF)


def fold_counts(counts, vals):
    """EXACT sum(counts[c] * vals[c]) for int64 vals, python-int result."""
    counts = np.asarray(counts, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.int64)
    hi = vals >> np.int64(32)                 # arithmetic shift: signed-safe
    lo = vals & _MASK                         # [0, 2^32)
    return (int(counts @ hi) << 32) + int(counts @ lo)


def fold_values(vals):
    """EXACT sum of an int64 array (counts of one each), python-int result."""
    vals = np.asarray(vals, dtype=np.int64)
    hi = (vals >> np.int64(32)).sum(dtype=np.int64)
    lo = (vals & _MASK).sum(dtype=np.int64)
    return (int(hi) << 32) + int(lo)


def exact_avg(total, n):
    """Correctly-rounded float AVG from an exact python-int SUM."""
    if not n:
        return None
    from fractions import Fraction
    return float(Fraction(total, n))
