"""FD-reference codec (FD pipeline stage 3, standalone — NOT wired into the .wdb format yet).

When X->Y is a verified exact FD, the dependent column Y need not store its own per-row
codes: it is fully determined by X. We store y_by_xcode[c] = the single Y value for each
distinct X code c (length Vx, typically << N rows), and reconstruct Y[i] = y_by_xcode[X[i]].

Encode is a vectorized scatter (rows sharing an X-code share one Y by the FD, so any wins);
decode is a gather. Nulls need no special handling: X's null is just a reserved code, Y's
null is just a value. Losslessness holds IFF X->Y is exact — is_lossless() checks it, so the
codec is only ever applied to a verified FD. This module is pure arrays; format integration
(stage 3b) is separate so the risky read-path change stays isolated until this is bulletproof.
"""
import numpy as np

def fd_encode(x_codes, y_values):
    """x_codes: int array (N,) of the determinant's per-row codes (0..Vx-1).
    y_values: array (N,) of the dependent column's per-row values.
    Returns y_by_xcode: array (Vx,) mapping each X code to its Y value."""
    x_codes = np.asarray(x_codes)
    y_values = np.asarray(y_values)
    Vx = int(x_codes.max()) + 1 if x_codes.size else 0
    y_by_xcode = np.empty(Vx, dtype=y_values.dtype)
    y_by_xcode[x_codes] = y_values   # FD => all rows with code c carry the same y; any wins
    return y_by_xcode

def fd_decode(x_codes, y_by_xcode):
    """Reconstruct the dependent column: Y[i] = y_by_xcode[X[i]]."""
    return np.asarray(y_by_xcode)[np.asarray(x_codes)]

def _equal(a, b):
    """Element-wise equality tolerant of None/NaN/NaT, for lossless checks."""
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape: return False
    if a.dtype == object or b.dtype == object:
        for x, y in zip(a.tolist(), b.tolist()):
            xn = x is None or (isinstance(x, float) and np.isnan(x))
            yn = y is None or (isinstance(y, float) and np.isnan(y))
            if xn or yn:
                if xn != yn: return False
            elif x != y:
                return False
        return True
    if np.issubdtype(a.dtype, np.floating):
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)

def is_lossless(x_codes, y_values):
    """True iff encoding Y as a reference into X round-trips exactly (i.e. X->Y is exact)."""
    return _equal(fd_decode(x_codes, fd_encode(x_codes, y_values)), np.asarray(y_values))
