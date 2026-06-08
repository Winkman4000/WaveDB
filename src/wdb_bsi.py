"""Bit-sliced index (BSI) + per-value bitmaps for predicate evaluation.

A column's integer dictionary codes are indexed in one of two forms, chosen by
measured cardinality and predicate type:

  - bit-sliced (BSI): B bit-planes, plane[b] = packed bits of (code>>b)&1.
    Range / eq / IN resolve in O(B) packed bitwise passes, storage N*B bits,
    any cardinality. Range is only value-correct when the column dict is
    value-sorted (so code order == value order); the caller maps a value range
    to a code range via searchsorted before calling range().

  - per-value bitmaps: one packed bitmap per distinct code. Equality / IN in
    O(matching) packed ORs, storage N*K bits. Low-card only.

Every predicate op returns a packed uint8 bitmap (np.packbits layout, MSB-first,
length ceil(N/8)). Pad bits beyond N are held at 0 so AND/OR combiners and the
range/eq/in ops stay clean without re-masking. Combine across predicates with
& / | -- nearly free. Decode matches with the wdb_bsi numba kernels (Phase 2)
or to_positions()/popcount() here.
"""
import numpy as np


def _ones(N):
    """Packed all-true bitmap for N rows, with pad bits held at 0."""
    return np.packbits(np.ones(N, dtype=bool))


class BSI:
    """Bit-sliced index over integer codes. plane[b] = packed bit b of every code."""
    __slots__ = ('planes', 'B', 'N', 'ones')

    def __init__(self, planes, N):
        self.planes = planes                 # list[packed uint8], planes[0] = LSB
        self.B = len(planes)
        self.N = N
        self.ones = _ones(N)

    def ge(self, C):
        """Packed bitmap of rows where code >= C (O(B) bitwise passes)."""
        C = int(C)
        if C <= 0:
            return self.ones.copy()
        if C >= (1 << self.B):
            return np.zeros_like(self.ones)
        EQ = self.ones.copy()                 # equal to C on bits seen so far
        GT = np.zeros_like(self.ones)         # strictly greater on a higher bit
        for b in range(self.B - 1, -1, -1):
            pb = self.planes[b]
            if (C >> b) & 1:
                EQ &= pb                       # need bit b set to stay equal
            else:
                GT |= EQ & pb                  # bit b set while C's is 0 -> greater
                EQ &= ~pb                      # bit b clear -> stays equal
        GT |= EQ                               # >= : greater OR exactly equal
        return GT

    def range(self, lo, hi):
        """Packed bitmap of rows where lo <= code < hi (code-space)."""
        return self.ge(lo) & ~self.ge(hi)

    def eq(self, C):
        """Packed bitmap of rows where code == C."""
        return self.ge(C) & ~self.ge(int(C) + 1)

    def in_set(self, vals):
        """Packed bitmap of rows where code in vals (OR of eq, small sets)."""
        r = np.zeros_like(self.ones)
        for v in vals:
            r |= self.eq(v)
        return r

    def nbytes(self):
        return sum(p.nbytes for p in self.planes)


def build_bsi(codes, N=None):
    """Build a BSI from integer codes (1-D). B = bits needed for max code."""
    codes = np.ascontiguousarray(codes)
    if N is None:
        N = int(codes.shape[0])
    mx = int(codes.max()) if codes.size else 0
    B = max(1, mx.bit_length())
    planes = [np.packbits(((codes >> b) & 1).astype(np.uint8)) for b in range(B)]
    return BSI(planes, N)


def build_value_bitmaps(codes, V=None):
    """Build one packed bitmap per distinct code value (per-value index)."""
    codes = np.ascontiguousarray(codes)
    if V is None:
        V = (int(codes.max()) + 1) if codes.size else 0
    return [np.packbits(codes == v) for v in range(V)]


def vbm_in(bms, idxs):
    """OR the per-value bitmaps at the given code indices -> packed bitmap."""
    idxs = list(idxs)
    if not idxs:
        return np.zeros_like(bms[0])
    r = bms[idxs[0]].copy()
    for i in idxs[1:]:
        r |= bms[i]
    return r


def to_positions(bm, N):
    """Row indices where the packed bitmap is set (ascending)."""
    return np.nonzero(np.unpackbits(bm)[:N])[0]


def popcount(bm):
    """Number of set bits (pad bits are 0, so this is exact)."""
    return int(np.unpackbits(bm).sum())
