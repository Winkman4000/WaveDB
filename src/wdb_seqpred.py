"""
wdb_seqpred — predicate pushdown on mode-4 (WSQ1 affine) columns.

A mode-4 column is value[i] = base + i*stride plus exceptions; for stride==0 it is piecewise
constant, changing only at the n_exc exception positions. So `col <op> const` resolves against the
O(n_exc) change-points -- yielding the SURVIVOR ROW-RANGES -- without decoding the O(n) column. A
filtered operator then touches only the rows in those ranges (and only the columns it needs): don't
read the filter column, jump straight to the survivors.

Reusable by any filtered operator (group-by, count, scan). Returns None for cases it does not handle
(stride!=0, non-comparison predicate) so the caller falls back to a normal scan. The per-column
piecewise-constant structure is built once and cached on the segment's column dict.
"""
import numpy as np
import wdb_seqcodec as SQ

NEQ, EQ, LT, LTE, GT, GTE = 0, 1, 2, 3, 4, 5


def n_exceptions(seg, col):
    """O(1) exception count for a mode-4 column (the structural-path size), or None."""
    c = seg.cols.get(col)
    if not c or c.get('mode') != 4:
        return None
    blob = c.get('seqblob')
    if blob is None:
        return None
    return SQ.header(blob)[3]


def _struct(seg, col):
    """Cached piecewise-constant structure for a stride==0 mode-4 column:
    (seg_lo, seg_hi, seg_val) where rows [seg_lo[k], seg_hi[k]) all hold value seg_val[k].
    None if not a stride==0 WSQ1 column we can push predicates into."""
    c = seg.cols.get(col)
    if not c or c.get('mode') != 4:
        return None
    cached = c.get('seqpred', 0)
    if cached != 0:
        return cached
    blob = c.get('seqblob')
    if blob is None:
        c['seqpred'] = None
        return None
    base, stride, n, exc, corr = SQ.exceptions(blob)
    if stride != 0:
        c['seqpred'] = None                     # value varies within a run -> not handled (v1)
        return None
    starts = (exc + 1).astype(np.int64)         # value changes at row exc+1
    seg_lo = np.concatenate([[0], starts]).astype(np.int64)
    seg_hi = np.concatenate([starts, [n]]).astype(np.int64)
    seg_val = np.concatenate([[np.int64(base)], np.int64(base) + np.cumsum(corr)]).astype(np.int64)
    out = (seg_lo, seg_hi, seg_val)
    c['seqpred'] = out
    return out


def _seg_mask(seg_val, op, const):
    if op == NEQ: return seg_val != const
    if op == EQ:  return seg_val == const
    if op == LT:  return seg_val < const
    if op == LTE: return seg_val <= const
    if op == GT:  return seg_val > const
    if op == GTE: return seg_val >= const
    return None


def survivor_ranges(seg, col, op, const):
    """(los, his) ascending non-overlapping row-ranges where `col <op> const`, computed from the
    exception structure (no column decode), or None if unsupported. No survivors -> empty arrays."""
    st = _struct(seg, col)
    if st is None:
        return None
    seg_lo, seg_hi, seg_val = st
    m = _seg_mask(seg_val, op, int(const))
    if m is None:
        return None
    return seg_lo[m], seg_hi[m]


def survivor_count(los, his):
    return int((his - los).sum())


def ranges_to_ids(los, his):
    """Flatten sorted non-overlapping [lo,hi) ranges to ascending row indices (vectorized)."""
    lens = his - los
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    out[0] = los[0]
    ends = np.cumsum(lens)
    out[ends[:-1]] = los[1:] - his[:-1] + 1
    return np.cumsum(out)
