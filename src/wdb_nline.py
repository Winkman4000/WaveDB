"""wdb_nline: the fixed number line as a file.

A .nline.<col> sidecar is the column's value-ordered int table written as a flat
little-endian int64 array -- born mappable: layout IS the compute layout. _int_table
(wdb_window) memmaps it: one virtual address for the life of the process, the OS
page cache owns the bytes, and the per-query dict rebuild is abolished (measured:
0.54s/query -> 1us on UserID at 17.6M values). Sidecars are derived, regenerable,
optional: no sidecar, old rebuild path, bit-for-bit identical results.

Threshold: int-valued dict columns (modes 0/1/2) with n_dict >= NLINE_MIN -- below
it the rebuild is microseconds and a line adds file clutter for nothing.
"""
import os
import numpy as np

NLINE_MIN = 1_000_000


def line_path(seg_path, col):
    return os.path.realpath(seg_path) + '.nline.' + col


def eligible(seg):
    out = []
    for nm in seg.order:
        c = seg.cols[nm]
        if c.get('mode') in (0, 1, 2) and int(c.get('n_dict') or 0) >= NLINE_MIN:
            out.append(nm)
    return out


def build(seg, cols=None, force=False):
    """Write lines for the given (or all eligible) columns. Atomic per line: readers
    never see a half-written file. String dicts decline (the .sline twin's job).
    Returns [(col, n_values), ...] for lines actually written."""
    made = []
    import wdb_sidecar
    if not wdb_sidecar.births_on(os.path.dirname(seg.path)):
        return made                                   # THE SWITCH: no lines written
    for nm in (cols if cols is not None else eligible(seg)):
        p = line_path(seg.path, nm)
        if os.path.exists(p) and not force:
            continue
        c = seg.cols[nm]
        try:
            tab = np.asarray(seg._dict_ints(c), dtype=np.int64)
        except Exception:
            continue                    # string-valued dict: not this line's format
        tmp = p + '.tmp'
        tab.astype('<i8').tofile(tmp)
        os.replace(tmp, p)
        made.append((nm, int(tab.size)))
        seg.drop_derived()              # the build's own decode dies like any query's
    return made
