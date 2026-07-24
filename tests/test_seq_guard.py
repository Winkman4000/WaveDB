"""The cardinality guard: narrow columns must never wear mode 4 (production law)."""
import os
import numpy as np
import sys
sys.path.insert(0, __file__.rsplit('/tests/', 1)[0] + '/src')
import wdb_encode


def test_narrow_column_declines_seq():
    saved = os.environ.pop('WDB_SEQ_NARROW_OK', None)
    try:
        col = np.zeros(50000, np.int64)
        col[::7] = 3                                  # 2 distincts, highly affine runs
        p = wdb_encode._prep_column('flag', col)
        assert p['mode'] != 4, 'narrow column misfired into mode 4'
    finally:
        if saved is not None:
            os.environ['WDB_SEQ_NARROW_OK'] = saved


def test_wide_affine_still_seq():
    saved = os.environ.pop('WDB_SEQ_NARROW_OK', None)
    try:
        col = np.arange(100000, dtype=np.int64) * 3 + 7
        p = wdb_encode._prep_column('seqcol', col)
        assert p['mode'] == 4, 'wide affine column should keep mode 4'
    finally:
        if saved is not None:
            os.environ['WDB_SEQ_NARROW_OK'] = saved
