"""enc 19's row walk on every core (wdb_kernels.e19_rows_eq_all, 2026-10-18): the rows holding a code, found from
the block labels and walked block-parallel, must be exactly the rows the decode finds -- for values spread over
every block, values in one block, one-value blocks (pointer width 0), and spans that start and end mid-block."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from wdb_db import Database
import wdb_encode, wdb_join, wdb_wherescan as WS, wdb_qmem

_D = None


def _fixture():
    global _D
    if _D is not None:
        return _D
    rng = np.random.default_rng(19); N = 1 << 21
    spread = rng.integers(0, 3000, N)                         # every value in every block
    clumped = np.repeat(np.arange(N // 4096), 4096) * 7       # each value in one run of rows
    flat = clumped.copy(); flat[: 3 * (1 << 16)] = 5          # whole blocks holding one value
    df = pd.DataFrame({'s': spread.astype('int64'), 'c': clumped.astype('int64'), 'f': flat.astype('int64')})
    d = os.path.join(tempfile.gettempdir(), 'e19p_%s' % uuid.uuid4().hex[:8])
    db = Database.create(d)
    db.run("CREATE TABLE t (s BIGINT, c BIGINT, f BIGINT)")
    pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
    old = os.environ.get('WDB_E19_FORCE'); os.environ['WDB_E19_FORCE'] = '1'
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        if old is None: os.environ.pop('WDB_E19_FORCE', None)
        else: os.environ['WDB_E19_FORCE'] = old
    db.cat.add_segment('t', 't_0.wdb')
    seg, _ = wdb_join._solo_segment(db, 't')
    _D = (db, seg, N)
    return _D


def test_rows_equal_decode():
    db, seg, N = _fixture()
    n19 = 0
    for col in ('s', 'c', 'f'):
        if seg.cols[col].get('code_enc') != 19:
            continue
        n19 += 1
        codes = np.asarray(seg._raw_codes(col)).copy()
        cnt = np.bincount(codes)
        picks = [int(np.argmax(cnt)), int(np.flatnonzero(cnt)[0]), int(np.flatnonzero(cnt)[-1]), int(codes[N // 3])]
        for code in picks:
            for lo, hi in ((0, N), (12345, N - 777), (70000, 70001), (65536, 131072)):
                wdb_qmem.flush(db)
                got = np.asarray(WS._scan_eq(seg, col, code, lo, hi), np.int64)
                want = np.flatnonzero(codes[lo:hi] == code) + lo
                assert np.array_equal(got, want), (col, code, lo, hi, got.size, want.size)
    assert n19 >= 1, 'no column took enc 19'


def test_switch_matches_old_walk():
    db, seg, N = _fixture()
    for col in ('s', 'c', 'f'):
        if seg.cols[col].get('code_enc') != 19:
            continue
        code = int(np.asarray(seg._raw_codes(col))[N // 2])
        wdb_qmem.flush(db)
        new = np.asarray(WS._scan_eq(seg, col, code, 0, N), np.int64)
        os.environ['WDB_E19_PAR'] = '0'
        try:
            wdb_qmem.flush(db)
            old = np.asarray(WS._scan_eq(seg, col, code, 0, N), np.int64)
        finally:
            os.environ.pop('WDB_E19_PAR', None)
        assert np.array_equal(new, old), col
