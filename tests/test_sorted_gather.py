
"""THE SORTED GATHER (2026-10-01): codes_at on ascending rows of an enc-3 (zstd frames) or enc-18
(packed frames) column walks frame runs in lanes and gathers with a compiled kernel -- it must equal the
old argsort path (WDB_SORTGATHER=0) and the full decode at those rows, values and dtype, for random,
duplicated, one-frame, frame-edge and dense row sets; unsorted rows still take the old path."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode
import wdb_engine
from wdb_engine import Segment

TMP = tempfile.gettempdir()
_REAL = {'WDB_SEQ_NARROW_OK': '0', 'WDB_SEQ_REPEATS_OK': '0', 'WDB_E18_FORCE': '1'}


def _toy():
    rng = np.random.default_rng(57)
    n = 1_500_007
    pool = np.array(['u%05d' % i for i in range(4000)])
    runs = pool[np.minimum(rng.zipf(1.3, n), 4000) - 1]                  # skewed text codes: tag 3
    wide = np.char.add('w', rng.integers(0, 300000, n).astype(str))     # 19-bit text codes: tag 18
    d = os.path.join(TMP, 'sg_' + uuid.uuid4().hex[:8]); pq = d + '.parquet'
    os.makedirs(d)
    pd.DataFrame({'runs': runs, 'wide': wide}).to_parquet(pq, index=False)
    old = {k: os.environ.get(k) for k in _REAL}
    os.environ.update(_REAL)
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'), stream=True)
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return d, pq


def _sets(N):
    rng = np.random.default_rng(58)
    BR = 65536
    return {'ten': np.sort(rng.integers(0, N, 10)),
            'dups': np.sort(np.concatenate([rng.integers(0, N, 4000)] * 2)),
            'one_frame': np.arange(BR * 7 + 5, BR * 7 + 900, 3, dtype=np.int64),
            'edges': np.unique(np.clip(np.concatenate([np.arange(0, N, BR) + k for k in (-1, 0, 1)] + [[N - 1]]), 0, N - 1)),
            'dense': np.sort(rng.choice(N, N // 2, replace=False)),
            'pair': np.array([0, N - 1], np.int64)}


def _at(P, nm, rows, on):
    old = wdb_engine._SORTGATHER[0]
    try:
        wdb_engine._SORTGATHER[0] = on
        return Segment(P).codes_at(nm, rows)
    finally:
        wdb_engine._SORTGATHER[0] = old


def test_sorted_gather_equals_old_path_and_decode():
    d, pq = _toy()
    try:
        P = os.path.join(d, 't_0.wdb')
        s = Segment(P); N = int(s.N)
        tags = {nm: s.cols[nm].get('code_enc') for nm in ('runs', 'wide')}
        assert set(tags.values()) == {3, 18}, tags                      # both frame kinds exercised
        for nm in ('runs', 'wide'):
            full = np.asarray(Segment(P)._raw_codes(nm))
            for k, r in _sets(N).items():
                a = np.asarray(_at(P, nm, r, True)); b = np.asarray(_at(P, nm, r, False))
                assert a.dtype == b.dtype, (nm, k, a.dtype, b.dtype)
                assert np.array_equal(a.astype(np.int64), b.astype(np.int64)), (nm, k)
                assert np.array_equal(a.astype(np.int64), full[r].astype(np.int64)), (nm, k)
            r = np.random.default_rng(59).integers(0, N, 5000)          # unsorted: the old path
            assert np.array_equal(np.asarray(_at(P, nm, r, True)).astype(np.int64), full[r].astype(np.int64))
    finally:
        shutil.rmtree(d, ignore_errors=True)
        try: os.remove(pq)
        except OSError: pass
