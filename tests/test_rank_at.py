"""THE POINT READ BY RANK (2026-10-01): Segment.codes_at on tags 8 and 9 reads each row's rank from its
64K checkpoint plus the presence bits up to it, then the literal (tag 8) or the tiers and the tail (tag 9) --
it must equal the full decode for any rows (random, unsorted with repeats, every block edge, the first and
last rows, absent rows) and build neither the planes nor the full decode. WDB_RANKAT=0 keeps the old path."""
import sys, os, uuid, tempfile, shutil, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode
import wdb_engine
from wdb_engine import Segment

TMP = tempfile.gettempdir()


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def _toy(force8):
    rng = np.random.default_rng(31 if force8 else 32)
    n = 1_200_000
    tail = np.minimum(rng.zipf(1.6, n), 40)
    mp = np.where(rng.random(n) < 0.92, 0, tail).astype(np.int64)          # tag 9's shape
    words = np.array(['', 'alpha', 'beta', 'gamma', 'delta'] + ['w%05d' % i for i in range(3000)], dtype=object)
    sp = words[np.where(rng.random(n) < 0.85, 0, rng.integers(1, words.size, n))]   # mostly ''
    d = os.path.join(TMP, 'rk_' + uuid.uuid4().hex[:8]); pq = d + '.parquet'
    os.makedirs(d)
    pd.DataFrame({'mp': mp, 'sp': sp, 'x': rng.integers(0, 9, n).astype(np.int64)}).to_parquet(pq, index=False)
    env = {'WDB_SEQ_NARROW_OK': '0'}
    if force8: env['WDB_E8_FORCE'] = '1'
    with _env(**env):
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'), stream=True)
    return d, pq


def test_rank_point_reads_equal_the_full_decode():
    seen = set()
    for force8 in (False, True):
        d, pq = _toy(force8)
        try:
            P = os.path.join(d, 't_0.wdb')
            s0 = Segment(P); N = int(s0.N)
            cols = [c for c in ('mp', 'sp') if s0.cols[c].get('code_enc') in (8, 9)]
            seen |= {s0.cols[c]['code_enc'] for c in cols}
            rng = np.random.default_rng(5)
            edges = np.unique(np.clip(np.concatenate([np.arange(0, N, 65536) + k for k in (-1, 0, 1)]), 0, N - 1))
            sets = [rng.integers(0, N, 10), rng.integers(0, N, 5000)[rng.permutation(5000)],
                    np.concatenate([rng.integers(0, N, 300)] * 3), edges, np.array([N - 1, 0, 1, N - 2])]
            for nm in cols:
                full = np.asarray(Segment(P)._raw_codes(nm)).astype(np.int64)
                d0 = int(s0.cols[nm].get('e8d', s0.cols[nm].get('e9d')))
                absent = np.flatnonzero(full == d0)[:50]; present = np.flatnonzero(full != d0)[:50]
                for r in sets + [absent, present]:
                    s = Segment(P)
                    got = np.asarray(s.codes_at(nm, r)).astype(np.int64)
                    assert np.array_equal(got, full[r]), (nm, s0.cols[nm]['code_enc'], r[:5])
                    assert nm not in s._codes and nm not in s.__dict__.get('_e8pm', {}), nm
                wdb_engine._RANKAT[0] = False                       # the old path agrees
                try:
                    assert np.array_equal(np.asarray(Segment(P).codes_at(nm, sets[1])).astype(np.int64), full[sets[1]])
                finally:
                    wdb_engine._RANKAT[0] = True
        finally:
            shutil.rmtree(d, ignore_errors=True); os.remove(pq)
    assert seen == {8, 9}, ('the toys must hold a tag-8 and a tag-9 column', seen)
