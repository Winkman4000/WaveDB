"""THE TIERED DRESS, BY RANK (2026-09-30): Segment._raw_codes_range on a tag-9 column reads a small window
[lo, hi) from the presence bitmap, its checkpoints and the tiers' prefix bits -- not the full decode.
It must equal the full decode for every window (first block, block edges, the last rows, random), and
must not leave the column decoded behind it. stairs()' first-block "no" rides it (Q10/Q11's detect)."""
import sys, os, uuid, tempfile, shutil, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode
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


def test_tag9_range_equals_full_decode():
    rng = np.random.default_rng(29)
    n = 1_200_000                                                # rule eleven elects at >=65,536 non-default rows
    tail = np.minimum(rng.zipf(1.6, n), 40)                     # a skewed tail: the tiered dress's shape
    mp = np.where(rng.random(n) < 0.92, 0, tail).astype(np.int64)
    names = np.array(['m%02d' % v if v else '' for v in mp], dtype=object)
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'e9_{t}'); pq = d + '.parquet'
    os.makedirs(d)
    pd.DataFrame({'mp': mp, 'mm': names, 'x': rng.integers(0, 9, n).astype(np.int64)}).to_parquet(pq, index=False)
    try:
        with _env(WDB_SEQ_NARROW_OK='0'):
            wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'), stream=True)
        seg = Segment(os.path.join(d, 't_0.wdb'))
        nine = [c for c in ('mp', 'mm') if seg.cols[c].get('code_enc') == 9]
        assert nine, ('the toy holds no tag-9 column', {c: seg.cols[c].get('code_enc') for c in ('mp', 'mm')})
        N = int(seg.N)
        wins = [(0, 65535), (0, 1), (N - 3000, N), (65536, 65536 + 65535), (65535, 65537), (131071, 131072 + 900)]
        for _ in range(150):
            lo = int(rng.integers(0, N - 1)); wins.append((lo, min(N, lo + int(rng.integers(1, 65536)))))
        for nm in nine:
            fresh = Segment(os.path.join(d, 't_0.wdb'))
            got = [np.asarray(fresh._raw_codes_range(nm, lo, hi)).astype(np.int64) for lo, hi in wins]
            assert nm not in fresh._codes, ('the window read decoded the whole column', nm)
            full = np.asarray(seg._raw_codes(nm)).astype(np.int64)
            for (lo, hi), g in zip(wins, got):
                assert np.array_equal(g, full[lo:hi]), (nm, lo, hi)
            assert fresh.stairs(nm) is None                    # not a staircase, learned from one block
            assert nm not in fresh._codes
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)
