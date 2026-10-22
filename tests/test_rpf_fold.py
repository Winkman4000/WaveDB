"""THE REPEATS-PER-FRAME FOLD (Jackson, 2026-10-21, Q29's WatchID): SUM / AVG of a near-unique integer column with no
census. The switch is the column's repeats per frame, (N - V) / frames; below the line (1.0, from the sweep: the fold
won at every point up to 0.82 and the paths crossed near 1.8) one walk tallies rows and code sums per dictionary frame,
a frame with one extra row names its repeated code by arithmetic, crowded frames are counted on their own boards, and
one pass through the dictionary's frames adds every value once plus the repeats. Oracle: the census fold and DuckDB."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_engine as EN, controller
from wdb_engine import Segment
from wdb_db import Database

TMP = tempfile.gettempdir()
ENV = dict(WDB_I2CHUNK='1024', WDB_I2CHUNK_MIN='1000', WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0')


def _col(n_unique, extras, seed):
    rng = np.random.default_rng(seed)
    vals = rng.integers(-(1 << 62), 1 << 62, n_unique, dtype=np.int64)
    parts = [vals]
    for code_pick, times in extras:                  # (which value, how many extra copies)
        parts.append(np.full(times, vals[code_pick], np.int64))
    x = np.concatenate(parts)
    return x[rng.permutation(x.size)]


def _seg(x):
    keep = {k: os.environ.get(k) for k in ENV}
    os.environ.update(ENV)
    try:
        t = uuid.uuid4().hex[:8]; pq = f'{TMP}/rpf_{t}.parquet'; w = f'{TMP}/rpf_{t}.wdb'
        pd.DataFrame({'x': x}).to_parquet(pq, index=False); wdb_encode.encode(pq, w)
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return Segment(w)


def _census(seg):
    c = seg.cols['x']
    return seg.i2_fold_sum('x', seg.raw_census('x', int(c.get('n_dict', c['V']))))


def _shape(seg):
    c = seg.cols['x']
    assert c.get('mode') == 2 and c.get('code_enc', 0) == 0 and c.get('i2ch') == 1024, \
        (c.get('mode'), c.get('code_enc'), c.get('i2ch'))


def test_no_repeats_and_a_few():
    for extras in ([], [(5, 1), (900, 1), (150_000, 1)],                   # single repeats: named by arithmetic
                   [(7, 2), (8, 1), (20_000, 3), (20_001, 1), (299_999, 1)]):  # crowded frames, a code thrice+
        x = _col(300_000, extras, len(extras) + 1)
        seg = _seg(x); _shape(seg)
        r = seg.rpf_sum('x')
        assert r is not None, extras
        assert r == (int(x.astype(object).sum()), x.size) == (_census(seg), x.size), extras


def test_the_switch():
    x = _col(300_000, [(i * 37, 1) for i in range(600)], 9)        # 600 repeats / 293 frames: past the line
    seg = _seg(x); _shape(seg)
    assert seg.rpf_sum('x') is None
    old = EN._RPF_MAX[0]
    try:
        EN._RPF_MAX[0] = float('inf')                                # forced open: still exact, every frame crowded
        assert seg.rpf_sum('x') == (_census(seg), x.size)
        EN._RPF_MAX[0] = old
        EN._RPF[0] = False
        assert seg.rpf_sum('x') is None
    finally:
        EN._RPF_MAX[0] = old; EN._RPF[0] = True


def test_sql_sum_and_affine():
    x = _col(300_000, [(3, 1), (4, 2)], 21)
    keep = {k: os.environ.get(k) for k in ENV}
    os.environ.update(ENV)
    try:
        d = os.path.join(TMP, 'rpfdb_%s' % uuid.uuid4().hex[:8]); os.makedirs(d)
        pq = os.path.join(d, 't.parquet'); pd.DataFrame({'x': x}).to_parquet(pq, index=False)
        db = Database.create(d); db.cat.add_table('t', [['x', 'int']])
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb')
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM read_parquet('%s')" % pq)
    # (AVG is left out here: with the suite's sidecars on, blockstats answers AVG of this table first, and its float
    # per-block sums round big int64 values -- 1493747350946502.2 for DuckDB's ...477.8, the same on the tree before
    # this fold. A separate open item; on the disk-only board AVG(WatchID) comes through this fold, exact.)
    for q in ("SELECT SUM(x) FROM t", "SELECT SUM(x), SUM(x + 1), SUM(x + 2), COUNT(*) FROM t"):
        h0 = EN._RPF_HITS
        g = db.run(q); g = g[0] if isinstance(g, tuple) else g
        e = con.execute(q).fetchall()
        assert controller._SERVED[0] == 'fused_agg' and EN._RPF_HITS > h0, (q, controller._SERVED[0])
        assert [tuple(float(v) if isinstance(v, float) else int(v) for v in r) for r in g] == \
               [tuple(float(v) if isinstance(v, float) else int(v) for v in r) for r in e], (q, g, e)
