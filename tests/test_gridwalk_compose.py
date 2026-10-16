"""gridwalk's top-K composed from the atoms (wdb_gridwalk._compose_top, 2026-10-15, the Q16 board): the dense
cells (radix_acc + bincount_par) when the possible pairs fit DENSE_MAX, the flat lane otherwise -- no structure
built. The answer must be IDENTICAL, row for row, to the structure's canonical prefix (count DESC, gid ASC),
plateaus included, and its counts must match a brute-force pandas group-by."""
import sys, os, uuid, tempfile, shutil, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from wdb_db import Database
import wdb_encode, wdb_gridwalk as GW, wdb_bound

_D = None


def _fixture():
    global _D
    if _D is not None:
        return _D
    rng = np.random.default_rng(11); N = 200_000
    df = pd.DataFrame({
        'bin':  rng.integers(0, 2, N).astype('int64'),                              # yes/no
        'tiny': rng.integers(0, 7, N).astype('int64'),
        'skew': np.floor(3000 * rng.random(N) ** 4).astype('int64'),                # a heavy head, long tail
        'wide': rng.integers(0, 60000, N).astype('int64'),                          # nearly every value once
        'word': rng.choice(['', 'a', 'b', 'cc', 'ddd'], N, p=[.8, .05, .05, .05, .05]),   # '' holds 80%
    })
    d = os.path.join(tempfile.gettempdir(), 'gwc_%s' % uuid.uuid4().hex[:8])
    db = Database.create(d)
    db.run("CREATE TABLE t (bin BIGINT, tiny BIGINT, skew BIGINT, wide BIGINT, word VARCHAR)")
    pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb')
    _D = (db, df, d)
    return _D


def _rows(res):
    return [tuple(r) for r in (res[0] if isinstance(res, tuple) else res)]


def _both(q):
    db, _df, _d = _fixture()
    GW._CACHE.clear()
    c0 = GW._COMPOSED; h0 = GW._HITS
    on = _rows(db.run(q)); composed = GW._COMPOSED > c0; hit = GW._HITS > h0
    GW._COMPOSE[0] = False
    try:
        GW._CACHE.clear()
        off = _rows(db.run(q))
    finally:
        GW._COMPOSE[0] = True
        GW._CACHE.clear()
    return on, off, composed, hit


def test_composed_equals_structure_every_pair():
    """every pair of the fixture, LIMIT 1 / 10 / 100: composed == the structure's canonical prefix, exactly"""
    db, df, _d = _fixture()
    n_comp = 0
    for a, b in itertools.permutations(['bin', 'tiny', 'skew', 'wide', 'word'], 2):
        for lim in (1, 10, 100):
            q = 'SELECT %s, %s, COUNT(*) FROM t GROUP BY %s, %s ORDER BY COUNT(*) DESC LIMIT %d' % (a, b, a, b, lim)
            on, off, composed, hit = _both(q)
            if hit:
                assert on == off, (a, b, lim, on[:3], off[:3])
                n_comp += composed
            want = sorted(df.groupby([a, b]).size().sort_values(ascending=False).head(lim).tolist())
            assert sorted(int(r[-1]) for r in on) == want, (a, b, lim)
    assert n_comp >= 30, n_comp                      # (pairs with 'word' go to tripletop)


def test_flat_lane_branch():
    """the possible pairs past DENSE_MAX: the flat lane counts only those that occur -- still identical"""
    old = wdb_bound.DENSE_MAX
    wdb_bound.DENSE_MAX = 1000
    try:
        for a, b in (('skew', 'wide'), ('wide', 'bin'), ('tiny', 'skew')):
            for lim in (10, 100):
                q = 'SELECT %s, %s, COUNT(*) FROM t GROUP BY %s, %s ORDER BY COUNT(*) DESC LIMIT %d' % (a, b, a, b, lim)
                on, off, composed, hit = _both(q)
                assert hit and composed, (a, b, lim)
                assert on == off, (a, b, lim, on[:3], off[:3])
    finally:
        wdb_bound.DENSE_MAX = old


def test_secondary_order_and_switch():
    """the stored tiebreak (keys ascending in name order) is honoured; WDB_GRIDWALK_COMPOSE's switch builds"""
    q = 'SELECT tiny, bin, COUNT(*) FROM t GROUP BY tiny, bin ORDER BY COUNT(*) DESC, bin, tiny LIMIT 5'
    on, off, composed, hit = _both(q)
    if hit:
        assert composed and on == off, (on, off)
    GW._COMPOSE[0] = False
    try:
        db, _df, _d = _fixture()
        c0 = GW._COMPOSED
        db.run('SELECT skew, tiny, COUNT(*) FROM t GROUP BY skew, tiny ORDER BY COUNT(*) DESC LIMIT 10')
        assert GW._COMPOSED == c0
    finally:
        GW._COMPOSE[0] = True
