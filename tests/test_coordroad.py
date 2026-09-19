"""THE COORDINATE ROAD: block-local positions with sparse/dense containers must answer the postings
of any key set exactly as the flat road does -- including keys that straddle blocks, keys dense
enough to become bitmaps, keys with one row, and keys absent from the road."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import wdb_coordroad as CR
from wdb_semijoin import _postings_kernel


def _flat(vals):
    order = np.argsort(vals, kind='stable').astype(np.int32)
    sv = vals[order]
    u, starts = np.unique(sv, return_index=True)
    offs = np.append(starts, sv.size).astype(np.int64)
    return u, offs, order


def _check(vals, sk):
    u, offs, order = _flat(vals)
    road = CR.build(u, order, offs)
    exp = _postings_kernel(u.astype(np.int64), offs, order, np.asarray(sk, dtype=np.int64))
    got = road.rows(np.asarray(sk, dtype=np.int64))
    assert np.array_equal(np.sort(got), np.sort(exp)), (got.size, exp.size)
    assert np.array_equal(got, exp), 'order within a key must be ascending, keys in sk order'
    return road, u, offs, order


def test_coordroad_scattered_and_clustered():
    rng = np.random.default_rng(7)
    N = 300_000
    vals = np.concatenate([rng.integers(0, 5000, N // 2), np.repeat(np.arange(50), N // 100)]).astype(np.int64)
    rng.shuffle(vals[: N // 2])                                   # first half scattered, second half clustered runs
    sk = np.unique(rng.choice(np.unique(vals), 800, replace=False))
    road, u, offs, order = _check(vals, sk)
    assert road.hdr.size > 0 and road.pay.dtype == np.uint16


def test_coordroad_dense_containers():
    N = 200_000
    vals = np.zeros(N, np.int64); vals[::3] = 1; vals[::7] = 2       # value 0 fills most of every block: bitmaps
    road, u, offs, order = _check(vals, np.array([0, 1, 2, 9]))
    dense = ((road.hdr >> 15) & 1).sum()
    assert dense > 0, 'a key with more than 4096 rows in a block must be a bitmap'
    assert road.disk_bytes < CR.flat_bytes(order, offs) + u.nbytes


def test_coordroad_missing_and_singletons():
    vals = np.arange(70_000, dtype=np.int64)                        # every key has one row, straddling one block boundary
    _check(vals, np.array([0, 65535, 65536, 69999, 70000, 123456]))


def test_coordroad_plan_rule():
    rng = np.random.default_rng(3)
    scattered = rng.integers(0, 200_000, 2_000_000).astype(np.int64)      # ~10 rows per key, each in its own block: coordinates lose
    u, offs, order = _flat(scattered)
    assert CR.plan(order, offs)[-1] > CR.flat_bytes(order, offs)
    clustered = np.repeat(np.arange(2000), 1000).astype(np.int64)         # runs: coordinates win
    u, offs, order = _flat(clustered)
    assert CR.plan(order, offs)[-1] < CR.flat_bytes(order, offs) / 1.8   # 16-bit positions: ~2x; bitmaps go further


def test_coordroad_save_load_roundtrip():
    rng = np.random.default_rng(11)
    vals = rng.integers(0, 300, 100_000).astype(np.int64)
    u, offs, order = _flat(vals)
    road = CR.build(u, order, offs)
    d = os.path.join(tempfile.gettempdir(), 'coord_%s' % uuid.uuid4().hex[:8]); os.makedirs(d)
    fn = os.path.join(d, 't_0.wdb.k.inv')
    CR.save(fn, road)
    back = CR.load(fn)
    sk = np.array([0, 5, 299, 300])
    assert np.array_equal(back.rows(sk), road.rows(sk))
    assert sorted(os.listdir(d)) == sorted('t_0.wdb.k.inv.%s.npy' % s for s in CR.SUFFIXES)
