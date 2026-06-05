"""Tests for clustered encode (wdb_encode cluster_by=).
Clustering is a result-preserving row permutation + a .cluster slice-boundary sidecar.
All synthetic + tiny: runs in the suite with no bench DB."""
import sys, os, tempfile, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode
from wdb_engine import Segment


def _enc(df, cluster_by=None):
    d = tempfile.mkdtemp(); p = os.path.join(d, 't.parquet'); df.to_parquet(p)
    w = os.path.join(d, 't.wdb'); wdb_encode.encode(p, w, cluster_by=cluster_by)
    return w


def test_clustered_encode_preserves_data():
    df = pd.DataFrame({'k': [5, 1, 3, 1, 5, 2], 'v': [10, 20, 30, 40, 50, 60]})
    s0 = Segment(_enc(df)); s1 = Segment(_enc(df, cluster_by='k'))
    r0 = sorted(zip(s0.values('k').tolist(), s0.values('v').tolist()))
    r1 = sorted(zip(s1.values('k').tolist(), s1.values('v').tolist()))
    assert r0 == r1                                   # same multiset of rows


def test_clustered_key_is_physically_sorted():
    df = pd.DataFrame({'k': [5, 1, 3, 1, 5, 2], 'v': [10, 20, 30, 40, 50, 60]})
    k1 = Segment(_enc(df, cluster_by='k')).values('k').tolist()
    assert k1 == sorted(k1)                           # rows now in cluster-key order


def test_cluster_sidecar_boundaries_locate_a_slice():
    df = pd.DataFrame({'k': np.arange(100) % 10, 'v': np.arange(100)})
    w = _enc(df, cluster_by='k')
    cl = pickle.load(open(w + '.cluster', 'rb'))
    assert cl['key'] == 'k' and cl['nn'] == 100
    vals = np.asarray(cl['values']); off = np.asarray(cl['offsets'])
    lo = off[np.searchsorted(vals, 3, 'left')]; hi = off[np.searchsorted(vals, 6, 'left')]
    k = Segment(w).values('k')[lo:hi]                 # the slice the executor will read
    assert len(k) == 30 and set(k.tolist()) == {3, 4, 5}


def test_cluster_boundaries_datetime():
    days = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.arange(50) % 5, unit='D')
    df = pd.DataFrame({'d': days.values, 'v': np.arange(50)})
    w = _enc(df, cluster_by='d')
    cl = pickle.load(open(w + '.cluster', 'rb'))
    assert cl['dtype'] == 3 and cl['nn'] == 50
    d = Segment(w).values('d')
    assert list(d.view('int64')) == sorted(d.view('int64').tolist())


def test_default_encode_writes_no_cluster_sidecar():
    df = pd.DataFrame({'k': [3, 1, 2], 'v': [1, 2, 3]})
    w = _enc(df)                                      # cluster_by=None
    assert not os.path.exists(w + '.cluster')
    assert sorted(Segment(w).values('v').tolist()) == [1, 2, 3]


# --- step 2: cluster-key slice primitives on the Segment ---------------------------------

def _seg_clustered():
    n = 500
    df = pd.DataFrame({'k': np.arange(n) % 20, 'big': np.arange(n), 'small': np.arange(n) % 7})
    return Segment(_enc(df, cluster_by='k'))

def test_raw_codes_range_matches_full_decode():
    s = _seg_clustered()
    for nm in ('big', 'small', 'k'):
        full = s.codes(nm)
        for lo, hi in [(0, 1), (5, 37), (0, len(full)), (100, 100), (255, 499)]:
            assert np.array_equal(s._raw_codes_range(nm, lo, hi), full[lo:hi]), (nm, lo, hi)

def test_values_range_matches_full_decode():
    s = _seg_clustered()
    for nm in ('big', 'small'):
        full = s.values(nm)
        for lo, hi in [(0, 3), (5, 37), (128, 256), (0, len(full))]:
            assert np.array_equal(s.values_range(nm, lo, hi), full[lo:hi]), (nm, lo, hi)

def test_slice_for_predicate_matches_bruteforce():
    s = _seg_clustered(); k = s.values('k')
    cases = [('=', 7), ('=', 999), ('>', 7), ('>=', 7), ('<', 7), ('<=', 7), ('>', 19), ('<', 0)]
    for op, thr in cases:
        lo, hi = s.slice_for_predicate('k', op, thr)
        got = set(range(lo, hi))
        m = {'=': k == thr, '>': k > thr, '>=': k >= thr, '<': k < thr, '<=': k <= thr}[op]
        assert got == set(np.where(m)[0].tolist()), (op, thr, lo, hi)

def test_slice_endtoend_sum_equals_full():
    s = _seg_clustered(); k = s.values('k'); big = s.values('big')
    lo, hi = s.slice_for_predicate('k', '>', 12)
    assert float(s.values_range('big', lo, hi).sum()) == float(big[k > 12].sum())

def test_slice_for_predicate_none_when_not_cluster_key():
    s = _seg_clustered()
    assert s.slice_for_predicate('big', '>', 5) is None      # not the cluster key

def test_slice_datetime_key():
    days = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.arange(200) % 10, unit='D')
    df = pd.DataFrame({'d': days.values, 'v': np.arange(200)})
    s = Segment(_enc(df, cluster_by='d')); unit = s.unit('d')
    thr = np.datetime64('2024-01-04').astype(f'datetime64[{unit}]').view('int64').item()
    lo, hi = s.slice_for_predicate('d', '>=', thr)
    d = s.values('d')
    assert set(range(lo, hi)) == set(np.where(d >= np.datetime64('2024-01-04'))[0].tolist())


# --- step 2b: wired slice fast-path in the executor (clustered == unclustered) -----------
import wdb_sql

def _pair(df, key):
    return Segment(_enc(df)), Segment(_enc(df, cluster_by=key))

def test_executor_slice_matches_full_path():
    n = 400
    df = pd.DataFrame({'k': np.arange(n) % 25, 'v': (np.arange(n) * 7) % 1000, 'm': np.arange(n) % 5})
    full, clus = _pair(df, 'k')
    queries = [
        "SELECT COUNT(*) FROM t WHERE k > 10",
        "SELECT SUM(v) FROM t WHERE k > 10",
        "SELECT SUM(v) FROM t WHERE k >= 5 AND k < 15",
        "SELECT SUM(v) FROM t WHERE k > 5 AND m < 3",
        "SELECT COUNT(*), SUM(v), MIN(v), MAX(v) FROM t WHERE k BETWEEN 4 AND 8",
        "SELECT SUM(v) FROM t WHERE k = 7",
        "SELECT COUNT(DISTINCT m) FROM t WHERE k > 3",
        "SELECT SUM(v) FROM t WHERE k > 100",          # empty slice
    ]
    before = wdb_sql._SLICE_HITS
    for q in queries:
        r_full, _ = wdb_sql.execute(full, q)
        r_clus, _ = wdb_sql.execute(clus, q)
        assert r_full == r_clus, (q, r_full, r_clus)
    assert wdb_sql._SLICE_HITS == before + len(queries)   # every one took the slice path


def test_executor_no_key_predicate_falls_back():
    df = pd.DataFrame({'k': np.arange(200) % 10, 'v': np.arange(200), 'm': np.arange(200) % 4})
    full, clus = _pair(df, 'k')
    before = wdb_sql._SLICE_HITS
    q = "SELECT SUM(v) FROM t WHERE m < 2"             # no cluster-key predicate
    assert wdb_sql.execute(full, q) == wdb_sql.execute(clus, q)
    assert wdb_sql._SLICE_HITS == before               # slice path NOT taken


def test_executor_slice_datetime_range():
    days = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.arange(300) % 12, unit='D')
    df = pd.DataFrame({'d': days.values, 'v': (np.arange(300) * 3) % 500})
    full, clus = _pair(df, 'd')
    q = ("SELECT SUM(v), COUNT(*) FROM t "
         "WHERE d >= '2024-01-04' AND d < '2024-01-09'")
    before = wdb_sql._SLICE_HITS
    assert wdb_sql.execute(full, q) == wdb_sql.execute(clus, q)
    assert wdb_sql._SLICE_HITS == before + 1


def test_executor_slice_float_sum_isclose():
    import math
    n = 300
    df = pd.DataFrame({'k': np.arange(n) % 20, 'p': np.round(np.arange(n) * 1.07, 2)})
    full, clus = _pair(df, 'k')
    q = "SELECT SUM(p) FROM t WHERE k > 8"
    a = wdb_sql.execute(full, q)[0][0][0]; b = wdb_sql.execute(clus, q)[0][0][0]
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)
