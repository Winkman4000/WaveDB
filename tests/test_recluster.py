"""wdb_recluster: reorder an existing segment by a key by permuting codes + reusing dicts,
instead of re-encoding from source. The gold check: the reclustered segment must decode
identically to encode(cluster_by=key) built from scratch. Synthetic; forces a mode-1
(front-coded) string column and a nullable column."""
import sys, os, tempfile, struct, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode as ENC, wdb_recluster as RC
from wdb_engine import Segment


def _df(n=120_000, seed=4):
    rng = np.random.default_rng(seed)
    ts = (np.datetime64('2013-07-01') + rng.integers(0, 4000, n).astype('timedelta64[s]'))
    url = np.array(['u%d' % i for i in rng.integers(0, 60000, n)])     # >50k distinct -> mode 1
    phrase = rng.choice(np.array(['', 'apple', 'banana', 'cherry']), n, p=[0.4, 0.3, 0.2, 0.1])
    uid = rng.integers(0, 500_000, n)
    nn = np.where(rng.random(n) < 0.1, np.nan, rng.random(n) * 100)    # nullable float
    return pd.DataFrame({'ts': ts, 'url': url, 'phrase': phrase, 'uid': uid, 'nn': nn})


def _enc(df, **kw):
    d = tempfile.mkdtemp(); pq = os.path.join(d, 't.parquet'); df.to_parquet(pq)
    w = os.path.join(d, 't.wdb'); ENC.encode(pq, w, columns=list(df.columns), **kw)
    return w


def _eqcol(a, b):
    if a.dtype.kind == 'f' or b.dtype.kind == 'f':
        return np.allclose(a.astype(float), b.astype(float), equal_nan=True)
    return np.array_equal(a, b)


def test_recluster_matches_fresh_cluster_encode():
    df = _df()
    s0 = _enc(df)                                   # unclustered
    gold = _enc(df, cluster_by='ts')                # from-scratch clustered (the oracle)
    s1 = os.path.join(tempfile.mkdtemp(), 'r.wdb')
    RC.recluster(s0, 'ts', s1)                      # reorder the existing segment
    g, r = Segment(gold), Segment(s1)
    assert list(g.cols) == list(r.cols)
    for nm in g.cols:
        assert _eqcol(g.values(nm), r.values(nm)), nm        # identical decoded content
    gc, rc = g.cluster_meta(), r.cluster_meta()
    assert gc['key'] == rc['key'] == 'ts' and gc['nn'] == rc['nn']
    assert np.array_equal(np.asarray(gc['offsets']), np.asarray(rc['offsets']))


def test_recluster_is_permutation_of_original():
    df = _df()
    s0 = _enc(df); seg0 = Segment(s0)
    order = np.argsort(seg0.values('ts').view('int64'), kind='stable')
    s1 = os.path.join(tempfile.mkdtemp(), 'r.wdb'); RC.recluster(s0, 'ts', s1); seg1 = Segment(s1)
    for nm in seg0.cols:
        assert _eqcol(seg0.values(nm)[order], seg1.values(nm)), nm


def test_nullable_column_survives():
    df = _df()
    s0 = _enc(df); s1 = os.path.join(tempfile.mkdtemp(), 'r.wdb'); RC.recluster(s0, 'ts', s1)
    seg0, seg1 = Segment(s0), Segment(s1)
    a = seg0.values('nn'); b = seg1.values('nn')
    assert np.isnan(a.astype(float)).sum() == np.isnan(b.astype(float)).sum() > 0   # nulls preserved


def test_cluster_topk_fires_on_reclustered():
    import wdb_clustertopk as CTK, sqlglot
    df = _df()
    s0 = _enc(df); s1 = os.path.join(tempfile.mkdtemp(), 'r.wdb'); RC.recluster(s0, 'ts', s1)
    seg1 = Segment(s1)
    tree = sqlglot.parse_one("SELECT phrase FROM t WHERE phrase <> '' ORDER BY ts, phrase LIMIT 10",
                             read='duckdb')
    spec = CTK.detect(seg1, tree, {})
    assert spec is not None
    rows, _ = CTK.execute(seg1, spec)
    got = [r[0].decode() if isinstance(r[0], bytes) else r[0] for r in rows]
    exp = df[df.phrase != ''].sort_values(['ts', 'phrase'], kind='mergesort').head(10).phrase.tolist()
    assert got == exp
