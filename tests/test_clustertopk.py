"""Tests for wdb_clustertopk -- cluster-ordered projection top-K.

When a segment is clustered by the primary ORDER BY column, SELECT cols [WHERE c<>'']
ORDER BY clusterkey [DESC][, sec] LIMIT k is answered by walking the clustered end and
stopping at the limit (the row-order twin of the value-sorted dict read). detect is a
pure state-gate: primary ORDER BY col must equal seg.cluster_meta()['key']. All
synthetic + tiny; pandas is the oracle. Compound/DESC use ts+phrase so the order is
fully determined (no tie ambiguity)."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode, wdb_clustertopk as CTK
from wdb_db import Database


def _clustered_db(df, key):
    d = tempfile.mkdtemp(); db = Database.create(d)
    cols = [[c, ('string' if df[c].dtype.kind == 'O'
                 else 'datetime' if df[c].dtype.kind == 'M'
                 else 'float' if df[c].dtype.kind == 'f' else 'int')]
            for c in df.columns]
    db.cat.add_table('t', cols)
    pq = os.path.join(d, 't.parquet'); df.to_parquet(pq)
    seg = 't_0.wdb'; wdb_encode.encode(pq, os.path.join(d, seg), cluster_by=key)
    db.cat.add_segment('t', seg)
    return db


def _df(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    ts = (np.datetime64('2013-07-01') + rng.integers(0, 60, n).astype('timedelta64[D]'))
    vocab = np.array(['', 'apple', 'banana', 'cherry', 'date', 'fig', 'grape', 'kiwi'])
    phrase = rng.choice(vocab, n, p=[0.40, 0.10, 0.10, 0.10, 0.08, 0.06, 0.06, 0.10])
    return pd.DataFrame({'ts': ts, 'phrase': phrase, 'uid': rng.integers(0, 1000, n)})


def _dec(rows):
    return [tuple(x.decode() if isinstance(x, (bytes, bytearray)) else x for x in r) for r in rows]


def _oracle(df, order_cols, asc, lim, where_nonempty=True):
    d = df[df.phrase != ''] if where_nonempty else df
    d = d.sort_values(order_cols, ascending=asc, kind='mergesort').head(lim)
    return d


def test_q24_asc_fires_and_matches():
    df = _df(); db = _clustered_db(df, 'ts')
    h0 = CTK._HITS
    rows, _ = db.run("SELECT phrase FROM t WHERE phrase <> '' ORDER BY ts, phrase LIMIT 10")
    assert CTK._HITS == h0 + 1                                   # cluster_topk fired
    exp = [(p,) for p in _oracle(df, ['ts', 'phrase'], [True, True], 10).phrase.tolist()]
    assert _dec(rows) == exp


def test_q24_desc_fires_and_matches():
    df = _df(); db = _clustered_db(df, 'ts')
    h0 = CTK._HITS
    rows, _ = db.run("SELECT phrase FROM t WHERE phrase <> '' ORDER BY ts DESC, phrase LIMIT 10")
    assert CTK._HITS == h0 + 1
    exp = [(p,) for p in _oracle(df, ['ts', 'phrase'], [False, True], 10).phrase.tolist()]
    assert _dec(rows) == exp


def test_limit_spans_multiple_key_values():
    df = _df(); db = _clustered_db(df, 'ts')
    rows, _ = db.run("SELECT phrase FROM t WHERE phrase <> '' ORDER BY ts, phrase LIMIT 50")
    exp = [(p,) for p in _oracle(df, ['ts', 'phrase'], [True, True], 50).phrase.tolist()]
    assert _dec(rows) == exp


def test_multi_projection():
    df = _df(); db = _clustered_db(df, 'ts')
    h0 = CTK._HITS
    rows, _ = db.run("SELECT phrase, uid FROM t WHERE phrase <> '' ORDER BY ts, phrase, uid LIMIT 12")
    assert CTK._HITS == h0 + 1
    o = _oracle(df, ['ts', 'phrase', 'uid'], [True, True, True], 12)
    exp = list(zip(o.phrase.tolist(), o.uid.tolist()))
    assert _dec(rows) == exp


def test_no_where_clause():
    df = _df(); db = _clustered_db(df, 'ts')
    h0 = CTK._HITS
    rows, _ = db.run("SELECT phrase FROM t ORDER BY ts, phrase LIMIT 10")
    assert CTK._HITS == h0 + 1
    exp = [(p,) for p in _oracle(df, ['ts', 'phrase'], [True, True], 10, where_nonempty=False).phrase.tolist()]
    assert _dec(rows) == exp


def test_declines_order_not_cluster_key():
    df = _df(); db = _clustered_db(df, 'ts')
    h0 = CTK._HITS
    rows, _ = db.run("SELECT phrase FROM t WHERE phrase <> '' ORDER BY phrase LIMIT 10")
    assert CTK._HITS == h0                                       # NOT taken (order != cluster key)
    # still correct via the general path
    exp = [(p,) for p in _oracle(df, ['phrase'], [True], 10).phrase.tolist()]
    assert sorted(_dec(rows)) == sorted(exp)


def test_declines_no_limit():
    import sqlglot
    from wdb_engine import Segment
    df = _df()
    d = tempfile.mkdtemp(); pq = os.path.join(d, 't.parquet'); df.to_parquet(pq)
    w = os.path.join(d, 't_0.wdb'); wdb_encode.encode(pq, w, cluster_by='ts')
    seg = Segment(w)
    tree = sqlglot.parse_one("SELECT phrase FROM t WHERE phrase <> '' ORDER BY ts", read='duckdb')
    assert CTK.detect(seg, tree, {}) is None                   # no LIMIT -> declines


def test_declines_group_by():
    import sqlglot
    from wdb_engine import Segment
    df = _df()
    d = tempfile.mkdtemp(); pq = os.path.join(d, 't.parquet'); df.to_parquet(pq)
    w = os.path.join(d, 't_0.wdb'); wdb_encode.encode(pq, w, cluster_by='ts')
    seg = Segment(w)
    tree = sqlglot.parse_one("SELECT phrase, COUNT(*) FROM t GROUP BY phrase ORDER BY ts LIMIT 10",
                             read='duckdb')
    assert CTK.detect(seg, tree, {}) is None                   # GROUP BY -> declines


def test_declines_when_not_clustered():
    import sqlglot
    from wdb_engine import Segment
    df = _df()
    d = tempfile.mkdtemp()
    pq = os.path.join(d, 't.parquet'); df.to_parquet(pq)
    w = os.path.join(d, 't_0.wdb'); wdb_encode.encode(pq, w)   # NOT clustered -> no .cluster sidecar
    seg = Segment(w)
    tree = sqlglot.parse_one("SELECT phrase FROM t WHERE phrase <> '' ORDER BY ts, phrase LIMIT 10",
                             read='duckdb')
    assert seg.cluster_meta() is None                          # state: not clustered
    assert CTK.detect(seg, tree, {}) is None                   # so the state-gate declines
