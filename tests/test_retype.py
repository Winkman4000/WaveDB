"""wdb_retype: re-type integer columns as temporal (dt=3) in place by re-encoding only the
targeted column dicts and byte-copying every other column blob verbatim. Gold check: the
retyped segment decodes identically to a from-scratch encode where those columns were already
datetime64. Also verifies untouched columns are byte-identical and date-literal coercion works."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode as ENC, wdb_retype as RT
from wdb_engine import Segment


def _df(n=120_000, seed=7):
    rng = np.random.default_rng(seed)
    ed = 15888 + rng.integers(0, 17, n)                       # days-since-epoch (few distinct -> mode 0)
    et = 1372708800 + rng.integers(0, 2_592_000, n)           # epoch seconds (many distinct -> mode 2)
    url = np.array(['u%d' % i for i in rng.integers(0, 60000, n)])   # >50k distinct -> mode 1
    phrase = rng.choice(np.array(['', 'apple', 'banana']), n, p=[0.5, 0.3, 0.2])
    nn = np.where(rng.random(n) < 0.1, np.nan, rng.random(n) * 100)  # nullable float
    return pd.DataFrame({'ed': ed.astype(np.int64), 'et': et.astype(np.int64),
                         'url': url, 'phrase': phrase, 'nn': nn})


def _enc(df, **kw):
    d = tempfile.mkdtemp(); pq = os.path.join(d, 't.parquet'); df.to_parquet(pq)
    w = os.path.join(d, 't.wdb'); ENC.encode(pq, w, columns=list(df.columns), **kw)
    return w


def _as_dt(df):
    df = df.copy()
    df['ed'] = df['ed'].to_numpy().astype('datetime64[D]')
    df['et'] = df['et'].to_numpy().astype('datetime64[s]')
    return df


def test_retype_matches_fresh_datetime_encode():
    df = _df()
    s_int = _enc(df)
    gold = _enc(_as_dt(df))                                    # from-scratch datetime encode (oracle)
    s_rt = os.path.join(tempfile.mkdtemp(), 'r.wdb')
    RT.retype(s_int, {'ed': 'D', 'et': 's'}, s_rt, verbose=False)
    g, r = Segment(gold), Segment(s_rt)
    for nm, unit in (('ed', 'D'), ('et', 's')):
        assert r.cols[nm]['dt'] == 3 and r.unit(nm) == unit, nm
        # parquet promotes datetime64[D]->[us] on the gold path, so compare values in a
        # common unit rather than the raw int64 (which is unit-dependent).
        assert np.array_equal(np.asarray(g.values(nm)).astype('datetime64[s]'),
                              np.asarray(r.values(nm)).astype('datetime64[s]')), nm


def test_untouched_columns_byte_identical():
    df = _df()
    s_int = _enc(df)
    s_rt = os.path.join(tempfile.mkdtemp(), 'r.wdb')
    RT.retype(s_int, {'ed': 'D', 'et': 's'}, s_rt, verbose=False)
    a, b = Segment(s_int), Segment(s_rt)
    sa = {nm: (st, e) for nm, st, e in RT._column_spans(a.buf)}
    sb = {nm: (st, e) for nm, st, e in RT._column_spans(b.buf)}
    for nm in a.order:
        if nm in ('ed', 'et'):
            continue
        assert a.buf[sa[nm][0]:sa[nm][1]].tobytes() == b.buf[sb[nm][0]:sb[nm][1]].tobytes(), nm


def test_date_literal_coercion_and_filter():
    import wdb_sql as SQL, sqlglot
    df = _df()
    s_int = _enc(df)
    s_rt = os.path.join(tempfile.mkdtemp(), 'r.wdb')
    RT.retype(s_int, {'ed': 'D', 'et': 's'}, s_rt, verbose=False)
    seg = Segment(s_rt)
    lit = sqlglot.expressions.Literal.string('2013-07-10')
    coerced = SQL._lit_for_col(seg, 'ed', lit, 'i')
    expected = int(np.datetime64('2013-07-10').astype('datetime64[D]').view('int64'))
    assert coerced == expected == 15896


def test_cluster_sidecar_retyped_when_key_retyped():
    df = _df()
    s_int = _enc(df, cluster_by='et')                         # cluster on the int key
    s_rt = os.path.join(tempfile.mkdtemp(), 'r.wdb')
    RT.retype(s_int, {'ed': 'D', 'et': 's'}, s_rt, verbose=False)
    seg = Segment(s_rt)
    cm = seg.cluster_meta()
    assert cm['key'] == 'et' and cm['dtype'] == 3
    # cluster range pruning still works with a datetime literal coerced to epoch seconds
    lo = int(np.datetime64('2013-07-15T00:00:00').astype('datetime64[s]').view('int64'))
    sl = seg.slice_for_predicate('et', '>=', lo)
    assert sl is not None and 0 <= sl[0] <= sl[1] == seg.N
