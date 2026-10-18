"""gridwalk's GRAB (2026-10-18, the Q14 board): an axis filter keeping at most GRAB_MAX rows is answered by finding
those rows and reading the other key at them only. The answer must be IDENTICAL to the remove lane's (count every
row, drop the filtered cells) -- the canonical top, plateaus included -- and its counts must match pandas; a
one-value key filtered '<>' its value keeps nothing."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from wdb_db import Database
import wdb_encode, wdb_join, wdb_qmem, wdb_gridwalk as GW

_D = None


def _fixture():
    global _D
    if _D is not None:
        return _D
    rng = np.random.default_rng(23); N = 300_000
    rare = np.zeros(N, np.int64); ix = rng.choice(N, 900, replace=False); rare[ix] = rng.integers(1, 40, 900)
    df = pd.DataFrame({
        'tiny': rng.integers(0, 7, N).astype('int64'),
        'skew': np.floor(3000 * rng.random(N) ** 4).astype('int64'),
        'wide': rng.integers(0, 60000, N).astype('int64'),
        'rare': rare,                                                          # 0 holds 99.7%
        'one': np.full(N, 7, np.int64),                                        # a single value
        'word': rng.choice(['', 'a', 'b', 'cc'], N, p=[.97, .01, .01, .01]),
    })
    d = os.path.join(tempfile.gettempdir(), 'gwg_%s' % uuid.uuid4().hex[:8])
    db = Database.create(d)
    db.run("CREATE TABLE t (tiny BIGINT, skew BIGINT, wide BIGINT, rare BIGINT, one BIGINT, word VARCHAR)")
    pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
    # production encodings (as test_bound_hunt): the suite's runner lets toys take mode 4, which sends these
    # filters to compound_filter's mode-4 path instead of gridwalk
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None: os.environ[k] = v
    db.cat.add_segment('t', 't_0.wdb')
    seg, _ = wdb_join._solo_segment(db, 't')
    _D = (db, df, seg)
    return _D


def _lanes(cols, af):
    db, df, seg = _fixture()
    out = []
    for gmax in (-1, int(seg.N)):
        GW._GRAB_MAX[0] = gmax
        wdb_qmem.flush(db)
        out.append(GW._compose_top(seg, cols, 10, af))
    GW._GRAB_MAX[0] = 1 << 20
    return out


def test_grab_equals_remove():
    db, df, seg = _fixture()
    n = 0
    for f, o in (('rare', 'wide'), ('rare', 'tiny'), ('skew', 'wide'), ('tiny', 'skew'), ('word', 'wide'), ('wide', 'tiny')):
        codes = np.asarray(seg._raw_codes(f)); cnt = np.bincount(codes)
        nz = np.argsort(-cnt)
        for code in (int(nz[0]), int(nz[1]), int(nz[min(5, nz.size - 1)])):
            for neg in (False, True):
                rem, grab = _lanes(sorted([f, o]), (f, neg, code))
                if rem is None or grab is None:
                    continue
                n += 1
                assert np.array_equal(rem[0], grab[0]) and np.array_equal(rem[1], grab[1]), (f, o, code, neg)
    assert n >= 20, n


def test_counts_match_pandas():
    db, df, seg = _fixture()
    for where, a, b in (("rare <> 0", 'rare', 'wide'), ("word <> ''", 'word', 'skew'), ("rare = 3", 'rare', 'tiny')):
        q = 'SELECT %s, %s, COUNT(*) AS c FROM t WHERE %s GROUP BY %s, %s ORDER BY c DESC LIMIT 10' % (a, b, where, a, b)
        res = db.run(q); rows = res[0] if isinstance(res, tuple) else res
        sub = df.query(where.replace("<> ''", "!= ''").replace('<>', '!=').replace(' = ', ' == '))
        want = sorted(sub.groupby([a, b]).size().sort_values(ascending=False).head(10).tolist())
        assert sorted(int(r[-1]) for r in rows) == want, (q, rows[:3])


def test_one_value_key_keeps_nothing():
    db, df, seg = _fixture()
    q = 'SELECT one, tiny, COUNT(*) AS c FROM t WHERE one <> 7 GROUP BY one, tiny ORDER BY c DESC LIMIT 10'
    res = db.run(q); rows = res[0] if isinstance(res, tuple) else res
    assert len(rows) == 0, rows
