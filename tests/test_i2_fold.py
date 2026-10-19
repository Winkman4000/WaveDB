"""THE FOLD THROUGH THE DICTIONARY'S OWN FRAMES (wdb_kernels.i2_fold, 2026-10-19): an exact SUM/AVG over
a chunked integer dictionary (mode 2) folds counts x values chunk by chunk as libzstd inflates the
frames, with no spine built. Small chunks are forced at encode (WDB_I2CHUNK / WDB_I2CHUNK_MIN), with a
partial last chunk; values span the int64 range (two limbs, negatives) and repeat (counts above one).
Checked at the segment (against python-int arithmetic, any counts, a short counts array declines) and
through SQL (the scalar path takes the fold; oracle DuckDB, SUM exact as HUGEINT; same answers with the
fold off)."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import duckdb, wdb_encode, wdb_engine
from wdb_db import Database
from wdb_engine import Segment

_FIX = None


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    rng = np.random.default_rng(11)
    ux = np.unique(rng.integers(-(1 << 62), 1 << 62, 90721, dtype=np.int64))      # past NUM_THRESHOLD: mode 2
    uy = np.unique(rng.integers(-(1 << 40), 1 << 40, 70001, dtype=np.int64))
    n = 200000
    df = pd.DataFrame({'x': ux[rng.integers(0, ux.size, n)], 'y': uy[rng.integers(0, uy.size, n)],
                       'g': rng.integers(0, 7, n).astype(np.int64)})
    d = os.path.join(tempfile.gettempdir(), 'i2f_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet'); df.to_parquet(pq, index=False)
    dbs = {}
    for ch in (1000, 8192):
        dd = os.path.join(d, 'db%d' % ch)
        db = Database.create(dd)
        db.cat.add_table('t', [['x', 'int'], ['y', 'int'], ['g', 'int']])
        old = {k: os.environ.get(k) for k in ('WDB_I2CHUNK', 'WDB_I2CHUNK_MIN')}
        os.environ['WDB_I2CHUNK'] = str(ch); os.environ['WDB_I2CHUNK_MIN'] = '1000'
        try:
            wdb_encode.encode(pq, os.path.join(dd, 't_0.wdb'))
        finally:
            for k, v in old.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
        db.cat.add_segment('t', 't_0.wdb')
        dbs[ch] = db
    con = duckdb.connect(); con.register('df', df); con.execute('CREATE TABLE t AS SELECT * FROM df')
    _FIX = (dbs, con, df, d)
    return _FIX


def _seg(db):
    return db.open_segment(db.cat.segment_paths('t')[0], 't')


def test_segment_fold_is_exact_for_any_counts():
    if wdb_engine._libzstd() is None:
        return                                      # no libzstd on this machine: the spine road answers
    dbs, _con, _df, _d = _fixture()
    rng = np.random.default_rng(5)
    for ch, db in dbs.items():
        for col in ('x', 'y'):
            seg = _seg(db); c = seg.cols[col]
            assert c['mode'] == 2 and c.get('i2ch') is not None and int(c['i2ch']) == ch, (ch, col, c['mode'])
            nd = int(c['n_dict'])
            assert seg.i2_foldable(col)
            s2 = Segment(seg.path)
            vals = np.asarray(s2._dict_ints(s2.cols[col]))[:nd]
            # counts total within the two-limb bound (rows up to ~4e9, wdb_exactint); the last code's count
            # is the largest so the partial last chunk weighs in
            heavy = np.ones(nd, np.int64); heavy[-1] = 1 << 30
            for k, cnt in enumerate((np.ones(nd, np.int64), rng.integers(0, 1000, nd).astype(np.int64),
                                     heavy, np.zeros(nd + 3, np.int64))):
                got = seg.i2_fold_sum(col, cnt)
                want = sum(int(a) * int(b) for a, b in zip(cnt[:nd].tolist(), vals.tolist()))
                assert got == want, (ch, col, k, got, want)
            assert seg.i2_fold_sum(col, np.ones(nd - 1, np.int64)) is None      # too few counts: decline
            seg._dict_ints(c)
            assert not seg.i2_foldable(col)                                     # spine inflated: the cheaper road


def test_sql_takes_the_fold_and_matches_duckdb():
    if wdb_engine._libzstd() is None:
        return
    dbs, con, _df, _d = _fixture()
    sqls = ["SELECT SUM(x) FROM t", "SELECT AVG(x) FROM t", "SELECT SUM(y), AVG(y), COUNT(*) FROM t",
            "SELECT SUM(x), AVG(x), MIN(x), MAX(x), COUNT(x), SUM(y) FROM t"]
    for ch, db in dbs.items():
        calls = []
        orig = Segment.i2_fold_sum
        def spy(self, nm, counts):
            r = orig(self, nm, counts); calls.append((nm, r is not None)); return r
        Segment.i2_fold_sum = spy
        try:
            got = [db.run(q)[0] for q in sqls]
        finally:
            Segment.i2_fold_sum = orig
        assert calls and all(ok for _nm, ok in calls), calls
        for q, g in zip(sqls, got):
            want = con.execute(q).fetchall()
            assert len(g) == 1 and len(g[0]) == len(want[0]), (q, g, want)
            for a, b in zip(g[0], want[0]):
                if isinstance(b, float):
                    assert abs(float(a) - b) <= 1e-12 * max(1.0, abs(b)), (ch, q, a, b)
                else:
                    assert int(a) == int(b), (ch, q, a, b)
        wdb_engine._I2FOLD[0] = False
        try:
            for q, g in zip(sqls, got):
                assert db.run(q)[0] == g, (ch, q)
        finally:
            wdb_engine._I2FOLD[0] = True


if __name__ == '__main__':
    try:
        test_segment_fold_is_exact_for_any_counts()
        test_sql_takes_the_fold_and_matches_duckdb()
        print('ok')
    finally:
        if _FIX is not None:
            shutil.rmtree(_FIX[3], ignore_errors=True)
