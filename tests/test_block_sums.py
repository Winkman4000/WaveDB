"""THE EXACT BLOCK SUMS (2026-11-08): the load writes each block's exact integer sum (two int64 halves) for every
integer column; SUM / AVG with no WHERE and no GROUP BY add a few thousand numbers. SUM is exact past int64 (a
column of values near 2^62), AVG is the exact sum over the count, correctly rounded -- DuckDB's to the bit; NULLs
stay out of both; SUM(col + k) is S + k * N; WDB_BLOCK_SUMS off reads none and another read answers the same.
MIN / MAX of a dictionary column are its dictionary's two ends (no statistics read)."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
_FX = None
N = 200_000


def _fixture():
    global _FX
    if _FX is not None:
        return _FX
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq, duckdb
    import wdb_encode
    from wdb_db import Database
    rng = np.random.default_rng(8)
    big = rng.integers(-(2 ** 62), 2 ** 62, N, dtype=np.int64)        # sums far past int64, both signs
    big[:N // 2] = rng.integers(2 ** 61, 2 ** 62, N // 2, dtype=np.int64)
    small = rng.integers(0, 3000, N).astype(np.int64)                  # a narrow dictionary
    neg = rng.integers(-50, 50, N).astype(np.int64)
    nul = rng.integers(0, 1000, N).astype(np.int64)
    mask = rng.random(N) < 0.3                                         # 30% NULL
    dd = np.datetime64('2013-07-01') + rng.integers(0, 31, N).astype('timedelta64[D]')
    d = os.path.join(tempfile.gettempdir(), 'bsum_%s' % uuid.uuid4().hex[:8]); os.makedirs(d)
    try:
        import helpers as _H; _H.register_dir(d)
    except Exception:
        pass
    p = os.path.join(d, 't.parquet')
    pq.write_table(pa.table({'big': big, 'small': small, 'neg': neg,
                             'nul': pa.array(nul, mask=mask), 'id': np.arange(N, dtype=np.int64),
                             'dd': pa.array(dd.astype('datetime64[D]'), pa.date32())}), p)
    db = Database.create(d)
    db.cat.add_table('t', [['big', 'int'], ['small', 'int'], ['neg', 'int'], ['nul', 'int'], ['id', 'int'],
                           ['dd', 'date']])
    wdb_encode.encode(p, os.path.join(d, 't_0.wdb'))
    db.cat.add_segment('t', 't_0.wdb')
    con = duckdb.connect(); con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('%s')" % p)
    _FX = (db, con, d)
    return _FX


def _rows(r):
    return [tuple(r9) for r9 in (r[0] if isinstance(r, tuple) else r)]


def test_the_load_wrote_them():
    import numpy as np, wdb_join, wdb_blockstats as B
    db, con, d = _fixture()
    seg, _x = wdb_join._solo_segment(db, 't')
    z = np.load(B.stats_path(seg.path))
    for col in ('big', 'small', 'neg', 'nul'):
        assert col + '.shi' in z.files and col + '.slo' in z.files, col


def test_sum_avg_are_duckdbs_to_the_bit():
    import controller
    db, con, d = _fixture()
    for col in ('big', 'small', 'neg', 'nul', 'id'):
        q = 'SELECT SUM(%s), AVG(%s), COUNT(%s), COUNT(*) FROM t' % (col, col, col)
        got = _rows(db.run(q))
        want = con.execute(q.replace('SUM(%s)' % col, 'SUM(%s)::HUGEINT' % col)).fetchall()
        assert controller._SERVED[0] == 'blockstats', (q, controller._SERVED[0])
        g, w = got[0], want[0]
        assert int(g[0]) == int(w[0]) and float(g[1]) == float(w[1]) and int(g[2]) == int(w[2]) \
            and int(g[3]) == int(w[3]), (q, g, w)


def test_affine_sums():
    import controller
    db, con, d = _fixture()
    for col in ('big', 'small', 'neg'):
        q = 'SELECT SUM(%s), SUM(%s + 1), SUM(%s + 7), SUM(3 + %s) FROM t' % (col, col, col, col)
        got = [int(v) for v in _rows(db.run(q))[0]]
        assert controller._SERVED[0] == 'affinesum', (q, controller._SERVED[0])
        h = 'CAST(%s AS HUGEINT)' % col
        dq = 'SELECT SUM(%s), SUM(%s + 1), SUM(%s + 7), SUM(3 + %s) FROM t' % (h, h, h, h)
        want = [int(v) for v in con.execute(dq).fetchall()[0]]
        assert got == want, (q, got, want)


def test_min_max_are_the_dictionary_ends():
    import controller, wdb_blockstats as B
    db, con, d = _fixture()
    build = B.build
    def _no_stats(seg, col):
        if seg.cols[col].get('mode') in (0, 2):
            raise AssertionError('MIN/MAX read the statistics of %s' % col)
        return build(seg, col)
    B.build = _no_stats
    try:
        for col in ('big', 'small', 'neg', 'nul', 'id', 'dd'):
            q = 'SELECT MIN(%s), MAX(%s) FROM t' % (col, col)
            got = _rows(db.run(q))[0]
            assert controller._SERVED[0] == 'blockstats', (q, controller._SERVED[0])
            w = con.execute(q).fetchall()[0]
            assert tuple(str(v) for v in got) == tuple(str(v) for v in w), (q, got, w)
    finally:
        B.build = build


def test_the_switch_off():
    import controller, wdb_blockstats as B
    db, con, d = _fixture()
    keep = (B._SUMS[0], B._ANSWERS[0])
    B._SUMS[0] = False; B._ANSWERS[0] = False
    try:
        for col in ('small', 'neg', 'nul'):
            q = 'SELECT SUM(%s), AVG(%s) FROM t' % (col, col)
            got = _rows(db.run(q))[0]
            assert controller._SERVED[0] != 'blockstats', q
            w = con.execute(q).fetchall()[0]
            assert int(got[0]) == int(w[0]) and abs(float(got[1]) - float(w[1])) <= 1e-9 * abs(float(w[1])), (q, got, w)
    finally:
        B._SUMS[0], B._ANSWERS[0] = keep


if __name__ == '__main__':
    for nm, fn in list(globals().items()):
        if nm.startswith('test_'):
            fn(); print('PASS', nm)
