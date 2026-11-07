"""THE SHARD THAT FILLED (2026-11-07): pairagg's per-shard hash tables had a fixed 2^18 slots, so more than 262,144
distinct pairs landing in one shard (g % 64 the same -- here every pair's second key has code 0 of 64 values) looped
forever. Each shard's table is now sized from its own rows. The answer must come back, and be DuckDB's."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_one_shard_with_more_pairs_than_slots():
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq, duckdb, sqlglot
    import wdb_encode, wdb_pairagg, wdb_join
    from wdb_db import Database
    NPAIR = 300_000
    N = NPAIR + 63
    rng = np.random.default_rng(5)
    a = rng.permutation(N).astype(np.int64) * 7 + 3
    a[:2000] = a[2000:4000]                               # some pairs repeat: counts of 2 at the top
    b = np.zeros(N, np.int64); b[NPAIR:] = np.arange(1, 64)
    f = (np.arange(N) % 3).astype(np.int64)
    d = os.path.join(tempfile.gettempdir(), 'pas_%s' % uuid.uuid4().hex[:8]); os.makedirs(d)
    try:
        import helpers as _H; _H.register_dir(d)
    except Exception:
        pass
    p = os.path.join(d, 't.parquet'); pq.write_table(pa.table({'a': a, 'b': b, 'f': f}), p)
    db = Database.create(d); db.cat.add_table('t', [['a', 'int'], ['b', 'int'], ['f', 'int']])
    saved = {k9: os.environ.pop(k9, None) for k9 in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:                                      # the runner's sequence levers would store b (0...0, 1..63) as a sequence
        wdb_encode.encode(p, os.path.join(d, 't_0.wdb'))
    finally:
        for k9, v9 in saved.items():
            if v9 is not None:
                os.environ[k9] = v9
    db.cat.add_segment('t', 't_0.wdb')
    con = duckdb.connect(); con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('%s')" % p)
    seg, _x = wdb_join._solo_segment(db, 't')
    for q in ['SELECT a, b, COUNT(*) AS c, SUM(f), AVG(f) FROM t GROUP BY a, b ORDER BY c DESC LIMIT 10',
              'SELECT a, b, COUNT(*) AS c FROM t WHERE f <> 1 GROUP BY a, b ORDER BY c DESC LIMIT 10']:
        spec = wdb_pairagg.detect(seg, sqlglot.parse_one(q), None)
        assert spec is not None, ('pairagg should take it', q)
        r = wdb_pairagg.execute(seg, spec)
        rows = r[0] if isinstance(r, tuple) else r
        full = {(x[0], x[1]): tuple(float(v) for v in x[2:]) for x in con.execute(q.split(' ORDER BY')[0]).fetchall()}
        want = sorted((float(x[2]) for x in con.execute(q).fetchall()), reverse=True)
        got = [tuple(float(v) for v in row) for row in rows]
        assert sorted((g[2] for g in got), reverse=True) == want, (q, got[:3])
        assert all(full[(int(g[0]), int(g[1]))] == g[2:] for g in got), (q, got[:3])


if __name__ == '__main__':
    test_one_shard_with_more_pairs_than_slots(); print('PASS')
