"""H2O.ai db-benchmark JOIN realm: x joins small (N/1e6), medium (N/1e3), big (N).
usage: python3 bench/h2o_join.py gen N   |  python3 bench/h2o_join.py board
"""
import sys, os, time, json
sys.path.insert(0, 'src')
import numpy as np

ROOT = '/workspace/data/h2oj'
DB = ROOT + '/db'

QS = [
    ("j1", "SELECT x.id1, x.id2, x.id3, x.id4, x.id5, x.id6, x.v1, small.id4 AS small_id4, small.v2 FROM x INNER JOIN small ON x.id1 = small.id1"),
    ("j2", "SELECT x.id1, x.id2, x.id3, x.id4, x.id5, x.id6, x.v1, medium.id1 AS medium_id1, medium.id4 AS medium_id4, medium.id5 AS medium_id5, medium.v2 FROM x INNER JOIN medium ON x.id2 = medium.id2"),
    ("j3", "SELECT x.id1, x.id2, x.id3, x.id4, x.id5, x.id6, x.v1, medium.id1 AS medium_id1, medium.id4 AS medium_id4, medium.id5 AS medium_id5, medium.v2 FROM x LEFT JOIN medium ON x.id2 = medium.id2"),
    ("j4", "SELECT x.id1, x.id2, x.id3, x.id4, x.id5, x.id6, x.v1, medium.id1 AS medium_id1, medium.id2 AS medium_id2, medium.id4 AS medium_id4, medium.v2 FROM x INNER JOIN medium ON x.id5 = medium.id5"),
    ("j5", "SELECT x.id1, x.id2, x.id3, x.id4, x.id5, x.id6, x.v1, big.id1 AS big_id1, big.id2 AS big_id2, big.id4 AS big_id4, big.id5 AS big_id5, big.id6 AS big_id6, big.v2 FROM x INNER JOIN big ON x.id3 = big.id3"),
    # aggregate forms: the same joins, engine-speed visible under the emission
    ("a1", "SELECT small.id4, SUM(x.v1) AS v1, SUM(small.v2) AS v2 FROM x INNER JOIN small ON x.id1 = small.id1 GROUP BY small.id4"),
    ("a2", "SELECT medium.id4, SUM(x.v1) AS v1, SUM(medium.v2) AS v2 FROM x INNER JOIN medium ON x.id2 = medium.id2 GROUP BY medium.id4"),
    ("a5", "SELECT COUNT(*) AS n, SUM(x.v1 + big.v2) AS s FROM x INNER JOIN big ON x.id3 = big.id3"),
]

def gen(N, seed=108):
    import pyarrow as pa, pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    os.makedirs(DB, exist_ok=True)
    K1, K2, K3 = max(1, N // 1_000_000), max(1, N // 1_000), N
    def sid(v):
        return pa.array(np.char.add('id', v.astype(str)))
    id1 = rng.integers(1, K1 + 1, N); id2 = rng.integers(1, K2 + 1, N); id3 = rng.integers(1, K3 + 1, N)
    x = pa.table({'id1': pa.array(id1.astype(np.int32)), 'id2': pa.array(id2.astype(np.int32)), 'id3': pa.array(id3.astype(np.int32)),
                  'id4': sid(id1), 'id5': sid(id2), 'id6': sid(id3), 'v1': pa.array(np.round(rng.random(N) * 100, 6))})
    def dim(n, cols):
        keys = rng.permutation(n) + 1
        d = {}
        for c in cols:
            if c == 'id1': d['id1'] = pa.array(keys.astype(np.int32)); d['id4'] = sid(keys)
            if c == 'id2': d['id2'] = pa.array(keys.astype(np.int32)); d['id5'] = sid(keys)
            if c == 'id3': d['id3'] = pa.array(keys.astype(np.int32)); d['id6'] = sid(keys)
        d['v2'] = pa.array(np.round(rng.random(n) * 100, 6))
        return pa.table(d)
    small = dim(K1, ['id1'])
    medium = pa.table({'id1': pa.array((rng.integers(1, K1 + 1, K2)).astype(np.int32)),
                       'id2': pa.array((rng.permutation(K2) + 1).astype(np.int32))})
    medium = medium.append_column('id4', sid(medium['id1'].to_numpy())).append_column('id5', sid(medium['id2'].to_numpy())).append_column('v2', pa.array(np.round(rng.random(K2) * 100, 6)))
    bk = rng.permutation(K3) + 1
    big = pa.table({'id1': pa.array(rng.integers(1, K1 + 1, K3).astype(np.int32)), 'id2': pa.array(rng.integers(1, K2 + 1, K3).astype(np.int32)),
                    'id3': pa.array(bk.astype(np.int32))})
    big = big.append_column('id4', sid(big['id1'].to_numpy())).append_column('id5', sid(big['id2'].to_numpy())).append_column('id6', sid(bk)).append_column('v2', pa.array(np.round(rng.random(K3) * 100, 6)))
    import wdb_encode
    cat = {'tables': {}}
    for name, t in (('x', x), ('small', small), ('medium', medium), ('big', big)):
        p = '%s/%s.parquet' % (ROOT, name)
        pq.write_table(t, p, row_group_size=4_000_000)
        t0 = time.perf_counter(); wdb_encode.encode(p, '%s/%s_0.wdb' % (DB, name))
        cols = t.column_names
        cat['tables'][name] = {'segments': ['%s_0.wdb' % name], 'columns': cols,
                               'schema': [[c, ('float' if c.startswith('v') else ('str' if c in ('id4', 'id5', 'id6') else 'int'))] for c in cols]}
        print('%s: %d rows encoded in %.0fs' % (name, t.num_rows, time.perf_counter() - t0), flush=True)
    json.dump(cat, open(DB + '/catalog.json', 'w'))

def board():
    import duckdb
    import wdb_kernels; wdb_kernels.warm()
    from wdb_db import Database
    db = Database.open(DB)
    con = duckdb.connect()
    for name in ('x', 'small', 'medium', 'big'):
        con.execute("CREATE VIEW %s AS SELECT * FROM read_parquet('%s/%s.parquet')" % (name, ROOT, name))
    def same(w, e):
        if len(w) != len(e): return False
        keyf = lambda t: tuple((str(x) if not isinstance(x, float) else '%.9g' % x) for x in t)   # full-row order: x.id3 is NOT unique
        kw = sorted(w, key=keyf); ke = sorted(e, key=keyf)
        for rw, re_ in zip(kw, ke):
            for a, b in zip(rw, re_):
                if isinstance(b, float):
                    if a is None or abs(float(a) - b) > 1e-6 * max(1.0, abs(b)): return False
                elif str(a) != str(b): return False
        return True
    skip = set(os.environ.get('H2OJ_SKIP', '').split(','))
    ok = holes = wins = 0; wt = dt = 0.0
    for name, q in QS:
        if name in skip: print('%-3s SKIP' % name, flush=True); continue
        try:
            e = con.execute(q).fetchall()
            ds = []
            for _ in range(3):
                t0 = time.perf_counter(); con.execute(q).fetchall(); ds.append(time.perf_counter() - t0)
            dm = min(ds)
        except Exception as ex:
            print('%-3s DUCK-ERR %s' % (name, str(ex)[:80]), flush=True); continue
        try:
            w = db.run(q); w = w[0] if isinstance(w, tuple) else w
            ws = []
            for _ in range(3):
                t0 = time.perf_counter(); db.run(q); ws.append(time.perf_counter() - t0)
            wm = min(ws)
        except BaseException as ex:
            holes += 1
            print('%-3s HOLE  %s: %s' % (name, type(ex).__name__, str(ex)[:100]), flush=True); continue
        exact = same(w, e)
        ok += exact; wt += wm; dt += dm
        if exact and wm < dm: wins += 1
        print('%-3s %s wave=%6.2fs duck=%6.2fs x%5.2f rows=%d/%d' % (name, 'OK   ' if exact else 'WRONG', wm, dm, dm / wm, len(w), len(e)), flush=True)
    print('H2O JOIN BOARD: %d queries | ok=%d holes=%d wins=%d | wave %.1fs duck %.1fs' % (len(QS), ok, holes, wins, wt, dt), flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'gen': gen(int(float(sys.argv[2])))
    else: board()
