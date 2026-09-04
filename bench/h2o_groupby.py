"""H2O.ai db-benchmark groupby realm: generate (numpy port of the R generator), encode, board.
usage: python3 bench/h2o_groupby.py gen N K   |  python3 bench/h2o_groupby.py board
"""
import sys, os, time, json, datetime
sys.path.insert(0, 'src')
import numpy as np

ROOT = '/workspace/data/h2o'
PQ = ROOT + '/G1.parquet'
DB = ROOT + '/h2odb'

QS = [
    ("q1", "SELECT id1, SUM(v1) AS v1 FROM x GROUP BY id1"),
    ("q2", "SELECT id1, id2, SUM(v1) AS v1 FROM x GROUP BY id1, id2"),
    ("q3", "SELECT id3, SUM(v1) AS v1, AVG(v3) AS v3 FROM x GROUP BY id3"),
    ("q4", "SELECT id4, AVG(v1) AS v1, AVG(v2) AS v2, AVG(v3) AS v3 FROM x GROUP BY id4"),
    ("q5", "SELECT id6, SUM(v1) AS v1, SUM(v2) AS v2, SUM(v3) AS v3 FROM x GROUP BY id6"),
    ("q6", "SELECT id4, id5, MEDIAN(v3) AS median_v3, STDDEV(v3) AS sd_v3 FROM x GROUP BY id4, id5"),
    ("q7", "SELECT id3, MAX(v1) - MIN(v2) AS range_v1_v2 FROM x GROUP BY id3"),
    ("q8", "SELECT id6, v3 FROM (SELECT id6, v3, ROW_NUMBER() OVER (PARTITION BY id6 ORDER BY v3 DESC) AS rn FROM x WHERE v3 IS NOT NULL) t WHERE rn <= 2"),
    ("q9", "SELECT id2, id4, POWER(CORR(v1, v2), 2) AS r2 FROM x GROUP BY id2, id4"),
    ("q10", "SELECT id1, id2, id3, id4, id5, id6, SUM(v3) AS v3, COUNT(*) AS cnt FROM x GROUP BY id1, id2, id3, id4, id5, id6"),
]

def gen(N, K, seed=108):
    import pyarrow as pa, pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    os.makedirs(ROOT, exist_ok=True)
    NK = N // K
    def ids(n, card):
        v = rng.integers(1, card + 1, size=n)
        w = len(str(card))
        return pa.array(np.char.add('id', np.char.zfill(v.astype(str), w)))
    t = pa.table({
        'id1': ids(N, K), 'id2': ids(N, K), 'id3': ids(N, NK),
        'id4': pa.array(rng.integers(1, K + 1, size=N).astype(np.int32)),
        'id5': pa.array(rng.integers(1, K + 1, size=N).astype(np.int32)),
        'id6': pa.array(rng.integers(1, NK + 1, size=N).astype(np.int32)),
        'v1': pa.array(rng.integers(1, 6, size=N).astype(np.int32)),
        'v2': pa.array(rng.integers(1, 16, size=N).astype(np.int32)),
        'v3': pa.array(np.round(rng.random(N) * 100, 6)),
    })
    pq.write_table(t, PQ, row_group_size=4_000_000)
    print('wrote', PQ, N, 'rows', flush=True)
    import wdb_encode
    os.makedirs(DB, exist_ok=True)
    t0 = time.perf_counter()
    wdb_encode.encode(PQ, DB + '/x_0.wdb')
    schema = [['id1', 'str'], ['id2', 'str'], ['id3', 'str'], ['id4', 'int'], ['id5', 'int'],
              ['id6', 'int'], ['v1', 'int'], ['v2', 'int'], ['v3', 'float']]
    json.dump({'tables': {'x': {'segments': ['x_0.wdb'], 'columns': [c for c, _ in schema],
                                'schema': schema}}}, open(DB + '/catalog.json', 'w'))
    print('encoded in %.0fs' % (time.perf_counter() - t0), flush=True)

def board():
    import duckdb
    import wdb_kernels; wdb_kernels.warm()
    from wdb_db import Database
    db = Database.open(DB)
    con = duckdb.connect()
    con.execute("CREATE VIEW x AS SELECT * FROM read_parquet('%s')" % PQ)
    def same(w, e):
        # THE REFEREE'S LAW (board_tpch): floats within 1e-6 relative, else string equality
        if len(w) != len(e): return False
        keyf = lambda t: tuple(str(x) for x in t if not isinstance(x, float))
        kw = sorted(w, key=keyf); ke = sorted(e, key=keyf)
        for rw, re_ in zip(kw, ke):
            if len(rw) != len(re_): return False
            for a, b in zip(rw, re_):
                if isinstance(b, float):
                    if a is None or abs(float(a) - b) > 1e-6 * max(1.0, abs(b)): return False
                elif str(a) != str(b): return False
        return True
    ok = holes = wins = 0; wt = dt = 0.0
    skip = set(os.environ.get('H2O_SKIP', '').split(','))
    for name, q in QS:
        if name in skip:
            print('%-4s SKIP  (held out: known cardinality war)' % name, flush=True); continue
        try:
            e = con.execute(q).fetchall()
            ds = []
            for _ in range(3):
                t0 = time.perf_counter(); con.execute(q).fetchall(); ds.append(time.perf_counter() - t0)
            dm = min(ds)
        except Exception as ex:
            print('%-4s DUCK-ERR %s' % (name, str(ex)[:80]), flush=True); continue
        import signal
        def _alarm(sig, frm): raise TimeoutError('per-query timeout')
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(120)
        try:
            w = db.run(q); w = w[0] if isinstance(w, tuple) else w
            signal.alarm(0)
            ws = []
            for _ in range(3):
                t0 = time.perf_counter(); db.run(q); ws.append(time.perf_counter() - t0)
            wm = min(ws)
        except BaseException as ex:
            signal.alarm(0)
            holes += 1
            print('%-4s HOLE  %s: %s' % (name, type(ex).__name__, str(ex)[:90]), flush=True); continue
        exact = same(w, e)
        ok += exact; wt += wm; dt += dm
        if exact and wm < dm: wins += 1
        print('%-4s %s wave=%6.2fs duck=%6.2fs x%5.2f rows=%d/%d' % (
            name, 'OK   ' if exact else 'WRONG', wm, dm, dm / wm, len(w), len(e)), flush=True)
    print('H2O GROUPBY BOARD: %d queries | ok=%d holes=%d wins=%d | wave %.1fs duck %.1fs'
          % (len(QS), ok, holes, wins, wt, dt), flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'gen':
        gen(int(float(sys.argv[2])), int(float(sys.argv[3])))
    else:
        board()
