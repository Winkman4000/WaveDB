"""THE SCOPE SPEED BOARD: every construct of the scope probe, at 10M rows, wave vs
duck, min-of-3, exact-checked. Shows which faces are already fast and which need
the dict-space treatment.
usage: python3 bench/board_scope.py gen | python3 bench/board_scope.py run
"""
import sys, os, time, json, signal
sys.path.insert(0, 'src'); sys.path.insert(0, 'bench')
import numpy as np

ROOT = '/workspace/data/scope10m'
DB = ROOT + '/db'
N = 10_000_000

def gen():
    import pyarrow as pa, pyarrow.parquet as pq
    import wdb_encode
    os.makedirs(DB, exist_ok=True)
    t = pq.read_table('/workspace/data/h2o/G1.parquet').slice(0, N)
    rng = np.random.default_rng(7)
    n = t.num_rows
    nul_i = rng.integers(0, 50, n).astype(np.int32); mask = rng.random(n) < 0.1
    nul_i = pa.array(np.where(mask, None, nul_i).tolist(), type=pa.int32())
    sv = rng.integers(0, 20, n)
    nul_s = pa.array([None if m else ('s%d' % v) for m, v in zip(mask.tolist(), sv.tolist())])
    days = pa.array((rng.integers(0, 3650, n) + 8000).astype(np.int32))
    flag = pa.array((rng.random(n) < 0.5).tolist())
    t = t.append_column('ni', nul_i).append_column('ns', nul_s).append_column('d', days).append_column('b', flag)
    d = pa.table({'id4': pa.array(np.arange(1, 101, dtype=np.int32)), 'dname': pa.array(['name%03d' % i for i in range(1, 101)]),
                  'w': pa.array(np.round(rng.random(100) * 10, 3))})
    cat = {'tables': {}}
    for name, tb in (('x', t), ('d', d)):
        p = '%s/%s.parquet' % (ROOT, name); pq.write_table(tb, p, row_group_size=2_000_000)
        t0 = time.perf_counter(); wdb_encode.encode(p, '%s/%s_0.wdb' % (DB, name))
        sch = []
        for f in tb.schema:
            ty = str(f.type)
            sch.append([f.name, 'float' if 'double' in ty or 'float' in ty else ('str' if 'string' in ty else ('bool' if ty == 'bool' else 'int'))])
        cat['tables'][name] = {'segments': ['%s_0.wdb' % name], 'columns': tb.column_names, 'schema': sch}
        print('%s: %d rows encoded in %.0fs' % (name, tb.num_rows, time.perf_counter() - t0), flush=True)
    json.dump(cat, open(DB + '/catalog.json', 'w'))

def run():
    import duckdb
    from sql_scope import Q
    import wdb_kernels; wdb_kernels.warm()
    from wdb_db import Database
    db = Database.open(DB)
    con = duckdb.connect()
    for name in ('x', 'd'):
        con.execute("CREATE VIEW %s AS SELECT * FROM read_parquet('%s/%s.parquet')" % (name, ROOT, name))
    def norm(v):
        if isinstance(v, bool): return 'b:%d' % v
        if isinstance(v, float): return '%.6g' % v
        if v is None: return 'NULL'
        try:
            import decimal
            if isinstance(v, decimal.Decimal): return '%.6g' % float(v)
        except Exception: pass
        return str(v)
    def same(w, e):
        if len(w) != len(e): return False
        return sorted(tuple(norm(v) for v in r) for r in w) == sorted(tuple(norm(v) for v in r) for r in e)
    def _alarm(sig, frm): raise TimeoutError('timeout')
    signal.signal(signal.SIGALRM, _alarm)
    skip = set(os.environ.get('SCOPE_SKIP', '').split(','))
    rows_out = []
    for cat, name, q in Q:
        if name in skip: continue
        # the 10M realm: literal-only queries and tiny d x d joins are not scale tests, keep them anyway
        try:
            e = con.execute(q).fetchall(); ds = []
            for _ in range(3):
                t0 = time.perf_counter(); con.execute(q).fetchall(); ds.append(time.perf_counter() - t0)
            dm = min(ds)
        except Exception as ex:
            print('%-7s %-20s DUCK-ERR %s' % (cat, name, str(ex)[:60]), flush=True); continue
        signal.alarm(300)
        try:
            w = db.run(q); w = w[0] if isinstance(w, tuple) else w
            ws = []
            for _ in range(2):
                t0 = time.perf_counter(); db.run(q); ws.append(time.perf_counter() - t0)
            signal.alarm(0)
            wm = min(ws)
            st = 'OK' if same(w, e) else 'WRONG'
        except BaseException as ex:
            signal.alarm(0)
            print('%-7s %-20s HOLE  %s: %s' % (cat, name, type(ex).__name__, str(ex)[:60]), flush=True); continue
        r = dm / wm if wm > 0 else float('inf')
        rows_out.append((cat, name, st, wm, dm, r))
        print('%-7s %-20s %-5s wave=%7.3fs duck=%7.3fs x%6.2f rows=%d' % (cat, name, st, wm, dm, r, len(w)), flush=True)
    oks = [x for x in rows_out if x[2] == 'OK']
    ratios = sorted(x[5] for x in oks)
    wins = sum(1 for x in oks if x[5] >= 1.0)
    print('SCOPE SPEED BOARD: %d measured | ok=%d wrong=%d | wins=%d | median x%.2f | wave %.1fs duck %.1fs' % (
        len(rows_out), len(oks), len(rows_out) - len(oks), wins, ratios[len(ratios) // 2] if ratios else 0,
        sum(x[3] for x in rows_out), sum(x[4] for x in rows_out)), flush=True)
    print('FACES (slowest first):', flush=True)
    for x in sorted(oks, key=lambda x: x[5])[:15]:
        print('  %-7s %-20s x%5.2f  wave %6.2fs duck %6.2fs' % (x[0], x[1], x[5], x[3], x[4]), flush=True)

if __name__ == '__main__':
    gen() if sys.argv[1] == 'gen' else run()
