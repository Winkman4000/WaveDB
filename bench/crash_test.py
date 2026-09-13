"""THE CRASH HARNESS: kill the engine at every named write step and prove the database
opens and still answers exactly afterward.
usage: python3 bench/crash_test.py [K]        (K segments, built as K loads; /workspace/data/scope/x.parquet)
"""
import sys, os, shutil, subprocess, json, time
sys.path.insert(0, 'src')

SRC = '/workspace/data/scope/x.parquet'
DB = '/tmp/crashdb'
PY = sys.executable

def run(script, env=None):
    e = dict(os.environ); e.update(env or {})
    p = subprocess.run([PY, '-c', script], capture_output=True, text=True, env=e, timeout=600)
    return p.returncode, (p.stdout + p.stderr)[-600:]

def truth(db_dir):
    """the answers a correct database must give (computed by duck on the same rows)"""
    import duckdb
    con = duckdb.connect(); con.execute("CREATE VIEW x AS SELECT * FROM read_parquet('%s') WHERE NOT (id4 = 7)" % SRC)
    return [con.execute(q).fetchall() for q in QS]

QS = ["SELECT COUNT(*) FROM x", "SELECT id4, COUNT(*) AS c FROM x GROUP BY id4 ORDER BY c DESC, id4 LIMIT 3",
      "SELECT SUM(v1), MIN(id6), MAX(id6) FROM x", "SELECT COUNT(*) FROM x WHERE id4 = 7"]

def answers(db_dir):
    from wdb_db import Database
    db = Database.open(db_dir)
    out = []
    for q in QS:
        r = db.run(q); out.append(r[0] if isinstance(r, tuple) else r)
    return out

def same(a, b):
    def nm(v): return ('%.6g' % v) if isinstance(v, float) else str(v)
    return [sorted(tuple(nm(v) for v in r) for r in x) for x in a] == [sorted(tuple(nm(v) for v in r) for r in x) for x in b]

K = int(sys.argv[1]) if len(sys.argv) > 1 else 1        # segments: the database is built as K loads


def fresh():
    shutil.rmtree(DB, ignore_errors=True); os.makedirs(DB)
    import pyarrow.parquet as pq
    t = pq.read_table(SRC); step = (t.num_rows + K - 1) // K
    for i in range(K):
        p = '/tmp/crash_slice%d.parquet' % i; pq.write_table(t.slice(i * step, step), p)
        r = subprocess.run([PY, 'bin/wdb', 'load', DB, 'x', p, '--workers', '4'], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-300:]
    rc, out = run("import sys; sys.path.insert(0,'src'); from wdb_db import Database; import wdb_dml\n"
                  "db = Database.open('%s'); print('deleted', wdb_dml.delete(db.cat, 'DELETE FROM x WHERE id4 = 7'))" % DB)
    assert rc == 0, out

def main():
    print('truth from duck...', flush=True); T = truth(DB)
    results = []
    print('segments per database: %d (tier ceiling: %d rows -> a tiered compaction merges a subset)' % (K, 120_000 if K > 1 else 25_000_000), flush=True)
    for point in ('encode:renamed', 'compact:segment-written', 'compact:catalog-saved'):
        fresh()
        rc, out = run("import sys; sys.path.insert(0,'src'); from wdb_db import Database\n"
                      "db = Database.open('%s'); print(db.compact('x', tier_rows=%d))" % (DB, 120_000 if K > 1 else 25_000_000), env={'WDB_CRASH_AT': point})
        died = rc == -9 or 'CRASH POINT' in out
        files = sorted(f for f in os.listdir(DB) if f.endswith('.wdb') or f.endswith('.partial'))
        try:
            A = answers(DB); ok = same(A, T); state = 'opens, EXACT' if ok else 'opens, WRONG: %s' % str(A[0])[:40]
        except Exception as e:
            ok = False; state = 'refuses: %s' % str(e)[:80]
        # a second compaction after recovery must also work
        try:
            rc2, out2 = run("import sys; sys.path.insert(0,'src'); from wdb_db import Database\n"
                            "db = Database.open('%s'); r = db.compact('x', tier_rows=%d); print('recompacted', r['new_segment'], r['rows'])" % (DB, 120_000 if K > 1 else 25_000_000))
            B = answers(DB); ok2 = same(B, T) and rc2 == 0
        except Exception as e:
            ok2 = False
        results.append(ok and ok2)
        print('%-26s died=%-5s files=%-28s -> %s | recompact exact=%s' % (point, died, ','.join(files), state, ok2), flush=True)
    print('CRASH HARNESS: %d/%d crash points survive with exact answers' % (sum(results), len(results)), flush=True)

if __name__ == '__main__':
    main()
