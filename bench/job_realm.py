"""THE JOB REALM: IMDB (21 tables) + the 113 Join Order Benchmark queries.
usage: python3 bench/job_realm.py build   -> duckdb referee + parquet + wdb encode + catalog
"""
import sys, os, time, json, glob, re
sys.path.insert(0, 'src')

ROOT = '/workspace/data/job'
REF = ROOT + '/imdb.duckdb'
PQ = ROOT + '/pq'
DB = ROOT + '/db'

def build():
    import duckdb
    import wdb_encode
    os.makedirs(PQ, exist_ok=True); os.makedirs(DB, exist_ok=True)
    if os.path.exists(REF): os.remove(REF)
    con = duckdb.connect(REF)
    schema = open(ROOT + '/q/schema.sql').read()
    for stmt in schema.split(';'):
        if stmt.strip():
            con.execute(stmt)
    tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()]
    cat = {'tables': {}}
    for t in sorted(tables):
        csv = '%s/csv/%s.csv' % (ROOT, t)
        if not os.path.exists(csv):
            print('MISSING csv for', t); continue
        t0 = time.perf_counter()
        con.execute("COPY %s FROM '%s' (DELIMITER ',', QUOTE '\"', ESCAPE '\\', NULL '')" % (t, csv))
        n = con.execute('SELECT COUNT(*) FROM %s' % t).fetchall()[0][0]
        p = '%s/%s.parquet' % (PQ, t)
        con.execute("COPY %s TO '%s' (FORMAT PARQUET, ROW_GROUP_SIZE 2000000)" % (t, p))
        t1 = time.perf_counter()
        wdb_encode.encode(p, '%s/%s_0.wdb' % (DB, t))
        cols = [r[0] for r in con.execute("DESCRIBE %s" % t).fetchall()]
        types = [r[1] for r in con.execute("DESCRIBE %s" % t).fetchall()]
        sch = [[c, ('int' if 'INT' in ty.upper() else ('str' if 'CHAR' in ty.upper() or 'TEXT' in ty.upper() else 'float'))] for c, ty in zip(cols, types)]
        cat['tables'][t] = {'segments': ['%s_0.wdb' % t], 'columns': cols, 'schema': sch}
        print('%-18s %10d rows | load+parquet %.0fs | encode %.0fs' % (t, n, t1 - t0, time.perf_counter() - t1), flush=True)
    json.dump(cat, open(DB + '/catalog.json', 'w'))
    con.close()
    print('JOB REALM BUILT: %d tables' % len(cat['tables']), flush=True)

if __name__ == '__main__':
    build()
