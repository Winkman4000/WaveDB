"""THE ROUTE CENSUS: which read answers each board query, cold and hot. Per query in a fresh process
(as bench/true_cold.py: open, evict the database files and numba's caches), the query runs three
times; for each run: the read that answered (the first read whose execute returned rows; 'none'
when the answer came from outside the read order), its time, and the whole run's time.

Usage: [ENV...] PYTHONPATH=src python bench/route_census.py DB_DIR QUERIES_SQL [q ...]
"""
import sys, os, time, json, glob, subprocess


def one(db_dir, sql):
    import controller, read_methods as R, wdb_db
    db = wdb_db.Database.open(db_dir)
    src = os.path.dirname(os.path.abspath(wdb_db.__file__))
    for f in [f for f in glob.glob(os.path.join(db_dir, '*')) if os.path.isfile(f)] + \
            glob.glob(os.path.join(src, '__pycache__', '*.nb*')):
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    used = []
    new = []
    for rd in controller._READ_ORDER:
        def mk(rd):
            def e(ctx, spec):
                t = time.perf_counter(); r = rd.execute(ctx, spec)
                if r is not None:
                    used.append((rd.name, round((time.perf_counter() - t) * 1e3)))
                return r
            return e
        new.append(R.Read(rd.name, rd.detect, mk(rd), ''))
    controller._READ_ORDER = tuple(new)
    runs = []
    for _ in range(3):
        used.clear()
        t = time.perf_counter(); db.run(sql); ms = (time.perf_counter() - t) * 1e3
        runs.append({'ms': round(ms), 'read': used[0][0] if used else 'none', 'read_ms': used[0][1] if used else None})
    return runs


if __name__ == '__main__':
    db_dir, qfile = sys.argv[1], sys.argv[2]
    qs = [l.strip() for l in open(qfile) if l.strip() and not l.strip().startswith('--')]
    todo = [int(x) for x in sys.argv[3:]] or list(range(len(qs)))
    if len(todo) == 1:
        print(json.dumps({'q': todo[0], 'runs': one(db_dir, qs[todo[0]])})); sys.exit(0)
    for q in todo:
        o = subprocess.run([sys.executable, __file__, db_dir, qfile, str(q)], capture_output=True, text=True, env=os.environ)
        ls = o.stdout.strip().splitlines()
        print(ls[-1] if ls else json.dumps({'q': q, 'err': o.stderr[-300:]}), flush=True)
