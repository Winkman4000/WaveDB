"""THE TRUE COLD RUN (2026-09-23): what ClickBench's cold run measures -- the database restarted
and the OS caches dropped. /proc/sys/vm/drop_caches is refused in the pod, but a file can leave
the page cache on its own: posix_fadvise(DONTNEED) needs no root (measured on /workspace: 6.7 GB/s
warm -> 0.71 GB/s after eviction). Per query, in a FRESH process:
  1. open the database (the program loads -- ClickBench's restart),
  2. evict every data file of the database and numba's compiled kernels (they load lazily, so
     ClickBench's dropped caches make them cold too),
  3. time the query (cold), then twice more (hot = the better of the two);
  and fincore reports how many bytes of the segment the cold run pulled from storage.

Usage: PYTHONPATH=src python bench/true_cold.py DB_DIR QUERIES_SQL [q ...]   (one JSON line per query)
"""
import sys, os, time, json, glob, subprocess


def _resident(f):
    out = subprocess.run(['fincore', '--bytes', '--noheadings', '--output', 'RES', f],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else -1


def one(db_dir, sql):
    import wdb_db
    db = wdb_db.Database.open(db_dir)
    segs = glob.glob(os.path.join(db_dir, '*.wdb'))
    src = os.path.dirname(os.path.abspath(wdb_db.__file__))
    evict = [f for f in glob.glob(os.path.join(db_dir, '*')) if os.path.isfile(f)]
    evict += glob.glob(os.path.join(src, '__pycache__', '*.nb*'))
    for f in evict:
        fd = os.open(f, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    r0 = sum(_resident(s) for s in segs)
    t = time.perf_counter(); db.run(sql); cold = (time.perf_counter() - t) * 1e3
    r1 = sum(_resident(s) for s in segs)
    hot = []
    for _ in range(2):
        t = time.perf_counter(); db.run(sql); hot.append((time.perf_counter() - t) * 1e3)
    return {'cold': round(cold), 'hot': round(min(hot)), 'read_mb': round((r1 - r0) / 1e6, 1)}


if __name__ == '__main__':
    db_dir, qfile = sys.argv[1], sys.argv[2]
    qs = [l.strip() for l in open(qfile) if l.strip() and not l.strip().startswith('--')]
    todo = [int(x) for x in sys.argv[3:]] or list(range(len(qs)))
    if len(todo) == 1:
        r = one(db_dir, qs[todo[0]]); r['q'] = todo[0]; print(json.dumps(r))
    else:                                        # every query in its own fresh process
        for q in todo:
            out = subprocess.run([sys.executable, __file__, db_dir, qfile, str(q)],
                                 capture_output=True, text=True, env=os.environ).stdout.strip().splitlines()
            print(out[-1] if out else json.dumps({'q': q, 'err': True}), flush=True)
