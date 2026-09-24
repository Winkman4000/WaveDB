"""THE KERNEL LOAD CENSUS (2026-09-24): how much of a cold query is spent bringing compiled kernels
into the process rather than running them.

A numba kernel's first call in a process goes through Dispatcher._compile_for_args: it loads the
cached machine code from __pycache__ (or compiles, on a cache miss), links it, and only then runs.
Every cold ClickBench query is a fresh process, so it pays this for every kernel it touches.
Per query, exactly as bench/true_cold.py (open, evict the database files and numba's caches, run
once cold), the time inside _compile_for_args is summed (it returns before the kernel runs), and
numba's own counters say whether each first call was a cache hit or a compile.

Usage: PYTHONPATH=src python bench/kernel_load_census.py DB_DIR QUERIES_SQL [q ...]
"""
import sys, os, time, json, glob, subprocess


def one(db_dir, sql):
    from numba.core import dispatcher as D
    rec = []
    orig = D.Dispatcher._compile_for_args
    seen = {}
    def timed(self, *a, **k):
        t = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            t1 = time.perf_counter()
            nm = getattr(self.py_func, '__module__', '?') + '.' + self.py_func.__name__
            seen[nm] = self
            rec.append((nm, (t1 - t) * 1e3, t, t1))
    D.Dispatcher._compile_for_args = timed
    import wdb_db
    db = wdb_db.Database.open(db_dir)
    src = os.path.dirname(os.path.abspath(wdb_db.__file__))
    evict = [f for f in glob.glob(os.path.join(db_dir, '*')) if os.path.isfile(f)]
    evict += glob.glob(os.path.join(src, '__pycache__', '*.nb*'))
    for f in evict:
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    n0 = len(rec)
    t = time.perf_counter(); db.run(sql); cold = (time.perf_counter() - t) * 1e3
    during = rec[n0:]
    iv = sorted((x[2], x[3]) for x in during); wall = 0.0; cur = None
    for a, b in iv:                                  # time during which ANY thread was loading
        if cur is None or a > cur[1]:
            if cur: wall += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur: wall += cur[1] - cur[0]
    names = sorted(set(x[0] for x in during))
    miss = {n: sum(seen[n].stats.cache_misses.values()) for n in names}
    hits = {n: sum(seen[n].stats.cache_hits.values()) for n in names}
    per = {}
    for n, ms, a, b in during:
        per.setdefault(n, [0, 0.0]); per[n][0] += 1; per[n][1] = max(per[n][1], ms)
    t = time.perf_counter(); db.run(sql); hot = (time.perf_counter() - t) * 1e3
    return {'cold': round(cold), 'hot': round(hot), 'load_wall_ms': round(wall * 1e3),
            'load_sum_ms': round(sum(x[1] for x in during)), 'calls': len(during), 'kernels': len(names),
            'compiled': {n: miss[n] for n in names if miss[n]},
            'top': sorted(((n, per[n][0], round(per[n][1]), hits[n], miss[n]) for n in names), key=lambda x: -x[2])[:8]}


if __name__ == '__main__':
    db_dir, qfile = sys.argv[1], sys.argv[2]
    qs = [l.strip() for l in open(qfile) if l.strip() and not l.strip().startswith('--')]
    todo = [int(x) for x in sys.argv[3:]] or list(range(len(qs)))
    if len(todo) == 1:
        r = one(db_dir, qs[todo[0]]); r['q'] = todo[0]; print(json.dumps(r))
    else:
        for q in todo:
            o = subprocess.run([sys.executable, __file__, db_dir, qfile, str(q)],
                               capture_output=True, text=True, env=os.environ)
            ls = o.stdout.strip().splitlines()
            print(ls[-1] if ls else json.dumps({'q': q, 'err': o.stderr[-400:]}), flush=True)
