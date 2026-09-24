"""THE PRELOAD PROBE (2026-09-24): the ceiling of loading a query's kernels WHILE it waits.

The kernel load census: 1.56-1.78 s of the cold board is numba bringing cached kernels into the
process, ~88% of it CPU (parsing and linking the cached machine code), and a query meets each
load in line -- its threads idle while it happens. A cold query also spends its first hundreds of
ms waiting on storage. So load the kernels on a spare thread at the moment the query starts.

Pass 1 (--record): per query in a fresh process, record every first-call dispatcher and the
argument types it was called with (pickled numba types). Pass 2: per query, alternating, a fresh
process runs it cold exactly as bench/true_cold.py -- plain, or with a background thread that
starts the moment the query does and loads the recorded list (Dispatcher.compile(signature): a
cache load, no run). Perfect prediction: this is the ceiling of the idea, not a design.

Usage: PYTHONPATH=src python bench/preload_probe.py DB_DIR QUERIES_SQL OUT_DIR [q ...]
"""
import sys, os, time, json, glob, pickle, subprocess, threading


def evict_all(db_dir, src):
    fs = [f for f in glob.glob(os.path.join(db_dir, '*')) if os.path.isfile(f)]
    fs += glob.glob(os.path.join(src, '__pycache__', '*.nb*'))
    for f in fs:
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)


def record(db_dir, sql, out):
    from numba.core import dispatcher as D
    got = []
    orig = D.Dispatcher._compile_for_args
    def rec(self, *a, **k):
        try:
            sig = tuple(self.typeof_pyval(x) for x in a)
            got.append((self.py_func.__module__, self.py_func.__name__, sig))
        except Exception:
            pass
        return orig(self, *a, **k)
    import wdb_db
    db = wdb_db.Database.open(db_dir)
    D.Dispatcher._compile_for_args = rec
    db.run(sql)
    seen = []; keys = set()
    for m, n, s in got:
        if (m, n, s) not in keys:
            keys.add((m, n, s)); seen.append((m, n, s))
    pickle.dump(seen, open(out, 'wb'))
    return len(seen)


def run(db_dir, sql, plan, preload):
    import importlib, wdb_db
    db = wdb_db.Database.open(db_dir)
    evict_all(db_dir, os.path.dirname(os.path.abspath(wdb_db.__file__)))
    items = pickle.load(open(plan, 'rb')) if preload else []
    done = {}
    def loader():
        t0 = time.perf_counter()
        for m, n, sig in items:
            try:
                getattr(importlib.import_module(m), n).compile(sig)
            except Exception as e:
                done.setdefault('err', repr(e))
        done['ms'] = (time.perf_counter() - t0) * 1e3
    t = time.perf_counter()
    th = threading.Thread(target=loader, daemon=True) if items else None
    if th: th.start()
    db.run(sql)
    cold = (time.perf_counter() - t) * 1e3
    if th: th.join()
    return {'cold': round(cold), 'n': len(items), 'loader_ms': round(done.get('ms', 0)), 'err': done.get('err')}


if __name__ == '__main__':
    if sys.argv[1] == '--record':
        print(record(sys.argv[2], sys.argv[3], sys.argv[4])); sys.exit(0)
    if sys.argv[1] == '--run':
        print(json.dumps(run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == '1'))); sys.exit(0)
    db_dir, qfile, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    qs = [l.strip() for l in open(qfile) if l.strip() and not l.strip().startswith('--')]
    todo = [int(x) for x in sys.argv[4:]] or list(range(len(qs)))
    env = os.environ
    for q in todo:
        plan = os.path.join(out, 'q%02d.pkl' % q)
        subprocess.run([sys.executable, __file__, '--record', db_dir, qs[q], plan], capture_output=True, env=env)
        r = {False: [], True: []}; info = None
        for rep in range(2):
            for pre in (False, True):
                o = subprocess.run([sys.executable, __file__, '--run', db_dir, qs[q], plan, '1' if pre else '0'],
                                   capture_output=True, text=True, env=env)
                x = json.loads(o.stdout.strip().splitlines()[-1]); r[pre].append(x['cold'])
                if pre: info = x
        print(json.dumps({'q': q, 'plain': r[False], 'preload': r[True], 'kernels': info['n'],
                          'loader_ms': info['loader_ms'], 'err': info['err']}), flush=True)
