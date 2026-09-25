"""Q27's timeline after the overlap: how long the dictionary lengths take (foreground, in detect), how
long execute then waits for the background unpacking of the two columns, and the pour. Cold, fresh process.
Usage: PYTHONPATH=src python bench/q27_timeline.py DB_DIR
"""
import sys, os, glob, time
import wdb_db, wdb_lenagg

d = sys.argv[1]
db = wdb_db.Database.open(d)
src = os.path.dirname(os.path.abspath(wdb_db.__file__))
for f in [f for f in glob.glob(d + '/*') if os.path.isfile(f)] + glob.glob(src + '/__pycache__/*.nb*'):
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
qs = [l.strip() for l in open('benchmark/clickbench/queries.sql') if l.strip() and not l.strip().startswith('--')]
T = {}
_fn, _ex = wdb_lenagg._fn_table, wdb_lenagg.execute
def fn(*a, **k):
    t = time.perf_counter(); r = _fn(*a, **k)
    T.setdefault('dictionary lengths (foreground)', (time.perf_counter() - t) * 1e3); return r
def ex(seg, spec):
    t9 = wdb_lenagg._PF.get(id(seg))
    t = time.perf_counter()
    if t9 is not None: t9.join()
    T['wait for the column unpacking after that'] = (time.perf_counter() - t) * 1e3
    t = time.perf_counter(); r = _ex(seg, spec); T['pour + finish'] = (time.perf_counter() - t) * 1e3
    return r
wdb_lenagg._fn_table, wdb_lenagg.execute = fn, ex
t = time.perf_counter(); db.run(qs[27]); tot = (time.perf_counter() - t) * 1e3
for k, v in T.items():
    print('%-45s %6.0f ms' % (k, v))
print('%-45s %6.0f ms' % ('whole query', tot))
