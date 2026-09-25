"""Q27 with Jackson's byte-unit lengths (one byte = one unit, as ClickHouse's length()): the URL lengths
come straight off the dictionary's offsets, no character counting. Cold, fresh process; and how the
answer differs from the character-counted one (DuckDB's length()).
Usage: PYTHONPATH=src python bench/q27_bytes.py DB_DIR {chars|bytes}
"""
import sys, os, glob, time, json
import wdb_db, wdb_lenagg

d, mode = sys.argv[1], sys.argv[2]
db = wdb_db.Database.open(d)
src = os.path.dirname(os.path.abspath(wdb_db.__file__))
for f in [f for f in glob.glob(d + '/*') if os.path.isfile(f)] + glob.glob(src + '/__pycache__/*.nb*'):
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
if mode == 'bytes':
    _fn = wdb_lenagg._fn_table
    def fn(seg, col, lkind):
        if isinstance(lkind, tuple) and lkind[0] == 'LENGTH':
            return seg.dict_bytelens(col)
        return _fn(seg, col, lkind)
    wdb_lenagg._fn_table = fn
qs = [l.strip() for l in open('benchmark/clickbench/queries.sql') if l.strip() and not l.strip().startswith('--')]
t = time.perf_counter(); r = db.run(qs[27])[0]; cold = (time.perf_counter() - t) * 1e3
t = time.perf_counter(); db.run(qs[27]); hot = (time.perf_counter() - t) * 1e3
print(json.dumps({'mode': mode, 'cold': round(cold), 'hot': round(hot), 'rows': [(int(a), round(float(b), 3), int(c)) for a, b, c in r]}))
