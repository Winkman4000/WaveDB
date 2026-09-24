"""A/B of one column's full decode, cold, old path against new, through an environment switch
(default WDB_PIPE3; e.g. AB_SWITCH=WDB_PLANES). Every measurement is a fresh process that opens
the database first (as a real query process does: kernels warmed, the parallel runtime started),
then evicts the segment file and times one _raw_codes. The new codes must equal the old path's,
element for element, dtype included.

Usage: [AB_SWITCH=VAR] PYTHONPATH=src python bench/pipe_ab.py DB_DIR col ...
"""
import sys, os, glob, time, json, subprocess, hashlib


def one(path, col):
    import numpy as np, wdb_db, wdb_engine
    wdb_db.Database.open(os.path.dirname(path))
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    s = wdb_engine.Segment(path)
    t = time.perf_counter(); cc = s._raw_codes(col); ms = (time.perf_counter() - t) * 1e3
    return {'ms': round(ms, 1), 'sum': hashlib.md5(np.ascontiguousarray(cc).tobytes()).hexdigest(),
            'dtype': str(np.asarray(cc).dtype)}


if __name__ == '__main__':
    if sys.argv[1] == '--one':
        print(json.dumps(one(sys.argv[2], sys.argv[3]))); sys.exit(0)
    sw = os.environ.get('AB_SWITCH', 'WDB_PIPE3')
    path = glob.glob(os.path.join(sys.argv[1], '*.wdb'))[0]
    for col in sys.argv[2:]:
        r = {True: [], False: []}; h = {}
        for rep in range(3):
            for new in (False, True):
                env = dict(os.environ); env[sw] = '1' if new else '0'
                o = subprocess.run([sys.executable, __file__, '--one', path, col], capture_output=True, text=True, env=env)
                x = json.loads(o.stdout.strip().splitlines()[-1])
                r[new].append(x['ms']); h.setdefault(new, set()).add((x['sum'], x['dtype']))
        same = h[True] == h[False] and len(h[True]) == 1
        print('%-18s %s old %s   new %s   median %.0f -> %.0f ms   codes identical: %s' % (
            col, sw, sorted(r[False]), sorted(r[True]), sorted(r[False])[1], sorted(r[True])[1], same), flush=True)
        assert same, (col, 'new codes differ')
