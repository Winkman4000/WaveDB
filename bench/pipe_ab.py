"""A/B of the enc-3 full decode, cold: read-all-then-decode (warm_span + 8 lanes) against the
pipelined lanes (16 lanes, each reading a run of frames then inflating it). Every measurement is
a fresh process: evict the segment file, open it, time one _raw_codes. The pipelined codes must
equal the old path's, element for element.

Usage: PYTHONPATH=src python bench/pipe_ab.py DB_DIR col ...
"""
import sys, os, glob, time, json, subprocess


def one(path, col, pipe):
    import numpy as np, wdb_engine
    wdb_engine._PIPE3[0] = pipe
    import wdb_db                                  # a real query process: the database is open
    wdb_db.Database.open(os.path.dirname(path))    # (kernels warmed, the parallel runtime started)
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    s = wdb_engine.Segment(path)
    t = time.perf_counter(); cc = s._raw_codes(col); ms = (time.perf_counter() - t) * 1e3
    import hashlib
    return {'ms': round(ms, 1), 'sum': hashlib.md5(np.ascontiguousarray(cc).tobytes()).hexdigest()}


if __name__ == '__main__':
    if sys.argv[1] == '--one':
        print(json.dumps(one(sys.argv[2], sys.argv[3], sys.argv[4] == '1'))); sys.exit(0)
    path = glob.glob(os.path.join(sys.argv[1], '*.wdb'))[0]
    for col in sys.argv[2:]:
        r = {True: [], False: []}; h = {}
        for rep in range(3):
            for pipe in (False, True):
                o = subprocess.run([sys.executable, __file__, '--one', path, col, '1' if pipe else '0'],
                                   capture_output=True, text=True, env=os.environ)
                x = json.loads(o.stdout.strip().splitlines()[-1])
                r[pipe].append(x['ms']); h.setdefault(pipe, set()).add(x['sum'])
        same = h[True] == h[False] and len(h[True]) == 1
        print('%-16s old %s   pipelined %s   median %.0f -> %.0f ms   codes identical: %s' % (
            col, sorted(r[False]), sorted(r[True]), sorted(r[False])[1], sorted(r[True])[1], same), flush=True)
        assert same, (col, 'pipelined codes differ')
