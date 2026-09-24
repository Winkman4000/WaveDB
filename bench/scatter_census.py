"""THE SCATTER CENSUS (2026-09-24): where the cold run's point reads go.

Per query, in a FRESH process, exactly as bench/true_cold.py (open, evict every data file and
numba's compiled kernels, run once cold): every Segment read method is wrapped, and each call
records its wall time, the rows or codes it was asked for, and the major page faults (pages the
mmap pulled from storage) during the call. Nested calls are kept apart by depth, so a codes_at
that calls _e19_at is not counted twice when summed at depth 0. Faults are process-wide, so
calls running at the same time in threads share the count.

Usage: PYTHONPATH=src python bench/scatter_census.py DB_DIR QUERIES_SQL [q ...]   (one JSON line per query)
"""
import sys, os, time, json, glob, subprocess, resource, threading

METHODS = {  # method -> index of the argument that names the column (or the column dict), and of the rows
    'codes_at': (0, 1), '_raw_codes': (0, None), '_raw_codes_range': (0, None),
    'values_at': (0, 1), 'values_at_rows': (0, 1), 'inline_at': (0, 1), 'fetch': (0, None),
    '_e19_at': (0, 1), '_dict_ints_at': (0, 1), '_pk18_frame': (0, None),
    'read_span': (None, None), 'warm_span': (None, None),
}


def _flt():
    return resource.getrusage(resource.RUSAGE_SELF).ru_majflt


def _instrument(agg):
    import wdb_engine as W
    tl = threading.local()
    for m, (ci, ri) in METHODS.items():
        orig = getattr(W.Segment, m, None)
        if orig is None: continue
        if isinstance(orig, staticmethod): continue
        def make(m, orig, ci, ri):
            def w(self, *a, **k):
                d = getattr(tl, 'd', 0); tl.d = d + 1
                f0 = _flt(); t0 = time.perf_counter()
                try:
                    return orig(self, *a, **k)
                finally:
                    dt = (time.perf_counter() - t0) * 1e3; df = _flt() - f0; tl.d = d
                    col = '-'; enc = '-'
                    if ci is not None and len(a) > ci:
                        x = a[ci]
                        if isinstance(x, dict):
                            col = next((n for n, c in self.cols.items() if c is x), '?'); cc = x
                        else:
                            col = str(x); cc = self.cols.get(x, {})
                        enc = cc.get('code_enc', 0); mode = cc.get('mode')
                        enc = '%s/m%s' % (enc, mode)
                    n = 0
                    if ri is not None and len(a) > ri:
                        try: n = int(getattr(a[ri], 'size', 1))
                        except Exception: n = -1
                    if m == 'read_span' and len(a) >= 2: n = int(a[1]) - int(a[0])
                    key = '%d|%s|%s|%s' % (d, m, col, enc)
                    r = agg.setdefault(key, [0, 0, 0.0, 0])
                    r[0] += 1; r[1] += n; r[2] += dt; r[3] += df
            return w
        setattr(W.Segment, m, make(m, orig, ci, ri))


def one(db_dir, sql):
    agg = {}
    import wdb_db
    db = wdb_db.Database.open(db_dir)
    src = os.path.dirname(os.path.abspath(wdb_db.__file__))
    evict = [f for f in glob.glob(os.path.join(db_dir, '*')) if os.path.isfile(f)]
    evict += glob.glob(os.path.join(src, '__pycache__', '*.nb*'))
    for f in evict:
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    _instrument(agg)
    f0 = _flt(); t = time.perf_counter(); db.run(sql); cold = (time.perf_counter() - t) * 1e3
    calls = sorted(([k] + [round(x, 1) if isinstance(x, float) else x for x in v] for k, v in agg.items()),
                   key=lambda r: -r[3])
    return {'cold': round(cold), 'majflt': _flt() - f0, 'calls': calls}


if __name__ == '__main__':
    db_dir, qfile = sys.argv[1], sys.argv[2]
    qs = [l.strip() for l in open(qfile) if l.strip() and not l.strip().startswith('--')]
    todo = [int(x) for x in sys.argv[3:]] or list(range(len(qs)))
    if len(todo) == 1:
        r = one(db_dir, qs[todo[0]]); r['q'] = todo[0]; print(json.dumps(r))
    else:
        for q in todo:
            out = subprocess.run([sys.executable, __file__, db_dir, qfile, str(q)],
                                 capture_output=True, text=True, env=os.environ)
            lines = out.stdout.strip().splitlines()
            print(lines[-1] if lines else json.dumps({'q': q, 'err': out.stderr[-400:]}), flush=True)
