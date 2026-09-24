"""THE GENERATOR CENSUS (2026-09-24): which board queries reach wdb_exprjit's runtime code
generator, what shape each sends it, and what the compile costs.

Per query, in a FRESH process (the in-process kernel cache starts empty, as after a restart):
run it once (first: pays any generated-kernel compile) and twice more (hot = the better).
Every kernel the generator builds is wrapped so its first call (compile + run) and its later
calls (run only) are timed separately.

Usage: PYTHONPATH=src python bench/jit_census.py DB_DIR QUERIES_SQL [q ...]   (one JSON line per query)
"""
import sys, os, time, json, subprocess

LOG = []


def _instrument():
    import wdb_exprjit as X
    real_njit = X._njit
    cur = {}

    class Timed:
        def __init__(self, disp, rec):
            self.disp = disp; self.rec = rec
        def __call__(self, *a):
            t = time.perf_counter(); r = self.disp(*a); dt = (time.perf_counter() - t) * 1e3
            self.rec['calls'].append(round(dt, 2))
            return r

    def njit(**kw):
        def deco(f):
            rec = dict(cur['rec']); rec['calls'] = []; LOG.append(rec)
            return Timed(real_njit(**kw)(f), rec)
        return deco
    X._njit = njit

    def wrap(name, fields):
        orig = getattr(X, name)
        def w(*a, **k):
            cur['rec'] = {'builder': name}
            for f, v in zip(fields, a): cur['rec'][f] = repr(v)
            for f, v in k.items(): cur['rec'][f] = repr(v)
            return orig(*a, **k)
        setattr(X, name, w)
    wrap('_build', ['body', 'slot_gathered', 'gk_gathered', 'nkeys', 'has_mask', 'need_minmax'])
    wrap('_build_multi', ['bodies', 'mm_flags', 'slot_gathered', 'slot_code', 'gk_gathered',
                          'nkeys', 'has_mask', 'pred'])
    wrap('_build_scalar', ['bodies', 'slot_gathered', 'slot_code', 'has_mask', 'pred'])


def one(db_dir, sql):
    _instrument()
    import wdb_db
    db = wdb_db.Database.open(db_dir)
    ts = []
    for _ in range(3):
        t = time.perf_counter(); db.run(sql); ts.append(round((time.perf_counter() - t) * 1e3))
    return {'first': ts[0], 'hot': min(ts[1:]), 'kernels': LOG}


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
