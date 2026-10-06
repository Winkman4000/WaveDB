"""THE BUILD (2026-10-01): compile every numba kernel signature the engine uses, ahead of time, the way a
C++ engine compiles at install. A fresh machine has an empty numba cache; without this step each query's
first (cold) run would compile its kernels inside the timer (seconds each). The signatures are types only
(array dtypes, ranks, layouts, scalars) -- never data.

  python tools/kernel_build.py dump    record every kernel signature compiled and valid in this checkout's
                                       numba cache into src/kernels.manifest (merged with the existing one)
  python tools/kernel_build.py build   compile every signature in src/kernels.manifest (cache hits are free);
                                       run by benchmark/clickbench/install on the benchmark machine

The manifest is a pickle of {(module, attribute): [signature, ...]} written by the same numba version the
install pins. Workers own whole functions, so two processes never write one function's cache index."""
import os, sys, pickle, time

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
MANIFEST = os.path.join(SRC, 'kernels.manifest')
sys.path.insert(0, SRC)


def _dispatchers():
    import wdb_preload
    from numba.core.registry import CPUDispatcher
    seen = set()
    for m in wdb_preload._kernel_modules():
        for name, v in sorted(vars(m).items()):
            if isinstance(v, CPUDispatcher) and id(v) not in seen and getattr(v.py_func, '__module__', None) == m.__name__:
                seen.add(id(v))
                yield m.__name__, name, v


def dump():
    man = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST, 'rb') as f:
            man = pickle.load(f)
    before = sum(len(v) for v in man.values())
    disp = list(_dispatchers())
    live = {(mod, name) for mod, name, _v in disp}
    man = {k: v for k, v in man.items() if k in live}       # a kernel that no longer exists leaves the manifest
    for mod, name, v in disp:
        cache = getattr(v, '_cache', None); cf = getattr(cache, '_cache_file', None)
        if cf is None:
            continue
        cg = v.targetctx.codegen()
        sigs = man.setdefault((mod, name), [])
        have = {repr(s) for s in sigs}
        for key in cf._load_index():
            if key == cache._index_key(key[0], cg) and repr(key[0]) not in have:
                sigs.append(key[0]); have.add(repr(key[0]))
    man = {k: v for k, v in man.items() if v}
    with open(MANIFEST + '.tmp', 'wb') as f:
        pickle.dump(man, f, protocol=4)
    os.replace(MANIFEST + '.tmp', MANIFEST)
    n = sum(len(v) for v in man.values())
    print('kernels.manifest: %d functions, %d signatures (%d new)' % (len(man), n, n - before))


def _build_one(item):
    (mod, name), sigs = item
    import importlib
    t = time.perf_counter(); ok = bad = 0; errs = []
    try:
        v = getattr(importlib.import_module(mod), name)
    except Exception as e:
        return mod, name, 0, len(sigs), ['import: %s' % e], 0.0
    for s in sigs:
        try:
            v.compile(s); ok += 1
        except Exception as e:
            bad += 1; errs.append('%s: %s' % (s, str(e).splitlines()[0][:120]))
    return mod, name, ok, bad, errs, time.perf_counter() - t


def build(workers=None):
    with open(MANIFEST, 'rb') as f:
        man = pickle.load(f)
    items = sorted(man.items(), key=lambda kv: -len(kv[1]))
    from multiprocessing import get_context
    t0 = time.perf_counter(); ok = bad = 0
    with get_context('spawn').Pool(workers or max(1, os.cpu_count() or 1)) as pool:
        for mod, name, o, b, errs, dt in pool.imap_unordered(_build_one, items):
            ok += o; bad += b
            for e in errs:
                print('  FAILED %s.%s %s' % (mod, name, e), flush=True)
    print('kernel build: %d signatures compiled or found, %d failed, %d functions, %.0f s'
          % (ok, bad, len(items), time.perf_counter() - t0), flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'dump':
        dump()
    elif cmd == 'build':
        sys.exit(build())
    else:
        print(__doc__); sys.exit(2)
