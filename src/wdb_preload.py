"""THE PROGRAM LOADS ITSELF (Jackson, 2026-10-01): at server startup, every numba kernel we ship is loaded
from numba's on-disk cache -- every compiled signature in each kernel's cache index that matches today's
code -- so no query pays for loading machine code. A C++ engine has no such step: its code is in its
binary and starting the process makes it runnable. Measured before this (kpre.py, cb_van0929): ~3.3 s
of the 43 cold queries was numba loading kernels (50-260 ms a query; ~250 ms of it the first kernel in
a process). ClickBench's driver restarts the database and waits for ./check before the timed query, so
startup is outside the timer. Nothing here touches data: kernels only, all of them, not a list picked
from any queries. A stale cache entry (the code changed since it was compiled) is skipped, never
compiled here. WDB_PRELOAD=0 turns it off."""
import os, glob, importlib, time


def _kernel_modules():
    src = os.path.dirname(os.path.abspath(__file__))
    mods = []
    for f in sorted(glob.glob(os.path.join(src, 'wdb_*.py'))):
        name = os.path.basename(f)[:-3]
        if name == 'wdb_preload':
            continue
        try:
            with open(f, encoding='utf-8', errors='replace') as fh:
                if 'njit' not in fh.read():
                    continue
            mods.append(importlib.import_module(name))
        except Exception:
            continue
    return mods


def preload_all(verbose=False):
    """Load every cached kernel signature that matches today's code. Returns a dict of counts and seconds."""
    if os.environ.get('WDB_PRELOAD', '1') != '1':
        return {'off': True}
    t0 = time.perf_counter()
    try:
        from numba.core.registry import CPUDispatcher
    except Exception:
        return {'no_numba': True}
    seen = set(); kernels = loaded = stale = failed = 0
    for m in _kernel_modules():
        for v in list(vars(m).values()):
            if not isinstance(v, CPUDispatcher) or id(v) in seen:
                continue
            seen.add(id(v))
            cache = getattr(v, '_cache', None)
            cf = getattr(cache, '_cache_file', None)
            if cf is None:
                continue                                   # cache=False kernels: nothing on disk to load
            try:
                idx = cf._load_index()
                cg = v.targetctx.codegen()
            except Exception:
                failed += 1; continue
            kernels += 1
            for key in list(idx):
                sig = key[0]
                try:
                    if key != cache._index_key(sig, cg):
                        stale += 1; continue               # compiled from older code or another CPU
                    v.compile(sig)                         # a cache hit: loads, never compiles
                    loaded += 1
                except Exception:
                    failed += 1
    out = {'kernels': kernels, 'signatures': loaded, 'stale': stale, 'failed': failed,
           'seconds': round(time.perf_counter() - t0, 2)}
    if verbose:
        print('wdb preload: %(kernels)d kernels, %(signatures)d signatures loaded, %(stale)d stale skipped, '
              '%(failed)d failed, %(seconds).2f s' % out, flush=True)
    return out


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print(preload_all(verbose=True))
