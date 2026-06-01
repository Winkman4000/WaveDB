#!/usr/bin/env python3
"""Zero-dependency test runner. Usage: python3 tests/run.py [substring-filter]"""
import sys, os, importlib, traceback, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else ''
    mods = sorted(f[:-3] for f in os.listdir(HERE)
                  if f.startswith('test_') and f.endswith('.py'))
    passed = failed = 0; fails = []
    t0 = time.time()
    for mname in mods:
        mod = importlib.import_module(mname)
        tests = sorted(n for n in dir(mod) if n.startswith('test_'))
        for tname in tests:
            if filt and filt not in f"{mname}.{tname}": continue
            try:
                getattr(mod, tname)()
                passed += 1; print(f"  PASS  {mname}.{tname}")
            except Exception as e:
                failed += 1; fails.append((f"{mname}.{tname}", e, traceback.format_exc()))
                print(f"  FAIL  {mname}.{tname}: {e}")
    dt = time.time() - t0
    print(f"\n{'='*60}\n{passed} passed, {failed} failed in {dt:.1f}s")
    if fails:
        print("\nFAILURES:")
        for name, e, tb in fails:
            print(f"\n--- {name} ---\n{tb}")
    sys.exit(1 if failed else 0)

if __name__ == '__main__':
    main()
