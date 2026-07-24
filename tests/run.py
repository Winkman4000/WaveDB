#!/usr/bin/env python3
"""Zero-dependency test runner. Usage: python3 tests/run.py [substring-filter] [--report]
On completion prints the scoreboard summary (examples/report.md); --report regenerates it."""
import os
os.environ.setdefault('WDB_SEQ_NARROW_OK', '1')   # machinery tests build narrow mode-4 toys
import sys, os, importlib, traceback, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

def _scoreboard():
    rep = os.path.join(ROOT, 'examples', 'report.md')
    if '--report' in sys.argv:
        print("\nregenerating scoreboard (bench/report.py) ...")
        subprocess.run([sys.executable, os.path.join(ROOT, 'bench', 'report.py')])
    if os.path.exists(rep):
        print("\n" + "=" * 60 + "\nSCOREBOARD  (examples/report.md)")
        for l in open(rep).read().splitlines():
            if l.startswith('_TPC-H') or 'WaveDB total' in l or l.startswith('**'):
                print("  " + l.replace('**', '').replace('|', ' ').strip())
        print("  regenerate: python bench/report.py   (or tests/run.py --report)")

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    filt = args[0] if args else ''
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
    _scoreboard()
    sys.exit(1 if failed else 0)

if __name__ == '__main__':
    main()
