#!/usr/bin/env python3
"""Zero-dependency test runner. Usage: python3 tests/run.py [substring-filter] [--report]
On completion prints the scoreboard summary (examples/report.md); --report regenerates it."""
import os
os.environ.setdefault('WDB_SEQ_NARROW_OK', '1')   # machinery tests build narrow mode-4 toys
os.environ.setdefault('WDB_SIDECARS', '1')        # THE SWITCH: new databases are born off; the suite exercises births
os.environ.setdefault('WDB_SIDECAR_STRICT', '1')  # THE SENTINEL raises: a birth under an off switch is a defect
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

def _static_gate():
    """THE STATIC GATE (Jackson, 2026-09-15): a one-second lint with ZERO false positives on this
    codebase -- undefined names, invalid syntax-level errors, mutable defaults, closures over loop
    variables -- run before the four-minute suite so a typo is caught first. Rules chosen by
    measurement: E711 (`!= None`) is EXCLUDED because it is right on numpy object arrays. Skipped
    when ruff is not installed."""
    import shutil, subprocess, sys as _s
    ruff = shutil.which('ruff') or (_s.executable.rsplit('/', 1)[0] + '/ruff')
    import os
    if not os.path.exists(ruff): return True
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
    r = subprocess.run([ruff, 'check', '--select', 'F821,F822,F823,PLE,B006,B008,B023', '--quiet', src], capture_output=True, text=True)
    if r.returncode != 0:
        print('STATIC GATE FAILED:\n' + (r.stdout + r.stderr)[-2000:], flush=True)
        return False
    print('static gate: clean', flush=True)
    return True


def main():
    if not _static_gate():
        sys.exit(2)
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
