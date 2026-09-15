"""THE JOB BOARD: 113 Join Order Benchmark queries, wave vs duck (the duckdb
referee file), exact-checked, one timing each (first pass: coverage).
usage: python3 bench/board_job.py [--time]
"""
import sys, os, time, glob, signal, traceback
sys.path.insert(0, 'src')

ROOT = '/workspace/data/job'
REF = ROOT + '/imdb.duckdb'
DB = ROOT + '/db'

def main():
    import duckdb
    import wdb_kernels; wdb_kernels.warm()
    from wdb_db import Database
    db = Database.open(DB)
    con = duckdb.connect(REF, read_only=True)
    timed = '--time' in sys.argv
    def norm(v):
        if v is None: return 'NULL'
        if isinstance(v, float): return '%.6g' % v
        return str(v)
    def same(w, e):
        if len(w) != len(e): return False
        return sorted(tuple(norm(v) for v in r) for r in w) == sorted(tuple(norm(v) for v in r) for r in e)
    def _alarm(sig, frm): raise TimeoutError('timeout')
    signal.signal(signal.SIGALRM, _alarm)
    files = sorted(glob.glob(ROOT + '/q/*.sql'), key=lambda p: (int(''.join(ch for ch in os.path.basename(p) if ch.isdigit()) or 0), os.path.basename(p)))
    files = [f for f in files if os.path.basename(f)[0].isdigit()]
    only = set(a for a in sys.argv[1:] if not a.startswith('--'))
    if '--warm' in sys.argv:
        # THE WARM STEP (as the ClickBench kit): every query once, untimed -- births, the shelved
        # predicates and key columns, the roads' first touch -- before the clock starts
        import re as _re
        t0 = time.perf_counter(); n = 0
        for f in files:
            name = os.path.basename(f)[:-4]
            if only and name not in only: continue
            q = open(f).read().strip().rstrip(';'); q = _re.sub(r'\bAS at\b', 'AS at1', q); q = _re.sub(r'\bat\.', 'at1.', q)
            try: db.run(q); n += 1
            except Exception: pass
        print('WARM: %d queries in %.1fs' % (n, time.perf_counter() - t0), flush=True)
    tally = {}; wt = dt = 0.0; wins = 0; holes = {}
    for f in files:
        name = os.path.basename(f)[:-4]
        if only and name not in only: continue
        q = open(f).read().strip().rstrip(';')
        # JOB aliases aka_title AS at -- a reserved word in duck's parser: rename on BOTH sides
        import re as _re
        q = _re.sub(r'\bAS at\b', 'AS at1', q); q = _re.sub(r'\bat\.', 'at1.', q)
        try:
            t0 = time.perf_counter(); e = con.execute(q).fetchall(); dm = time.perf_counter() - t0
        except Exception as ex:
            print('%-4s DUCK-ERR %s' % (name, str(ex)[:60]), flush=True); continue
        signal.alarm(180)
        try:
            t0 = time.perf_counter(); w = db.run(q); wm = time.perf_counter() - t0
            signal.alarm(0)
            w = w[0] if isinstance(w, tuple) else w
            st = 'OK' if same(w, e) else 'WRONG'
            detail = '' if st == 'OK' else ' wave=%s duck=%s' % (str(w[:1])[:60], str(e[:1])[:60])
            wt += wm; dt += dm
            if st == 'OK' and wm < dm: wins += 1
            print('%-4s %-5s wave=%7.2fs duck=%6.2fs x%6.2f%s' % (name, st, wm, dm, dm / wm if wm else 0, detail), flush=True)
        except NotImplementedError as ex:
            signal.alarm(0); st = 'HOLE'; msg = str(ex)[:70]
            holes[msg] = holes.get(msg, 0) + 1
            print('%-4s HOLE  %s' % (name, msg), flush=True)
        except BaseException as ex:
            signal.alarm(0); st = 'CRASH'
            print('%-4s CRASH %s: %s' % (name, type(ex).__name__, str(ex)[:70]), flush=True)
        tally[st] = tally.get(st, 0) + 1
    print('JOB BOARD: %d queries | %s | wins=%d | wave %.1fs duck %.1fs' % (
        sum(tally.values()), ' '.join('%s=%d' % kv for kv in sorted(tally.items())), wins, wt, dt), flush=True)
    if holes:
        print('HOLE FAMILIES:', flush=True)
        for msg, n in sorted(holes.items(), key=lambda kv: -kv[1]):
            print('  %3d  %s' % (n, msg), flush=True)

if __name__ == '__main__':
    main()
