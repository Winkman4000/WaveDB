"""JOB-COUNT: every Join Order Benchmark query as COUNT(*) through its joins --
multiplicity through junction tables, vs the duckdb referee."""
import sys, os, time, glob, re, signal
sys.path.insert(0, 'src')
ROOT = '/workspace/data/job'

def main():
    import duckdb
    import wdb_kernels; wdb_kernels.warm()
    from wdb_db import Database
    db = Database.open(ROOT + '/db'); con = duckdb.connect(ROOT + '/imdb.duckdb', read_only=True)
    files = sorted(glob.glob(ROOT + '/q/*.sql'), key=lambda p: (int(''.join(ch for ch in os.path.basename(p) if ch.isdigit()) or 0), os.path.basename(p)))
    files = [f for f in files if os.path.basename(f)[0].isdigit()]
    def _alarm(sig, frm): raise TimeoutError('timeout')
    signal.signal(signal.SIGALRM, _alarm)
    if '--warm' in sys.argv:
        t0 = time.perf_counter(); n = 0
        for f in files:
            q = open(f).read().strip().rstrip(';'); q = re.sub(r'\bAS at\b', 'AS at1', q); q = re.sub(r'\bat\.', 'at1.', q)
            m = re.search(r'\bFROM\b', q); q = 'SELECT COUNT(*) AS c ' + q[m.start():]
            try: db.run(q); n += 1
            except Exception: pass
        print('WARM: %d queries in %.1fs' % (n, time.perf_counter() - t0), flush=True)
    tally = {}; wt = dt = 0.0; wins = 0; holes = {}
    for f in files:
        name = os.path.basename(f)[:-4]
        q = open(f).read().strip().rstrip(';')
        q = re.sub(r'\bAS at\b', 'AS at1', q); q = re.sub(r'\bat\.', 'at1.', q)
        m = re.search(r'\bFROM\b', q); q = 'SELECT COUNT(*) AS c ' + q[m.start():]
        try:
            t0 = time.perf_counter(); e = con.execute(q).fetchall(); dm = time.perf_counter() - t0
        except Exception as ex:
            print('%-4s DUCK-ERR %s' % (name, str(ex)[:60]), flush=True); continue
        signal.alarm(300)
        try:
            t0 = time.perf_counter(); w = db.run(q); wm = time.perf_counter() - t0; signal.alarm(0)
            w = w[0] if isinstance(w, tuple) else w
            st = 'OK' if [tuple(map(str, r)) for r in w] == [tuple(map(str, r)) for r in e] else 'WRONG'
            wt += wm; dt += dm; wins += (st == 'OK' and wm < dm)
            print('%-4s %-5s wave=%7.2fs duck=%6.2fs x%6.2f %s%s' % (name, st, wm, dm, dm / wm if wm else 0, w[0], '' if st == 'OK' else ' duck=%s' % e[0]), flush=True)
        except NotImplementedError as ex:
            signal.alarm(0); st = 'HOLE'; msg = str(ex)[:70]; holes[msg] = holes.get(msg, 0) + 1
            print('%-4s HOLE  %s' % (name, msg), flush=True)
        except BaseException as ex:
            signal.alarm(0); st = 'CRASH'; print('%-4s CRASH %s: %s' % (name, type(ex).__name__, str(ex)[:70]), flush=True)
        tally[st] = tally.get(st, 0) + 1
    print('JOB-COUNT BOARD: %d | %s | wins=%d | wave %.1fs duck %.1fs' % (sum(tally.values()), ' '.join('%s=%d' % kv for kv in sorted(tally.items())), wins, wt, dt), flush=True)
    for msg, n in sorted(holes.items(), key=lambda kv: -kv[1]): print('  %3d  %s' % (n, msg), flush=True)

if __name__ == '__main__':
    main()
