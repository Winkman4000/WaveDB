"""THE SEGMENTS BOARD (B, step 1 -- the instrument): the same rows as ONE segment and as K
segments, the same 100 scope constructs, exact vs duck on both, timed on both. The output
is the cost of segmentation per query family, before any organ changes.

usage: python3 bench/board_segments.py gen K      build /workspace/data/scope10m/segdb as K loads
       python3 bench/board_segments.py run        board: single (db) vs segmented (segdb)
"""
import sys, os, time, json, shutil, subprocess, signal
sys.path.insert(0, 'src'); sys.path.insert(0, 'bench')

ROOT = '/workspace/data/scope10m'
SINGLE = ROOT + '/db'; SEG = ROOT + '/segdb'


def gen(K):
    import pyarrow.parquet as pq
    shutil.rmtree(SEG, ignore_errors=True); os.makedirs(SEG)
    t = pq.read_table(ROOT + '/x.parquet'); n = t.num_rows; step = (n + K - 1) // K
    for i in range(K):
        sl = t.slice(i * step, step); p = '%s/x_slice%d.parquet' % (ROOT, i); pq.write_table(sl, p)
        r = subprocess.run([sys.executable, 'bin/wdb', 'load', SEG, 'x', p, '--workers', '8'], capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:], flush=True)
    # the dimension table rides along as one segment (the scope joins need it)
    r = subprocess.run([sys.executable, 'bin/wdb', 'load', SEG, 'd', ROOT + '/d.parquet', '--workers', '4'], capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:], flush=True)
    cat = json.load(open(SEG + '/catalog.json'))
    print('segdb:', {t: len(v['segments']) for t, v in cat['tables'].items()}, flush=True)


def run():
    import duckdb
    from sql_scope import Q
    import wdb_kernels; wdb_kernels.warm()
    from wdb_db import Database
    dbs = {'single': Database.open(SINGLE), 'segmented': Database.open(SEG)}
    con = duckdb.connect()
    for tname in ('x', 'd'):
        con.execute("CREATE VIEW %s AS SELECT * FROM read_parquet('%s/%s.parquet')" % (tname, ROOT, tname))
    def nm(v): return ('%.6g' % v) if isinstance(v, float) else str(v)
    def norm(rows): return sorted(tuple(nm(v) for v in r) for r in rows)
    def _alarm(sig, frm): raise TimeoutError('timeout')
    signal.signal(signal.SIGALRM, _alarm)
    fam = {}; rows_out = []
    for cat, name, q in Q:
        try: e = con.execute(q).fetchall()
        except Exception: continue
        res = {}
        for tag, db in dbs.items():
            signal.alarm(180)
            try:
                db.run(q); t0 = time.perf_counter(); w = db.run(q); dt = time.perf_counter() - t0; signal.alarm(0)
                w = w[0] if isinstance(w, tuple) else w
                res[tag] = ('OK' if norm(w) == norm(e) else 'WRONG', dt)
            except NotImplementedError as ex:
                signal.alarm(0); res[tag] = ('HOLE', None, str(ex)[:50])
            except BaseException as ex:
                signal.alarm(0); res[tag] = ('CRASH', None, '%s: %s' % (type(ex).__name__, str(ex)[:40]))
        s1, sk = res['single'], res['segmented']
        ratio = (sk[1] / s1[1]) if (s1[1] and sk[1]) else None
        f = fam.setdefault(cat, {'n': 0, 'ok_single': 0, 'ok_seg': 0, 't_single': 0.0, 't_seg': 0.0, 'holes_seg': 0, 'wrong_seg': 0})
        f['n'] += 1; f['ok_single'] += s1[0] == 'OK'; f['ok_seg'] += sk[0] == 'OK'; f['holes_seg'] += sk[0] == 'HOLE'; f['wrong_seg'] += sk[0] in ('WRONG', 'CRASH')
        if s1[1] and sk[1]: f['t_single'] += s1[1]; f['t_seg'] += sk[1]
        print('%-6s %-22s single=%-5s %6s  segmented=%-5s %6s  cost=%s%s' % (cat, name, s1[0], ('%.2fs' % s1[1]) if s1[1] else '-', sk[0], ('%.2fs' % sk[1]) if sk[1] else '-',
              ('%.2fx' % ratio) if ratio else '-', ('  ' + sk[2]) if len(sk) > 2 else ''), flush=True)
    print('\nSEGMENTS BOARD (%d segments): family | n | single ok | segmented ok/hole/wrong | time single -> segmented (cost)' % len(json.load(open(SEG + '/catalog.json'))['tables']['x']['segments']))
    T1 = T2 = 0.0
    for cat, f in fam.items():
        T1 += f['t_single']; T2 += f['t_seg']
        print('  %-6s n=%-3d single %3d/%-3d  segmented %3d ok %2d hole %2d wrong   %6.1fs -> %6.1fs (%.2fx)' % (
            cat, f['n'], f['ok_single'], f['n'], f['ok_seg'], f['holes_seg'], f['wrong_seg'], f['t_single'], f['t_seg'], (f['t_seg'] / f['t_single']) if f['t_single'] else 0))
    print('  TOTAL %.1fs -> %.1fs (%.2fx)' % (T1, T2, T2 / T1 if T1 else 0), flush=True)


if __name__ == '__main__':
    gen(int(sys.argv[2])) if sys.argv[1] == 'gen' else run()
