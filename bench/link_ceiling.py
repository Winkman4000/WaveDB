"""THE LINK CEILING (Jackson's idea, Q40 by hand): at encode time record that one column's value
points to another's ("RefererHash = X -> CounterID = 62") and store the exception rows. Then
  - the region shortcut: search RefererHash only inside the blocks where counter 62 can live
    (load-time min/max), and
  - the link shortcut: skip CounterID entirely -- the answer is the RefererHash rows minus the
    stored exceptions.
Nothing here is wired into the engine; this measures the ceiling the idea can reach.

Variants (each a fresh process, database opened first, files evicted, then timed):
  A  the engine as it stands (db.run Q40)
  D  RefererHash searched over the whole table, then CounterID checked on its rows
  B  region only: RefererHash searched inside counter 62's blocks, then CounterID checked
  C  region + link: RefererHash searched inside counter 62's blocks, minus the stored exceptions

Usage: PYTHONPATH=src python bench/link_ceiling.py DB_DIR [reps]
"""
import sys, os, glob, time, json, subprocess
import numpy as np

Q40 = ("SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 "
       "AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 "
       "AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 "
       "GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100")
RH = 3594120000172545465
EXC = '/workspace/link_q40.npz'          # the stored link: stands in for the load statistics file


def _open(dbdir):
    import wdb_db
    db = wdb_db.Database.open(dbdir)
    seg = db.open_segment(db.cat.segment_paths('hits')[0], 'hits')
    return db, seg


def _evict(seg):
    for p in glob.glob(seg.path + '*') + [EXC]:
        if os.path.isfile(p):
            fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)


def _blocks_runs(seg, col, code):
    """Row spans of the blocks whose load-time min/max can hold `code` (neighbours merged)."""
    import wdb_blockstats
    st = wdb_blockstats._from_load(seg, col)
    BR = wdb_blockstats._BR
    hit = np.flatnonzero((st['cmin'] <= code) & (st['cmax'] >= code))
    brk = np.flatnonzero(np.diff(hit) != 1) + 1
    return [(int(r[0]) * BR, min(int(seg.N), (int(r[-1]) + 1) * BR)) for r in np.split(hit, brk)], hit.size


def by_hand(seg, variant):
    """Returns (result rows, survivor row positions, stage ms)."""
    import wdb_funnel, wdb_wherescan
    T = {}; t = time.perf_counter()
    def mark(k):
        nonlocal t
        n = time.perf_counter(); T[k] = round((n - t) * 1e3, 1); t = n
    rh = wdb_funnel._code_of(seg, 'RefererHash', RH)
    c62 = wdb_funnel._code_of(seg, 'CounterID', 62)
    r0 = wdb_funnel._code_of(seg, 'IsRefresh', 0)
    ts = [c for c in (wdb_funnel._code_of(seg, 'TraficSourceID', v) for v in (-1, 6)) if c is not None]
    mark('literals -> codes')
    if variant == 'D':
        rows = wdb_funnel.positions(seg, 'RefererHash', rh)
        mark('RefererHash, whole table')
    else:
        runs, _ = _blocks_runs(seg, 'CounterID', c62)
        mark('counter 62 region (min/max)')
        rows = np.concatenate([np.asarray(wdb_wherescan._scan_eq(seg, 'RefererHash', rh, a, b), np.int64)
                               for a, b in runs])
        mark('RefererHash, inside region')
    T['rows after RefererHash'] = int(rows.size)
    if variant == 'C':
        exc = np.load(EXC)['exc']
        rows = rows[~np.isin(rows, exc, assume_unique=True)]
        mark('minus stored exceptions')
    else:
        rows = rows[np.asarray(seg.codes_at('CounterID', rows)) == c62]
        mark('CounterID check')
    rows = rows[np.asarray(seg.codes_at('IsRefresh', rows)) == r0]
    mark('IsRefresh')
    rows = rows[np.isin(np.asarray(seg.codes_at('TraficSourceID', rows)), ts)]
    mark('TraficSourceID')
    ed = np.asarray(seg.codes_at('EventDate', rows)).astype(np.int64)
    days = np.array([wdb_funnel._days_of(seg.fetch('EventDate', int(c))) for c in range(int(seg.cols['EventDate']['V']))])
    lo = wdb_funnel._days_of(np.datetime64('2013-07-01')); hi = wdb_funnel._days_of(np.datetime64('2013-07-31'))
    keep = (days[ed] >= lo) & (days[ed] <= hi)
    rows, ed = rows[keep], ed[keep]
    uh = np.asarray(seg.codes_at('URLHash', rows)).astype(np.int64)
    mark('EventDate + URLHash')
    VE = int(seg.cols['EventDate']['V'])
    key, cnt = np.unique(uh * VE + ed, return_counts=True)
    order = np.argsort(-cnt, kind='stable')[100:110]
    mark('group + sort')
    out = [(seg.fetch('URLHash', int(key[i] // VE)), seg.fetch('EventDate', int(key[i] % VE)), int(cnt[i]))
           for i in order]
    mark('labels')
    T['survivors'] = int(rows.size)
    return out, rows, T


def one(dbdir, variant):
    db, seg = _open(dbdir)
    _evict(seg)
    t = time.perf_counter()
    if variant == 'A':
        res = db.run(Q40)[0]; T = {}
    else:
        res, _, T = by_hand(seg, variant)
    cold = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    if variant == 'A':
        db.run(Q40)
    else:
        by_hand(seg, variant)
    hot = (time.perf_counter() - t) * 1e3
    return {'cold': round(cold, 1), 'hot': round(hot, 1), 'stages': T,
            'res': [(str(a), str(b), int(c)) for a, b, c in res]}


def prep(dbdir):
    """Encode-time stand-in: the exception rows of the link RefererHash=X -> CounterID=62."""
    import wdb_funnel
    db, seg = _open(dbdir)
    rh = wdb_funnel._code_of(seg, 'RefererHash', RH)
    c62 = wdb_funnel._code_of(seg, 'CounterID', 62)
    allrh = wdb_funnel.positions(seg, 'RefererHash', rh)
    exc = allrh[np.asarray(seg.codes_at('CounterID', allrh)) != c62].astype(np.int64)
    runs, nblk = _blocks_runs(seg, 'CounterID', c62)
    inside = sum(int(((exc >= a) & (exc < b)).sum()) for a, b in runs)
    np.savez(EXC, exc=exc)
    c = seg.cols['RefererHash']
    print('RefererHash: mode %s enc %s V %s' % (c.get('mode'), c.get('code_enc'), c.get('V')))
    print('RefererHash=X rows %d, of them counter 62: %d, exceptions %d (%d inside counter 62\'s %d blocks), stored %d bytes'
          % (allrh.size, allrh.size - exc.size, exc.size, inside, nblk, os.path.getsize(EXC)), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == '--one':
        print(json.dumps(one(sys.argv[2], sys.argv[3]))); sys.exit(0)
    dbdir = sys.argv[1]; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    prep(dbdir)
    ref = None
    for v in ('A', 'D', 'B', 'C'):
        runs = []
        for _ in range(reps):
            o = subprocess.run([sys.executable, __file__, '--one', dbdir, v], capture_output=True, text=True)
            if o.returncode:
                print(o.stderr[-3000:]); raise SystemExit(1)
            runs.append(json.loads(o.stdout.strip().splitlines()[-1]))
        cs = sorted(r['cold'] for r in runs); hs = sorted(r['hot'] for r in runs)
        res = runs[0]['res']
        if ref is None:
            ref = res
        print('%s cold %s median %.0f ms | hot median %.0f ms | answer matches engine: %s'
              % (v, cs, cs[len(cs) // 2], hs[len(hs) // 2], res == ref), flush=True)
        if runs[0]['stages']:
            print('   stages (cold, first run):', runs[0]['stages'], flush=True)
        if res != ref:
            print('   engine:', ref); print('   by hand:', res)
