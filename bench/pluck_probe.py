"""THE PLUCK PROBE (2026-09-24): which call sites pluck answer values one fetch at a time, and
what the same dictionary chunks cost cold read one after another against all at once.

Per query in a fresh process: (1) run warm with Segment.fetch wrapped -> the caller (file:line)
and the codes of every fetch; (2) for each (column, codes) set: evict the database files, read the
chunks those codes touch one call at a time (as the plucks do); evict again, read them in one
batched call (the pooled chunk pop). Mode-2 (integer) dictionaries only in this probe.

Usage: PYTHONPATH=src python bench/pluck_probe.py DB_DIR QUERIES_SQL q ...
"""
import sys, os, time, json, glob, subprocess, traceback, collections
import numpy as np


def evict(db_dir):
    for f in glob.glob(os.path.join(db_dir, '*')):
        if os.path.isfile(f):
            fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)


def one(db_dir, sql):
    import wdb_db, wdb_engine as W
    db = wdb_db.Database.open(db_dir)
    seen = collections.defaultdict(list)       # (segid, col, caller) -> codes
    segs = {}
    orig = W.Segment.fetch
    def f(self, nm, code):
        st = traceback.extract_stack(limit=3)[0]
        k = (id(self), nm, '%s:%d' % (os.path.basename(st.filename), st.lineno))
        segs[id(self)] = self; seen[k].append(int(code))
        return orig(self, nm, code)
    W.Segment.fetch = f
    db.run(sql)
    W.Segment.fetch = orig
    out = []
    for (sid, nm, caller), codes in seen.items():
        seg = segs[sid]; c = seg.cols[nm]
        rec = {'col': nm, 'caller': caller, 'n': len(codes), 'mode': c.get('mode')}
        if c.get('mode') == 2 and c.get('i2ch') is not None and c.get('intvals') is None:
            CH = int(c['i2ch']); chs = sorted(set(x // CH for x in codes if x < c['V']))
            rec['chunks'] = len(chs)
            rec['chunk_kb'] = round(float(np.mean([c['i2zoffs'][j + 1] - c['i2zoffs'][j] for j in chs])) / 1024, 1) if chs else 0
            idx = np.array([x for x in codes if x < c['V'] - (1 if c['has_null'] else 0)], np.int64)
            c['i2chunks'].clear(); evict(db_dir)
            t = time.perf_counter()
            for x in idx: seg._dict_ints_at(c, np.array([x]))
            rec['serial_ms'] = round((time.perf_counter() - t) * 1e3, 1)
            c['i2chunks'].clear(); evict(db_dir)
            t = time.perf_counter(); seg._dict_ints_at(c, idx)
            rec['batch_ms'] = round((time.perf_counter() - t) * 1e3, 1)
        out.append(rec)
    return out


if __name__ == '__main__':
    db_dir, qfile = sys.argv[1], sys.argv[2]
    qs = [l.strip() for l in open(qfile) if l.strip() and not l.strip().startswith('--')]
    todo = [int(x) for x in sys.argv[3:]]
    if len(todo) == 1:
        print(json.dumps({'q': todo[0], 'plucks': one(db_dir, qs[todo[0]])}))
    else:
        for q in todo:
            o = subprocess.run([sys.executable, __file__, db_dir, qfile, str(q)],
                               capture_output=True, text=True, env=os.environ)
            ls = o.stdout.strip().splitlines()
            print(ls[-1] if ls else json.dumps({'q': q, 'err': o.stderr[-400:]}), flush=True)
