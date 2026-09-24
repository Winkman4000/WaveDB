"""Full integer-dictionary reads, cold: the old reader (one pread per chunk, 16 in flight) on the
old database against the new reader (neighbouring chunks share one pread of up to 8 MB, then
every chunk decompresses in parallel) on both databases. The small chunks must not make the
whole-dictionary read slower. Each read runs after evicting the database files.

Usage: PYTHONPATH=src python bench/i2_fullread.py OLD_DB NEW_DB col ...
"""
import sys, os, glob, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import wdb_engine


def evict(db):
    for f in glob.glob(os.path.join(db, '*')):
        if os.path.isfile(f):
            fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)


def old_reader(seg, c):
    nch = len(c['i2zoffs']) - 1; parts = [None] * nch
    def _popi(ch):
        import zstandard as _z
        a = c['i2base'] + int(c['i2zoffs'][ch]); b = c['i2base'] + int(c['i2zoffs'][ch + 1])
        return ch, np.cumsum(np.frombuffer(_z.ZstdDecompressor().decompress(seg.read_span(a, b)), dtype=np.int64))
    with ThreadPoolExecutor(max_workers=16) as ex:
        for ch, arr in ex.map(_popi, range(nch)): parts[ch] = arr
    return np.concatenate(parts)


def timed(db, fn):
    evict(db); t = time.perf_counter(); v = fn(); return (time.perf_counter() - t) * 1e3, v


if __name__ == '__main__':
    old_db, new_db = sys.argv[1], sys.argv[2]
    A = wdb_engine.Segment(glob.glob(os.path.join(old_db, '*.wdb'))[0])
    B = wdb_engine.Segment(glob.glob(os.path.join(new_db, '*.wdb'))[0])
    for nm in sys.argv[3:]:
        ca, cb = A.cols[nm], B.cols[nm]
        r = {'old reader, old db': [], 'new reader, old db': [], 'new reader, new db': []}
        for _ in range(3):
            dt, v0 = timed(old_db, lambda: old_reader(A, ca)); r['old reader, old db'].append(dt)
            ca['intvals'] = None; ca['i2chunks'].clear()
            dt, v1 = timed(old_db, lambda: A._dict_ints(ca)); r['new reader, old db'].append(dt)
            ca['intvals'] = None; ca['i2chunks'].clear()
            dt, v2 = timed(new_db, lambda: B._dict_ints(cb)); r['new reader, new db'].append(dt)
            cb['intvals'] = None; cb['i2chunks'].clear()
            assert np.array_equal(v0, v1) and np.array_equal(v0, v2), nm
        print('%-12s %s' % (nm, '   '.join('%s %s' % (k, sorted(round(x) for x in v)) for k, v in r.items())), flush=True)
