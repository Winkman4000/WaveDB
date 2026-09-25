"""The ceiling of Jackson's encode-time character lengths for Q27: if the load stored each dictionary
entry's CHARACTER length (bytes corrected for the multi-byte characters, flagged at encode), the query
reads that array instead of walking the dictionary. Stand-in: a u16 array per URL dictionary entry in a
file (raw and zstd), read cold in place of dict_charlens. Same answer is checked.
Usage: PYTHONPATH=src python bench/q27_storedlens.py DB_DIR {build|raw|zstd|off}
"""
import sys, os, glob, time, hashlib
import numpy as np

d, mode = sys.argv[1], sys.argv[2]
P = '/workspace/q27_charlens'
import wdb_db, wdb_lenagg
if mode == 'build':
    db = wdb_db.Database.open(d)
    seg = db.open_segment(db.cat.segment_paths('hits')[0], 'hits')
    cl = np.asarray(seg.dict_charlens('URL'))
    assert cl.max() < 65536
    a = cl.astype(np.uint16)
    a.tofile(P + '.u16')
    import zstandard as zs
    open(P + '.zst', 'wb').write(zs.ZstdCompressor(level=9).compress(a.tobytes()))
    print('entries %d; raw %.1f MB; zstd %.1f MB' % (a.size, os.path.getsize(P + '.u16') / 1e6, os.path.getsize(P + '.zst') / 1e6))
    sys.exit(0)
db = wdb_db.Database.open(d)
src = os.path.dirname(os.path.abspath(wdb_db.__file__))
for f in [f for f in glob.glob(d + '/*') if os.path.isfile(f)] + glob.glob(src + '/__pycache__/*.nb*') + glob.glob(P + '*'):
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
if mode != 'off':
    _fn = wdb_lenagg._fn_table
    def fn(seg, col, lkind):
        if col == 'URL':
            if mode == 'raw':
                return np.fromfile(P + '.u16', np.uint16).astype(np.int64)
            import zstandard as zs
            return np.frombuffer(zs.ZstdDecompressor().decompress(open(P + '.zst', 'rb').read()), np.uint16).astype(np.int64)
        return _fn(seg, col, lkind)
    wdb_lenagg._fn_table = fn
qs = [l.strip() for l in open('benchmark/clickbench/queries.sql') if l.strip() and not l.strip().startswith('--')]
t = time.perf_counter(); r = db.run(qs[27])[0]; cold = (time.perf_counter() - t) * 1e3
t = time.perf_counter(); db.run(qs[27]); hot = (time.perf_counter() - t) * 1e3
print('%-4s cold %4.0f ms  hot %4.0f ms  answer %s' % (mode, cold, hot, hashlib.md5(repr(r).encode()).hexdigest()[:10]))
