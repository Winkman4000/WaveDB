"""URL lengths off the front-coded dictionary: characters (every suffix byte scanned for UTF-8
continuation bytes) against bytes (the prefix/suffix headers alone). Cold, fresh process each call
kind; plus how many distinct URLs have characters != bytes (non-ASCII).
Usage: PYTHONPATH=src python bench/url_lens.py DB_DIR {chars|bytes|census}
"""
import sys, os, glob, time
import numpy as np
import wdb_db

d, kind = sys.argv[1], sys.argv[2]
db = wdb_db.Database.open(d)
seg = db.open_segment(db.cat.segment_paths('hits')[0], 'hits')
src = os.path.dirname(os.path.abspath(wdb_db.__file__))
for f in [f for f in glob.glob(d + '/*') if os.path.isfile(f)] + glob.glob(src + '/__pycache__/*.nb*'):
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
t = time.perf_counter()
if kind == 'chars':
    x = seg.dict_charlens('URL')
elif kind == 'bytes':
    x = seg.dict_bytelens('URL')
else:
    c = np.asarray(seg.dict_charlens('URL')); b = np.asarray(seg.dict_bytelens('URL'))
    diff = int((c != b).sum())
    print('distinct URLs %d; characters != bytes on %d (%.2f%%); total chars %d, total bytes %d'
          % (c.size, diff, 100.0 * diff / c.size, int(c.sum()), int(b.sum())))
    sys.exit(0)
print('%s: %.0f ms cold, %d lengths' % (kind, (time.perf_counter() - t) * 1e3, np.asarray(x).size))
