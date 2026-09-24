"""Q40's last lookups, split: EventDate by column read vs by the staircase (position alone), and
URLHash by column read; then how many bytes it takes to tell the survivors' URLs apart
(Jackson: read only as many bytes as the population needs).
Usage: PYTHONPATH=src python bench/url_bytes.py DB_DIR
"""
import sys, time
import numpy as np
sys.path.insert(0, 'bench')
import link_ceiling as L

db, seg = L._open(sys.argv[1])
_, rows, _ = L.by_hand(seg, 'C')
rows = np.ascontiguousarray(rows)
c = seg.cols['URLHash']
print('URLHash meta:', {k: v for k, v in c.items() if not hasattr(v, '__len__') or isinstance(v, str)})
print('EventDate meta:', {k: v for k, v in seg.cols['EventDate'].items() if not hasattr(v, '__len__') or isinstance(v, str)})


def cold(label, fn):
    L._evict(seg)
    t = time.perf_counter(); r = fn(); ms = (time.perf_counter() - t) * 1e3
    t = time.perf_counter(); fn(); hot = (time.perf_counter() - t) * 1e3
    print('%-34s cold %6.1f ms  hot %6.1f ms' % (label, ms, hot), flush=True)
    return r

ed = cold('EventDate, column read', lambda: np.asarray(seg.codes_at('EventDate', rows)).astype(np.int64))
st = np.asarray(seg.stairs('EventDate'), np.int64)
ed2 = cold('EventDate, from position (stairs)', lambda: np.searchsorted(st, rows, side='right'))
print('   staircase dates equal column dates:', bool(np.array_equal(ed, ed2)))
uh = cold('URLHash codes, column read', lambda: np.asarray(seg.codes_at('URLHash', rows)).astype(np.int64))

u = np.unique(uh)
vals = np.array([int(seg.fetch('URLHash', int(x))) for x in u], dtype=np.int64).view(np.uint64)
print('survivors %d, distinct URLs among them %d, distinct (URL, day) piles %d'
      % (rows.size, u.size, np.unique(uh * 64 + ed2).size))
for nb in range(1, 9):
    m = np.uint64((1 << (8 * nb)) - 1) if nb < 8 else np.uint64(0xFFFFFFFFFFFFFFFF)
    lo = np.unique(vals & m).size
    hi = np.unique(vals >> np.uint64(64 - 8 * nb)).size
    cd = np.unique(u & ((1 << (8 * nb)) - 1)).size if nb < 8 else u.size
    print('  %d byte(s): low bytes of hash tell apart %d / %d, high bytes %d, low bytes of dict code %d'
          % (nb, lo, u.size, hi, cd))
fr = np.unique(rows // int(c['BR'])); print('URLHash frames touched by survivors: %d (of %d), first %d last %d; average compressed frame %.0f KB' % (fr.size, -(-int(seg.N) // int(c['BR'])), fr[0], fr[-1], int(c['czlen']) / 1024 / -(-int(seg.N) // int(c['BR']))))
print('URLHash dict size V = %d -> %d bits per code' % (int(c['V']), int(c['V']).bit_length()))
