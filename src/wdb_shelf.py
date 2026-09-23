"""THE SHELF: one process-wide store for resident derived objects (decoded
dictionaries, inline-string text, reverse-road uniques, decoded key arrays)
with a byte CEILING and LRU eviction. The engine never crashes itself: past
the ceiling the least-recently-used objects go, and an object larger than
the whole ceiling is refused by name (the caller works without it).

Ceiling: WDB_SHELF_MB, default 25% of physical RAM (or of the cgroup limit
when that is smaller).  db.stats() / shelf.stats() answer "why is RAM high".
"""
import os, sys, time, threading
from collections import OrderedDict


def _physical_bytes():
    try:
        b = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except Exception:
        b = 16 << 30
    try:
        with open('/sys/fs/cgroup/memory.max') as f:
            v = f.read().strip()
            if v.isdigit(): b = min(b, int(v))
    except Exception:
        pass
    return b


def ceiling_bytes():
    v = os.environ.get('WDB_SHELF_MB')
    if v:
        try: return int(float(v)) << 20
        except Exception: pass
    return _physical_bytes() // 4


class ShelfRefused(Exception):
    pass


class Shelf:
    def __init__(self):
        self._items = OrderedDict()      # key -> (obj, nbytes, kind, born)
        self._bytes = 0
        self._lock = threading.Lock()
        self.evictions = 0
        self.refusals = 0

    def get(self, key):
        with self._lock:
            it = self._items.get(key)
            if it is None: return None
            self._items.move_to_end(key)
            return it[0]

    def put(self, key, obj, nbytes, kind='derived'):
        nbytes = int(nbytes)
        cap = ceiling_bytes()
        if nbytes > cap:
            self.refusals += 1
            raise ShelfRefused('%s (%.1f MB) exceeds the shelf ceiling (%.0f MB)' % (kind, nbytes / 1e6, cap / 1e6))
        with self._lock:
            if key in self._items:
                self._bytes -= self._items[key][1]; del self._items[key]
            while self._items and self._bytes + nbytes > cap:
                k0, (o0, b0, kind0, t0) = self._items.popitem(last=False)     # LRU out
                self._bytes -= b0; self.evictions += 1
            self._items[key] = (obj, nbytes, kind, time.time())
            self._bytes += nbytes
        return obj

    def drop(self, key):
        with self._lock:
            it = self._items.pop(key, None)
            if it is not None: self._bytes -= it[1]

    def keep_only(self, kinds):
        """THE QUERY'S END (wdb_qmem): only the kinds Jackson's law lets outlive a query stay."""
        with self._lock:
            for k in [k for k, it in self._items.items() if it[2] not in kinds]:
                self._bytes -= self._items[k][1]; del self._items[k]

    def drop_prefix(self, prefix):
        with self._lock:
            for k in [k for k in self._items if str(k).startswith(prefix)]:
                self._bytes -= self._items[k][1]; del self._items[k]

    def stats(self, print_out=True):
        with self._lock:
            by = {}
            for k, (o, b, kind, t) in self._items.items():
                d = by.setdefault(kind, [0, 0]); d[0] += 1; d[1] += b
            rows = sorted(by.items(), key=lambda kv: -kv[1][1])
            tot = self._bytes; n = len(self._items)
        if print_out:
            print('SHELF: %d objects, %.2f GB of %.2f GB ceiling | evictions=%d refusals=%d' % (
                n, tot / 1e9, ceiling_bytes() / 1e9, self.evictions, self.refusals))
            for kind, (c, b) in rows:
                print('  %-22s x%-5d %7.2f GB' % (kind, c, b / 1e9))
        return {'objects': n, 'bytes': tot, 'ceiling': ceiling_bytes(), 'by_kind': {k: v for k, v in rows}}


SHELF = Shelf()

# What may outlive a query (wdb_qmem, Jackson's law): a decoded DICTIONARY is V-scale vocabulary of
# the source data -- a buffer-pool of the file's own values, cleared before every cold run anyway.
# Everything a query COMPUTED (roads, keys, ranks, orders, predicates, settled rows, block stats,
# inline N-scale text) dies with the query that computed it.
VOCABULARY = frozenset({'dictionary', 'joined-text', 'dictionary-sarray', 'inline-text'})


def vocabulary():
    """TIER 1 on the shelf (decoded source data); empty under WDB_HOT_KEEP=0, the pure-cold A/B."""
    return frozenset() if os.environ.get('WDB_HOT_KEEP', '1') == '0' else VOCABULARY


def nbytes_of(obj):
    """Best-effort resident size: numpy arrays exactly; lists of bytes sampled."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                n = obj.shape[0]; step = max(1, n // 4096)
                samp = obj[::step]
                per = sum((len(x) if isinstance(x, (bytes, bytearray, str)) else 8) + 56 for x in samp) / max(1, len(samp))
                return int(obj.nbytes + per * n)
            return int(obj.nbytes)
        if isinstance(obj, (bytes, bytearray)):
            return len(obj)
        if isinstance(obj, (list, tuple)):
            n = len(obj); step = max(1, n // 4096)
            samp = obj[::step]
            per = sum((len(x) if isinstance(x, (bytes, bytearray, str)) else 8) + 56 for x in samp) / max(1, len(samp))
            return int(per * n + 8 * n)
    except Exception:
        pass
    return 64
