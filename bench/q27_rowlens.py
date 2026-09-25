"""Q27 from a ROW-ORDER length column (ClickHouse's URL.size, in characters): each row's URL length stored
in row order, zstd per 65,536-row block with a block table -- the offsets ARE the answer, no URL identity
per row. The query reads the website numbers (unpacked on a thread, overlapped) and the length column
(blocks in parallel), then pours. Size, cold/hot time, and the answer against the engine's.
Usage: PYTHONPATH=src python bench/q27_rowlens.py DB_DIR {build|run}
"""
import sys, os, glob, time, json
import numpy as np
import numba

P = '/workspace/q27_rowlens'
FR = 65536
d, mode = sys.argv[1], sys.argv[2]
import wdb_db


@numba.njit(parallel=True, cache=True)
def pour(cc, L, K, T):
    n = cc.size
    S = np.zeros((T, K), np.int64); C = np.zeros((T, K), np.int64)
    per = (n + T - 1) // T
    for t in numba.prange(T):
        for i in range(t * per, min(n, (t + 1) * per)):
            l = L[i]
            if l > 0:
                S[t, cc[i]] += l; C[t, cc[i]] += 1
    return S.sum(0), C.sum(0)


if mode == 'build':
    import zstandard as zs
    db = wdb_db.Database.open(d)
    seg = db.open_segment(db.cat.segment_paths('hits')[0], 'hits')
    lens = np.asarray(seg.dict_charlens('URL'))
    L = lens[np.asarray(seg._raw_codes('URL'))].astype(np.uint16)
    cz = zs.ZstdCompressor(level=9)
    blocks = [cz.compress(L[i:i + FR].tobytes()) for i in range(0, L.size, FR)]
    off = np.zeros(len(blocks) + 1, np.int64); np.cumsum([len(b) for b in blocks], out=off[1:])
    with open(P + '.bin', 'wb') as f:
        f.write(b''.join(blocks))
    np.save(P + '_off.npy', off)
    print('row-order character lengths: %d rows, %.1f MB (%.2f bits/row); raw u16 would be %.1f MB'
          % (L.size, off[-1] / 1e6, off[-1] * 8 / L.size, L.size * 2 / 1e6))
    sys.exit(0)

db = wdb_db.Database.open(d)
seg = db.open_segment(db.cat.segment_paths('hits')[0], 'hits')
pour(np.zeros(4, np.uint16), np.zeros(4, np.uint16), 2, 2)            # kernel loaded, as a real process
src = os.path.dirname(os.path.abspath(wdb_db.__file__))
for f in [f for f in glob.glob(d + '/*') if os.path.isfile(f)] + glob.glob(src + '/__pycache__/*.nb*') + glob.glob(P + '*'):
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)


def run():
    import threading, zstandard as zs
    from concurrent.futures import ThreadPoolExecutor
    T = {}
    t0 = time.perf_counter()
    box = {}
    th = threading.Thread(target=lambda: box.setdefault('cc', np.asarray(seg._raw_codes('CounterID'))))
    th.start()                                            # the website numbers, overlapped
    off = np.load(P + '_off.npy'); N = int(seg.N); nb = off.size - 1
    L = np.empty(N, np.uint16)
    fd = os.open(P + '.bin', os.O_RDONLY)
    runs = [(a, min(nb, a + 24)) for a in range(0, nb, 24)]   # ~24 blocks a read, all lanes busy
    def task(ab):
        a, b = ab
        raw = os.pread(fd, int(off[b] - off[a]), int(off[a]))
        dz = zs.ZstdDecompressor()
        for j in range(a, b):
            x = np.frombuffer(dz.decompress(raw[off[j] - off[a]:off[j + 1] - off[a]]), np.uint16)
            L[j * FR:j * FR + x.size] = x
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(task, runs))
    os.close(fd)
    T['length column read'] = (time.perf_counter() - t0) * 1e3
    th.join(); T['website numbers ready'] = (time.perf_counter() - t0) * 1e3
    t1 = time.perf_counter()
    K = int(seg.cols['CounterID']['V'])
    S, C = pour(box['cc'], L, K, 16)
    keep = np.flatnonzero(C > 100000)
    avg = S[keep] / C[keep]
    o = np.argsort(-avg, kind='stable')[:25]
    vals = seg.values_at('CounterID', keep[o])
    rows = [(int(vals[i]), float(avg[o[i]]), int(C[keep[o[i]]])) for i in range(len(o))]
    T['pour + top 25'] = (time.perf_counter() - t1) * 1e3
    T['total'] = (time.perf_counter() - t0) * 1e3
    return rows, T


rows, T = run(); cold = T['total']
_, T2 = run()
qs = [l.strip() for l in open('benchmark/clickbench/queries.sql') if l.strip() and not l.strip().startswith('--')]
eng = [(int(a), float(b), int(c)) for a, b, c in db.run(qs[27])[0]]
same = [(a, round(b, 9), c) for a, b, c in rows] == [(a, round(b, 9), c) for a, b, c in eng]
print(json.dumps({'cold_ms': round(cold), 'hot_ms': round(T2['total']), 'stages_cold': {k: round(v) for k, v in T.items()},
                  'same_answer_as_engine': same}))
