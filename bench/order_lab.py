"""THE ORDER LAB (for Jackson): row order is a compression parameter. Encode a slice of a
parquet under several cluster orders and print, per column, the blob bytes and whether the
column became a STAIRCASE (enc=2: free ordering for the window/range doors).
usage: python3 bench/order_lab.py PARQUET ROWS "EventTime" "EventDate,CounterID,EventTime" ...
"""
import sys, os, time, struct, tempfile
sys.path.insert(0, 'src')
import numpy as np

def slice_parquet(src, rows, dst):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(src); t = pf.read_row_groups(list(range(min(pf.num_row_groups, 64)))).slice(0, rows)
    pq.write_table(t, dst); return t.num_rows

def sizes(path):
    from wdb_engine import Segment
    s = Segment(path); raw = bytes(np.memmap(path, dtype=np.uint8, mode='r')[:])
    pos = []; off = 11
    for nm in s.order:
        pat = struct.pack('<H', len(nm)) + nm.encode(); i = raw.find(pat, off); pos.append(i); off = i + len(pat)
    pos.append(len(raw))
    return {nm: (pos[k + 1] - pos[k], s.cols[nm].get('mode'), s.cols[nm].get('code_enc')) for k, nm in enumerate(s.order)}

def main():
    import wdb_encode
    src, rows = sys.argv[1], int(sys.argv[2]); orders = sys.argv[3:] or ['']
    tmp = tempfile.mkdtemp(prefix='orderlab_'); sl = os.path.join(tmp, 'slice.parquet')
    n = slice_parquet(src, rows, sl); print('slice: %d rows' % n, flush=True)
    casts = {'EventDate': 'date_days', 'EventTime': 'timestamp_s'}
    results = {}
    for o in orders:
        keys = [k for k in o.split(',') if k]
        out = os.path.join(tmp, 'seg_%s.wdb' % ('_'.join(keys) or 'fileorder'))
        t0 = time.perf_counter()
        wdb_encode.encode(sl, out, stream=True, workers=8, casts=casts, cluster_by=(keys or None))
        results[o or '(file order)'] = (sizes(out), time.perf_counter() - t0)
    cols = list(next(iter(results.values()))[0].keys())
    hdr = '%-22s' % 'column' + ''.join('%18s' % (o or 'file order')[:18] for o in results)
    print(hdr)
    for c in sorted(cols, key=lambda c: -max(r[0][c][0] for r in results.values())):
        row = '%-22s' % c[:22]
        for o, (sz, _) in results.items():
            b, mode, enc = sz[c]
            row += '%12.1fMB%s' % (b / 1e6, ' STAIR' if enc == 2 else ('  m%s' % mode if mode == 4 else '     '))
        print(row)
    print('%-22s' % 'TOTAL' + ''.join('%14.1fMB %3.0fs' % (sum(v[0] for v in sz.values()) / 1e6, t) for sz, t in results.values()))

if __name__ == '__main__':
    main()
