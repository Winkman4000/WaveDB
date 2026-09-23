"""THE THREE STREAMS, converted in place of a reload: rewrite every chunked front-coded dictionary
of a database as headers + mask + text (wdb_encode writes the same layout on a fresh load).
Everything else -- other columns, every code section, the catalog, the ledger, the load
statistics -- is copied byte for byte. Each rewritten chunk is checked: the three streams must
rejoin to the original interleaved bytes exactly (FAIL-LOUD).

Usage: PYTHONPATH=src python bench/fc3_convert.py SRC_DB DST_DB
"""
import os, sys, shutil, struct, time
import numpy as np, zstandard as zstd
from concurrent.futures import ThreadPoolExecutor

import wdb_engine, wdb_encode, wdb_kernels as WK


def _split_chunk(seg, c, j, n, level):
    a = seg.fc_chunk(c, j)
    hdr = np.empty(4 * n, np.uint8); text = np.empty(a.size - 4 * n, np.uint8)
    mask = np.zeros(((text.size + 63) // 64) * 8, np.uint8)
    tl = int(WK.fc3_split(a, np.int64(n), hdr, text, mask))
    assert tl == text.size, ('split', j, tl, text.size)
    back = np.empty(a.size, np.uint8)
    assert int(WK.fc3_join(hdr, text, back)) == a.size and np.array_equal(back, a), ('rejoin differs', j)
    cz = zstd.ZstdCompressor(level=level)
    return cz.compress(hdr.tobytes()), cz.compress(mask.tobytes()), cz.compress(text.tobytes())


def convert_segment(src, dst, level=wdb_encode.ZSTD_LEVEL):
    seg = wdb_engine.Segment(src)
    buf = seg.buf
    ex = ThreadPoolExecutor(16)
    with open(dst, 'wb') as f:
        head_end = min(c['blob'][0] for c in seg.cols.values())
        f.write(bytes(buf[:head_end]))
        for nm in seg.order:
            c = seg.cols[nm]; b0, end = c['blob']
            if not (c.get('mode') == 1 and c.get('chunked') and not c.get('fc3')):
                for p in range(b0, end, 1 << 26):
                    f.write(bytes(buf[p:min(end, p + (1 << 26))]))
                continue
            t = time.perf_counter()
            nl = struct.unpack_from('<H', buf, b0)[0]
            aux_at = b0 + 2 + nl + 4 + 4                   # name, V u32, bits/dt/mode/has_null
            aux = int(buf[aux_at]); assert aux & 0x40 and not aux & 0x80, (nm, aux)
            nch = int(c['nch']); CH = int(c['CHUNK']); nd = int(c.get('n_dict') or c['V'])
            tab_at = int(c['chunk_base']) - 4 * nch      # the old frame-length table
            assert int(c['chunk_base']) + int(c['chunk_foff'][-1]) == int(c['code_off']), nm
            trip = list(ex.map(lambda j: _split_chunk(seg, c, j, min(CH, nd - j * CH), level), range(nch)))
            f.write(bytes(buf[b0:aux_at])); f.write(bytes([aux | 0x80]))
            f.write(bytes(buf[aux_at + 1:tab_at]))
            for k in range(3):
                f.write(np.array([len(x[k]) for x in trip], dtype=np.uint32).tobytes())
            for k in range(3):
                for x in trip: f.write(x[k])
            for p in range(int(c['code_off']), end, 1 << 26):
                f.write(bytes(buf[p:min(end, p + (1 << 26))]))
            old = int(c['code_off']) - b0; new = sum(len(x[0]) + len(x[1]) + len(x[2]) for x in trip)
            print('  %-14s %5d chunks  dictionary %8.1f -> %8.1f MB   (%.1fs)' % (
                nm, nch, int(c['chunk_foff'][-1]) / 1e6, new / 1e6, time.perf_counter() - t), flush=True)
    ex.shutdown()


def main(src_db, dst_db):
    os.makedirs(dst_db, exist_ok=False)
    for fn in sorted(os.listdir(src_db)):
        s = os.path.join(src_db, fn); d = os.path.join(dst_db, fn)
        if fn.endswith('.wdb'):
            print(fn, flush=True)
            convert_segment(s, d)
        elif os.path.isfile(s):
            shutil.copy2(s, d)
    # the new segment must parse, and every dictionary must read back equal to the source
    for fn in sorted(os.listdir(src_db)):
        if not fn.endswith('.wdb'):
            continue
        a = wdb_engine.Segment(os.path.join(src_db, fn)); b = wdb_engine.Segment(os.path.join(dst_db, fn))
        assert a.order == b.order and a.N == b.N
        for nm in a.order:
            ca, cb = a.cols[nm], b.cols[nm]
            if ca.get('mode') == 1 and ca.get('chunked'):
                assert cb.get('fc3'), nm
                for j in range(ca['nch']):
                    assert a.fc_chunk(ca, j, as_bytes=True) == b.fc_chunk(cb, j, as_bytes=True), (nm, j)
        print('verified', fn, os.path.getsize(os.path.join(src_db, fn)), '->', os.path.getsize(os.path.join(dst_db, fn)), 'bytes', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
