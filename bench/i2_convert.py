"""THE READ MATCHED TO THE QUESTION, converted in place of a reload: rewrite every chunked integer
dictionary (mode 2) of a database at I2CH values per chunk (wdb_encode writes the same on a fresh
load). Everything else -- other columns, every code section, the catalog, the ledger, the load
statistics -- is copied byte for byte. FAIL-LOUD: every rewritten dictionary must read back
equal to the source, value for value.

Usage: PYTHONPATH=src python bench/i2_convert.py SRC_DB DST_DB [values_per_chunk]
"""
import os, sys, shutil, struct, time
import numpy as np, zstandard as zstd
from concurrent.futures import ThreadPoolExecutor

import wdb_engine, wdb_encode


def convert_segment(src, dst, I2CH, level):
    seg = wdb_engine.Segment(src)
    buf = seg.buf
    ex = ThreadPoolExecutor(16)
    with open(dst, 'wb') as f:
        head_end = min(c['blob'][0] for c in seg.cols.values())
        f.write(bytes(buf[:head_end]))
        for nm in seg.order:
            c = seg.cols[nm]; b0, end = c['blob']
            if not (c.get('mode') == 2 and c.get('i2ch') is not None):
                for p in range(b0, end, 1 << 26):
                    f.write(bytes(buf[p:min(end, p + (1 << 26))]))
                continue
            t = time.perf_counter()
            nch = len(c['i2zoffs']) - 1
            sec0 = int(c['i2base']) - 4 * nch - 8 - 4               # sentinel, (i2ch, nch), lengths
            sec1 = int(c['i2base']) + int(c['i2zoffs'][-1])
            assert struct.unpack_from('<I', buf, sec0)[0] == 0xFFFFFFFF, (nm, 'no sentinel')
            assert struct.unpack_from('<II', buf, sec0 + 4) == (int(c['i2ch']), nch), (nm, 'header')
            v = np.asarray(seg._dict_ints(c), np.int64)
            def comp(a):
                return zstd.ZstdCompressor(level=level).compress(
                    np.diff(v[a:a + I2CH], prepend=np.int64(0)).astype(np.int64).tobytes())
            zs = list(ex.map(comp, range(0, v.size, I2CH)))
            f.write(bytes(buf[b0:sec0]))
            f.write(struct.pack('<I', 0xFFFFFFFF) + struct.pack('<II', I2CH, len(zs)))
            f.write(np.array([len(z) for z in zs], dtype=np.uint32).tobytes())
            for z in zs:
                f.write(z)
            for p in range(sec1, end, 1 << 26):
                f.write(bytes(buf[p:min(end, p + (1 << 26))]))
            print('  %-16s %6d -> %6d chunks   dictionary %8.2f -> %8.2f MB   (%.1fs)' % (
                nm, nch, len(zs), int(c['i2zoffs'][-1]) / 1e6, sum(len(z) for z in zs) / 1e6,
                time.perf_counter() - t), flush=True)
            c['intvals'] = None; v = None
    ex.shutdown()


def main(src_db, dst_db, I2CH):
    level = wdb_encode.ZSTD_LEVEL       # the encoder's own level for dictionaries (_serialize_column)
    os.makedirs(dst_db, exist_ok=False)
    for fn in sorted(os.listdir(src_db)):
        s = os.path.join(src_db, fn); d = os.path.join(dst_db, fn)
        if fn.endswith('.wdb'):
            print(fn, flush=True)
            convert_segment(s, d, I2CH, level)
        elif os.path.isfile(s):
            shutil.copy2(s, d)
    for fn in sorted(os.listdir(src_db)):
        if not fn.endswith('.wdb'):
            continue
        a = wdb_engine.Segment(os.path.join(src_db, fn)); b = wdb_engine.Segment(os.path.join(dst_db, fn))
        assert a.order == b.order and a.N == b.N
        for nm in a.order:
            ca, cb = a.cols[nm], b.cols[nm]
            assert ca.get('mode') == cb.get('mode'), nm
            if ca.get('mode') == 2 and ca.get('i2ch') is not None:
                assert int(cb['i2ch']) == I2CH, nm
                assert np.array_equal(a._dict_ints(ca), b._dict_ints(cb)), (nm, 'dictionary differs')
                ca['intvals'] = cb['intvals'] = None
            if 'code_off' in ca:
                ea = ca['blob'][1] - ca['code_off']; eb = cb['blob'][1] - cb['code_off']
                assert ea == eb, (nm, 'code section length')
        print('verified', fn, os.path.getsize(os.path.join(src_db, fn)), '->',
              os.path.getsize(os.path.join(dst_db, fn)), 'bytes', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8192)
