"""wdb_gbshelf -- Jackson's tiered-absence count shelf (.gbc2).

The census of per-code row counts, dressed by its own histogram:
count==1 stored as NOTHING (absence from every structure IS the answer),
count==2 as one presence bit, count==3 as one presence bit, count>=4 as a
presence bit plus a rank-ordered u16 tail with 512-code popcount
checkpoints. mmap-shaped: opening costs microseconds (the pickle it
replaces paid 65ms per query to answer one entry). Point lookups are three
independent bit tests plus at most one tail touch; bulk rebuilds are three
vectorized unpacks and one scatter.
"""
import os
import mmap
import numpy as np

_MAGIC = 0x67626332          # 'gbc2'
_CK = 512                    # checkpoint every 512 codes


def _path(seg, col):
    return seg.path + '.%s.gbc2' % col


def birth(seg, col):
    """Build the shelf from one bincount of the raw code stream. Returns True
    on success. Birth-on-first-touch: pays the pooled read once."""
    try:
        c = seg.cols[col]
        V = int(c['V'])
        cnt = np.bincount(np.asarray(seg._raw_codes(col)), minlength=V)
        if cnt.size and int(cnt.max()) >= 65536:
            return False                     # u16 tail law: this column outgrew it
        bm2 = np.packbits(cnt == 2)
        bm3 = np.packbits(cnt == 3)
        b4 = cnt >= 4
        bm4 = np.packbits(b4)
        nb = (V + 7) // 8
        # exclusive cumulative popcount of bm4 per _CK-code block
        pop8 = np.unpackbits(bm4[:nb]).reshape(-1, 8).sum(1)
        blocks = (V + _CK - 1) // _CK
        bp = np.zeros(blocks, np.int64)
        per = _CK // 8
        s = np.add.reduceat(pop8, np.arange(0, pop8.size, per))
        bp[:s.size] = s
        ck4 = np.concatenate([[0], np.cumsum(bp)[:-1]]).astype(np.uint32)
        tail = cnt[b4].astype(np.uint16)
        hdr = np.array([_MAGIC, V, tail.size, int(seg.N)], np.int64)
        with open(_path(seg, col), 'wb') as f:
            f.write(hdr.tobytes())
            f.write(bm2[:nb].tobytes())
            f.write(bm3[:nb].tobytes())
            f.write(bm4[:nb].tobytes())
            f.write(ck4.tobytes())
            f.write(tail.tobytes())
        return True
    except Exception:
        return False


def open_shelf(seg, col):
    """mmap the shelf; returns views dict or None. No caching needed: opening
    is priced in microseconds by design."""
    p = _path(seg, col)
    if not os.path.exists(p):
        return None
    try:
        f = open(p, 'rb')
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        hdr = np.frombuffer(mm, np.int64, 4)
        if int(hdr[0]) != _MAGIC or int(hdr[3]) != int(seg.N):
            return None
        V = int(hdr[1]); T = int(hdr[2])
        nb = (V + 7) // 8
        blocks = (V + _CK - 1) // _CK
        o = 32
        bm2 = np.frombuffer(mm, np.uint8, nb, o); o += nb
        bm3 = np.frombuffer(mm, np.uint8, nb, o); o += nb
        bm4 = np.frombuffer(mm, np.uint8, nb, o); o += nb
        ck4 = np.frombuffer(mm, np.uint32, blocks, o); o += 4 * blocks
        tail = np.frombuffer(mm, np.uint16, T, o)
        return {'V': V, 'bm2': bm2, 'bm3': bm3, 'bm4': bm4,
                'ck4': ck4, 'tail': tail, '_mm': mm, '_f': f}
    except Exception:
        return None


def _bit(bm, code):
    return (bm[code >> 3] >> (7 - (code & 7))) & 1


def point(sh, code):
    """count for one code: three independent bit tests, one tail touch."""
    if _bit(sh['bm4'], code):
        blk = code >> 9
        base = int(sh['ck4'][blk])
        start = blk << 6                      # first byte of this block
        endb = code >> 3
        seg9 = sh['bm4'][start:endb]
        r = int.from_bytes(seg9.tobytes(), 'big').bit_count() if seg9.size else 0
        last = int(sh['bm4'][endb]) >> (7 - (code & 7))
        r += bin(last).count('1') - 1         # bits at/above ours, minus ours
        return int(sh['tail'][base + r])
    if _bit(sh['bm3'], code):
        return 3
    if _bit(sh['bm2'], code):
        return 2
    return 1                                  # absence IS the answer


def bulk(sh):
    """Full census, vectorized: ones + tier bits + tail scatter."""
    V = sh['V']
    cnt = np.ones(V, np.int64)
    cnt += np.unpackbits(sh['bm2'])[:V]
    cnt += 2 * np.unpackbits(sh['bm3'])[:V]
    idx = np.flatnonzero(np.unpackbits(sh['bm4'])[:V])
    cnt[idx] = sh['tail']
    return cnt
