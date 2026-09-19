"""THE COORDINATE ROAD (Jackson's blocks): a reverse road whose row numbers are stored as
(block, position-in-block) instead of absolute 32-bit rows.

A key's rows ascend, so they visit blocks in order; every (key, block) pair is one CONTAINER:
    header (uint32) = block_id << 16 | dense << 15 | (count - 1)       -- one word per container
    sparse container: `count` uint16 positions (count <= 4096)
    dense  container: a 65,536-bit bitmap as 4,096 uint16 words (count > 4096: the bitmap is smaller)
Per key: boffs (int32, its container range) -- the flat road's offs, at half the width; no row
count is stored: the walk sizes its output from the headers (a bitmap's rows are its popcount).
The read is what it was -- find the key in u, walk its containers, add the block base to each
position -- so a flat road's cost model holds; only the bytes move. Chosen at birth by measuring
the whole file against the whole flat file: a road whose keys scatter across blocks
(cast_info.movie_id: 15 rows in 15 blocks) stays flat, because each row would pay its own header.
"""
import numpy as np
from numba import njit

B = 65536
SHIFT = 16
DENSE_MIN = 4097            # more rows than this in one block: the 8 KB bitmap is smaller than the positions
WORDS = B // 16
SUFFIXES = ('u', 'boffs', 'hdr', 'pay')
_POP8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)


@njit(nogil=True, cache=True)
def _fill(pos, starts, counts, dense, bpos, pay):
    for b in range(starts.size):
        s = starts[b]; c = counts[b]; o = bpos[b]
        if dense[b]:
            for w in range(WORDS): pay[o + w] = 0
            for q in range(c):
                p = pos[s + q]
                pay[o + (p >> 4)] |= np.uint16(1 << (p & 15))
        else:
            for q in range(c):
                pay[o + q] = pos[s + q]


@njit(nogil=True, cache=True)
def _postings(u, boffs, hdr, bpos, pay, sk, pop8):
    """rows of the keys in sk (int64): searchsorted, then every container of each hit"""
    m = sk.size
    hits = np.empty(m, np.int64); nh = 0; total = 0
    for i in range(m):
        k = sk[i]
        lo = 0; hi = u.size
        while lo < hi:
            mid = (lo + hi) >> 1
            if u[mid] < k: lo = mid + 1
            else: hi = mid
        if lo < u.size and u[lo] == k:
            hits[nh] = lo; nh += 1
            for b in range(boffs[lo], boffs[lo + 1]):
                hv = hdr[b]
                if (hv >> 15) & 1:
                    o = bpos[b]
                    for w in range(WORDS):
                        word = pay[o + w]
                        total += pop8[word & 255] + pop8[word >> 8]
                else:
                    total += (hv & 0x7FFF) + 1
    out = np.empty(total, np.int64); p = 0
    for j in range(nh):
        h = hits[j]
        for b in range(boffs[h], boffs[h + 1]):
            hv = hdr[b]
            base = np.int64(hv >> 16) << SHIFT
            o = bpos[b]
            if (hv >> 15) & 1:
                for w in range(WORDS):
                    word = pay[o + w]
                    if word == 0: continue
                    wb = base + (w << 4)
                    for t in range(16):
                        if (word >> t) & 1:
                            out[p] = wb + t; p += 1
            else:
                c = (hv & 0x7FFF) + 1
                for q in range(c):
                    out[p] = base + pay[o + q]; p += 1
    return out


class CoordRoad:
    __slots__ = ('u', 'boffs', 'hdr', 'bpos', 'pay')

    def __init__(self, u, boffs, hdr, pay):
        self.u = np.asarray(u); self.boffs = np.asarray(boffs, dtype=np.int32)      # u keeps its width (int32 keys: half the file)
        self.hdr = np.asarray(hdr, dtype=np.uint32); self.pay = pay
        sizes = np.where((self.hdr >> 15) & 1, WORDS, (self.hdr & 0x7FFF) + 1).astype(np.int64)
        self.bpos = np.concatenate(([0], np.cumsum(sizes)[:-1])) if sizes.size else np.zeros(0, np.int64)

    @property
    def disk_bytes(self):
        return int(self.u.nbytes + self.boffs.nbytes + self.hdr.nbytes + self.pay.nbytes)

    @property
    def resident_bytes(self):
        """what lives in RAM once loaded (positions stay mmap'd)"""
        return int(self.u.nbytes + self.boffs.nbytes + self.hdr.nbytes + self.bpos.nbytes)

    nbytes = disk_bytes

    def rows(self, sk):
        return _postings(self.u, self.boffs, self.hdr, self.bpos, np.asarray(self.pay), np.asarray(sk, dtype=np.int64), _POP8)

    def arrays(self):
        return {'u': self.u, 'boffs': self.boffs, 'hdr': self.hdr, 'pay': np.asarray(self.pay)}


def plan(order, offs):
    """the containers of a flat road: (starts, counts, dense, bid, key_of_container) and the coordinate
    file's bytes beyond u -- boffs + headers + payload -- against the flat file's offs + order"""
    order = np.asarray(order); offs = np.asarray(offs, dtype=np.int64)
    k = np.diff(offs)
    grp = np.repeat(np.arange(k.size, dtype=np.int64), k)
    blk = order.astype(np.int64) >> SHIFT
    if order.size:
        chg = np.flatnonzero((np.diff(grp) != 0) | (np.diff(blk) != 0)) + 1
        starts = np.concatenate(([0], chg)).astype(np.int64)
    else:
        starts = np.zeros(0, np.int64)
    counts = np.diff(np.append(starts, order.size)).astype(np.int64)
    dense = counts >= DENSE_MIN
    bid = blk[starts] if starts.size else np.zeros(0, np.int64)
    kc = grp[starts] if starts.size else np.zeros(0, np.int64)
    pay_words = int(np.where(dense, WORDS, counts).sum())
    nbytes = offs.size * 4 + starts.size * 4 + pay_words * 2
    return starts, counts, dense, bid, kc, nbytes


def flat_bytes(order, offs):
    return int(np.asarray(offs).nbytes + np.asarray(order).nbytes)


def build(u, order, offs):
    """a CoordRoad from a flat road (u sorted keys; order grouped by key, ascending within a key; offs per key)"""
    order = np.asarray(order); offs = np.asarray(offs, dtype=np.int64)
    starts, counts, dense, bid, kc, _nb = plan(order, offs)
    V = offs.size - 1
    boffs = np.searchsorted(kc, np.arange(V + 1, dtype=np.int64), side='left')
    if bid.size and int(bid.max()) >= (1 << 16):
        raise ValueError('coordinate road: more than 65,536 blocks (N > 2^32)')
    if starts.size >= (1 << 31):
        raise ValueError('coordinate road: more than 2^31 containers')
    hdr = ((bid.astype(np.uint32) << 16) | (dense.astype(np.uint32) << 15)
           | (np.where(dense, 0, counts - 1).astype(np.uint32) & 0x7FFF)).astype(np.uint32)
    sizes = np.where(dense, WORDS, counts).astype(np.int64)
    bpos = np.concatenate(([0], np.cumsum(sizes)[:-1])) if sizes.size else np.zeros(0, np.int64)
    pay = np.empty(int(sizes.sum()), np.uint16)
    pos = (order.astype(np.int64) & (B - 1)).astype(np.uint16)
    _fill(pos, starts, counts, dense, bpos, pay)
    return CoordRoad(u, boffs.astype(np.int32), hdr, pay)


def save(fn, road):
    """fn is the road's stem ('<segment>.<col>.inv'); four .npy files, the rename law"""
    import os
    for nm, arr in road.arrays().items():
        p = '%s.%s.npy' % (fn, nm)
        np.save(p + '.tmp.npy', np.ascontiguousarray(arr)); os.replace(p + '.tmp.npy', p)


def load(fn):
    """None unless every part is present; u/boffs/hdr in RAM, pay mmap'd (read at slice time)"""
    import os
    paths = {nm: '%s.%s.npy' % (fn, nm) for nm in SUFFIXES}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    u = np.load(paths['u']); boffs = np.load(paths['boffs']); hdr = np.load(paths['hdr'])
    pay = np.load(paths['pay'], mmap_mode='r')
    return CoordRoad(u, boffs, hdr, pay)
