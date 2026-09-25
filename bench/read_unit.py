"""How much does the machine actually pull in when we ask for 4 bytes? Evict a file, read 4 bytes at a
few far-apart spots (pread, then separately an mmap touch), then ask the kernel (mincore) which 4 KB
pages of the file are now in memory -- the size of each resident island is the real read unit.
Usage: python bench/read_unit.py FILE
"""
import sys, os, mmap, ctypes, time
import numpy as np

path = sys.argv[1]
size = os.path.getsize(path)
PG = os.sysconf('SC_PAGE_SIZE')
libc = ctypes.CDLL('libc.so.6', use_errno=True)
print('page size %d bytes; file %.1f MB' % (PG, size / 1e6))
try:
    dev = os.stat(path).st_dev
    print('readahead (blockdev) n/a on this mount; mount:', [l.split()[:3] for l in open('/proc/mounts') if ' /workspace ' in l])
except Exception:
    pass


def evict():
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)


def resident():
    mm = np.memmap(path, np.uint8, 'r')
    n = (size + PG - 1) // PG
    vec = (ctypes.c_ubyte * n)()
    r = libc.mincore(ctypes.c_void_p(mm.ctypes.data), ctypes.c_size_t(size), vec)
    assert r == 0, ctypes.get_errno()
    v = np.frombuffer(vec, np.uint8) & 1
    del mm
    return v.copy()


spots = [int(size * f) for f in (0.10, 0.30, 0.50, 0.70, 0.90)]
for how in ('pread', 'mmap touch'):
    evict()
    v0 = resident()
    fd = os.open(path, os.O_RDONLY)
    t = time.perf_counter()
    if how == 'pread':
        for s in spots:
            os.pread(fd, 4, s)
    else:
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
        for s in spots:
            _ = mm[s]
        mm.close()
    ms = (time.perf_counter() - t) * 1e3
    os.close(fd)
    v = resident()
    idx = np.flatnonzero(v)
    brk = np.flatnonzero(np.diff(idx) != 1) + 1
    isl = [(int(g[0]), int(g[-1] - g[0] + 1)) for g in np.split(idx, brk)] if idx.size else []
    print('%-10s 5 reads of 4 bytes -> %d pages resident (%.1f KB), islands (first page, pages): %s; %.1f ms'
          % (how, int(v.sum()), v.sum() * PG / 1024, isl[:8], ms))
