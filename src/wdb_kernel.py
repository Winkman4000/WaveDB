"""
wdb_kernel -- lazily-compiled AVX-512 helper for the scanpair high-match path.

A two-stage read-less, hash-aggregating filtered 2-key top-K: scan the filter column C alone (in its
NATIVE width) for survivor positions, gather A/B only there, count the fused pairs with an
open-addressing hash table (counts, never sorts; memory scales with pairs that occur, not the full
key space). It is an OPTIONAL accelerator: compiled on first use with `gcc -march=native`, used only
when the build succeeds AND the CPU reports AVX-512; otherwise every caller falls back to the numpy
path in wdb_scanpair with identical results. Nothing here is required for correctness.

Critically, the code arrays are read in their NATIVE dtype (uint8/uint16/uint32) -- never upcast to
uint32 -- because an upcast of a 100M-element column per call erases the win.

scanpair_topk(cC, cA, cB, vC, Vb, k) -> (a_codes, b_codes, counts) for the top-k pairs by count,
or None to decline (kernel unavailable, or inputs outside the supported shape).
"""
import os, ctypes, subprocess, hashlib
import numpy as np

_LIB = 0           # 0 = not yet attempted; None = unavailable; else the loaded CDLL
_SRC = os.path.join(os.path.dirname(__file__), 'kernels', 'scanpair_kernel.c')


def _has_avx512():
    try:
        with open('/proc/cpuinfo') as f:
            return 'avx512f' in f.read()
    except Exception:
        return False


def _load():
    """Compile (once) and load the kernel, or return None if unavailable. Cached after first call.
    The .so is built next to the source, keyed by a hash of the source so a changed kernel rebuilds."""
    global _LIB
    if _LIB != 0:
        return _LIB
    _LIB = None
    if not _has_avx512() or not os.path.exists(_SRC):
        return None
    try:
        h = hashlib.md5(open(_SRC, 'rb').read()).hexdigest()[:10]
        so = os.path.join(os.path.dirname(_SRC), f'scanpair_kernel.{h}.so')
        if not os.path.exists(so):
            r = subprocess.run(
                ['gcc', '-O3', '-march=native', '-mavx512f', '-mavx512bw', '-mavx512vl',
                 '-mavx512dq', '-shared', '-fPIC', '-o', so, _SRC],
                capture_output=True, timeout=60)
            if r.returncode != 0 or not os.path.exists(so):
                return None
        lib = ctypes.CDLL(so)
        for nm in ('wdb_scan_u8', 'wdb_scan_u16', 'wdb_scan_u32'):
            getattr(lib, nm).restype = ctypes.c_int64
        lib.wdb_scan_u8.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint8, ctypes.c_void_p]
        lib.wdb_scan_u16.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint16, ctypes.c_void_p]
        lib.wdb_scan_u32.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint32, ctypes.c_void_p]
        lib.wdb_gather_fuse.restype = None
        lib.wdb_gather_fuse.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        lib.wdb_tally.restype = ctypes.c_int64
        lib.wdb_tally.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int] + [ctypes.c_void_p] * 4
        _LIB = lib
    except Exception:
        _LIB = None
    return _LIB


_SCAN = {1: 'wdb_scan_u8', 2: 'wdb_scan_u16', 4: 'wdb_scan_u32'}


def _capbits(m):
    """Pick the open-addressing table size (power-of-two slots) for m survivor rows.

    The table only needs to hold the DISTINCT pairs, which is usually far below the row count m.
    Sizing to 2*m badly oversizes when pairs repeat (the common high-match case): a 1GB table spends
    all its time zeroing and cache-missing. Measured: for ~1.8M distinct pairs, a 2M-slot table
    (load factor ~0.85, fits in cache) tallied ~2.4x faster than a 67M-slot one. So we size to ~2*m
    but CAP at 2^22 (4M slots, 64MB) -- past that, a higher load factor on a cache-resident table
    beats a sparse table that thrashes memory. Linear probing stays fine up to ~0.85 here."""
    b = 10
    target = max(m * 2, 1024)
    while (1 << b) < target:
        b += 1
    return min(b, 22)


def scanpair_topk(cC, cA, cB, vC, Vb, k):
    """Top-k (a,b) pairs by count among rows where C == vC. Returns (a_codes, b_codes, counts)
    int64 arrays of length <= k sorted by count desc, or None to decline.
    cC, cA, cB are native-dtype (uint8/uint16/uint32) code arrays of equal length; vC a single C
    code; Vb the B cardinality; k the LIMIT."""
    lib = _load()
    if lib is None:
        return None
    for arr in (cC, cA, cB):
        if arr.dtype.kind not in 'u' or arr.dtype.itemsize not in (1, 2, 4):
            return None
    N = cC.shape[0]
    if cA.shape[0] != N or cB.shape[0] != N or N >= (1 << 31):
        return None                                   # int32 positions; >2^31 rows out of scope
    cC = np.ascontiguousarray(cC); cA = np.ascontiguousarray(cA); cB = np.ascontiguousarray(cB)
    scan = getattr(lib, _SCAN[cC.dtype.itemsize])
    pos = np.empty(N, dtype=np.int32)
    m = scan(cC.ctypes.data, N, int(vC), pos.ctypes.data)
    if m == 0:
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))
    keys = np.empty(m, dtype=np.int64)
    lib.wdb_gather_fuse(cA.ctypes.data, cA.dtype.itemsize, cB.ctypes.data, cB.dtype.itemsize,
                        pos.ctypes.data, m, int(Vb), keys.ctypes.data)
    cb = _capbits(m)                                  # distinct <= m; size table to that
    cap = 1 << cb
    htab_k = np.empty(cap, np.int64); htab_c = np.empty(cap, np.int64)
    out_k = np.empty(cap, np.int64); out_c = np.empty(cap, np.int64)
    d = lib.wdb_tally(keys.ctypes.data, m, cb, htab_k.ctypes.data, htab_c.ctypes.data,
                      out_k.ctypes.data, out_c.ctypes.data)
    if d < 0:
        return None                                   # table full (distinct > capped size) -> fall back
    uk = out_k[:d]; cnt = out_c[:d]
    kk = min(k, d)
    if kk < d:
        part = np.argpartition(cnt, -kk)[-kk:]
        order = part[np.argsort(cnt[part])[::-1]]
    else:
        order = np.argsort(cnt)[::-1]
    uk = uk[order[:kk]]; cnt = cnt[order[:kk]]
    return (uk // Vb, uk % Vb, cnt)
