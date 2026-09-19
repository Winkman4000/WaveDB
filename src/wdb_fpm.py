"""
wdb_fpm — frame-presence map: which dictionary codes appear in which zstd frame.

Jackson's granularity find: CounterID = 62 lives in 25 of 191 frames, yet the
equality filter decompressed all 191 (7.6x read amplification). One bit per
(code, frame) — V x F bits, 155KB for CounterID — and an eq-hunt pops ONLY the
frames that contain its code. The survivors then span those same few frames, so
every downstream codes_at collapses by the same factor.

Species law: disk sidecar at V-scale (V x F bits; F is file-structural, ~191),
born lazily on first eligible eq-touch, persisted as <seg>.<col>.fpm, staleness-
guarded by (N, V, BR). Fail-closed: absent/ineligible -> caller keeps the full
read. Exact by construction: the map only SKIPS frames proven empty of the code.
"""
import os, pickle, numpy as np
import wdb_qmem

_HITS = 0
_CACHE = wdb_qmem.register({})   # (seg.path, col) -> unpacked bit matrix


def _path(seg, col):
    return f"{seg.path}.{col}.fpm"


def _eligible(seg, col):
    c = seg.cols.get(col)
    return (c is not None and c.get('code_enc') == 3
            and int(c.get('V', 1 << 30)) <= 65536 and 'BR' in c)


def _load_or_birth(seg, col):
    """The map for (seg, col): from cache, from disk, or born from one full read
    (a toll the very query asking was about to pay anyway)."""
    key = (seg.path, col)
    m = _CACHE.get(key)
    if m is not None:
        return m
    c = seg.cols[col]
    N = int(seg.N); V = int(c['V']); BR = int(c['BR'])
    F = (N + BR - 1) // BR
    p = _path(seg, col)
    if os.path.exists(p):
        try:
            with open(p, 'rb') as fh:
                hdr = pickle.load(fh)
            if hdr['N'] == N and hdr['V'] == V and hdr['BR'] == BR:
                bits = np.unpackbits(hdr['bits'], count=V * F).reshape(V, F).astype(bool)
                _CACHE[key] = bits
                return bits
        except Exception:
            pass                                  # stale/corrupt: rebirth below
    codes = np.asarray(seg._raw_codes(col))
    bits = np.zeros((V, F), bool)
    for f in range(F):
        seen = np.unique(codes[f * BR: (f + 1) * BR])
        bits[seen, f] = True
    try:
        import wdb_sidecar, os as _os9
        if not wdb_sidecar.births_on(_os9.path.dirname(p)): raise OSError('sidecars off')   # THE SWITCH
        with open(p, 'wb') as fh:
            pickle.dump({'N': N, 'V': V, 'BR': BR,
                         'bits': np.packbits(bits)}, fh)
    except Exception:
        pass                                      # disk full etc.: serve from RAM cache
    _CACHE[key] = bits
    return bits


def eq_positions(seg, col, code):
    """Row positions where col == code, popping ONLY the frames whose presence bit is
    set. Returns None when ineligible (caller keeps its full read)."""
    global _HITS
    if not _eligible(seg, col):
        return None
    c = seg.cols[col]
    N = int(seg.N); BR = int(c['BR'])
    bits = _load_or_birth(seg, col)
    frames = np.flatnonzero(bits[int(code)])
    if frames.size == 0:
        return np.empty(0, np.int64)
    if frames.size * 2 > bits.shape[1]:          # COMMON code (over half the frames):
        return None                              # materializing millions of positions
                                                 # loses to the block scan -- decline
    out = []
    for f in frames.tolist():
        lo = f * BR
        hi = min(lo + BR, N)
        seg8 = np.asarray(seg._raw_codes_range(col, lo, hi))
        loc = np.flatnonzero(seg8 == int(code))
        if loc.size:
            out.append(loc.astype(np.int64) + lo)
    _HITS += 1
    return np.concatenate(out) if out else np.empty(0, np.int64)
