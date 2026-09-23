"""THE THREE READS (Jackson, 2026-09-23) -- the only ways the engine reads a string.

  DIFFERENTIATION  same or different. The dictionary code IS the differentiator (dictionaries are
                   stored sorted, so code order is string order: ORDER BY / MIN / MAX live here too).
                   String bytes read: none.
  IDENTIFICATION   is the thing at this code X, or does it have property P. Against a literal: one
                   dictionary lookup, then it is differentiation. Against a property (LIKE, length,
                   host): decided once per DISTINCT string, reading only the decisive bytes; a
                   decision made inside a shared prefix is inherited by the neighbour for free.
  RETURN           the full bytes -- only for the strings that go into the answer.

A front-coded dictionary entry is [shared-prefix length u16][suffix length u16][suffix bytes],
restarting every R entries; the dictionary is split into zstd chunks that begin on restarts.
Identification walks those chunks as they decompress -- never a whole-dictionary blob."""
import numpy as np


def differentiate(seg, col):
    """codes per row: equal codes <=> equal strings, code order == string order"""
    return seg.codes(col)


def retrieve(seg, col, codes):
    """the answer's strings: bytes for these codes only"""
    return [seg.fetch(col, int(c)) for c in np.asarray(codes).tolist()]


def _chunk_plan(seg, col):
    """[(chunk j, first code, code count, chunk-local restarts)] for a chunked front-coded dict;
    FAIL-LOUD if a chunk does not begin on a restart (the carry would start mid-chain)."""
    c = seg.cols[col]
    R = int(c['R']); V0 = int(c.get('n_dict') or c['V'])
    rs = np.asarray(c['restarts'], dtype=np.int64)
    us = np.asarray(c['chunk_ustart'], dtype=np.int64)
    nch = int(c['nch'])
    g = np.searchsorted(rs, us, side='left')
    plan = []
    for j in range(nch):
        g0 = int(g[j]); g1 = int(g[j + 1]) if j + 1 < nch else int(rs.size)
        assert g0 < rs.size and int(rs[g0]) == int(us[j]), ('chunk does not begin on a restart', col, j)
        lo = g0 * R; hi = min(g1 * R, V0)
        plan.append((j, lo, hi - lo, rs[g0:g1] - int(us[j])))
    return plan


def _chunk_bytes(seg, col, j):
    """chunk j interleaved (<cp sl>+suffix), either layout"""
    return seg.fc_chunk(seg.cols[col], j)


def charlens_chunk(seg, col, p, out):
    """IDENTIFICATION, length in characters, for one chunk of the plan into out[:n]. On the three
    streams: headers + mask only -- no text byte is read (Jackson's fixed width and mask)."""
    import wdb_kernels as _WK
    c = seg.cols[col]; j, lo, n, _rl = p; R = int(c['R'])
    if c.get('fc3'):
        h = seg.fc_part(c, j, 'h'); m = seg.fc_part(c, j, 'm')
        assert h.size == 4 * n and m.size % 8 == 0, ('fc3 chunk shape', col, j, h.size, n, m.size)
        return int(_WK.fc3_charlens(h.view(np.uint16), m.view(np.uint64), np.int64(R), out))
    return int(_WK.fc_charlens(_chunk_bytes(seg, col, j), np.int64(R), out))


def bytelens_chunk(seg, col, p, out):
    """length in bytes for one chunk: on the three streams it is the headers alone (cp + sl)"""
    import wdb_kernels as _WK
    c = seg.cols[col]; j, lo, n, _rl = p
    if c.get('fc3'):
        h = seg.fc_part(c, j, 'h').view(np.uint16)
        assert h.size == 2 * n, ('fc3 headers', col, j, h.size, n)
        out[:n] = h[0::2].astype(np.int64) + h[1::2]
        return n
    return int(_WK.fc_bytelens(_chunk_bytes(seg, col, j), np.int64(c['R']), out))


def identify_contains(seg, col, n1, n2=b''):
    """IDENTIFICATION by property: keep[code] = the string contains n1 (and n2 after it). One pass
    over the dictionary as stored, chunk by chunk in parallel, prefix carry inside each chain.
    Returns bool[V] (the null bin, if any, is False). None when the column is not front-coded."""
    import wdb_kernels as _WK, wdb_engine
    c = seg.cols.get(col) or {}
    if 'restarts' not in c:
        return None
    R = int(c['R']); V0 = int(c.get('n_dict') or c['V'])
    a1 = np.frombuffer(n1 if isinstance(n1, bytes) else n1.encode(), np.uint8)
    a2 = np.frombuffer(n2 if isinstance(n2, bytes) else n2.encode(), np.uint8)
    keep = np.zeros(int(c['V']), np.bool_)
    if not c.get('chunked'):
        raw = seg._dz.decompress(c['z'])
        _WK.plike_fc_serial(np.frombuffer(raw, np.uint8), np.asarray(c['restarts'], np.int64),
                            np.int64(R), np.int64(V0), a1, a2, keep)
        return keep
    def _one(p):
        j, lo, n, rl = p
        if c.get('fc3'):                         # headers + text, read in place: no rejoin
            h = seg.fc_part(c, j, 'h').view(np.uint16)
            assert h.size == 2 * n, ('fc3 headers', col, j, h.size, n)
            _WK.plike_fc3(h, seg.fc_part(c, j, 't'), np.int64(R), a1, a2, keep[lo:lo + n])
            return
        _WK.plike_fc_serial(_chunk_bytes(seg, col, j), rl, np.int64(R), np.int64(n), a1, a2, keep[lo:lo + n])
    list(wdb_engine._leaf_pool().map(_one, _chunk_plan(seg, col)))
    return keep
