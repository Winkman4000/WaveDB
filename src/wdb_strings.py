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


def chunks_touched(seg, col, u):
    """how many dictionary chunks hold at least one of the sorted codes u (the inflate bill of
    deciding only u), and the chunk count"""
    c = seg.cols[col]
    if not c.get('chunked'):
        return 1, 1
    CH = int(c['CHUNK']); nch = int(c['nch'])
    b = np.searchsorted(u, np.arange(nch + 1, dtype=np.int64) * CH)
    return int(np.count_nonzero(np.diff(b))), nch


def identify_contains_at(seg, col, u, n1, n2=b''):
    """IDENTIFICATION AT THE SURVIVORS: out[i] = the string of code u[i] contains n1 (then n2 after
    it), for sorted unique codes u only. Chunks holding none of u are never inflated; inside a
    chunk only the restart groups holding a needed code are walked, each from its restart to its
    last needed entry. Codes past the dictionary (the NULL bin) are False. None when the column is
    not front-coded."""
    import wdb_kernels as _WK, wdb_engine
    c = seg.cols.get(col) or {}
    if 'restarts' not in c:
        return None
    u = np.ascontiguousarray(u, dtype=np.int64)
    assert u.size == 0 or bool(np.all(u[1:] > u[:-1])), 'identify_contains_at wants sorted unique codes'
    R = int(c['R']); V0 = int(c.get('n_dict') or c['V'])
    a1 = np.frombuffer(n1 if isinstance(n1, bytes) else n1.encode(), np.uint8)
    a2 = np.frombuffer(n2 if isinstance(n2, bytes) else n2.encode(), np.uint8)
    out = np.zeros(u.size, np.bool_)
    live = int(np.searchsorted(u, V0))                  # u[:live] are real strings
    if live == 0:
        return out
    if not c.get('chunked'):
        raw = np.frombuffer(seg._dz.decompress(c['z']), np.uint8)
        _WK.plike_sel_fc(raw, np.asarray(c['restarts'], np.int64), np.int64(R), u[:live], a1, a2, out[:live])
        return out
    CH = int(c['CHUNK'])
    plan = _chunk_plan(seg, col)
    b = np.searchsorted(u[:live], np.arange(len(plan) + 1, dtype=np.int64) * CH)
    def _one(j):
        a, z = int(b[j]), int(b[j + 1])
        if a == z:
            return
        _jj, lo, n, rl = plan[j]
        assert lo == j * CH, ('chunk start', col, j, lo)
        need = u[a:z] - lo
        rl = np.asarray(rl, np.int64)
        if c.get('fc3'):
            h = seg.fc_part(c, j, 'h').view(np.uint16)
            gto = rl - 4 * R * np.arange(rl.size, dtype=np.int64)   # text offset of each restart
            _WK.plike_sel_fc3(h, seg.fc_part(c, j, 't'), gto, np.int64(R), need, a1, a2, out[a:z])
        else:
            _WK.plike_sel_fc(_chunk_bytes(seg, col, j), rl, np.int64(R), need, a1, a2, out[a:z])
    list(wdb_engine._leaf_pool().map(_one, [j for j in range(len(plan)) if b[j] < b[j + 1]]))
    return out


# THE POTENCY OF A SUBSET (Jackson's law: a filter earns its place only while the work it saves
# exceeds its own cost). Deciding the whole dictionary inflates every chunk and decides every entry
# -- the two halves measured about equal (URL 292 / 323 ms). Deciding only codes u inflates the
# chunks holding u and walks, in each restart group holding u, from its restart to its last needed
# entry. Bill in units of "one whole pass = 2.0"; the subset road is taken below AT_COST_SHARE of it.
AT_COST_SHARE = 0.8
AT_FORCE = [None]                     # tests: 'survivors' | 'whole' pins the road


def at_cost(seg, col, u):
    """(bill as a fraction of the whole pass, chunks touched, chunks, entries walked) for deciding
    only the sorted unique codes u (NULL-bin codes past the dictionary cost nothing)"""
    c = seg.cols[col]
    R = int(c['R']); V0 = int(c.get('n_dict') or c['V'])
    u = u[:int(np.searchsorted(u, V0))]
    if u.size == 0:
        return 0.0, 0, int(c.get('nch') or 1), 0
    touched, nch = chunks_touched(seg, col, u)
    g = u // R
    last = np.flatnonzero(np.r_[g[1:] != g[:-1], True])       # the last needed code of each group
    walked = int((u[last] % R + 1).sum())
    return (touched / nch + walked / V0) / 2.0, touched, nch, walked


def take_subset_road(seg, col, u):
    """True when deciding only u is cheaper than deciding the whole dictionary; and the bill"""
    bill = at_cost(seg, col, u)
    if AT_FORCE[0] is not None:
        return AT_FORCE[0] == 'survivors', bill
    return bill[0] < AT_COST_SHARE, bill
