s = open('/home/jack/WaveDB/src/wdb_engine.py').read()
old = '''        out = np.empty(self.N, dtype=np.bool_)
        import wdb_kernels as _WK15
'''
assert s.count(old) == 1
s = s.replace(old, '''        out = np.empty(self.N, dtype=np.bool_)
        import wdb_kernels as _WK15
        # SIBLING SHARE (Jackson's walk): every clock consumer drinks from
        # ONE decompression -- the band loads the cached streams (paying for
        # them exactly once per query) and later consumers ride free.
        Y5, M5, D5, DL5, BB5 = self._e15_streams(anm)
        _WK15.e15_band_from_streams(Y5, M5, D5, DL5, BB5, role,
                                    ca['ybase'], day_lo, day_hi, out)
        return out
''', 1)
open('/home/jack/WaveDB/src/wdb_engine.py', 'w').write(s)
print('band shares the streams')

s = open('/home/jack/WaveDB/src/wdb_kernels.py').read()
assert 'def e15_band_from_streams' not in s
s += '''

@njit(cache=True, parallel=True, nogil=True)
def e15_band_from_streams(Y, M, D, DL, BB, role, ybase, lo, hi, out):
    """Clock band over ALREADY-DECOMPRESSED streams (the sibling share)."""
    n = out.shape[0]
    for i in prange(n):
        mm = np.int64(M[i])
        y = np.int64(ybase) + np.int64(Y[i])
        if mm <= 1:
            y -= 1
        if y >= 0:
            era = y // 400
        else:
            era = (y - 399) // 400
        yoe = y - era * 400
        mp = (mm + 10) % 12
        doe = yoe * 365 + yoe // 4 - yoe // 100 + (153 * mp + 2) // 5 + np.int64(D[i])
        days = era * 146097 + doe - 719468
        b = (BB[i >> 3] >> (7 - (i & 7))) & 1
        if b != role:
            days += np.int64(DL[i])
        out[i] = (days >= lo) and (days < hi)
'''
open('/home/jack/WaveDB/src/wdb_kernels.py', 'w').write(s)
print('kernel in')
