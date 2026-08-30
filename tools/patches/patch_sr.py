import numpy as np
s = open('/home/jack/WaveDB/src/wdb_engine.py').read()
old = '''        answer from their own point paths (bitpack arithmetic, stair searchsorted, full decode)."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return np.empty(0, dtype=np.int64)
        c = self.cols[nm]
        if c.get('code_enc') == 13:
'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '''        answer from their own point paths (bitpack arithmetic, stair searchsorted, full decode)."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return np.empty(0, dtype=np.int64)
        c = self.cols[nm]
        if c.get('code_enc') == 14 and nm not in self._codes:
            # SURVIVOR READ (the cascade's prerequisite organ): streams once
            # (query-cached), civil math + LUT only at the asked rows.
            cache14 = getattr(self, '_e14_pl', None)
            pls = (cache14 or {}).get(nm)
            if pls is None:
                pls = self._e14_planes(nm)
                if cache14 is None:
                    cache14 = self._e14_pl = {}
                cache14[nm] = pls
            inv14, dmin14 = self._e14_inv_of(nm)
            out14 = np.empty(rows.size, dtype=np.int64)
            import wdb_kernels as _WKa
            _WKa.e14_reconstruct_at(pls[0], pls[1], pls[2], rows,
                                    c['ybase'], inv14, dmin14, out14)
            return out14
        if c.get('code_enc') in (15, 16) and nm not in self._codes:
            anm15 = c['e16_partner'] if c['code_enc'] == 16 else nm
            role15 = 0 if c['code_enc'] == 16 else 1
            sts = self._e15_streams(anm15)
            inv15, dmin15 = self._e14_inv_of(nm)
            out15 = np.empty(rows.size, dtype=np.int64)
            import wdb_kernels as _WKa
            _WKa.e15_reconstruct_at(sts[0], sts[1], sts[2], sts[3], sts[4], rows,
                                    role15, self.cols[anm15]['ybase'], inv15, dmin15, out15)
            return out15
        if c.get('code_enc') == 13:
'''
, 1)
old = '''    def _e15_band(self, nm, day_lo, day_hi):
'''
assert s.count(old) == 1
s = s.replace(old, '''    def _e15_streams(self, anm):
        """All five clock streams of an anchor column, decompressed once and
        CACHED per query (the flush law clears _e14_pl)."""
        cache = getattr(self, '_e14_pl', None)
        if cache is None:
            cache = self._e14_pl = {}
        key = (anm, 'e15s')
        got = cache.get(key)
        if got is not None:
            return got
        ca = self.cols[anm]
        base = ca['cstart']; FR = ca['e15_FR']; nfr = ca['e15_nfr']; offs = ca['e15_offs']
        st = [base]
        for _k in range(4):
            st.append(st[-1] + int(offs[_k][-1]))
        outs = [np.empty(self.N, np.uint8) for _ in range(4)]
        outs.append(np.empty((self.N + 7) >> 3, np.uint8))
        def _ws(job):
            import zstandard as _zs15
            p9, j9 = job
            a9 = st[p9] + int(offs[p9][j9]); b9 = st[p9] + int(offs[p9][j9 + 1])
            raw = np.frombuffer(_zs15.ZstdDecompressor().decompress(
                self.buf[a9:b9].tobytes()), np.uint8)
            o9 = j9 * ((FR >> 3) if p9 == 4 else FR)
            outs[p9][o9:o9 + raw.size] = raw
        jobs = [(p9, j9) for p9 in range(5) for j9 in range(nfr)]
        if len(jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor as _TPs
            with _TPs(max_workers=min(len(jobs), 8)) as exs:
                list(exs.map(_ws, jobs))
        else:
            _ws(jobs[0])
        cache[key] = outs
        return outs

    def _e15_band(self, nm, day_lo, day_hi):
'''
, 1)
old = '''    def plane_test(self, nm, day_lo, day_hi):
'''
assert s.count(old) == 1
s = s.replace(old, '''    def _e14_inv_of(self, nm):
        """The inverse day->code LUT for a date-dressed column, cached."""
        td = np.asarray(self._typed_dict(nm)).astype(np.int64)
        inv_map = getattr(self, '_e14_inv', None)
        if inv_map is None:
            inv_map = self._e14_inv = {}
        dmin = int(td[0])
        inv = inv_map.get(nm)
        if inv is None:
            inv = np.zeros(int(td[-1]) - dmin + 1, dtype=np.uint32)
            inv[td - dmin] = np.arange(td.size, dtype=np.uint32)
            inv_map[nm] = inv
        return inv, dmin

    def plane_test(self, nm, day_lo, day_hi):
'''
, 1)
open('/home/jack/WaveDB/src/wdb_engine.py', 'w').write(s)
print('ENGINE survivor reads')
