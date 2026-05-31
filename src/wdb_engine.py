#!/usr/bin/env python3
"""WaveDB engine — loads a WVDB3 segment, resolves columns by name from the header.
Generic: knows nothing about any specific dataset. Handles plain (mode 0) and
front-coded (mode 1) string dictionaries transparently."""
import struct, numpy as np, zstandard as zstd

class Segment:
    def __init__(self, path):
        buf = open(path,'rb').read(); assert buf[:5]==b'WVDB3', "not a WVDB3 segment"
        off = 5
        self.n_cols = struct.unpack_from('<H',buf,off)[0]; off += 2
        self.N = struct.unpack_from('<I',buf,off)[0]; off += 4
        self.cols = {}; self.order = []; self._dz = zstd.ZstdDecompressor()
        for _ in range(self.n_cols):
            nl = struct.unpack_from('<H',buf,off)[0]; off += 2
            nm = buf[off:off+nl].decode(); off += nl
            V = struct.unpack_from('<I',buf,off)[0]; off += 4
            bits = buf[off]; off += 1; dt = buf[off]; off += 1; mode = buf[off]; off += 1
            meta = dict(V=V, bits=bits, dt=dt, mode=mode)
            if mode == 0:
                vals = []
                for _ in range(V):
                    vl = struct.unpack_from('<I',buf,off)[0]; off += 4
                    vals.append(buf[off:off+vl]); off += vl
                meta['vals'] = vals
            else:
                Rr = struct.unpack_from('<H',buf,off)[0]; off += 2
                nr = struct.unpack_from('<I',buf,off)[0]; off += 4
                off += nr*4  # restart offsets (for random access; full decode not needed here)
                fclen = struct.unpack_from('<I',buf,off)[0]; off += 4
                zlen = struct.unpack_from('<I',buf,off)[0]; off += 4
                meta['R'] = Rr; meta['z'] = buf[off:off+zlen]; off += zlen
                meta['vals'] = None  # decoded lazily
            nb = (self.N*bits+7)//8; meta['cstart'] = off; off += nb
            self.cols[nm] = meta; self.order.append(nm)
        self.buf = np.frombuffer(buf, dtype=np.uint8); self._codes = {}
    def _decode_fc(self, c):
        raw = self._dz.decompress(c['z']); vals = []; prev = b''; o = 0; i = 0; R = c['R']
        while o < len(raw):
            if i % R == 0: prev = b''
            cp, sl = struct.unpack_from('<HH', raw, o); o += 4
            suf = raw[o:o+sl]; o += sl; s = prev[:cp]+suf; vals.append(s); prev = s; i += 1
        return vals
    def dict_vals(self, nm):
        c = self.cols[nm]
        if c['vals'] is None: c['vals'] = self._decode_fc(c)
        return c['vals']
    def codes(self, nm):
        if nm in self._codes: return self._codes[nm]
        c = self.cols[nm]; raw = self.buf[c['cstart']:c['cstart']+(self.N*c['bits']+7)//8]
        b = np.unpackbits(raw)[:self.N*c['bits']].reshape(self.N, c['bits'])
        w = (1 << np.arange(c['bits']-1,-1,-1)).astype(np.uint64)
        cc = (b*w).sum(1).astype(np.int64); self._codes[nm] = cc; return cc
    def values(self, nm):
        c = self.cols[nm]
        if c['dt'] == 0:
            iv = np.array([int(v) for v in c['vals']], dtype=np.int64); return iv[self.codes(nm)]
        vb = self.dict_vals(nm); return np.array(vb, dtype=object)[self.codes(nm)]
    def cardinality(self, nm): return self.cols[nm]['V']
    def group_by_count(self, nm):
        return np.bincount(self.codes(nm), minlength=self.cols[nm]['V'])
