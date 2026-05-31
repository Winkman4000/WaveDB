#!/usr/bin/env python3
"""WaveDB engine — loads a WVDB2 segment, resolves columns by name from the header.
Generic: knows nothing about any specific dataset. Provides decode, group-by, count, distinct."""
import struct, numpy as np

class Segment:
    def __init__(self, path):
        buf = open(path,'rb').read(); assert buf[:5]==b'WVDB2', "not a WVDB2 segment"
        off = 5
        self.n_cols = struct.unpack_from('<H',buf,off)[0]; off += 2
        self.N = struct.unpack_from('<I',buf,off)[0]; off += 4
        self.cols = {}; self.order = []
        for _ in range(self.n_cols):
            nl = struct.unpack_from('<H',buf,off)[0]; off += 2
            nm = buf[off:off+nl].decode(); off += nl
            V = struct.unpack_from('<I',buf,off)[0]; off += 4
            bits = buf[off]; off += 1; dt = buf[off]; off += 1
            vals = []
            for _ in range(V):
                vl = struct.unpack_from('<I',buf,off)[0]; off += 4
                vals.append(buf[off:off+vl]); off += vl
            self.cols[nm] = dict(off=off, bits=bits, V=V, dt=dt, vals=vals)
            off += (self.N*bits+7)//8; self.order.append(nm)
        self.buf = np.frombuffer(buf, dtype=np.uint8)
        self._codes = {}
    def codes(self, nm):
        if nm in self._codes: return self._codes[nm]
        c = self.cols[nm]; raw = self.buf[c['off']:c['off']+(self.N*c['bits']+7)//8]
        b = np.unpackbits(raw)[:self.N*c['bits']].reshape(self.N, c['bits'])
        w = (1 << np.arange(c['bits']-1,-1,-1)).astype(np.uint64)
        cc = (b*w).sum(1).astype(np.int64); self._codes[nm] = cc; return cc
    def values(self, nm):
        c = self.cols[nm]
        if c['dt'] == 0:
            iv = np.array([int(v) for v in c['vals']], dtype=np.int64)
            return iv[self.codes(nm)]
        vb = c['vals']; return np.array([vb[x] for x in self.codes(nm)], dtype=object)
    def cardinality(self, nm): return self.cols[nm]['V']
    def group_by_count(self, nm):
        cc = self.codes(nm); return np.bincount(cc, minlength=self.cols[nm]['V'])
