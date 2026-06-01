#!/usr/bin/env python3
"""WaveDB engine — loads a WVDB3 segment, resolves columns by name from the header.
Generic: knows nothing about any specific dataset. Handles plain (mode 0) and
front-coded (mode 1) string dictionaries transparently."""
import struct, numpy as np, zstandard as zstd
_DT_UNITS = ['us','ns','ms','s','D','h','m','M','Y','W']

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
            has_null = buf[off]; off += 1; aux = buf[off]; off += 1
            n_dict = V - has_null
            meta = dict(V=V, bits=bits, dt=dt, mode=mode, has_null=has_null, n_dict=n_dict, aux=aux)
            if mode == 0:
                vals = []
                for _ in range(n_dict):
                    vl = struct.unpack_from('<I',buf,off)[0]; off += 4
                    vals.append(buf[off:off+vl]); off += vl
                meta['vals'] = vals
            elif mode == 2:
                zlen = struct.unpack_from('<I',buf,off)[0]; off += 4
                meta['z2'] = bytes(buf[off:off+zlen]); off += zlen
                meta['vals'] = None; meta['intvals'] = None
            else:
                Rr = struct.unpack_from('<H',buf,off)[0]; off += 2
                nr = struct.unpack_from('<I',buf,off)[0]; off += 4
                meta['restarts'] = np.frombuffer(buf, dtype=np.uint32, count=nr, offset=off); off += nr*4
                fclen = struct.unpack_from('<I',buf,off)[0]; off += 4
                zlen = struct.unpack_from('<I',buf,off)[0]; off += 4
                meta['R'] = Rr; meta['z'] = buf[off:off+zlen]; off += zlen
                meta['vals'] = None; meta['raw'] = None  # decoded lazily
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
    def _dict_ints(self, c):
        # mode 2: reconstruct sorted int64 dictionary from delta+zstd (once, cached)
        if c.get('intvals') is None:
            raw = self._dz.decompress(c['z2'])
            d = np.frombuffer(raw, dtype=np.int64)
            c['intvals'] = np.cumsum(d)
        return c['intvals']
    def dict_vals(self, nm):
        c = self.cols[nm]
        if c['vals'] is None: c['vals'] = self._decode_fc(c)
        return c['vals']
    def codes(self, nm):
        if nm in self._codes: return self._codes[nm]
        c = self.cols[nm]; bits = c['bits']; N = self.N; base = c['cstart']
        w = (1 << np.arange(bits-1,-1,-1)).astype(np.uint64)
        cc = np.empty(N, dtype=np.int64)
        CH = 2_000_000  # chunk rows so we never build the full N x bits matrix
        for lo in range(0, N, CH):
            hi = min(lo+CH, N)
            bit_lo = lo*bits; bit_hi = hi*bits
            byte_lo = bit_lo//8; byte_hi = (bit_hi+7)//8
            allb = np.unpackbits(self.buf[base+byte_lo:base+byte_hi])
            s = bit_lo - byte_lo*8
            b = allb[s:s+(hi-lo)*bits].reshape(hi-lo, bits)
            cc[lo:hi] = (b.astype(np.uint64)*w).sum(1).astype(np.int64)
        self._codes[nm] = cc; return cc
    def _typed_dict(self, nm):
        c = self.cols[nm]
        if c['mode'] == 2: return self._dict_ints(c)  # int64 array (dt 0/3)
        if c['dt'] == 0: return [int(v) for v in c['vals']]
        if c['dt'] == 2: return [struct.unpack('<d', v)[0] for v in c['vals']]
        if c['dt'] == 3: return [struct.unpack('<q', v)[0] for v in c['vals']]   # int64 epoch
        return self.dict_vals(nm)  # bytes
    def unit(self, nm):
        return _DT_UNITS[self.cols[nm]['aux']]
    def values(self, nm):
        c = self.cols[nm]; codes = self.codes(nm); dvals = self._typed_dict(nm)
        if c['has_null']:
            nullcode = c['V'] - 1
            lut = np.empty(c['V'], dtype=object)
            if c['dt'] == 3:
                unit = _DT_UNITS[c['aux']]
                for i, v in enumerate(dvals): lut[i] = np.int64(v).view(f'datetime64[{unit}]')
            else:
                for i, v in enumerate(dvals): lut[i] = v
            lut[nullcode] = None
            return lut[codes]
        if c['dt'] == 0: return np.array(dvals, dtype=np.int64)[codes]
        if c['dt'] == 2: return np.array(dvals, dtype=np.float64)[codes]
        if c['dt'] == 3:
            unit = _DT_UNITS[c['aux']]
            return np.array(dvals, dtype=np.int64)[codes].view(f'datetime64[{unit}]')
        return np.array(dvals, dtype=object)[codes]
    def fetch(self, nm, code):
        """Random-access the dictionary value for a given code. O(1) for plain columns,
        O(R) for front-coded columns (jump to restart block, walk <=R deltas)."""
        c = self.cols[nm]
        if c['has_null'] and code == c['V'] - 1: return None
        if c['mode'] == 2:
            arr = self._dict_ints(c)
            if c['dt'] == 3:
                unit = _DT_UNITS[c['aux']]
                return np.int64(arr[code]).view(f'datetime64[{unit}]')
            return int(arr[code])
        if c['mode'] == 0:
            v = c['vals'][code]
            if c['dt'] == 0: return int(v)
            if c['dt'] == 2: return struct.unpack('<d', v)[0]
            if c['dt'] == 3:
                unit = _DT_UNITS[c['aux']]
                return np.int64(struct.unpack('<q', v)[0]).view(f'datetime64[{unit}]')
            return v
        if c.get('raw') is None: c['raw'] = self._dz.decompress(c['z'])
        raw = c['raw']; R = c['R']; o = int(c['restarts'][code // R]); prev = b''
        for _ in range(code % R + 1):
            cp, sl = struct.unpack_from('<HH', raw, o); o += 4
            suf = raw[o:o+sl]; o += sl; prev = prev[:cp] + suf
        return prev
    def cardinality(self, nm): return self.cols[nm]['V']
    def group_by_count(self, nm):
        return np.bincount(self.codes(nm), minlength=self.cols[nm]['V'])
