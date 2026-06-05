#!/usr/bin/env python3
"""WaveDB engine — loads a WVDB3 segment, resolves columns by name from the header.
Generic: knows nothing about any specific dataset. Handles plain (mode 0) and
front-coded (mode 1) string dictionaries transparently."""
import struct, numpy as np, zstandard as zstd
_DT_UNITS = ['us','ns','ms','s','D','h','m','M','Y','W']

class Segment:
    def __init__(self, path):
        buf = open(path,'rb').read(); assert buf[:5]==b'WVDB4', "not a WVDB3 segment"
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
            elif mode == 3:
                # FD-reference: dependent column stored as y_by_xcode (Vx Y-codes) into a
                # determinant column; no per-row codes. Y dict stored plain (mode-0 style).
                meta['det_idx'] = struct.unpack_from('<H',buf,off)[0]; off += 2
                Vx = struct.unpack_from('<I',buf,off)[0]; off += 4
                meta['Vx'] = Vx
                vals = []
                for _ in range(n_dict):
                    vl = struct.unpack_from('<I',buf,off)[0]; off += 4
                    vals.append(buf[off:off+vl]); off += vl
                meta['vals'] = vals
                meta['map_start'] = off
                off += (Vx*bits+7)//8      # packed y_by_xcode, NOT N per-row codes
                meta['fdmap'] = None
                self.cols[nm] = meta; self.order.append(nm)
                continue
            elif mode == 4:
                # affine/sequence (WSQ1) blob: no dict, no per-row codes. Self-describing
                # length: 32-byte fixed part + optional (4-byte zlen + zstd exception payload).
                n_exc = struct.unpack_from('<I', buf, off+28)[0]
                blob_len = 32 if n_exc == 0 else 36 + struct.unpack_from('<I', buf, off+32)[0]
                meta['seqblob'] = bytes(buf[off:off+blob_len]); off += blob_len
                meta['seqvals'] = None   # decoded lazily
                self.cols[nm] = meta; self.order.append(nm)
                continue
            elif mode == 5:
                # inline string column: no dict, no per-row codes. zstd(lengths u32)+zstd(bytes).
                zll = struct.unpack_from('<I', buf, off)[0]; off += 4
                meta['ilen'] = bytes(buf[off:off+zll]); off += zll
                zvl = struct.unpack_from('<I', buf, off)[0]; off += 4
                meta['ival'] = bytes(buf[off:off+zvl]); off += zvl
                meta['ivals'] = None
                self.cols[nm] = meta; self.order.append(nm)
                continue
            else:
                Rr = struct.unpack_from('<H',buf,off)[0]; off += 2
                nr = struct.unpack_from('<I',buf,off)[0]; off += 4
                meta['restarts'] = np.frombuffer(buf, dtype=np.uint32, count=nr, offset=off); off += nr*4
                fclen = struct.unpack_from('<I',buf,off)[0]; off += 4
                zlen = struct.unpack_from('<I',buf,off)[0]; off += 4
                meta['R'] = Rr; meta['z'] = buf[off:off+zlen]; off += zlen
                meta['vals'] = None; meta['raw'] = None  # decoded lazily
            code_enc = buf[off]; off += 1; meta['code_enc'] = code_enc   # 0=raw bitpack, 1=zstd codes
            if code_enc == 0:
                nb = (self.N*bits+7)//8; meta['cstart'] = off; off += nb
            else:
                meta['cwidth'] = buf[off]; off += 1
                czlen = struct.unpack_from('<I', buf, off)[0]; off += 4
                meta['czlen'] = czlen; meta['cstart'] = off; off += czlen
            self.cols[nm] = meta; self.order.append(nm)
        self.buf = np.frombuffer(buf, dtype=np.uint8); self._codes = {}
        self.path = path; self._presence = 0   # 0 = not yet loaded
        self._ov = 0                            # override sidecar: 0 = not yet loaded
        self._synth = {}                        # mode-6 synthetic constant columns (ADD COLUMN)
        self._tdict = {}                        # memo: decoded base dict per column (immutable .wdb)
    def add_const_column(self, name, value, dt, aux=0):
        """Register a synthetic constant column: `value` for ALL N live rows. Used for a logical
        column this immutable segment PREDATES (ADD COLUMN) -- the default is materialized lazily
        at read instead of rewriting the segment. Engine-internal mode 6; never serialized. value
        None means the column reads as NULL everywhere. Overrides/presence ride on top normally."""
        self.cols[name] = dict(V=1, bits=1, dt=dt, mode=6, has_null=(1 if value is None else 0),
                               n_dict=1, aux=aux)
        self._synth[name] = value

    def _const_array(self, nm):
        v = self._synth[nm]; c = self.cols[nm]; N = self.N
        if v is None: return np.full(N, None, dtype=object)
        if c['dt'] == 0: return np.full(N, int(v), dtype=np.int64)
        if c['dt'] == 2: return np.full(N, float(v), dtype=np.float64)
        if c['dt'] == 3:
            unit = _DT_UNITS[c['aux']]
            ev = int(v) if isinstance(v, (int, np.integer)) else np.datetime64(v).astype(f'datetime64[{unit}]').view('int64')
            return np.full(N, ev, dtype=np.int64).view(f'datetime64[{unit}]')
        return np.full(N, v, dtype=object)           # string/bytes constant

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
    def _fd_map(self, c):
        """Decode a mode-3 column's y_by_xcode: Vx Y-codes packed at bits, from map_start."""
        if c['fdmap'] is None:
            Vx = c['Vx']; bits = c['bits']; base = c['map_start']
            w = (1 << np.arange(bits-1,-1,-1)).astype(np.uint64)
            nbytes = (Vx*bits+7)//8
            allb = np.unpackbits(self.buf[base:base+nbytes])
            b = allb[:Vx*bits].reshape(Vx, bits)
            c['fdmap'] = (b.astype(np.uint64)*w).sum(1).astype(np.int64)
        return c['fdmap']
    def _inline_values(self, c):
        """Mode-5: decode the inline string column to an object array of bytes (cached). Rows are
        stored directly (lengths + concatenated bytes); offsets are cumsum(lengths)."""
        if c.get('ivals') is None:
            lengths = np.frombuffer(self._dz.decompress(c['ilen']), dtype=np.uint32)
            data = self._dz.decompress(c['ival']); mv = memoryview(data)
            off = np.zeros(len(lengths) + 1, dtype=np.int64); np.cumsum(lengths, out=off[1:])
            c['ivals'] = np.array([bytes(mv[off[i]:off[i+1]]) for i in range(len(lengths))], dtype=object)
        return c['ivals']
    def _seq_decode(self, c):
        """Decode a mode-4 affine column to its int64 array (cached). dt-3 epochs stay int64
        here; _base_values/fetch view them as datetime64."""
        if c.get('seqvals') is None:
            import wdb_seqcodec
            c['seqvals'] = wdb_seqcodec.decode(c['seqblob'])
        return c['seqvals']
    def _raw_codes(self, nm):
        if nm in self._codes: return self._codes[nm]
        c = self.cols[nm]
        if c['mode'] == 6:
            cc = np.zeros(self.N, dtype=np.int64)    # constant column: a single group
            self._codes[nm] = cc; return cc
        if c['mode'] == 4:
            cc = np.arange(self.N, dtype=np.int64)   # identity codes: value = f(position)
            self._codes[nm] = cc; return cc
        if c['mode'] == 5:
            uniq, inv = np.unique(self._inline_values(c), return_inverse=True)
            c['_idict'] = uniq                       # sorted distinct values, for fetch()
            cc = inv.astype(np.int64); self._codes[nm] = cc; return cc
        if c['mode'] == 3:
            # dependent column: gather Y-codes through the determinant's per-row codes
            x_codes = self._raw_codes(self.order[c['det_idx']])
            cc = self._fd_map(c)[x_codes].astype(np.int64)
            self._codes[nm] = cc; return cc
        if c.get('code_enc', 0) == 1:                # zstd of byte-aligned codes (clustered/skewed)
            raw = self._dz.decompress(self.buf[c['cstart']:c['cstart']+c['czlen']].tobytes())
            wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
            cc = np.frombuffer(raw, dtype=wdt).astype(np.int64)
            self._codes[nm] = cc; return cc
        bits = c['bits']; N = self.N; base = c['cstart']
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
    def _effective(self, nm):
        """Effective code space for a column with overrides. An override value that ALREADY
        exists in the dictionary reuses that value's code (so it merges with existing rows in
        GROUP BY); a genuinely-new value is appended as a synthetic dict entry at code V, V+1,
        ... (above the null slot). Overridden rows' codes are patched accordingly. Returns
        (eff_codes, ov_typed) or None when the column has no overrides. Cached. ov_typed are
        the distinct NEW values only, in dict-typed form."""
        ov = self._overrides(nm)
        if ov is None: return None
        cache = getattr(self, '_eff', None)
        if cache is None: cache = self._eff = {}
        if nm in cache: return cache[nm]
        idx, vals = ov
        c = self.cols[nm]; V = c['V']; dt = c['dt']
        real = self._typed_dict(nm)
        def _key(x): return bytes(x) if isinstance(x, (bytes, bytearray)) else x
        val2code = {}
        for code, dvv in enumerate(real):
            k = _key(dvv)
            if k not in val2code: val2code[k] = code
        new_seen = {}; ov_typed = []; row_codes = np.empty(len(vals), dtype=np.int64)
        for i, v in enumerate(list(vals)):
            if v is None:
                # override to NULL: use the null code (valid when this segment has a null
                # slot). UPDATE ... SET col = NULL on a segment with no null slot is a
                # deferred edge (the UPDATE statement does not emit it yet).
                row_codes[i] = c['V'] - 1
                continue
            tv = self._coerce_override(v, dt); k = _key(tv)
            if k in val2code:
                row_codes[i] = val2code[k]                 # reuse existing dict code
            else:
                j = new_seen.get(k)
                if j is None:
                    j = len(ov_typed); new_seen[k] = j; ov_typed.append(tv)
                row_codes[i] = V + j                       # synthetic code for new value
        eff = self._raw_codes(nm).copy()
        eff[np.asarray(idx, dtype=np.int64)] = row_codes
        cache[nm] = (eff, ov_typed); return cache[nm]
    @staticmethod
    def _coerce_override(v, dt):
        if dt == 0: return int(v)
        if dt == 2: return float(v)
        if dt == 1:
            return v.encode('utf-8') if isinstance(v, str) else bytes(v)
        return v   # dt 3 datetime: stored as-is (UPDATE on datetime is a later case)
    def codes(self, nm):
        eff = self._effective(nm)
        return eff[0] if eff is not None else self._raw_codes(nm)
    def _override_vals_typed(self, nm):
        eff = self._effective(nm)
        return eff[1] if eff is not None else []
    def _typed_dict(self, nm):
        if nm in self._tdict: return self._tdict[nm]
        r = self._typed_dict_uncached(nm)
        if self.cols[nm]['mode'] != 6:   # mode-6 synth value can change via register_synth; don't memo
            self._tdict[nm] = r
        return r
    def _typed_dict_uncached(self, nm):
        c = self.cols[nm]
        if c['mode'] == 6: return [self._synth[nm]]
        if c['mode'] == 4: return list(self._seq_decode(c))  # decoded values (override path only)
        if c['mode'] == 5: self._raw_codes(nm); return list(c['_idict'])  # factorized (override path)
        if c['mode'] == 2: return self._dict_ints(c)  # int64 array (dt 0/3)
        if c['dt'] == 0: return [int(v) for v in c['vals']]
        if c['dt'] == 2: return np.frombuffer(b''.join(c['vals']), dtype='<f8')   # vectorized + memoized
        if c['dt'] == 3: return [struct.unpack('<q', v)[0] for v in c['vals']]   # int64 epoch
        return self.dict_vals(nm)  # bytes
    def unit(self, nm):
        return _DT_UNITS[self.cols[nm]['aux']]
    def values(self, nm):
        """Decoded column values with overrides applied. Overrides are scattered in after
        the normal decode (vectorized, dtype-preserving where the override dtype is
        compatible). No-override columns take the base fast path unchanged."""
        out = self._base_values(nm)
        ov = self._overrides(nm)
        if ov is not None:
            idx, vals = ov
            try:
                out = out.copy(); out[idx] = vals          # dtype-preserving scatter
            except (ValueError, TypeError):
                out = np.asarray(out, dtype=object); out[idx] = vals
        return out
    def _base_values(self, nm):
        c = self.cols[nm]
        if c['mode'] == 6:
            return self._const_array(nm)
        if c['mode'] == 4:
            arr = self._seq_decode(c)
            return arr.view(f"datetime64[{_DT_UNITS[c['aux']]}]") if c['dt'] == 3 else arr
        if c['mode'] == 5:
            return self._inline_values(c)
        codes = self._raw_codes(nm); dvals = self._typed_dict(nm)
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
        O(R) for front-coded columns (jump to restart block, walk <=R deltas).
        Synthetic codes (>= V) resolve to override values."""
        c = self.cols[nm]
        if code >= c['V']:
            ov = self._override_vals_typed(nm)
            v = ov[int(code) - c['V']]
            if c['dt'] == 3:
                return np.int64(v).view(f"datetime64[{_DT_UNITS[c['aux']]}]")
            return v
        if c['has_null'] and code == c['V'] - 1: return None
        if c['mode'] == 6: return self._synth[nm]
        if c['mode'] == 4:
            v = int(self._seq_decode(c)[code])
            return np.int64(v).view(f"datetime64[{_DT_UNITS[c['aux']]}]") if c['dt'] == 3 else v
        if c['mode'] == 5:
            d = c.get('_idict')
            if d is None: self._raw_codes(nm); d = c['_idict']
            return d[code]
        if c['mode'] == 2:
            arr = self._dict_ints(c)
            if c['dt'] == 3:
                unit = _DT_UNITS[c['aux']]
                return np.int64(arr[code]).view(f'datetime64[{unit}]')
            return int(arr[code])
        if c['mode'] in (0, 3):
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
    def presence_mask(self):
        """Bool array (len N, True=live) from the presence sidecar, or None if all rows live.
        Lazily loaded and cached. None lets callers take the unmasked fast path."""
        if isinstance(self._presence, int):          # int sentinel = not yet loaded (loaded = ndarray|None)
            import wdb_presence
            self._presence = wdb_presence.load(self.path, self.N)
        return self._presence
    def _overrides(self, nm):
        """(row_idx, vals) overriding column nm from the override sidecar, or None.
        Lazily loaded and cached. None lets callers take the no-override fast path."""
        if self._ov == 0:
            import wdb_override
            self._ov = wdb_override.load(self.path) or {}
        return self._ov.get(nm)
    def cardinality(self, nm): return self.cols[nm]['V']
    def group_by_count(self, nm):
        return np.bincount(self.codes(nm), minlength=self.cols[nm]['V'])
