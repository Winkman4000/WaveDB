#!/usr/bin/env python3
"""WaveDB engine — loads a WVDB3 segment, resolves columns by name from the header.
Generic: knows nothing about any specific dataset. Handles plain (mode 0) and
front-coded (mode 1) string dictionaries transparently."""
import struct
import threading, numpy as np, zstandard as zstd
from concurrent.futures import ThreadPoolExecutor
_DT_UNITS = ['us','ns','ms','s','D','h','m','M','Y','W']

_POOL = None


def _pool():
    """Persistent 8-lane executor for span decompression: pool spin-up per column read
    was measurable overhead at board scale."""
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=8)
    return _POOL


_LEAF_POOL = None


def _leaf_pool():
    """ONE shared executor for all frame-decompress leaf jobs -- three
    loaders each spawning cpu_count() threads oversubscribed the box 3x
    (measured on Q12's shadow). Leaf jobs never submit, so no deadlock."""
    global _LEAF_POOL
    if _LEAF_POOL is None:
        import os as _osl
        from concurrent.futures import ThreadPoolExecutor as _TPl
        _LEAF_POOL = _TPl(max_workers=(_osl.cpu_count() or 8))
    return _LEAF_POOL


class Segment:
    def __init__(self, path):
        # memmap instead of read(): the file is demand-paged by the OS, so a Segment
        # only makes resident the column code/dict pages a query actually touches, and
        # those pages are file-backed (reclaimable under pressure) rather than anonymous
        # heap. Parse below reads only small metadata; the big code regions stay un-faulted
        # until decoded. zstd payload blobs are kept as memmap views (decompress accepts
        # them) so even compressed dictionaries load lazily.
        buf = np.memmap(path, dtype=np.uint8, mode='r'); assert bytes(buf[:5])==b'WVDB4', "not a WVDB4 segment"
        off = 5
        self.n_cols = struct.unpack_from('<H',buf,off)[0]; off += 2
        self.N = struct.unpack_from('<I',buf,off)[0]; off += 4
        self.cols = {}; self.order = []; self._dzl = threading.local()
        for _ in range(self.n_cols):
            blob0 = off                                        # this column's first byte
            nl = struct.unpack_from('<H',buf,off)[0]; off += 2
            nm = bytes(buf[off:off+nl]).decode(); off += nl
            V = struct.unpack_from('<I',buf,off)[0]; off += 4
            bits = int(buf[off]); off += 1; dt = int(buf[off]); off += 1; mode = int(buf[off]); off += 1
            has_null = int(buf[off]); off += 1; aux = int(buf[off]); off += 1
            chunked = bool(aux & 0x40); aux &= 0x3F            # bit6 = front-coded chunked dict
            n_dict = V - has_null
            meta = dict(V=V, bits=bits, dt=dt, mode=mode, has_null=has_null, n_dict=n_dict, aux=aux, chunked=chunked)
            if mode == 0:
                vals = []
                for _ in range(n_dict):
                    vl = struct.unpack_from('<I',buf,off)[0]; off += 4
                    vals.append(bytes(buf[off:off+vl])); off += vl
                meta['vals'] = vals
            elif mode == 2:
                zlen = struct.unpack_from('<I',buf,off)[0]; off += 4
                if zlen == 0xFFFFFFFF:               # chunked spine (hits_6 dress)
                    i2ch, nch = struct.unpack_from('<II', buf, off); off += 8
                    zl = np.frombuffer(buf, dtype=np.uint32, count=nch, offset=off)
                    off += 4 * nch
                    zoffs = np.zeros(nch + 1, np.int64)
                    np.cumsum(zl.astype(np.int64), out=zoffs[1:])
                    meta['i2ch'] = int(i2ch)
                    meta['i2base'] = off
                    meta['i2zoffs'] = zoffs
                    meta['i2chunks'] = {}
                    off += int(zoffs[-1])
                    meta['z2'] = None
                else:
                    meta['z2'] = buf[off:off+zlen]; off += zlen
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
                    vals.append(bytes(buf[off:off+vl])); off += vl
                meta['vals'] = vals
                meta['map_start'] = off
                off += (Vx*bits+7)//8      # packed y_by_xcode, NOT N per-row codes
                meta['fdmap'] = None
                meta['blob'] = (blob0, off); self.cols[nm] = meta; self.order.append(nm)
                continue
            elif mode == 4:
                # affine/sequence (WSQ1) blob: no dict, no per-row codes. Self-describing
                # length: 32-byte fixed part + optional (4-byte zlen + zstd exception payload).
                n_exc = struct.unpack_from('<I', buf, off+28)[0]
                blob_len = 32 if n_exc == 0 else 36 + struct.unpack_from('<I', buf, off+32)[0]
                meta['seqblob'] = bytes(buf[off:off+blob_len]); off += blob_len
                meta['seqvals'] = None   # decoded lazily
                meta['blob'] = (blob0, off); self.cols[nm] = meta; self.order.append(nm)
                continue
            elif mode == 5:
                # inline string column: no dict, no per-row codes. zstd(lengths u32)+zstd(bytes).
                zll = struct.unpack_from('<I', buf, off)[0]; off += 4
                meta['ilen'] = buf[off:off+zll]; off += zll
                zvl = struct.unpack_from('<I', buf, off)[0]; off += 4
                meta['ival'] = buf[off:off+zvl]; off += zvl
                meta['ivals'] = None
                meta['blob'] = (blob0, off); self.cols[nm] = meta; self.order.append(nm)
                continue
            else:
                Rr = struct.unpack_from('<H',buf,off)[0]; off += 2
                meta['R'] = Rr
                if meta['chunked']:
                    CH = struct.unpack_from('<I',buf,off)[0]; off += 4
                    nch = struct.unpack_from('<I',buf,off)[0]; off += 4
                    nr = struct.unpack_from('<I',buf,off)[0]; off += 4
                    meta['restarts'] = buf[off:off+nr*4].view(np.uint32); off += nr*4
                    fclen = struct.unpack_from('<I',buf,off)[0]; off += 4
                    meta['CHUNK'] = CH
                    meta['chunk_ustart'] = buf[off:off+nch*4].view(np.uint32); off += nch*4
                    czlen = buf[off:off+nch*4].view(np.uint32); off += nch*4
                    meta['chunk_czlen'] = czlen
                    foff = np.empty(nch+1, dtype=np.int64); foff[0] = 0
                    np.cumsum(czlen.astype(np.int64), out=foff[1:])
                    meta['chunk_foff'] = foff; meta['chunk_base'] = off
                    off += int(foff[-1])
                    meta['chunks'] = {}; meta['vals'] = None
                else:
                    nr = struct.unpack_from('<I',buf,off)[0]; off += 4
                    meta['restarts'] = buf[off:off+nr*4].view(np.uint32); off += nr*4
                    fclen = struct.unpack_from('<I',buf,off)[0]; off += 4
                    zlen = struct.unpack_from('<I',buf,off)[0]; off += 4
                    meta['z'] = buf[off:off+zlen]; off += zlen
                    meta['vals'] = None; meta['raw'] = None  # decoded lazily
            meta['code_off'] = off                                             # the code section's first byte (the redress splices here)
            code_enc = int(buf[off]); off += 1; meta['code_enc'] = code_enc   # 0=bitpack, 1=zstd, 2=staircase, 3=blocked, 18=packed frames
            if code_enc == 0:
                nb = (self.N*bits+7)//8; meta['cstart'] = off; off += nb
            elif code_enc == 17:               # RAW PACKED CODES: mmap-direct, no toll
                meta['pk_bits'] = int(buf[off]); off += 1
                meta['pk_n'] = int(struct.unpack_from('<I', buf, off)[0]); off += 4
                meta['cstart'] = off
                meta['czlen'] = (meta['pk_n'] * meta['pk_bits'] + 7) // 8 + 2
                off += meta['czlen']
            elif code_enc == 15:               # CLOCK DRESS (Jackson's dial): anchor+delta+bit
                meta['ywidth'] = int(buf[off]); off += 1
                meta['ybase'] = int(struct.unpack_from('<H', buf, off)[0]); off += 2
                fr15, nfr15 = struct.unpack_from('<II', buf, off); off += 8
                meta['e15_FR'] = int(fr15); meta['e15_nfr'] = int(nfr15)
                meta['e15_offs'] = []
                for _p15 in range(5):
                    meta['e15_offs'].append(np.frombuffer(buf, dtype=np.uint32,
                                                          count=nfr15 + 1, offset=off))
                    off += 4 * (nfr15 + 1)
                pnl15 = int(struct.unpack_from('<H', buf, off)[0]); off += 2
                meta['e15_partner'] = bytes(buf[off:off + pnl15]).decode(); off += pnl15
                meta['cstart'] = off
                meta['czlen'] = int(sum(int(o[-1]) for o in meta['e15_offs']))
                off += meta['czlen']
            elif code_enc == 16:               # clock stub: dressed by partner
                pnl16 = int(struct.unpack_from('<H', buf, off)[0]); off += 2
                meta['e16_partner'] = bytes(buf[off:off + pnl16]).decode(); off += pnl16
                meta['cstart'] = off; meta['czlen'] = 0
            elif code_enc == 14:               # FIELD PLANES (Jackson's dress): y/m/d, FRAMED
                meta['ywidth'] = int(buf[off]); off += 1
                meta['ybase'] = int(struct.unpack_from('<H', buf, off)[0]); off += 2
                fr14, nfr14 = struct.unpack_from('<II', buf, off); off += 8
                meta['e14_FR'] = int(fr14); meta['e14_nfr'] = int(nfr14)
                meta['e14_offs'] = []
                for _p14 in range(3):
                    meta['e14_offs'].append(np.frombuffer(buf, dtype=np.uint32,
                                                          count=nfr14 + 1, offset=off))
                    off += 4 * (nfr14 + 1)
                meta['cstart'] = off
                meta['czlen'] = int(sum(int(o[-1]) for o in meta['e14_offs']))
                off += meta['czlen']
            elif code_enc == 12:               # vertical planes (the tapes)
                meta['nwords'] = int(struct.unpack_from('<I', buf, off)[0]); off += 4
                meta['cstart'] = off; off += bits * meta['nwords'] * 8
            elif code_enc == 2:                # staircase: gap-packed step rows (see Segment.stairs)
                meta['gbits'] = int(buf[off]); off += 1
                meta['nsteps'] = struct.unpack_from('<I', buf, off)[0]; off += 4
                nb = (meta['nsteps']*meta['gbits']+7)//8
                meta['cstart'] = off; off += nb
            elif code_enc == 3:                # blocked: independent zstd frame per BR rows + offset index
                meta['cwidth'] = int(buf[off]); off += 1
                meta['BR'], nfr = struct.unpack_from('<II', buf, off); off += 8
                meta['boffs'] = np.frombuffer(buf, dtype=np.uint32, count=nfr+1, offset=off); off += 4*(nfr+1)
                meta['cstart'] = off; meta['czlen'] = int(meta['boffs'][-1]); off += meta['czlen']
            elif code_enc == 18:               # PACKED FRAMES: zstd frame per BR rows of LE bit-packed codes
                meta['pbits'] = int(buf[off]); off += 1        # (no 'boffs'/'cwidth': the enc-3 readers do not see it)
                meta['BR'], nfr = struct.unpack_from('<II', buf, off); off += 8
                meta['poffs'] = np.frombuffer(buf, dtype=np.uint32, count=nfr+1, offset=off); off += 4*(nfr+1)
                meta['cstart'] = off; meta['czlen'] = int(meta['poffs'][-1]); off += meta['czlen']
            elif code_enc == 13:               # byte-planes: each plane its own zstd frame
                meta['vnby'] = int(buf[off]); off += 1
                meta['BR'], nfr = struct.unpack_from('<II', buf, off); off += 8
                meta['boffs'] = np.frombuffer(buf, dtype=np.uint64, count=nfr+1, offset=off); off += 8*(nfr+1)
                meta['cstart'] = off; meta['czlen'] = int(meta['boffs'][-1]); off += meta['czlen']
            elif code_enc == 10:               # segmented bitpack-plus
                meta['pXbits'] = int(buf[off]); off += 1
                pXn, pXnb = struct.unpack_from('<QI', buf, off); off += 12
                meta['pXn'] = int(pXn); meta['pXnblk'] = int(pXnb)
                meta['pXdir'] = off; off += 8 * int(pXnb)
                meta['pXpay'] = off
                dirX = np.frombuffer(buf, np.int64, int(pXnb), meta['pXdir'])
                lastoff = int(dirX[-1]) >> 1
                B10 = 4096
                last_rows = int(pXn) - (int(pXnb) - 1) * B10
                if int(dirX[-1]) & 1:
                    nrX, = struct.unpack_from('<H', buf, meta['pXpay'] + lastoff)
                    off = meta['pXpay'] + lastoff + 2 + 4 * int(nrX)
                else:
                    off = meta['pXpay'] + lastoff + (last_rows * meta['pXbits'] + 7) // 8
                meta['bits'] = meta['pXbits']
            elif code_enc == 9:                # tiered dress (rule eleven)
                meta['e9bits'] = int(buf[off]); off += 1
                meta['e9d'], meta['e9n'], e9rows = struct.unpack_from('<IQQ', buf, off); off += 20
                meta['e9pres'] = off; off += (e9rows + 7) // 8
                meta['e9ck'] = off; off += ((e9rows + 65535) // 65536) * 8
                nt9 = int(buf[off]); off += 1
                meta['e9tiers'] = []
                for _ in range(nt9):
                    tc9, tn9 = struct.unpack_from('<IQ', buf, off); off += 12
                    meta['e9tiers'].append((int(tc9), int(tn9), off))
                    off += (tn9 + 7) // 8
                meta['e9tail_n'], = struct.unpack_from('<Q', buf, off); off += 8
                meta['e9tail'] = off; off += meta['e9tail_n']
            elif code_enc == 8:                # sparse-default: presence + checkpoints + literals
                meta['e8bits'] = int(buf[off]); off += 1
                meta['e8d'], meta['e8n'], e8rows = struct.unpack_from('<IQQ', buf, off); off += 20
                meta['e8pres'] = off; off += (e8rows + 7) // 8
                meta['e8ck'] = off; off += ((e8rows + 65535) // 65536) * 8
                meta['cstart'] = off
                off += (meta['e8n'] * meta['e8bits'] + 7) // 8
            elif code_enc == 6:                # warm buckets: hot nibble -> warm byte -> u16 cold
                meta['cwidth'] = int(buf[off]); off += 1
                meta['BR'], nb6, nwb, npatch6, nhot = struct.unpack_from('<IIQQH', buf, off); off += 26
                meta['e5hot'] = np.frombuffer(buf, dtype=np.uint16, count=nhot, offset=off); off += 2*nhot
                meta['e6warm'] = np.frombuffer(buf, dtype=np.uint16, count=255, offset=off); off += 2*255
                meta['e6o1'] = np.frombuffer(buf, dtype=np.uint32, count=nb6+1, offset=off); off += 4*(nb6+1)
                meta['e6o2'] = np.frombuffer(buf, dtype=np.uint32, count=nb6+1, offset=off); off += 4*(nb6+1)
                meta['e6wb'] = np.frombuffer(buf, dtype=np.uint8, count=nwb, offset=off); off += nwb
                meta['e5patch'] = np.frombuffer(buf, dtype=np.uint16, count=npatch6, offset=off); off += 2*npatch6
                meta['cstart'] = off; meta['czlen'] = (self.N + 1) // 2; off += meta['czlen']
            elif code_enc == 5:                # patched buckets: 4-bit hot pointers + escape patches
                meta['cwidth'] = int(buf[off]); off += 1
                meta['BR'], nb5, npatch, nhot = struct.unpack_from('<IIQH', buf, off); off += 18
                meta['e5hot'] = np.frombuffer(buf, dtype=np.uint16, count=nhot, offset=off); off += 2*nhot
                meta['e5off'] = np.frombuffer(buf, dtype=np.uint32, count=nb5+1, offset=off); off += 4*(nb5+1)
                meta['e5patch'] = np.frombuffer(buf, dtype=np.uint16, count=npatch, offset=off); off += 2*npatch
                meta['cstart'] = off; meta['czlen'] = (self.N + 1) // 2; off += meta['czlen']
            else:
                meta['cwidth'] = int(buf[off]); off += 1
                czlen = struct.unpack_from('<I', buf, off)[0]; off += 4
                meta['czlen'] = czlen; meta['cstart'] = off; off += czlen
            meta['blob'] = (blob0, off); self.cols[nm] = meta; self.order.append(nm)
        assert off == len(buf), ('segment parse did not consume the file', off, len(buf))
        self.buf = buf; self._codes = {}   # buf is a read-only np.memmap (uint8); code/dict reads fault lazily
        self.path = path; self._presence = 0   # 0 = not yet loaded
        self._ov = 0                            # override sidecar: 0 = not yet loaded
        self._cluster = 0                       # .cluster slice-boundary sidecar: 0 = not loaded
        self._cubes = 0                         # .cube materialised-aggregate sidecar: 0 = not loaded
        self._synth = {}                        # mode-6 synthetic constant columns (ADD COLUMN)
        self._tdict = {}                        # memo: decoded base dict per column (immutable .wdb)
        self._resident = {}                     # memo: full per-row decoded array, kept resident for chunk reads
    def drop_derived(self):
        """Jackson's law (see wdb_qmem): everything derived from data dies at query end --
        decompressed codes, decoded dicts, resident arrays, effective/count memos, and
        sidecar loads (presence, override, cluster, cube: they lazy-reload on next touch).
        The memmap stays: it IS the file. cols/order/synth stay: file-shape metadata."""
        self._codes.clear(); self._resident.clear()
        # THE SHELF: a decoded dictionary is V-scale vocabulary, not N-scale residue --
        # it survives the flush only while the process-wide shelf keeps it (byte
        # ceiling, LRU). Anything the shelf evicted is dropped here too.
        try:
            import wdb_shelf
            _keep9 = {}
            for _k9, _v9 in list(self._tdict.items()):
                _key9 = ('tdict', getattr(self, 'path', id(self)), _k9)
                if wdb_shelf.SHELF.get(_key9) is None:
                    try:
                        wdb_shelf.SHELF.put(_key9, _v9, wdb_shelf.nbytes_of(_v9), kind='dictionary')
                    except wdb_shelf.ShelfRefused:
                        continue
                _keep9[_k9] = _v9
            self._tdict.clear(); self._tdict.update(_keep9)
        except Exception:
            self._tdict.clear()
        _pl9 = getattr(self, '_e14_pl', None)
        if _pl9 is not None: _pl9.clear()   # planes obey the same forget-law as codes
        for a in ('_eff', '_ccounts'):
            d = getattr(self, a, None)
            if isinstance(d, dict):
                d.clear()
        for c in self.cols.values():
            if c.get('mode') not in (0, 3) and 'vals' in c:
                c['vals'] = None                 # lazily-decoded dicts die too (modes 0/3
            for k in ('intvals', 'raw', 'seqvals', 'ivals', 'fdmap'):   # parse eagerly at
                if k in c:                       # open: file-shape metadata, they stay)
                    c[k] = None
            ch = c.get('chunks')
            if isinstance(ch, dict):
                ch.clear()
        self._presence = 0; self._ov = 0; self._cluster = 0; self._cubes = 0

    def resident_values(self, nm):
        """Decode the column ONCE and keep the full per-row array resident, so cluster-slice
        reads over a sorted segment are a plain memory slice with no per-query decode. This
        trades RAM (N * itemsize) for eliminating the dict gather on every query; callers gate
        it on a RAM budget and only materialize columns a chunk path actually aggregates. The
        array is the same object values() would produce (overrides applied), cached by name."""
        r = self._resident.get(nm)
        if r is None:
            r = np.ascontiguousarray(self.values(nm))
            self._resident[nm] = r
        return r
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
        R = c['R']
        if c.get('chunked'):
            vals = []
            for j in range(len(c['chunk_czlen'])):
                fb = c['chunk_base'] + int(c['chunk_foff'][j])
                fe = c['chunk_base'] + int(c['chunk_foff'][j+1])
                raw = self._dz.decompress(bytes(self.buf[fb:fe]))
                o = 0; i = 0; prev = b''
                while o < len(raw):
                    if i % R == 0: prev = b''
                    cp, sl = struct.unpack_from('<HH', raw, o); o += 4
                    suf = raw[o:o+sl]; o += sl; prev = prev[:cp]+suf; vals.append(prev); i += 1
            return vals
        raw = self._dz.decompress(c['z']); vals = []; prev = b''; o = 0; i = 0
        while o < len(raw):
            if i % R == 0: prev = b''
            cp, sl = struct.unpack_from('<HH', raw, o); o += 4
            suf = raw[o:o+sl]; o += sl; s = prev[:cp]+suf; vals.append(s); prev = s; i += 1
        return vals
    def dict_charlens(self, nm):
        _m9 = getattr(self, '_dlens', None)
        if _m9 is None:
            _m9 = self._dlens = {}
        if ('c', nm) in _m9:
            return _m9[('c', nm)]
        """V-sized CHARACTER lengths read straight off the front-coded dict bytes:
        UTF-8 chars = bytes minus continuation bytes, tracked through the prefix
        chain arithmetically. No string is ever decoded. None -> caller falls back."""
        c = self.cols.get(nm)
        if c is None or c.get('mode') not in (0, 1) or c.get('dt') != 1:
            return None
        got = c.get('charlens')
        if got is not None:
            return got
        import wdb_kernels as _WK
        if c.get('R') is None:
            return None                          # not front-coded: fall back
        R = int(c['R'])
        outs = []
        try:
            if c.get('chunked'):
                bufs = []
                for j in range(len(c['chunk_czlen'])):
                    fb = c['chunk_base'] + int(c['chunk_foff'][j])
                    fe = c['chunk_base'] + int(c['chunk_foff'][j + 1])
                    bufs.append(bytes(self.buf[fb:fe]))
                def _one(fb2):
                    raw = __import__('zstandard').ZstdDecompressor().decompress(fb2)
                    a = np.frombuffer(raw, dtype=np.uint8)
                    out = np.empty(a.size // 4 + 1, np.int64)
                    n = _WK.fc_charlens(a, np.int64(R), out)
                    return out[:n]
                outs = list(_pool().map(_one, bufs))
            else:
                raw = self._dz.decompress(c['z'])
                a = np.frombuffer(raw, dtype=np.uint8)
                out = np.empty(a.size // 4 + 1, np.int64)
                n = _WK.fc_charlens(a, np.int64(R), out)
                outs = [out[:n]]
        except Exception:
            return None
        res = np.concatenate(outs) if outs else np.empty(0, np.int64)
        if res.size != int(c.get('n_dict', c['V'])):
            return None                          # layout surprise: fail closed
        if c['has_null'] and res.size == int(c['V']) - 1:
            res = np.concatenate([res, np.zeros(1, np.int64)])
        c['charlens'] = res
        _r9 = res
        _m9[('c', nm)] = _r9
        return _r9

    def dict_bytelens(self, nm):
        _m9 = getattr(self, '_dlens', None)
        if _m9 is None:
            _m9 = self._dlens = {}
        if ('b', nm) in _m9:
            return _m9[('b', nm)]
        """V-sized BYTE lengths off the front-coded dict (STRLEN semantics).
        No decode; pure header arithmetic. None -> caller falls back."""
        c = self.cols.get(nm)
        if c is None or c.get('mode') not in (0, 1) or c.get('dt') != 1:
            return None
        got = c.get('bytelens')
        if got is not None:
            return got
        import wdb_kernels as _WK
        if c.get('R') is None:
            return None                          # not front-coded: fall back
        R = int(c['R'])
        outs = []
        try:
            if c.get('chunked'):
                bufs = []
                for j in range(len(c['chunk_czlen'])):
                    fb = c['chunk_base'] + int(c['chunk_foff'][j])
                    fe = c['chunk_base'] + int(c['chunk_foff'][j + 1])
                    bufs.append(bytes(self.buf[fb:fe]))
                def _one(fb2):
                    raw = __import__('zstandard').ZstdDecompressor().decompress(fb2)
                    a = np.frombuffer(raw, dtype=np.uint8)
                    out = np.empty(a.size // 4 + 1, np.int64)
                    n = _WK.fc_bytelens(a, np.int64(R), out)
                    return out[:n]
                outs = list(_pool().map(_one, bufs))
            else:
                raw = self._dz.decompress(c['z'])
                a = np.frombuffer(raw, dtype=np.uint8)
                out = np.empty(a.size // 4 + 1, np.int64)
                n = _WK.fc_bytelens(a, np.int64(R), out)
                outs = [out[:n]]
        except Exception:
            return None
        res = np.concatenate(outs) if outs else np.empty(0, np.int64)
        if res.size != int(c.get('n_dict', c['V'])):
            return None                          # layout surprise: fail closed
        if c['has_null'] and res.size == int(c['V']) - 1:
            res = np.concatenate([res, np.zeros(1, np.int64)])
        c['bytelens'] = res
        _r9 = res
        _m9[('b', nm)] = _r9
        return _r9

    def _dict_ints(self, c):
        # mode 2: the .nline sidecar memmaps directly when present -- the fixed number
        # line: zero per-query rebuild, the OS page cache owns the bytes, and the map
        # survives drop_derived lawfully (it IS a file, seg.buf's class). Every caller
        # of this choke point points at the line for free. Else delta+zstd -> cumsum,
        # memoized per query as before.
        if '_nline' not in c:
            import os
            nm = next((k for k, v in self.cols.items() if v is c), None)
            p = os.path.realpath(self.path) + '.nline.' + str(nm)
            c['_nline'] = np.memmap(p, dtype='<i8', mode='r').view(np.ndarray) \
                if nm and os.path.exists(p) else None   # ndarray view: same mapped bytes,
        if c['_nline'] is not None:                     # no memmap-subclass gather tax
            return c['_nline']
        if c.get('intvals') is None:
            if c.get('i2ch') is not None:            # chunked spine: inflate ALL chunks
                nch = len(c['i2zoffs']) - 1          # POOLED for full-dict consumers
                parts = [None] * nch
                def _popi(ch):
                    import zstandard as _z
                    a = c['i2base'] + int(c['i2zoffs'][ch])
                    b = c['i2base'] + int(c['i2zoffs'][ch + 1])
                    return ch, np.cumsum(np.frombuffer(
                        _z.ZstdDecompressor().decompress(bytes(self.buf[a:b])),
                        dtype=np.int64))
                with ThreadPoolExecutor(max_workers=8) as ex:
                    for ch, arr in ex.map(_popi, range(nch)):
                        parts[ch] = arr
                c['intvals'] = np.concatenate(parts)
            else:
                raw = self._dz.decompress(c['z2'])
                d = np.frombuffer(raw, dtype=np.int64)
                c['intvals'] = np.cumsum(d)
        return c['intvals']

    def _dict_ints_at(self, c, idx):
        """Values at dict indices, popping ONLY the touched chunks (pooled). Falls back
        to the full spine when unchunked or already inflated."""
        idx = np.asarray(idx, dtype=np.int64)
        if c.get('i2ch') is None or c.get('intvals') is not None or idx.size == 0:
            return self._dict_ints(c)[idx] if idx.size else np.empty(0, np.int64)
        CH = int(c['i2ch'])
        chs = np.unique(idx // CH)
        cache = c['i2chunks']
        missing = [int(ch) for ch in chs.tolist() if ch not in cache]
        if missing:
            def _popi(ch):
                import zstandard as _z
                a = c['i2base'] + int(c['i2zoffs'][ch])
                b = c['i2base'] + int(c['i2zoffs'][ch + 1])
                return ch, np.cumsum(np.frombuffer(
                    _z.ZstdDecompressor().decompress(bytes(self.buf[a:b])),
                    dtype=np.int64))
            if len(missing) > 1:
                with ThreadPoolExecutor(max_workers=min(8, len(missing))) as ex:
                    for ch, arr in ex.map(_popi, missing):
                        cache[ch] = arr
            else:
                ch, arr = _popi(missing[0])
                cache[ch] = arr
        out = np.empty(idx.size, np.int64)
        for ch in chs.tolist():
            m = (idx // CH) == ch
            out[m] = cache[int(ch)][idx[m] - ch * CH]
        return out
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
    def inline_at(self, nm, rows):
        """Mode-5 POINT READ: bytes at ROW positions. Decompresses lengths+bytes
        once per column (cached on the column), then slices -- no full object
        array, no np.unique. (values_at's mode-5 path built the whole sorted
        dict to answer 20 lookups: 7s per query on Q10, 2026-09-01.)"""
        c = self.cols[nm]
        st = c.get('_istream')
        if st is None:
            lengths = np.frombuffer(self._dz.decompress(c['ilen']), dtype=np.uint32)
            off = np.zeros(len(lengths) + 1, dtype=np.int64); np.cumsum(lengths, out=off[1:])
            data = self._dz.decompress(c['ival'])
            st = c['_istream'] = (off, memoryview(data))
        off, mv = st
        return [bytes(mv[off[r]:off[r + 1]]) for r in np.asarray(rows, dtype=np.int64).tolist()]

    def inline_stream(self, nm):
        """(blob_u8, off) of a mode-5 column's inline stream, cached -- the raw
        substrate for byte kernels (LIKE, prefixes) that never touch Python."""
        self.inline_at(nm, np.empty(0, dtype=np.int64))     # warm the cache
        off, mv = self.cols[nm]['_istream']
        blob = np.frombuffer(mv, dtype=np.uint8)
        return blob, off

    def like_mask(self, nm, needles):
        """Bool[N]: rows whose bytes contain the needles IN ORDER (1 or 2)."""
        import wdb_kernels as _WKl
        blob, off = self.inline_stream(nm)
        n1 = np.frombuffer(needles[0].encode() if isinstance(needles[0], str) else needles[0], dtype=np.uint8)
        n2 = (np.frombuffer(needles[1].encode() if isinstance(needles[1], str) else needles[1], dtype=np.uint8)
              if len(needles) > 1 else np.empty(0, dtype=np.uint8))
        out = np.empty(self.N, dtype=np.bool_)
        _WKl.plike2(blob, np.ascontiguousarray(off, dtype=np.int64), n1, n2, out)
        return out

    def dict_bytes(self, nm):
        """The FULL front-coded dict byte space + restarts (chunks decompressed
        in parallel and concatenated -- restarts are global offsets already)."""
        c = self.cols[nm]
        got = c.get('_dictbytes')
        if got is not None:
            return got
        if 'restarts' not in c:
            raise KeyError('not front-coded')
        if c.get('chunked'):
            nch = len(c['chunk_foff']) - 1
            import zstandard as _z
            def _dc(ch):
                fb = c['chunk_base'] + int(c['chunk_foff'][ch])
                fe = c['chunk_base'] + int(c['chunk_foff'][ch + 1])
                return _z.ZstdDecompressor().decompress(bytes(self.buf[fb:fe]))
            bufs = list(_leaf_pool().map(_dc, range(nch)))
            blob = np.frombuffer(b''.join(bufs), dtype=np.uint8)
        else:
            raw = c.get('raw')
            if raw is None:
                raw = c['raw'] = self._dz.decompress(c['z'])
            blob = np.frombuffer(raw, dtype=np.uint8)
        got = c['_dictbytes'] = (blob, np.ascontiguousarray(np.asarray(c['restarts'], dtype=np.int64)))
        return got

    def like_mask_dict(self, nm, needles, invert=False):
        """Bool[N] for an ordered-needle LIKE on a front-coded dict column:
        the kernel tests V dictionary values, a LUT paints the rows."""
        import wdb_kernels as _WKl
        c = self.cols[nm]
        blob, restarts = self.dict_bytes(nm)
        V = int(c.get('n_dict') or c['V'])
        n1 = np.frombuffer(needles[0].encode(), dtype=np.uint8)
        n2 = (np.frombuffer(needles[1].encode(), dtype=np.uint8)
              if len(needles) > 1 else np.empty(0, dtype=np.uint8))
        keepd = np.empty(V, dtype=np.bool_)
        _WKl.plike_fc(blob, restarts, int(c['R']), V, n1, n2, keepd)
        if invert:
            keepd = ~keepd
        cds = np.asarray(self.codes(nm))
        out = np.empty(cds.shape[0], dtype=np.bool_)
        _WKl.plut_u8(cds, keepd, out)
        return out

    def prefix2_codes(self, nm):
        """u16[N]: each row's first two bytes packed -- 2-byte reads at stream
        level, cached per column (3MB at 1.5M rows)."""
        c = self.cols[nm]
        got = c.get('_prefix2')
        if got is not None:
            return got
        import wdb_kernels as _WKl
        blob, off = self.inline_stream(nm)
        out = np.empty(self.N, dtype=np.uint16)
        _WKl.pprefix2(blob, np.ascontiguousarray(off, dtype=np.int64), out)
        c['_prefix2'] = out
        return out

    def values_at_rows(self, nm, rows):
        """Emission-ready Python values at ROW positions (the row-side twin of
        values_at, which takes dict codes)."""
        c = self.cols[nm]
        if c['mode'] == 5 and not c.get('has_null'):
            import wdb_sql as _WS5
            return [_WS5._pyval(b) for b in self.inline_at(nm, rows)]
        return self.values_at(nm, np.asarray(self.codes_at(nm, rows)))

    def _seq_decode(self, c):
        """Decode a mode-4 affine column to its int64 array (cached). dt-3 epochs stay int64
        here; _base_values/fetch view them as datetime64."""
        if c.get('seqvals') is None:
            import wdb_seqcodec
            c['seqvals'] = wdb_seqcodec.decode(c['seqblob'])
        return c['seqvals']
    def _bitunpack(self, base, lo, hi, bits):
        """Decode bit-packed per-row codes for rows [lo,hi) using 8-byte sliding windows + a
        single shift+mask -- no 8x np.unpackbits expansion (measured ~3.4x faster, bit-exact).
        Big-endian because codes are packed MSB-first. Chunked to bound peak memory; the final
        column is zero-padded so the 8-byte window never reads past the buffer."""
        from numpy.lib.stride_tricks import sliding_window_view
        n = hi - lo
        _od = np.uint8 if bits <= 8 else (np.uint16 if bits <= 16 else (np.uint32 if bits <= 32 else np.int64))
        out = np.empty(n, dtype=_od)
        if n <= 0: return out
        if bits % 8 == 0:                        # BYTE-ALIGNED FAST PATH (the throughput
            w = bits // 8                        # law's read): lanes ARE the memory; the
            raw = self.buf[base + lo * w: base + hi * w]   # sliding-window machinery cost
            if w == 1:                           # 1.4s at 100Mx24b -- slower than the
                return np.asarray(raw, dtype=np.uint8).copy()   # zstd it replaced
            if w == 2:
                return np.frombuffer(raw, dtype='>u2').astype(np.uint16)
            if w == 3:
                import wdb_kernels as _WK
                return _WK.unpack24_be(np.ascontiguousarray(raw), n)
            if w == 4:
                return np.frombuffer(raw, dtype='>u4').astype(np.uint32)
        mask = np.uint64((1 << bits) - 1); CH = 4_000_000
        for c0 in range(lo, hi, CH):
            c1 = min(c0 + CH, hi)
            bo = np.arange(c0, c1, dtype=np.int64) * bits
            byte0 = bo >> 3
            first = int(byte0[0]); last = int(byte0[-1]) + 8
            seg = self.buf[base + first: base + last]
            if len(seg) < (last - first):
                seg = np.concatenate([seg, np.zeros((last - first) - len(seg), np.uint8)])
            win = sliding_window_view(seg, 8)
            sel = np.ascontiguousarray(win[byte0 - first])
            w64 = sel.view('>u8').reshape(-1)
            shift = np.uint64(64 - bits) - (bo & 7).astype(np.uint64)
            out[c0 - lo:c1 - lo] = ((w64 >> shift) & mask).astype(_od)
        return out

    @staticmethod
    def _cdt(ndist):
        # smallest unsigned width that holds codes 0..ndist-1 (keeps the scanned array compact)
        return (np.uint8 if ndist <= 256 else np.uint16 if ndist <= 65536
                else np.uint32 if ndist <= 4294967296 else np.int64)

    @property
    def _dz(self):
        """Thread-local zstd decompressor. ZstdDecompressor instances are NOT thread-safe for
        concurrent operations: a single shared one under threaded reads produced intermittent
        'Data corruption detected' errors and hard segfaults (measured: 8-way parallel column
        decodes failed with the shared instance, 3/3 clean with per-thread instances). Every
        existing self._dz.decompress(...) call site works unchanged through this property."""
        d = getattr(self._dzl, 'dz', None)
        if d is None:
            d = self._dzl.dz = zstd.ZstdDecompressor()
        return d

    def vplanes(self, nm):
        """Aligned u64 view of an enc-12 column's planes, memoized. Zero-
        copy when the file offset lands 8-aligned; one copy otherwise."""
        pv = getattr(self, '_vpl', None)
        if pv is None:
            pv = self._vpl = {}
        if nm in pv:
            return pv[nm]
        c = self.cols[nm]
        nb = int(c['bits']) * int(c['nwords']) * 8
        cs = int(c['cstart'])
        if cs % 8 == 0:
            arr = np.frombuffer(self.buf, np.uint64, nb // 8, cs)
        else:
            arr = np.frombuffer(bytes(memoryview(self.buf)[cs:cs + nb]), np.uint64)
        pv[nm] = arr
        return arr

    @staticmethod
    def _pk18_dtype(c):
        b = int(c['pbits'])
        return np.uint8 if b <= 8 else (np.uint16 if b <= 16 else np.uint32)

    def _pk18_frame(self, c, j, dec=None, mv=None, need=None):
        """enc-18: frame j's inflated LE bit-stream as u64 WORDS (wdb_kernels.pk32_words -- one load per
        code). `need` limits the inflate to a byte prefix (zstd streams: a point read pays only to its
        highest row; need = ceil(rows*bits/8) + 8 keeps the straddle word inside the prefix)"""
        import zstandard as _zs
        from wdb_kernels import pk32_words as _words
        if dec is None: dec = _zs.ZstdDecompressor()
        if mv is None: mv = memoryview(self.buf)
        po = c['poffs']; base = c['cstart']
        fb = mv[base + int(po[j]):base + int(po[j + 1])]
        if need is not None:
            import io as _io
            return _words(dec.stream_reader(_io.BytesIO(fb)).read(int(need)))
        return _words(dec.decompress(fb))

    def _raw_codes(self, nm):
        if nm in self._codes: return self._codes[nm]
        c = self.cols[nm]
        if c.get('code_enc') == 13:
            cc = self._e13_band(nm, 0, self.N)
            self._codes[nm] = cc
            return cc
        if c['mode'] == 6:
            cc = np.zeros(self.N, dtype=np.uint8)    # constant column: a single group
            self._codes[nm] = cc; return cc
        if c['mode'] == 4:
            dt = np.uint32 if self.N <= 4294967296 else np.int64
            cc = np.arange(self.N, dtype=dt)         # identity codes: value = f(position)
            self._codes[nm] = cc; return cc
        if c['mode'] == 5:
            uniq, inv = np.unique(self._inline_values(c), return_inverse=True)
            c['_idict'] = uniq                       # sorted distinct values, for fetch()
            cc = inv.astype(self._cdt(len(uniq))); self._codes[nm] = cc; return cc
        if c['mode'] == 3:
            # dependent column: gather Y-codes through the determinant's per-row codes
            x_codes = self._raw_codes(self.order[c['det_idx']])
            fm = self._fd_map(c)
            cc = fm[x_codes].astype(self._cdt(int(fm.max()) + 1 if fm.size else 1))
            self._codes[nm] = cc; return cc
        if c.get('code_enc', 0) == 2:                # staircase: codes = repeat(arange, step gaps)
            st = self.stairs(nm)
            cnts = np.diff(np.concatenate(([0], st, [self.N])))
            cc = np.repeat(np.arange(cnts.size, dtype=self._cdt(cnts.size)), cnts)
            self._codes[nm] = cc; return cc
        if c.get('code_enc', 0) == 6:                # warm buckets: parallel two-tier decode
            import wdb_kernels as _WK
            pk = np.frombuffer(self.buf, dtype=np.uint8, count=c['czlen'], offset=c['cstart'])
            cc = _WK.enc6_stream(pk, np.asarray(c['e5hot']), np.asarray(c['e6warm']),
                                 np.asarray(c['e6wb']), np.asarray(c['e5patch']),
                                 np.asarray(c['e6o1']).astype(np.int64),
                                 np.asarray(c['e6o2']).astype(np.int64),
                                 np.int64(self.N), np.int64(c['BR']))
            self._codes[nm] = cc; return cc
        if c.get('code_enc', 0) == 5:                # patched buckets: parallel nibble decode
            import wdb_kernels as _WK
            pk = np.frombuffer(self.buf, dtype=np.uint8, count=c['czlen'], offset=c['cstart'])
            cc = _WK.enc5_stream(pk, np.asarray(c['e5hot']),
                                 np.asarray(c['e5patch']),
                                 np.asarray(c['e5off']).astype(np.int64),
                                 np.int64(self.N), np.int64(c['BR']))
            self._codes[nm] = cc; return cc
        if c.get('code_enc', 0) == 8:                # sparse-default: 3-pass parallel expand
            import wdb_kernels as _WK
            pb = np.ascontiguousarray(self.buf[c['e8pres']:c['e8pres'] + (self.N + 7) // 8])
            ck = np.frombuffer(self.buf[c['e8ck']:c['e8ck'] + ((self.N + 65535) // 65536) * 8],
                               dtype=np.uint64)
            pos = np.empty(c['e8n'], dtype=np.int64)
            _WK.e8_pos(pb, ck, self.N, pos)
            lb = np.ascontiguousarray(
                self.buf[c['cstart']:c['cstart'] + (c['e8n'] * c['e8bits'] + 7) // 8 + 8])
            lits8 = _WK.unpack_any(lb, c['e8n'], c['e8bits'])
            out8 = _WK.e8_scatter(pos, lits8, self.N, c['e8d'])
            self._codes[nm] = out8; return out8
        if c.get('code_enc', 0) == 10:               # segmented bitpack-plus
            bitsX = int(c['pXbits'])
            nblkX = int(c['pXnblk'])
            dirX = np.frombuffer(self.buf, np.int64, nblkX, c['pXdir'])
            cc = np.zeros(int(c['pXn']), dtype=np.uint16 if bitsX > 8 else np.uint8)
            import wdb_kernels as _WK
            bufX = np.frombuffer(self.buf, np.uint8)
            _WK.bp10_decode(bufX, np.ascontiguousarray(dirX), int(c['pXpay']),
                            bitsX, int(c['pXn']), cc)
            self._codes[nm] = cc; return cc
        if c.get('code_enc', 0) == 9:                # tiered dress: absence + tier planes + tail
            import wdb_kernels as _WK
            pb = np.ascontiguousarray(self.buf[c['e9pres']:c['e9pres'] + (self.N + 7) // 8])
            pres = np.unpackbits(pb, count=self.N).astype(bool)
            cc = np.full(self.N, c['e9d'], dtype=np.uint8 if c['e9bits'] <= 8 else np.uint16)
            rem_idx = np.flatnonzero(pres)
            for tc9, tn9, toff in c['e9tiers']:
                tb = np.unpackbits(np.ascontiguousarray(
                    self.buf[toff:toff + (tn9 + 7) // 8]), count=tn9).astype(bool)
                cc[rem_idx[tb]] = tc9
                rem_idx = rem_idx[~tb]
            if c['e9tail_n']:
                tail = np.frombuffer(self.buf, np.uint8, c['e9tail_n'], c['e9tail'])
                cc[rem_idx] = tail
            self._codes[nm] = cc; return cc
        if c.get('code_enc', 0) == 1:                # zstd of byte-aligned codes (clustered/skewed)
            raw = self._dz.decompress(self.buf[c['cstart']:c['cstart']+c['czlen']].tobytes())
            wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
            cc = np.frombuffer(raw, dtype=wdt)       # native width (was upcast to int64)
            self._codes[nm] = cc; return cc
        if c.get('code_enc', 0) == 17:               # RAW PACKED: one parallel unpack
            wdt17 = np.uint8 if c['bits'] <= 8 else np.uint16
            cc = np.empty(self.N, dtype=wdt17)
            import wdb_kernels as _WK17
            _WK17.pk_unpack(np.asarray(self.buf[c['cstart']:c['cstart'] + c['czlen']]),
                            c['pk_bits'], self.N, cc)
            self._codes[nm] = cc; return cc
        if c.get('code_enc', 0) in (15, 16):         # CLOCK DRESS: anchor+delta+bit -> codes
            if c['code_enc'] == 16:
                anm = c['e16_partner']; role = 0
            else:
                anm = nm; role = 1
            ca = self.cols[anm]
            td15 = np.asarray(self._typed_dict(nm)).astype(np.int64)
            dmin15 = int(td15[0])
            inv_map = getattr(self, '_e14_inv', None)
            if inv_map is None:
                inv_map = self._e14_inv = {}
            inv15 = inv_map.get(nm)
            if inv15 is None:
                inv15 = np.zeros(int(td15[-1]) - dmin15 + 1, dtype=np.uint32)
                inv15[td15 - dmin15] = np.arange(td15.size, dtype=np.uint32)
                inv_map[nm] = inv15
            wdt15 = np.uint8 if c['bits'] <= 8 else (np.uint16 if c['bits'] <= 16 else np.uint32)
            cc = np.empty(self.N, dtype=wdt15)
            import wdb_kernels as _WK15
            base15 = ca['cstart']; FR15 = ca['e15_FR']; nfr15 = ca['e15_nfr']; offs15 = ca['e15_offs']
            st15 = [base15]
            for _k in range(4):
                st15.append(st15[-1] + int(offs15[_k][-1]))
            def _wr15(j9):
                import zstandard as _zs15
                dz9 = _zs15.ZstdDecompressor()
                pls9 = []
                for p9 in range(5):
                    a9 = st15[p9] + int(offs15[p9][j9]); b9 = st15[p9] + int(offs15[p9][j9 + 1])
                    pls9.append(np.frombuffer(dz9.decompress(self.buf[a9:b9].tobytes()), np.uint8))
                l9 = j9 * FR15
                _WK15.e15_reconstruct_chunk(pls9[0], pls9[1], pls9[2], pls9[3], pls9[4],
                                            role, ca['ybase'], inv15, dmin15,
                                            cc[l9:l9 + pls9[0].size])
            if nfr15 > 1:
                from concurrent.futures import ThreadPoolExecutor as _TPc
                with _TPc(max_workers=min(nfr15, 8)) as exc15:
                    list(exc15.map(_wr15, range(nfr15)))
            else:
                _wr15(0)
            self._codes[nm] = cc; return cc
        if c.get('code_enc', 0) == 14:               # FIELD PLANES: framed y/m/d -> codes
            if __import__('os').environ.get('WDB_JOIN_BILL'):
                print('JOIN BILL: E14 FULL RECONSTRUCT fired: %s' % nm, flush=True)
            td14 = np.asarray(self._typed_dict(nm)).astype(np.int64)
            inv_map = getattr(self, '_e14_inv', None)
            if inv_map is None:
                inv_map = self._e14_inv = {}
            dmin14 = int(td14[0])
            inv14 = inv_map.get(nm)
            if inv14 is None:
                inv14 = np.zeros(int(td14[-1]) - dmin14 + 1, dtype=np.uint32)
                inv14[td14 - dmin14] = np.arange(td14.size, dtype=np.uint32)
                inv_map[nm] = inv14
            wdt14 = np.uint8 if c['bits'] <= 8 else (np.uint16 if c['bits'] <= 16 else np.uint32)
            cc = np.empty(self.N, dtype=wdt14)
            import wdb_kernels as _WK14
            base14 = c['cstart']; FR14 = c['e14_FR']; nfr14 = c['e14_nfr']; offs14 = c['e14_offs']
            st14 = [base14,
                    base14 + int(offs14[0][-1]),
                    base14 + int(offs14[0][-1]) + int(offs14[1][-1])]
            def _wr14(j9):
                import zstandard as _zs14
                dz9 = _zs14.ZstdDecompressor()
                pls9 = []
                for p9 in range(3):
                    a9 = st14[p9] + int(offs14[p9][j9]); b9 = st14[p9] + int(offs14[p9][j9 + 1])
                    pls9.append(np.frombuffer(dz9.decompress(self.buf[a9:b9].tobytes()), np.uint8))
                l9 = j9 * FR14
                _WK14.e14_reconstruct_chunk(pls9[0], pls9[1], pls9[2], c['ybase'],
                                            inv14, dmin14, cc[l9:l9 + pls9[0].size])
            if nfr14 > 1:
                from concurrent.futures import ThreadPoolExecutor as _TPr
                with _TPr(max_workers=min(nfr14, 8)) as exr:
                    list(exr.map(_wr14, range(nfr14)))
            else:
                _wr14(0)
            self._codes[nm] = cc; return cc
        if c.get('code_enc', 0) == 3:                # blocked: decompress every frame, concat
            wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
            cc = np.empty(self.N, dtype=wdt)
            BR = c['BR']; base = c['cstart']; bo = c['boffs']
            isz = wdt().itemsize
            nb = int(bo.size) - 1
            def _span(t, T=8):                       # non-overlapping writes; zstd drops the
                import zstandard as _zs              # GIL, so 8 lanes decompress side by side.
                dec = _zs.ZstdDecompressor()         # One span task per worker and one
                lo = t * nb // T                     # decompressor per span: the per-frame
                hi = (t + 1) * nb // T               # task dispatch was 2.3s of lock.acquire
                for j in range(lo, hi):              # across the trench (the shared-function
                    raw = dec.decompress(            # audit's whale).
                        self.buf[base + int(bo[j]):base + int(bo[j + 1])].tobytes())
                    cc[j * BR:j * BR + len(raw) // isz] = np.frombuffer(raw, dtype=wdt)
            if nb > 4:
                list(_pool().map(_span, range(8)))
            else:
                _span(0, 1)
            self._codes[nm] = cc; return cc
        if c.get('code_enc', 0) == 18:               # PACKED FRAMES: inflate + unpack, fourteen lanes
            cc = np.empty(self.N, dtype=self._pk18_dtype(c))
            BR = c['BR']; nb = int(c['poffs'].size) - 1
            import wdb_kernels as _WK18
            def _span18(js):
                import zstandard as _zs
                dec = _zs.ZstdDecompressor(); mv = memoryview(self.buf)
                for j in js:
                    fb = self._pk18_frame(c, j, dec, mv)
                    n9 = min(BR, self.N - j * BR)
                    _WK18.pk32_unpack_serial(fb, int(c['pbits']), n9, cc[j * BR:j * BR + n9])
            if nb > 4:
                from concurrent.futures import ThreadPoolExecutor as _TP18
                with _TP18(max_workers=14) as ex18:
                    list(ex18.map(_span18, np.array_split(np.arange(nb), 14)))
            else:
                _span18(np.arange(nb))
            self._codes[nm] = cc; return cc
        bits = c['bits']; base = c['cstart']
        if c.get('code_enc', 0) == 12:
            import wdb_kernels as _WK
            outF = np.zeros(self.N, np.uint64)
            _WK.vp_window(self.vplanes(nm), int(c['nwords']), int(bits), 0, self.N, outF)
            wdtF = np.uint8 if bits <= 8 else (np.uint16 if bits <= 16 else np.uint32)
            cc = outF.astype(wdtF)
        elif c.get('code_enc', 0) == 0 and 'boffs' not in c and 0 < bits <= 32 \
                and self.N >= (1 << 22):             # kernel full decode: the scan
            import wdb_kernels as _WK                # family rides bit math too;
            wdt0 = np.uint8 if bits <= 8 else (np.uint16 if bits <= 16 else np.uint32)
            cc = np.zeros(self.N, wdt0)              # native width, kernel at scale
            _WK.bp0_decode(np.frombuffer(self.buf, np.uint8), int(base),
                           int(bits), self.N, cc)
        else:
            cc = self._bitunpack(base, 0, self.N, bits)  # native width
        self._codes[nm] = cc; return cc
    def pair_bits(self, nm):
        """THE BIT READ (Jackson's declared clock, its purpose): for a pair
        column, decompress ONLY the delta and orientation streams. Returns
        (bit_bool[N] meaning anchor-column <= partner, delta_u8[N])."""
        c = self.cols[nm]
        anm = c['e16_partner'] if c['code_enc'] == 16 else nm
        cached = (getattr(self, '_e14_pl', None) or {}).get((anm, 'e15s'))
        if cached is not None:
            # SIBLING SHARE: the streams are up -- the bit costs an unpack only.
            BB = cached[4]
            bit = np.unpackbits(BB)[:self.N].astype(np.bool_)
            return bit, cached[3]
        ca = self.cols[anm]
        base = ca['cstart']; FR = ca['e15_FR']; nfr = ca['e15_nfr']; offs = ca['e15_offs']
        st = [base]
        for _k in range(4):
            st.append(st[-1] + int(offs[_k][-1]))
        DL = np.empty(self.N, np.uint8)
        BB = np.empty((self.N + 7) >> 3, np.uint8)
        def _ws(job):
            import zstandard as _zs15
            p9, j9 = job
            a9 = st[p9] + int(offs[p9][j9]); b9 = st[p9] + int(offs[p9][j9 + 1])
            raw = np.frombuffer(_zs15.ZstdDecompressor().decompress(
                self.buf[a9:b9].tobytes()), np.uint8)
            if p9 == 3:
                DL[j9 * FR: j9 * FR + raw.size] = raw
            else:
                f8 = FR >> 3
                BB[j9 * f8: j9 * f8 + raw.size] = raw
        jobs = [(3, j) for j in range(nfr)] + [(4, j) for j in range(nfr)]
        if len(jobs) > 1:
            list(_leaf_pool().map(_ws, jobs))
        else:
            _ws(jobs[0])
        bit = np.unpackbits(BB)[:self.N].astype(bool)
        return bit, DL

    def _e15_streams(self, anm):
        """All five clock streams of an anchor column, decompressed once and
        CACHED per query (the flush law clears _e14_pl)."""
        cache = getattr(self, '_e14_pl', None)
        if cache is None:
            cache = self._e14_pl = {}
        key = (anm, 'e15s')
        got = cache.get(key)
        if got is not None:
            return got
        import threading as _th15
        lk = getattr(self, '_e15_lock', None)
        if lk is None:
            lk = self._e15_lock = _th15.Lock()
        with lk:                                  # concurrent consumers load ONCE
            got = cache.get(key)
            if got is not None:
                return got
            return self._e15_streams_load(anm, cache, key)

    def _e15_streams_load(self, anm, cache, key):
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
            list(_leaf_pool().map(_ws, jobs))
        else:
            _ws(jobs[0])
        cache[key] = outs
        return outs

    def _e15_band(self, nm, day_lo, day_hi):
        """The clock dress's band test: fused per-frame decompress + civil
        anchor + delta swing + numeric band. Serves BOTH pair columns."""
        c = self.cols[nm]
        if c['code_enc'] == 16:
            anm = c['e16_partner']; role = 0
        else:
            anm = nm; role = 1
        ca = self.cols[anm]
        base = ca['cstart']; FR = ca['e15_FR']; nfr = ca['e15_nfr']; offs = ca['e15_offs']
        st = [base]
        for _k in range(4):
            st.append(st[-1] + int(offs[_k][-1]))
        out = np.empty(self.N, dtype=np.bool_)
        import wdb_kernels as _WK15
        # SIBLING SHARE (Jackson's walk): every clock consumer drinks from
        # ONE decompression -- the band loads the cached streams (paying for
        # them exactly once per query) and later consumers ride free.
        Y5, M5, D5, DL5, BB5 = self._e15_streams(anm)
        ystart, mcum = self._civil_luts(ca['ybase'])
        _WK15.e15_band_lut(Y5, M5, D5, DL5, BB5, role, ystart, mcum,
                           day_lo, day_hi, out)
        return out
    def cost_of(self, nm, n):
        """THE COST CURVE: predicted serve cost (ms) for this column at n
        rows -- cost = a + b*n. a is the fixed stream floor (stored bytes at
        the calibrated medium bandwidth), b the family's measured per-row
        rate. All constants calibrated from the deals/cascade benches;
        derived from file metadata, deterministic, zero-time at plan."""
        c = self.cols.get(nm)
        if c is None:
            return 0.0
        enc = c.get('code_enc', 0)
        MSPB = 1.0 / 1.0e6                    # 1 GB/s decompress -> ms per byte
        if enc == 14:
            a = 3.2 * self.N * MSPB           # y+m+d planes
            b = 13e-6                          # civil math + LUT, measured
        elif enc in (15, 16):
            a = 3.4 * self.N * MSPB           # y/m/d + delta + bit streams
            b = 14e-6
        elif enc == 17:
            a = (c.get('czlen') or 0) * MSPB * 0.15   # mmap-direct unpack
            b = 2e-6
        elif enc == 3:
            a = (c.get('czlen') or 0) * MSPB
            b = 1.5e-6                         # LUT/compare pass
        else:
            a = (c.get('czlen') or 0) * MSPB
            b = 2e-6
        return a + b * float(n)

    def _civil_luts(self, ybase):
        """ystart[k] = epoch-day of Jan 1 of (ybase+k), k in 0..257;
        mcum[m] = non-leap cumulative days before month m (1-based)."""
        luts = getattr(self, '_civil_lut_cache', None)
        if luts is None:
            luts = self._civil_lut_cache = {}
        got = luts.get(ybase)
        if got is not None:
            return got
        import datetime as _dt
        ep = _dt.date(1970, 1, 1)
        ystart = np.array([(_dt.date(ybase + k, 1, 1) - ep).days for k in range(258)],
                          dtype=np.int64)
        mcum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
                        dtype=np.int64)        # 0-based months, non-leap
        luts[ybase] = (ystart, mcum)
        return luts[ybase]

    @staticmethod
    def mask_rows(mask):
        """Parallel flatnonzero for big bool masks."""
        import wdb_kernels as _WKm
        n = mask.shape[0]
        chunk = 1 << 20
        nc = (n + chunk - 1) // chunk
        counts = np.empty(nc, np.int64)
        _WKm.pcount_chunks(mask, counts, chunk)
        offs = np.zeros(nc + 1, np.int64)
        np.cumsum(counts, out=offs[1:])
        out = np.empty(int(offs[-1]), np.int64)
        _WKm.pfill_rows(mask, offs, out, chunk)
        return out
    def _e14_inv_of(self, nm):
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
        """THE PLANE-TEST READ (the field-plane dress's primary consumer):
        a [day_lo, day_hi) date-range mask served straight from the y/m/d
        planes -- a lexicographic band over calendar tuples. No civil math,
        no inverse LUT, no code reconstruction. Returns a bool mask of N."""
        if self.cols[nm].get('code_enc') in (15, 16):
            return self._e15_band(nm, day_lo, day_hi)
        import datetime as _dt14
        cache = getattr(self, '_e14_pl', None)
        if cache is None:
            cache = self._e14_pl = {}
        c = self.cols[nm]
        if __import__('os').environ.get('WDB_JOIN_BILL'):
            print('JOIN BILL: PLANE-TEST %s [%d,%d)' % (nm, day_lo, day_hi), flush=True)
        e0 = _dt14.date(1970, 1, 1)
        a9 = e0 + _dt14.timedelta(days=int(day_lo))
        b9 = e0 + _dt14.timedelta(days=int(day_hi))
        out = np.empty(self.N, dtype=np.bool_)
        import wdb_kernels as _WK14
        if (a9.month, a9.day, b9.month, b9.day) == (1, 1, 1, 1):
            # YEAR-ALIGNED band, FUSED (Jackson's shape): each frame worker
            # tests its chunk THE MOMENT it decompresses it -- bytes hot in
            # that core's cache, no full-plane round-trip through RAM, one
            # pool doing both jobs. The plane never materialises.
            lo14 = max(0, a9.year - c['ybase'])
            hi14 = min(256, b9.year - c['ybase'])
            if hi14 <= lo14:
                out[:] = False; return out
            if lo14 == 0 and hi14 == 256:
                out[:] = True; return out
            base = c['cstart']; FR = c['e14_FR']; nfr = c['e14_nfr']; offs = c['e14_offs']
            def _wy(j9):
                import zstandard as _zs14
                a1 = base + int(offs[0][j9]); b1 = base + int(offs[0][j9 + 1])
                raw = np.frombuffer(_zs14.ZstdDecompressor().decompress(
                    self.buf[a1:b1].tobytes()), np.uint8)
                l1 = j9 * FR
                np.less(raw - np.uint8(lo14), np.uint8(hi14 - lo14),
                        out=out[l1:l1 + raw.size])      # unsigned trick: lo<=v<hi
            if nfr > 1:
                from concurrent.futures import ThreadPoolExecutor as _TPy
                with _TPy(max_workers=min(nfr, 8)) as exy:
                    list(exy.map(_wy, range(nfr)))
            else:
                _wy(0)
            return out
        pls = cache.get(nm)
        if pls is None:
            pls = cache[nm] = self._e14_planes(nm)
        _WK14.e14_band_test(pls[0], pls[1], pls[2], c['ybase'],
                            a9.year, a9.month - 1, a9.day - 1,
                            b9.year, b9.month - 1, b9.day - 1, out)
        return out

    def _e14_plane(self, nm, p9):
        """Decompress ONE field plane (0=y, 1=m, 2=d), frames in parallel."""
        c = self.cols[nm]
        base = c['cstart']; FR = c['e14_FR']; nfr = c['e14_nfr']; offs = c['e14_offs']
        starts = [base, base + int(offs[0][-1]), base + int(offs[0][-1]) + int(offs[1][-1])]
        out = np.empty(self.N, np.uint8)
        def _w1(j9):
            import zstandard as _zs14
            a9 = starts[p9] + int(offs[p9][j9]); b9 = starts[p9] + int(offs[p9][j9 + 1])
            raw = _zs14.ZstdDecompressor().decompress(self.buf[a9:b9].tobytes())
            lo9 = j9 * FR
            out[lo9:lo9 + len(raw)] = np.frombuffer(raw, np.uint8)
        if nfr > 1:
            from concurrent.futures import ThreadPoolExecutor as _TP14
            with _TP14(max_workers=min(nfr, 8)) as ex14:
                list(ex14.map(_w1, range(nfr)))
        else:
            _w1(0)
        return out

    def _e14_planes(self, nm):
        """Decompress the three field planes, ALL FRAMES IN PARALLEL (zstd
        releases the GIL). Returns (Y, M, D) u8 arrays of length N."""
        c = self.cols[nm]
        base = c['cstart']; FR = c['e14_FR']; nfr = c['e14_nfr']; offs = c['e14_offs']
        starts = [base, base + int(offs[0][-1]), base + int(offs[0][-1]) + int(offs[1][-1])]
        outs = [np.empty(self.N, np.uint8) for _ in range(3)]
        jobs = []
        for p9 in range(3):
            for j9 in range(nfr):
                jobs.append((p9, j9))
        def _w14(job):
            import zstandard as _zs14
            p9, j9 = job
            a9 = starts[p9] + int(offs[p9][j9]); b9 = starts[p9] + int(offs[p9][j9 + 1])
            raw = _zs14.ZstdDecompressor().decompress(self.buf[a9:b9].tobytes())
            lo9 = j9 * FR
            outs[p9][lo9:lo9 + len(raw)] = np.frombuffer(raw, np.uint8)
        if len(jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor as _TP14
            list(_leaf_pool().map(_w14, jobs))
        else:
            _w14(jobs[0])
        return outs

    def codes_band(self, nm, lo, hi):
        """enc-3 band read: decompress ONLY the frames covering [lo,hi),
        8 zstd lanes, band-relative result, nothing cached -- the range
        path the blocked dress never had."""
        c = self.cols[nm]
        if nm in self._codes:
            return self._codes[nm][lo:hi]
        if c.get('code_enc') == 13:
            return self._e13_band(nm, lo, hi)
        if c.get('code_enc') == 14:
            return np.asarray(self.codes(nm))[lo:hi]
        if c.get('code_enc') == 18:
            return self._raw_codes_range(nm, lo, hi)
        wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
        BR = int(c['BR']); base = c['cstart']; bo = c['boffs']
        isz = wdt().itemsize
        j0 = lo // BR
        j1 = (hi + BR - 1) // BR
        out = np.empty(j1 * BR - j0 * BR, dtype=wdt)
        nb = j1 - j0
        def _span(t, T=8):
            import zstandard as _zs
            dec = _zs.ZstdDecompressor()
            a = t * nb // T
            b = (t + 1) * nb // T
            for jj in range(a, b):
                j = j0 + jj
                raw = dec.decompress(
                    self.buf[base + int(bo[j]):base + int(bo[j + 1])].tobytes())
                out[jj * BR:jj * BR + len(raw) // isz] = np.frombuffer(raw, dtype=wdt)
        if nb > 2:
            list(_pool().map(_span, range(8)))
        else:
            _span(0, 1)
        return out[lo - j0 * BR:hi - j0 * BR]

    def _e13_band(self, nm, lo, hi, planes=None):
        """Byte-plane range read: per touched frame, inflate only the
        planes asked for (all, for exact codes; fewer for coarse reads)
        and weave bytes back into codes. Eight lanes, nothing cached."""
        c = self.cols[nm]
        nby = int(c['vnby']); BR = int(c['BR'])
        base = c['cstart']; bo = c['boffs']
        pb = nby if planes is None else min(planes, nby)
        j0 = lo // BR; j1 = (hi + BR - 1) // BR
        nb = j1 - j0
        out = np.zeros((j1 - j0) * BR, dtype=np.int64)
        def _span(t, T=8):
            import zstandard as _zs
            dec = _zs.ZstdDecompressor()
            for jj in range(t * nb // T, (t + 1) * nb // T):
                j = j0 + jj
                seg9 = out[jj * BR:(jj + 1) * BR]
                for b in range(pb):
                    fi = j * nby + b
                    raw = dec.decompress(
                        self.buf[base + int(bo[fi]):base + int(bo[fi + 1])]
                        .tobytes(), max_output_size=BR)
                    seg9 |= np.frombuffer(raw, np.uint8, BR).astype(np.int64) \
                        << np.int64(8 * (nby - 1 - b))
        if nb > 2:
            list(_pool().map(_span, range(8)))
        else:
            _span(0, 1)
        return out[lo - j0 * BR:hi - j0 * BR]

    def e13_scan_eq(self, nm, code, neq=False):
        """THE BYTE-DESCENT (Jackson's 10-vs-42 economics): per frame,
        inflate plane 0 and compare one byte -- 255/256 of rows die per
        level -- descending only while candidates live. A dead frame
        never inflates its remaining planes. Returns a bool row mask and
        bytes actually inflated, for the referee."""
        import zstandard as _zs
        c = self.cols[nm]
        nby = int(c['vnby']); BR = int(c['BR'])
        base = c['cstart']; bo = c['boffs']
        nfr = (len(bo) - 1) // nby
        out = np.zeros(nfr * BR, bool)
        spent = np.zeros(8, np.int64)
        def _span(t, T=8):
            dec = _zs.ZstdDecompressor()
            for j in range(t * nfr // T, (t + 1) * nfr // T):
                cand = None
                for b in range(nby):
                    fi = j * nby + b
                    comp = self.buf[base + int(bo[fi]):base + int(bo[fi + 1])]
                    raw = dec.decompress(comp.tobytes(), max_output_size=BR)
                    spent[t] += len(comp)
                    pl = np.frombuffer(raw, np.uint8, BR)
                    tb = (code >> (8 * (nby - 1 - b))) & 0xFF
                    m9 = (pl == tb)
                    cand = m9 if cand is None else (cand & m9)
                    if not cand.any():
                        cand = None
                        break
                if cand is not None:
                    out[j * BR:(j + 1) * BR] = cand
        if nfr > 2:
            list(_pool().map(_span, range(8)))
        else:
            _span(0, 1)
        if neq:
            np.invert(out, out=out)
        return out, int(spent.sum())

    def stairs(self, nm):
        """Step rows of a STAIRCASE column (codes non-decreasing in row order, e.g. time-ordered
        ingest): the row indices where the code ticks +1. Per-value counts and row spans derive
        from these by diff -- GROUP BY/point reads with NO code decode. code_enc=2 columns carry
        them gap-packed in the file (unpacked here in ~ms); legacy encodings get a one-time
        monotonicity probe over the decoded codes, cached. None when not a staircase."""
        c = self.cols.get(nm)
        if c is None or c.get('mode') in (4, 6):
            return None
        if '_steps' in c:
            return c['_steps']
        if c.get('code_enc', 0) == 2:
            gaps = self._bitunpack(c['cstart'], 0, c['nsteps'], c['gbits']).astype(np.int64)
            c['_steps'] = np.cumsum(gaps)
            return c['_steps']
        a = self._raw_codes(nm)
        d = np.diff(a.astype(np.int64)) if a.size else np.empty(0, np.int64)
        if a.size and int(a[0]) == 0 and (d.size == 0 or (int(d.min()) >= 0 and int(d.max()) <= 1)):
            c['_steps'] = (np.nonzero(d)[0] + 1).astype(np.int64)
        else:
            c['_steps'] = None
        return c['_steps']
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
    def code_counts(self, nm):
        """Per-code row counts (np.bincount of the code array), cached. Length V (includes the
        null bin at V-1 when has_null). Lets COUNT(*) WHERE P(col) be summed over the dictionary
        in O(distinct) instead of materialising + scanning N values."""
        cc = getattr(self, '_ccounts', None)
        if cc is None: cc = self._ccounts = {}
        if nm not in cc:
            # THE CENSUS SIDECAR (Jackson's split: facts cached, decisions
            # computed). Counts are immutable per-file facts -- persisted
            # once, mmap'd forever; the PEMDAS consults them in microseconds
            # instead of re-scanning 60M rows per query to score potency.
            import os as _os
            fn = self.path + '.' + nm + '.cnt.npy'
            try:
                import wdb_sidecar as _wsc9
                if _wsc9.exists(fn) and _wsc9.is_fresh(_os.path.dirname(self.path), _os.path.basename(fn)):
                    v9 = np.load(fn, mmap_mode='r')
                    if v9.shape[0] == int(self.cols[nm]['V']):
                        cc[nm] = v9
                        return cc[nm]
            except Exception:
                pass
            codes = self.codes(nm)
            import wdb_kernels as _WKc
            cc[nm] = _WKc.bincount_par(codes, self.cols[nm]['V'])            # THE PARALLEL CENSUS
            try:
                import wdb_sidecar as _wsc9
                if not _wsc9.births_on(_os.path.dirname(self.path)):
                    return cc[nm]                                            # THE SWITCH: computed, not born
                np.save(fn + '.tmp.npy', np.asarray(cc[nm]))
                _os.replace(fn + '.tmp.npy', fn)
            except Exception:
                pass
        return cc[nm]
    def _override_vals_typed(self, nm):
        eff = self._effective(nm)
        return eff[1] if eff is not None else []
    def _typed_dict(self, nm):
        if nm in self._tdict: return self._tdict[nm]
        r = self._typed_dict_uncached(nm)
        if self.cols[nm]['mode'] != 6:   # mode-6 synth value can change via register_synth; don't memo
            self._tdict[nm] = r
        # THE WIDTH LAW at the source: a STRING dictionary is an object array, never a Python
        # list -- np.asarray(list_of_bytes) builds a fixed-width S<maxlen> array (6M URLs x a
        # thousands-byte longest value = tens of GB in one C call, GIL held: the fused cascade
        # OOM-killed the server on AVG(length(URL)) before any watchdog could see it)
        _c9 = self.cols[nm]
        if _c9.get('dt') == 1 and isinstance(r, list) and _c9.get('mode') in (0, 1, 2):
            r = np.array(r, dtype=object)
            if _c9['mode'] != 6: self._tdict[nm] = r
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
        if c.get('aux') == 9 and c['dt'] == 1:               # THE BOOL MARKER: decode to Python bools at the source
            return [(v == b'True') for v in self.dict_vals(nm)]
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
        v9 = self._fetch_raw(nm, code)
        if isinstance(v9, (bytes, bytearray)) and self.cols[nm].get('aux') == 9 and self.cols[nm]['dt'] == 1:
            return v9 == b'True'                                  # THE BOOL MARKER at the point read
        return v9

    def _fetch_raw(self, nm, code):
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
            v2 = int(self._dict_ints_at(c, np.array([code], np.int64))[0])
            if c['dt'] == 3:
                unit = _DT_UNITS[c['aux']]
                return np.int64(v2).view(f'datetime64[{unit}]')
            return v2
        if c['mode'] in (0, 3):
            v = c['vals'][code]
            if c['dt'] == 0: return int(v)
            if c['dt'] == 2: return struct.unpack('<d', v)[0]
            if c['dt'] == 3:
                unit = _DT_UNITS[c['aux']]
                return np.int64(struct.unpack('<q', v)[0]).view(f'datetime64[{unit}]')
            return v
        if c.get('chunked'):
            CH = c['CHUNK']; j = code // CH
            buf = c['chunks'].get(j)
            if buf is None:
                fb = c['chunk_base'] + int(c['chunk_foff'][j])
                fe = c['chunk_base'] + int(c['chunk_foff'][j+1])
                buf = self._dz.decompress(bytes(self.buf[fb:fe])); c['chunks'][j] = buf
            R = c['R']; o = int(c['restarts'][code // R]) - int(c['chunk_ustart'][j]); prev = b''
            for _ in range(code % R + 1):
                cp, sl = struct.unpack_from('<HH', buf, o); o += 4
                suf = buf[o:o+sl]; o += sl; prev = prev[:cp] + suf
            return prev
        if c.get('raw') is None: c['raw'] = self._dz.decompress(c['z'])
        raw = c['raw']; R = c['R']; o = int(c['restarts'][code // R]); prev = b''
        for _ in range(code % R + 1):
            cp, sl = struct.unpack_from('<HH', raw, o); o += 4
            suf = raw[o:o+sl]; o += sl; prev = prev[:cp] + suf
        return prev
    def values_at(self, nm, codes):
        """BATCH fetch+pyval: emission-ready Python values for an array of dict codes, exactly
        matching [_pyval(fetch(nm, c)) for c in codes] but without 2M Python calls at deep K.
        Dedupes codes, decompresses each touched dict CHUNK once, walks each front-coded restart
        block once, and converts scalars via .tolist() (C-speed). min(point, pop) inside the dict."""
        import wdb_sql
        codes = np.asarray(codes)
        if codes.size == 0:
            return []
        c = self.cols[nm]
        V = int(c['V'])
        if int(codes.max()) >= V:                    # synthetic override codes: rare, per-code path
            return [wdb_sql._pyval(self.fetch(nm, int(x))) for x in codes]
        u, inv = np.unique(codes, return_inverse=True)
        uv = [None] * u.size                         # value per unique code
        nullmask = c['has_null'] and int(u[-1]) == V - 1
        work = u[:-1] if nullmask else u             # null code decodes to None; skip the walk for it
        mode = c['mode']
        if work.size == 0:
            pass
        elif mode == 6:
            v = wdb_sql._pyval(self._synth[nm])
            for i in range(work.size): uv[i] = v
        elif mode == 4:
            arr = self._seq_decode(c)[work]
            vals = (arr.view(f"datetime64[{_DT_UNITS[c['aux']]}]") if c['dt'] == 3 else arr)
            for i, v in enumerate(vals): uv[i] = wdb_sql._pyval(v)
        elif mode == 5:
            d = c.get('_idict')
            if d is None: self._raw_codes(nm); d = c['_idict']
            for i, cd in enumerate(work.tolist()): uv[i] = wdb_sql._pyval(d[cd])
        elif mode == 2:
            arr = self._dict_ints_at(c, work)
            if c['dt'] == 3:
                vals = arr.view(f"datetime64[{_DT_UNITS[c['aux']]}]")
                for i in range(vals.size): uv[i] = wdb_sql._pyval(vals[i])
            else:
                uv[:work.size] = arr.tolist()        # C-speed int conversion
        elif mode in (0, 3):
            vals = c['vals']; dt = c['dt']
            if dt == 0:
                for i, cd in enumerate(work.tolist()): uv[i] = int(vals[cd])
            elif dt == 2:
                for i, cd in enumerate(work.tolist()): uv[i] = struct.unpack('<d', vals[cd])[0]
            elif dt == 3:
                unit = _DT_UNITS[c['aux']]
                for i, cd in enumerate(work.tolist()):
                    uv[i] = wdb_sql._pyval(np.int64(struct.unpack('<q', vals[cd])[0]).view(f'datetime64[{unit}]'))
            else:
                for i, cd in enumerate(work.tolist()): uv[i] = wdb_sql._pyval(vals[cd])
        else:                                        # front-coded string dict (chunked or whole)
            R = c['R']; restarts = c['restarts']
            chunked = c.get('chunked')
            if not chunked and c.get('raw') is None:
                c['raw'] = self._dz.decompress(c['z'])
            if chunked:
                CHK = c['CHUNK']
                need_ch = sorted({int(x) // CHK for x in work.tolist()})
                missing = [ch for ch in need_ch if ch not in c['chunks']]
                if len(missing) > 1:             # POOL the page pops: 321 serial pops
                    from concurrent.futures import ThreadPoolExecutor   # were 265ms of
                    def _popc(ch):               # j-dump; zstd releases the GIL
                        import zstandard as _z
                        fb = c['chunk_base'] + int(c['chunk_foff'][ch])
                        fe = c['chunk_base'] + int(c['chunk_foff'][ch + 1])
                        return ch, _z.ZstdDecompressor().decompress(bytes(self.buf[fb:fe]))
                    with ThreadPoolExecutor(max_workers=min(8, len(missing))) as ex:
                        for ch, buf2 in ex.map(_popc, missing):
                            c['chunks'][ch] = buf2
            wl = work.tolist()
            i = 0
            while i < len(wl):                       # one walk per touched restart block
                cd = wl[i]; rb = cd // R
                j = i
                while j < len(wl) and wl[j] // R == rb:
                    j += 1
                if chunked:
                    ch = cd // c['CHUNK']
                    buf = c['chunks'].get(ch)
                    if buf is None:
                        fb = c['chunk_base'] + int(c['chunk_foff'][ch])
                        fe = c['chunk_base'] + int(c['chunk_foff'][ch + 1])
                        buf = self._dz.decompress(bytes(self.buf[fb:fe])); c['chunks'][ch] = buf
                    o = int(restarts[rb]) - int(c['chunk_ustart'][ch])
                else:
                    buf = c['raw']; o = int(restarts[rb])
                prev = b''; k = i
                last_rel = wl[j - 1] - rb * R
                for step in range(last_rel + 1):
                    cp, sl = struct.unpack_from('<HH', buf, o); o += 4
                    prev = prev[:cp] + buf[o:o + sl]; o += sl
                    if rb * R + step == wl[k]:
                        try: uv[k] = prev.decode('utf-8', 'surrogatepass')
                        except Exception: uv[k] = prev
                        k += 1
                i = j
        out = [None] * codes.size                    # scatter uniques back to input order
        uvl = uv
        for pos, ui in enumerate(inv.tolist()):
            out[pos] = uvl[ui]
        return out
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
    def group_by_count(self, nm):
        return np.bincount(self.codes(nm), minlength=self.cols[nm]['V'])

    # ---- narrow-before-expand: cluster-key slicing -----------------------------------------
    def cluster_meta(self):
        """Slice-boundary index from the .cluster sidecar, or None. Keys: key, dtype, aux,
        values (sorted unique key, int64/float), offsets (first-row of each, len+1), nn, n."""
        if isinstance(self._cluster, int):            # 0 = not yet loaded
            import os, pickle
            p = self.path + '.cluster'
            self._cluster = pickle.load(open(p, 'rb')) if os.path.exists(p) else None
        return self._cluster

    def cubes(self):
        """Materialised low-card GROUP BY cubes from the .cube sidecar, or [] if none. Each is a dict
        {dims, B, keys, count, sums} -- a precomputed aggregate the executor can answer from directly."""
        if isinstance(self._cubes, int):              # 0 = not yet loaded
            import os, pickle
            p = self.path + '.cube'
            self._cubes = pickle.load(open(p, 'rb')) if os.path.exists(p) else []
        return self._cubes

    def slice_for_predicate(self, nm, op, lit):
        """If nm is the cluster key and (op, lit) is a range/eq, return the contiguous (lo, hi)
        row bounds of the matching slice; else None. lit must be in the key's native numeric
        domain (int, float, or int64 epoch for datetime). Nulls live at [nn, n) and never match,
        so range/eq bounds stay inside [0, nn)."""
        cm = self.cluster_meta()
        if cm is None or cm['key'] != nm or cm['dtype'] == 1 or op not in ('=', '>', '>=', '<', '<='):
            return None
        vals = cm['values']; off = cm['offsets']; nn = int(cm['nn'])
        if op == '=':
            i = int(np.searchsorted(vals, lit, 'left'))
            if i < len(vals) and vals[i] == lit:
                return (int(off[i]), int(off[i + 1]))
            return (0, 0)
        if op in ('>', '>='):
            i = int(np.searchsorted(vals, lit, 'right' if op == '>' else 'left'))
            return (int(off[i]), nn)
        i = int(np.searchsorted(vals, lit, 'left' if op == '<' else 'right'))
        return (0, int(off[i]))

    def e8_planes(self, nm):
        """The sparse dress's planes, served RAW (Jackson's differential read): row
        positions of the literals + the literal codes + the default. Consumers that
        count or mask never densify -- 13.2M elements instead of a 400MB write."""
        c = self.cols.get(nm)
        if c is None or c.get('code_enc', 0) not in (8, 9):
            return None
        pm = self.__dict__.setdefault('_e8pm', {})
        hit = pm.get(nm)                     # planes get an unbounded home:
        if hit is not None:                  # the codes LRU evicted them and
            return hit                       # every query rebuilt 13.2M planes
        import wdb_kernels as _WK
        if c.get('code_enc', 0) == 9:        # the tiered dress speaks planes too
            pb = np.ascontiguousarray(self.buf[c['e9pres']:c['e9pres'] + (self.N + 7) // 8])
            ck = np.frombuffer(self.buf[c['e9ck']:c['e9ck'] + ((self.N + 65535) // 65536) * 8],
                               dtype=np.uint64)
            pos = np.empty(c['e9n'], dtype=np.int64)
            _WK.e8_pos(pb, ck, self.N, pos)
            lits = np.empty(c['e9n'], dtype=np.int64)
            rem9 = np.arange(c['e9n'], dtype=np.int64)
            for tc9, tn9, toff in c['e9tiers']:
                tb = np.unpackbits(np.ascontiguousarray(
                    self.buf[toff:toff + (tn9 + 7) // 8]), count=tn9).astype(bool)
                lits[rem9[tb]] = tc9
                rem9 = rem9[~tb]
            if c['e9tail_n']:
                lits[rem9] = np.frombuffer(self.buf, np.uint8, c['e9tail_n'], c['e9tail'])
            res = (pos, lits, int(c['e9d']))
            pm[nm] = res
            return res
        pb = np.ascontiguousarray(self.buf[c['e8pres']:c['e8pres'] + (self.N + 7) // 8])
        ck = np.frombuffer(self.buf[c['e8ck']:c['e8ck'] + ((self.N + 65535) // 65536) * 8],
                           dtype=np.uint64)
        pos = np.empty(c['e8n'], dtype=np.int64)
        _WK.e8_pos(pb, ck, self.N, pos)
        lb = np.ascontiguousarray(
            self.buf[c['cstart']:c['cstart'] + (c['e8n'] * c['e8bits'] + 7) // 8 + 8])
        lits = _WK.unpack_any(lb, c['e8n'], c['e8bits'])
        res = (pos, lits, int(c['e8d']))
        pm[nm] = res
        return res

    def _raw_codes_range(self, nm, lo, hi):
        """Per-row codes for rows [lo, hi) ONLY. Raw bit-packed columns (code_enc 0) touch just
        the covering bytes -- the narrow-before-expand read. mode 4/6 are positional (free slice).
        Other encodings full-decode then slice (correct; bigger win awaits block decode)."""
        enc_r = self.cols[nm].get('code_enc', 0)
        if enc_r == 8 and (nm in self._codes or hi - lo >= 65536):
            # STREAMING consumers (block-sized ranges): 382 rank-arithmetic calls cost
            # ~570ms where ONE shared 3-pass read costs ~150 and every block becomes a
            # view. Small ranges below keep pure rank arithmetic (point-read economics).
            return self._raw_codes(nm)[lo:hi]
        if enc_r == 8:                               # rank arithmetic: checkpoint + local bits
            c = self.cols[nm]
            pb = np.asarray(self.buf[c['e8pres']:c['e8pres'] + (self.N + 7) // 8], dtype=np.uint8)
            ck = np.frombuffer(self.buf[c['e8ck']:c['e8ck'] + ((self.N + 65535) // 65536) * 8],
                               dtype=np.uint64)
            cb = lo >> 16
            base_row = cb << 16
            up = np.unpackbits(pb[base_row >> 3:(hi + 7) >> 3],
                               count=hi - base_row).astype(bool)
            rank_lo = int(ck[cb]) + int(up[:lo - base_row].sum())
            seg8 = up[lo - base_row:]
            nlit = int(seg8.sum())
            out8 = np.full(hi - lo, c['e8d'], dtype=np.uint32)
            if nlit:
                import wdb_kernels as _WK
                bits8 = c['e8bits']
                b0 = (rank_lo * bits8) >> 3      # parallel window unpack from the
                sh0 = (rank_lo * bits8) & 7      # rank's byte, not the slider
                nb8 = ((sh0 + nlit * bits8) + 7) // 8 + 8
                lb8 = np.ascontiguousarray(self.buf[c['cstart'] + b0:c['cstart'] + b0 + nb8])
                lits8 = _WK.unpack_any_off(lb8, nlit, bits8, sh0)
                out8[np.nonzero(seg8)[0]] = lits8
            return out8
        if enc_r in (5, 6):
            return self._raw_codes(nm)[lo:hi]    # bucket tags: the range reader predates them;
                                                 # the cached full decode is exact and 54ms-class
        if lo >= hi:
            return np.empty(0, dtype=np.int64)
        c = self.cols[nm]
        if c['mode'] == 4: return np.arange(lo, hi, dtype=np.int64)
        if c['mode'] == 6: return np.zeros(hi - lo, dtype=np.int64)
        if c.get('code_enc', 0) == 2:                # staircase: code(row) = #steps at-or-before row
            st = self.stairs(nm)                     # O((hi-lo) log nsteps), touches no code bytes
            return np.searchsorted(st, np.arange(lo, hi), side='right').astype(np.int64)
        if c.get('code_enc', 0) == 3:                # blocked: touched frames only
            if nm in self._codes:
                return self._codes[nm][lo:hi]
            wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
            BR = c['BR']; base = c['cstart']; bo = c['boffs']; dz = self._dz
            out = np.empty(hi - lo, dtype=wdt)
            for j in range(lo // BR, (hi - 1) // BR + 1):
                raw = np.frombuffer(dz.decompress(self.buf[base+int(bo[j]):base+int(bo[j+1])].tobytes()), dtype=wdt)
                a = max(lo, j*BR); b = min(hi, j*BR + raw.size)
                out[a-lo:b-lo] = raw[a-j*BR:b-j*BR]
            return out
        if c.get('code_enc', 0) == 18:               # packed frames: touched frames, unpacked
            if nm in self._codes:
                return self._codes[nm][lo:hi]
            import wdb_kernels as _WK18
            BR = c['BR']; bits18 = int(c['pbits'])
            if hi - lo > 8 * BR:
                return self._raw_codes(nm)[lo:hi]    # a wide window: the fourteen-lane full decode
            out = np.empty(hi - lo, dtype=self._pk18_dtype(c))
            for j in range(lo // BR, (hi - 1) // BR + 1):
                a = max(lo, j * BR); b = min(hi, j * BR + BR, self.N)
                fb = self._pk18_frame(c, j)
                rel = np.arange(a - j * BR, b - j * BR, dtype=np.int64)
                tmp = np.empty(rel.size, np.int64)
                _WK18.pk32_gather(fb, bits18, rel, tmp)
                out[a - lo:b - lo] = tmp
            return out
        if c.get('code_enc', 0) == 10 and nm not in self._codes:
            import wdb_kernels as _WK             # window: gather the span only
            rowsW = np.arange(lo, hi, dtype=np.int64)
            outW = np.zeros(rowsW.size, dtype=np.uint16 if int(c['pXbits']) > 8 else np.uint8)
            dirW = np.frombuffer(self.buf, np.int64, int(c['pXnblk']), c['pXdir'])
            _WK.bp10_gather(np.frombuffer(self.buf, np.uint8), np.ascontiguousarray(dirW),
                            int(c['pXpay']), int(c['pXbits']), rowsW, outW)
            return outW
        if c['mode'] in (3, 5) or c.get('code_enc', 0) in (1, 8, 9, 10):
            return self._raw_codes(nm)[lo:hi]      # dresses without frames
        if c.get('code_enc', 0) == 12:
            import wdb_kernels as _WK
            bitsR = int(c['bits'])
            outR = np.zeros(hi - lo, np.uint64)
            _WK.vp_window(self.vplanes(nm), int(c['nwords']), bitsR, lo, hi, outR)
            wdtR = np.uint8 if bitsR <= 8 else (np.uint16 if bitsR <= 16 else np.uint32)
            return outR.astype(wdtR)
        if c.get('code_enc', 0) == 0 and 'boffs' not in c and c.get('bits') \
                and 0 < int(c['bits']) <= 32 and (hi - lo) >= (1 << 16):
            import wdb_kernels as _WK            # window decode at kernel speed
            bitsW = int(c['bits'])
            wdtW = np.uint8 if bitsW <= 8 else (np.uint16 if bitsW <= 16 else np.uint32)
            outW = np.zeros(hi - lo, wdtW)
            _WK.bp0_gather(np.frombuffer(self.buf, np.uint8), int(c['cstart']),
                           bitsW, np.arange(lo, hi, dtype=np.int64), outW)
            return outW
        return self._bitunpack(c['cstart'], lo, hi, c['bits'])

    def codes_at(self, nm, rows):
        """Batch point-pop: codes at the given sorted-or-not row positions, decompressing ONLY the
        touched enc=3 frames (~0.6 ms each). min(point, pop): callers with huge scattered row sets
        fall through to the cached full decode automatically once it exists. Non-blocked encodings
        answer from their own point paths (bitpack arithmetic, stair searchsorted, full decode)."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return np.empty(0, dtype=np.int64)
        c = self.cols[nm]
        if c.get('code_enc') == 17 and nm not in self._codes:
            # RAW PACKED: direct bit gather at rows, no full unpack
            out17 = np.empty(rows.size, dtype=np.int64)
            import wdb_kernels as _WKp
            _WKp.pk_gather(np.asarray(self.buf[c['cstart']:c['cstart'] + c['czlen']]),
                           c['pk_bits'], rows, out17)
            return out17
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
            lo13 = int(rows.min()); hi13 = int(rows.max()) + 1
            band13 = self._e13_band(nm, lo13, hi13)
            return band13[rows - lo13]
        if c.get('code_enc', 0) == 10:
            if nm in self._codes:                    # already decoded: gather free
                return self._codes[nm][np.asarray(rows, np.int64)]
            import wdb_kernels as _WK
            rowsX = np.asarray(rows, np.int64)
            srt = np.argsort(rowsX, kind='stable')   # kernel wants sorted rows
            outX = np.zeros(rowsX.size, dtype=np.uint16 if int(c['pXbits']) > 8 else np.uint8)
            dirX = np.frombuffer(self.buf, np.int64, int(c['pXnblk']), c['pXdir'])
            _WK.bp10_gather(np.frombuffer(self.buf, np.uint8), np.ascontiguousarray(dirX),
                            int(c['pXpay']), int(c['pXbits']), rowsX[srt], outX)
            inv9 = np.empty_like(srt); inv9[srt] = np.arange(srt.size)
            return outX[inv9]
        if c.get('code_enc', 0) == 9:
            rows9 = np.asarray(rows, np.int64)
            pl9 = self.e8_planes(nm)         # planes speak tag-9, memoized
            pos9 = pl9[0]; lit9 = pl9[1]
            j9 = np.searchsorted(pos9, rows9)
            j9c = np.minimum(j9, max(0, pos9.size - 1))
            hit9 = (pos9[j9c] == rows9) if pos9.size else np.zeros(rows9.size, bool)
            out9 = np.full(rows9.size, pl9[2], np.int64)
            if pos9.size:
                out9[hit9] = lit9[j9c[hit9]]
            return out9
        if c.get('code_enc', 0) == 8 and nm not in self._codes:
            pos8, lits8, d8 = self.e8_planes(nm)   # point reads: binary search the
            out = np.full(rows.size, d8, dtype=np.int64)   # positions, never densify
            if pos8.size:
                idx = np.searchsorted(pos8, rows)
                idx2 = np.minimum(idx, pos8.size - 1)
                hit = pos8[idx2] == rows
                out[hit] = lits8[idx2[hit]].astype(np.int64)
            return out
        if c.get('code_enc', 0) == 6 and nm not in self._codes and rows.size < (self.N >> 2):
            import wdb_kernels as _WK
            pk = np.frombuffer(self.buf, dtype=np.uint8, count=c['czlen'], offset=c['cstart'])
            order = np.argsort(rows, kind='stable')
            rs = np.ascontiguousarray(np.asarray(rows, dtype=np.int64)[order])
            got = _WK.enc6_at2(pk, np.asarray(c['e5hot']), np.asarray(c['e6warm']),
                               np.asarray(c['e6wb']), np.asarray(c['e5patch']),
                               np.asarray(c['e6o1']).astype(np.int64),
                               np.asarray(c['e6o2']).astype(np.int64), rs,
                               np.int64(self.N), np.int64(c['BR']))
            out = np.empty_like(got)
            out[order] = got
            return out
        if c.get('code_enc', 0) == 5 and nm not in self._codes and rows.size < (self.N >> 2):
            import wdb_kernels as _WK
            pk = np.frombuffer(self.buf, dtype=np.uint8, count=c['czlen'], offset=c['cstart'])
            order = np.argsort(rows, kind='stable')
            rs = np.ascontiguousarray(np.asarray(rows, dtype=np.int64)[order])
            got = _WK.enc5_at2(pk, np.asarray(c['e5hot']), np.asarray(c['e5patch']),
                               np.asarray(c['e5off']).astype(np.int64), rs,
                               np.int64(self.N), np.int64(c['BR']))
            out = np.empty_like(got)
            out[order] = got
            return out
        if c.get('code_enc', 0) == 12 and rows.size < (self.N >> 1):
            import wdb_kernels as _WK
            rows0 = np.ascontiguousarray(np.asarray(rows, np.int64))
            outV = np.zeros(rows0.size, np.int64)
            _WK.vp_gather(self.vplanes(nm), int(c['nwords']), int(c['bits']), rows0, outV)
            bitsV = int(c['bits'])
            wdtV = np.uint8 if bitsV <= 8 else (np.uint16 if bitsV <= 16 else np.uint32)
            return outV.astype(wdtV)
        if c.get('code_enc', 0) == 0 and c.get('bits') and 'cstart' in c \
                and 'boffs' not in c and nm not in self._codes \
                and rows.size < (self.N >> 2):
            import wdb_kernels as _WK                # plain bitpack: pure bit math
            rows0 = np.ascontiguousarray(np.asarray(rows, np.int64))
            bits0 = int(c['bits'])
            wdt0 = np.uint8 if bits0 <= 8 else (np.uint16 if bits0 <= 16 else np.uint32)
            out0 = np.zeros(rows0.size, wdt0)
            _WK.bp0_gather(np.frombuffer(self.buf, np.uint8), int(c['cstart']),
                           int(c['bits']), rows0, out0)
            return out0
        if c.get('code_enc', 0) == 1 and nm not in self._codes and rows.size <= 65536:
            # ONE FRAME, READ TO THE LAST ROW ASKED: enc-1 is a single zstd frame, and a point read
            # inflated all 100M codes (ParamPrice: 87 ms for ten rows). zstd streams, so the prefix
            # up to the highest row is what is inflated -- the first ten hits of a scan are early rows
            import zstandard as _z1, io as _io1
            wdt1 = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
            isz1 = np.dtype(wdt1).itemsize
            need1 = (int(rows.max()) + 1) * isz1
            if need1 * 2 <= int(self.N) * isz1:
                raw1 = np.frombuffer(_z1.ZstdDecompressor().stream_reader(
                    _io1.BytesIO(memoryview(self.buf)[c['cstart']:c['cstart'] + c['czlen']])).read(need1), dtype=wdt1)
                return raw1[rows]
        _sc = self.__dict__.get('_scan_codes')
        if _sc is not None and nm in _sc and c.get('code_enc', 0) in (3, 18) and nm not in self._codes:
            # THE SCANNED CODES: a frame scan on this column kept its codes at the hit positions; a
            # gather over those positions (or a subset of them) is a lookup, not 1,526 decompressions
            _sp, _scc = _sc[nm]
            if rows.size <= _sp.size:
                _ix = np.searchsorted(_sp, rows)
                _ok = _ix < _sp.size
                if _ok.all() and np.array_equal(_sp[_ix], rows):
                    return _scc[_ix]
        if c.get('code_enc', 0) == 18 and nm not in self._codes and rows.size < (self.N >> 2):
            # PACKED FRAMES, AT ROWS: inflate each touched frame to the byte its highest row needs,
            # gather by bit arithmetic; frames in lanes when there are many
            import wdb_kernels as _WK18
            bits18 = int(c['pbits']); BR = c['BR']
            out18 = np.empty(rows.size, dtype=np.int64)
            blks = rows // BR
            order = np.argsort(blks, kind='stable'); rs = rows[order]; bs = blks[order]
            ub, starts = np.unique(bs, return_index=True); ends = np.append(starts[1:], bs.size)
            def _pop18(js):
                import zstandard as _zs
                dec = _zs.ZstdDecompressor(); mv = memoryview(self.buf)
                for t in js:
                    j = int(ub[t]); s = int(starts[t]); e = int(ends[t])
                    rel = (rs[s:e] - j * BR).astype(np.int64)
                    frame_rows = min(BR, self.N - j * BR)
                    need = ((int(rel.max()) + 1) * bits18 + 7) // 8 + 8
                    fb = self._pk18_frame(c, j, dec, mv, need if need * 4 <= frame_rows * bits18 // 8 * 3 else None)
                    _WK18.pk32_gather(fb, bits18, rel, out18[s:e])
            nt = int(ub.size)
            if nt > 16:
                from concurrent.futures import ThreadPoolExecutor as _TP18
                _W = 14 if nt >= 512 else 8
                with _TP18(max_workers=min(_W, nt)) as ex18:
                    list(ex18.map(_pop18, np.array_split(np.arange(nt), min(_W, nt))))
            else:
                _pop18(np.arange(nt))
            inv = np.empty_like(order); inv[order] = np.arange(order.size)
            return out18[inv]
        if c.get('code_enc', 0) != 3 or nm in self._codes or rows.size >= (self.N >> 2):
            # huge row sets: ONE full decode + one vectorized gather beats touching every
            # frame through a positional walk (sq-nested passed ~90M positions here)
            return np.asarray(self._raw_codes(nm))[rows] if c.get('code_enc', 0) != 2 else \
                np.searchsorted(self.stairs(nm), rows, side='right').astype(np.int64)
        wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
        BR = c['BR']; base = c['cstart']; bo = c['boffs']; dz = self._dz
        out = np.empty(rows.size, dtype=wdt)
        blks = rows // BR
        order = np.argsort(blks, kind='stable')     # group rows by frame in one sort --
        rs = rows[order]; bs = blks[order]          # the old per-block mask rescan was
        ub, starts = np.unique(bs, return_index=True)   # O(blocks x pos) comparisons
        ends = np.append(starts[1:], bs.size)
        import io as _io
        isz = np.dtype(wdt).itemsize
        mv = memoryview(self.buf)                        # slices without the copy

        def _popf(t):
            j, s, e = t
            import zstandard as _z                       # per-task decompressor: the
            fb = mv[base + int(bo[j]):base + int(bo[j + 1])]                  # shared one
            frame_rows = min(BR, self.N - j * BR)        # is not thread-safe
            mx = int(rs[s:e].max()) - j * BR
            need = (mx + 1) * isz
            dz2 = _z.ZstdDecompressor()
            if need * 4 <= frame_rows * isz * 3:         # PARTIAL-FRAME READ: zstd streams
                raw = np.frombuffer(                     # decompress prefixes, so a frame
                    dz2.stream_reader(_io.BytesIO(fb)).read(need),   # is only inflated to
                    dtype=wdt)                           # its highest requested row
            else:
                raw = np.frombuffer(dz2.decompress(fb), dtype=wdt)
            return j, s, e, raw

        tasks = list(zip(ub.tolist(), starts.tolist(), ends.tolist()))
        if len(tasks) > 16:                              # POOLED POPS: 95 scattered (a pool for ten frames cost more than the frames)
            from concurrent.futures import ThreadPoolExecutor   # winners were 95 serial
            _W = 14 if len(tasks) >= 512 else 8          # inflations; wide sets use every core
            with ThreadPoolExecutor(max_workers=min(_W, len(tasks))) as ex:
                for j, s, e, raw in ex.map(_popf, tasks):
                    out[order[s:e]] = raw[rs[s:e] - j * BR]
        else:
            for t in tasks:
                j, s, e, raw = _popf(t)
                out[order[s:e]] = raw[rs[s:e] - j * BR]
        return out

    def values_range(self, nm, lo, hi):
        """Decoded values for rows [lo, hi) only (the cluster-slice read). Mirrors _base_values'
        code->value mapping over the slice. No override support (clustered segments are fresh)."""
        c = self.cols[nm]
        if c['mode'] == 4:
            arr = self._seq_decode(c)[lo:hi]
            return arr.view(f"datetime64[{_DT_UNITS[c['aux']]}]") if c['dt'] == 3 else arr
        if c['mode'] == 5: return self._inline_values(c)[lo:hi]
        if c['mode'] == 6: return self._const_array(nm)[lo:hi]
        codes = self._raw_codes_range(nm, lo, hi); dvals = self._typed_dict(nm)
        if c['has_null']:
            nullcode = c['V'] - 1; lut = np.empty(c['V'], dtype=object)
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
