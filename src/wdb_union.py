"""THE MERGED-DICTIONARY VIEW (B, step 3): K segments as ONE table for every door that reads
the segment API. Each segment has its own dictionary per column, so the same code means
different values in different segments. The union merges the K sorted dictionaries into one
(V-scale), keeps a remap table per segment (old code -> merged code), and presents
codes(col) as the concatenation of the remapped streams; values, fetch, typed_dict,
presence and counts derive from those two. Row order is the segments' order (segment 0's
rows, then segment 1's...). Row order serves the code streams; dictionary order serves the
values -- the two levers are separate (Jackson's reading), and the merged dictionary is
sorted by value like every segment's is, so binary search and range predicates still hold.

Memory: the merged dictionary is V-scale (shelf-governed); the remapped code stream is
N-scale per touched column and lives in this view's query-lifetime caches (freed by
drop_derived like a segment's). Everything a door might ask that the union cannot answer
raises NotImplementedError BY NAME -- a decline, never a wrong.
"""
import numpy as np


class SegmentUnion:
    def __init__(self, segs, paths):
        assert len(segs) >= 2
        self.segs = list(segs); self.paths = list(paths)
        self.N = int(sum(int(s.N) for s in segs))
        import os, hashlib
        # THE UNION BIRTHMARK: the virtual name carries a hash of the segment paths AND their
        # mtimes/sizes, so a sidecar born under one union can never be read by another with
        # different rows (the suite's matrix rebuilt a two-segment table under the same name
        # and a stale census answered 69,829 for 116,469)
        stamp = hashlib.sha1('|'.join('%s:%d:%d' % (p, int(os.stat(p).st_mtime_ns), os.stat(p).st_size) for p in paths).encode()).hexdigest()[:12]
        self.path = os.path.join(os.path.dirname(paths[0]), os.path.basename(paths[0]).split('_')[0] + '.union-%s.wdb' % stamp)
        self._tdict = {}; self._codes = {}; self._remap = {}; self._meta = {}
        self._starts = np.zeros(len(segs) + 1, np.int64); np.cumsum([int(s.N) for s in segs], out=self._starts[1:])
        # the column set is the intersection in segment 0's order; metadata merges lazily
        names = [c for c in segs[0].order if all(c in s.cols for s in segs)]
        self.order = names
        self.cols = _LazyCols(self)
        self._presence = 0

    # ---- metadata ----
    def _merge(self, col):
        """merged typed dictionary + per-segment remap for one column (once)"""
        if col in self._remap: return
        segs = self.segs
        dts = {s.cols[col].get('dt') for s in segs}
        if len(dts) != 1: raise NotImplementedError('union: column %s has mixed types across segments' % col)
        dt = dts.pop()
        parts = []; nulls = []
        for s in segs:
            c = s.cols[col]
            if c.get('mode') in (0, 1, 2):
                td = s._typed_dict(col)
                td = np.asarray(td, dtype=object) if (dt == 1 and not isinstance(td, np.ndarray)) else np.asarray(td)
                if c.get('has_null'):
                    parts.append(td[: int(c['V']) - 1]); nulls.append(True)
                else:
                    parts.append(td); nulls.append(False)
            elif c.get('mode') in (4, 5):
                vals = np.asarray(s.values(col))
                if dt == 1 and vals.dtype != object: vals = vals.astype(object)
                u = np.unique(vals[vals != None]) if vals.dtype == object else np.unique(vals[~_isnan(vals)])
                parts.append(u); nulls.append(bool((vals == None).any()) if vals.dtype == object else bool(_isnan(vals).any()))
            else:
                raise NotImplementedError('union: column %s mode %s' % (col, c.get('mode')))
        cat = np.concatenate([np.asarray(p, dtype=(object if dt == 1 else None)) for p in parts]) if parts else np.zeros(0)
        if dt == 1:
            cat = np.array([v if isinstance(v, (bytes, bytearray)) else (v.encode() if isinstance(v, str) else v) for v in cat], dtype=object)
        merged = np.unique(cat)
        has_null = any(nulls)
        V = int(merged.size) + (1 if has_null else 0)
        remaps = []
        for s, p, hn in zip(segs, parts, nulls):
            c = s.cols[col]
            if c.get('mode') in (0, 1, 2):
                r = np.searchsorted(merged, np.asarray(p, dtype=(object if dt == 1 else None))).astype(np.int64)
                if c.get('has_null'): r = np.append(r, V - 1)             # the segment's null code -> the merged null code
                remaps.append(('codes', r))
            else:
                remaps.append(('values', None))                          # mode 4/5: remap by value at stream time
        self._remap[col] = (merged, remaps, has_null, V, dt)
        unit = None
        try: unit = segs[0].unit(col)
        except Exception: pass
        self._meta[col] = {'V': V, 'n_dict': V - (1 if has_null else 0), 'dt': dt, 'mode': 0 if dt != 1 else 1, 'has_null': 1 if has_null else 0, 'bits': max(1, int(np.ceil(np.log2(max(V, 2))))), 'aux': segs[0].cols[col].get('aux', 0), 'unit': unit, 'union': True}

    def unit(self, col):
        return self.segs[0].unit(col)

    # ---- the two primitives ----
    def _typed_dict(self, col):
        self._merge(col)
        merged, remaps, has_null, V, dt = self._remap[col]
        return merged                              # the null code (V-1) is implicit, as in a segment

    def codes(self, col):
        if col in self._codes: return self._codes[col]
        self._merge(col)
        merged, remaps, has_null, V, dt = self._remap[col]
        out = np.empty(self.N, np.int32 if V < (1 << 31) else np.int64); pos = 0     # the narrowest width: 40 MB per column at 10M rows
        for s, (kind, r) in zip(self.segs, remaps):
            n = int(s.N)
            if kind == 'codes':
                out[pos:pos + n] = r[np.asarray(s.codes(col), dtype=np.int64)]
            else:
                vals = np.asarray(s.values(col))
                if dt == 1:
                    vals = np.array([v if isinstance(v, (bytes, bytearray)) else (None if v is None else v.encode()) for v in vals], dtype=object)
                    nn = np.array([v is not None for v in vals], dtype=bool)
                else:
                    nn = ~_isnan(vals)
                seg_codes = np.full(n, V - 1 if has_null else 0, np.int64)
                seg_codes[nn] = np.searchsorted(merged, vals[nn])
                out[pos:pos + n] = seg_codes
            pos += n
        self._codes[col] = out
        return out

    _raw_codes = codes

    def codes_at(self, col, rows):
        return self.codes(col)[np.asarray(rows)]

    def values(self, col):
        td = self._typed_dict(col); c = self.cols[col]
        codes = self.codes(col)
        if c['has_null']:
            # the segment's convention: a nullable column reads as an object array with None
            tdn = np.append(np.asarray(td, dtype=object), None)
            return tdn[codes]
        return np.asarray(td)[codes]

    def values_at_rows(self, col, rows):
        return self.values(col)[np.asarray(rows)]

    def fetch(self, col, code):
        td = self._typed_dict(col); c = self.cols[col]
        if c['has_null'] and int(code) == c['V'] - 1: return None
        return td[int(code)]

    def dict_vals(self, col):
        """the dictionary values in code order (strings as an object array of bytes)"""
        td = self._typed_dict(col)
        return td if isinstance(td, np.ndarray) and td.dtype == object else np.asarray(td, dtype=object)

    def _dict_ints(self, col):
        self._merge(col); merged = self._remap[col][0]
        return np.asarray(merged, dtype=np.int64)

    def code_counts(self, col):
        return np.bincount(self.codes(col), minlength=self.cols[col]['V'])

    def presence_mask(self):
        masks = [s.presence_mask() for s in self.segs]
        if all(m is None for m in masks): return None
        return np.concatenate([m if m is not None else np.ones(int(s.N), bool) for m, s in zip(masks, self.segs)])

    # ---- things a union cannot do: decline by name ----
    def e8_planes(self, col): return None
    def cluster_meta(self): return None
    def stairs(self, col): return None
    def _override_vals_typed(self, col): return []              # no overrides on a union (the presence gate keeps dirty segments away)
    def _overrides(self, col): return None

    def _raw_codes_range(self, col, lo, hi):
        return self.codes(col)[int(lo):int(hi)]

    def values_range(self, col, lo, hi):
        return self.values(col)[int(lo):int(hi)]
    def _seq_decode(self, c): raise NotImplementedError('union: no sequence codec')
    def add_const_column(self, *a, **k): raise NotImplementedError('union: read-only')

    def drop_derived(self):
        # the remapped code streams derive from IMMUTABLE segments: they persist across queries
        # (set ops rebuilt them per leaf: 176x); only a catalog refresh discards the union itself.
        # The per-query caches of the underlying segments are dropped as usual.
        for s in self.segs:
            try: s.drop_derived()
            except Exception: pass

    def __getattr__(self, name):
        if name.startswith('_'):
            # a PRIVATE probe (getattr(seg, '_cache', None) / hasattr): AttributeError, so the
            # caller creates its cache on the union like on a segment
            raise AttributeError(name)
        raise NotImplementedError('union view: %s is not served across segments' % name)


class _LazyCols(dict):
    """column metadata that merges on first access"""
    def __init__(self, u): super().__init__(); self._u = u
    def __contains__(self, k): return k in self._u.order
    def __getitem__(self, k):
        if k not in self._u.order: raise KeyError(k)
        self._u._merge(k); return self._u._meta[k]
    def get(self, k, default=None):
        try: return self[k]
        except KeyError: return default
    def keys(self): return list(self._u.order)
    def items(self): return [(k, self[k]) for k in self._u.order]
    def __iter__(self): return iter(self._u.order)
    def __len__(self): return len(self._u.order)


def _isnan(a):
    a = np.asarray(a)
    return np.isnan(a) if a.dtype.kind == 'f' else np.zeros(a.shape, bool)
