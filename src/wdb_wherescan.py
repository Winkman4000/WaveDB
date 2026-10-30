"""wherescan: the disk-first WHERE read -- predicate scan over blocked (enc=3) code frames.

The shape this serves is the ClickBench trench: a conjunction of selective predicates feeding a
GROUP BY or aggregate, e.g. WHERE CounterID = 62 AND EventDate BETWEEN .. AND IsRefresh = 0.
Today those queries full-decode every involved column (0.6-1.9 s each; Q39 times out). Here:

  1. STAIR predicates (range/eq on a staircase column) become exact row spans via searchsorted
     over the step rows -- no bytes touched.
  2. The first dict-equality predicate becomes a PARALLEL FRAME SCAN: decompress each enc=3 frame
     inside the span (thread pool, per-thread zstd via the thread-local property), compare, keep
     positions. Measured primitive: 100M rows in ~95 ms.
  3. Remaining dict-eq predicates check via codes_at (touched frames only); mode-4 flag
     predicates check in place (bitpacked values, cached seq decode).
  4. Survivors group/aggregate: keys gathered by codes_at, composite built factorize-then-combine
     (overflow-safe), counts by bincount, canonical count-desc order, OFFSET/LIMIT, winners
     decoded one value each. Supports the CASE WHEN <eq-conj> THEN col ELSE '' derived key (Q39).

Zero columns enter the codes cache; the pipeline's residency is its survivors. Declines: joins,
distinct, having, deleted rows, overrides on involved columns, non-conjunctive WHERE, LIKE/regex.
Prototype of this exact pipeline ran Q39 (a 45 s timeout) in ~1 s steady-state, exact vs DuckDB.
"""
import os
import numpy as np
import sqlglot.expressions as E
from concurrent.futures import ThreadPoolExecutor
import wdb_sql
import wdb_policies as P
import wdb_gdsidecar
import wdb_scalar
import os

_ENABLED = True
_HITS = 0
_SCAN_THREADS = 14
_FLAG_LANES = [os.environ.get('WDB_FLAG_LANES', '1') == '1']   # A/B: 0 restores the thread-pool frame scan (enc 3)
_SCAN_CODES_MAX = 4_000_000    # the scanned codes are kept for codes_at up to this many hits
_SLOT_DIV = [4]                # the lane scan's pass 1 keeps up to BR // 4 hits per frame; a frame with more is
                               # inflated again in pass 2 (measured, server: a quarter beat 1/16 and 1/64 on the
                               # dense LIKEs and tied on URL; a whole frame's slot was slower everywhere)


def enable():
    global _ENABLED; _ENABLED = True


def disable():
    global _ENABLED; _ENABLED = False


# ---------------------------------------------------------------- literal -> code
def _lit_bytes(v):
    return v if isinstance(v, (bytes, bytearray)) else str(v).encode()


def _litval(node):
    """Python value of a Literal, unwrapping CAST('..' AS DATE/TIMESTAMP) and unary minus."""
    n = node
    neg = False
    while isinstance(n, (E.Cast, E.Neg, E.Paren)):
        if isinstance(n, E.Neg):
            neg = not neg
        n = n.this
    if isinstance(n, E.Literal):
        v = wdb_sql._literal_value(n)
        return -v if neg and isinstance(v, (int, float)) else v
    return None


def _time_units(seg, col, val):
    """A literal for a time/date column (dt 3) as an int in the column's stored unit, or None when no stored
    value can equal it. A numeric literal is native units (wdb_sql._lit_for_col's convention); text, bytes,
    dates and datetimes are parsed. Raises ValueError on text that is not a time."""
    import datetime as _dtm
    unit = seg.unit(col)
    if isinstance(val, (bool, np.bool_)):
        raise ValueError('boolean literal for a time column')
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return int(val) if float(val).is_integer() else None
    if isinstance(val, bytes):
        val = val.decode('utf-8', 'surrogatepass')
    if isinstance(val, str):
        t = np.datetime64(val.strip().replace(' ', 'T'))
    elif isinstance(val, (np.datetime64, _dtm.datetime, _dtm.date)):
        t = np.datetime64(val)
    else:
        raise ValueError('unsupported literal for a time column: %r' % (val,))
    iv = int(t.astype('datetime64[%s]' % unit).view('int64'))
    return iv if np.datetime64(iv, unit) == t else None


def _code_of(seg, col, val):
    """Dict code of a literal, or None when absent. Dicts are value-sorted: mode 2 answers by
    searchsorted over the int dict; string dicts by fetch-bisect (O(log V) point fetches, each
    sub-ms against chunked dicts -- never materializes the dictionary)."""
    c = seg.cols[col]
    V = int(c['V']) - (1 if c['has_null'] else 0)
    if V <= 0:
        return None
    if c['dt'] == 3:
        # A TIME/DATE LITERAL IN THE COLUMN'S STORED UNITS (2026-10-17, the Q14 board): EventTime <> '2013-07-21
        # 19:37:51' reached the int lookup as text, int() failed, and None read as "absent" -- '<>' then kept
        # every row (a wrong answer) and the composed reads declined. The text is parsed to the column's unit;
        # an unparseable literal raises (never guessed absent); a value finer than the unit is absent.
        val = _time_units(seg, col, val)
        if val is None:
            return None
        if c['mode'] != 2:
            d = np.asarray(seg._typed_dict(col))[:V]
            k = int(np.searchsorted(d, val))
            return k if k < V and int(d[k]) == val else None
    if c['mode'] == 2:
        try:
            iv = int(val)
        except (TypeError, ValueError):
            return None
        if c.get('i2ch') is not None and c.get('intvals') is None:
            lo2, hi2 = 0, V - 1              # chunked spine: fetch-bisect pops ~6
            while lo2 <= hi2:                # chunks (cached) instead of inflating
                mid = (lo2 + hi2) // 2       # the 102MB monolith per lookup
                fv = int(seg._dict_ints_at(c, np.array([mid], np.int64))[0])
                if fv == iv:
                    return mid
                if fv < iv:
                    lo2 = mid + 1
                else:
                    hi2 = mid - 1
            return None
        arr = seg._dict_ints(c)
        k = int(np.searchsorted(arr[:V], iv))
        return k if k < V and int(arr[k]) == iv else None
    if c['dt'] == 0:
        try:
            iv = int(val)
        except (TypeError, ValueError):
            return None
        lo, hi = 0, V - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            fv = int(seg.fetch(col, mid))
            if fv == iv: return mid
            if fv < iv: lo = mid + 1
            else: hi = mid - 1
        return None
    tgt = _lit_bytes(val)
    lo, hi = 0, V - 1
    if c.get('mode') == 1 and c.get('chunked') and c.get('CHUNK') and int(c.get('nch', 0)) > 1 \
            and seg._overrides(col) is None and col not in getattr(seg, '_tdict', {}):
        # THE CHUNK FIRST, BY ITS HEAD (2026-10-03): the last chunk whose first value is <= the literal holds it,
        # if anything does; the heads are read from the chunks' first bytes, then one chunk is inflated for the
        # bisect inside it (the plain bisect inflated ~3-6 chunks per literal: movie_info.info, 8 countries, ~26)
        CH = int(c['CHUNK']); a, b = 0, min(int(c['nch']), (V + CH - 1) // CH) - 1
        if seg.fc_head(c, 0) > tgt:
            return None
        while a < b:
            m = (a + b + 1) // 2
            h = seg.fc_head(c, m)
            if h == tgt:
                return m * CH
            if h < tgt: a = m
            else: b = m - 1
        lo, hi = a * CH, min(V, (a + 1) * CH) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        fv = seg.fetch(col, mid)
        fv = fv if isinstance(fv, (bytes, bytearray)) else _lit_bytes(fv)
        if fv == tgt: return mid
        if fv < tgt: lo = mid + 1
        else: hi = mid - 1
    return None


# ---------------------------------------------------------------- WHERE parsing
def _conjuncts(node):
    if isinstance(node, E.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    if isinstance(node, E.Paren):
        return _conjuncts(node.this)
    return [node]


def _col_lit(node):
    """(colname, literal, op) for col <op> literal, either side. op in ('=','<>','>=','<=')."""
    ops = {E.EQ: '=', E.NEQ: '<>', E.GTE: '>=', E.LTE: '<='}
    op = ops.get(type(node))
    if op is None:
        return None
    a, b = node.this, node.expression
    def _is_lit(x):                                  # a Cast/Neg only counts when it WRAPS a literal:
        n = x                                        # CAST(ts AS DATE) = d is a column expression, not a literal
        while isinstance(n, (E.Cast, E.Neg, E.Paren)): n = n.this
        return isinstance(n, E.Literal)
    if isinstance(a, E.Column) and _is_lit(b):
        return a.name, _litval(b), op
    if isinstance(b, E.Column) and _is_lit(a):
        flip = {'>=': '<=', '<=': '>='}
        return b.name, _litval(a), flip.get(op, op)
    return None


def _in_list(node):
    """(colname, [literals], negate) for `col [NOT] IN (lit, ...)`."""
    neg = False
    if isinstance(node, E.Not):
        node = node.this; neg = True
    if not isinstance(node, E.In) or not isinstance(node.this, E.Column):
        return None
    if node.args.get('_codes') is not None:       # same-column subquery, pre-resolved to
        return node.this.name, {'_codes': node.args['_codes']}, neg   # a code set upstream
    if node.args.get('query') is not None:
        return None                               # unresolved subquery: not a literal list
    vals = []
    for x in node.expressions:
        v = _litval(x)
        if v is None:
            return None
        vals.append(v)
    return node.this.name, vals, neg

def _case_key(p, seg, col_map):
    """The Q39 pattern: CASE WHEN <conj of col = int-literal> THEN <column> ELSE <literal>."""
    inner = p.this if isinstance(p, E.Alias) else p
    if not isinstance(inner, E.Case) or len(inner.args.get('ifs', [])) != 1:
        return None
    iff = inner.args['ifs'][0]
    default = inner.args.get('default')
    if default is not None and not isinstance(default, E.Literal):
        return None
    then = iff.args.get('true')
    if not isinstance(then, E.Column):
        return None
    conds = []
    for cn in _conjuncts(iff.this):
        cl = _col_lit(cn)
        if cl is None or cl[2] != '=':
            return None
        col = col_map.get(cl[0], cl[0]) if col_map else cl[0]
        if not P.columns_exist(seg, col):
            return None
        if seg.cols[col].get('mode') not in (0, 1, 2, 4):
            return None
        conds.append((col, cl[1]))       # literal resolved at execute (code for dicts, int for mode 4)
    src = col_map.get(then.name, then.name) if col_map else then.name
    if not P.columns_exist(seg, src):
        return None
    dflt = _litval(default) if default is not None else ''
    return {'kind': 'case', 'conds': conds, 'src': src, 'default': dflt}


def _count_distinct(p):
    """colname for COUNT(DISTINCT col)."""
    inner = p.this if isinstance(p, E.Alias) else p
    if isinstance(inner, E.Count) and isinstance(inner.this, E.Distinct):
        exprs = inner.this.expressions
        if len(exprs) == 1 and isinstance(exprs[0], E.Column):
            return exprs[0].name
    return None


_SCMP = {'GT': '>', 'GTE': '>=', 'LT': '<', 'LTE': '<=', 'EQ': '=', 'NEQ': '<>'}


def _scalar_cmp(seg, node, col_map):
    """(spec, op, lit) for <scalar(col)> <cmp> <literal> -- evaluated as a flag table over the
    scalar's code->value table: the comparison happens once per DISTINCT value."""
    import wdb_scalar
    tn = type(node).__name__
    if tn not in _SCMP:
        return None
    a, b = node.this, node.expression
    lit = None
    if isinstance(b, E.Literal):
        expr, lit = a, b
    elif isinstance(a, E.Literal):
        expr, lit = b, a
        tn = {'GT': 'LT', 'GTE': 'LTE', 'LT': 'GT', 'LTE': 'GTE'}.get(tn, tn)
    else:
        return None
    sp = wdb_scalar.parse(seg, expr, col_map)
    if sp is None:
        return None
    v = lit.this
    if getattr(lit, 'is_string', False):
        v = str(v)                              # a STRING literal stays a string: '01' is not 1
    else:
        try:
            v = int(str(v))
        except Exception:
            try: v = float(str(v))
            except Exception: v = str(v)
    return sp, _SCMP[tn], v


def _scalar_flag(seg, sp, op, lit):
    """flag[code] = table[code] <op> lit, memoized via the underlying scalar table."""
    import wdb_scalar
    t = wdb_scalar.table(seg, sp)
    if op == '>':   return t > lit
    if op == '>=':  return t >= lit
    if op == '<':   return t < lit
    if op == '<=':  return t <= lit
    if op == '=':   return t == lit
    return t != lit


def _like(node):
    """(col, needle, kind, negate) for `col [NOT] LIKE pat` -- kinds: contains '%x%',
    prefix 'x%', suffix '%x' (no interior wildcards), general (any other pattern), all ('%' alone)."""
    neg = False
    n = node
    if isinstance(n, E.Not):
        n = n.this; neg = True
    if not isinstance(n, E.Like) or not isinstance(n.this, E.Column):
        return None
    neg = neg or bool(n.args.get('negate'))      # sqlglot: NOT LIKE == Like(negate=True)
    pat = _litval(n.expression)
    if isinstance(pat, str) and pat and pat.strip('%') == '':
        # '%' (or '%%...'): every non-NULL value matches -- kind 'all' (2026-10-19, the LIKE board: it was refused
        # here as too short, and Q21/Q23 with K LIKE '%' fell to the general path, past 30 s)
        return n.this.name, '%', 'all', neg
    if not isinstance(pat, str) or len(pat) < 2:
        return None
    if pat.startswith('%') and pat.endswith('%'):
        needle, kind = pat[1:-1], 'contains'
    elif pat.endswith('%'):
        needle, kind = pat[:-1], 'prefix'
    elif pat.startswith('%'):
        needle, kind = pat[1:], 'suffix'
    else:
        needle, kind = pat, 'general'
    if kind != 'general' and (not needle or '%' in needle or '_' in needle):
        needle, kind = pat, 'general'    # interior wildcards: the general regex path
    if not needle:
        return None
    return n.this.name, needle, kind, neg


from numba import njit, prange


@njit(nogil=True, cache=True)
def _flag_hits_kernel(raw, bits, s0, e0, base, hits, codes):
    """positions (base + i) and codes for i in [s0, e0) whose code's bit is set in the packed flag
    (little-endian bit order) -- one frame, no GIL"""
    n = 0
    for i in range(s0, e0):
        v = raw[i]
        if (bits[v >> 3] >> (v & 7)) & 1:
            hits[n] = base + (i - s0); codes[n] = v; n += 1
    return n


@njit(nogil=True, parallel=True, cache=True)
def _find_all_kernel(buf, nd):
    """all start positions of nd in buf -- two passes over parallel chunks (count, then fill into
    exact offsets): no big zeroed buffer per call"""
    n = buf.size; m = nd.size
    if m == 0 or n < m: return np.zeros(0, np.int64)
    nchunks = 64
    step = (n + nchunks - 1) // nchunks
    counts = np.zeros(nchunks, np.int64)
    first = nd[0]
    for c in prange(nchunks):
        lo = c * step; hi = min(n - m + 1, (c + 1) * step)
        k = 0; i = lo
        while i < hi:
            if buf[i] == first:
                ok = True
                for j in range(1, m):
                    if buf[i + j] != nd[j]:
                        ok = False; break
                if ok: k += 1
            i += 1
        counts[c] = k
    starts = np.zeros(nchunks + 1, np.int64)
    for c in range(nchunks): starts[c + 1] = starts[c] + counts[c]
    out = np.empty(starts[nchunks], np.int64)
    for c in prange(nchunks):
        lo = c * step; hi = min(n - m + 1, (c + 1) * step)
        p = starts[c]; i = lo
        while i < hi:
            if buf[i] == first:
                ok = True
                for j in range(1, m):
                    if buf[i + j] != nd[j]:
                        ok = False; break
                if ok:
                    out[p] = i; p += 1
            i += 1
    return out


class FramedText:
    """THE FRAMED TEXT: the joined dictionary bytes as zstd frames split at VALUE boundaries
    (URLs compress ~8x: 3.4 GB of text is ~400 MB of I/O), exposing exactly what the LIKE scan
    needs -- find(needle, start) and slicing -- decompressing one frame at a time. A needle never
    spans two values, so it never spans two frames. The last frame is cached."""
    def __init__(self, path, index):
        self.path = path; self.hay_start = np.asarray(index[:, 0], dtype=np.int64); self.file_start = np.asarray(index[:, 1], dtype=np.int64)
        self.total = int(index[-1, 0]); self.nf = int(index.shape[0]) - 1
        self.fh = open(path, 'rb'); self._last = (-1, None)
        import zstandard; self._dz = zstandard.ZstdDecompressor()
    def __len__(self): return self.total
    def _frame_bytes(self, i):
        """decompress frame i without touching the shared cache (thread-safe)"""
        with open(self.path, 'rb') as fh:
            fh.seek(int(self.file_start[i])); raw = fh.read(int(self.file_start[i + 1] - self.file_start[i]))
        import zstandard
        return zstandard.ZstdDecompressor().decompress(raw, max_output_size=int(self.hay_start[i + 1] - self.hay_start[i]))
    def _frame(self, i):
        if self._last[0] == i: return self._last[1]
        self.fh.seek(int(self.file_start[i])); raw = self.fh.read(int(self.file_start[i + 1] - self.file_start[i]))
        b = self._dz.decompress(raw, max_output_size=int(self.hay_start[i + 1] - self.hay_start[i]))
        self._last = (i, b); return b
    def find(self, sub, start=0):
        if start >= self.total: return -1
        i = int(np.searchsorted(self.hay_start, start, side='right')) - 1
        while i < self.nf:
            b = self._frame(i); off = int(self.hay_start[i])
            p = b.find(sub, max(0, start - off))
            if p >= 0: return off + p
            i += 1
        return -1
    def __getitem__(self, sl):
        a, b = sl.start or 0, (sl.stop if sl.stop is not None else self.total)
        i = int(np.searchsorted(self.hay_start, a, side='right')) - 1
        fb = self._frame(i); off = int(self.hay_start[i])
        return fb[a - off:b - off]
    def find_all_parallel(self, sub, threads=8):
        """every occurrence position of sub in the whole text (frames in parallel)"""
        nd = np.frombuffer(sub, dtype=np.uint8)
        def one(i):
            self.fh_local = getattr(self, 'fh_local', None)
            with open(self.path, 'rb') as fh:
                fh.seek(int(self.file_start[i])); raw = fh.read(int(self.file_start[i + 1] - self.file_start[i]))
            import zstandard
            b = zstandard.ZstdDecompressor().decompress(raw, max_output_size=int(self.hay_start[i + 1] - self.hay_start[i]))
            pos = _find_all_kernel(np.frombuffer(b, dtype=np.uint8), nd)
            return pos + int(self.hay_start[i])
        with ThreadPoolExecutor(max_workers=threads) as ex:
            parts = list(ex.map(one, range(self.nf)))
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)


def _framed_write(path, hay, offs, target=32 << 20):
    """write hay as zstd frames split at value boundaries; returns the (F+1, 2) index"""
    import zstandard
    cz = zstandard.ZstdCompressor(level=3, threads=4)      # a birth happens once, at warm: fast beats small here
    idx = []; fpos = 0; i = 0; V = int(offs.size) - 1
    with open(path + '.partial', 'wb') as f:
        while i < V:
            j = int(np.searchsorted(offs, int(offs[i]) + target, side='left'))
            j = max(j, i + 1); j = min(j, V)
            chunk = hay[int(offs[i]):int(offs[j])]
            z = cz.compress(chunk); idx.append((int(offs[i]), fpos)); f.write(z); fpos += len(z); i = j
        idx.append((int(offs[V]), fpos)); f.flush(); os.fsync(f.fileno())
    os.replace(path + '.partial', path)
    return np.asarray(idx, dtype=np.int64)


@njit(nogil=True, parallel=True, cache=True)
def _char_count_kernel(buf, starts, out, base):
    """UTF-8 character count per value: bytes that are not continuation bytes (10xxxxxx)"""
    V = starts.size - 1
    for v in prange(V):
        a = starts[v] - base; b = starts[v + 1] - base
        n = 0
        for i in range(a, b):
            if (buf[i] & 0xC0) != 0x80: n += 1
        out[v] = n


def char_lengths(seg, col):
    """THE LENGTH SIDECAR: length(col) over a string dictionary as a V-scale int32 array -- born
    from the framed text with one parallel pass (UTF-8 characters, not bytes), never a Python
    string. AVG(length(URL)) had front-decoded 6M URLs into objects: 12s cold for a number that
    is a difference of offsets."""
    base = getattr(seg, 'path', None)
    if not base: return None
    lp = base + '.' + col + '.clen.npy'
    import wdb_sidecar
    dbdir = os.path.dirname(base)
    if wdb_sidecar.exists(lp) and wdb_sidecar.is_fresh(dbdir, os.path.basename(lp)):
        return np.load(lp)
    # THE DICTIONARY'S OWN LENGTHS FIRST (2026-09-26): the load stores them (wdb_lens) and the
    # front-coded walk derives them without a string -- neither births the text buffer
    try:
        dl = seg.dict_charlens(col)
        if dl is not None:
            return np.asarray(dl)
    except Exception:
        pass
    hay, offs = _text_buffer(seg, col)            # births the framed text if needed
    V = int(offs.size) - 1
    out = np.zeros(V, np.int32)
    if isinstance(hay, FramedText):
        def one(i):
            b = np.frombuffer(hay._frame_bytes(i), dtype=np.uint8)
            lo = int(np.searchsorted(offs, int(hay.hay_start[i]), side='left')); hi = int(np.searchsorted(offs, int(hay.hay_start[i + 1]), side='left'))
            part = np.zeros(hi - lo, np.int32)
            _char_count_kernel(b, np.asarray(offs[lo:hi + 1], dtype=np.int64), part, int(hay.hay_start[i]))
            return lo, part
        with ThreadPoolExecutor(max_workers=8) as ex:
            for lo, part in ex.map(one, range(hay.nf)):
                out[lo:lo + part.size] = part
    else:
        _char_count_kernel(np.frombuffer(hay, dtype=np.uint8), np.asarray(offs, dtype=np.int64), out, 0)
    try:
        if wdb_sidecar.may_birth(dbdir, out.nbytes, 'length %s' % col):
            np.save(lp + '.partial.npy', out); os.replace(lp + '.partial.npy', lp)
    except Exception:
        pass
    return out


def _text_buffer(seg, col):
    """THE TEXT BUFFER SIDECAR: the dictionary's values joined into one byte string plus their
    offsets -- born once from _typed_dict (6M front-coded URLs decoded into Python objects:
    16 seconds, every process, never written down) and mmap'd thereafter. The LIKE family reads
    it without a single Python object; the hay bytes are memoised per process."""
    memo = seg.__dict__.setdefault('_ws_text_memo', {})
    if col in memo: return memo[col]
    base = getattr(seg, 'path', None)
    tp = (base + '.' + col + '.txz.bin') if base else None          # zstd frames split at value boundaries
    ip = (base + '.' + col + '.txi.npy') if base else None          # the frame index
    op = (base + '.' + col + '.txo.npy') if base else None          # value offsets
    hay = offs = None
    import wdb_sidecar as _wsc
    if tp and _wsc.exists(tp) and _wsc.exists(op) and _wsc.exists(ip):
        try:
            import wdb_sidecar
            dbdir = os.path.dirname(base)
            if all(wdb_sidecar.is_fresh(dbdir, os.path.basename(p)) for p in (tp, op, ip)):
                offs = np.load(op, mmap_mode='r'); hay = FramedText(tp, np.load(ip))
        except Exception:
            hay = offs = None
    if hay is None:
        vals = seg._typed_dict(col)
        V = len(vals)
        bs = [v if isinstance(v, (bytes, bytearray)) else (v.encode() if isinstance(v, str) else bytes(v)) for v in vals]
        lens = np.fromiter((len(v) for v in bs), np.int64, V)
        offs = np.zeros(V + 1, np.int64); np.cumsum(lens, out=offs[1:])
        hay = b''.join(bs)
        if tp and not str(base).startswith('union'):
            try:
                import wdb_sidecar
                if wdb_sidecar.may_birth(os.path.dirname(base), len(hay) // 4 + offs.nbytes, 'text-buffer %s' % col):
                    idx = _framed_write(tp, hay, offs)                                  # the rename law inside
                    np.save(ip + '.partial.npy', idx); os.replace(ip + '.partial.npy', ip)
                    np.save(op + '.partial.npy', offs); os.replace(op + '.partial.npy', op)
            except Exception:
                pass
    memo[col] = (hay, offs)
    return hay, offs


# the road is priced by wdb_strings.take_subset_road (chunks inflated + entries walked, against
# the whole pass); the last residual LIKE's road is kept for inspection:
# (col, needle, distinct codes, chunks touched, chunks, entries walked, bill, road)
_LIKE_AT_LAST = [None]


def _like_at(seg, col, needle, kind, cc):
    """Boolean[len(cc)]: the string at each code in cc matches the LIKE. THE FILTER BEFORE THE READ
    (Jackson): a LIKE that runs after other filters decides only the distinct codes still alive --
    18.3M URLs become the 2,450 behind Q22's survivors. Falls back to the whole dictionary when
    deciding the survivors would cost most of a whole pass (wdb_strings.take_subset_road), or when
    the LIKE is not a plain contains."""
    memo = seg.__dict__.setdefault('_ws_like_memo', {})
    if (col, needle, kind) in memo:              # already decided whole (the NULL bin padded)
        return _like_flags(seg, col, needle, kind)[cc]
    c0 = seg.cols.get(col) or {}
    nd = needle.encode() if isinstance(needle, str) else needle
    if kind == 'contains' and 'restarts' in c0 and len(nd) >= 1 and cc.size:
        import wdb_strings
        V = int(c0['V'])
        pres = np.zeros(V, np.bool_); pres[cc] = True
        u = np.flatnonzero(pres)
        sub, (bill, touched, nch, walked) = wdb_strings.take_subset_road(seg, col, u)
        if sub:
            got = wdb_strings.identify_contains_at(seg, col, u, nd)
            if got is not None:
                _LIKE_AT_LAST[0] = (col, nd, int(u.size), touched, nch, walked, round(bill, 3), 'survivors')
                flag = np.zeros(V, np.bool_); flag[u] = got
                return flag[cc]
        _LIKE_AT_LAST[0] = (col, nd, int(u.size), touched, nch, walked, round(bill, 3), 'whole dictionary')
    return _like_flags(seg, col, needle, kind)[cc]


def _like_flags(seg, col, needle, kind='contains'):
    """flag[code] for EVERY code of the column, the NULL bin included (False: NULL LIKE x is not
    true). The text-buffer roads size their flag to the dictionary without its NULL bin; a NULL
    row's code indexed one past it (found 2026-09-23, a residual LIKE over a nullable column)."""
    fl = _like_flags_dict(seg, col, needle, kind)
    V = int((seg.cols.get(col) or {}).get('V') or 0)
    if fl.size < V:
        fl = np.concatenate([np.asarray(fl, np.bool_), np.zeros(V - fl.size, np.bool_)])
    return fl


def _like_flags_dict(seg, col, needle, kind='contains'):
    """Boolean flag[code] = dict value contains needle. THE DICT IS THE HAYSTACK: the row data
    is never string-compared -- all distinct values are scanned once (C-speed buffer find with
    per-string skip), and the predicate collapses to a code-set membership test. Memoized for
    the process lifetime (transient; dies with the worker -- not a persisted structure)."""
    memo = seg.__dict__.setdefault('_ws_like_memo', {})
    mk = (col, needle, kind)
    if mk in memo:
        return memo[mk]
    nd = needle.encode() if isinstance(needle, str) else needle
    c0 = seg.cols.get(col) or {}
    if kind == 'all':
        # LIKE '%': every value of the dictionary matches (the NULL bin, past it, is padded False by _like_flags)
        V0 = int(c0.get('V') or 0) - (1 if c0.get('has_null') else 0)
        flag = np.ones(max(V0, 0), bool)
        memo[mk] = flag
        return flag
    if kind == 'contains' and 'restarts' in c0 and len(nd) >= 1:
        # THE FRONT-CODED HAYSTACK (vanilla law's find): the old path decoded 18M URLs into Python
        # objects to join a haystack -- 19 s, every process. The kernel walks the dictionary's own
        # front-coded bytes: no objects, blocks in parallel, nothing born.
        # THE THREE READS: identification over the dictionary AS STORED -- chunk by chunk as they
        # decompress, prefix carry inside each chain, no whole-dictionary blob (the join of 2 GB of
        # URL text was 1.4-2.3 s of every first LIKE).
        # ALWAYS, sidecars on or off (2026-09-26): this road was gated on the switch being off and
        # no text-buffer file existing, so the kit's warm step (switch on, text buffer born) sent
        # Q20-22 back to the Sept 13 text buffer. Same kit database, text buffer removed: Q20 cold
        # 2906 -> 854 ms, Q21 3349 -> 1223, Q22 3053 -> 2305.
        import wdb_strings
        keep = wdb_strings.identify_contains(seg, col, nd)
        if keep is not None:
            memo[mk] = keep
            return keep
    if kind == 'prefix' and c0.get('dt') == 1 and c0.get('mode') in (0, 1):
        # THE PREFIX NEEDS NO HAYSTACK (2026-10-05): a value-sorted dictionary holds a prefix as one code range,
        # two bisections by point fetch -- the road below built the whole text buffer first (18M URLs unpacked
        # into Python: 7.7 s of URL LIKE 'http://holodilnik%') and then never read it
        V0 = int(c0['V']) - (1 if c0.get('has_null') else 0)
        lo = _bound_code(seg, col, nd, 'left')
        up = nd.rstrip(b'\xff')
        hi = V0 if not up else _bound_code(seg, col, up[:-1] + bytes([up[-1] + 1]), 'left')
        if lo is not None and hi is not None:
            flag = np.zeros(V0, bool); flag[lo:hi] = True
            memo[mk] = flag
            return flag
    import wdb_scalar
    st9 = wdb_scalar._stream(seg, col)
    if st9 is not None:
        # THE LIKE ON THE STREAM (2026-10-05): the dictionary's compiled byte stream, one parallel pass -- the
        # token kernel without '_', the general kernel with it. The text buffer below unpacked every value into
        # Python and ran a regex per value ('%карт%мир%' over 6M phrases: 6.8 s; '_оскв%' over 9.4M titles: 11.9 s)
        ns9 = needle if isinstance(needle, str) else needle.decode('utf-8', 'replace')
        pat9 = {'contains': '%' + ns9 + '%', 'prefix': ns9 + '%', 'suffix': '%' + ns9}.get(kind, ns9)
        fl9 = wdb_sql._like_stream(st9[0], st9[1], pat9)
        if fl9 is not None:
            memo[mk] = fl9
            return fl9
    hay, offs = _text_buffer(seg, col)
    V = int(offs.size) - 1
    flag = np.zeros(V, bool)
    if kind == 'general':
        # arbitrary wildcards, evaluated once per DISTINCT value (the dict-level law).
        # '%' spans bytes safely in UTF-8 (literal segments byte-match exactly), but '_'
        # means one CHARACTER -- multi-byte text forces the decoded path.
        import re
        if b'_' in nd:
            pat = needle if isinstance(needle, str) else needle.decode('utf-8', 'replace')
            rx = re.compile('^' + re.escape(pat).replace('\\%', '.*').replace('%', '.*')
                            .replace('_', '.') + '$', re.DOTALL)
            pos0 = 0
            for i in range(V):
                if rx.match(hay[pos0:int(offs[i + 1])].decode('utf-8', 'replace')):
                    flag[i] = True
                pos0 = int(offs[i + 1])
        else:
            rx = re.compile(b'^' + re.escape(nd).replace(b'\\%', b'.*').replace(b'%', b'.*')
                            + b'$', re.DOTALL)
            pos0 = 0
            for i in range(V):
                if rx.match(hay[pos0:int(offs[i + 1])]):
                    flag[i] = True
                pos0 = int(offs[i + 1])
        memo[mk] = flag
        return flag
    if kind == 'prefix':
        # value-sorted dict: a prefix is a contiguous code range -- two bisects, no scan
        lo = _bound_code(seg, col, nd, 'left')
        up = nd.rstrip(b'\xff')
        hi = V if not up else _bound_code(seg, col, up[:-1] + bytes([up[-1] + 1]), 'left')
        if lo is not None and hi is not None:
            flag[lo:hi] = True
    elif kind == 'suffix':
        # end-anchored: every occurrence checked (no per-string skip -- a mid-string hit
        # must not shadow a real end hit)
        pos = hay.find(nd)
        while pos >= 0:
            i = int(np.searchsorted(offs, pos, side='right')) - 1
            if pos + len(nd) == offs[i + 1]:
                flag[i] = True
                pos = hay.find(nd, int(offs[i + 1]))
            else:
                pos = hay.find(nd, pos + 1)
    elif isinstance(hay, FramedText) and len(nd) >= 2:
        # THE PARALLEL NEEDLE: frames decompress in threads (zstd releases the GIL) and a numba
        # kernel finds every occurrence across cores; a hit maps to its value by offsets. Python's
        # single-threaded find took 1.9s over 3.4 GB of URL text; this is memory-bound across 16 cores.
        hits = hay.find_all_parallel(nd)
        if hits.size:
            codes = np.searchsorted(offs, hits, side='right') - 1
            flag[np.unique(codes)] = True
    else:
        pos = hay.find(nd)
        while pos >= 0:
            i = int(np.searchsorted(offs, pos, side='right')) - 1
            if pos + len(nd) <= offs[i + 1]:
                flag[i] = True
            pos = hay.find(nd, int(offs[i + 1]))     # skip the rest of this string either way
    memo[mk] = flag
    return flag


def _disjuncts(node):
    if isinstance(node, E.Or):
        return _disjuncts(node.this) + _disjuncts(node.expression)
    if isinstance(node, E.Paren):
        return _disjuncts(node.this)
    return [node]


def _atom(seg, node, col_map):
    """Parse one disjunct into a row-evaluable atom, or None.
    Atoms: ('eq', col, val, neg) | ('in', col, vals, neg) | ('like', col, needle, kind, neg)
         | ('null', col, want) | ('flag', col, int, op) | ('range', col, val, op)"""
    isn = node
    want_null = None
    if isinstance(isn, E.Not) and isinstance(isn.this, E.Is):
        isn = isn.this; want_null = False
    elif isinstance(isn, E.Is):
        want_null = True
    if want_null is not None:
        if not isinstance(isn.this, E.Column) or not isinstance(isn.expression, E.Null):
            return None
        col = col_map.get(isn.this.name, isn.this.name) if col_map else isn.this.name
        if not P.columns_exist(seg, col) or seg._effective(col) is not None:
            return None
        if seg.cols[col].get('mode') not in (0, 1, 2, 4):
            return None
        return ('null', col, want_null)
    cl = _col_lit(node)
    if cl is not None:
        col = col_map.get(cl[0], cl[0]) if col_map else cl[0]
        if not P.columns_exist(seg, col) or seg._effective(col) is not None:
            return None
        c = seg.cols[col]
        if cl[2] in ('=', '<>'):
            if c.get('mode') == 4:
                return ('flag', col, int(cl[1]), cl[2])
            if c.get('mode') in (0, 1, 2):
                return ('eq', col, cl[1], cl[2] == '<>')
            return None
        if cl[2] in ('>=', '<=') and seg.stairs(col) is not None:
            return ('range', col, cl[1], cl[2])
        return None
    il = _in_list(node)
    if il is not None:
        col = col_map.get(il[0], il[0]) if col_map else il[0]
        if not P.columns_exist(seg, col) or seg._effective(col) is not None:
            return None
        if seg.cols[col].get('mode') not in (0, 1, 2, 4):
            return None
        return ('in', col, il[1], il[2])
    lk = _like(node)
    if lk is not None:
        col = col_map.get(lk[0], lk[0]) if col_map else lk[0]
        if not P.columns_exist(seg, col) or seg._effective(col) is not None:
            return None
        if seg.cols[col].get('mode') not in (0, 1):
            return None
        return ('like', col, lk[1], lk[2], lk[3])
    return None


def detect(seg, tree, col_map):
    # enc-5 (patched buckets) columns: this path's direct block machinery predates the
    # species; decline so the _raw_codes route (exact, 8-lane) serves until v2 learns nibbles
    for _c5 in tree.find_all(E.Column):
        _n5 = (col_map or {}).get(_c5.name, _c5.name) if col_map else _c5.name
        if _n5 in seg.cols and seg.cols[_n5].get('code_enc') in (5, 6):
            return None
    if not _ENABLED:                    return None
    if not P.no_joins(tree):            return None
    if not P.no_select_distinct(tree):  return None
    having = tree.args.get('having')
    hterms = []
    if having is not None:
        for hc in _conjuncts(having.this):
            tn = type(hc).__name__
            if tn not in _SCMP:
                return None
            hexpr, hlit = hc.this, hc.expression
            if isinstance(hexpr, E.Literal):
                hexpr, hlit = hlit, hexpr
                tn = {'GT': 'LT', 'GTE': 'LTE', 'LT': 'GT', 'LTE': 'GTE'}.get(tn, tn)
            if not isinstance(hlit, E.Literal):
                return None
            hk = wdb_sql._agg_kind(hexpr)
            if hk is None or hk[0] not in ('COUNT_STAR', 'SUM', 'AVG'):
                return None
            hcol = None
            if hk[0] != 'COUNT_STAR':
                hcol = col_map.get(hk[1], hk[1]) if col_map else hk[1]
                if not P.columns_exist(seg, hcol) or seg.cols[hcol].get('dt') != 0:
                    return None
            try:
                hval = float(str(hlit.this))
            except Exception:
                return None
            hterms.append((hk[0], hcol, _SCMP[tn], hval))
    if not P.no_deleted_rows(seg):      return None
    where = tree.args.get('where')
    if where is None:
        return None
    spans, eqs, flags, ins, likes, nulls, sflags = [], [], [], [], [], [], []
    conjs = []
    for cn in _conjuncts(where.this):
        if isinstance(cn, E.Between):            # sugar: col >= low AND col <= high
            if not isinstance(cn.this, E.Column):
                return None
            conjs.append(E.GTE(this=cn.this.copy(), expression=cn.args['low']))
            conjs.append(E.LTE(this=cn.this.copy(), expression=cn.args['high']))
        else:
            conjs.append(cn)
    ors = []
    for cn in conjs:
        node = cn.this if isinstance(cn, E.Paren) else cn
        if isinstance(node, E.Or):
            atoms = []
            for dj in _disjuncts(node):
                a = _atom(seg, dj, col_map)
                if a is None:
                    return None          # every disjunct must be in vocabulary
                atoms.append(a)
            ors.append(atoms)
            continue
        isn = cn
        want_null = None
        if isinstance(isn, E.Not) and isinstance(isn.this, E.Is):
            isn = isn.this; want_null = False
        elif isinstance(isn, E.Is):
            want_null = True
        if want_null is not None:
            if not isinstance(isn.this, E.Column) or not isinstance(isn.expression, E.Null):
                return None
            col = col_map.get(isn.this.name, isn.this.name) if col_map else isn.this.name
            if not P.columns_exist(seg, col):   return None
            if seg._effective(col) is not None: return None
            if seg.cols[col].get('mode') not in (0, 1, 2, 4): return None
            nulls.append((col, want_null))
            continue
        cl = _col_lit(cn)
        if cl is None:
            il = _in_list(cn)
            if il is not None:
                col = col_map.get(il[0], il[0]) if col_map else il[0]
                if not P.columns_exist(seg, col):   return None
                if seg._effective(col) is not None: return None
                if seg.cols[col].get('mode') not in (0, 1, 2, 4): return None
                ins.append((col, il[1], il[2]))
                continue
            lk = _like(cn)
            if lk is None:
                sc = _scalar_cmp(seg, cn, col_map)
                if sc is None:
                    return None
                sflags.append(sc)
                continue
            col = col_map.get(lk[0], lk[0]) if col_map else lk[0]
            if not P.columns_exist(seg, col):   return None
            if seg._effective(col) is not None: return None
            if seg.cols[col].get('mode') not in (0, 1):
                return None              # substring haystack needs a string dict
            likes.append((col, lk[1], lk[2], lk[3]))
            continue
        col = col_map.get(cl[0], cl[0]) if col_map else cl[0]
        if not P.columns_exist(seg, col):   return None
        if seg._effective(col) is not None: return None
        c = seg.cols[col]
        if c.get('mode') == 4 and cl[2] in ('=', '<>'):
            flags.append((col, int(cl[1]), cl[2]))
        elif seg.stairs(col) is not None and cl[2] in ('>=', '<=', '='):
            spans.append((col, cl[1], cl[2]))
        elif c.get('mode') in (0, 1, 2) and cl[2] in ('=', '<>'):
            eqs.append((col, cl[1], cl[2]))
        else:
            return None
    if not eqs and not spans and not likes and not ors and not sflags and not ins:
        return None                      # unselective flag-only shapes stay with the scan family
    def _eq_cost(col, val):
        # the driver decides the whole query's scale: RegionID=229 drove cte-chain at
        # 18.3M survivors when CounterID=62 (738K) sat right next to it. Exact per-code
        # counts when a .gbc sidecar ALREADY EXISTS (peek only -- driver selection must
        # never trigger a lazy build); N/V average otherwise.
        import os
        try:
            import wdb_gbshelf
            sh9 = wdb_gbshelf.open_shelf(seg, col)   # the tiered shelf prices in
            if sh9 is not None:                      # microseconds; the pickle
                k = _code_of(seg, col, val)          # below paid 65ms per plan
                if k is None:
                    return 0
                return int(wdb_gbshelf.point(sh9, int(k)))
        except Exception:
            pass
        if os.path.exists(seg.path + '.' + col + '.gbc'):
            try:
                import wdb_gbcount
                loaded = wdb_gbcount._load(seg, col)
                if loaded is not None:
                    k = _code_of(seg, col, val)
                    if k is None:
                        return 0                    # absent literal: empty drive, best possible
                    hc, hn = loaded
                    j = np.flatnonzero(hc == int(k))
                    return int(hn[j[0]]) if j.size else 1
            except Exception:
                pass
        V = int(seg.cols[col].get('V') or 1)
        return max(1, int(seg.N) // max(1, V))
    _cand = [i for i, e in enumerate(eqs) if e[2] == '=']
    # one candidate is its own driver: pricing it read the whole .gbc count file for nothing (Q19 cold:
    # 274 ms of a 577 ms query, 2026-09-27) -- min() calls the key even for a single item
    drive_eq = (_cand[0] if len(_cand) == 1 else
                min(_cand, key=lambda i: _eq_cost(eqs[i][0], eqs[i][1]))) if _cand else None
    # a like is only THE driver when no eq drives -- execute skips lflags[drive_like],
    # so marking one while an eq drives silently drops that LIKE as a filter
    drive_like = None if drive_eq is not None else \
        next((i for i, l in enumerate(likes) if not l[3]), None)
    drive_or = None
    if drive_eq is None and drive_like is None:
        # an OR group of positive membership atoms on dict columns can drive: one union flag
        # per column, one frame scan each, positions unioned
        for oi, atoms in enumerate(ors):
            if all(((a[0] == 'eq' and not a[3]) or (a[0] == 'in' and not a[3]) or
                    (a[0] == 'like' and not a[4])) and
                   seg.cols[a[1]].get('mode') in (0, 1, 2) for a in atoms):
                drive_or = oi
                break
    drive_sf = None
    if drive_eq is None and drive_like is None and drive_or is None:
        for si, (sp2, _op2, _l2) in enumerate(sflags):
            if seg.cols[sp2['col']].get('code_enc') in (3, 18):
                drive_sf = si
                break
    drive_in = None
    if drive_eq is None and drive_like is None and drive_or is None and drive_sf is None:
        for ii, (icol, _iv, ineg) in enumerate(ins):
            c2 = seg.cols[icol]
            if not ineg and c2.get('mode') in (0, 1, 2) and c2.get('code_enc') in (3, 18):
                drive_in = ii            # a positive IN drives: union flag over its codes
                break
    drive_neq = None
    if drive_eq is None and drive_like is None and drive_or is None and drive_sf is None \
            and drive_in is None and not spans:
        for ei, (ecol, _ev, eop) in enumerate(eqs):
            c2 = seg.cols[ecol]
            if eop == '<>' and c2.get('mode') in (0, 1, 2) and c2.get('code_enc') in (3, 18):
                drive_neq = ei           # a lone <> drives: same frame scan, inverted card
                break
    if drive_eq is None and drive_like is None and drive_or is None and drive_sf is None \
            and drive_in is None and drive_neq is None and not spans:
        return None                      # need a driver (=, <>, LIKE, OR, IN, scalar) or a span
    # projections: plain key columns, aggregates, the CASE derived key, or bare * (rows mode)
    proj = tree.expressions
    star = len(proj) == 1 and isinstance(proj[0], E.Star)
    plain_cols = None
    if not star and tree.args.get('group') is None:
        pc = []
        for p in proj:
            inner = p.this if isinstance(p, E.Alias) else p
            if not isinstance(inner, E.Column):
                pc = None; break
            nm2 = col_map.get(inner.name, inner.name) if col_map else inner.name
            if not P.columns_exist(seg, nm2):
                pc = None; break
            pc.append(nm2)
        plain_cols = pc
    if star or plain_cols:
        if tree.args.get('group') is not None or hterms:
            return None
        order = tree.args.get('order')
        lim = wdb_sql._limit(tree)
        if order is None or not order.expressions:
            # THE UNORDERED ROWS (vanilla law's first find): no ORDER BY means any order is a
            # lawful answer, and the driver's positions in row order are one. Without this,
            # `SELECT UserID FROM hits WHERE UserID = k` went to the general path, which decoded
            # all 100M values before applying the predicate: 1,115 ms for 78 rows. No LIMIT is
            # lim -1 (execute declines past 250K hits -- there the O(N) path is the cheaper one).
            if lim is not None and (lim <= 0 or lim > 100000):
                return None
            ocol = None; tiebreak = []
            lim = int(lim) if lim is not None else -1
        else:
            oe = order.expressions[0]
            ocol = oe.this.name if isinstance(oe.this, E.Column) else None
            ocol = col_map.get(ocol, ocol) if (col_map and ocol) else ocol
            if ocol is None or oe.args.get('desc') or seg.stairs(ocol) is None:
                return None              # rows mode rides the cluster order: ASC on a stair column
            tiebreak = []
            for oe2 in order.expressions[1:]:
                if oe2.args.get('desc') or not isinstance(oe2.this, E.Column):
                    return None
                tc = oe2.this.name
                tc = col_map.get(tc, tc) if col_map else tc
                if not P.columns_exist(seg, tc):
                    return None
                tiebreak.append(tc)
            if lim is None or lim <= 0 or lim > 100000:
                return None
        return {'spans': spans, 'eqs': eqs, 'flags': flags, 'ins': ins, 'likes': likes,
                'nulls': nulls, 'ors': ors, 'sflags': sflags, 'drive_eq': drive_eq,
                'drive_like': drive_like, 'drive_or': drive_or, 'drive_sf': drive_sf,
                'drive_in': drive_in, 'drive_neq': drive_neq, 'mode': 'rows',
                'ocol': ocol, 'tiebreak': tiebreak,
                'out_cols': (list(col_map.values()) if col_map else None) if star else plain_cols,
                'lim': int(lim), 'off': int(wdb_sql._offset(tree) or 0)}
    keys, aggs = [], []
    for pi, p in enumerate(proj):
        cd = _count_distinct(p)
        if cd is not None:
            col = col_map.get(cd, cd) if col_map else cd
            if not P.columns_exist(seg, col):   return None
            if seg.cols[col].get('mode') not in (0, 1, 2): return None
            aggs.append((pi, 'COUNT_D', col, )); continue
        ak = wdb_sql._agg_kind(p)
        if ak is not None:
            if ak[0] == 'COUNT_STAR':
                aggs.append((pi, 'COUNT_STAR', None)); continue
            if ak[0] == 'COUNT' and len(ak) > 1 and isinstance(ak[1], str):
                col = col_map.get(ak[1], ak[1]) if col_map else ak[1]
                if not P.columns_exist(seg, col):   return None
                if seg.cols[col].get('mode') not in (0, 1, 2, 4): return None
                aggs.append((pi, 'COUNT_COL', col)); continue
            if ak[0] not in ('SUM', 'AVG', 'MIN', 'MAX') or not isinstance(ak[1], str):
                return None
            col = col_map.get(ak[1], ak[1]) if col_map else ak[1]
            if not P.columns_exist(seg, col):   return None
            if ak[0] == 'MIN' and seg.cols[col].get('mode') in (0, 1, 2):
                aggs.append((pi, 'MIN_DICT', col)); continue
            if ak[0] == 'MAX' and seg.cols[col].get('mode') in (0, 1, 2):
                aggs.append((pi, 'MAX_DICT', col)); continue
            if seg.cols[col].get('dt') != 0:    return None
            if ak[0] in ('MIN', 'MAX') and seg.cols[col].get('mode') != 4:
                return None                       # MIN/MAX are answered here for sequence columns only
            aggs.append((pi, ak[0], col)); continue
        ck = _case_key(p, seg, col_map)
        if ck is not None:
            keys.append((pi, ck)); continue
        sp = wdb_scalar.parse(seg, p, col_map)
        if sp is not None:
            keys.append((pi, {'kind': 'scalar', 'spec': sp, 'src': sp['col']})); continue
        nm = wdb_sql._proj_colname(p)
        if nm is None:
            return None
        col = col_map.get(nm, nm) if col_map else nm
        if not P.columns_exist(seg, col):   return None
        if seg._effective(col) is not None: return None
        keys.append((pi, {'kind': 'col', 'src': col}))
    group = tree.args.get('group')
    if group is None and keys:
        return None
    if group is not None and len(group.expressions) != len(keys):
        return None                      # every key projection must be grouped (and vice versa)
    if hterms and not keys:
        return None                      # HAVING without groups: the fallback's territory
    # resident structures outrank the scan: a gd sidecar serving this exact
    # (group key, COUNT DISTINCT target) shape keeps the query (doctrine: min(point, pop)).
    # LIKE conjuncts disqualify the sidecar (it can only filter eq/in shapes), so no defer.
    if not likes and len(keys) == 1 and keys[0][1].get('kind') == 'col':
        for a in aggs:
            if a[1] == 'COUNT_D' and os.path.exists(
                    wdb_gdsidecar.sidecar_path(seg.path, keys[0][1]['src'], a[2])):
                return None
    order = tree.args.get('order')
    if order is not None:
        if not order.expressions or not order.expressions[0].args.get('desc'):
            return None
        onm = order.expressions[0].this
        cnt_idx = [pi for pi, k, _ in aggs if k == 'COUNT_STAR']
        if not cnt_idx:
            return None
        tgt = onm.name if isinstance(onm, E.Column) else None
        if tgt is None:
            ak2 = wdb_sql._agg_kind(onm)
            if ak2 is None or ak2[0] != 'COUNT_STAR':
                return None
        elif tgt not in (wdb_sql._alias(proj[cnt_idx[0]]), 'COUNT', 'count'):
            return None
        # trailing ASC key columns are this read's canonical emission order (count desc, then the
        # composite asc = each key's value asc, IN KEY ORDER) only when they are the keys in that
        # order, from the first: ORDER BY c DESC, WatchID over keys (ClientIP, WatchID) broke the
        # count ties on ClientIP and cut the wrong rows at the LIMIT (the shifting board's G,
        # 2026-10-12). A prefix is enough -- ties past it are SQL's to leave in any order.
        tail = order.expressions[1:]
        if len(tail) > len(keys):
            return None
        for oe2, (pi2, k2) in zip(tail, keys):
            if oe2.args.get('desc') or not isinstance(oe2.this, E.Column) or k2.get('kind') != 'col':
                return None
            n2 = oe2.this.name
            n2m = col_map.get(n2, n2) if col_map else n2
            if n2m != k2['src'] and n2 != wdb_sql._alias(proj[pi2]):
                return None
    lim = wdb_sql._limit(tree)
    off = wdb_sql._offset(tree) or 0
    # THE COMPOSED COUNT OUTRANKS THE SCAN (2026-10-17, the Q14 board): two keys, COUNT only, ordered by the
    # count, the WHERE one '=' / '<>' on one of the keys -- gridwalk's composed count applies that filter to
    # the counted cells (exact, every core); this scan walked the survivors on one key's frames and passed the
    # board's 20 s cap on 30 of the Q14 samples (IsOldCounter x ClientEventTime 30 s, DuckDB 0.3 s)
    if len(keys) == 2 and order is not None:
        try:
            import wdb_gridwalk
            gsp = wdb_gridwalk.detect(seg, tree, col_map)
            if gsp is not None and gsp.get('afilter') is not None:
                return None
        except Exception:
            pass
    return {'spans': spans, 'eqs': eqs, 'flags': flags, 'ins': ins, 'likes': likes,
            'nulls': nulls, 'ors': ors, 'sflags': sflags, 'drive_eq': drive_eq,
            'drive_like': drive_like, 'drive_or': drive_or, 'drive_sf': drive_sf,
            'drive_in': drive_in, 'drive_neq': drive_neq, 'mode': 'group', 'having': hterms,
            'keys': keys, 'aggs': aggs, 'proj': proj, 'ordered': order is not None,
            'lim': lim, 'off': int(off)}

# ---------------------------------------------------------------- execution
def _bound_code(seg, col, val, side):
    """Insertion point of a literal in the value-sorted dict: the first code whose value is
    >= (side='left') or > (side='right') the literal. Handles int, datetime (dt==3, compared as
    numpy datetime64), and byte-string dicts via fetch-bisect -- O(log V) point fetches."""
    c = seg.cols[col]
    V = int(c['V']) - (1 if c['has_null'] else 0)
    dt = c['dt']
    if dt == 3:
        try:
            tgt = np.datetime64(str(val))
        except Exception:
            return None
        def key(k):
            return np.datetime64(seg.fetch(col, k))
    elif dt == 0:
        try:
            tgt = int(val)
        except (TypeError, ValueError):
            return None
        def key(k):
            return int(seg.fetch(col, k))
    else:
        tgt = _lit_bytes(val)
        def key(k):
            fv = seg.fetch(col, k)
            return fv if isinstance(fv, (bytes, bytearray)) else _lit_bytes(fv)
    lo, hi = 0, V
    while lo < hi:
        mid = (lo + hi) // 2
        kv = key(mid)
        if kv < tgt or (side == 'right' and kv == tgt):
            lo = mid + 1
        else:
            hi = mid
    return lo


def _span_rows(seg, spans):
    lo, hi = 0, int(seg.N)
    for col, val, op in spans:
        st = seg.stairs(col)
        V = int(seg.cols[col]['V']) - (1 if seg.cols[col]['has_null'] else 0)
        def row_of(code, st=st, V=V):            # first row of a code (code==V -> N); bound, not late-bound
            if code <= 0: return 0
            return int(seg.N) if code >= V else (int(st[code - 1]) if code - 1 < st.size else int(seg.N))
        if op == '=':
            k0 = _bound_code(seg, col, val, 'left')
            k1 = _bound_code(seg, col, val, 'right')
            if k0 is None or k1 is None or k0 == k1:
                return 0, 0                      # literal absent -> empty
            lo, hi = max(lo, row_of(k0)), min(hi, row_of(k1))
        elif op == '>=':
            k = _bound_code(seg, col, val, 'left')
            if k is None: return 0, 0
            lo = max(lo, row_of(k))
        else:                                    # '<='
            k = _bound_code(seg, col, val, 'right')
            if k is None: return 0, 0
            hi = min(hi, row_of(k))
    return lo, max(lo, hi)


_BP0EQ = [os.environ.get('WDB_BP0_EQ', '1') != '0']


def _keep_eq_codes(seg, col, pos, code, negate):
    """the codes at an equality scan's hits are the code itself: kept for codes_at (as the frame scans keep theirs),
    so the projection that follows reads no frame -- ClientEventTime = k's scan took 27 ms of the query's 85 without"""
    if not negate and pos.size <= 4_000_000:
        seg.__dict__.setdefault('_scan_codes', {})[col] = (pos, np.full(pos.size, code, np.int64))
    return pos


def _scan_eq18_lanes(seg, c, code, lo, hi, negate):
    """THE ONE-CODE SCAN IN COMPILED LANES (2026-10-21, Q19's board): enc 18's frames inflate through libzstd inside
    the kernel and their codes are tested against the one code as they unpack (wdb_kernels.z18_eq_count / _fill);
    the frame scan below inflated through Python's zstandard on a thread pool and tested a flag table. None (the
    caller scans as before) without libzstd, with frames not yet in memory, or on a frame error."""
    import wdb_engine as _EN
    import wdb_kernels as _WK
    if _EN._libzstd() is None or not _EN._ZLANES[0]:
        return None
    base = int(c['cstart']); po = np.asarray(c['poffs'], np.int64)
    BR = int(c['BR']); bits = int(c['pbits'])
    j0, j1 = lo // BR, (hi - 1) // BR + 1
    if seg._resident_share(base + int(po[j0]), base + int(po[j1])) < 0.9:
        return None
    P = int(_WK.wide_threads())
    scr = np.zeros((P, (BR * bits + 63) // 64 + 4), np.uint64)
    cnt = np.zeros(j1 - j0, np.int64)
    src = np.uint64(seg.buf.ctypes.data); zabs = po + base
    ok = seg._zlanes(lambda zfn, dctx, err: _WK.run_wide(_WK.z18_eq_count, np.uint64(zfn), src, zabs, np.int64(j0),
                     np.int64(j1), dctx, scr, np.int64(bits), np.int64(BR), np.int64(seg.N), np.int64(lo),
                     np.int64(hi), np.int64(code), np.bool_(negate), cnt, err), P)
    if not ok:
        return None
    offs = np.zeros(cnt.size + 1, np.int64)
    np.cumsum(cnt, out=offs[1:])
    out = np.empty(max(1, int(offs[-1])), np.int64)
    ok = seg._zlanes(lambda zfn, dctx, err: _WK.run_wide(_WK.z18_eq_fill, np.uint64(zfn), src, zabs, np.int64(j0),
                     np.int64(j1), dctx, scr, np.int64(bits), np.int64(BR), np.int64(seg.N), np.int64(lo),
                     np.int64(hi), np.int64(code), np.bool_(negate), offs, out, err), P)
    if not ok:
        return None
    return out[:int(offs[-1])]


def _scan_eq(seg, col, code, lo, hi, negate=False):
    """Positions in [lo, hi) where the column's code equals (or differs from) `code`.
    enc=3: parallel decompress of the frames covering the span; other encodings fall back to the
    range read (bitpack touches covering bytes; sealed uses the cached full decode)."""
    import wdb_fpm
    p9 = wdb_fpm.eq_positions(seg, col, code)    # the map: pop only frames that
    if p9 is not None:                           # CONTAIN the code (25/191 for
        if lo > 0 or hi < seg.N:                 # classroom 62)
            a9 = np.searchsorted(p9, lo)
            b9 = np.searchsorted(p9, hi)
            return p9[a9:b9].astype(np.int64)
        return p9.astype(np.int64)
    c = seg.cols[col]
    if int(c.get('code_enc', 0) or 0) == 10:
        # enc 10 reads its own blocks even when the decoded codes are held: searching 100M held codes costs more
        # than walking the blocks (a run is a range, a 1-bit block skips its empty bytes)
        p = _eq_e10(seg, col, c, int(code), lo, hi, negate)
        if p is not None:
            return p
    if col not in seg._codes:
        # THE JUMP TABLE (Jackson, 2026-10-21): the column's encoding picks its scanners; each answers or declines
        # (None) and the next is tried; the decode-and-compare below is the floor under all of them
        for scan in _EQ_TABLE.get(int(c.get('code_enc', 0) or 0), ()):
            p = scan(seg, col, c, int(code), lo, hi, negate)
            if p is not None:
                return p
    cc = np.asarray(seg._raw_codes_range(col, lo, hi))
    h = np.nonzero(cc != code)[0] if negate else np.nonzero(cc == code)[0]
    return h + lo


def _eq_e18_lanes(seg, col, c, code, lo, hi, negate):
    """enc 18 in compiled lanes (a4e9321)"""
    if not _BP0EQ[0] or (hi - lo) < (1 << 16):
        return None
    p18 = _scan_eq18_lanes(seg, c, code, lo, hi, negate)
    return None if p18 is None else _keep_eq_codes(seg, col, p18, code, negate)


def _eq_e18_frames(seg, col, c, code, lo, hi, negate):
    """enc 18 through the packed-frame flag scan"""
    V18 = 1 << int(c['pbits'])                   # every code the stream can hold
    flag18 = np.full(V18, bool(negate), np.bool_)
    if 0 <= code < V18:
        flag18[code] = not negate
    return _scan_flag18(seg, col, np.packbits(flag18, bitorder='little'), lo, hi)


_E3_BATCH = [os.environ.get('WDB_E3_BATCH', '1') != '0']


def _eq_e3_batch(seg, col, c, code, lo, hi):
    """THE ONE-CODE FRAME SCAN (2026-10-24, the dashboard's fixed cost): the span's frames decompressed in ONE call on
    zstd's own C threads (multi_decompress_to_buffer over a BufferWithSegments view of the file -- no per-frame copy,
    no Python thread per frame), each frame then compared with the one code. The dashboard's selector, CounterID = 62
    (738K rows, the 193 frames the load statistics leave): the frame scan 45-50 ms, of which decompressing on 14
    Python threads 8.5 ms (one thread 11.5; zstd's threads 2.6) and the flag-bit test 10.8 ms on one thread against
    the equality's 1.1 -- the rest contention. Whole dashboard queries 58-93 -> 20-55 ms (DuckDB 30-40)."""
    import zstandard as zstd
    BR = int(c['BR']); bo = c['boffs']; base = int(c['cstart'])
    wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
    j0, j1 = lo // BR, (hi - 1) // BR + 1
    if j1 <= j0:
        return np.empty(0, np.int64)
    o0 = int(bo[j0])
    seg.warm_span(base + o0, base + int(bo[j1]))             # cold frames: parallel large reads first
    offs = np.asarray(bo[j0:j1 + 1], np.int64) - o0
    segs = np.empty((j1 - j0, 2), np.uint64)
    segs[:, 0] = offs[:-1]
    segs[:, 1] = np.diff(offs)
    data = memoryview(seg.buf)[base + o0: base + int(bo[j1])]
    dz9 = zstd.ZstdDecompressor()                # one per call: a shared one is not thread-safe
    out = dz9.multi_decompress_to_buffer(zstd.BufferWithSegments(data, segs.tobytes()),
                                              threads=min(_SCAN_THREADS, j1 - j0))
    parts = []
    for i in range(len(out)):
        raw = np.frombuffer(out[i], dtype=wdt)
        j = j0 + i
        a = max(lo, j * BR) - j * BR
        b = min(hi, j * BR + raw.size) - j * BR
        h = np.flatnonzero(raw[a:b] == code)
        if h.size:
            parts.append(h + (j * BR + a))
    pos = np.concatenate(parts) if parts else np.empty(0, np.int64)
    if pos.size <= 4_000_000:                    # THE SCANNED CODES, kept for codes_at (as the frame scan keeps them)
        seg.__dict__.setdefault('_scan_codes', {})[col] = (pos, np.full(pos.size, code, np.int64))
    return pos


def _eq_e3(seg, col, c, code, lo, hi, negate):
    # = : the batch decompress + the equality (_eq_e3_batch); <> keeps the frame scan below, whose kernel writes the
    # tens of millions of positions a common code's complement makes better than a compare per frame (measured
    # 0.087 vs 0.102 s, 0.080 vs 0.106 on Robotness / ResponseEndTiming's commonest)
    if not negate and code >= 0 and _E3_BATCH[0] and col not in seg._codes:
        try:
            return _eq_e3_batch(seg, col, c, int(code), lo, hi)
        except Exception:
            pass                                 # a frame the batch call cannot take: the frame scan below
    # one code IS a flag: the nogil frame scan (THE SCAN WITHOUT THE GIL) instead of a numpy
    # compare per frame under the GIL -- UserID = k measured 155 -> ~60 ms on 1,526 frames
    V3 = int(c.get('V') or 0)
    flag3 = np.full(max(V3, code + 1), bool(negate), np.bool_)
    if code >= 0:
        flag3[code] = not negate
    return _scan_flag(seg, col, flag3, lo, hi)


def _eq_e19(seg, col, c, code, lo, hi, negate):
    if negate or os.environ.get('WDB_E19_LABELS', '1') == '0':
        return None
    return _scan_eq19(seg, col, code, lo, hi)


def _eq_e20_pattern(seg, col, c, code, lo, hi, negate):
    """THE DIFFERENTIATION PATH (Jackson, 2026-10-21): '= k' only needs to tell rows apart, so the back-reference
    stream is not parsed where k cannot be. A row equal to k leads back, inside its own block, to a literal holding
    k: the bits '0, then k's code' sit in that block's bytes. One flat search of the bytes for that pattern
    (wdb_kernels.e20_pattern_blocks) flags the blocks that can hold k; only those are walked exactly (e20_eq_list).
    A stray match (the pattern straddling rows) costs one block's walk, never a wrong row.
    THE SWITCH: code width >= 23 (the two-byte key needs width + 1 >= 24; the width sweep put strays at <= ~10% of
    blocks there: 25 -> 1.9%, 24 -> 5.6%, 23 -> 9.9%; below, 22% at 21, 46% at 20); sixteen evenly spaced blocks probed
    first, then (after the search) fewer than a quarter of the blocks flagged -- a common value: the full walk. RefererHash = k: search 28 ms + 12 of 1,526 blocks
    walked, where the full walk is 50-65 ms (DuckDB 36). WDB_E20_PATTERN=0 declines; WDB_E20_PATTERN_MIN is the width."""
    import wdb_kernels as _WK
    bits = int(c['e20bits'])
    if (negate or not _E20PAT[0] or bits < max(23, _E20PAT_MIN[0]) or code < 0 or code >= (1 << bits)
            or (hi - lo) < (1 << 16)):
        return None
    BR = int(c['e20BR']); b0 = lo // BR; b1 = (hi - 1) // BR + 1
    boff = c['e20boff']; buf8 = np.frombuffer(seg.buf, np.uint8); base = np.int64(c['cstart'])
    P = code << 1
    table = np.zeros(65536, np.uint8)
    for o in range(8):
        table[(P >> (8 - o)) & 0xFFFF] |= (1 << o)
    # THE PROBE (2026-10-21, Q41): a common value flags most blocks and the search is wasted (Q41's URLHash over one
    # counter's 12.6M rows: 4.1-4.9 ms thrown away before the walk). Sixteen evenly spaced blocks are searched first;
    # a rare value flags almost none (strays: ~2% at width 25), a common one most -- a quarter flagged: the walk.
    nsp = b1 - b0
    probe = np.unique(np.linspace(b0, b1 - 1, min(16, nsp)).astype(np.int64))
    pc = np.zeros(probe.size, np.uint8)
    _WK.e20_pattern_blocks(buf8, base, boff, probe, np.int64(bits), np.uint64(P), table, pc)
    if 4 * int(pc.sum()) >= probe.size:
        return None
    allb = np.arange(b0, b1, dtype=np.int64)
    cand = np.zeros(nsp, np.uint8)
    _WK.run_wide(_WK.e20_pattern_blocks, buf8, base, boff, allb, np.int64(bits), np.uint64(P), table, cand)
    blocks = allb[cand != 0]
    if 4 * blocks.size >= nsp:
        return None                                  # a common value the probe missed: the full walk
    global _E20PAT_HITS
    _E20PAT_HITS += 1
    offs = np.zeros(blocks.size + 1, np.int64)
    args = (buf8, base, np.int64(BR), np.int64(seg.N), np.int64(bits), boff, blocks, np.int64(lo), np.int64(hi),
            np.int64(code), offs)
    _WK.run_wide(_WK.e20_eq_list, *(args + (np.empty(1, np.int64), False)))
    np.cumsum(offs, out=offs)
    out = np.empty(max(1, int(offs[-1])), np.int64)
    _WK.run_wide(_WK.e20_eq_list, *(args + (out, True)))
    return _keep_eq_codes(seg, col, out[:int(offs[-1])], code, negate)


def _eq_e20_walk(seg, col, c, code, lo, hi, negate):
    # THE ONE-CODE SCAN THROUGH THE BACK-REFERENCE (2026-10-21, Q19's board): the full decode wrote 100M codes to
    # compare them with one (RefererHash = k: 0.12 s for one row, DuckDB 0.036 s); the walk keeps the hits of
    # one block on its lane's board, and a second walk writes positions for the blocks that hold any
    if (hi - lo) < (1 << 16) or not _BP0EQ[0]:
        return None
    import wdb_kernels as _WK
    BR20 = int(c['e20BR']); b0 = lo // BR20; b1 = (hi - 1) // BR20 + 1
    P20 = int(_WK.wide_threads())
    scr20 = np.empty((P20, BR20), np.bool_)
    cnt20 = np.zeros(b1 - b0, np.int64)
    args20 = (np.frombuffer(seg.buf, np.uint8), np.int64(c['cstart']), np.int64(BR20), np.int64(seg.N),
              np.int64(c['e20bits']), c['e20boff'], np.int64(lo), np.int64(hi), np.int64(code), np.bool_(negate),
              scr20)
    _WK.run_wide(_WK.e20_eq_count, *(args20 + (cnt20,)))
    off20 = np.zeros(cnt20.size + 1, np.int64)
    np.cumsum(cnt20, out=off20[1:])
    out20 = np.empty(max(1, int(off20[-1])), np.int64)
    _WK.run_wide(_WK.e20_eq_fill, *(args20 + (off20, out20)))
    return _keep_eq_codes(seg, col, out20[:int(off20[-1])], code, negate)


def _eq_e0_packed(seg, col, c, code, lo, hi, negate):
    # THE ONE-CODE SCAN ON THE PACKED BITS (2026-10-21, Q19's board): plain bitpack had no equality scanner, so
    # one code decoded every row of the span into an array and compared it (WatchID = k: 100M codes, 400 MB,
    # 0.18 s for one row, DuckDB 0.034 s). The tile reader tests each packed code against the one target in a
    # register; a second pass writes positions for the chunks that hold any. WDB_BP0_EQ=0 restores the decode.
    if ('boffs' in c or not c.get('bits') or not (0 < int(c['bits']) <= 32) or (hi - lo) < (1 << 16)
            or not _BP0EQ[0]):
        return None
    import wdb_kernels as _WK
    bufE = np.frombuffer(seg.buf, np.uint8)
    CH = 1 << 18
    nch = (hi - lo + CH - 1) // CH
    cntE = np.zeros(nch, np.int64)
    argsE = (bufE, np.int64(c['cstart']), np.int64(c['bits']), np.int64(lo), np.int64(hi), np.int64(code),
             np.bool_(negate))
    _WK.run_wide(_WK.bp0_eq_count, *(argsE + (cntE, np.int64(CH))))
    offE = np.zeros(nch + 1, np.int64)
    np.cumsum(cntE, out=offE[1:])
    outE = np.empty(max(1, int(offE[-1])), np.int64)
    _WK.run_wide(_WK.bp0_eq_fill, *(argsE + (offE, outE, np.int64(CH))))
    return _keep_eq_codes(seg, col, outE[:int(offE[-1])], code, negate)


# THE JUMP TABLE: encoding -> its equality scanners, in the order they are tried
_E10EQ = [os.environ.get('WDB_E10_EQ', '1') != '0']


def _eq_e10(seg, col, c, code, lo, hi, negate):
    """enc 10 straight from its blocks (wdb_kernels.bp10_eq_pos, 2026-10-24)"""
    bits = int(c.get('pXbits') or 0)
    if not _E10EQ[0] or not 1 <= bits <= 16 or (hi - lo) < (1 << 16):
        return None
    import wdb_kernels as _WK
    dirX = np.frombuffer(seg.buf, np.int64, int(c['pXnblk']), c['pXdir'])
    return _WK.bp10_eq_pos(np.frombuffer(seg.buf, np.uint8), np.ascontiguousarray(dirX), np.int64(c['pXpay']),
                           np.int64(bits), np.int64(c['pXn']), np.int64(lo), np.int64(hi), np.int64(code),
                           bool(negate))


_EQ_TABLE = {
    0: (_eq_e0_packed,),
    10: (_eq_e10,),
    3: (_eq_e3,),
    18: (_eq_e18_lanes, _eq_e18_frames),
    19: (_eq_e19,),
    20: (_eq_e20_pattern, _eq_e20_walk),
}
_E20PAT = [os.environ.get('WDB_E20_PATTERN', '1') != '0']
_E20PAT_MIN = [int(os.environ.get('WDB_E20_PATTERN_MIN', '23'))]
_E20PAT_HITS = 0


def _scan_eq19(seg, col, code, lo, hi):
    """THE BOX LABELS (Jackson, 2026-09-27): a block-dictionary (tag 19) column carries, per block of
    rows, the sorted list of the codes in it. 'Which rows hold code k' reads the labels -- through the
    load's signposts when present (Q19: 0.15 ms for 1,526 blocks), else walking each label to k --
    then the pointers of only the blocks whose label holds k. The full decode it replaces unpacked
    all 100M codes (48 ms hot, 247 MB cold) to keep 4 rows. None: the caller scans as before."""
    import wdb_kernels as _WK
    import wdb_blockstats
    c = seg.cols[col]
    if c.get('e19dc') is None or code < 0 or lo >= hi:
        return None
    BR = int(c['e19BR']); N = int(seg.N)
    b0 = lo // BR; b1 = (hi - 1) // BR + 1
    pw, dw = seg._e19_words(c)
    dc = np.asarray(c['e19dc'], np.int64)
    lid = np.full(dc.size, -1, np.int64)
    gb = np.int64(c['e19bits'])
    if 'e19R' in c:
        # THE SHELVES (Jackson, 2026-09-27): the labels by code range -- code k's pieces of every label sit
        # together on shelf k // W (~256 KB for UserID), so the lookup reads that one shelf, not 58 MB
        W = int(c['e19W']); r = code // W
        if r < int(c['e19R']) and os.environ.get('WDB_E19_WARM', '1') != '0':
            ds = int(c['e19dstart']); SW = c['e19SW']
            seg.warm_span(ds + 8 * int(SW[r]), min(len(seg.buf), ds + 8 * int(SW[r + 1])))
        _WK.e19s_eq_blocks(dw, np.int64(W), np.int64(c['e19wb']), c['e19gw'], c['e19pre'], c['e19soff'],
                           c['e19SW'], np.int64(code), np.int64(b0), np.int64(b1), lid)
    elif os.environ.get('WDB_E19_WARM', '1') != '0':
        # THE GATE (Jackson, 2026-09-27): the walk touches ~1 page of every block's label, and cold each
        # touch was one page fault waited on alone (483 x ~0.34 ms = 163 ms on the network volume). mincore
        # asks which label pages are in memory (~70 us warm); the missing ones come in by parallel large
        # reads first (warm_span), then the walk runs in memory. Warm, nothing is read.
        ds = int(c['e19dstart']); doff = c['e19doff']
        seg.warm_span(ds + 8 * int(doff[b0]), min(len(seg.buf), ds + 8 * (int(doff[b1]) + 1)))
    spp = None if 'e19R' in c else wdb_blockstats.signposts_from_load(seg, col)
    if 'e19R' in c:
        pass                                           # (the shelf lookup above filled lid)
    elif spp is not None:
        sp, spo, S = spp
        _WK.e19_eq_blocks_sp(dw, gb, c['e19gw'], dc, c['e19doff'], np.int64(S), spo, sp,
                             np.int64(code), np.int64(b0), np.int64(b1), lid)
    else:
        _WK.e19_eq_blocks(dw, gb, c['e19gw'], dc, c['e19doff'], np.int64(code), np.int64(b0), np.int64(b1), lid)
    if os.environ.get('WDB_E19_PAR', '1') != '0':
        pos = _WK.e19_rows_eq_all(pw, np.int64(BR), np.int64(N), c['e19lb'], c['e19poff'], lid,
                                  np.int64(b0), np.int64(b1), np.int64(lo), np.int64(hi))
    else:
        out = np.empty(1024, np.int64)
        k = _WK.e19_rows_eq(pw, np.int64(BR), np.int64(N), c['e19lb'], c['e19poff'], lid,
                            np.int64(b0), np.int64(b1), np.int64(lo), np.int64(hi), out)
        if k > out.size:                               # more rows than the first guess: once more, sized
            out = np.empty(k, np.int64)
            k = _WK.e19_rows_eq(pw, np.int64(BR), np.int64(N), c['e19lb'], c['e19poff'], lid,
                                np.int64(b0), np.int64(b1), np.int64(lo), np.int64(hi), out)
        pos = out[:k].copy()
    if pos.size <= 4_000_000:                          # the codes at the hits, for codes_at (as enc 3/18)
        seg.__dict__.setdefault('_scan_codes', {})[col] = (pos, np.full(pos.size, code, np.int64))
    return pos

def _num_table(seg, col):
    """code -> numeric value for any int-valued dict column (mode 2 int-dict layout or a
    generic mode-0/1 dict whose values are ints)."""
    c = seg.cols[col]
    try:
        return np.asarray(seg._dict_ints(c), dtype=np.float64)
    except Exception:
        return np.array([float(v) for v in seg._typed_dict(col)], dtype=np.float64)


def _in_codes(seg, col, vals):
    """Dict codes for an IN list. Pre-resolved code sets (same-column subqueries) pass
    straight through. The dict-pass path decodes the ENTIRE dictionary
    (V-proportional: ~6s for SearchPhrase's 6M) while bisects cost only the probes
    (each _code_of is O(log V) point-fetches, never materializing). The old fixed
    500-value threshold was calibrated in the leak era when the decoded dict was
    retained and free; honestly priced, the crossover scales with the dictionary."""
    if isinstance(vals, dict):
        return np.asarray(vals['_codes'], dtype=np.int64)
    c = seg.cols[col]
    nd = int(c.get('n_dict') or c.get('V') or 0)
    if len(vals) * 64 > nd:
        dv = seg._typed_dict(col)
        idx = {(x if isinstance(x, (bytes, bytearray)) else str(x).encode()
                if isinstance(x, str) else x): i for i, x in enumerate(dv)}
        codes = []
        for v in vals:
            kk = idx.get(v.encode() if isinstance(v, str) else v)
            if kk is None and isinstance(v, str):
                # coerce like _code_of: string literals against numeric dicts
                try:
                    kk = idx.get(int(v))
                except ValueError:
                    try:
                        kk = idx.get(float(v))
                    except ValueError:
                        kk = None
            if kk is None and not isinstance(v, (str, bytes, bytearray)):
                kk = idx.get(v)
            if kk is not None:
                codes.append(kk)
        return np.array(codes, dtype=np.int64)
    codes = [_code_of(seg, col, v) for v in vals]
    return np.array([k for k in codes if k is not None], dtype=np.int64)


def _flag_bits(seg, flag):
    """THE FLAG AS BITS: an 18 MB bool table missed the cache on every one of 100M lookups; packed
    to 2.3 MB it sits in L2 (130 -> 102 ms for the lookups alone, one thread); memoized per flag
    object (the object is held, so its id cannot be reused while remembered)"""
    _pk = seg.__dict__.setdefault('_flag_bits', {})
    hit = _pk.get(id(flag))
    if hit is None or hit[0] is not flag:
        hit = (flag, np.packbits(np.ascontiguousarray(flag, dtype=np.bool_), bitorder='little'))
        if len(_pk) > 64: _pk.clear()
        _pk[id(flag)] = hit
    return hit[1]


def _scan_flag18(seg, col, fbits, lo, hi):
    """THE PACKED-FRAME SCAN (enc 18): each zstd frame inflates to a bit-stream (memcpy speed --
    the packed bytes compress little) and one kernel unpacks and tests membership in a single
    loop; the scanned column's codes at the hits are kept for codes_at, as for enc 3"""
    import wdb_kernels as _WK18
    c = seg.cols[col]; BR = c['BR']; bits18 = int(c['pbits'])
    j0, j1 = lo // BR, (hi - 1) // BR + 1
    seg.warm_span(int(c['cstart']) + int(c['poffs'][j0]), int(c['cstart']) + int(c['poffs'][j1]))   # cold frames first
    mv = memoryview(seg.buf)
    def scan(js):
        import zstandard as zstd
        dz = zstd.ZstdDecompressor(); outp = []; outc = []
        hitbuf = np.empty(BR, np.int64); codebuf = np.empty(BR, np.int64)
        for j in js:
            fb = seg._pk18_frame(c, j, dz, mv)
            a, b = max(lo, j * BR), min(hi, j * BR + BR, int(seg.N))
            n = _WK18.pk32_flag_hits(fb, bits18, a - j * BR, b - j * BR, a, fbits, hitbuf, codebuf)
            if n:
                outp.append(hitbuf[:n].copy()); outc.append(codebuf[:n].copy())
        return (np.concatenate(outp), np.concatenate(outc)) if outp else (np.empty(0, np.int64), np.empty(0, np.int64))
    W = min(_SCAN_THREADS, max(1, j1 - j0))
    with ThreadPoolExecutor(W) as ex:
        parts = list(ex.map(scan, np.array_split(np.arange(j0, j1), W)))
    parts = [p for p in parts if p[0].size]
    if not parts:
        return np.empty(0, np.int64)
    pos = np.concatenate([p[0] for p in parts]); codes = np.concatenate([p[1] for p in parts])
    if pos.size <= 4_000_000:
        seg.__dict__.setdefault('_scan_codes', {})[col] = (pos, codes)
    return pos


def _scan_flag3_lanes(seg, c, bitsB, lo, hi, j0, j1):
    """THE FRAME SCAN IN COMPILED LANES (2026-10-30, E: Q21's small-text LIKEs on the PC). The thread pool below hands
    each frame across threads three times (inflate, test, copy out); under a virtual machine every handoff wakes a
    virtual CPU, so 14 threads ran slower than 4 (HitColor LIKE '%g', 1,526 frames: 390 ms in the query; inflating
    alone 20 ms on 4 threads, 59 on 8). Here one compiled call holds the whole scan: pass 1 inflates each frame into
    its lane's board, counts its hits and keeps up to a quarter frame of them in the frame's slot (offset, code); the
    running total places every frame's hits; pass 2 copies each frame's hits from its slot -- inflating again only a
    frame that held more -- and writes positions (and codes, when they will be kept) in place. Inflating everything
    twice made URL LIKE '%google%' (16K hits in 1,432 of 1,526 frames) 51 -> 85 ms on the server; with the slots 47.
    Returns (positions, codes or None), or None to fall back to the thread pool (no libzstd or numba, switched off,
    a frame that fails)."""
    if not _FLAG_LANES[0]:
        return None
    import wdb_engine as _WE
    import wdb_kernels as _WK
    kern = getattr(_WK, 'z3_flag_hits', None)
    if kern is None or _WE._libzstd() is None or not _WE._ZLANES[0]:
        return None
    wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
    BR = int(c['BR']); base = int(c['cstart'])
    zabs = np.asarray(c['boffs'][j0:j1 + 1], np.int64) + base
    nf = j1 - j0
    P = int(_WK.wide_threads())
    scr = np.empty((P, BR), wdt)
    cnt = np.zeros(nf, np.int64)
    e0 = np.empty(0, np.int64)
    K = max(1, BR // int(_SLOT_DIV[0]))
    soff = np.empty(nf * K, np.int32); scode = np.empty(nf * K, np.uint32)   # touched only where hits land
    src = np.uint64(seg.buf.ctypes.data)
    out = []
    def run(zfn, dctx, err):
        args = (np.uint64(zfn), src, zabs, dctx, scr, np.int64(BR), np.int64(j0), np.int64(lo), np.int64(hi), bitsB,
                np.int64(K), soff, scode, cnt)
        _WK.run_wide(kern, *args, e0, e0, e0, False, err)
        if err.any():
            return
        offs = np.zeros(nf + 1, np.int64)
        np.cumsum(cnt, out=offs[1:])
        tot = int(offs[-1])
        pos = np.empty(tot, np.int64)
        codes = np.empty(tot, np.int64) if tot <= _SCAN_CODES_MAX else e0
        if tot:
            _WK.run_wide(kern, *args, offs, pos, codes, True, err)
        out.append((pos, codes if tot <= _SCAN_CODES_MAX else None))
    if not seg._zlanes(run, P) or not out:
        return None
    return out[0]


def _scan_flag(seg, col, flag, lo, hi):
    """Positions in [lo, hi) where flag[code] is set -- the LIKE frame scan: parallel enc=3
    decompress, one fancy-index per frame; the substring test happened once, in the dict."""
    c = seg.cols[col]
    if c.get('code_enc', 0) == 8 and hasattr(seg, 'e8_planes'):
        pl = seg.e8_planes(col)
        if pl is not None:
            pos8, lits8, d8 = pl
            if not bool(flag[d8]):               # the default can't match (a blank never
                ckm = '_e8sf_' + col             # contains anything): the answer lives
                hit = seg._codes.get(ckm)        # ENTIRELY in the literals -- and the
                if hit is not None and hit[0] == id(flag):   # block loop calls this 382
                    rows, bnd, BSTEP = hit[1], hit[2], hit[3]
                else:
                    rows = pos8[flag[lits8]]
                    BSTEP = 262144               # boundaries for EVERY block, once:
                    bnd = np.searchsorted(rows, np.arange(0, int(seg.N) + BSTEP, BSTEP))
                    seg._codes[ckm] = (id(flag), rows, bnd, BSTEP)
                if lo > 0 or hi < seg.N:
                    if lo % BSTEP == 0 and (hi % BSTEP == 0 or hi >= seg.N):
                        a = int(bnd[lo // BSTEP])          # O(1) per block: the 382
                        b = int(bnd[min(hi // BSTEP, len(bnd) - 1)])   # searches die
                    else:
                        a = int(np.searchsorted(rows, lo))
                        b = int(np.searchsorted(rows, hi))
                    return rows[a:b]
                return rows
    if c.get('code_enc', 0) == 18 and col not in seg._codes:
        return _scan_flag18(seg, col, _flag_bits(seg, flag), lo, hi)
    if c.get('code_enc', 0) == 3 and col not in seg._codes:
        # THE FRAME SCAN WITHOUT THE GIL: zstd releases it, but flag[raw] and nonzero held it, so
        # fourteen threads ran one at a time (221 ms for 1,526 frames; 63 ms per 200 frames on one
        # thread was the whole cost). The membership test is a nogil kernel now, and it keeps the
        # scanned column's codes at the hits -- the caller's codes_at over the same positions would
        # have decompressed every frame again (93 ms for 16K rows of URL)
        wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
        BR = c['BR']; base = c['cstart']; bo = c['boffs']; buf = seg.buf
        j0, j1 = lo // BR, (hi - 1) // BR + 1
        seg.warm_span(base + int(bo[j0]), base + int(bo[j1]))   # cold frames: parallel large reads first
        bitsB = _flag_bits(seg, flag)
        got = _scan_flag3_lanes(seg, c, bitsB, lo, hi, j0, j1)
        if got is not None:
            pos, codes = got
            if codes is not None:
                seg.__dict__.setdefault('_scan_codes', {})[col] = (pos, codes)
            return pos
        mv = memoryview(buf)                     # slices without the copy: 89 -> 61 ms of decompression
        def scan(js):
            import zstandard as zstd
            dz = zstd.ZstdDecompressor(); outp = []; outc = []
            hitbuf = np.empty(BR, np.int64); codebuf = np.empty(BR, np.int64)
            for j in js:
                raw = np.frombuffer(dz.decompress(mv[base+int(bo[j]):base+int(bo[j+1])]), dtype=wdt)
                a, b = max(lo, j*BR), min(hi, j*BR + raw.size)
                n = _flag_hits_kernel(raw, bitsB, a - j*BR, b - j*BR, a, hitbuf, codebuf)
                if n:
                    outp.append(hitbuf[:n].copy()); outc.append(codebuf[:n].copy())
            return (np.concatenate(outp), np.concatenate(outc)) if outp else (np.empty(0, np.int64), np.empty(0, np.int64))
        W = min(_SCAN_THREADS, max(1, j1 - j0))
        with ThreadPoolExecutor(W) as ex:
            parts = list(ex.map(scan, np.array_split(np.arange(j0, j1), W)))
        parts = [p for p in parts if p[0].size]
        if not parts:
            return np.empty(0, np.int64)
        pos = np.concatenate([p[0] for p in parts]); codes = np.concatenate([p[1] for p in parts])
        if pos.size <= _SCAN_CODES_MAX:                                    # THE SCANNED CODES, kept for codes_at:
            seg.__dict__.setdefault('_scan_codes', {})[col] = (pos, codes)  # a fact about the segment, never stale
        return pos
    if c.get('code_enc', 0) == 12 and (hi - lo) >= (1 << 16):
        import wdb_kernels as _WK                # vertical membership scan
        plV = seg.vplanes(col)
        CH = 1 << 18
        nch = (hi - lo + CH - 1) // CH
        cnts = np.zeros(nch, np.int64)
        flagV = np.ascontiguousarray(flag, dtype=np.bool_)
        _WK.vp_scan_flag_count(plV, int(c['nwords']), int(c['bits']), lo, hi, flagV, cnts, CH)
        offs = np.zeros(nch + 1, np.int64)
        np.cumsum(cnts, out=offs[1:])
        outV = np.zeros(max(1, int(offs[-1])), np.int64)
        _WK.vp_scan_flag_fill(plV, int(c['nwords']), int(c['bits']), lo, hi, flagV, offs, outV, CH)
        return outV[:int(offs[-1])]
    if c.get('code_enc', 0) == 0 and 'boffs' not in c and c.get('bits') \
            and 0 < int(c['bits']) <= 32 and (hi - lo) >= (1 << 16):
        import wdb_kernels as _WK                # RULE 3: the fused scan --
        bufS = np.frombuffer(seg.buf, np.uint8)  # values live in registers,
        CH = 1 << 18                             # only positions ever write
        nch = (hi - lo + CH - 1) // CH
        cnts = np.zeros(nch, np.int64)
        flagS = np.ascontiguousarray(flag, dtype=np.bool_)
        _WK.bp0_scan_count(bufS, int(c['cstart']), int(c['bits']), lo, hi, flagS, cnts, CH)
        offs = np.zeros(nch + 1, np.int64)
        np.cumsum(cnts, out=offs[1:])
        outS = np.zeros(max(1, int(offs[-1])), np.int64)
        _WK.bp0_scan_fill(bufS, int(c['cstart']), int(c['bits']), lo, hi, flagS, offs, outS, CH)
        return outS[:int(offs[-1])]
    # THE GENERAL FLAG SCAN (2026-10-19, the LIKE board): encodings with no fused scanner above (enc 10 / 9 / 2 ...)
    # built arange for the range read and indexed the flag on one thread -- UTMContent 250 ms flat for codes that
    # decode in 7-12 ms. The whole column decodes once (kept for the codes_at that follow), the scan runs on
    # every core.
    if lo == 0 and hi >= int(seg.N):
        cc = np.asarray(seg._raw_codes(col))
    else:
        cc = np.asarray(seg._raw_codes_range(col, lo, hi))
    fl9 = np.asarray(flag, bool)
    if cc.size >= (1 << 16) and cc.dtype.kind in 'iu' and fl9.size >= int(c.get('V') or 0) > 0:
        import wdb_kernels as _WK
        return _WK.flag_positions(cc, fl9, lo)
    return np.nonzero(fl9[cc])[0] + lo


def _or_union_flag(seg, atoms):
    """Union flag table for a same-column OR group of positive membership atoms."""
    col = atoms[0][1]
    V = int(seg.cols[col]['V'])
    flag = np.zeros(V, bool)
    for a in atoms:
        if a[0] == 'eq':
            k = _code_of(seg, col, a[2])
            if k is not None:
                flag[k] = True
        elif a[0] == 'in':
            flag[_in_codes(seg, col, a[2])] = True
        else:                                    # like
            flag |= _like_flags(seg, col, a[2], a[3])
    return flag


def _atom_mask(seg, a, pos, ccache):
    """Boolean mask over pos for one atom; per-column code gathers cached for the query."""
    col = a[1]
    c = seg.cols[col]
    if a[0] == 'flag' or (c['mode'] == 4):
        if col not in ccache:
            ccache[col] = np.asarray(seg._seq_decode(c))[pos]
        vv = ccache[col]
        if a[0] == 'flag':
            return vv != a[2] if a[3] == '<>' else vv == a[2]
        if a[0] == 'in':
            m = np.isin(vv, np.array([int(v) for v in a[2]], np.int64))
            return ~m if a[3] else m
        return np.zeros(pos.size, bool) if a[0] == 'null' and a[2] else np.ones(pos.size, bool)
    if col not in ccache:
        ccache[col] = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
    cc = ccache[col]
    if a[0] == 'eq':
        k = _code_of(seg, col, a[2])
        if k is None:
            return np.ones(pos.size, bool) if a[3] else np.zeros(pos.size, bool)
        return cc != k if a[3] else cc == k
    if a[0] == 'in':
        ks = _in_codes(seg, col, a[2])
        if not ks.size:
            m = np.zeros(pos.size, bool)
        else:
            V = int(c.get('V', 0))
            if 0 < V <= 100_000_000:
                flag = np.zeros(V + 1, dtype=bool)   # flag-gather beats np.isin's
                flag[ks] = True                      # sort/unique over 100M codes
                m = flag[cc]
            else:
                m = np.isin(cc, ks)
        return ~m if a[3] else m
    if a[0] == 'like':
        fl = _like_flags(seg, col, a[2], a[3])
        m = fl[cc]
        return ~m if a[4] else m
    if a[0] == 'null':
        if not c.get('has_null'):
            return np.zeros(pos.size, bool) if a[2] else np.ones(pos.size, bool)
        nc = int(c['V']) - 1
        return cc == nc if a[2] else cc != nc
    if a[0] == 'range':
        b = _bound_code(seg, col, a[2], 'left' if a[3] == '>=' else 'right')
        if b is None:
            return np.zeros(pos.size, bool)
        return cc >= b if a[3] == '>=' else cc < b
    return np.zeros(pos.size, bool)


_COUNT_CENSUS_V = 1 << 20   # COUNT + LIKE from the census up to this many values (past it the scan is cheaper)
_DENSE_SPAN = 1 << 26    # a key's code range up to 64M (past 4M the census is the radix one: no per-lane boards)
_DENSE_MIN_ROWS = 1 << 20   # the parallel factorization from this many rows (below, numpy's one pass is enough)
_LATE_MIN_ROWS = 1 << 16    # the late aggregates from this many survivors


def _fact_counts(a):
    """(sorted values present, row -> group, rows per group): _fact_small plus the counts, in one pass when the
    parallel dense road serves (a single dictionary key: its codes are the group ids)"""
    if a.size >= _DENSE_MIN_ROWS and a.size:
        amin = int(a.min()); amax = int(a.max())
        span = amax - amin + 1
        if span <= _DENSE_SPAN:
            import wdb_kernels as _WK
            return _WK.dense_factorize(a.astype(np.int64, copy=False), amin, span)
    u, inv = _fact_small(a)
    return u, inv, np.bincount(inv, minlength=u.size)


def _fact_small(a):
    """np.unique(a, return_inverse=True) without the hidden sort: when the value range
    is narrow (derived keys like minute 0-59 or LENGTH 0-500, and composites over them),
    a bincount presence table factorizes in O(N). Identical contract; sorted u; falls
    back to np.unique for wide or empty inputs."""
    if a.size == 0:
        return np.unique(a, return_inverse=True)
    amin = int(a.min()); amax = int(a.max())
    span = amax - amin + 1
    if span > (1 << 20) and span <= _DENSE_SPAN and a.size >= _DENSE_MIN_ROWS:
        import wdb_kernels as _WK                # dictionary codes as group ids, in parallel (2026-10-19)
        u, inv, _c = _WK.dense_factorize(a.astype(np.int64, copy=False), amin, span)
        return u, inv
    if span > (1 << 20):
        return np.unique(a, return_inverse=True)
    shifted = (a - amin).astype(np.int64, copy=False)
    pres = np.bincount(shifted, minlength=span)
    uvals = np.flatnonzero(pres)
    rank = np.empty(span, dtype=np.int64)
    rank[uvals] = np.arange(uvals.size)
    return (uvals + amin), rank[shifted]


_POTENCY = [__import__('os').environ.get('WDB_WS_POTENCY', '1') == '1']   # A/B: 0 restores the fixed driver ladder
_POT_BLOCKS, _POT_BLOCK = 32, 2048   # the sample: evenly spaced blocks of neighbouring rows (frames open once)
_POT_SWITCH = 0.5           # the potency pick replaces the ladder's driver only when it keeps under half its rows
_POT_LAST = [None]          # (driver kind, index, keep shares) of the last pick: the boards' witness


def _pot_sample(seg, lo, hi):
    n = hi - lo
    if n <= _POT_BLOCKS * _POT_BLOCK:
        return np.arange(lo, hi, dtype=np.int64)
    starts = lo + (np.arange(_POT_BLOCKS, dtype=np.int64) * (n - _POT_BLOCK)) // (_POT_BLOCKS - 1)
    return (starts[:, None] + np.arange(_POT_BLOCK, dtype=np.int64)[None, :]).ravel()


def _pot_keep(seg, col, test, sample, cc_memo):
    """share of the sampled rows a single-column filter keeps: test(codes) -> bool per code, over the codes of
    the sampled rows (one read of each column's sample, shared by its filters). A ranking, not an answer: the
    sample is _POT_BLOCKS blocks of _POT_BLOCK neighbouring rows spread over the span -- a frame-coded column
    opens _POT_BLOCKS frames, a point-read column reads 64K rows, a census would cost up to a whole decode"""
    if col not in cc_memo:
        cc_memo[col] = np.asarray(seg.codes_at(col, sample)).astype(np.int64)
    cc = cc_memo[col]
    if cc.size == 0:
        return 0.0
    return float(np.count_nonzero(np.asarray(test(cc), bool))) / cc.size


def _potency_pick(seg, spec, lo, hi):
    """THE POTENCY PICK (2026-10-19, Jackson: "a failure of the potency based system to detect that search phrase
    filter as an option"). The driver -- the one filter that scans the span and writes positions -- was chosen by
    a fixed ladder (an '=', else the first LIKE, then OR / scalar / IN, and a '<>' only on enc 3/18 with nothing
    else): Q21 with a one-letter needle drove on the LIKE that keeps 80% of the rows, wrote 80M positions, then
    the residual SearchPhrase <> '' dropped 85% of them. Every positive LIKE, '=', '<>' and IN on a dictionary
    column is a candidate here; each one's keep share is measured (_pot_keep), and the one keeping fewest rows
    drives when it keeps under _POT_SWITCH of the ladder's driver. The ladder's driver then runs as a residual."""
    if not _POTENCY[0] or hi - lo < (1 << 16):
        return spec
    if spec.get('drive_or') is not None or spec.get('drive_sf') is not None:
        return spec
    cands = []          # (kind, index, column, test(codes) -> keep)
    for i, (col, needle, kind, neg) in enumerate(spec.get('likes', [])):
        if not neg:
            # the whole dictionary's flags (memoized for the query): the driver reuses them, and a LIKE moved to the
            # residuals answers from them at once. Deciding only the sampled values cost more: 4,096 random URL
            # codes open ~4,096 front-coded chunks (192 ms), where the whole decision is the driver's own cost anyway
            cands.append(('like', i, col,
                          lambda cc, col=col, needle=needle, kind=kind:
                          np.asarray(_like_flags(seg, col, needle, kind), bool)[cc]))
    for i, (col, val, op) in enumerate(spec['eqs']):
        c2 = seg.cols.get(col) or {}
        if c2.get('mode') not in (0, 1, 2) or seg._effective(col) is not None:
            continue
        def fe(cc, col=col, val=val, op=op, c2=c2):
            code = _code_of(seg, col, val)
            if code is None:
                m = np.zeros(cc.size, bool) if op == '=' else np.ones(cc.size, bool)
            else:
                m = (cc == int(code)) if op == '=' else (cc != int(code))
            if op == '<>' and c2.get('has_null'):
                m &= cc != int(c2['V']) - 1      # SQL: NULL <> x is not TRUE
            return m
        cands.append(('eq', i, col, fe))
    for i, (col, vals, ineg) in enumerate(spec.get('ins', ())):
        c2 = seg.cols.get(col) or {}
        if ineg or c2.get('mode') not in (0, 1, 2):
            continue
        cands.append(('in', i, col, lambda cc, col=col, vals=vals: np.isin(cc, _in_codes(seg, col, vals))))
    if len(cands) < 2:
        return spec
    cur = (('eq', spec['drive_eq']) if spec.get('drive_eq') is not None else
           ('like', spec['drive_like']) if spec.get('drive_like') is not None else
           ('in', spec['drive_in']) if spec.get('drive_in') is not None else
           ('eq', spec['drive_neq']) if spec.get('drive_neq') is not None else None)
    sample = _pot_sample(seg, lo, hi)
    cc_memo = {}
    shares = {}
    for kind, i, col, test in cands:
        try:
            shares[(kind, i)] = _pot_keep(seg, col, test, sample, cc_memo)
        except Exception:
            continue
    if not shares:
        return spec
    best = min(shares, key=lambda k: shares[k])
    cur_share = shares.get(cur, 1.0) if cur is not None else 1.0
    _POT_LAST[0] = (best, cur, {('%s%d' % k): round(v, 4) for k, v in shares.items()})
    if best == cur or shares[best] >= _POT_SWITCH * cur_share:
        return spec
    spec = dict(spec)
    for k in ('drive_eq', 'drive_like', 'drive_in', 'drive_neq'):
        spec[k] = None
    kind, i = best
    if kind == 'like':
        spec['drive_like'] = i
    elif kind == 'in':
        spec['drive_in'] = i
    elif spec['eqs'][i][2] == '=':
        spec['drive_eq'] = i
    else:
        spec['drive_neq'] = i
    return spec


_COL_EMIT_MIN = 10000      # past this many groups the answer is emitted by column (THE COLUMNAR EMISSION)


def _key_values(seg, kk, a):
    """one key's values for the returned groups, exactly as the per-group loop emits them: a sequence column's own
    values, a scalar's surrogate values, a CASE key's default for -1, else seg.values_at -- [_pyval(fetch(code))]
    by contract, each front-coded block walked once (the point fetches were 223 s of 232 for SearchPhrase's 5.9M
    groups: ~65 Python steps per value, 17 us against values_at's 1 us)"""
    if kk['kind'] == 'scalar':
        # each surrogate value as the per-group loop emits it (v.item() for a numpy scalar): the value list holds
        # numpy ints for the arithmetic and length scalars, and an object array's tolist hands them back as numpy
        # ints (np.int64(5), not 5) -- converted once per distinct value, then indexed (fixed 2026-10-24)
        sv = wdb_scalar.surrogate(seg, kk['spec'])[1]
        conv = np.empty(len(sv), dtype=object)
        conv[:] = [v.item() if hasattr(v, 'item') else v for v in sv]
        return conv[np.asarray(a, np.int64)].tolist()
    src = kk['src']; c = seg.cols[src]
    if kk['kind'] == 'case':
        dflt = kk['default'] if isinstance(kk['default'], str) else str(kk['default'])
        out = np.empty(a.size, dtype=object)
        neg = a < 0
        out[neg] = dflt
        if (~neg).any():
            live = a[~neg].astype(np.int64)
            vals = live.tolist() if c['mode'] == 4 else seg.values_at(src, live)
            out[~neg] = np.array(list(vals) + [None], dtype=object)[:-1]
        return out.tolist()
    if c['mode'] == 4:
        return a.astype(np.int64).tolist()
    return seg.values_at(src, a.astype(np.int64))


def _emit_columns(seg, spec, keyarr, sel, sel_w, rep, cnt, cells):
    """THE COLUMNAR EMISSION (2026-10-24, the GROUP BY holes): the per-group loop below sorted a list and point-fetched
    every key value of every group -- SearchPhrase, COUNT(*) WHERE IsRefresh = 0 GROUP BY SearchPhrase was 102.6 s for
    5.9M groups, 223 s of a profiled 232 in the point fetches. Here every projected column is one array over the
    returned groups, each front-coded dictionary block walked once, the same values the loop emits (seg.values_at: the loop's _pyval(fetch) by contract); a ColRows past COL_GATE
    (wdb_colresult), tuples below."""
    import wdb_colresult as CR
    n = int(sel.size)
    byp = {}
    for pi, kind, col in spec['aggs']:
        if kind == 'COUNT_STAR':
            byp[pi] = cnt[sel].astype(np.int64).tolist()
        elif kind in ('COUNT_COL', 'COUNT_D', 'SUM', 'MIN', 'MAX'):
            byp[pi] = np.asarray(cells[pi])[sel_w].astype(np.int64).tolist()
        elif kind in ('MIN_DICT', 'MAX_DICT'):
            k2 = np.asarray(cells[pi])[sel_w].astype(np.int64)
            dead = (k2 < 0) if kind == 'MAX_DICT' else (k2 == np.iinfo(np.int64).max)
            out = np.empty(n, dtype=object)
            if (~dead).any():
                out[~dead] = np.array(list(seg.values_at(col, k2[~dead])) + [None], dtype=object)[:-1]
            out[dead] = None
            byp[pi] = out.tolist()
        else:                                            # AVG
            byp[pi] = (np.asarray(cells[pi], dtype=np.float64)[sel_w] / cnt[sel]).tolist()
    for (pi, kk), (_x, aa) in zip(spec['keys'], keyarr):
        byp[pi] = _key_values(seg, kk, np.asarray(aa)[rep])
    cols = [byp[pi] for pi in sorted(byp)]
    if n > CR.COL_GATE:
        return CR.ColRows([('arr', cl) for cl in cols], n)
    return list(zip(*cols))


_EXC_ROUTE = [os.environ.get('WDB_WS_EXCEPTIONS', '1') != '0']
_COUNT_BOARD = [os.environ.get('WDB_WS_COUNTBOARD', '1') != '0']
_EXC_HITS = 0
_BOARD_HITS = 0


def _count_key(seg, spec):
    """the one key of a COUNT-only GROUP BY counted by code -> (key spec, source column, ids, surrogate or None):
    a dictionary column (ids = its codes) or a scalar over one (ids = the surrogate's sorted value ids)"""
    if spec.get('mode') != 'group' or len(spec['keys']) != 1:
        return None
    if any(a[1] != 'COUNT_STAR' for a in spec['aggs']):
        return None
    if any(h[0] != 'COUNT_STAR' for h in (spec.get('having') or ())):
        return None
    _pi, k = spec['keys'][0]
    if k['kind'] not in ('col', 'scalar'):
        return None
    c = seg.cols[k['src']]
    if c.get('mode') not in (0, 1, 2) or seg._effective(k['src']) is not None:
        return None
    if k['kind'] == 'col':
        return k, k['src'], int(c['V']), None
    surr, svals = wdb_scalar.surrogate(seg, k['spec'])
    surr = np.asarray(surr, np.int64)
    if surr.size < int(c['V']):
        return None
    return k, k['src'], len(svals), surr


def _exceptions_counts(seg, spec, lo, hi, ck):
    """THE EXCEPTIONS ROAD (2026-10-24, Jackson: "if the filter is only getting rid of 10% of the rows and then we
    list 90M things that will be rough -- list the exceptions instead"). COUNT(*) WHERE filters GROUP BY k is
        k's census over every row  -  k's census over the rows some filter fails,
    and the filters' own censuses say, before any scan, how many rows each keeps: the exceptions (at most the sum of
    the failing counts) are listed instead of the survivors (at least the smallest keeping count) whenever they are
    fewer. Measured (excproto.py, same counts): RegionID WHERE IsRefresh = 0 (93.3% kept) 0.550 -> 0.228 s,
    DontCountHits = 0 0.486 -> 0.278, AdvEngineID = 0 0.528 -> 0.149, SearchPhrase / UserID by IsRefresh 0.64 /
    0.78 -> 0.34 / 0.35; SearchPhrase WHERE TraficSourceID = 0 (18.3% kept) keeps the survivors (0.49 vs 1.17).
    = / <> filters only, the whole table (no span), COUNT only -- subtraction undoes a count; MIN / MAX it cannot."""
    if not _EXC_ROUTE[0] or lo != 0 or hi < int(seg.N) or not spec['eqs']:
        return None
    if spec['flags'] or spec['ins'] or spec['likes'] or spec['nulls'] or spec['ors'] or spec['sflags']:
        return None
    N = int(seg.N)
    fails, keeps, parts = 0, N, []
    for col, val, op in spec['eqs']:
        c = seg.cols[col]
        V = int(c['V'])
        code = _code_of(seg, col, val)
        cen = np.asarray(seg.raw_census(col, V))[:V]
        nullc = V - 1 if c.get('has_null') else None
        if op == '=':
            kept = int(cen[code]) if code is not None else 0
        else:                                        # SQL: NULL <> x is not TRUE
            kept = N - (int(cen[code]) if code is not None else 0) - (int(cen[nullc]) if nullc is not None else 0)
        fails += N - kept
        keeps = min(keeps, kept)
        parts.append((col, code, op, nullc))
    if fails >= keeps:
        return None                                  # the survivors are the smaller list: the scan below
    import wdb_kernels as _WKu
    xpos = None
    for col, code, op, nullc in parts:
        if op == '=':
            p = _scan_eq(seg, col, int(code), 0, N, negate=True)        # every row not holding the code, NULLs too
        else:
            ps = []
            if code is not None:
                ps.append(_scan_eq(seg, col, int(code), 0, N))
            if nullc is not None and nullc != code:
                ps.append(_scan_eq(seg, col, int(nullc), 0, N))
            p = (ps[0] if len(ps) == 1 else _WKu.union_sorted(np.asarray(ps[0], np.int64), np.asarray(ps[1], np.int64))
                 if ps else np.empty(0, np.int64))
        p = np.asarray(p, np.int64)
        xpos = p if xpos is None else _WKu.union_sorted(xpos, p)      # the scans hand back ascending positions
    k, src, nid, surr = ck
    Vs = int(seg.cols[src]['V'])
    cen = np.asarray(seg.raw_census(src, Vs))[:Vs].astype(np.int64)
    if xpos.size:
        xc = np.asarray(seg.codes_at(src, xpos)).astype(np.int64)
        cen = cen - np.bincount(xc, minlength=Vs)[:Vs]
    if surr is None:
        return cen
    return np.rint(np.bincount(surr[:Vs], weights=cen, minlength=nid)).astype(np.int64)


def _kth_count(cnt, K9):
    """the K9-th largest count, ties kept (the GROUP BY tail's count of the counts)"""
    cmax9 = int(cnt.max())
    if K9 <= 0:
        return cmax9 + 1
    if cmax9 <= (1 << 24):
        hist9 = np.bincount(cnt, minlength=cmax9 + 1)
        above9 = np.cumsum(hist9[::-1])
        return cmax9 - int(np.searchsorted(above9, K9))
    return np.partition(cnt, cnt.size - K9)[cnt.size - K9]


def _emit_counts(seg, spec, k, counts):
    """a COUNT-only GROUP BY's answer from its counts by id (codes, or a scalar's sorted value ids): the groups are
    the ids with rows; HAVING on the count, then the order the GROUP BY tail below gives (count descending, then
    id ascending -- its factorised groups in code order), the LIMIT window, keys through _key_values"""
    import wdb_colresult as CR
    counts = np.asarray(counts, np.int64)
    g = np.flatnonzero(counts)
    cnt = counts[g]
    for _hk, _hc, hop, hval in (spec.get('having') or ()):
        a = cnt.astype(np.float64)
        m = (a > hval if hop == '>' else a >= hval if hop == '>=' else a < hval if hop == '<' else
             a <= hval if hop == '<=' else a == hval if hop == '=' else a != hval)
        g = g[m]; cnt = cnt[m]
    K9 = None if spec['lim'] is None else spec['off'] + spec['lim']
    if spec['ordered'] and g.size:
        if K9 is not None and cnt.size > 4 * K9 + 64:
            cand = np.flatnonzero(cnt >= _kth_count(cnt, K9))
            order = cand[np.lexsort((g[cand], -cnt[cand]))]
        else:
            order = np.lexsort((g, -cnt))
    else:
        order = np.arange(g.size)
    sel = order[spec['off']: spec['off'] + spec['lim']] if spec['lim'] is not None else order[spec['off']:]
    byp = {spec['keys'][0][0]: _key_values(seg, k, g[sel])}
    for pi, _kind, _c in spec['aggs']:
        byp[pi] = cnt[sel].tolist()
    cols = [byp[pi] for pi in sorted(byp)]
    hdr = [wdb_sql._alias(p) for p in spec['proj']]
    n = int(sel.size)
    if n > CR.COL_GATE:
        return CR.ColRows([('arr', cl) for cl in cols], n), hdr
    return list(zip(*cols)) if cols else [], hdr


def execute(seg, spec):
    global _HITS, _EXC_HITS, _BOARD_HITS
    lo, hi = _span_rows(seg, spec['spans'])
    spec = _potency_pick(seg, spec, lo, hi)
    _ck9 = _count_key(seg, spec)
    if _ck9 is not None:
        _xc9 = _exceptions_counts(seg, spec, lo, hi, _ck9)
        if _xc9 is not None:
            _HITS += 1; _EXC_HITS += 1
            return _emit_counts(seg, spec, _ck9[0], _xc9)
    likes = spec.get('likes', [])
    de, dl = spec.get('drive_eq'), spec.get('drive_like')
    # only the DRIVING like decides its whole dictionary (it scans every row); every other like is
    # decided later, at the survivors, for the codes still alive (_like_at)
    lflags = [(col, _like_flags(seg, col, needle, kind) if i == dl else None, neg)
              for i, (col, needle, kind, neg) in enumerate(likes)] if likes else []
    _dn = None
    pos_full = False      # pos is every row of the span: the first residual = / <> may scan instead (2026-10-19)
    if (dl is not None and not spec.get('keys') and spec.get('mode') != 'rows' and len(likes) == 1
            and spec.get('aggs') and all(a9[1] == 'COUNT_STAR' for a9 in spec['aggs'])
            and not spec['eqs'] and not spec.get('ins') and not spec.get('nulls') and not spec.get('ors')
            and not spec.get('sflags') and not spec['flags'] and lo == 0 and hi >= int(seg.N)):
        # THE COUNT FROM THE CENSUS (2026-10-19, the LIKE board): COUNT(*) WHERE col LIKE p over a dictionary
        # of at most ~1M values is the rows of the matching values -- the census of the codes, counted from the
        # encoding itself where it can be (raw_census), with no row positions written. The scan wrote every hit's
        # position to count them (HitColor LIKE '%5%': 80M positions, 366 ms; UTMContent 250 ms flat). Big
        # dictionaries keep the scan: their census costs more than it (Q20 on URL: 480 ms against 150).
        col9, fl9, _n9 = lflags[dl]
        c9 = seg.cols[col9]
        V9 = int(c9['V'])
        n9 = None
        if likes[dl][2] == 'all' and not c9.get('has_null'):
            n9 = int(seg.N)                      # LIKE '%' over a column with no NULL: every row
        elif V9 <= _COUNT_CENSUS_V:
            cn9 = np.asarray(seg.raw_census(col9, V9))[:V9]
            f9 = np.zeros(V9, bool)
            m9 = min(V9, int(np.asarray(fl9).size))
            f9[:m9] = np.asarray(fl9, bool)[:m9]
            n9 = int(cn9[f9].sum())
        if n9 is not None:
            _HITS += 1
            return [tuple(n9 for _a in spec['aggs'])], [wdb_sql._alias(p) for p in spec['proj']]
    if hi <= lo:
        pos = np.empty(0, np.int64)
    elif dl is not None and not np.asarray(lflags[dl][1]).any():
        pos = np.empty(0, np.int64)      # the driving LIKE matches no value: no row can pass (2026-10-19)
    elif de is not None:
        col, val, _op = spec['eqs'][de]
        code = _code_of(seg, col, val)
        pos = _scan_eq(seg, col, int(code), lo, hi) if code is not None else np.empty(0, np.int64)
    elif dl is not None:
        col, fl, _n = lflags[dl]
        if np.asarray(fl).all():
            pos = np.arange(lo, hi, dtype=np.int64)   # every code matches (LIKE '%', no NULL): no scan to run
            pos_full = True
        else:
            pos = _scan_flag(seg, col, fl, lo, hi)
    elif spec.get('drive_sf') is not None:
        sp2, op2, l2 = spec['sflags'][spec['drive_sf']]
        pos = _scan_flag(seg, sp2['col'], _scalar_flag(seg, sp2, op2, l2), lo, hi)
    elif spec.get('drive_in') is not None:
        icol, ivals, _n = spec['ins'][spec['drive_in']]
        ks = _in_codes(seg, icol, ivals)
        fl = np.zeros(int(seg.cols[icol]['V']), bool)
        fl[ks] = True
        pos = _scan_flag(seg, icol, fl, lo, hi)
    elif spec.get('drive_neq') is not None:
        ncol, nval, _op = spec['eqs'][spec['drive_neq']]
        fl = np.ones(int(seg.cols[ncol]['V']), bool)
        kc = _code_of(seg, ncol, nval)
        if kc is not None:
            fl[kc] = False
        if seg.cols[ncol].get('has_null'):
            fl[-1] = False               # SQL: NULL <> x is not TRUE
        pos = _scan_flag(seg, ncol, fl, lo, hi)
    elif spec.get('drive_or') is not None:
        atoms = spec['ors'][spec['drive_or']]
        bycol = {}
        for a in atoms:
            bycol.setdefault(a[1], []).append(a)
        parts = [_scan_flag(seg, col, _or_union_flag(seg, grp), lo, hi)
                 for col, grp in bycol.items()]
        pos = parts[0]
        for pp in parts[1:]:
            pos = np.union1d(pos, pp)
    else:
        for _ni, (_ncol, _wn) in enumerate(spec.get('nulls', ())):
            _c2 = seg.cols[_ncol]
            if _c2['mode'] != 4 and _c2.get('has_null'):
                _dn = _ni
                break
        if _dn is not None:
            # drive on the null predicate: null is just code V-1, so IS [NOT] NULL is
            # an equality scan -- never materialize arange(N) to filter it afterward
            _ncol, _wn = spec['nulls'][_dn]
            pos = _scan_eq(seg, _ncol, int(seg.cols[_ncol]['V']) - 1, lo, hi,
                           negate=not _wn)
        else:
            pos = np.arange(lo, hi, dtype=np.int64)
            pos_full = True
    # residual order is ours to choose (conjunction commutes): equalities before
    # inequalities (eq slashes survivors, neq barely trims), cheap dictionaries
    # before heavy ones -- the cte-chain lesson: testing a 6.5M-V string column
    # at 738K rows before a 9K-V column cut them to 5K cost half the query
    _res = sorted(range(len(spec['eqs'])),
                  key=lambda _i: (0 if spec['eqs'][_i][2] == '=' else 1,
                                  int(seg.cols.get(spec['eqs'][_i][0], {}).get('V') or 0)))
    _eqcc = {}            # col -> (pos, its codes there): kept when no later filter moves pos (the GROUP BY key)
    for i in _res:
        col, val, op = spec['eqs'][i]
        if i == de or i == spec.get('drive_neq') or pos.size == 0: continue
        code = _code_of(seg, col, val)
        if code is None:
            if op == '=': pos = np.empty(0, np.int64)
            continue
        c_e = seg.cols.get(col) or {}
        if pos_full and c_e.get('mode') in (0, 1, 2) and seg._effective(col) is None:
            # THE RESIDUAL THAT SCANS (2026-10-19): nothing has filtered yet (a LIKE every value passes), so this
            # filter reads the column through its own scanners -- SearchPhrase <> '' from the e8 planes' literals
            # -- instead of gathering codes at every row and comparing them on one thread
            if op == '=':
                pos = _scan_eq(seg, col, int(code), lo, hi)
            else:
                fl_e = np.ones(int(c_e['V']), bool)
                fl_e[int(code)] = False
                if c_e.get('has_null'):
                    fl_e[-1] = False             # SQL: NULL <> x is not TRUE
                pos = _scan_flag(seg, col, fl_e, lo, hi)
            pos_full = False
            continue
        cc = np.asarray(seg.codes_at(col, pos))
        keep9 = (cc != code) if op == '<>' else (cc == code)
        pos = pos[keep9]
        _eqcc[col] = (pos, cc[keep9])
    for ii, (col, vals, ineg) in enumerate(spec.get('ins', ())):
        if ii == spec.get('drive_in') or pos.size == 0:
            continue
        c = seg.cols[col]
        if c['mode'] == 4:
            want = np.array([int(v) for v in vals], dtype=np.int64)
            vv = np.asarray(seg._seq_decode(c))[pos]
            m = np.isin(vv, want)
            pos = pos[~m] if ineg else pos[m]
        else:
            codes = _in_codes(seg, col, vals)
            if codes.size == 0:
                if not ineg:
                    pos = np.empty(0, np.int64); break
                continue
            cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
            V = int(c.get('V', 0))
            if 0 < V <= 100_000_000:
                flag = np.zeros(V + 1, dtype=bool)   # flag-gather: one pass, no sort --
                flag[codes] = True                   # np.isin uniques/sorts 100M codes
                m = flag[cc]
            else:
                m = np.isin(cc, codes)
            pos = pos[~m] if ineg else pos[m]
    for _ni, (col, want_null) in enumerate(spec.get('nulls', ())):
        if _ni == _dn:
            continue                             # already driven
        if pos.size == 0: break
        c = seg.cols[col]
        if c['mode'] == 4 or not c.get('has_null'):
            if want_null:
                pos = np.empty(0, np.int64)      # column carries no nulls
            continue
        nc = int(c['V']) - 1
        cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
        pos = pos[cc == nc] if want_null else pos[cc != nc]
    for oi, atoms in enumerate(spec.get('ors', ())):
        if oi == spec.get('drive_or') or pos.size == 0:
            continue
        ccache = {}
        m = _atom_mask(seg, atoms[0], pos, ccache)
        for a in atoms[1:]:
            m |= _atom_mask(seg, a, pos, ccache)
        pos = pos[m]
    for si, (sp2, op2, l2) in enumerate(spec.get('sflags', ())):
        if si == spec.get('drive_sf') or pos.size == 0:
            continue
        fl = _scalar_flag(seg, sp2, op2, l2)
        cc = np.asarray(seg.codes_at(sp2['col'], pos)).astype(np.int64)
        pos = pos[fl[cc]]
    for col, val, op in spec['flags']:
        if pos.size == 0: break
        vv = np.asarray(seg._seq_decode(seg.cols[col]))[pos]
        pos = pos[vv != val] if op == '<>' else pos[vv == val]
    # THE LIKES LAST (Jackson: the cheap filter before the expensive read): a residual LIKE costs
    # the distinct strings still alive, so every code-compare filter above runs first and shrinks
    # what it must decide. Conjunction commutes: the answer cannot change, only the bill.
    for i, (col, fl, neg) in enumerate(lflags):
        if i == dl or pos.size == 0: continue
        cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
        _lc, needle_i, kind_i, _ln = likes[i]
        m = _like_at(seg, col, needle_i, kind_i, cc)
        if neg:
            m = ~m
            c_n = seg.cols.get(col) or {}
            if c_n.get('has_null'):
                m &= cc != int(c_n['V']) - 1     # SQL: NULL NOT LIKE x is not true
        pos = pos[m]

    # ---- rows mode: SELECT * ordered by the cluster column -- positions ARE the order
    if spec.get('mode') == 'rows':
        if spec['lim'] < 0:                          # THE UNORDERED ROWS, no LIMIT: every hit
            if pos.size > 250_000:
                return None                          # past this the O(N) general path is cheaper
            spec = dict(spec); spec['lim'] = int(pos.size)
        K = spec['off'] + spec['lim']
        if spec.get('tiebreak') and pos.size > K:
            # total order: extend to the full plateau of the K-th row's cluster value,
            # decode, sort by the complete ORDER BY column list, then slice
            oc = np.asarray(seg.codes_at(spec['ocol'], pos)).astype(np.int64)
            kc = oc[K - 1]
            cut = int(np.searchsorted(oc, kc, side='right'))
            sel = pos[:cut]
        else:
            sel = pos[spec['off']: spec['off'] + spec['lim']]
        cols = spec.get('out_cols') or list(seg.cols.keys())
        vals = []
        for cn in cols:
            c = seg.cols[cn]
            if c['mode'] == 4:
                vals.append([int(x) for x in np.asarray(seg._seq_decode(c))[sel]])
            else:
                cc = np.asarray(seg.codes_at(cn, sel)).astype(np.int64)
                vals.append([wdb_sql._pyval(seg.fetch(cn, int(k))) for k in cc])
        out = [tuple(vals[ci][ri] for ci in range(len(cols))) for ri in range(sel.size)]
        if spec.get('tiebreak') and len(out) > spec['lim']:
            oi = [cols.index(spec['ocol'])] + [cols.index(c) for c in spec['tiebreak']]
            out.sort(key=lambda r: tuple(('' if r[i] is None else r[i]) for i in oi))
            out = out[spec['off']: spec['off'] + spec['lim']]
        _HITS += 1
        return out, cols

    # ---- aggregates only (no GROUP BY)
    if not spec['keys']:
        row = []
        for _pi, kind, col in spec['aggs']:
            if kind == 'COUNT_STAR':
                row.append(int(pos.size)); continue
            if pos.size == 0:
                row.append(None); continue
            c = seg.cols[col]
            if kind == 'COUNT_COL':
                c2 = seg.cols[col]
                if c2['mode'] == 4 or not c2.get('has_null'):
                    row.append(int(pos.size)); continue
                cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
                row.append(int((cc != int(c2['V']) - 1).sum())); continue
            if kind in ('MIN_DICT', 'MAX_DICT'):
                c2 = seg.cols[col]
                cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
                if kind == 'MAX_DICT' and c2.get('has_null'):
                    cc = cc[cc != int(c2['V']) - 1]        # nulls sit at V-1: MAX ignores them
                    if cc.size == 0:
                        row.append(None); continue
                k2 = int(cc.min()) if kind == 'MIN_DICT' else int(cc.max())
                row.append(wdb_sql._pyval(seg.fetch(col, k2))); continue
            if kind == 'COUNT_D':
                cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
                row.append(int(np.unique(cc).size)); continue
            if kind in ('MIN', 'MAX'):
                # MIN/MAX OF A SEQUENCE COLUMN (2026-10-03): it fell through to the SUM/AVG fold below and
                # answered the AVERAGE -- MIN(id) WHERE info = '...' gave 82867.0 for DuckDB's 21106
                v = np.asarray(seg._seq_decode(c))[pos]
                row.append(int(v.min() if kind == 'MIN' else v.max())); continue
            if c['mode'] == 4:
                v = np.asarray(seg._seq_decode(c))[pos]
                import wdb_exactint as XI
                row.append(XI.fold_values(v) if kind == 'SUM'
                           else XI.exact_avg(XI.fold_values(v), v.size)); continue
            cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
            if c['mode'] == 2 and c.get('dt') != 0:
                v = _num_table(seg, col)[cc]
                row.append(float(v.sum(dtype=np.float64)) if kind == 'SUM'
                           else float(v.sum(dtype=np.float64)) / v.size); continue
            # int values (dict modes 0/1, and mode-2 ints): EXACT at any magnitude via
            # dictionary arithmetic -- bincount the survivor codes (V-sized counts),
            # two-limb fold (wdb_exactint). Also faster than the 33M gather+sum, and
            # nulls (codes past the value table) are properly excluded.
            import wdb_exactint as XI
            import wdb_window as _WN
            tab = _WN._int_table(seg, col)
            cnts = np.bincount(cc, minlength=tab.size)
            if cnts.size > tab.size:
                cnts = cnts[:tab.size]     # null codes live past the dict's values:
            n = int(cnts.sum())            # SQL aggs exclude them
            s = XI.fold_counts(cnts, tab)
            row.append((s if n else None) if kind == 'SUM' else XI.exact_avg(s, n))
        _HITS += 1
        return [tuple(row)], [wdb_sql._alias(p) for p in spec['proj']]

    # ---- GROUP BY: gather keys at survivor positions, factorize-then-combine (overflow-safe)
    kcols = {}
    for _pi, k in spec['keys']:
        if k['kind'] == 'scalar':
            srcs = [k['src']]
        elif k['kind'] == 'col':
            srcs = [k['src']]
        else:
            srcs = [k['src']] + [c for c, _ in k['conds']]
        for s in srcs:
            if s in kcols: continue
            c = seg.cols[s]
            _h9 = _eqcc.get(s)
            if c['mode'] == 4:
                kcols[s] = np.asarray(seg._seq_decode(c))[pos]
            elif _h9 is not None and _h9[0] is pos:
                kcols[s] = _h9[1].astype(np.int64, copy=False)   # read already by the <> (2026-10-19)
            else:
                kcols[s] = np.asarray(seg.codes_at(s, pos)).astype(np.int64)
    keyarr = []
    for _pi, k in spec['keys']:
        if k['kind'] == 'scalar':
            surr, _vals = wdb_scalar.surrogate(seg, k['spec'])
            keyarr.append((k, surr[kcols[k['src']]]))
        elif k['kind'] == 'col':
            keyarr.append((k, kcols[k['src']]))
        else:
            m = np.ones(pos.size, bool)
            for c, v in k['conds']:
                if seg.cols[c]['mode'] == 4:
                    m &= kcols[c] == int(v)
                else:                            # dict cond: compare in CODE space
                    code = _code_of(seg, c, v)
                    m &= (kcols[c] == code) if code is not None else False
            keyarr.append((k, np.where(m, kcols[k['src']], -1)))
    if _ck9 is not None and _COUNT_BOARD[0]:
        # THE COUNT BOARD (2026-10-24): one key, COUNT only -- the groups are the ids with rows, counted on a board by
        # id; the factorise below (0.47 s of a 90M-survivor key) and the first-row walk are not needed
        a9 = np.asarray(keyarr[0][1], np.int64)
        nid9 = _ck9[2]
        if a9.size == 0 or int(a9.max()) < nid9:
            import wdb_kernels as _WK
            c9 = (_WK.bincount_par(a9, nid9) if (nid9 <= (1 << 22) and a9.size) else np.bincount(a9, minlength=nid9))
            _HITS += 1; _BOARD_HITS += 1
            return _emit_counts(seg, spec, _ck9[0], np.asarray(c9, np.int64)[:nid9])
    if len(keyarr) == 1:
        # ONE KEY (2026-10-19): its factorization is the grouping -- the composite's second factorization was the
        # same pass again, and both sorted when the key's codes spanned past 1M (_fact_counts: dense, parallel)
        u, ginv, cnt = _fact_counts(keyarr[0][1])
        g = np.arange(u.size, dtype=np.int64)
    else:
        comp = np.zeros(pos.size, dtype=np.int64)
        locals_ = []
        for _k, a in keyarr:
            u, inv = _fact_small(a)
            comp = comp * u.size + inv
            locals_.append(u)
        g, ginv = _fact_small(comp)
        cnt = np.bincount(ginv)
    K9 = None if spec['lim'] is None else spec['off'] + spec['lim']
    if spec['ordered'] and K9 is not None and not spec.get('having') and cnt.size > 4 * K9 + 64:
        # THE CANDIDATES ONLY (2026-10-19): ORDER BY count LIMIT k needs the groups at or above the k-th count,
        # ties included, ordered exactly as before (count descending, then group) -- not a sort of every group
        cmax9 = int(cnt.max())
        if K9 <= 0:
            kth = cmax9 + 1
        elif cmax9 <= (1 << 24):
            # the k-th largest count from a count of the counts: one pass, no selection worst case
            # (np.partition over 6M mostly-tied counts: 0.97 s under Q21 with K LIKE '%')
            hist9 = np.bincount(cnt, minlength=cmax9 + 1)
            above9 = np.cumsum(hist9[::-1])           # groups with count >= cmax9 - j
            kth = cmax9 - int(np.searchsorted(above9, K9))
        else:
            kth = np.partition(cnt, cnt.size - K9)[cnt.size - K9]
        cand = np.flatnonzero(cnt >= kth)
        order = cand[np.lexsort((g[cand], -cnt[cand]))]
    else:
        order = np.lexsort((g, -cnt)) if spec['ordered'] else np.arange(g.size)
    if spec.get('having'):
        hmask = np.ones(cnt.size, bool)
        for hkind, hcol, hop, hval in spec['having']:
            if hkind == 'COUNT_STAR':
                arr = cnt.astype(np.float64)
            else:
                c2 = seg.cols[hcol]
                if c2['mode'] == 4:
                    v = np.asarray(seg._seq_decode(c2))[pos].astype(np.float64)
                else:
                    v = _num_table(seg, hcol)[np.asarray(seg.codes_at(hcol, pos)).astype(np.int64)]
                sums = np.bincount(ginv, weights=v, minlength=cnt.size)
                arr = sums if hkind == 'SUM' else sums / np.maximum(cnt, 1)
            m = (arr > hval if hop == '>' else arr >= hval if hop == '>=' else
                 arr < hval if hop == '<' else arr <= hval if hop == '<=' else
                 arr == hval if hop == '=' else arr != hval)
            hmask &= m
        order = order[hmask[order]]
    sel = order[spec['off']: spec['off'] + spec['lim']] if spec['lim'] is not None else order[spec['off']:]
    # THE LATE AGGREGATES (2026-10-19, the LIKE board): the groups the answer returns are known once the counts
    # are; every other aggregate is computed over THEIR rows only. Q21 with a one-letter needle gathered MIN(K)'s
    # codes for 60M survivors and folded them with ufunc.at, to keep ten groups. From here pos / ginv / G are
    # the returned groups' rows (sel_w: each returned group's local index); rep maps back to every survivor's row.
    G_all = cnt.size
    rix9 = None
    if sel.size < G_all and pos.size >= _LATE_MIN_ROWS:
        loc9 = np.full(G_all, -1, np.int64)
        loc9[sel] = np.arange(sel.size, dtype=np.int64)
        lg9 = loc9[ginv]
        rix9 = np.flatnonzero(lg9 >= 0)
        pos = pos[rix9]; ginv = lg9[rix9]
        sel_w = np.arange(sel.size, dtype=np.int64)
    else:
        sel_w = sel
    G = int(sel.size) if rix9 is not None else G_all
    first = np.empty(G, np.int64)                   # first row of every group, one pass,
    first[ginv[::-1]] = np.arange(pos.size - 1, -1, -1, dtype=np.int64)   # no sort: reversed
    rep = first[sel_w]                              # writes -- last write is the earliest row
    if rix9 is not None:
        rep = rix9[rep]                             # back to rows of every survivor (the key arrays' rows)
    # ONE gather per agg column over all pos, then every group's aggregate vectorized --
    # the per-group codes_at gathers re-decompressed the frames each group's scattered
    # members touch (gs-cube: 1,165 gathers, 8,773 zstd calls for a 4-column query).
    _cells = {}
    _gath = {}
    def _codes_all(col):
        if col not in _gath:
            c3 = seg.cols[col]
            if c3['mode'] == 4:
                _gath[col] = np.asarray(seg._seq_decode(c3))[pos]
            else:
                _gath[col] = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
        return _gath[col]
    for pi3, kind3, col3 in spec['aggs']:
        if kind3 == 'COUNT_STAR':
            continue
        c3 = seg.cols[col3]
        if kind3 == 'COUNT_COL':
            if c3['mode'] == 4 or not c3.get('has_null'):
                _cells[pi3] = np.bincount(ginv, minlength=G)
            else:
                m3 = _codes_all(col3) != int(c3['V']) - 1
                _cells[pi3] = np.bincount(ginv[m3], minlength=G)
        elif kind3 in ('MIN_DICT', 'MAX_DICT'):
            cc3 = _codes_all(col3)
            if kind3 == 'MAX_DICT':
                if c3.get('has_null'):
                    m3 = cc3 != int(c3['V']) - 1
                    acc = np.full(G, -1, np.int64)
                    np.maximum.at(acc, ginv[m3], cc3[m3])
                else:
                    acc = np.full(G, -1, np.int64)
                    np.maximum.at(acc, ginv, cc3)
            else:
                acc = np.full(G, np.iinfo(np.int64).max, np.int64)
                np.minimum.at(acc, ginv, cc3)
            _cells[pi3] = acc
        elif kind3 == 'COUNT_D':
            cc3 = _codes_all(col3)
            Vc3 = int(cc3.max()) + 1 if cc3.size else 1
            u3 = np.unique(ginv.astype(np.int64) * Vc3 + cc3)
            _cells[pi3] = np.bincount((u3 // Vc3).astype(np.int64), minlength=G)
        elif kind3 in ('MIN', 'MAX'):                    # a sequence column's own values (no NULLs)
            v3 = _codes_all(col3).astype(np.int64)
            if kind3 == 'MIN':
                acc = np.full(G, np.iinfo(np.int64).max, np.int64); np.minimum.at(acc, ginv, v3)
            else:
                acc = np.full(G, np.iinfo(np.int64).min, np.int64); np.maximum.at(acc, ginv, v3)
            _cells[pi3] = acc
        else:                                            # SUM / AVG
            if c3['mode'] == 4:
                v3 = _codes_all(col3)
            else:
                v3 = _num_table(seg, col3)[_codes_all(col3)]
            if kind3 == 'SUM':
                acc = np.zeros(G, np.int64)
                np.add.at(acc, ginv, v3.astype(np.int64))
                _cells[pi3] = acc
            else:
                _cells[pi3] = np.bincount(ginv, weights=v3.astype(np.float64), minlength=G)
    if sel.size > _COL_EMIT_MIN:
        _HITS += 1
        return _emit_columns(seg, spec, keyarr, sel, sel_w, rep, cnt, _cells), [wdb_sql._alias(p) for p in spec['proj']]
    rows_out = []
    for i, gi in enumerate(sel):
        r = int(rep[i]); row = []
        gl = int(sel_w[i])                           # the group's index among the aggregates' groups
        for _pi, kind_or_key in sorted(
                [(pi, ('agg', kind, col)) for pi, kind, col in spec['aggs']] +
                [(pi, ('key', kk, aa)) for (pi, kk), (_x, aa) in zip(spec['keys'], keyarr)]):
            tag = kind_or_key[0]
            if tag == 'agg':
                _t, kind, col = kind_or_key
                if kind == 'COUNT_STAR':
                    row.append(int(cnt[gi]))
                elif kind in ('COUNT_COL', 'COUNT_D'):
                    row.append(int(_cells[_pi][gl]))
                elif kind in ('MIN_DICT', 'MAX_DICT'):
                    k2 = int(_cells[_pi][gl])
                    if (kind == 'MAX_DICT' and k2 < 0) or \
                            (kind == 'MIN_DICT' and k2 == np.iinfo(np.int64).max):
                        row.append(None)             # a group whose every member was null
                    else:
                        row.append(wdb_sql._pyval(seg.fetch(col, k2)))
                elif kind in ('SUM', 'MIN', 'MAX'):
                    row.append(int(_cells[_pi][gl]))
                else:                                # AVG
                    row.append(float(_cells[_pi][gl]) / int(cnt[gi]))
            else:
                _t, kk, aa = kind_or_key
                a = int(aa[r])
                if kk['kind'] == 'scalar':
                    v = wdb_scalar.surrogate(seg, kk['spec'])[1][a]
                    row.append(v.item() if hasattr(v, 'item') else v)
                    continue
                if kk['kind'] == 'case' and a < 0:
                    row.append(kk['default'] if isinstance(kk['default'], str) else str(kk['default']))
                else:
                    src = kk['src']; c = seg.cols[src]
                    row.append(int(a) if c['mode'] == 4 else wdb_sql._pyval(seg.fetch(src, a)))
        rows_out.append(tuple(row))
    _HITS += 1
    return rows_out, [wdb_sql._alias(p) for p in spec['proj']]
