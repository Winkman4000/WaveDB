"""wdb_lens -- STRING LENGTHS AS LOAD DATA (Jackson, 2026-09-25).

Two kinds, both written by the encoder after the segment is sealed, counted in load time, and read
only when their header still matches the segment (rows, distinct values, segment file size) -- a
rewritten segment silently falls back to the dictionary walk.

  DICTIONARY LENGTHS (the default, every front-coded text column): each dictionary entry's CHARACTER
  length, once, zstd. length() over a text column reads this instead of walking the front-coded
  dictionary for UTF-8 continuation bytes.            <seg>.clen.<col>

  ROW LENGTHS (the operator's flag, bin/wdb load --row-lengths C,..): each ROW's character length in
  row order, zstd per 65,536-row block with a block table -- the offsets are the answer, a length
  aggregate never needs the row's dictionary number.   <seg>.rlen.<col>
  Its size is bounded by the column's own number stream (a length is a function of the value).
"""
import os, struct
import numpy as np

FR = 65536
_M_DICT = b'WCLN'
_M_ROW = b'WRLN'


def _dt(mx):
    return np.uint16 if mx < 65536 else np.uint32


def dict_path(seg_path, col):
    return '%s.clen.%s' % (seg_path, col)


def row_path(seg_path, col):
    return '%s.rlen.%s' % (seg_path, col)


def _head_ok(seg, N, V, size):
    try:
        return int(N) == int(seg.N) and int(size) == os.path.getsize(seg.path)
    except Exception:
        return False


def write_for_segment(seg_path, row_cols=(), verbose=False):
    """Called by the encoder once the segment file is final. Returns bytes written."""
    import zstandard as zs
    from wdb_engine import Segment
    seg = Segment(seg_path)
    size = os.path.getsize(seg_path)
    cz = zs.ZstdCompressor(level=9)
    total = 0
    for col, c in seg.cols.items():
        if c.get('dt') != 1 or c.get('mode') != 1 or c.get('has_null'):
            continue                                   # long text: the front-coded dictionaries
        cl = seg.dict_charlens(col)
        if cl is None:
            continue
        cl = np.asarray(cl)
        dt = _dt(int(cl.max()) if cl.size else 0)
        body = cz.compress(cl.astype(dt).tobytes())
        with open(dict_path(seg_path, col) + '.tmp', 'wb') as f:
            f.write(_M_DICT + struct.pack('<qqqB', int(seg.N), int(c['V']), size, np.dtype(dt).itemsize) + body)
        os.replace(dict_path(seg_path, col) + '.tmp', dict_path(seg_path, col))
        total += len(body) + 29
        if verbose:
            print('  dictionary lengths %s: %d entries, %.1f MB' % (col, cl.size, (len(body) + 29) / 1e6), flush=True)
    for col in row_cols:
        c = seg.cols.get(col)
        if c is None or c.get('dt') != 1 or c.get('mode') not in (0, 1) or c.get('has_null'):
            if verbose:
                print('  row lengths %s: not a non-null text dictionary column, skipped' % col, flush=True)
            continue
        cl = seg.dict_charlens(col)
        if cl is None:
            continue
        cl = np.asarray(cl)
        dt = _dt(int(cl.max()) if cl.size else 0)
        L = cl.astype(dt)[np.asarray(seg._raw_codes(col))]
        blocks = [cz.compress(L[i:i + FR].tobytes()) for i in range(0, L.size, FR)]
        off = np.zeros(len(blocks) + 1, np.int64)
        np.cumsum([len(b) for b in blocks], out=off[1:])
        with open(row_path(seg_path, col) + '.tmp', 'wb') as f:
            f.write(_M_ROW + struct.pack('<qqqBqq', int(seg.N), int(c['V']), size, np.dtype(dt).itemsize, FR, len(blocks)))
            f.write(off.tobytes())
            for b in blocks:
                f.write(b)
        os.replace(row_path(seg_path, col) + '.tmp', row_path(seg_path, col))
        total += int(off[-1]) + off.nbytes + 45
        if verbose:
            print('  row lengths %s: %d rows, %.1f MB' % (col, L.size, (int(off[-1]) + off.nbytes) / 1e6), flush=True)
    return total


def dict_lens(seg, col):
    """The stored character length of every dictionary entry of col, or None (absent / stale)."""
    p = dict_path(getattr(seg, 'path', ''), col)
    try:
        with open(p, 'rb') as f:
            head = f.read(29)
            if head[:4] != _M_DICT:
                return None
            N, V, size, isz = struct.unpack('<qqqB', head[4:])
            c = seg.cols.get(col)
            if c is None or int(V) != int(c['V']) or not _head_ok(seg, N, V, size):
                return None
            import zstandard as zs
            a = np.frombuffer(zs.ZstdDecompressor().decompress(f.read()), {2: np.uint16, 4: np.uint32}[isz])
    except (OSError, KeyError, ValueError):
        return None
    return a.astype(np.int64) if a.size == int(V) else None


def has_row_lens(seg, col):
    p = row_path(getattr(seg, 'path', ''), col)
    if not os.path.exists(p):
        return False
    try:
        with open(p, 'rb') as f:
            head = f.read(45)
        N, V, size, isz, fr, nb = struct.unpack('<qqqBqq', head[4:])
        c = seg.cols.get(col)
        return head[:4] == _M_ROW and c is not None and int(V) == int(c['V']) and _head_ok(seg, N, V, size)
    except Exception:
        return False


def row_lens(seg, col, lanes=16):
    """Every row's character length of col, in row order (blocks read and inflated on `lanes` threads,
    neighbouring blocks in one read). None when absent or stale."""
    if not has_row_lens(seg, col):
        return None
    import zstandard as zs
    from concurrent.futures import ThreadPoolExecutor
    p = row_path(seg.path, col)
    fd = os.open(p, os.O_RDONLY)
    try:
        head = os.pread(fd, 45, 0)
        N, V, size, isz, fr, nb = struct.unpack('<qqqBqq', head[4:])
        dt = {2: np.uint16, 4: np.uint32}[isz]
        off = np.frombuffer(os.pread(fd, 8 * (nb + 1), 45), np.int64)
        base = 45 + 8 * (nb + 1)
        out = np.empty(int(N), dt)
        step = max(1, nb // (2 * lanes))
        runs = [(a, min(nb, a + step)) for a in range(0, nb, step)]

        def task(ab):
            a, b = ab
            raw = os.pread(fd, int(off[b] - off[a]), base + int(off[a]))
            dz = zs.ZstdDecompressor()
            for j in range(a, b):
                x = np.frombuffer(dz.decompress(raw[int(off[j] - off[a]):int(off[j + 1] - off[a])]), dt)
                out[j * fr:j * fr + x.size] = x
        with ThreadPoolExecutor(min(lanes, len(runs))) as ex:
            list(ex.map(task, runs))
    finally:
        os.close(fd)
    return out


def clean(seg):
    """The row lengths describe the rows AS ENCODED: a segment carrying tombstones or overrides
    (DELETE/UPDATE) must not be answered from them."""
    try:
        if seg.presence_mask() is not None:
            return False
        if hasattr(seg, '_overrides_any') and seg._overrides_any():
            return False
        import wdb_override
        return not bool(wdb_override.load(seg.path))
    except Exception:
        return False


try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def row_pour(kc, L, K, excl, T):
        """per key: sum of lengths and count of rows (rows of length 0 -- the empty string -- left out
        when excl), T partial tables merged at the end"""
        n = kc.size
        S = np.zeros((T, K), np.int64)
        C = np.zeros((T, K), np.int64)
        per = (n + T - 1) // T
        for t in prange(T):
            for i in range(t * per, min(n, (t + 1) * per)):
                l = L[i]
                if excl and l == 0:
                    continue
                S[t, kc[i]] += l
                C[t, kc[i]] += 1
        return S.sum(0), C.sum(0)
except Exception:                                     # pragma: no cover
    def row_pour(kc, L, K, excl, T):
        L = np.asarray(L, np.int64); m = (L > 0) if excl else np.ones(L.size, bool)
        return (np.bincount(kc[m], weights=L[m], minlength=K).astype(np.int64),
                np.bincount(kc[m], minlength=K).astype(np.int64))
