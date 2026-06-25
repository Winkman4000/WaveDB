"""Shared value normalization + hashing for the ClickBench board and the correctness verifier.

The board's old check md5-hashed repr(round(v,3)) per cell, which false-failed on benign differences:
int SUM (5) vs float SUM (5.0), DATE-as-string ('2013-07-01') vs datetime.date, and last-ULP float
drift -- all read as wrong despite being correct. This normalizes so only real differences show:
  - numbers: int (any size) and integer-valued float collapse to ('i', exact_int); non-integer
    floats round to 3 dp as ('f', x). Big ints stay exact (no float64 collapse that could hide a diff).
  - dates/datetimes and their string renderings collapse to a canonical 'YYYY-MM-DD[ HH:MM:SS]'.
  - bytes decode to str.
Hashes are order-independent (GROUP BY output order is unspecified): a small-result sorted md5
(limit_hash) and a commutative md5-sum fingerprint for large full sets (full_fp), both stable across
processes (md5, not Python hash())."""
import datetime, re, hashlib
_TS = re.compile(r'^(\d{4}-\d\d-\d\d)(?:[ T](\d\d:\d\d:\d\d))?(?:\.\d+)?$')

def _cn(v):                                   # canonical number
    if isinstance(v, bool): return ('b', v)
    if isinstance(v, int):  return ('i', v)                       # exact, any magnitude
    f = float(v)
    if f != f:              return ('f', 'nan')
    if f.is_integer() and abs(f) < 2**53: return ('i', int(f))    # 5.0 -> ('i',5) == int 5
    return ('f', round(f, 3))

def norm_cell(v):
    if v is None: return None
    if isinstance(v, (bool, int, float)): return _cn(v)
    if isinstance(v, (bytes, bytearray)):
        try: return v.decode('utf-8', 'replace')
        except Exception: return repr(v)
    if isinstance(v, datetime.datetime):
        return v.strftime('%Y-%m-%d') if (v.hour, v.minute, v.second, v.microsecond) == (0,0,0,0) \
               else v.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(v, datetime.date):
        return v.strftime('%Y-%m-%d')
    if isinstance(v, str):
        m = _TS.match(v)
        if m:
            return m.group(1) if (m.group(2) in (None, '00:00:00')) else f"{m.group(1)} {m.group(2)}"
        return v
    return str(v)

def norm_row(r): return tuple(norm_cell(v) for v in r)

def limit_hash(rows):                         # order-insensitive multiset hash (small results)
    h = hashlib.md5()
    for r in sorted((norm_row(x) for x in rows), key=repr): h.update(repr(r).encode())
    return h.hexdigest()

_M = (1 << 64) - 1
def full_fp(rows):                            # order-independent commutative fingerprint for large sets.
    # Uses the built-in hash() for speed -> ONLY comparable when both sides are hashed in the SAME
    # process (the verifier does; it never crosses a process boundary). The board uses limit_hash, not
    # this. norm_row returns a hashable tuple of canonical cells, so hash() is exact-on-equality.
    acc = 0; n = 0
    for r in rows:
        acc = (acc + (hash(norm_row(r)) & _M)) & _M; n += 1
    return (n, acc)
