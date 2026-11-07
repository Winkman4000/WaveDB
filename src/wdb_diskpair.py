"""diskpair: disk-only bare SELECT DISTINCT over 2-4 keys or one derived key -- the scan-merge read.

THE COUNT LANE RETIRED (2026-10-23, Jackson). This read was built for SELECT k1..kN, COUNT(*) GROUP BY ...
ORDER BY COUNT(*) DESC LIMIT k, before the census, the heavy split, the flat bucket scan (wdb_bound) and the
fused counts existed. Measured against the read behind it (fused_agg -> wdb_bound) on every count query it
claimed, same counts each time, the read behind was faster on every one (dpvs.py, dpvs2.py). Its
one-thread bill (dpprof.py): the frames' decompress 4%, the minute by a binary search per row 22%, the i64 key
fold 8%, the concatenate 5%, np.unique's sort 61% -- nowhere near the bytes' floor. It now declines every
aggregate; the count queries go to the general reads.

ONE PLAIN COLUMN RETIRED TOO (2026-10-24, Jackson): DISTINCT over a single dict column is 5-15x faster on the
general reads on all 38 enc-3 columns measured, bare and LIMIT 10, the same answers (dpvs3.py). A single DERIVED
key stays here -- the general path has no dictionary-level DISTINCT for it: a minute extracted from a datetime
column, or a column plus a constant, is over 100x faster here (dpvs4.py).

What stays: bare DISTINCT (no GROUP BY / ORDER BY, LIMIT optional) over 2-4 keys, or over one derived key, which
still beats the general path (DuckDB is faster on the measured two-key DISTINCT -- both behind, the dense grid's
next).
The blocked (enc=3) code frames are streamed -- never touching the codes cache -- keys may be blocked dict
columns, a scalar over one, or EXTRACT(part FROM datetime column) keys off the stair column's step rows (a
code->part table over the dictionary, in the column's own time unit: row position IS the timestamp; second, minute,
hour, day, month, quarter, year, dow, isodow, doy -- the parts a dictionary value alone decides; NULLs, overrides
and the other parts decline). Parallel per-thread block scan -> mixed-radix
i64 key fold (declines if the key space overflows i64) -> per-thread np.unique (a dense mark when the key
space is small) -> one merge -> decode.

Lore: the retired count lane, ClickBench pairs: ResponseEndTiming x minute x Robotness 1.362 -> 0.740 s, Referer x
minute x TraficSourceID 2.230 -> 1.005, ResponseStartTiming x minute x SilverlightVersion1 1.471 -> 0.379, minute x
Robotness 0.641 -> 0.252, ResponseEndTiming x Robotness x PageCharset 1.155 -> 0.776. The plain column: CounterID
0.352 -> 0.041 s, FetchTiming 0.341 -> 0.045, WindowName 0.445 -> 0.078, LocalEventTime 1.247 -> 0.338, Referer
27.4 -> 11.0, URL 28.0 -> 10.0. The derived key: extract(minute FROM EventTime) 0.374 s here vs 60.6 s;
ResponseEndTiming + 1 0.457 vs 61.1. What stays: DISTINCT ResponseEndTiming, Robotness 1.145 s vs 2.370; DuckDB
0.314. The stair table for EventTime has 1.4M entries. The part key (2026-11-07) replaced the minute key, which read
the dictionary as seconds: on a millisecond column it found 18 of DuckDB's 60 minutes.
"""
import numpy as np
import zstandard as zstd
import sqlglot.expressions as E
from concurrent.futures import ThreadPoolExecutor
import wdb_sql
import wdb_policies as P
import wdb_kernels as K
import wdb_scalar

_ENABLED = True
_HITS = 0
_SCAN_THREADS = 14
_MAX_KEYS = 4
_DENSE_KS = 1 << 22        # key spaces up to this many slots are marked in place (one int64 array per worker)


def enable():
    global _ENABLED; _ENABLED = True


def disable():
    global _ENABLED; _ENABLED = False


# the EXTRACT parts a part key computes (DuckDB's names; the function forms minute(ts), dayofweek(ts), ... alike)
_PARTS = {'second': 'second', 'minute': 'minute', 'hour': 'hour', 'day': 'day', 'dayofmonth': 'day',
          'month': 'month', 'quarter': 'quarter', 'year': 'year', 'dow': 'dow', 'dayofweek': 'dow',
          'weekday': 'dow', 'isodow': 'isodow', 'doy': 'doy', 'dayofyear': 'doy'}
_PART_FN = {'Second': 'second', 'Minute': 'minute', 'Hour': 'hour', 'Day': 'day', 'DayOfMonth': 'day',
            'Month': 'month', 'Quarter': 'quarter', 'Year': 'year', 'DayOfWeek': 'dow', 'DayOfWeekIso': 'isodow',
            'DayOfYear': 'doy'}


def _part_key(p, seg, col_map):
    """EXTRACT(part FROM col) over a stair datetime column -> derived key spec. The part is a table over the
    dictionary (the stair's code -> value), so any part a dictionary value alone decides is exact; the rest
    (epoch, week, sub-second parts, ...) decline."""
    inner = p.this if isinstance(p, E.Alias) else p
    if isinstance(inner, E.Extract):
        unit = inner.this.name.lower() if hasattr(inner.this, 'name') else str(inner.this).lower()
        part, arg = _PARTS.get(unit), inner.expression
    else:
        part, arg = _PART_FN.get(type(inner).__name__), inner.this
    if part is None or not isinstance(arg, E.Column):
        return None
    col = arg.name
    col = col_map.get(col, col) if col_map else col
    if not P.columns_exist(seg, col):
        return None
    c = seg.cols[col]
    if seg.stairs(col) is None or c.get('dt') != 3 or c.get('has_null') or seg._effective(col) is not None:
        return None                      # NULL has no part value in the table; overrides falsify the steps
    return {'kind': 'part', 'part': part, 'src': col, 'V': _part_table(seg, col, part)[2]}


def _part_table(seg, col, part):
    """(code -> part value - lo, lo, V): the part of every dictionary value, in the column's own time unit;
    memoized on the segment, as wdb_scalar's tables"""
    memo = seg.__dict__.setdefault('_dp_part_memo', {})
    if (col, part) not in memo:
        memo[(col, part)] = _part_values(seg, col, part)
    return memo[(col, part)]


def _part_values(seg, col, part):
    import wdb_scalar
    secs = wdb_scalar.time_table(seg, col).astype('datetime64[s]').astype(np.int64)  # floors, as DuckDB's parts
    days = secs // 86400
    if part == 'second':
        out = secs % 60
    elif part == 'minute':
        out = (secs // 60) % 60
    elif part == 'hour':
        out = (secs // 3600) % 24
    elif part == 'dow':
        out = (days + 4) % 7                         # 1970-01-01 was a Thursday; Sunday is 0
    elif part == 'isodow':
        out = (days + 3) % 7 + 1                     # Monday 1 .. Sunday 7
    else:
        D = days.astype('datetime64[D]')
        M = D.astype('datetime64[M]')
        mo = M.astype(np.int64) % 12 + 1
        if part == 'day':
            out = (D - M.astype('datetime64[D]')).astype(np.int64) + 1
        elif part == 'month':
            out = mo
        elif part == 'quarter':
            out = (mo - 1) // 3 + 1
        elif part == 'doy':
            out = (D - D.astype('datetime64[Y]').astype('datetime64[D]')).astype(np.int64) + 1
        else:
            out = D.astype('datetime64[Y]').astype(np.int64) + 1970
    lo = int(out.min()) if out.size else 0
    hi = int(out.max()) if out.size else 0
    return (out - lo).astype(np.int64), lo, hi - lo + 1


def detect(seg, tree, col_map):
    # enc-5 (patched buckets) columns: this path's direct block machinery predates the
    # species; decline so the _raw_codes route (exact, 8-lane) serves until v2 learns nibbles
    for _c5 in tree.find_all(E.Column):
        _n5 = (col_map or {}).get(_c5.name, _c5.name) if col_map else _c5.name
        if _n5 in seg.cols and seg.cols[_n5].get('code_enc') in (5, 6):
            return None
    if not _ENABLED:                    return None
    if not tree.args.get('distinct'):   return None     # the count lane is retired (docstring)
    if not P.no_joins(tree):            return None
    if not P.no_having(tree):           return None
    if not P.no_deleted_rows(seg):      return None
    if tree.args.get('where') is not None:
        return None
    if tree.args.get('group') is not None or tree.args.get('order') is not None:
        return None                      # bare DISTINCT only (no group/order); LIMIT optional
    proj = tree.expressions
    keys = []
    for pi, p in enumerate(proj):
        if wdb_sql._agg_kind(p) is not None:
            return None                  # keys only, no aggregates
        mk = _part_key(p, seg, col_map)
        if mk is not None:
            keys.append((pi, mk))
            continue
        sp = wdb_scalar.parse(seg, p, col_map)
        if sp is not None:
            c2 = seg.cols[sp['col']]
            if c2.get('code_enc') == 3 or seg.stairs(sp['col']) is not None:
                keys.append((pi, {'kind': 'scalar', 'spec': sp, 'src': sp['col'],
                                  'V': int(c2['V'])}))      # surrogate V <= dict V (safe fold bound)
                continue
        nm = wdb_sql._proj_colname(p)
        if nm is None:
            return None
        col = col_map.get(nm, nm) if col_map else nm
        if not P.columns_exist(seg, col):   return None
        if seg._effective(col) is not None: return None
        c = seg.cols[col]
        if c.get('mode') not in (0, 1, 2) or c.get('code_enc') != 3:
            return None
        keys.append((pi, {'kind': 'col', 'src': col, 'V': int(c['V'])}))
    if not (1 <= len(keys) <= _MAX_KEYS):
        return None
    if len(keys) == 1 and keys[0][1]['kind'] == 'col':
        return None                      # one plain column: the general reads (docstring, 2026-10-24)
    span = 1
    for _pi, k in keys:
        span *= k['V']
        if span > (1 << 62):
            return None                 # composite would overflow the i64 key fold
    lim = wdb_sql._limit(tree)
    return {'keys': keys, 'lim': None if lim is None else int(lim),
            'off': int(wdb_sql._offset(tree) or 0), 'proj': proj}


def _block_geometry(seg, keys):
    """(NB, BR): all enc=3 columns share BLOCK_ROWS, so any blocked column supplies the
    geometry -- a key column first (col or scalar over one), else any blocked column in the
    segment (a pure stair-derived key still needs block bounds for the parallel split)."""
    for _pi, k in keys:
        if k['kind'] in ('col', 'scalar'):
            c = seg.cols.get(k['src'])
            if c is not None and c.get('code_enc') == 3:
                return c['boffs'].size - 1, int(c['BR'])
    for c in seg.cols.values():
        if c.get('code_enc') == 3:
            return c['boffs'].size - 1, int(c['BR'])
    return 0, 0


def answers(seg, spec):
    """whether execute will answer this spec: a block geometry exists and THE GRID LAW holds -- every blocked
    (enc=3) key column shares the segment's block geometry; an encode where candidates won at different frame
    sizes (the code-stream level can change which tag wins) declines by name instead of broadcasting two grids
    together. wdb_counted asks this before taking diskpair's shapes (2026-10-24).

    Lore: g-exyear2, 2026-09-12: EventTime at 524,288 rows, SearchEngineID at 65,536; IsMobile at 524,288 beside
    RegionID's 65,536."""
    keys = spec['keys']
    NB, BR = _block_geometry(seg, keys)
    if NB == 0:
        return False
    for _pi, k in keys:
        if k['kind'] in ('col', 'scalar'):
            c = seg.cols.get(k['src'])
            if c is not None and c.get('code_enc') == 3 and int(c['BR']) != BR:
                return False
    return True


def execute(seg, spec):
    global _HITS
    keys = spec['keys']
    if not answers(seg, spec):
        return None
    NB, BR = _block_geometry(seg, keys)
    N = int(seg.N)
    # per-key block readers
    readers = []
    for _pi, k in keys:
        if k['kind'] == 'part':
            readers.append(('part', (_part_table(seg, k['src'], k['part'])[0], seg.stairs(k['src']))))
        elif k['kind'] == 'scalar':
            surr, svals = wdb_scalar.surrogate(seg, k['spec'])
            k['V'] = len(svals)                      # true surrogate cardinality for the fold
            c = seg.cols[k['src']]
            if c.get('code_enc') == 3:
                readers.append(('scalar', (surr, c['boffs'], c['cstart'],
                                           {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']])))
            else:
                readers.append(('scalar_stair', (surr, seg.stairs(k['src']))))
        else:
            c = seg.cols[k['src']]
            readers.append(('col', (c['boffs'], c['cstart'],
                                    {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']])))
    radix = [k['V'] for _pi, k in keys]
    KS = 1
    for _r9 in radix:
        KS *= int(_r9)
    buf = seg.buf

    def work(js):
        dz = zstd.ZstdDecompressor()
        exc = []
        for j in js:
            a, b = j * BR, min(N, (j + 1) * BR)
            cols = []
            for kind, meta in readers:
                if kind == 'part':
                    tab, st = meta
                    codes = np.searchsorted(st, np.arange(a, b), side='right')
                    cols.append(tab[codes])
                elif kind == 'scalar':
                    surr, bo2, base2, wdt2 = meta
                    codes = np.frombuffer(
                        dz.decompress(buf[base2 + int(bo2[j]):base2 + int(bo2[j + 1])].tobytes()),
                        dtype=wdt2)
                    cols.append(surr[codes])
                elif kind == 'scalar_stair':
                    surr, st2 = meta
                    codes = np.searchsorted(st2, np.arange(a, b), side='right')
                    cols.append(surr[codes])
                else:
                    bo, base, wdt = meta
                    cols.append(np.frombuffer(
                        dz.decompress(buf[base + int(bo[j]):base + int(bo[j + 1])].tobytes()),
                        dtype=wdt))
            key = cols[0].astype(np.int64)
            for ci in range(1, len(cols)):
                key = key * radix[ci] + cols[ci]
            exc.append(key)
        ek = np.concatenate(exc) if exc else np.empty(0, np.int64)
        if 0 < KS <= _DENSE_KS and ek.size > KS // 4:
            # THE DENSE COUNT (2026-10-05): a key space this small is marked in place, no sort -- np.unique
            # sorted each worker's keys
            # lore: ~6M keys a worker; the sort was a third of GROUP BY SUBSTR(URL, 1, 20) once the keys were cheap
            acc = np.zeros(KS, np.int64)
            K.count_into(ek, acc)
            return np.flatnonzero(acc)
        return np.unique(ek)

    W = min(_SCAN_THREADS, NB) or 1
    with ThreadPoolExecutor(W) as ex:
        parts = list(ex.map(work, np.array_split(np.arange(NB), W)))
    uniq = np.unique(np.concatenate(parts))
    if spec['off'] or spec['lim'] is not None:
        end = None if spec['lim'] is None else spec['off'] + spec['lim']
        uniq = uniq[spec['off']: end]
    # decompose composites into per-key code columns
    comp = uniq
    keycodes = []
    for ci in range(len(keys) - 1, -1, -1):
        keycodes.append(comp % radix[ci]); comp = comp // radix[ci]
    keycodes.reverse()
    # decode: point fetches when small, one bulk dict gather per column when large
    BULK = uniq.size > 10000
    colvals = []
    for (pi, k), codes in zip(keys, keycodes):
        if k['kind'] == 'part':
            colvals.append(codes.astype(np.int64) + _part_table(seg, k['src'], k['part'])[1])
        elif k['kind'] == 'scalar':
            svals = wdb_scalar.surrogate(seg, k['spec'])[1]
            colvals.append([svals[int(c)] for c in codes] if not BULK
                           else np.array(svals, dtype=object)[codes])
        elif BULK:
            dvals = np.array(seg._typed_dict(k['src']), dtype=object)
            colvals.append(dvals[codes])
        else:
            colvals.append([wdb_sql._pyval(seg.fetch(k['src'], int(c))) for c in codes])
    out = []
    for ri in range(uniq.size):
        row = [None] * len(spec['proj'])
        for (pi, k), cv in zip(keys, colvals):
            v = cv[ri]
            if isinstance(v, (bytes, bytearray)):
                v = v.decode('utf-8', 'replace')
            elif hasattr(v, 'item'):
                v = v.item()
            row[pi] = v
        out.append(tuple(row))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
