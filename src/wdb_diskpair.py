"""diskpair: disk-only bare SELECT DISTINCT over 2-4 keys or one derived key -- the scan-merge read.

THE COUNT LANE RETIRED (2026-10-23, Jackson). This read was built for SELECT k1..kN, COUNT(*) GROUP BY ...
ORDER BY COUNT(*) DESC LIMIT k, before the census, the heavy split, the flat bucket scan (wdb_bound) and the
fused counts existed. Measured against the read behind it (fused_agg -> wdb_bound) on every count query it
claimed, same counts each time: ResponseEndTiming x minute x Robotness 1.362 -> 0.740 s, Referer x minute x
TraficSourceID 2.230 -> 1.005, ResponseStartTiming x minute x SilverlightVersion1 1.471 -> 0.379, minute x
Robotness 0.641 -> 0.252, ResponseEndTiming x Robotness x PageCharset 1.155 -> 0.776 (dpvs.py, dpvs2.py). Its
one-thread bill (dpprof.py): the frames' decompress 4%, the minute by a binary search per row 22%, the i64 key
fold 8%, the concatenate 5%, np.unique's sort 61% -- nowhere near the bytes' floor. It now declines every
aggregate; the count queries go to the general reads.

ONE PLAIN COLUMN RETIRED TOO (2026-10-24, Jackson): DISTINCT over a single dict column is 5-15x faster on the
general reads on all 38 enc-3 columns measured, bare and LIMIT 10, the same answers (dpvs3.py): CounterID 0.352 ->
0.041 s, FetchTiming 0.341 -> 0.045, WindowName 0.445 -> 0.078, LocalEventTime 1.247 -> 0.338, Referer 27.4 ->
11.0, URL 28.0 -> 10.0. A single DERIVED key stays here -- the general path has no dictionary-level DISTINCT for
it: extract(minute FROM EventTime) 0.374 s here vs 60.6 s; ResponseEndTiming + 1 0.457 vs 61.1 (dpvs4.py).

What stays: bare DISTINCT (no GROUP BY / ORDER BY, LIMIT optional) over 2-4 keys, or over one derived key, which
still beats the general path (DISTINCT ResponseEndTiming, Robotness 1.145 s vs 2.370; DuckDB 0.314 -- both
behind, the dense grid's next).
The blocked (enc=3) code frames are streamed -- never touching the codes cache -- keys may be blocked dict
columns, a scalar over one, or ONE derived minute-of(EventTime) key off the stair column's step rows (the
1.4M-entry code->minute table: row position IS the timestamp). Parallel per-thread block scan -> mixed-radix
i64 key fold (declines if the key space overflows i64) -> per-thread np.unique (a dense mark when the key
space is small) -> one merge -> decode.
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


def _minute_key(p, seg, col_map):
    """extract(minute FROM col) over a stair datetime column -> derived key spec."""
    inner = p.this if isinstance(p, E.Alias) else p
    if not isinstance(inner, E.Extract):
        return None
    unit = inner.this.name.lower() if hasattr(inner.this, 'name') else str(inner.this).lower()
    if unit != 'minute' or not isinstance(inner.expression, E.Column):
        return None
    col = inner.expression.name
    col = col_map.get(col, col) if col_map else col
    if not P.columns_exist(seg, col):
        return None
    if seg.stairs(col) is None or seg.cols[col].get('dt') != 3:
        return None
    return {'kind': 'minute', 'src': col, 'V': 60}


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
        mk = _minute_key(p, seg, col_map)
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
    if sum(1 for _pi, k in keys if k['kind'] == 'minute') > 1:
        return None
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


def execute(seg, spec):
    global _HITS
    keys = spec['keys']
    NB, BR = _block_geometry(seg, keys)
    if NB == 0:
        return None
    # THE GRID LAW: every blocked (enc=3) key column must share the segment's block geometry;
    # an encode where candidates won at different frame sizes (EventTime at 524,288 rows,
    # SearchEngineID at 65,536 -- the code-stream level changed which tag won) declines by
    # name instead of broadcasting two grids together (g-exyear2, 2026-09-12)
    for _pi, k in keys:
        if k['kind'] in ('col', 'scalar'):
            c = seg.cols.get(k['src'])
            if c is not None and c.get('code_enc') == 3 and int(c['BR']) != BR:
                return None
    N = int(seg.N)
    # per-key block readers
    readers = []
    minute_tab = None
    for _pi, k in keys:
        if k['kind'] == 'minute':
            st = seg.stairs(k['src'])
            secs = np.asarray(seg._dict_ints(seg.cols[k['src']]), dtype=np.int64)
            minute_tab = ((secs // 60) % 60).astype(np.int64)
            readers.append(('minute', st))
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
                if kind == 'minute':
                    codes = np.searchsorted(meta, np.arange(a, b), side='right')
                    cols.append(minute_tab[codes])
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
            # sorted each worker's ~6M keys (a third of GROUP BY SUBSTR(URL, 1, 20) once the keys were cheap)
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
        if k['kind'] == 'minute':
            colvals.append(codes.astype(np.int64))
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
