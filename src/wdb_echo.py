"""THE ECHO, DISK-ONLY (2026-10-21, Q19's board): SELECT c FROM t WHERE c = lit [LIMIT k] [OFFSET o].

The projection is the filter column, so every row the query emits IS the literal: the answer is the literal's value,
repeated as many times as its code holds rows. No row is read -- only one code's count.

wdb_gbcount's echo (inside the general scan) took its count from the gbc shelf, which is never born with sidecars off,
so on the disk-only board the shape fell through to the general SQL path: IsRefresh 0.29 s, Income 0.19 s, Age 0.15 s
against DuckDB's 0.03 s, and the binary columns (JavaEnable, GoodEvent, IsMobile, SocialNetwork) past the 10 s cap.

Here the count comes from the stored codes, by whichever road needs no row decode:
  - a staircase (enc 2, or a column with steps): the gap between the code's two steps, O(V) (raw_census);
  - a dictionary of at most 64K codes: the census of the stored codes (Segment.raw_census -- from the dress where it
    can, enc 5 by nibbles, enc 10 by run lengths; else the codes counted on per-thread boards). Measured against the
    low count of codes 0..code (Segment.low_counts), the census was the faster on every column of the board, cold
    and hot: IsRefresh 1.9 against 60.7 ms, IsMobile 20 against 147, Age 11 against 45.
A large dictionary with neither is declined at detect: wherescan's equality scan serves it, as before.
Exact: no deletes (checked), no UPDATE overrides (checked). WDB_ECHO=0 declines."""
import os
import wdb_sql
import wdb_policies as P

_ON = [os.environ.get('WDB_ECHO', '1') != '0']
_LOW_V = 1 << 16
_HITS = 0


def _how(seg, col):
    c = seg.cols[col]
    if c.get('mode') not in (0, 1, 2):
        return None
    if c.get('code_enc', 0) == 2 or c.get('_steps') is not None:
        return 'stairs'
    if int(c['V']) <= _LOW_V:
        return 'low'
    return None


def detect(seg, tree, col_map):
    if not _ON[0]:
        return None
    import wdb_gbcount
    sp = wdb_gbcount._echo_detect(seg, tree, col_map)
    if sp is None:
        return None
    col = sp['col']
    if seg._effective(col) is not None or seg.presence_mask() is not None:
        return None                                  # overrides / deletes falsify the stored codes' counts
    how = _how(seg, col)
    if how is None:
        return None
    if sp['lim'] is not None and sp['lim'] < 0:
        return None
    return dict(sp, how=how, off=int(wdb_sql._offset(tree) or 0))


def execute(seg, spec):
    global _HITS
    import wdb_wherescan as WS
    col = spec['col']
    names = [wdb_sql._alias(p) for p in spec['proj']]
    kc = WS._code_of(seg, col, spec['lit'])
    if kc is None:
        _HITS += 1
        return [], names                             # not in the dictionary: no row holds it
    kc = int(kc)
    cn = seg.raw_census(col, int(seg.cols[col]['V']))
    cnt = int(cn[kc]) if kc < len(cn) else 0
    n = max(0, cnt - spec['off'])
    if spec['lim'] is not None:
        n = min(n, int(spec['lim']))
    v = wdb_sql._pyval(seg.fetch(col, kc))
    if isinstance(v, (bytes, bytearray)):
        v = v.decode('utf-8', 'replace')
    _HITS += 1
    return [(v,)] * n, names
