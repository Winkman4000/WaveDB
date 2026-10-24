"""wdb_complement: THE COMPLEMENT COUNT (2026-10-24, Q1; Jackson's exceptions idea at its simplest).

    COUNT(*) WHERE k <> v  =  N  -  (rows where k = v)  -  (rows where k IS NULL)       (NULL <> v is not TRUE)
    COUNT(*) WHERE k = v   =  (rows where k = v)

The rows where k = v come from whichever side is cheap, by a number the column carries -- its value count V:
  V <= 2^20, or a staircase column: k's census (raw_census, from the encoding where it can be) -- 16-44 ms
             whatever v's share, a staircase's from the gaps between its steps;
  past it:   the equality scan of v (wherescan's per-encoding scanners) -- the census board itself grows
             (0.12-0.25 s) while a large dictionary's literal is usually rare.
Measured on the Q1 board's failing variants (q1proto.py, hot, every count DuckDB's, both ways): Robotness today
0.525 s (wherescan listed the ~33M survivors to count them) -> census 0.016; ResponseEndTiming 0.479 -> 0.021;
UTMSource 0.244 -> 0.016; LocalEventTime (1.43M values) 0.621 -> scan 0.068; ClientEventTime 0.832 -> 0.053;
RefererHash 0.355 (fused_agg) -> 0.172; WatchID 0.460 (fused_agg) -> 0.037.
One = / <> conjunct on a dictionary column; COUNT(*) projections only. WDB_COMPLEMENT=0 off."""
import os
import numpy as np
import sqlglot.expressions as E
import wdb_sql
import wdb_policies as P

_ENABLED = [os.environ.get('WDB_COMPLEMENT', '1') != '0']
_CENSUS_V = 1 << 20
_HITS = 0


def detect(seg, tree, col_map):
    if not _ENABLED[0] or not isinstance(tree, E.Select):
        return None
    if not P.no_joins(tree) or not P.no_deleted_rows(seg):
        return None
    if tree.args.get('group') is not None or tree.args.get('having') is not None or tree.args.get('distinct'):
        return None
    if tree.args.get('order') is not None or wdb_sql._limit(tree) is not None or wdb_sql._offset(tree):
        return None
    proj = tree.expressions
    if not proj:
        return None
    for p in proj:
        ak = wdb_sql._agg_kind(p)
        if ak is None or ak[0] != 'COUNT_STAR':
            return None
    where = tree.args.get('where')
    if where is None:
        return None
    import wdb_wherescan as WS
    cl = WS._col_lit(where.this)
    if cl is None or cl[2] not in ('=', '<>') or cl[1] is None:
        return None
    col = col_map.get(cl[0], cl[0]) if col_map else cl[0]
    if not P.columns_exist(seg, col) or seg._effective(col) is not None:
        return None
    c = seg.cols[col]
    if c.get('mode') not in (0, 1, 2) or not c.get('V'):
        return None
    return {'col': col, 'val': cl[1], 'op': cl[2], 'proj': proj}


def execute(seg, spec):
    global _HITS
    import wdb_wherescan as WS
    col = spec['col']
    c = seg.cols[col]
    V = int(c['V'])
    N = int(seg.N)
    code = WS._code_of(seg, col, spec['val'])
    nullc = V - 1 if c.get('has_null') else None
    if code is not None and nullc is not None and int(code) == nullc:
        return None                                  # the literal decoded to the NULL code: not this read's to judge
    if V <= _CENSUS_V or seg.stairs(col) is not None:
        # a staircase's census comes from the gaps between its steps, whatever V (EventTime, 1.43M values: the
        # equality scan decoded the column, 1.6 s against fused_agg's 0.063 -- the Q1 board, 2026-10-24)
        cen = np.asarray(seg.raw_census(col, V))
        eq = int(cen[code]) if code is not None else 0
        nn = int(cen[nullc]) if nullc is not None else 0
    else:
        eq = int(WS._scan_eq(seg, col, int(code), 0, N).size) if code is not None else 0
        nn = (int(WS._scan_eq(seg, col, int(nullc), 0, N).size)
              if (nullc is not None and spec['op'] == '<>') else 0)
    n = eq if spec['op'] == '=' else N - eq - nn
    _HITS += 1
    return [tuple(n for _p in spec['proj'])], [wdb_sql._alias(p) for p in spec['proj']]
