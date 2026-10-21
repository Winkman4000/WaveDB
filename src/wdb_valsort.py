"""
wdb_valsort -- value-sorted projection top-K, read from the dictionary (general since 2026-10-21, Q24-Q26 board).

Answers   SELECT C FROM t [WHERE <a predicate on C>] ORDER BY C [ASC|DESC] LIMIT k [OFFSET o]
for any dictionary column C (numbers, times, text). Every dictionary is stored in VALUE ORDER (encode:
np.unique / factorize(sort=True); measured on all 35 column profiles of hits), so VALUE order == CODE order,
and every code has at least one row. The answer is a walk of the dictionary from the front (or, DESC, from
the back): skip the codes the filter rejects (wdb_keepflag: a keep flag over the codes), emit each kept value
as many times as it occurs, stop at k. The first k kept codes are always enough (each holds >= 1 row), so
counts are needed for those codes only: low in the dictionary, Segment.low_counts (codes below a bound, one
parallel counting pass, nothing kept); high (DESC, or a filter like > x), one pass counting just that run. Only the values
emitted ever decode. NULLs sort last both ways (DuckDB's NULLS LAST); with no WHERE they fill after the values.

Before 2026-10-21 this read took only text columns filtered by `C <> ''`; every number and time column, and
any other filter, fell to the general top-k door, which sorts the survivors (Age: 7.0 s, of it 6.1 s one
argpartition over 95M values, against DuckDB's 0.036 s).

detect (fail-closed): single table; projection is exactly one bare column C (mode 0/1/2, no overrides);
ORDER BY C alone; LIMIT present; WHERE absent or a predicate over C alone that wdb_keepflag can judge;
no GROUP BY / HAVING / DISTINCT / JOIN; no deleted rows. Anything else -> None (caller scans).
"""
import numpy as np
import wdb_qmem
import wdb_sql
import wdb_gbcount
import wdb_policies as P
import wdb_keepflag as KF
E = wdb_sql.E

_HITS = 0
_CACHE = wdb_qmem.register({})   # (seg.path, col, N) -> (dup_codes_asc, dup_counts) | None


def _order_dir(tree, colname):
    """'asc' / 'desc' when the ORDER BY is exactly this column, else None"""
    order = tree.args.get('order')
    if order is None or len(order.expressions) != 1:
        return None
    first = order.expressions[0]
    if not isinstance(first, E.Ordered) or first.args.get('nulls_first'):
        return None
    tgt = first.this
    if not (isinstance(tgt, E.Column) and tgt.name == colname):
        return None
    return 'desc' if first.args.get('desc') else 'asc'


def detect(seg, tree, col_map):
    if not P.no_joins(tree):           return None
    if not P.no_select_distinct(tree): return None
    if not P.no_having(tree):          return None
    if not P.has_limit(tree):          return None
    if tree.args.get('group') is not None: return None
    proj = tree.expressions
    if len(proj) != 1:                 return None
    if wdb_sql._agg_kind(proj[0]) is not None: return None
    colname = wdb_sql._proj_colname(proj[0])
    if colname is None:                return None
    d = _order_dir(tree, colname)
    if d is None:                      return None
    col = col_map.get(colname, colname) if col_map else colname
    if not P.columns_exist(seg, col):  return None
    if not P.not_positional(seg, col): return None
    if not P.no_deleted_rows(seg):     return None
    c = seg.cols[col]
    if c.get('mode') not in (0, 1, 2) or seg._overrides(col) is not None:
        return None
    w = tree.args.get('where')
    fl = rs = None
    if w is not None:
        if KF.column_of(w.this, col_map) != col:
            return None                                  # a filter on another column needs the rows: not here
        rs = KF.runs(seg, w.this, col)                   # value ranges: the kept codes as runs, no V-sized table
        if rs is None:
            fl = KF.flag(seg, w.this, col)
            if fl is None:
                return None
    try:
        off = int(wdb_sql._offset(tree) or 0)
    except Exception:
        return None
    return {'col': col, 'lim': wdb_sql._limit(tree), 'off': off, 'desc': d == 'desc', 'flag': fl, 'runs': rs,
            'proj': proj, 'V': c['V'], 'has_null': c['has_null']}


_SMALL_V = 1 << 16       # codes below this bound are counted by the parallel low-count pass


_HEAD = 1 << 18           # the head window: rows read from the top of the file before any counting pass


def _head_first(seg, col, code, need):
    """need, when the file's first window alone holds `need` rows of `code`; else None"""
    N = int(seg.N)
    hc = np.asarray(seg.codes_at(col, np.arange(0, min(N, _HEAD), dtype=np.int64)), np.int64)
    return need if int(np.count_nonzero(hc == code)) >= need else None


def _counts_run(seg, col, lo, hi):
    """rows per code for codes lo..hi: one pass over the column's codes"""
    codes = np.asarray(seg._raw_codes(col))
    m = (codes >= lo) & (codes <= hi)
    return np.bincount((codes[m].astype(np.int64) - lo), minlength=hi - lo + 1)


def execute(seg, spec):
    global _HITS
    col = spec['col']; V = int(spec['V'])
    lim = spec['lim']
    if lim is None:
        return None
    need = int(lim) + int(spec.get('off') or 0)
    null = (V - 1) if spec['has_null'] else None
    Vn = V - (1 if null is not None else 0)
    fl = spec.get('flag'); rs = spec.get('runs')
    if rs is None and fl is None:
        rs = [(0, Vn)]                                   # no WHERE: every code
    if rs is not None:
        take = KF.first_codes(rs, need, spec['desc'])
    else:
        keep = np.flatnonzero(fl[:Vn])
        take = (keep[::-1] if spec['desc'] else keep)[:need]
    # every code holds >= 1 row: the first `need` kept codes suffice
    rows = []
    if take.size and need > 0:
        lo9, hi9 = int(take.min()), int(take.max())
        c9 = seg.cols[col]
        # the head window pays only where the smallest kept value is likely common: a small dictionary, and not the
        # sparse dress whose low count reads just its literal lane (SearchPhrase, the submitted Q25: 8.8 -> 14.8 ms
        # with the window, which a 6M-value dictionary's first phrase never fills)
        hc = (_head_first(seg, col, int(take[0]), need)
              if V <= _SMALL_V and c9.get('code_enc') != 8 else None)
        if hc is not None:
            # THE HEAD WINDOW: the first kept value already shows `need` rows in the file's first window -- the
            # answer is that value `need` times, exact without any count (JavaEnable <> 1: 59 -> ~6 ms)
            cnt = np.asarray([need], np.int64); take = take[:1]
        elif hi9 < _SMALL_V:
            # the codes sit low in the dictionary: the parallel low-count pass (one counter per code below the
            # bound, per lane -- cheap only while the bound is small)
            cnt = np.asarray(seg.low_counts(col, hi9 + 1), np.int64)[take]
        else:
            # high codes (DESC, or a filter like UserID > x): count just the run lo..hi in one pass over the codes
            # (UserID > 5e18 ASC: the low-count pass with a 9M-code bound took 680 ms, this road ~85)
            cnt = _counts_run(seg, col, lo9, hi9)[take - lo9]
        if hc is None and (cnt == 0).any():
            # A CODE WITH NO ROWS breaks the walk's premise (each code holds >= 1 row; a fresh segment's dictionary
            # is built from its rows): recount the same codes in one direct pass, and if they truly hold fewer
            # than `need` rows, widen the walk to every kept code
            cnt = _counts_run(seg, col, lo9, hi9)[take - lo9]
            if int(cnt.sum()) < need:
                take = KF.first_codes(rs, Vn, spec['desc']) if rs is not None else \
                    (np.flatnonzero(fl[:Vn])[::-1] if spec['desc'] else np.flatnonzero(fl[:Vn]))
                if take.size:
                    lo9, hi9 = int(take.min()), int(take.max())
                    cnt = _counts_run(seg, col, lo9, hi9)[take - lo9]
                    nz = cnt > 0
                    take, cnt = take[nz], cnt[nz]
        fill = np.cumsum(cnt)
        used = int(np.searchsorted(fill, need)) + 1          # codes needed to reach `need` rows
        used = min(used, take.size)
        vals = seg.values_at(col, take[:used]) if used else []
        for v, n in zip(vals, cnt[:used].tolist()):
            rows.extend([(wdb_sql._pyval(v),)] * min(int(n), need - len(rows)))
            if len(rows) >= need:
                break
    if len(rows) < need and fl is None and null is not None:
        nn = int(_counts_run(seg, col, null, null)[0])                  # NULLS LAST (DuckDB's default)
        rows.extend([(None,)] * min(nn, need - len(rows)))
    rows = rows[int(spec.get('off') or 0):need]
    _HITS += 1
    return rows, [wdb_sql._alias(spec['proj'][0])]


def try_valsort(seg, tree, col_map):
    spec = detect(seg, tree, col_map)
    if spec is None:
        return None
    return execute(seg, spec)
