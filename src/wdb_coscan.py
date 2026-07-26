"""
wdb_coscan -- the fused conjunctive count: duck's homework, our dialect.

COUNT(*) under ANDed dict predicates used to run driver-then-residuals: scan one
column fully, then codes_at the others at scattered survivor positions (re-opening
most of their blocks anyway). Here every predicate is tested block-by-block in one
walk -- and before any byte decompresses, the per-block min/max codes (blockstats,
folded from 32K stat blocks to compression blocks) veto blocks that cannot contain
the literals. On counter-ordered data an equality like CounterID=62 lives in a
handful of blocks; the veto kills the rest for EVERY column at once.

Soundness: the veto is a necessary condition (min<=code<=max), never sufficient --
false positives decompress and test, false negatives cannot exist. Ranges ride
code order and are therefore admitted ONLY on mode-2 int dicts (numeric dict
order); string dicts sort by string, where code ranges lie.
"""
import numpy as np
import zstandard
from concurrent.futures import ThreadPoolExecutor
import wdb_sql
import wdb_policies as P
import wdb_blockstats as BS
import sqlglot.expressions as E

_HITS = 0


def _lit(x):
    return x.this if isinstance(x, E.Literal) else None


def detect(seg, tree, col_map):
    if not P.no_joins(tree) or not P.no_having(tree) or not P.no_select_distinct(tree):
        return None
    if tree.args.get('group') is not None or tree.args.get('qualify') is not None:
        return None
    w = tree.args.get('where')
    if w is None:
        return None
    proj = tree.expressions
    if len(proj) != 1:
        return None
    kind = wdb_sql._agg_kind(proj[0])
    if kind is None or kind[0] != 'COUNT_STAR':
        return None
    sc = (lambda c: col_map.get(c, c)) if col_map else (lambda c: c)
    preds = []
    def walk(node):
        if isinstance(node, E.And):
            return walk(node.this) and walk(node.expression)
        if isinstance(node, E.Paren):
            return walk(node.this)
        if isinstance(node, (E.EQ, E.NEQ)):
            nm = wdb_sql._colname(node.this)
            lv = _lit(node.expression)
            if nm is None or lv is None:
                return False
            preds.append(('eq' if isinstance(node, E.EQ) else 'neq', sc(nm), lv))
            return True
        if isinstance(node, E.In):
            nm = wdb_sql._colname(node.this)
            exprs = node.args.get('expressions') or []
            if nm is None or not exprs or any(_lit(x) is None for x in exprs):
                return False
            neg = bool(node.args.get('is_negated')) or isinstance(node.parent, E.Not)
            preds.append(('notin' if neg else 'in', sc(nm), tuple(_lit(x) for x in exprs)))
            return True
        if isinstance(node, E.Not) and isinstance(node.this, E.In):
            inner = node.this
            nm = wdb_sql._colname(inner.this)
            exprs = inner.args.get('expressions') or []
            if nm is None or not exprs or any(_lit(x) is None for x in exprs):
                return False
            preds.append(('notin', sc(nm), tuple(_lit(x) for x in exprs)))
            return True
        if isinstance(node, E.Between):
            nm = wdb_sql._colname(node.this)
            lo, hi = _lit(node.args.get('low')), _lit(node.args.get('high'))
            if nm is None or lo is None or hi is None:
                return False
            preds.append(('range', sc(nm), (lo, hi)))
            return True
        return False
    if not walk(w.this) or not preds:
        return None
    for kd, nm, _v in preds:
        c = seg.cols.get(nm)
        if c is None or c.get('has_null') or c.get('mode') not in (0, 1, 2):
            return None
        if kd == 'range' and c.get('mode') != 2:
            return None                                 # code order == value order only there
    if not P.no_deleted_rows(seg):
        return None
    return {'preds': preds, 'alias': wdb_sql._alias(proj[0])}


def _block_codes(seg, col, j):
    c = seg.cols[col]
    base = c['cstart']; bo = c['boffs']
    wdt = np.uint8 if c['cwidth'] == 1 else (np.uint16 if c['cwidth'] == 2 else np.uint32)
    dz = zstandard.ZstdDecompressor()
    return np.frombuffer(dz.decompress(seg.buf[base + int(bo[j]):base + int(bo[j + 1])].tobytes()),
                         dtype=wdt)


def _comp_minmax(seg, col, nbc, BR):
    st = BS.build(seg, col)
    if st is None or st.get('mode4'):
        return None
    f = BR // 32768                                     # stat blocks per compression block
    if f * 32768 != BR:
        return None
    cmin, cmax = st['cmin'], st['cmax']
    lo = np.empty(nbc, np.int64); hi = np.empty(nbc, np.int64)
    for j in range(nbc):
        a, b = j * f, min((j + 1) * f, cmin.size)
        lo[j] = cmin[a:b].min(); hi[j] = cmax[a:b].max()
    return lo, hi


def execute(seg, spec):
    global _HITS
    import wdb_wherescan as WS
    N = int(seg.N)
    resolved = []                                       # (kind, col, payload)
    for kd, nm, v in spec['preds']:
        c = seg.cols[nm]
        if kd in ('eq', 'neq'):
            k = WS._code_of(seg, nm, v)
            if k is None:
                if kd == 'eq':
                    _HITS += 1
                    return [(0,)], [spec['alias']]      # impossible conjunct: the answer is 0
                continue                                # <> absent-value: always true, drop
            resolved.append((kd, nm, int(k)))
        elif kd in ('in', 'notin'):
            ks = [WS._code_of(seg, nm, x) for x in v]
            ks = sorted(int(k) for k in ks if k is not None)
            if kd == 'in' and not ks:
                _HITS += 1
                return [(0,)], [spec['alias']]
            if kd == 'notin' and not ks:
                continue
            resolved.append((kd, nm, np.asarray(ks, np.int64)))
        else:                                           # range, mode 2 only
            arr = np.asarray(seg._dict_ints(c))
            V = int(c['V'])
            try:
                lo_v, hi_v = int(v[0]), int(v[1])
            except (TypeError, ValueError):
                return None
            lo_c = int(np.searchsorted(arr[:V], lo_v, side='left'))
            hi_c = int(np.searchsorted(arr[:V], hi_v, side='right')) - 1
            if lo_c > hi_c:
                _HITS += 1
                return [(0,)], [spec['alias']]
            resolved.append(('range', nm, (lo_c, hi_c)))
    if not resolved:
        _HITS += 1
        return [(N,)], [spec['alias']]
    conforming = [nm for _kd, nm, _pl in resolved
                  if seg.cols[nm].get('BR') and seg.cols[nm].get('boffs') is not None]
    BRs = {int(seg.cols[nm]['BR']) for nm in conforming}
    BR = BRs.pop() if len(BRs) == 1 else (524288 if not BRs else None)
    if BR is None:
        return None                                     # mixed block sizes: not this path
    chunked = set(conforming)
    full = {}                                           # unchunked/odd columns: one read, sliced
    for _kd, nm, _pl in resolved:
        if nm not in chunked and nm not in full:
            full[nm] = np.asarray(seg._raw_codes(nm))
    nbc = (N + BR - 1) // BR
    admit = np.ones(nbc, bool)
    for kd, nm, pl in resolved:
        if nm not in chunked:
            continue                                    # no per-block veto for full-read cols
        mm = _comp_minmax(seg, nm, nbc, BR)
        if mm is None:
            continue                                    # no stats: mask-only conjunct
        lo, hi = mm
        if kd == 'eq':
            admit &= (lo <= pl) & (pl <= hi)
        elif kd == 'range':
            admit &= (lo <= pl[1]) & (pl[0] <= hi)
        elif kd == 'neq':
            admit &= ~((lo == hi) & (lo == pl))         # constant excluded block: veto
        else:                                           # in / notin
            if kd == 'in':
                a = np.zeros(nbc, bool)
                for k in pl:
                    a |= (lo <= k) & (k <= hi)
                admit &= a
            else:
                const = lo == hi
                excl = np.isin(lo, pl)
                admit &= ~(const & excl)
    blocks = np.flatnonzero(admit)
    def one(j):
        m = None
        a = int(j) * BR
        for kd, nm, pl in resolved:
            if nm in chunked:
                raw = _block_codes(seg, nm, int(j))
            else:
                raw = full[nm][a:a + BR]
            if kd == 'eq':
                mm = raw == pl
            elif kd == 'neq':
                mm = raw != pl
            elif kd == 'range':
                mm = (raw >= pl[0]) & (raw <= pl[1])
            elif kd == 'in':
                mm = np.isin(raw.astype(np.int64), pl)
            else:
                mm = ~np.isin(raw.astype(np.int64), pl)
            m = mm if m is None else (m & mm)
            if not m.any():
                return 0
        return int(m.sum())
    total = 0
    if blocks.size:
        with ThreadPoolExecutor(max_workers=min(8, blocks.size)) as ex:
            for n in ex.map(one, blocks.tolist()):
                total += n
    _HITS += 1
    return [(total,)], [spec['alias']]
