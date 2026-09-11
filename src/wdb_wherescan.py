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


def _code_of(seg, col, val):
    """Dict code of a literal, or None when absent. Dicts are value-sorted: mode 2 answers by
    searchsorted over the int dict; string dicts by fetch-bisect (O(log V) point fetches, each
    sub-ms against chunked dicts -- never materializes the dictionary)."""
    c = seg.cols[col]
    V = int(c['V']) - (1 if c['has_null'] else 0)
    if V <= 0:
        return None
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
    prefix 'x%', suffix '%x'. No interior wildcards."""
    neg = False
    n = node
    if isinstance(n, E.Not):
        n = n.this; neg = True
    if not isinstance(n, E.Like) or not isinstance(n.this, E.Column):
        return None
    neg = neg or bool(n.args.get('negate'))      # sqlglot: NOT LIKE == Like(negate=True)
    pat = _litval(n.expression)
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


def _like_flags(seg, col, needle, kind='contains'):
    """Boolean flag[code] = dict value contains needle. THE DICT IS THE HAYSTACK: the row data
    is never string-compared -- all distinct values are scanned once (C-speed buffer find with
    per-string skip), and the predicate collapses to a code-set membership test. Memoized for
    the process lifetime (transient; dies with the worker -- not a persisted structure)."""
    memo = seg.__dict__.setdefault('_ws_like_memo', {})
    mk = (col, needle, kind)
    if mk in memo:
        return memo[mk]
    vals = seg._typed_dict(col)
    V = len(vals)
    bs = [v if isinstance(v, (bytes, bytearray)) else
          (v.encode() if isinstance(v, str) else bytes(v)) for v in vals]
    lens = np.fromiter((len(v) for v in bs), np.int64, V)
    offs = np.zeros(V + 1, np.int64); np.cumsum(lens, out=offs[1:])
    hay = b''.join(bs)
    nd = needle.encode() if isinstance(needle, str) else needle
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
    drive_eq = min(_cand, key=lambda i: _eq_cost(eqs[i][0], eqs[i][1])) if _cand else None
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
            if seg.cols[sp2['col']].get('code_enc') == 3:
                drive_sf = si
                break
    drive_in = None
    if drive_eq is None and drive_like is None and drive_or is None and drive_sf is None:
        for ii, (icol, _iv, ineg) in enumerate(ins):
            c2 = seg.cols[icol]
            if not ineg and c2.get('mode') in (0, 1, 2) and c2.get('code_enc') == 3:
                drive_in = ii            # a positive IN drives: union flag over its codes
                break
    drive_neq = None
    if drive_eq is None and drive_like is None and drive_or is None and drive_sf is None \
            and drive_in is None and not spans:
        for ei, (ecol, _ev, eop) in enumerate(eqs):
            c2 = seg.cols[ecol]
            if eop == '<>' and c2.get('mode') in (0, 1, 2) and c2.get('code_enc') == 3:
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
        if order is None or not order.expressions:
            return None
        oe = order.expressions[0]
        ocol = oe.this.name if isinstance(oe.this, E.Column) else None
        ocol = col_map.get(ocol, ocol) if (col_map and ocol) else ocol
        if ocol is None or oe.args.get('desc') or seg.stairs(ocol) is None:
            return None                  # rows mode rides the cluster order: ASC on a stair column
        tiebreak = []
        for oe2 in order.expressions[1:]:
            if oe2.args.get('desc') or not isinstance(oe2.this, E.Column):
                return None
            tc = oe2.this.name
            tc = col_map.get(tc, tc) if col_map else tc
            if not P.columns_exist(seg, tc):
                return None
            tiebreak.append(tc)
        lim = wdb_sql._limit(tree)
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
        # trailing ASC key columns (the validator's total-order tiebreak) are exactly the
        # canonical emission order (count desc, composite asc = value asc) -- accept them
        knames = {k['src'] for _pi, k in keys if k.get('kind') == 'col'}
        kalias = {wdb_sql._alias(proj[pi]) for pi, _k in keys}
        for oe2 in order.expressions[1:]:
            if oe2.args.get('desc') or not isinstance(oe2.this, E.Column):
                return None
            n2 = oe2.this.name
            n2m = col_map.get(n2, n2) if col_map else n2
            if n2m not in knames and n2 not in kalias:
                return None
    lim = wdb_sql._limit(tree)
    off = wdb_sql._offset(tree) or 0
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
        def row_of(code):                        # first row of a code (code==V -> N)
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
    if c.get('code_enc', 0) == 3 and col not in seg._codes:
        wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
        BR = c['BR']; base = c['cstart']; bo = c['boffs']; buf = seg.buf
        j0, j1 = lo // BR, (hi - 1) // BR + 1
        def scan(js):
            import zstandard as zstd
            dz = zstd.ZstdDecompressor(); out = []
            for j in js:
                raw = np.frombuffer(dz.decompress(buf[base+int(bo[j]):base+int(bo[j+1])].tobytes()), dtype=wdt)
                a, b = max(lo, j*BR), min(hi, j*BR + raw.size)
                seg_ = raw[a-j*BR:b-j*BR]
                h = np.nonzero(seg_ != code)[0] if negate else np.nonzero(seg_ == code)[0]
                if h.size: out.append(h + a)
            return np.concatenate(out) if out else np.empty(0, np.int64)
        W = min(_SCAN_THREADS, max(1, j1 - j0))
        with ThreadPoolExecutor(W) as ex:
            parts = list(ex.map(scan, np.array_split(np.arange(j0, j1), W)))
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.empty(0, np.int64)
    cc = np.asarray(seg._raw_codes_range(col, lo, hi))
    h = np.nonzero(cc != code)[0] if negate else np.nonzero(cc == code)[0]
    return h + lo

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
    if c.get('code_enc', 0) == 3 and col not in seg._codes:
        wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
        BR = c['BR']; base = c['cstart']; bo = c['boffs']; buf = seg.buf
        j0, j1 = lo // BR, (hi - 1) // BR + 1
        def scan(js):
            import zstandard as zstd
            dz = zstd.ZstdDecompressor(); out = []
            for j in js:
                raw = np.frombuffer(dz.decompress(buf[base+int(bo[j]):base+int(bo[j+1])].tobytes()), dtype=wdt)
                a, b = max(lo, j*BR), min(hi, j*BR + raw.size)
                h = np.nonzero(flag[raw[a-j*BR:b-j*BR]])[0]
                if h.size: out.append(h + a)
            return np.concatenate(out) if out else np.empty(0, np.int64)
        W = min(_SCAN_THREADS, max(1, j1 - j0))
        with ThreadPoolExecutor(W) as ex:
            parts = list(ex.map(scan, np.array_split(np.arange(j0, j1), W)))
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.empty(0, np.int64)
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
    cc = np.asarray(seg._raw_codes_range(col, lo, hi))
    return np.nonzero(flag[cc])[0] + lo


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


def _fact_small(a):
    """np.unique(a, return_inverse=True) without the hidden sort: when the value range
    is narrow (derived keys like minute 0-59 or LENGTH 0-500, and composites over them),
    a bincount presence table factorizes in O(N). Identical contract; sorted u; falls
    back to np.unique for wide or empty inputs."""
    if a.size == 0:
        return np.unique(a, return_inverse=True)
    amin = int(a.min()); amax = int(a.max())
    span = amax - amin + 1
    if span > (1 << 20):
        return np.unique(a, return_inverse=True)
    shifted = (a - amin).astype(np.int64, copy=False)
    pres = np.bincount(shifted, minlength=span)
    uvals = np.flatnonzero(pres)
    rank = np.empty(span, dtype=np.int64)
    rank[uvals] = np.arange(uvals.size)
    return (uvals + amin), rank[shifted]


def execute(seg, spec):
    global _HITS
    lo, hi = _span_rows(seg, spec['spans'])
    likes = spec.get('likes', [])
    lflags = [(col, _like_flags(seg, col, needle, kind), neg) for col, needle, kind, neg in likes] if likes else []
    de, dl = spec.get('drive_eq'), spec.get('drive_like')
    _dn = None
    if hi <= lo:
        pos = np.empty(0, np.int64)
    elif de is not None:
        col, val, _op = spec['eqs'][de]
        code = _code_of(seg, col, val)
        pos = _scan_eq(seg, col, int(code), lo, hi) if code is not None else np.empty(0, np.int64)
    elif dl is not None:
        col, fl, _n = lflags[dl]
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
    # residual order is ours to choose (conjunction commutes): equalities before
    # inequalities (eq slashes survivors, neq barely trims), cheap dictionaries
    # before heavy ones -- the cte-chain lesson: testing a 6.5M-V string column
    # at 738K rows before a 9K-V column cut them to 5K cost half the query
    _res = sorted(range(len(spec['eqs'])),
                  key=lambda _i: (0 if spec['eqs'][_i][2] == '=' else 1,
                                  int(seg.cols.get(spec['eqs'][_i][0], {}).get('V') or 0)))
    for i in _res:
        col, val, op = spec['eqs'][i]
        if i == de or i == spec.get('drive_neq') or pos.size == 0: continue
        code = _code_of(seg, col, val)
        if code is None:
            if op == '=': pos = np.empty(0, np.int64)
            continue
        cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
        pos = pos[cc != code] if op == '<>' else pos[cc == code]
    for i, (col, fl, neg) in enumerate(lflags):
        if i == dl or pos.size == 0: continue
        cc = np.asarray(seg.codes_at(col, pos)).astype(np.int64)
        m = fl[cc]
        pos = pos[~m] if neg else pos[m]
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

    # ---- rows mode: SELECT * ordered by the cluster column -- positions ARE the order
    if spec.get('mode') == 'rows':
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
            if c['mode'] == 4:
                kcols[s] = np.asarray(seg._seq_decode(c))[pos]
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
    comp = np.zeros(pos.size, dtype=np.int64)
    locals_ = []
    for _k, a in keyarr:
        u, inv = _fact_small(a)
        comp = comp * u.size + inv
        locals_.append(u)
    g, ginv = _fact_small(comp)
    cnt = np.bincount(ginv)
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
    first = np.empty(cnt.size, np.int64)            # first row of every group, one pass,
    first[ginv[::-1]] = np.arange(pos.size - 1, -1, -1, dtype=np.int64)   # no sort: reversed
    rep = first[sel]                                # writes -- last write is the earliest row
    # ONE gather per agg column over all pos, then every group's aggregate vectorized --
    # the per-group codes_at gathers re-decompressed the frames each group's scattered
    # members touch (gs-cube: 1,165 gathers, 8,773 zstd calls for a 4-column query).
    G = cnt.size
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
    rows_out = []
    for i, gi in enumerate(sel):
        r = int(rep[i]); row = []
        for _pi, kind_or_key in sorted(
                [(pi, ('agg', kind, col)) for pi, kind, col in spec['aggs']] +
                [(pi, ('key', kk, aa)) for (pi, kk), (_x, aa) in zip(spec['keys'], keyarr)]):
            tag = kind_or_key[0]
            if tag == 'agg':
                _t, kind, col = kind_or_key
                if kind == 'COUNT_STAR':
                    row.append(int(cnt[gi]))
                elif kind in ('COUNT_COL', 'COUNT_D'):
                    row.append(int(_cells[_pi][gi]))
                elif kind in ('MIN_DICT', 'MAX_DICT'):
                    k2 = int(_cells[_pi][gi])
                    if (kind == 'MAX_DICT' and k2 < 0) or \
                            (kind == 'MIN_DICT' and k2 == np.iinfo(np.int64).max):
                        row.append(None)             # a group whose every member was null
                    else:
                        row.append(wdb_sql._pyval(seg.fetch(col, k2)))
                elif kind == 'SUM':
                    row.append(int(_cells[_pi][gi]))
                else:                                # AVG
                    row.append(float(_cells[_pi][gi]) / int(cnt[gi]))
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
