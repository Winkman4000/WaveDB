"""WaveDB JOIN executor — step 1: two-table INNER equi-join.

Decode the needed columns from each side via WaveDB's own read path, join them, then run the SELECT
clauses (projection / WHERE / GROUP BY / aggregates / ORDER BY / LIMIT). The join itself has two
kernels: a hash join (baseline, any equi-join) and -- when the child join key is a pre-resolved
foreign-key POINTER into the parent's rows -- a gather join (array index, no hash build). This module
is the hash baseline + the shared post-join evaluator; the gather kernel layers on top.
Correctness first: verified against DuckDB. Unsupported shapes raise NotImplementedError.
"""
import re
import sqlglot, sqlglot.expressions as E
import numpy as np
import time, pandas as pd, os
from wdb_engine import Segment
import wdb_sql
import wdb_kernels, wdb_dml, wdb_agg, wdb_fkptr, wdb_exprjit, wdb_radix
import wdb_measure_runtime as RT

_CMP = {E.EQ: '==', E.NEQ: '!=', E.GT: '>', E.LT: '<', E.GTE: '>=', E.LTE: '<='}


def _materialize(db, table, cols):
    """Return {col: np.array} for the given LOGICAL columns of `table`, decoded via WaveDB. Fast
    path (single live segment) reads columns directly; otherwise fall back to the row SELECT path
    so presence/synth/merge correctness is inherited."""
    cols = list(dict.fromkeys(cols))
    phys = db.cat.phys_map(table)
    paths = db.cat.segment_paths(table)
    hp = wdb_dml.hot_path(db.cat, table); hot = os.path.exists(hp)
    if len(paths) == 1 and not hot:
        seg = db.open_segment(paths[0], table)
        if seg.presence_mask() is None:
            return {c: seg.values(phys.get(c, c)) for c in cols}
    if len(paths) > 1 and not hot:
        # THE UNION: typed arrays exactly as a single segment gives them (the row-SELECT fallback
        # built object arrays that pandas merged to nothing: a two-key self-join answered 0)
        try:
            u, _ = _solo_segment(db, table)
            return {c: u.values(phys.get(c, c)) for c in cols}
        except _FastUnsupported:
            pass
    rows, _ = db.run(f"SELECT {', '.join(cols)} FROM {table}")
    arrs = list(zip(*rows)) if rows else [()] * len(cols)
    return {c: np.array(arrs[i], dtype=object) for i, c in enumerate(cols)}


def _all_columns(node):
    """Every (table_alias, colname) referenced under a node."""
    return [(c.table, c.name) for c in node.find_all(E.Column)]


def _chain_pandas(db, tree, ctx):
    """Fallback that REUSES the FK-chain resolution: gather every referenced column to fact-row space via
    the same composed pointers the fast path uses, then run the pandas WHERE/aggregate tail on that frame.
    So an FK-chain query that the fused fast path can't take (high-card group, non-value-identity column,
    plain projection, exotic predicate) still resolves the SAME join -- no separate single-join engine, no
    'multi-join' cliff. pandas only does the aggregation/predicate part that couldn't be fused."""
    alias2t = ctx['alias2t']; seg_of = ctx['seg_of']; composed = ctx['composed']
    cols_of = {a: set(db.cat.column_names(t)) for a, t in alias2t.items()}
    phys_of = {a: db.cat.phys_map(t) for a, t in alias2t.items()}
    _memo = {}
    def gather(alias, name):
        k = (alias, name)
        if k not in _memo:
            seg = seg_of[alias]; pcol = phys_of[alias].get(name, name); cptr = composed[alias]
            arr, _nm = wdb_sql._col(seg, pcol)
            _memo[k] = arr if cptr is None else arr[cptr]        # gather parent rows to fact rows
        return _memo[k]
    def owner(node):
        a = node.table
        if a:
            if a not in alias2t or node.name not in cols_of[a]: raise _FastUnsupported
            return a
        owners = [al for al, cs in cols_of.items() if node.name in cs]
        if len(owners) != 1: raise _FastUnsupported
        return owners[0]
    R = lambda node: f"{owner(node)}.{node.name}"

    proj = tree.expressions
    frame = {}
    scan = list(proj)
    for key in ('where', 'group', 'order'):
        nd = tree.args.get(key)
        if nd is not None: scan.append(nd)
    alias9p = {p.alias for p in proj if isinstance(p, E.Alias)}
    where = tree.args.get('where')
    # SURVIVOR DISCIPLINE for the pandas tail (Q10 paid 41s decoding fact-
    # scale strings it then threw away): gather ONLY the WHERE's columns
    # full-width, mask, then fetch every remaining column at survivors.
    wcols9 = set()
    if where is not None:
        for col in where.find_all(E.Column):
            if col.table or col.name not in alias9p:
                wcols9.add((owner(col), col.name))
    need9 = []
    for rootn in scan:
        for col in rootn.find_all(E.Column):
            if not col.table and col.name in alias9p:
                continue                   # output alias (ORDER BY revenue): applies post-projection
            a = owner(col)
            if (a, col.name) not in need9:
                need9.append((a, col.name))
    for a, nm9 in need9:
        if (a, nm9) in wcols9:
            frame[f"{a}.{nm9}"] = gather(a, nm9)
    df = pd.DataFrame(frame) if frame else pd.DataFrame(index=range(ctx['n']))
    if where is not None:
        df = df[_mask(df, where.this, R)]
    sv9 = df.index.to_numpy()
    late9 = {}
    for a, nm9 in need9:
        fk = f"{a}.{nm9}"
        if fk in df.columns:
            continue
        seg = seg_of[a]; pcol = phys_of[a].get(nm9, nm9); cptr = composed[a]
        rows_a = sv9 if cptr is None else np.asarray(cptr)[sv9]
        c9m = seg.cols.get(pcol, {})
        try:
            if (c9m.get('mode', 0) in (0, 1, 2) and not c9m.get('has_null')
                    and not seg._override_vals_typed(pcol)):
                cd9 = np.asarray(seg.codes_at(pcol, rows_a))
                td9 = seg._typed_dict(pcol)
                late9[fk] = (pd.Series([td9[c] for c in cd9], index=sv9) if c9m.get('dt') == 1
                             else pd.Series(np.asarray(td9)[cd9], index=sv9))
            else:
                # values_at takes DICT CODES, never row positions (mode-5's
                # sorted dict punished the confusion with silent permutation)
                late9[fk] = pd.Series(seg.values_at_rows(pcol, rows_a), index=sv9)
        except Exception:
            late9[fk] = pd.Series(np.asarray(gather(a, nm9))[sv9], index=sv9)
    for fk, ser9 in late9.items():
        df[fk] = ser9
    group = tree.args.get('group')
    has_agg = any(wdb_sql._agg_kind(p) is not None
                  or any(True for _ in p.find_all(E.Sum, E.Avg, E.Min, E.Max, E.Count))
                  for p in proj)
    if group is not None or has_agg:
        rows = _aggregate(df, proj, group, R)
    else:
        keys = [R(p.this if isinstance(p, E.Alias) else p) for p in proj]
        rows = [tuple(_render(v) for v in t) for t in df[keys].itertuples(index=False, name=None)]
    having = tree.args.get('having')
    if having is not None:
        rows = wdb_sql._apply_having(rows, proj, having.this, None)   # fused path must filter too
    rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))
    lim = wdb_sql._limit(tree)
    if lim is not None: rows = rows[:lim]
    return rows, [wdb_sql._alias(p) for p in proj]


def _exact_scalar(db, tree):
    """No-WHERE, no-GROUP scalar aggregates over dict-int columns, by DICTIONARY
    ARITHMETIC: one bincount of raw codes per column, then SUM exact at any magnitude
    (two-limb fold, wdb_exactint -- float64 drifts and int64 wraps at SUM(UserID) scale),
    AVG correctly rounded from the exact SUM, MIN/MAX off the value-ordered dict's
    present codes, COUNTs from the counts. Declines (None) on any other shape."""
    import wdb_exactint as XI
    if (tree.args.get('where') is not None or tree.args.get('group') is not None
            or tree.args.get('joins') or tree.args.get('having') is not None
            or tree.args.get('distinct') is not None or tree.args.get('order') is not None
            or wdb_sql._limit(tree) is not None or wdb_sql._offset(tree)):
        return None
    frm = tree.find(E.From)
    if frm is None or not isinstance(frm.this, E.Table):
        return None
    name = frm.this.name
    try:
        paths = db.cat.segment_paths(name)
    except Exception:
        return None
    if len(paths) != 1:
        return None
    seg = db.open_segment(paths[0], name)
    if seg.presence_mask() is not None:
        return None                                  # DELETEs falsify raw-code counts
    specs = []
    for p in tree.expressions:
        ak = wdb_sql._agg_kind(p)
        if ak is None:
            return None
        if ak[0] == 'COUNT_STAR':
            specs.append(('COUNT_STAR', None)); continue
        if ak[0] not in ('SUM', 'AVG', 'MIN', 'MAX', 'COUNT') or not isinstance(ak[1], str):
            return None
        col = ak[1]
        c = seg.cols.get(col)
        if c is None or c.get('mode') not in (0, 1, 2, 4):
            return None
        if c.get('mode') in (2, 4) and ak[0] in ('SUM', 'AVG', 'MIN', 'MAX') and c.get('dt') != 0:
            return None                              # float/typed stay on their own paths
        if c.get('mode') == 4 and c.get('has_null'):
            return None                              # seq nulls: keep the mature path
        if seg._effective(col) is not None:
            return None                              # UPDATE overrides falsify raw codes
        if ak[0] in ('SUM', 'AVG', 'MIN', 'MAX') and c.get('dt') != 0:
            return None                              # datetimes keep their typed emission
        specs.append((ak[0], col))
    import wdb_window as WN
    cnts = {}
    def counts_of(col, tabsize):
        if col not in cnts:
            import wdb_kernels as _WKc
            cn = _WKc.bincount_par(seg._raw_codes(col), tabsize)   # THE PARALLEL CENSUS (no astype copy)
            if cn.size > tabsize:
                cn = cn[:tabsize]                    # null codes live past the dict's
            cnts[col] = cn                           # values: SQL aggs exclude them
        return cnts[col]
    N = int(seg.N)
    seqs = {}
    row = []
    for kind, col in specs:
        if kind == 'COUNT_STAR':
            row.append(N); continue
        c = seg.cols[col]
        if c.get('mode') == 4:                       # sequence codec: decode once, exact fold
            if col not in seqs:
                seqs[col] = np.asarray(seg._seq_decode(c))
            v = seqs[col]
            if kind == 'COUNT':
                row.append(int(v.size))
            elif kind == 'MIN':
                row.append(int(v.min()) if v.size else None)
            elif kind == 'MAX':
                row.append(int(v.max()) if v.size else None)
            elif kind == 'SUM':
                row.append(XI.fold_values(v) if v.size else None)
            else:
                row.append(XI.exact_avg(XI.fold_values(v), int(v.size)))
            continue
        if kind == 'COUNT' and c.get('dt') != 0:
            # COUNT(text/float column): the non-null codes are the dictionary's own (a NULL is the
            # last code, V - 1) -- no value table needed; _int_table on a text dict raised
            cn = counts_of(col, int(c['V']) - (1 if c.get('has_null') else 0))
            row.append(int(cn.sum())); continue
        tab = WN._int_table(seg, col)
        cn = counts_of(col, tab.size)
        if kind == 'COUNT':
            row.append(int(cn.sum())); continue
        if kind in ('MIN', 'MAX'):
            nz = np.flatnonzero(cn)
            row.append(None if nz.size == 0
                       else int(tab[int(nz[0] if kind == 'MIN' else nz[-1])])); continue
        n = int(cn.sum())
        s = XI.fold_counts(cn, tab)
        row.append((s if n else None) if kind == 'SUM' else XI.exact_avg(s, n))
    global _FAST_HITS
    _FAST_HITS += 1
    return [tuple(row)], [wdb_sql._alias(p) for p in tree.expressions]


def _sum_overflow_gate(db, tree):
    """Raise _FastUnsupported when a SUM/AVG argument is a dict-int column whose extremes
    times N could exceed 2^53: the fused engine accumulates float64, which drifts there
    (SUM(UserID) at 100M: ~1e13 absolute error). Falling through reaches wherescan's
    exact two-limb fold. Extremes cost two point-fetches on the value-ordered dict."""
    risky = []
    for p in tree.expressions:
        ak = wdb_sql._agg_kind(p)
        if ak and ak[0] in ('SUM', 'AVG') and isinstance(ak[1], str):
            risky.append(ak[1])
    if not risky:
        return
    frm = tree.find(E.From)
    if frm is None or not isinstance(frm.this, E.Table):
        return
    try:
        paths = db.cat.segment_paths(frm.this.name)
    except Exception:
        return
    if len(paths) != 1:
        return
    seg = db.open_segment(paths[0], frm.this.name)
    for col in risky:
        c = seg.cols.get(col)
        if c is None or c.get('mode') not in (0, 1, 2) or c.get('dt') != 0:
            continue
        nd = int(c.get('n_dict') or c['V'])
        if nd == 0:
            continue
        try:
            lov = int(seg.fetch(col, 0)); hiv = int(seg.fetch(col, nd - 1))
        except Exception:
            continue
        if max(abs(lov), abs(hiv)) * max(int(seg.N), 1) >= (1 << 53):
            raise _FastUnsupported


def table_agg(db, tree):
    """Single-table aggregate routed through the SAME fused engine as joins: a 0-join chain (fact only, every
    cptr is None). Reuses predicate fusion, high-card factorise, and vectorised assembly. Raises
    _FastUnsupported on anything not fusable so the caller falls back to the mature single-table executor."""
    w = tree.args.get('where')
    if w is None:
        ex = _exact_scalar(db, tree)         # dict-arithmetic scalars: exact hugeint SUM,
        if ex is not None:                   # correctly-rounded AVG, dict-edge MIN/MAX
            return ex
    _sum_overflow_gate(db, tree)             # big-int SUM/AVG: decline so the exact
    if w is not None:                        # wherescan fold serves instead of float64
        for innode in w.find_all(E.In):
            if len(innode.args.get('expressions') or []) > 256:
                raise _FastUnsupported       # giant literal lists: the pandas tail decodes the
                                             # world; the fallback's code-space isin is the path
    _dc9 = _descent_court(db, tree)
    if _dc9 is not None:
        return _dc9
    return _fast_pointer_agg(db, tree, _build_chain(db, tree))


def _sumslice_meta(db, tbl):
    """Memoised per-table sum-stamp metadata ({key, val, stamp, ceilings}) or None."""
    memo = getattr(db, '_sumslice_memo', None)
    if memo is None:
        memo = db._sumslice_memo = {}
    if tbl not in memo:
        import os as _os, json as _js
        p = _os.path.join(db.cat.dbdir, tbl + '.sumslice.json')
        m = None
        if _os.path.exists(p):
            try:
                m = _js.load(open(p))
                m.setdefault('stamp', 'l_sumslice')
            except Exception:
                m = None
        memo[tbl] = m
    return memo[tbl]


def _descent_exec(db, tree, meta, s9, k, fseg):
    """Jackson's pipeline, the probe's shape, no ceremony: (1) stream the
    one-byte stamp >= s9; (2) confirm survivors -- fact conjuncts at
    survivor scale, parent verdicts at PARENT scale flowing the mmap'd
    edge roads down, fact pays one road gather at survivor scale; (3) sum
    and top-k at the final survivors. The join context (segments + edge
    roads) is memoised like the plan cache: paths and mmaps, no row data.
    None on any doubt -> the general court serves."""
    import sqlglot.expressions as E9
    def _gate9(tag):
        if __import__('os').environ.get('WDB_JOIN_BILL'):
            print('JOIN BILL: EXEC gate-out %s' % tag, flush=True)
        return None
    try:
        import os as _os
        memo = db.__dict__.setdefault('_dc_ctx', {})
        tbls = tuple(sorted(t.name for t in tree.find_all(E9.Table)))
        ent = memo.get(tbls)
        if ent is not None:
            for p9, mt9 in ent['stamps']:
                if _os.path.getmtime(p9) != mt9:
                    ent = None; break
        if ent is None:
            ctx = _build_chain(db, tree.copy())
            ent = dict(fact=ctx['fact'], alias2t=ctx['alias2t'],
                       seg_of=ctx['seg_of'], edges=ctx.get('edge_ptrs') or {},
                       stamps=[(db.cat.segment_paths(t)[0], _os.path.getmtime(db.cat.segment_paths(t)[0]))
                               for t in ctx['alias2t'].values()])
            memo[tbls] = ent
        import time as _tt
        _bl = [] if __import__('os').environ.get('WDB_JOIN_BILL') else None
        _t0 = _tt.perf_counter()
        fact_a = ent['fact']; seg_of = ent['seg_of']; edges = ent['edges']
        own = {}
        for a9, t9 in ent['alias2t'].items():
            for c9 in db.cat.column_names(t9):
                own.setdefault(c9, a9)
        road = {pa: (ca, p9) for pa, (ca, p9) in edges.items()}
        # ---- 1. stream the byte
        std = np.asarray(fseg._typed_dict(meta['stamp']))
        clo = int(np.searchsorted(std, s9, side='left'))
        r = np.flatnonzero(np.asarray(fseg.codes(meta['stamp'])) >= clo)
        if _bl is not None: _bl.append(('ctx+stamp', _tt.perf_counter() - _t0)); _t0 = _tt.perf_counter()
        # ---- 2. confirm
        w9 = tree.args.get('where')
        def flat(x):
            if isinstance(x, E9.Paren): return flat(x.this)
            if isinstance(x, E9.And): return flat(x.this) + flat(x.expression)
            return [x]
        def keep_of(cn, sg, nm):
            td = np.asarray(sg._typed_dict(nm))
            V = int(sg.cols[nm]['V'])
            kx = np.zeros(V + 1, bool)
            lit = cn.args.get('expression') if not isinstance(cn, E9.Between) else None
            def num(e): return float(str(e.name if hasattr(e, 'name') else e))
            if td.dtype.kind in 'iuf':
                if   isinstance(cn, E9.GT):  kx[:td.size] = td > num(lit)
                elif isinstance(cn, E9.GTE): kx[:td.size] = td >= num(lit)
                elif isinstance(cn, E9.LT):  kx[:td.size] = td < num(lit)
                elif isinstance(cn, E9.LTE): kx[:td.size] = td <= num(lit)
                elif isinstance(cn, E9.EQ):  kx[:td.size] = td == num(lit)
                elif isinstance(cn, E9.Between):
                    kx[:td.size] = (td >= num(cn.args['low'])) & (td <= num(cn.args['high']))
                else: return _gate9('G1')
            elif isinstance(cn, E9.EQ) and lit is not None and getattr(lit, 'is_string', False):
                want = str(lit.name)
                kx[:td.size] = np.array([(x.decode() if isinstance(x, (bytes, bytearray)) else str(x)) == want
                                         for x in td])
            else:
                return _gate9('G2')
            return kx
        fact_cns = []; par_keep = {}
        for cn in (flat(w9.this) if w9 is not None else []):
            cols = list(cn.find_all(E9.Column))
            if (isinstance(cn, E9.EQ) and len(cols) == 2
                    and isinstance(cn.this, E9.Column)
                    and isinstance(cn.expression, E9.Column)):
                continue                    # a join road, already honored by the edges
            if len(cols) != 1: return _gate9('G3')
            nm = cols[0].name; a9 = own.get(nm)
            if a9 is None: return _gate9('G4')
            kx = keep_of(cn, seg_of[a9], nm)
            if kx is None: return _gate9('G5')
            if a9 == fact_a:
                fact_cns.append((nm, kx))
            else:
                m9 = kx[np.asarray(seg_of[a9].codes(nm))]
                par_keep[a9] = m9 if a9 not in par_keep else (par_keep[a9] & m9)
        if _bl is not None: _bl.append(('parse-where+parent-keeps', _tt.perf_counter() - _t0)); _t0 = _tt.perf_counter()
        for nm, kx in fact_cns:                       # fact confirms at survivor scale
            if r.size == 0: break
            c14c = fseg.cols.get(nm)
            if c14c is not None and c14c.get('code_enc') in (14, 15, 16):
                nz14 = np.flatnonzero(kx)
                if nz14.size and int(nz14[-1]) + 1 - int(nz14[0]) == nz14.size:
                    lo14c, hi14c = int(nz14[0]), int(nz14[-1]) + 1
                    td14c = np.asarray(fseg._typed_dict(nm))
                    V14c = int(c14c['V'])
                    if hi14c <= V14c:
                        dl14 = int(td14c[lo14c])
                        dh14 = int(td14c[hi14c]) if hi14c < V14c else int(td14c[V14c - 1]) + 1
                        r = r[fseg.plane_test(nm, dl14, dh14)[r]]
                        continue
            r = r[kx[np.asarray(fseg.codes_at(nm, r))]]
        if _bl is not None: _bl.append(('fact-confirm', _tt.perf_counter() - _t0)); _t0 = _tt.perf_counter()
        # flow parent verdicts DOWN the roads (parent scale), then one fact gather each
        depth = {fact_a: 0}; ch = True
        while ch:
            ch = False
            for pa, (ca, _p) in road.items():
                if ca in depth and pa not in depth:
                    depth[pa] = depth[ca] + 1; ch = True
        for pa in sorted([a for a in par_keep if a != fact_a], key=lambda a: -depth.get(a, 0)):
            ca, p9 = road.get(pa, (None, None))
            if ca is None: return _gate9('G6')
            if ca == fact_a:
                if r.size:
                    r = r[par_keep[pa][np.asarray(p9)[r]]]
            else:
                m9 = par_keep[pa][np.asarray(p9)]
                par_keep[ca] = m9 if ca not in par_keep else (par_keep[ca] & m9)
        # ---- 3. sum + top-k
        if _bl is not None: _bl.append(('road-flow', _tt.perf_counter() - _t0)); _t0 = _tt.perf_counter()
        group = tree.args.get('group')
        f2a = {}                                       # fact -> depth-1 alias road, at survivors
        gk = []
        for g9 in group.expressions:
            c9 = g9.find(E9.Column)
            nm = c9.name; a9 = own.get(nm)
            if a9 is None: return _gate9('G7')
            if a9 == fact_a:
                cd = np.asarray(fseg.codes_at(nm, r))
            else:
                ca, p9 = road.get(a9, (None, None))
                if ca != fact_a: return _gate9('G8')
                if a9 not in f2a: f2a[a9] = np.asarray(p9)[r]
                cd = np.asarray(seg_of[a9].codes(nm))[f2a[a9]]
            gk.append((nm, seg_of[a9], cd))
        sexpr = None
        for p in tree.expressions:
            ag = p.find(E9.Sum)
            if ag is not None: sexpr = ag.this
        def ev(e):
            if isinstance(e, E9.Paren): return ev(e.this)
            if isinstance(e, E9.Column):
                a9 = own.get(e.name)
                if a9 != fact_a: return _gate9('G9')
                _rdc = wdb_sql.raw_dict_col(seg_of[a9], e.name, want_codes=False)   # cached base
                td = (np.asarray(_rdc[0], dtype=np.float64) if _rdc is not None
                      else np.asarray(seg_of[a9]._typed_dict(e.name), dtype=np.float64))
                return td[np.asarray(fseg.codes_at(e.name, r))]
            if isinstance(e, E9.Literal): return float(str(e.name))
            if isinstance(e, E9.Mul):
                a, b = ev(e.this), ev(e.expression)
                return _gate9('G10') if a is None or b is None else a * b
            if isinstance(e, E9.Sub):
                a, b = ev(e.this), ev(e.expression)
                return _gate9('G11') if a is None or b is None else a - b
            if isinstance(e, E9.Add):
                a, b = ev(e.this), ev(e.expression)
                return _gate9('G12') if a is None or b is None else a + b
            return _gate9('G13')
        rev = ev(sexpr)
        if rev is None or not gk: return _gate9('G14')
        import pandas as pd
        comp = np.zeros(r.size, dtype=np.int64)
        for _, sg, cd in gk:
            comp = comp * (int(cd.max()) + 1 if cd.size else 1) + cd.astype(np.int64)
        gid, uniq = pd.factorize(comp, sort=False)
        sums = np.bincount(gid, weights=rev, minlength=len(uniq))       # not np.add.at
        _u9, firsts = np.unique(gid, return_index=True)                 # first row per group, vectorized
        _f9 = np.full(len(uniq), -1, np.int64); _f9[_u9] = firsts; firsts = _f9
        keys2 = []
        for oe in tree.args.get('order').expressions[1:]:
            nm2 = oe.this.name
            for (nm, sg, cd) in gk:
                if nm == nm2: keys2.append(cd[firsts])
        order = np.lexsort(tuple(reversed([-sums] + keys2)))
        top = order[:k]
        rows = []
        for t9i in top:
            row = []; i0 = firsts[t9i]
            for p in tree.expressions:
                if p.find(E9.Sum) is not None:
                    row.append(float(sums[t9i])); continue
                nm = p.find(E9.Column).name
                for (nmg, sg, cd) in gk:
                    if nmg == nm:
                        v9 = sg._typed_dict(nm)[int(cd[i0])]
                        c9m = sg.cols[nm]
                        if c9m.get('dt') == 3:
                            import datetime as _dt
                            v9 = _dt.date(1970, 1, 1) + _dt.timedelta(days=int(v9))
                        elif isinstance(v9, (bytes, bytearray)):
                            v9 = v9.decode()
                        elif c9m.get('dt') == 0 and float(v9) == int(v9):
                            v9 = int(v9)
                        row.append(v9); break
                else:
                    return _gate9('G15')
            rows.append(tuple(row))
        if _bl is not None:
            _bl.append(('sum+topk+emit', _tt.perf_counter() - _t0))
            print('JOIN BILL: EXEC ' + ' | '.join('%s=%.0fms' % (n9, v9 * 1000) for n9, v9 in _bl), flush=True)
        return rows, [wdb_sql._alias(p) for p in tree.expressions]
    except Exception:
        if __import__('os').environ.get('WDB_JOIN_BILL'):
            import traceback
            print('JOIN BILL: EXEC declined:', flush=True)
            traceback.print_exc()
        return _gate9('G16')


def _descent_court(db, tree):
    def _no9(tag):
        return None
    """THE DESCENT COURT (Jackson's stamp, given teeth): top-k-by-SUM served
    from the heaviest sum-slices down. Gate: GROUP BY includes the stamp's
    key; ORDER BY <sum alias> DESC with LIMIT k; the summed expression is
    provably bounded by the stamped value column (val itself, or
    val*(1-c)/(1-c)*val with c's dictionary within [0,1]). Then: run the
    ordinary court with 'stamp >= s' injected, and STOP when the kth sum
    STRICTLY exceeds the next slice's stored ceiling -- orders below cannot
    reach it (filtered sum <= total <= ceiling). Descend a slice otherwise.
    Any mismatch or doubt returns None: the plain path is always the law."""
    import sqlglot.expressions as E9
    try:
        group = tree.args.get('group'); lim = tree.args.get('limit')
        ordn = tree.args.get('order')
        if group is None or lim is None or ordn is None or tree.args.get('having') is not None:
            return _no9('shape')
        k = int(lim.expression.name)
        o0 = ordn.expressions[0]
        if not o0.args.get('desc'): return _no9('not-desc')
        onm = o0.this.name if isinstance(o0.this, E9.Column) else None
        if onm is None: return _no9('order-not-col')
        sum_idx = None; sum_expr = None; aliases = []
        for i, p in enumerate(tree.expressions):
            aliases.append(wdb_sql._alias(p))
            if aliases[-1] == onm:
                ag = p.find(E9.Sum)
                if ag is None: return _no9('no-sum')
                sum_idx = i; sum_expr = ag.this
        if sum_idx is None: return _no9('alias-miss')
        cols = list(sum_expr.find_all(E9.Column))
        if not cols: return _no9('no-cols')
        tbl = None
        for t9 in (tree.find_all(E9.Table)):
            m9 = _sumslice_meta(db, t9.name)
            if m9 is not None and m9['val'] in {c.name for c in cols}:
                tbl = t9.name; meta = m9; break
        if tbl is None: return _no9('no-meta-table')
        seg9 = db.open_segment(db.cat.segment_paths(tbl)[0], tbl)
        if meta['stamp'] not in db.cat.column_names(tbl): return _no9('stamp-not-registered')
        # bound proof: expr is val, or val*(1-c) with c's dict in [0,1]
        val = meta['val']; okb = False
        if isinstance(sum_expr, E9.Column) and sum_expr.name == val:
            okb = True
        elif isinstance(sum_expr, E9.Mul):
            a9, b9 = sum_expr.this, sum_expr.expression
            def unparen(x):
                return unparen(x.this) if isinstance(x, E9.Paren) else x
            a9, b9 = unparen(a9), unparen(b9)
            for u9, v9 in ((a9, b9), (b9, a9)):
                if isinstance(u9, E9.Column) and u9.name == val and isinstance(v9, E9.Sub):
                    l9, r9 = v9.this, v9.expression
                    if (isinstance(l9, E9.Literal) and str(l9.name) == '1'
                            and isinstance(r9, E9.Column)):
                        td9 = np.asarray(seg9._typed_dict(r9.name))
                        if td9.dtype.kind in 'if' and td9.size and 0.0 <= float(td9.min()) and float(td9.max()) <= 1.0:
                            okb = True
                    break
        if not okb: return _no9('bound-unproven')
        gnames = set()
        for g9 in group.expressions:
            for c9 in g9.find_all(E9.Column):
                gnames.add(c9.name)
        if meta['key'] not in gnames: return _no9('key-not-grouped')
        ceil9 = meta['ceilings']
        _bill9 = __import__('os').environ.get('WDB_JOIN_BILL')
        s9 = 255
        for _it in range(6):
            t2 = tree.copy()
            w2 = t2.args.get('where')
            cond9 = E9.GTE(this=E9.column(meta['stamp']),
                           expression=E9.Literal.number(s9))
            t2.set('where', E9.Where(this=cond9 if w2 is None
                                     else E9.And(this=w2.this, expression=cond9)))
            out = _descent_exec(db, tree, meta, s9, k, seg9)
            if out is None:
                out = _fast_pointer_agg(db, t2, _build_chain(db, t2))
            rows, als = out if isinstance(out, tuple) else (out, None)
            kth = float(rows[k - 1][sum_idx]) if len(rows) >= k else None
            nxt = max(ceil9[1:s9]) if s9 > 1 else None
            done = s9 <= 1 or (kth is not None and nxt is not None and kth > nxt)
            if _bill9:
                print('JOIN BILL: DESCENT slice>=%d rows=%d kth=%s next-ceil=%s %s'
                      % (s9, len(rows), kth, nxt, 'STOP' if done else 'DESCEND'),
                      flush=True)
            if done:
                return out
            if kth is None:
                s9 -= 1
            else:
                s9 = max(1, max((i for i in range(1, s9) if ceil9[i] >= kth), default=s9 - 1))
        return None
    except _FastUnsupported:
        if __import__('os').environ.get('WDB_JOIN_BILL'):
            import traceback
            print('JOIN BILL: DESCENT declined _FastUnsupported:', flush=True)
            traceback.print_exc()
        return None
    except Exception:
        if __import__('os').environ.get('WDB_JOIN_BILL'):
            import traceback
            print('JOIN BILL: DESCENT declined:', flush=True)
            traceback.print_exc()
        return None


def denorm_rewrite(db, tree):
    """Rewrite a single INNER join that GROUPs BY denormalised (stamped) parent columns into a
    single-table GROUP BY on the child's stamped columns -- which the cube path then answers, reading a
    few precomputed cells instead of gathering the whole child. Returns a rewritten joinless SQL string
    or None (keep the gather). Equivalence conditions, all required:
      - exactly one INNER join, a GROUP BY, no WHERE / HAVING / DISTINCT (conservative);
      - every GROUP BY column resolves to a child column: a TOTAL stamp of a parent column (so the inner
        join drops no child rows), or a child-native column;
      - every aggregate is COUNT(*) or over a CHILD-NATIVE column (never a stamped parent column -- that
        would be the grain trap of summing a parent attribute once per child row).
    Anything else returns None and stays on the gather, which already wins on high-card joins."""
    joins = tree.args.get('joins')
    if not joins or len(joins) != 1: return None
    if tree.args.get('group') is None: return None
    if (tree.args.get('where') is not None or tree.args.get('having') is not None
            or tree.args.get('distinct') is not None): return None
    jn = joins[0]
    if jn.args.get('side') or jn.args.get('kind'): return None          # INNER only
    frm = tree.find(E.From)
    if frm is None: return None
    frm = frm.this
    child_t, child_a = frm.name, (frm.alias or frm.name)
    parent_t, parent_a = jn.this.name, (jn.this.alias or jn.this.name)
    try:
        stamps = db.cat.stamps(child_t)
    except Exception:
        return None
    if not stamps: return None
    pmap = {m['parent_col']: cc for cc, m in stamps.items()
            if m.get('parent') == parent_t and m.get('total')}          # parent col -> child stamped col
    if not pmap: return None
    try:
        child_cols = set(db.cat.column_names(child_t)); parent_cols = set(db.cat.column_names(parent_t))
    except Exception:
        return None
    stamped = set(pmap.values())
    def to_child(node):
        if not isinstance(node, E.Column): return None
        a, nm = node.table, node.name
        if a == parent_a: return pmap.get(nm)
        if a == child_a:  return nm if nm in child_cols else None
        if nm in child_cols and nm not in parent_cols: return nm        # unqualified child-native
        if nm in parent_cols and nm not in child_cols: return pmap.get(nm)
        return None                                                      # ambiguous / unknown
    gcols = []
    for g in tree.args['group'].expressions:
        cc = to_child(g)
        if cc is None: return None
        gcols.append(cc)
    proj_sql = []
    _FN = {E.Sum: 'SUM', E.Avg: 'AVG', E.Min: 'MIN', E.Max: 'MAX'}
    for p in tree.expressions:
        inner = p.this if isinstance(p, E.Alias) else p
        explicit = p.alias if isinstance(p, E.Alias) else None
        if isinstance(inner, E.Count) and (inner.this is None or isinstance(inner.this, E.Star)):
            s = "COUNT(*)"
        elif isinstance(inner, E.Column):
            cc = to_child(inner)
            if cc is None: return None
            s = cc if cc == inner.name else f"{cc} AS {inner.name}"      # preserve output header
        elif isinstance(inner, E.Count):
            arg = inner.this
            cc = to_child(arg) if isinstance(arg, E.Column) else None
            if cc is None: return None
            s = f"COUNT({cc})"
        elif type(inner) in _FN:
            arg = inner.this
            cc = to_child(arg) if isinstance(arg, E.Column) else None
            if cc is None or cc in stamped: return None                  # measure must be child-native
            s = f"{_FN[type(inner)]}({cc})"
        else:
            return None
        if explicit: s += f" AS {explicit}"
        proj_sql.append(s)
    return f"SELECT {', '.join(proj_sql)} FROM {child_t} GROUP BY {', '.join(gcols)}"


def _rich_lonely_frame(db, inner):
    """Q22's inner dissolved: single-table SELECT over customer where every
    conjunct is one of {2-char-prefix IN literals (byte kernel), col > (scalar
    AVG subquery of the same shape), NOT EXISTS (child fk = key) (the CENSUS:
    a bincount of the child fk over the key's identity domain -- 'has no
    orders' is 'the slot is empty')}. Returns the inner's frame or None."""
    import pandas as pd
    if inner.args.get('joins') or inner.args.get('group'): return None
    frm = inner.args.get('from') or inner.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table): return None
    t1 = frm.this.name
    try:
        seg1, _sp = _solo_segment(db, t1)
    except _FastUnsupported:
        return None
    phys1 = db.cat.phys_map(t1)
    def _flat9(x):
        if isinstance(x, E.Paren): return _flat9(x.this)
        if isinstance(x, E.And): return _flat9(x.this) + _flat9(x.expression)
        return [x]
    def _prefix_node(nd):
        if isinstance(nd, E.Substring):
            a1 = nd.args.get('start'); a2 = nd.args.get('length')
            if (isinstance(nd.this, E.Column) and a1 is not None and a2 is not None
                    and str(a1.this) == '1' and str(a2.this) == '2'):
                return nd.this
        return None
    def _mask_of(w9, seg):
        m = None
        for cj in _flat9(w9):
            mm = None
            if isinstance(cj, E.In):
                pc = _prefix_node(cj.this)
                lits = [str(x.this) for x in (cj.args.get('expressions') or []) if isinstance(x, E.Literal)]
                if pc is None or not lits or any(len(x) != 2 for x in lits): return None
                pre = seg.prefix2_codes(phys1.get(pc.name, pc.name))
                want = np.array(sorted((ord(x[0]) << 8) | ord(x[1]) for x in lits), dtype=np.uint16)
                mm = np.isin(pre, want)
            elif type(cj) in (E.GT, E.GTE, E.LT, E.LTE) and isinstance(cj.this, E.Column):
                rhs = cj.expression
                if isinstance(rhs, E.Subquery) or isinstance(rhs, E.Select) or rhs.find(E.Select) is not None:
                    sub = rhs.this if isinstance(rhs, E.Subquery) else rhs
                    sv = _scalar_of(sub)
                    if sv is None: return None
                    rv = sv
                elif isinstance(rhs, E.Literal) and not rhs.is_string:
                    rv = float(rhs.this)
                else:
                    return None
                raw = wdb_sql.raw_dict_col(seg, phys1.get(cj.this.name, cj.this.name))
                if raw is None: return None
                vals = raw[0][np.asarray(raw[1])]
                op = {E.GT: np.greater, E.GTE: np.greater_equal,
                      E.LT: np.less, E.LTE: np.less_equal}[type(cj)]
                mm = op(vals, rv)
            elif isinstance(cj, E.Exists) or (isinstance(cj, E.Not) and isinstance(cj.this, E.Exists)):
                inv9 = isinstance(cj, E.Not)
                ex = (cj.this if inv9 else cj).this
                if not isinstance(ex, E.Select): return None
                efrm = ex.args.get('from') or ex.args.get('from_')
                if efrm is None or not isinstance(efrm.this, E.Table): return None
                t2 = efrm.this.name
                ew = ex.args.get('where')
                if ew is None: return None
                ecs = _flat9(ew.this)
                if len(ecs) != 1 or not isinstance(ecs[0], E.EQ): return None
                x9, y9 = ecs[0].this, ecs[0].expression
                c2 = set(db.cat.column_names(t2))
                if x9.name not in c2: x9, y9 = y9, x9
                if x9.name not in c2 or y9.name not in set(db.cat.column_names(t1)): return None
                seg2, _s2 = _solo_segment(db, t2)
                # THE CENSUS, CACHED per (child, fk, parent key) on the parent
                # segment -- Jackson's .cnt.npy law in process (106ms per query
                # rebuilt a 15M bincount that never changes).
                _cc9 = getattr(seg1, '_census_cache', None)
                if _cc9 is None: _cc9 = seg1._census_cache = {}
                _ck9 = ('fkcensus', t2, x9.name, y9.name)
                _ent9 = _cc9.get(_ck9)
                if _ent9 is None:
                    _rk1 = wdb_sql.raw_dict_col(seg1, phys1.get(y9.name, y9.name), want_codes=False)
                    kv1 = (np.asarray(_rk1[0], dtype=np.int64) if _rk1 is not None
                           else np.asarray(wdb_sql._col(seg1, phys1.get(y9.name, y9.name))[0]).astype(np.int64))
                    kmin = int(kv1.min())
                    if int(kv1.max()) - kmin + 1 != kv1.size: return None
                    fk9 = wdb_sql.raw_dict_col(seg2, db.cat.phys_map(t2).get(x9.name, x9.name))
                    if fk9 is None: return None
                    cen = np.bincount((fk9[0][np.asarray(fk9[1])]).astype(np.int64) - kmin,
                                      minlength=kv1.size)
                    _cc9[_ck9] = cen
                else:
                    cen = _ent9
                mm = (cen == 0) if inv9 else (cen > 0)
            else:
                return None
            m = mm if m is None else (m & mm)
        return m
    def _scalar_of(sub):
        if not isinstance(sub, E.Select): return None
        sf = sub.args.get('from') or sub.args.get('from_')
        if sf is None or not isinstance(sf.this, E.Table) or sf.this.name != t1: return None
        sp = list(sub.expressions)
        if len(sp) != 1: return None
        ag = sp[0].this if isinstance(sp[0], E.Alias) else sp[0]
        if not (isinstance(ag, E.Avg) and isinstance(ag.this, E.Column)): return None
        sm = _mask_of(sub.args['where'].this, seg1) if sub.args.get('where') is not None else None
        if sm is None and sub.args.get('where') is not None: return None
        raw = wdb_sql.raw_dict_col(seg1, phys1.get(ag.this.name, ag.this.name))
        if raw is None: return None
        vals = raw[0][np.asarray(raw[1])]
        return float(vals[sm].mean() if sm is not None else vals.mean())
    w = inner.args.get('where')
    if w is None: return None
    m = _mask_of(w.this, seg1)
    if m is None: return None
    outcols = {}
    for p in inner.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        nm9 = p.alias_or_name
        pc9 = _prefix_node(nd)
        if pc9 is not None:
            pre = seg1.prefix2_codes(phys1.get(pc9.name, pc9.name))[m]
            outcols[nm9] = [chr(x >> 8) + chr(x & 255) for x in pre.tolist()]
        elif isinstance(nd, E.Column):
            raw = wdb_sql.raw_dict_col(seg1, phys1.get(nd.name, nd.name))
            if raw is None: return None
            outcols[nm9] = raw[0][np.asarray(raw[1])][m]
        else:
            return None
    return pd.DataFrame(outcols)



def _left_count_frame(db, inner):
    """Q13's LEFT JOIN dissolved (Jackson's reading: the join fetches NO
    foreign data -- its whole meaning is 'the domain is ALL customers, absent
    matches count zero'): filter the child by the ON's extra conjunct (a
    front-coded LIKE runs as a byte kernel over the dict), bincount child fk
    values over the parent's FULL key domain, zeros by construction. Returns
    the inner's output frame or None."""
    import pandas as pd
    joins = inner.args.get('joins') or []
    if len(joins) != 1: return None
    jn = joins[0]
    if (jn.args.get('side') or '').upper() != 'LEFT': return None
    frm = inner.args.get('from') or inner.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table): return None
    t1 = frm.this.name; a1 = frm.this.alias or t1
    if not isinstance(jn.this, E.Table): return None
    t2 = jn.this.name; a2 = jn.this.alias or t2
    grp = inner.args.get('group')
    if grp is None or len(grp.expressions) != 1: return None
    gk = grp.expressions[0]
    if not isinstance(gk, E.Column): return None
    proj = list(inner.expressions)
    if len(proj) != 2: return None
    kcol = proj[0].this if isinstance(proj[0], E.Alias) else proj[0]
    cagg = proj[1]
    if not (isinstance(kcol, E.Column) and kcol.name == gk.name): return None
    if not (isinstance(cagg, E.Alias) and isinstance(cagg.this, E.Count)
            and isinstance(cagg.this.this, E.Column)): return None
    on = jn.args.get('on')
    if on is None: return None
    def _flat9(x):
        if isinstance(x, E.Paren): return _flat9(x.this)
        if isinstance(x, E.And): return _flat9(x.this) + _flat9(x.expression)
        return [x]
    eqs, extra = [], []
    c1 = set(db.cat.column_names(t1)); c2 = set(db.cat.column_names(t2))
    def owner9(c):
        if c.table: return c.table
        if c.name in c1 and c.name not in c2: return a1
        if c.name in c2 and c.name not in c1: return a2
        return None
    for cj in _flat9(on):
        if isinstance(cj, E.EQ) and isinstance(cj.this, E.Column) and isinstance(cj.expression, E.Column):
            eqs.append(cj); continue
        if all(owner9(x) == a2 for x in cj.find_all(E.Column)):
            extra.append(cj); continue
        return None
    if len(eqs) != 1: return None
    x, y = eqs[0].this, eqs[0].expression
    if owner9(x) == a2: x, y = y, x
    if owner9(x) != a1 or owner9(y) != a2: return None
    if x.name != gk.name: return None
    seg1, _sp1 = _solo_segment(db, t1)
    seg2, _sp2 = _solo_segment(db, t2)
    kv1 = np.asarray(wdb_sql._col(seg1, db.cat.phys_map(t1).get(x.name, x.name))[0]).astype(np.int64)
    if kv1.size == 0: return None
    kmin, kmax = int(kv1.min()), int(kv1.max())
    if kmax - kmin + 1 != kv1.size: return None            # contiguous identity domain only
    keep = None
    for cj in extra:
        node = cj; invert = False
        if isinstance(node, E.Not): node = node.this; invert = True
        if not isinstance(node, E.Like): return None
        if node.args.get('negate'): invert = not invert   # some sqlglots fold NOT into Like
        col9 = node.this; pat9 = node.expression
        if not (isinstance(col9, E.Column) and isinstance(pat9, E.Literal) and pat9.is_string): return None
        p = str(pat9.this)
        if '_' in p or not (p.startswith('%') and p.endswith('%')): return None
        needles = [t for t in p.split('%') if t]
        if not needles or len(needles) > 2: return None
        try:
            m9 = seg2.like_mask_dict(db.cat.phys_map(t2).get(col9.name, col9.name), needles, invert=invert)
        except Exception:
            return None
        keep = m9 if keep is None else (keep & m9)
    fk9 = wdb_sql.raw_dict_col(seg2, db.cat.phys_map(t2).get(y.name, y.name))
    if fk9 is None: return None
    fkv = fk9[0][np.asarray(fk9[1])]
    if keep is not None: fkv = fkv[keep]
    cc = np.bincount(fkv.astype(np.int64) - kmin, minlength=kv1.size)
    return pd.DataFrame({(proj[0].alias if isinstance(proj[0], E.Alias) else kcol.name):
                         np.arange(kmin, kmax + 1, dtype=np.int64),
                         cagg.alias: cc.astype(np.int64)})



_MOMENT9 = tuple(getattr(E, n) for n in ('Stddev', 'StddevSamp', 'StddevPop', 'Variance', 'VariancePop', 'Corr') if hasattr(E, n))


def has_agg_arith(tree):
    """True when a projection combines aggregates with arithmetic (Q14's
    Div-of-Sums, H2O q7's MAX-MIN) or uses a MOMENT aggregate (STDDEV,
    VARIANCE, CORR -- algebra over hidden sums): the single-table executor
    cannot plan these."""
    if not isinstance(tree, E.Select): return False
    _AGG = (E.Sum, E.Count, E.Avg, E.Min, E.Max)
    _ROW9 = tuple(getattr(E, n) for n in ('Filter', 'GroupConcat', 'AnyValue', 'IgnoreNulls', 'Mode', 'LogicalAnd', 'LogicalOr',
                                          'PercentileCont', 'PercentileDisc', 'Quantile') if hasattr(E, n))
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        if isinstance(nd, _ROW9): continue                    # the row-aggregate family: the single-table path owns it
        if nd.find(E.Window) is not None: continue           # windows own their aggregates (SUM(x) OVER is not SUM arithmetic)
        if nd.find(*_MOMENT9) is not None: return True
        if hasattr(E, 'Median') and nd.find(E.Median) is not None: return True
        if isinstance(nd, (E.Column, *_AGG)): continue
        if nd.find(*_AGG) is not None: return True
    return False


def _moment_to_algebra(nd):
    """STDDEV/VARIANCE/CORR as arithmetic over SUM/COUNT (sample forms, as
    duck defaults): var_samp = (Sxx - Sx*Sx/n) / (n-1); corr = (n*Sxy - Sx*Sy)
    / sqrt((n*Sxx - Sx^2) * (n*Syy - Sy^2)). Returns a rewritten node."""
    def S(expr):  return E.Sum(this=expr.copy())
    def N(expr):  return E.Count(this=expr.copy())
    def mul(a, b): return E.Mul(this=a, expression=b)
    def sub(a, b): return E.Sub(this=a, expression=b)
    def div(a, b): return E.Div(this=a, expression=b)
    def lit(x):   return E.Literal(this=str(x), is_string=False)
    def sq(a):    return E.Mul(this=a.copy(), expression=a.copy())
    t = type(nd).__name__
    if t in ('Stddev', 'StddevSamp', 'Variance', 'StddevPop', 'VariancePop'):
        x = nd.this
        n = N(x); sx = S(x); sxx = S(sq(x))
        denom = lit(1) if t.endswith('Pop') else sub(N(x), lit(1))
        var = div(sub(sxx, div(mul(sx, sx.copy()), n)), (N(x) if t.endswith('Pop') else denom))
        if t.startswith('Var'): return var
        return E.Sqrt(this=var)
    if t == 'Corr':
        x, y = nd.this, nd.expression
        n = N(x); sx = S(x); sy = S(y); sxy = S(mul(x.copy(), y.copy())); sxx = S(sq(x)); syy = S(sq(y))
        num = sub(mul(n, sxy), mul(sx, sy))
        dx = sub(mul(N(x), S(sq(x))), mul(S(x), S(x)))
        dy = sub(mul(N(x), S(sq(y))), mul(S(y), S(y)))
        return div(num, E.Sqrt(this=mul(dx, dy)))
    return nd


def _median_side(db, tree, med_nodes):
    """THE ORDER-STATISTIC SIDE PASS: MEDIAN(col) per group for a single-table
    GROUP BY -- composite group ids (mixed radix over key codes), ONE lexsort
    of (gid, value), the middle of each run (mean of the two middles on even
    counts, as duck's quantile_cont(0.5)). Returns {alias: {keytuple: median}}
    keyed by the group columns' Python values, in GROUP BY order."""
    frm = tree.args.get('from') or tree.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table) or tree.args.get('joins'): return None
    if tree.args.get('where') is not None: return None
    grp = tree.args.get('group')
    if grp is None: return None
    gcols = list(grp.expressions)
    if not all(isinstance(g, E.Column) for g in gcols): return None
    tname = frm.this.name
    try:
        seg, _sp = _solo_segment(db, tname)
    except _FastUnsupported:
        return None
    pm = db.cat.phys_map(tname)
    n = int(seg.N)
    comp = np.zeros(n, np.int64); K = 1
    keyinfo = []
    for g in gcols:
        pc = pm.get(g.name, g.name)
        c = seg.cols.get(pc, {})
        if c.get('has_null') or c.get('mode') not in (0, 1, 2): return None
        V = int(c['V'])
        codes = np.asarray(seg.codes(pc)).astype(np.int64, copy=False)
        if K * V > (1 << 62): return None
        comp = comp * V + codes; K *= V
        keyinfo.append((pc, V))
    out = {}
    for al, nd in med_nodes:
        if not isinstance(nd.this, E.Column): return None
        raw = wdb_sql.raw_dict_col(seg, pm.get(nd.this.name, nd.this.name))
        if raw is None: return None
        vals = raw[0][np.asarray(raw[1])].astype(np.float64, copy=False)
        if K <= (1 << 24):
            # THE COUNTING SCATTER: O(n) placement by group id, then each
            # group's slice sorts in parallel -- no 50M-row lexsort.
            cnt_all = np.bincount(comp, minlength=K)
            present9 = np.flatnonzero(cnt_all)
            offs0 = np.zeros(K + 1, np.int64); np.cumsum(cnt_all, out=offs0[1:])
            cur = offs0[:-1].copy()
            placed = np.empty(n, np.float64)
            wdb_kernels.pscatter_by_gid(np.ascontiguousarray(comp), np.ascontiguousarray(vals), cur, placed)
            starts = offs0[present9]; ends = offs0[present9 + 1]
            med = np.empty(present9.size, np.float64)
            wdb_kernels.pgroup_median(placed, starts, ends, med)
            gcomp = present9
        else:
            order = np.lexsort((vals, comp))
            cs = comp[order]; vs = vals[order]
            starts = np.concatenate(([0], np.flatnonzero(np.diff(cs) != 0) + 1))
            ends = np.concatenate((starts[1:], [n]))
            cnt = ends - starts
            lo = starts + (cnt - 1) // 2
            hi = starts + cnt // 2
            med = (vs[lo] + vs[hi]) / 2.0
            gcomp = cs[starts]
        # decode the composite back to key values, last key fastest
        cols_v = [None] * len(keyinfo)
        rem = gcomp.copy()
        for j in range(len(keyinfo) - 1, -1, -1):
            pc, V = keyinfo[j]
            kc = rem % V; rem //= V
            cols_v[j] = _bulk_keyvals(seg, pc, kc)
        out[al] = {tuple(cols_v[j][i] for j in range(len(keyinfo))): float(med[i]) for i in range(len(gcomp))}
    return out



def has_expr_group(tree):
    """A GROUP BY key that is arithmetic over one column (id4 % 7): the join
    engine's dict-space expression key serves it; the single-table planner
    declines."""
    if not isinstance(tree, E.Select): return False
    g = tree.args.get('group')
    if g is None: return False
    if tree.find(E.Window) is not None: return False
    for k in g.expressions:
        if isinstance(k, (E.Mod, E.Div, E.Mul, E.Add, E.Sub, E.IntDiv)) and len(list(k.find_all(E.Column))) == 1:
            return True
    return False


def _agg_expr_rewrite(db, tree):
    """AGGREGATE ARITHMETIC (Q14's Div-of-Sums): a projection combining
    aggregates with +-*/ and literals runs as hidden aggregate aliases
    through the fused engine; the arithmetic evaluates on the results.
    Returns rows or None (not this shape)."""
    if not isinstance(tree, E.Select): return None
    _AGG = (E.Sum, E.Count, E.Avg, E.Min, E.Max)
    proj = list(tree.expressions)
    def _plain(nd):
        return isinstance(nd, _AGG) or isinstance(nd, E.Column)
    need = False
    if tree.find(E.Window) is not None: return None       # windows own their aggregates
    for p in proj:
        nd = p.this if isinstance(p, E.Alias) else p
        if nd.find(*_MOMENT9) is not None or (hasattr(E, 'Median') and nd.find(E.Median) is not None):
            need = True
        elif not _plain(nd) and nd.find(*_AGG) is not None:
            need = True
    if not need: return None
    if tree.args.get('order') is not None or tree.args.get('having') is not None: return None
    grp = tree.args.get('group')
    hidden = []          # (alias, node)
    def _hoist(nd):
        # replace each aggregate subtree with a Column ref to a hidden alias
        if isinstance(nd, _AGG):
            al = '__agg%d' % len(hidden)
            hidden.append((al, nd.copy()))
            return E.Column(this=E.Identifier(this=al, quoted=False))
        for k, v in list(nd.args.items()):
            if isinstance(v, E.Expression):
                nd.set(k, _hoist(v))
            elif isinstance(v, list):
                nd.set(k, [_hoist(x) if isinstance(x, E.Expression) else x for x in v])
        return nd
    outer = []           # per projection: ('col', idx) | ('expr', ast)
    st = tree.copy()
    new_exprs = []
    med_nodes = []       # (alias, Median node) served by the side pass
    for p in proj:
        nd = (p.this if isinstance(p, E.Alias) else p).copy()
        for md9 in list(nd.find_all(E.Median)) if hasattr(E, 'Median') else []:
            al9 = '__med%d' % len(med_nodes)
            med_nodes.append((al9, md9.copy()))
            rep9 = E.Column(this=E.Identifier(this=al9, quoted=False))
            if md9 is nd: nd = rep9
            else: md9.replace(rep9)
        for m9 in list(nd.find_all(*_MOMENT9)):
            rep9 = _moment_to_algebra(m9)
            if m9 is nd: nd = rep9
            else: m9.replace(rep9)
        if isinstance(nd, E.Column) and nd.name.startswith('__med'):
            outer.append(('hid', nd.name))
        elif isinstance(nd, E.Column):
            outer.append(('col', len(new_exprs))); new_exprs.append(nd.copy())
        elif isinstance(nd, _AGG):
            al = '__agg%d' % len(hidden); hidden.append((al, nd.copy()))
            outer.append(('hid', al))
        else:
            outer.append(('expr', _hoist(nd)))
    for al, nd in hidden:
        new_exprs.append(E.Alias(this=nd, alias=E.Identifier(this=al, quoted=False)))
    meds = None
    if med_nodes:
        meds = _median_side(db, tree, med_nodes)
        if meds is None:
            raise NotImplementedError('MEDIAN outside the single-table GROUP BY side pass')
        gnames9 = [g.name for g in tree.args['group'].expressions]
        for g9 in gnames9:
            if not any((e.alias_or_name == g9) for e in new_exprs):
                new_exprs.append(E.Column(this=E.Identifier(this=g9, quoted=False)))
    st.set('expressions', new_exprs)
    rows = join_query(db, st.sql(dialect='duckdb'))
    rows = rows[0] if isinstance(rows, tuple) else rows
    names = [e.alias_or_name for e in new_exprs]
    if meds is not None:
        gidx9 = [names.index(g9) for g9 in gnames9]
        for al9, table9 in meds.items():
            names.append(al9)
            rows = [tuple(r) + (table9.get(tuple(r[j] for j in gidx9)),) for r in rows]
    import math as _m9
    def ev(nd, rec):
        if isinstance(nd, E.Paren): return ev(nd.this, rec)
        if isinstance(nd, E.Column): return rec[names.index(nd.name)]
        if isinstance(nd, E.Sqrt):
            v = ev(nd.this, rec); return None if v is None or v < 0 else _m9.sqrt(v)
        if isinstance(nd, E.Pow):
            a = ev(nd.this, rec); b = ev(nd.expression, rec)
            return None if a is None or b is None else a ** b
        if isinstance(nd, E.Literal): return float(nd.this) if not nd.is_string else nd.this
        if isinstance(nd, E.Neg): return -ev(nd.this, rec)
        if isinstance(nd, E.Mul): return ev(nd.this, rec) * ev(nd.expression, rec)
        if isinstance(nd, E.Div):
            d = ev(nd.expression, rec); return None if not d else ev(nd.this, rec) / d
        if isinstance(nd, E.Add): return ev(nd.this, rec) + ev(nd.expression, rec)
        if isinstance(nd, E.Sub): return ev(nd.this, rec) - ev(nd.expression, rec)
        raise _FastUnsupported
    out = []
    for rec in rows:
        row = []
        for kind, x in outer:
            if kind == 'col': row.append(rec[x])
            elif kind == 'hid': row.append(rec[names.index(x)])
            else: row.append(ev(x, rec))
        out.append(tuple(row))
    return out



def _lonely_rewrite(db, tree):
    """Q21's EXISTS pair dissolved (Jackson): EXISTS(l2: same key, other supp)
    and NOT EXISTS(l3: same key, other supp, ALSO LATE) are two PARENT KEEPS
    -- distinct-supplier censuses over the child's sorted road: keep orders
    with nsupp >= 2 and late-distinct == 1 (the outer's own lateness conjunct
    covers l1's side). The pair is replaced by one In carrying _codes on the
    parent key; the mask layer serves it."""
    w = tree.args.get('where')
    if w is None: return
    def _flat9(x):
        if isinstance(x, E.Paren): return _flat9(x.this)
        if isinstance(x, E.And): return _flat9(x.this) + _flat9(x.expression)
        return [x]
    conj = _flat9(w.this)
    ex_pos = ex_neg = None
    for cj in conj:
        if isinstance(cj, E.Exists): ex_pos = cj
        elif isinstance(cj, E.Not) and isinstance(cj.this, E.Exists): ex_neg = cj
    if ex_pos is None or ex_neg is None: return
    def _parts(ex):
        sub = ex.this
        if not isinstance(sub, E.Select): return None
        f9 = sub.args.get('from') or sub.args.get('from_')
        if f9 is None or not isinstance(f9.this, E.Table): return None
        al = f9.this.alias or f9.this.name
        ww = sub.args.get('where')
        if ww is None: return None
        return f9.this.name, al, _flat9(ww.this)
    P1 = _parts(ex_pos); P2 = _parts(ex_neg.this)
    if P1 is None or P2 is None or P1[0] != P2[0]: return
    childT = P1[0]
    def _corr_neq(cjs, al):
        key_eq = neq = None; extra = []
        for c in cjs:
            if isinstance(c, E.EQ) and isinstance(c.this, E.Column) and isinstance(c.expression, E.Column):
                key_eq = c
            elif isinstance(c, E.NEQ) and isinstance(c.this, E.Column) and isinstance(c.expression, E.Column):
                neq = c
            else:
                extra.append(c)
        return key_eq, neq, extra
    k1, n1, e1 = _corr_neq(P1[2], P1[1])
    k2, n2, e2 = _corr_neq(P2[2], P2[1])
    if k1 is None or n1 is None or e1: return
    if k2 is None or n2 is None or len(e2) != 1: return
    # the NOT-EXISTS extra conjunct, re-aliased to the OUTER, must appear in the outer WHERE
    x2 = e2[0].copy()
    outer_al = None
    for c9 in x2.find_all(E.Column):
        if c9.table == P2[1]:
            pass
    inner_al = P2[1]
    outer_al = (k2.this.table if k2.this.table != inner_al else k2.expression.table) or ''
    for c9 in x2.find_all(E.Column):
        if c9.table == inner_al:
            c9.set('table', E.Identifier(this=outer_al, quoted=False) if outer_al else None)
    want = x2.sql()
    if not any(c.sql() == want for c in conj): return
    # key/supp/flag columns on the child
    key_c = k1.this if (k1.this.table or '') == P1[1] else k1.expression
    sup_c = n1.this if (n1.this.table or '') == P1[1] else n1.expression
    try:
        cseg, _s = _solo_segment(db, childT)
        # the parent of the child's key via the outer key column (other side of k1)
        okc = k1.expression if key_c is k1.this else k1.this
        # find the parent table owning the outer key through the OUTER's edges:
        pT = None
        for t9 in tree.find_all(E.Table):
            if t9.find_ancestor(E.Select) is not tree: continue
            if t9.name == childT and (t9.alias or t9.name) != (okc.table or ''):
                continue
        # outer key column l1.l_orderkey belongs to the fact alias; the road's
        # PARENT key is found through the catalog FK or unique-key equality in
        # the outer WHERE: o_orderkey = l1.l_orderkey
        peq = None
        for c9 in conj:
            if (isinstance(c9, E.EQ) and isinstance(c9.this, E.Column) and isinstance(c9.expression, E.Column)
                    and {c9.this.name, c9.expression.name} >= {okc.name} and c9 is not k1):
                a9, b9 = c9.this, c9.expression
                other = b9 if a9.name == okc.name and (a9.table or '') == (okc.table or '') else (
                        a9 if b9.name == okc.name and (b9.table or '') == (okc.table or '') else None)
                if other is not None:
                    peq = other; break
        if peq is None: return
        pT9 = None
        for t9 in tree.find_all(E.Table):
            if t9.find_ancestor(E.Select) is tree and peq.name in set(db.cat.column_names(t9.name)):
                pT9 = t9.name; break
        if pT9 is None or not _key_is_unique(db, pT9, peq.name): return
        pseg, _s2 = _solo_segment(db, pT9)
        road = np.asarray(_hash_pointer(db, childT, db.cat.phys_map(childT).get(key_c.name, key_c.name),
                                        cseg, pT9, db.cat.phys_map(pT9).get(peq.name, peq.name), pseg))
        if not bool((np.diff(road) >= 0).all()): return       # runs law: sorted roads only
        # the flag: the NOT-EXISTS extra conjunct evaluated on the CHILD --
        # only the declared-clock strict compare is served (Q21's late)
        f9n = e2[0]
        if not (type(f9n) in (E.GT, E.LT) and isinstance(f9n.this, E.Column)
                and isinstance(f9n.expression, E.Column)): return
        a9c, b9c = f9n.this.name, f9n.expression.name
        big, small = (a9c, b9c) if isinstance(f9n, E.GT) else (b9c, a9c)
        cc9 = cseg.cols.get(db.cat.phys_map(childT).get(big, big), {})
        if cc9.get('code_enc') != 16 or cc9.get('e16_partner') != db.cat.phys_map(childT).get(small, small):
            return
        bit9, dl9 = cseg.pair_bits(db.cat.phys_map(childT).get(big, big))
        flag = np.ascontiguousarray(np.asarray(bit9) & (np.asarray(dl9) > 0), dtype=np.bool_)
        sup9 = wdb_sql.raw_dict_col(cseg, db.cat.phys_map(childT).get(sup_c.name, sup_c.name), want_codes=True)
        if sup9 is None: return
        supv = np.ascontiguousarray(np.asarray(sup9[1]), dtype=np.int64)
        ch9 = np.flatnonzero(np.diff(road) != 0) + 1
        starts = np.concatenate((np.array([0]), ch9, np.array([road.shape[0]]))).astype(np.int64)
        nsupp = np.zeros(pseg.N, np.int32); nflag = np.zeros(pseg.N, np.int32)
        wdb_kernels.pruns_distinct(starts, np.ascontiguousarray(road, dtype=np.int64), supv, flag, nsupp, nflag)
        qual = np.flatnonzero((nsupp >= 2) & (nflag == 1)).astype(np.int64)
    except Exception:
        return
    # replace the pair with one In carrying _codes on the parent key
    innode = E.In(this=peq.copy())
    innode.set('_codes', qual)
    kept = [c for c in conj if c is not ex_pos and c is not ex_neg] + [innode]
    new = kept[0]
    for c in kept[1:]:
        new = E.And(this=new.copy() if new is kept[0] else new, expression=c.copy() if not isinstance(c, E.In) else c)
    w.set('this', new)



def _factor_or_rewrite(tree):
    """SCHOOLBOOK ALGEBRA (Jackson's Q19): (E and A) or (E and B) = E and
    (A or B). For EVERY OR conjunct of the WHERE (top-level or inside the
    AND chain -- Q7), conjuncts common to all branches hoist out, and THE
    IMPLIED KEEP adds col IN (union of pins) for columns pinned in every
    branch -- a single-column keep the cascade serves at dict scale, the
    exact OR re-checked at survivors."""
    w = tree.args.get('where')
    if w is None: return
    def _ors(x):
        if isinstance(x, E.Paren): return _ors(x.this)
        if isinstance(x, E.Or): return _ors(x.this) + _ors(x.expression)
        return [x]
    def _ands(x):
        if isinstance(x, E.Paren): return _ands(x.this)
        if isinstance(x, E.And): return _ands(x.this) + _ands(x.expression)
        return [x]
    def _factor_one(ornode):
        branches = _ors(ornode)
        if len(branches) < 2: return [ornode]
        csets = [ {c.sql(): c for c in _ands(b)} for b in branches ]
        common = set(csets[0].keys())
        for cs in csets[1:]:
            common &= set(cs.keys())
        hoisted = [csets[0][k].copy() for k in sorted(common)]
        reduced = []
        collapse = False
        for cs in csets:
            rest = [c.copy() for k, c in cs.items() if k not in common]
            if not rest:
                collapse = True
                break
            rb = rest[0]
            for c in rest[1:]:
                rb = E.And(this=rb, expression=c)
            reduced.append(E.Paren(this=rb))
        parts = list(hoisted)
        if not collapse and reduced:
            ob = reduced[0]
            for r in reduced[1:]:
                ob = E.Or(this=ob, expression=r)
            parts.append(E.Paren(this=ob))
            per_branch = []
            for cs in csets:
                pins = {}
                for k, c in cs.items():
                    if k in common: continue
                    if isinstance(c, E.EQ) and isinstance(c.this, E.Column) and isinstance(c.expression, E.Literal):
                        pins.setdefault(c.this.sql(), set()).add(c.expression.sql())
                    elif (isinstance(c, E.In) and isinstance(c.this, E.Column) and c.args.get('query') is None
                          and all(isinstance(x, E.Literal) for x in (c.args.get('expressions') or []))):
                        pins.setdefault(c.this.sql(), set()).update(x.sql() for x in c.args['expressions'])
                per_branch.append(pins)
            cols_all = set(per_branch[0].keys())
            for pb in per_branch[1:]:
                cols_all &= set(pb.keys())
            for colsql in sorted(cols_all):
                union = set()
                for pb in per_branch:
                    union |= pb[colsql]
                if len(union) > 64: continue
                colnode = next(c.this for cs in csets for k, c in cs.items()
                               if isinstance(c, (E.EQ, E.In)) and isinstance(c.this, E.Column) and c.this.sql() == colsql)
                lits = [sqlglot.parse_one(x, read='duckdb') for x in sorted(union)]
                parts.append(E.In(this=colnode.copy(), expressions=lits))
        return parts if parts else [ornode]
    out = []
    changed = False
    for cj in _ands(w.this):
        if isinstance(cj, E.Or) or (isinstance(cj, E.Paren) and isinstance(cj.this, E.Or)):
            rep = _factor_one(cj)
            if len(rep) != 1 or rep[0] is not cj: changed = True
            out.extend(rep)
        else:
            out.append(cj)
    if not changed: return
    new = out[0]
    for p in out[1:]:
        new = E.And(this=new, expression=p)
    w.set('this', new)


def _lonely_rewrite(db, tree):
    """Q21's EXISTS pair dissolved (Jackson): EXISTS(l2: same key, other supp)
    and NOT EXISTS(l3: same key, other supp, ALSO LATE) are two PARENT KEEPS
    -- distinct-supplier censuses over the child's sorted road: keep orders
    with nsupp >= 2 and late-distinct == 1 (the outer's own lateness conjunct
    covers l1's side). The pair is replaced by one In carrying _codes on the
    parent key; the mask layer serves it."""
    w = tree.args.get('where')
    if w is None: return
    def _flat9(x):
        if isinstance(x, E.Paren): return _flat9(x.this)
        if isinstance(x, E.And): return _flat9(x.this) + _flat9(x.expression)
        return [x]
    conj = _flat9(w.this)
    ex_pos = ex_neg = None
    for cj in conj:
        if isinstance(cj, E.Exists): ex_pos = cj
        elif isinstance(cj, E.Not) and isinstance(cj.this, E.Exists): ex_neg = cj
    if ex_pos is None or ex_neg is None: return
    def _parts(ex):
        sub = ex.this
        if not isinstance(sub, E.Select): return None
        f9 = sub.args.get('from') or sub.args.get('from_')
        if f9 is None or not isinstance(f9.this, E.Table): return None
        al = f9.this.alias or f9.this.name
        ww = sub.args.get('where')
        if ww is None: return None
        return f9.this.name, al, _flat9(ww.this)
    P1 = _parts(ex_pos); P2 = _parts(ex_neg.this)
    if P1 is None or P2 is None or P1[0] != P2[0]: return
    childT = P1[0]
    def _corr_neq(cjs, al):
        key_eq = neq = None; extra = []
        for c in cjs:
            if isinstance(c, E.EQ) and isinstance(c.this, E.Column) and isinstance(c.expression, E.Column):
                key_eq = c
            elif isinstance(c, E.NEQ) and isinstance(c.this, E.Column) and isinstance(c.expression, E.Column):
                neq = c
            else:
                extra.append(c)
        return key_eq, neq, extra
    k1, n1, e1 = _corr_neq(P1[2], P1[1])
    k2, n2, e2 = _corr_neq(P2[2], P2[1])
    if k1 is None or n1 is None or e1: return
    if k2 is None or n2 is None or len(e2) != 1: return
    # the NOT-EXISTS extra conjunct, re-aliased to the OUTER, must appear in the outer WHERE
    x2 = e2[0].copy()
    outer_al = None
    for c9 in x2.find_all(E.Column):
        if c9.table == P2[1]:
            pass
    inner_al = P2[1]
    outer_al = (k2.this.table if k2.this.table != inner_al else k2.expression.table) or ''
    for c9 in x2.find_all(E.Column):
        if c9.table == inner_al:
            c9.set('table', E.Identifier(this=outer_al, quoted=False) if outer_al else None)
    want = x2.sql()
    if not any(c.sql() == want for c in conj): return
    # key/supp/flag columns on the child
    key_c = k1.this if (k1.this.table or '') == P1[1] else k1.expression
    sup_c = n1.this if (n1.this.table or '') == P1[1] else n1.expression
    try:
        cseg, _s = _solo_segment(db, childT)
        # the parent of the child's key via the outer key column (other side of k1)
        okc = k1.expression if key_c is k1.this else k1.this
        # find the parent table owning the outer key through the OUTER's edges:
        pT = None
        for t9 in tree.find_all(E.Table):
            if t9.find_ancestor(E.Select) is not tree: continue
            if t9.name == childT and (t9.alias or t9.name) != (okc.table or ''):
                continue
        # outer key column l1.l_orderkey belongs to the fact alias; the road's
        # PARENT key is found through the catalog FK or unique-key equality in
        # the outer WHERE: o_orderkey = l1.l_orderkey
        peq = None
        for c9 in conj:
            if (isinstance(c9, E.EQ) and isinstance(c9.this, E.Column) and isinstance(c9.expression, E.Column)
                    and {c9.this.name, c9.expression.name} >= {okc.name} and c9 is not k1):
                a9, b9 = c9.this, c9.expression
                other = b9 if a9.name == okc.name and (a9.table or '') == (okc.table or '') else (
                        a9 if b9.name == okc.name and (b9.table or '') == (okc.table or '') else None)
                if other is not None:
                    peq = other; break
        if peq is None: return
        pT9 = None
        for t9 in tree.find_all(E.Table):
            if t9.find_ancestor(E.Select) is tree and peq.name in set(db.cat.column_names(t9.name)):
                pT9 = t9.name; break
        if pT9 is None or not _key_is_unique(db, pT9, peq.name): return
        pseg, _s2 = _solo_segment(db, pT9)
        road = np.asarray(_hash_pointer(db, childT, db.cat.phys_map(childT).get(key_c.name, key_c.name),
                                        cseg, pT9, db.cat.phys_map(pT9).get(peq.name, peq.name), pseg))
        if not bool((np.diff(road) >= 0).all()): return       # runs law: sorted roads only
        # the flag: the NOT-EXISTS extra conjunct evaluated on the CHILD --
        # only the declared-clock strict compare is served (Q21's late)
        f9n = e2[0]
        if not (type(f9n) in (E.GT, E.LT) and isinstance(f9n.this, E.Column)
                and isinstance(f9n.expression, E.Column)): return
        a9c, b9c = f9n.this.name, f9n.expression.name
        big, small = (a9c, b9c) if isinstance(f9n, E.GT) else (b9c, a9c)
        cc9 = cseg.cols.get(db.cat.phys_map(childT).get(big, big), {})
        if cc9.get('code_enc') != 16 or cc9.get('e16_partner') != db.cat.phys_map(childT).get(small, small):
            return
        bit9, dl9 = cseg.pair_bits(db.cat.phys_map(childT).get(big, big))
        flag = np.ascontiguousarray(np.asarray(bit9) & (np.asarray(dl9) > 0), dtype=np.bool_)
        sup9 = wdb_sql.raw_dict_col(cseg, db.cat.phys_map(childT).get(sup_c.name, sup_c.name), want_codes=True)
        if sup9 is None: return
        supv = np.ascontiguousarray(np.asarray(sup9[1]), dtype=np.int64)
        ch9 = np.flatnonzero(np.diff(road) != 0) + 1
        starts = np.concatenate((np.array([0]), ch9, np.array([road.shape[0]]))).astype(np.int64)
        nsupp = np.zeros(pseg.N, np.int32); nflag = np.zeros(pseg.N, np.int32)
        wdb_kernels.pruns_distinct(starts, np.ascontiguousarray(road, dtype=np.int64), supv, flag, nsupp, nflag)
        qual = np.flatnonzero((nsupp >= 2) & (nflag == 1)).astype(np.int64)
    except Exception:
        return
    # replace the pair with one In carrying _codes on the parent key
    innode = E.In(this=peq.copy())
    innode.set('_codes', qual)
    kept = [c for c in conj if c is not ex_pos and c is not ex_neg] + [innode]
    new = kept[0]
    for c in kept[1:]:
        new = E.And(this=new.copy() if new is kept[0] else new, expression=c.copy() if not isinstance(c, E.In) else c)
    w.set('this', new)



def _grouped_in_rewrite(db, tree):
    """THE WEIGHTED CENSUS SERVE (Jackson's Q18): `pkey IN (SELECT ckey FROM
    child GROUP BY ckey HAVING AGG(col) cmp lit)` fetches no foreign data --
    it keeps parents whose slot in a weighted bincount over the child's road
    clears the bar. Resolve to the literal key list (monsters are rare) and
    the query becomes a plain tree query. Declines loudly past 100k keys."""
    w = tree.args.get('where')
    if w is None: return
    for node in list(w.find_all(E.In)):
        q = node.args.get('query')
        if q is None: continue
        sub = q.this if isinstance(q, E.Subquery) else q
        if not isinstance(sub, E.Select): continue
        if node.find_ancestor(E.Select) is not None and node.find_ancestor(E.Select) is not tree: continue
        if sub.args.get('joins') or sub.args.get('where'): continue
        grp = sub.args.get('group'); hav = sub.args.get('having')
        if grp is None or hav is None or len(grp.expressions) != 1: continue
        gk = grp.expressions[0]
        proj = list(sub.expressions)
        if len(proj) != 1: continue
        p0 = proj[0].this if isinstance(proj[0], E.Alias) else proj[0]
        if not (isinstance(gk, E.Column) and isinstance(p0, E.Column) and p0.name == gk.name): continue
        frm = sub.args.get('from') or sub.args.get('from_')
        if frm is None or not isinstance(frm.this, E.Table): continue
        childT = frm.this.name
        hv = hav.this
        if type(hv) not in (E.GT, E.GTE, E.LT, E.LTE): continue
        ag = hv.this; lit = hv.expression
        if not (isinstance(lit, E.Literal) and not lit.is_string): continue
        thr = float(lit.this)
        if isinstance(ag, E.Count) and (ag.this is None or isinstance(ag.this, E.Star)):
            wcol = None
        elif isinstance(ag, E.Sum) and isinstance(ag.this, E.Column):
            wcol = ag.this.name
        else:
            continue
        oc = node.this
        if not isinstance(oc, E.Column): continue
        # which alias/table owns the outer column
        parentT = None
        for t9 in tree.find_all(E.Table):
            if t9.find_ancestor(E.Select) is tree and oc.name in set(db.cat.column_names(t9.name)):
                if oc.table and (t9.alias or t9.name) != oc.table: continue
                parentT = t9.name; break
        if parentT is None or not _key_is_unique(db, parentT, oc.name): continue
        try:
            cseg, _s1 = _solo_segment(db, childT)
            pseg, _s2 = _solo_segment(db, parentT)
            road = np.asarray(_hash_pointer(db, childT, db.cat.phys_map(childT).get(gk.name, gk.name),
                                            cseg, parentT, db.cat.phys_map(parentT).get(oc.name, oc.name), pseg))
            cache = getattr(pseg, '_census_cache', None)
            if cache is None: cache = pseg._census_cache = {}
            ck9 = (childT, gk.name, wcol)
            qsum = cache.get(ck9)
            if qsum is None:
                if wcol is None:
                    qsum = np.bincount(road, minlength=pseg.N).astype(np.float64)
                else:
                    rw = wdb_sql.raw_dict_col(cseg, db.cat.phys_map(childT).get(wcol, wcol))
                    if rw is None: continue
                    qsum = np.bincount(road, weights=rw[0][np.asarray(rw[1])], minlength=pseg.N)
                cache[ck9] = qsum
            op9 = {E.GT: np.greater, E.GTE: np.greater_equal,
                   E.LT: np.less, E.LTE: np.less_equal}[type(hv)]
            prows = np.flatnonzero(op9(qsum, thr))
            if prows.size > 100000: continue                 # stays a loud hole
            keys = np.asarray(wdb_sql._col(pseg, db.cat.phys_map(parentT).get(oc.name, oc.name))[0])[prows]
        except Exception:
            continue
        node.set('expressions', [E.Literal(this=str(int(k)), is_string=False) for k in keys.tolist()])
        node.set('query', None)



def _window_topk_door(db, tree):
    """THE TOP-K-PER-GROUP DOOR (H2O q8): FROM (SELECT cols..., ROW_NUMBER()
    OVER (PARTITION BY p ORDER BY o [DESC]) AS rn FROM T [WHERE inner]) t
    WHERE rn <= k. Dissolved: composite group ids over p, a COUNTING SCATTER
    of row indices by group, each group's slice sorted by o in parallel,
    the first k rows gathered. ROW_NUMBER only (RANK ties would need more)."""
    frm = tree.args.get('from') or tree.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Subquery) or tree.args.get('joins'): return None
    inner = frm.this.this
    if not isinstance(inner, E.Select) or inner.args.get('group') or inner.args.get('joins'): return None
    ifrm = inner.args.get('from') or inner.args.get('from_')
    if ifrm is None or not isinstance(ifrm.this, E.Table): return None
    tname = ifrm.this.name
    wins = [p for p in inner.expressions if isinstance(p, E.Alias) and isinstance(p.this, E.Window)]
    if len(wins) != 1: return None
    walias = wins[0].alias; w = wins[0].this
    if not isinstance(w.this, E.RowNumber): return None
    part = [c for c in (w.args.get('partition_by') or [])]
    if not part or not all(isinstance(c, E.Column) for c in part): return None
    order = w.args.get('order')
    if order is None or len(order.expressions) != 1 or not isinstance(order.expressions[0].this, E.Column): return None
    ocol = order.expressions[0].this.name; desc = bool(order.expressions[0].args.get('desc'))
    ow = tree.args.get('where')
    if ow is None or tree.args.get('group'): return None
    cj = ow.this
    if isinstance(cj, E.Paren): cj = cj.this
    if not (isinstance(cj, (E.LTE, E.LT, E.EQ)) and isinstance(cj.this, E.Column) and cj.this.name == walias
            and isinstance(cj.expression, E.Literal)): return None
    k = int(cj.expression.this)
    if isinstance(cj, E.LT): k -= 1
    if k <= 0 or k > 1000: return None
    outcols = []
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        if not isinstance(nd, E.Column) or nd.name == walias: return None
        outcols.append(nd.name)
    try:
        seg, _sp = _solo_segment(db, tname)
    except _FastUnsupported:
        return None
    pm = db.cat.phys_map(tname)
    n = int(seg.N)
    # inner WHERE -> survivor rows
    iw = inner.args.get('where')
    if iw is not None:
        m = wdb_sql._eval_pred(seg, iw.this, lambda nm: pm.get(nm, nm))
        rows = np.flatnonzero(np.asarray(m, dtype=bool))
    else:
        rows = np.arange(n, dtype=np.int64)
    comp = np.zeros(rows.size, np.int64); K = 1
    for c in part:
        pc = pm.get(c.name, c.name); cd = seg.cols.get(pc, {})
        if cd.get('has_null') or cd.get('mode') not in (0, 1, 2, 4): return None
        codes = np.asarray(seg.codes_at(pc, rows)).astype(np.int64, copy=False)
        V = int(cd.get('V') or (int(codes.max()) + 1 if codes.size else 1))
        if K * V > (1 << 24): return None
        comp = comp * V + codes; K *= V
    raw = wdb_sql.raw_dict_col(seg, pm.get(ocol, ocol))
    if raw is None: return None
    vals = raw[0][np.asarray(raw[1])].astype(np.float64, copy=False)
    cnt_all = np.bincount(comp, minlength=K)
    present = np.flatnonzero(cnt_all)
    offs0 = np.zeros(K + 1, np.int64); np.cumsum(cnt_all, out=offs0[1:])
    cur = offs0[:-1].copy()
    placed = np.empty(rows.size, np.int64)
    wdb_kernels.pscatter_by_gid(np.ascontiguousarray(comp), np.ascontiguousarray(rows), cur, placed)
    starts = offs0[present]; ends = offs0[present + 1]
    out_rows = np.empty(present.size * k, np.int64); out_cnt = np.zeros(present.size, np.int64)
    wdb_kernels.pgroup_topk(placed, vals, starts, ends, k, desc, out_rows, out_cnt)
    keep = np.zeros(present.size * k, bool)
    for t in range(k):
        keep[t::k] = out_cnt > t
    sel = out_rows[keep]
    cols = []
    for nm in outcols:
        pc = pm.get(nm, nm)
        cols.append(list(seg.values_at_rows(pc, sel)) if seg.cols.get(pc, {}).get('mode') == 5
                    else _bulk_keyvals(seg, pc, np.asarray(seg.codes_at(pc, sel))))
    out9 = [tuple(col[i] for col in cols) for i in range(sel.size)]
    out9 = wdb_sql._apply_order(out9, list(tree.expressions), tree.args.get('order'))   # outer ORDER BY (DISTINCT ON)
    lim9 = tree.args.get('limit')
    if lim9 is not None: out9 = out9[:int(lim9.expression.this)]
    return out9



def _join_pointer(db, ctbl, ckey, cseg, ptbl, pkey, pseg):
    """A child->parent pointer that KEEPS -1 for unmatched child rows (the
    road refuses them; INNER drops them, LEFT emits NULLs). Cached per
    process on the child segment."""
    cache = getattr(cseg, '_jptr_cache', None)
    if cache is None:
        cache = cseg._jptr_cache = {}
    k = (ckey, ptbl, pkey)
    got = cache.get(k)
    if got is not None:
        return got
    import pandas as pd
    pk = np.asarray(wdb_sql._col(pseg, pkey)[0])
    ck = np.asarray(wdb_sql._col(cseg, ckey)[0])
    if pk.dtype.kind in 'OSU' or ck.dtype.kind in 'OSU':
        pk = pk.astype(object); ck = ck.astype(object)
    pidx = pd.Index(pk)
    if not pidx.is_unique: raise _FastUnsupported                 # many-to-many -> not a pointer
    ptr = np.ascontiguousarray(pidx.get_indexer(ck), dtype=np.int64)
    cache[k] = ptr
    return ptr


def _road_join_emit(db, tree):
    """THE ROAD JOIN (H2O joins): a row-emitting two-table equi-join whose
    right key is unique is a POINTER, not a merge -- the child's columns
    gather at its rows, the parent's at the pointer. INNER drops -1 rows;
    LEFT keeps them and emits NULLs for the parent side. Returns
    (rows, names) or None (not this shape)."""
    if not isinstance(tree, E.Select): return None
    joins = tree.args.get('joins') or []
    if len(joins) != 1 or tree.args.get('group') is not None: return None
    if tree.args.get('having') is not None: return None
    jn = joins[0]
    side = (jn.args.get('side') or '').upper(); kind = (jn.args.get('kind') or '').upper()
    if kind not in ('', 'INNER', 'OUTER') or side not in ('', 'LEFT', 'RIGHT', 'FULL'): return None
    frm = tree.args.get('from') or tree.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table) or not isinstance(jn.this, E.Table): return None
    lt, la = frm.this.name, (frm.this.alias or frm.this.name)
    rt, ra = jn.this.name, (jn.this.alias or jn.this.name)
    on = jn.args.get('on')
    if not (isinstance(on, E.EQ) and isinstance(on.this, E.Column) and isinstance(on.expression, E.Column)): return None
    lcols = set(db.cat.column_names(lt)); rcols = set(db.cat.column_names(rt))
    def own(c):
        if c.table: return c.table
        if c.name in lcols and c.name not in rcols: return la
        if c.name in rcols and c.name not in lcols: return ra
        return None
    a, b = on.this, on.expression
    if own(a) == ra: a, b = b, a
    if own(a) != la or own(b) != ra: return None
    proj = list(tree.expressions)
    for p in proj:
        nd = p.this if isinstance(p, E.Alias) else p
        if not isinstance(nd, E.Column) or own(nd) not in (la, ra): return None
    try:
        if not _key_is_unique(db, rt, b.name): return None
        lseg, _l = _solo_segment(db, lt); rseg, _r = _solo_segment(db, rt)
    except _FastUnsupported:
        return None
    lpm = db.cat.phys_map(lt); rpm = db.cat.phys_map(rt)
    # WHERE on the road: single-sided conjuncts filter their side BEFORE the pointer is read;
    # a conjunct touching both sides is applied to the joined rows
    _where9 = tree.args.get('where')
    def _fl9(x):
        if isinstance(x, E.Paren): return _fl9(x.this)
        if isinstance(x, E.And): return _fl9(x.this) + _fl9(x.expression)
        return [x]
    _conj9 = _fl9(_where9.this) if _where9 is not None else []
    def _sides9(cj): return {own(c) for c in cj.find_all(E.Column)}
    def _strip9(cj):
        c2 = cj.copy()
        for c in c2.find_all(E.Column): c.set('table', None)
        return c2
    def _null_side_truth9(cj, null_alias):
        """truth of a conjunct when every column of null_alias is NULL: only IS NULL survives"""
        if isinstance(cj, E.Is) and isinstance(cj.this, E.Column) and own(cj.this) == null_alias and isinstance(cj.expression, E.Null):
            return not bool(cj.args.get('not'))
        if isinstance(cj, E.Or):
            return _null_side_truth9(cj.this, null_alias) or _null_side_truth9(cj.expression, null_alias)
        return False
    for cj in _conj9:
        if None in _sides9(cj) or cj.find(E.Subquery) is not None: return None
    ptr = _join_pointer(db, lt, lpm.get(a.name, a.name), lseg, rt, rpm.get(b.name, b.name), rseg)
    n = int(lseg.N)
    rmiss = None                                      # RIGHT/FULL: parent rows nobody points at (NULL child side)
    if side == 'LEFT':
        rows = np.arange(n, dtype=np.int64); prow = ptr
        miss = prow < 0
    elif side in ('RIGHT', 'FULL'):
        if side == 'FULL':
            rows = np.arange(n, dtype=np.int64); prow = ptr; miss = prow < 0
        else:
            rows = np.flatnonzero(ptr >= 0); prow = ptr[rows]; miss = None
        hit = np.zeros(int(rseg.N), bool); hit[ptr[ptr >= 0]] = True
        rmiss = np.flatnonzero(~hit)
    else:
        rows = np.flatnonzero(ptr >= 0); prow = ptr[rows]; miss = None
    import wdb_govern
    wdb_govern.ask(int(rows.size) + (int(rmiss.size) if rmiss is not None else 0), len(proj), 'join result')   # THE GOVERNOR
    if _conj9:
        keep = np.ones(int(rows.size), bool)
        for cj in _conj9:
            sd = _sides9(cj)
            if sd == {la}:
                keep &= np.asarray(wdb_sql._eval_pred(lseg, _strip9(cj), lambda nm: lpm.get(nm, nm)), dtype=bool)[rows]
            elif sd == {ra}:
                m_r = np.asarray(wdb_sql._eval_pred(rseg, _strip9(cj), lambda nm: rpm.get(nm, nm)), dtype=bool)
                if miss is not None:
                    kk = np.zeros(int(rows.size), bool); okp = ~miss
                    kk[okp] = m_r[prow[okp]]; kk[miss] = _null_side_truth9(cj, ra)
                    keep &= kk
                else:
                    keep &= m_r[prow]
            else:
                return None                                  # a two-sided conjunct: the general joiner
        rows = rows[keep]; prow = prow[keep]
        if miss is not None: miss = miss[keep]
    def col_vals(seg, pc, rr):
        # MODE-AWARE point reads: plain dict numerics ride base[codes_at]
        # (cached base), everything else the general values_at_rows --
        # a float column mis-read through the key-decoder shipped j1 wrong.
        raw = wdb_sql.raw_dict_col(seg, pc, want_codes=False)
        if raw is not None:
            return raw[0][np.asarray(seg.codes_at(pc, rr))].tolist()
        cd = seg.cols.get(pc, {})
        if cd.get('mode') in (0, 1, 2) and cd.get('dt') == 1 and not cd.get('has_null'):
            return _bulk_keyvals(seg, pc, np.asarray(seg.codes_at(pc, rr)))   # dict strings: decode once per V
        if cd.get('mode') == 5 and int(seg.N) <= 4_000_000 and rr.size > 4 * int(seg.N):
            # SMALL MODE-5 SIDE READ MANY TIMES (a 10-row dim at 10M rows): decode
            # the whole column ONCE to an object array and gather -- the inline
            # slicer per row cost 3.9s here.
            cache = getattr(seg, '_m5_obj_cache', None)
            if cache is None:
                cache = seg._m5_obj_cache = {}
            arr = cache.get(pc)
            if arr is None:
                arr = np.array(list(seg.values_at_rows(pc, np.arange(int(seg.N), dtype=np.int64))), dtype=object)
                cache[pc] = arr
            return arr[rr].tolist()
        return list(seg.values_at_rows(pc, rr))
    def one_col(p):
        nd = p.this if isinstance(p, E.Alias) else p
        if own(nd) == la:
            return col_vals(lseg, lpm.get(nd.name, nd.name), rows)
        pc = rpm.get(nd.name, nd.name)
        if miss is not None and miss.any():
            safe = np.where(miss, 0, prow)
            v = col_vals(rseg, pc, safe)
            mi = miss.tolist()
            return [None if mi[i] else v[i] for i in range(len(v))]
        return col_vals(rseg, pc, prow)
    _bj = os.environ.get('WDB_JOIN_BILL')
    cols = []
    for p in proj:
        _t0 = time.perf_counter() if _bj else 0.0
        cols.append(one_col(p))                     # (threads measured: no gain -- GIL-bound materialisation)
        if _bj: print('ROAD-JOIN col %s: %.0fms' % (wdb_sql._alias(p), (time.perf_counter() - _t0) * 1e3), flush=True)
    _t0 = time.perf_counter() if _bj else 0.0
    rows_out = list(zip(*cols)) if cols else []
    if _bj: print('ROAD-JOIN zip: %.0fms' % ((time.perf_counter() - _t0) * 1e3), flush=True)
    if rmiss is not None and rmiss.size and _where9 is not None:
        # the WHERE applies to unmatched-parent rows too: child columns are NULL there
        keep_r = np.ones(int(rmiss.size), bool)
        for cj in _conj9:
            sd = _sides9(cj)
            if sd == {ra}:
                keep_r &= np.asarray(wdb_sql._eval_pred(rseg, _strip9(cj), lambda nm: rpm.get(nm, nm)), dtype=bool)[rmiss]
            else:
                # any conjunct that needs a child column sees NULL: IS NULL passes, everything else fails
                keep_r &= _null_side_truth9(cj, la)
        rmiss = rmiss[keep_r]
    if rmiss is not None and rmiss.size:
        # unmatched PARENT rows: child columns NULL, parent columns read at rmiss
        extra = []
        for p in proj:
            nd = p.this if isinstance(p, E.Alias) else p
            if own(nd) == la: extra.append([None] * int(rmiss.size))
            else: extra.append(col_vals(rseg, rpm.get(nd.name, nd.name), rmiss))
        rows_out += list(zip(*extra))
    rows_out = wdb_sql._apply_order(rows_out, proj, tree.args.get('order'))
    lim = tree.args.get('limit')
    if lim is not None: rows_out = rows_out[:int(lim.expression.this)]
    return rows_out, [wdb_sql._alias(p) for p in proj]



def hidden_rewrite(tree):
    """Three rewrites into shapes every door already serves:
    GROUP BY ALL -> the non-aggregate projections; HAVING over a non-projected
    aggregate -> a hidden projection, stripped after; ORDER BY an expression
    over projected aliases -> a hidden projection, stripped after.
    Returns (sql, n_hidden) or None."""
    if not isinstance(tree, E.Select): return None
    _g0 = tree.args.get('group')
    if tree.args.get('having') is None and tree.args.get('order') is None and not (_g0 is not None and _g0.args.get('all')):
        return None                       # nothing to rewrite: the copy below was 5 ms of Q29's 9 (ninety projections)
    t = tree.copy(); changed = False
    _AGG = (E.AggFunc,)
    g = t.args.get('group')
    if g is not None and not g.expressions and any(isinstance(x, E.Column) and x.name.upper() == 'ALL' for x in [g]) :
        pass
    if g is not None and g.args.get('all'):
        keys = [(p.this if isinstance(p, E.Alias) else p).copy() for p in t.expressions
                if (p.this if isinstance(p, E.Alias) else p).find(*_AGG) is None]
        t.set('group', E.Group(expressions=keys)); changed = True
    defs = {p.alias: p.this for p in t.expressions if isinstance(p, E.Alias)}
    def _subst(x):                                        # projected aliases -> their definitions
        x = x.copy()
        for c in list(x.find_all(E.Column)):
            if not c.table and c.name in defs:
                c.replace(defs[c.name].copy())
        return x
    hidden = 0
    h = t.args.get('having')
    if h is not None:
        for ag in list(h.this.find_all(*_AGG)):
            if not any(ag == (p.this if isinstance(p, E.Alias) else p) for p in t.expressions):
                al = '__h%d' % hidden; hidden += 1        # project it hidden; HAVING keeps the aggregate text
                t.set('expressions', list(t.expressions) + [E.Alias(this=ag.copy(), alias=E.Identifier(this=al, quoted=False))])
                changed = True
    o = t.args.get('order')
    if o is not None:
        proj_sqls = {(p.this if isinstance(p, E.Alias) else p).sql() for p in t.expressions}
        for oe in o.expressions:
            x = oe.this
            if isinstance(x, (E.Column, E.Literal, E.AggFunc)): continue   # doors order by aggregates natively
            if x.sql() in proj_sqls: continue                              # already projected
            al = '__o%d' % hidden; hidden += 1
            t.set('expressions', list(t.expressions) + [E.Alias(this=_subst(x), alias=E.Identifier(this=al, quoted=False))])
            oe.set('this', E.Column(this=E.Identifier(this=al, quoted=False))); changed = True
    if not changed: return None
    return t.sql(dialect='duckdb'), hidden


def qualify_rewrite(tree):
    """QUALIFY <window> <cmp> <lit>  ->  SELECT cols FROM (SELECT cols, <window>
    AS __q FROM ...) t WHERE __q <cmp> <lit>: the top-k-per-group door's shape.
    Returns SQL or None. (The clause was silently IGNORED before, 2026-09-08.)"""
    q = tree.args.get('qualify')
    if q is None: return None
    cond = q.this
    if isinstance(cond, E.Paren): cond = cond.this
    if not (type(cond) in (E.EQ, E.LTE, E.LT) and isinstance(cond.this, E.Window)
            and isinstance(cond.expression, E.Literal)):
        return None                                   # the window door evaluates general QUALIFY over its output
    inner = tree.copy(); inner.set('qualify', None)
    inner.set('order', None); inner.set('limit', None)
    inner.set('expressions', list(inner.expressions) + [E.Alias(this=cond.this.copy(), alias=E.Identifier(this='__q', quoted=False))])
    names = [p.alias_or_name for p in tree.expressions]
    outer_cols = ', '.join(names)
    op = {E.EQ: '=', E.LTE: '<=', E.LT: '<'}[type(cond)]
    tail = ''
    if tree.args.get('order') is not None: tail += ' ' + tree.args['order'].sql(dialect='duckdb')
    if tree.args.get('limit') is not None: tail += ' ' + tree.args['limit'].sql(dialect='duckdb')
    return 'SELECT %s FROM (%s) t WHERE __q %s %s%s' % (outer_cols, inner.sql(dialect='duckdb'), op, cond.expression.sql(), tail)


def distinct_on_rewrite(tree):
    """DISTINCT ON (k...) cols ORDER BY k..., rest  ->  ROW_NUMBER() OVER
    (PARTITION BY k ORDER BY rest) = 1 through the top-k-per-group door.
    Returns SQL or None. (The clause was silently IGNORED before.)"""
    d = tree.args.get('distinct')
    if d is None or d.args.get('on') is None: return None
    on = d.args['on']
    keys = list(on.expressions) if isinstance(on, E.Tuple) else [on]
    if not all(isinstance(k, E.Column) for k in keys): raise NotImplementedError('DISTINCT ON non-column keys')
    order = tree.args.get('order')
    if order is None: raise NotImplementedError('DISTINCT ON without ORDER BY (arbitrary pick)')
    knames = [k.name for k in keys]
    rest = [o for o in order.expressions if not (isinstance(o.this, E.Column) and o.this.name in knames)]
    if not rest: raise NotImplementedError('DISTINCT ON needs an ORDER BY beyond its keys')
    win = 'ROW_NUMBER() OVER (PARTITION BY %s ORDER BY %s)' % (', '.join(knames), ', '.join(o.sql(dialect='duckdb') for o in rest))
    inner = tree.copy(); inner.set('distinct', None); inner.set('order', None); inner.set('limit', None)
    inner.set('expressions', list(inner.expressions) + [sqlglot.parse_one('SELECT %s AS __q' % win, read='duckdb').expressions[0]])
    names = [p.alias_or_name for p in tree.expressions]
    tail = ' ' + order.sql(dialect='duckdb')
    if tree.args.get('limit') is not None: tail += ' ' + tree.args['limit'].sql(dialect='duckdb')
    return 'SELECT %s FROM (%s) t WHERE __q = 1%s' % (', '.join(names), inner.sql(dialect='duckdb'), tail)



def _window_door(db, tree):
    """THE WINDOW DOOR: single-table SELECT with window projections. Survivors,
    composite partition ids, ONE lexsort by (partition, order), boundaries,
    then each function as vectorised arithmetic over the sorted order:
    ROW_NUMBER, RANK, DENSE_RANK, NTILE, LAG/LEAD, FIRST_VALUE, and
    SUM/COUNT/AVG/MIN/MAX OVER (partition totals, or running when ORDER BY).
    Returns (rows, names) or None (not this shape)."""
    if not isinstance(tree, E.Select): return None
    if tree.args.get('group') is not None or tree.args.get('joins') or tree.args.get('having') is not None: return None
    frm = tree.args.get('from') or tree.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table): return None
    proj = list(tree.expressions)
    _ARITH9 = (E.Add, E.Sub, E.Mul, E.Div, E.Paren, E.Neg)
    def _win_expr_ok(nd):
        """a Window, or arithmetic with literals around exactly one Window"""
        if isinstance(nd, E.Window): return True
        if isinstance(nd, E.Literal): return True
        if isinstance(nd, _ARITH9): return all(_win_expr_ok(c) for c in nd.args.values() if isinstance(c, E.Expression))
        return False
    wins = [(i, p) for i, p in enumerate(proj) if isinstance(p, E.Alias) and p.this.find(E.Window) is not None]
    if not wins: return None
    for p in proj:
        nd = p.this if isinstance(p, E.Alias) else p
        if isinstance(nd, E.Column): continue
        if not (_win_expr_ok(nd) and len(list(nd.find_all(E.Window))) == 1): return None
    tname = frm.this.name
    try:
        seg, _sp = _solo_segment(db, tname)
    except _FastUnsupported:
        return None
    pm = db.cat.phys_map(tname)
    n = int(seg.N)
    w = tree.args.get('where')
    if w is not None:
        m = wdb_sql._eval_pred(seg, w.this, lambda nm: pm.get(nm, nm))
        rows = np.flatnonzero(np.asarray(m, dtype=bool))
    else:
        rows = np.arange(n, dtype=np.int64)
    R = rows.size
    def col_at(nm, rr):
        pc = pm.get(nm, nm)
        raw = wdb_sql.raw_dict_col(seg, pc, want_codes=False)
        if raw is not None:
            v = raw[0][np.asarray(seg.codes_at(pc, rr))]
            if seg.cols.get(pc, {}).get('dt') == 3 and np.asarray(v).dtype.kind != 'M':
                v = np.asarray(v).astype(np.int64).view('datetime64[%s]' % seg.unit(pc))   # a date column IS datetime64
            return v
        return np.asarray(_bulk_keyvals(seg, pc, np.asarray(seg.codes_at(pc, rr))), dtype=object)
    def sort_key(node, rr):
        """numeric sort key for an order expression (strings by dict code)."""
        if not isinstance(node, E.Column): raise _FastUnsupported
        pc = pm.get(node.name, node.name)
        cd = seg.cols.get(pc, {})
        if cd.get('dt') == 1:
            return np.asarray(seg.codes_at(pc, rr)).astype(np.float64)      # sorted dict: code order = value order
        raw = wdb_sql.raw_dict_col(seg, pc, want_codes=False)
        if raw is None: raise _FastUnsupported
        return raw[0][np.asarray(seg.codes_at(pc, rr))].astype(np.float64)
    out_cols = {}
    def _arith9(nd, wv):
        if isinstance(nd, E.Window): return wv
        if isinstance(nd, E.Literal): return float(nd.this)
        if isinstance(nd, E.Paren): return _arith9(nd.this, wv)
        if isinstance(nd, E.Neg): return -_arith9(nd.this, wv)
        a, b = _arith9(nd.this, wv), _arith9(nd.expression, wv)
        if isinstance(nd, E.Add): return a + b
        if isinstance(nd, E.Sub): return a - b
        if isinstance(nd, E.Mul): return a * b
        return a / b
    for i, p in wins:
        outer9 = p.this
        win = outer9.find(E.Window); fn = win.this
        part = list(win.args.get('partition_by') or [])
        order = win.args.get('order')
        ords = list(order.expressions) if order is not None else []
        spec = win.args.get('spec')
        # THE FRAME LAW: frames are named -- ROWS UNBOUNDED PRECEDING (cumulative), ROWS k PRECEDING
        # (sliding), and the default RANGE frame (peers share the running value); anything else
        # (FOLLOWING, RANGE with offsets) declines by name
        frame_k = None; range_peers = False
        if spec is not None:
            _k9 = (spec.args.get('kind') or '').upper(); _st9 = str(spec.args.get('start') or '').upper(); _en9 = str(spec.args.get('end') or 'CURRENT ROW').upper()
            _sside9 = str(spec.args.get('start_side') or '').upper(); _eside9 = str(spec.args.get('end_side') or '').upper()
            if 'CURRENT' not in _en9 and _en9 != '':
                raise NotImplementedError('window frame %s not served by the window door' % spec.sql()[:50])
            if _k9 == 'ROWS' and 'UNBOUNDED' in _st9: frame_k = None
            elif _k9 == 'ROWS' and _st9.isdigit() and 'PRECEDING' in _sside9: frame_k = int(_st9)
            elif _k9 == 'RANGE' and 'UNBOUNDED' in _st9: range_peers = True
            else:
                raise NotImplementedError('window frame %s not served by the window door' % spec.sql()[:50])
        elif ords and type(fn).__name__ in ('Sum', 'Count', 'Avg', 'Min', 'Max'):
            range_peers = True                                   # SQL's default frame
        # partition ids
        if part:
            gid = np.zeros(R, np.int64); K = 1
            for c in part:
                if not isinstance(c, E.Column): raise _FastUnsupported
                pc = pm.get(c.name, c.name); cd = seg.cols.get(pc, {})
                V = int(cd.get('V') or 1)
                codes = np.asarray(seg.codes_at(pc, rows)).astype(np.int64)
                if K * V > (1 << 62): raise _FastUnsupported
                gid = gid * V + codes; K *= V
            _u, gid = np.unique(gid, return_inverse=True)
        else:
            gid = np.zeros(R, np.int64)
        keys = [gid]
        for o in ords:
            k = sort_key(o.this, rows)
            keys.append(-k if o.args.get('desc') else k)
        if not ords:
            # PARTITION-ONLY windows need no sort: a counting scatter by group id (O(N))
            G9 = int(gid.max()) + 1 if R else 0
            cnt9 = np.bincount(gid, minlength=G9)
            offs9 = np.zeros(G9 + 1, np.int64); np.cumsum(cnt9, out=offs9[1:])
            cur9 = offs9[:-1].copy()
            order_idx = np.empty(R, np.int64)
            wdb_kernels.pscatter_by_gid(np.ascontiguousarray(gid), np.arange(R, dtype=np.int64), cur9, order_idx)
        else:
            order_idx = np.lexsort(tuple(reversed(keys)))      # primary = gid, then order keys
        g_sorted = gid[order_idx]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(g_sorted) != 0) + 1))
        seg_id = np.cumsum(np.concatenate(([0], (np.diff(g_sorted) != 0).astype(np.int64))))   # partition index per sorted pos
        pos_in = np.arange(R) - starts[seg_id]                  # 0-based position within partition
        sizes = np.diff(np.concatenate((starts, [R])))
        fname = type(fn).__name__
        val = None
        if fname == 'RowNumber':
            val = pos_in + 1
        elif fname in ('Rank', 'DenseRank'):
            if not ords: raise _FastUnsupported
            ok = np.stack([keys[j + 1][order_idx] for j in range(len(ords))], axis=1)
            new = np.ones(R, bool)
            new[1:] = (g_sorted[1:] != g_sorted[:-1]) | np.any(ok[1:] != ok[:-1], axis=1)
            if fname == 'DenseRank':
                dr = np.cumsum(new) - 1
                val = dr - (np.cumsum(new) - 1)[starts][seg_id] + 1
            else:
                idx_new = np.where(new, np.arange(R), 0)
                last_new = np.maximum.accumulate(idx_new)
                val = last_new - starts[seg_id] + 1
        elif fname == 'Ntile':
            nb = int(fn.this.this)
            sz = sizes[seg_id]; base = sz // nb; rem = sz % nb
            # first `rem` buckets get base+1 rows
            cut = rem * (base + 1)
            val = np.where(pos_in < cut, pos_in // np.maximum(base + 1, 1) + 1,
                           rem + (pos_in - cut) // np.maximum(base, 1) + 1)
        elif fname in ('Lag', 'Lead'):
            col = fn.this
            off = int(fn.args['offset'].this) if fn.args.get('offset') is not None else 1
            if not isinstance(col, E.Column): raise _FastUnsupported
            vals = col_at(col.name, rows[order_idx])
            shifted = np.empty(R, dtype=object)
            if fname == 'Lag':
                src = np.arange(R) - off; okm = pos_in >= off
            else:
                src = np.arange(R) + off; okm = pos_in + off < sizes[seg_id]
            src = np.clip(src, 0, R - 1)
            picked = np.asarray(vals, dtype=object)[src]
            val = np.where(okm, picked, None)
        elif fname == 'FirstValue':
            col = fn.this
            if not isinstance(col, E.Column): raise _FastUnsupported
            vals = np.asarray(col_at(col.name, rows[order_idx]), dtype=object)
            val = vals[starts[seg_id]]
        elif fname in ('Sum', 'Count', 'Avg', 'Min', 'Max'):
            arg = fn.this
            if fname == 'Count' and (arg is None or isinstance(arg, E.Star)):
                x = np.ones(R, np.float64)
            else:
                if not isinstance(arg, E.Column): raise _FastUnsupported
                x = np.asarray(col_at(arg.name, rows[order_idx]), dtype=np.float64)
            if ords and frame_k is not None:                    # SLIDING: ROWS k PRECEDING .. CURRENT ROW
                cs0 = np.concatenate(([0.0], np.cumsum(x)))
                lo_i = np.maximum(np.arange(R) - frame_k, starts[seg_id])
                nrow = np.arange(R) - lo_i + 1
                if fname in ('Sum', 'Count'): val = cs0[np.arange(R) + 1] - cs0[lo_i]
                elif fname == 'Avg': val = (cs0[np.arange(R) + 1] - cs0[lo_i]) / nrow
                else:
                    # min/max over a small sliding window: stack the k+1 shifted views (k <= 512)
                    if frame_k > 512: raise NotImplementedError('sliding MIN/MAX over a %d-row frame' % frame_k)
                    val = x.copy()
                    for sh in range(1, frame_k + 1):
                        shifted = np.concatenate((np.full(sh, np.nan), x[:-sh])) if sh < R else np.full(R, np.nan)
                        okm = (np.arange(R) - sh) >= starts[seg_id]
                        cand = np.where(okm, shifted, np.nan)
                        val = np.fmin(val, cand) if fname == 'Min' else np.fmax(val, cand)
            elif ords:                                          # running: cumulative, or RANGE peers
                cs = np.cumsum(x); base = np.concatenate(([0.0], cs))[starts][seg_id]
                if range_peers:
                    # peers (equal order keys within a partition) share the value at the LAST peer
                    ok = np.stack([keys[j + 1][order_idx] for j in range(len(ords))], axis=1)
                    new = np.ones(R, bool); new[1:] = (g_sorted[1:] != g_sorted[:-1]) | np.any(ok[1:] != ok[:-1], axis=1)
                    tie_id = np.cumsum(new) - 1
                    last = np.zeros(int(tie_id.max()) + 1, np.int64); np.maximum.at(last, tie_id, np.arange(R))
                    at = last[tie_id]
                    run = cs[at] - base
                    cnt = (at - starts[seg_id] + 1)
                else:
                    run = cs - base; cnt = pos_in + 1
                if fname in ('Sum', 'Count'): val = run
                elif fname == 'Avg': val = run / cnt
                else:
                    acc = np.minimum.accumulate if fname == 'Min' else np.maximum.accumulate
                    val = np.empty(R)
                    for gi9 in range(starts.size):
                        a9 = starts[gi9]; b9 = a9 + sizes[gi9]; val[a9:b9] = acc(x[a9:b9])
                    if range_peers: val = val[at]
            else:                                               # partition totals
                tot = np.add.reduceat(x, starts)
                if fname in ('Sum', 'Count'): val = tot[seg_id]
                elif fname == 'Avg': val = (tot / sizes)[seg_id]
                else:
                    red = np.minimum.reduceat if fname == 'Min' else np.maximum.reduceat
                    val = red(x, starts)[seg_id]
            if fname == 'Count' or (fname == 'Sum' and arg is not None and isinstance(arg, E.Column)
                                    and seg.cols.get(pm.get(arg.name, arg.name), {}).get('dt') == 0):
                val = np.asarray(val).astype(np.int64)
        else:
            raise NotImplementedError('window function %s' % fname)
        if outer9 is not win:
            val = _arith9(outer9, np.asarray(val, dtype=np.float64))
        # back to survivor order
        inv = np.empty(R, np.int64); inv[order_idx] = np.arange(R)
        out_cols[i] = np.asarray(val, dtype=object)[inv] if not isinstance(val, np.ndarray) or val.dtype == object else val[inv]
    names9 = [wdb_sql._alias(p) for p in proj]
    # column ARRAYS first (numeric where the column is numeric); QUALIFY filters the
    # arrays before a single row is materialised
    arrs = []
    for i, p in enumerate(proj):
        if i in out_cols:
            arrs.append(out_cols[i])
        else:
            nd = p.this if isinstance(p, E.Alias) else p
            pc = pm.get(nd.name, nd.name)
            raw = wdb_sql.raw_dict_col(seg, pc, want_codes=False)
            if raw is not None:
                _v9 = raw[0][np.asarray(seg.codes_at(pc, rows))]
                if seg.cols.get(pc, {}).get('dt') == 3 and np.asarray(_v9).dtype.kind != 'M':
                    _v9 = np.asarray(_v9).astype(np.int64).view('datetime64[%s]' % seg.unit(pc))
                arrs.append(_v9)
            elif seg.cols.get(pc, {}).get('mode') != 5:
                arrs.append(np.asarray(_bulk_keyvals(seg, pc, np.asarray(seg.codes_at(pc, rows))), dtype=object))
            else:
                arrs.append(np.asarray(list(seg.values_at_rows(pc, rows)), dtype=object))
    q9 = tree.args.get('qualify')
    if q9 is not None and R:
        env9 = {nm: a for nm, a in zip(names9, arrs)}
        m9 = np.asarray(wdb_sql._eval_rows(seg, q9.this, None, None, env=env9))
        if m9.dtype != bool:
            m9 = np.array([bool(v) if v is not None else False for v in m9], dtype=bool)
        arrs = [a[m9] for a in arrs]
    cols = [([wdb_sql._pyval(x) for x in a] if a.dtype.kind == 'M' else (a.tolist() if a.dtype != object else list(a))) for a in arrs]
    out = list(zip(*cols)) if cols else []
    out = [tuple(int(x) if isinstance(x, (np.integer,)) else (float(x) if isinstance(x, np.floating) else (wdb_sql._pyval(x) if isinstance(x, np.datetime64) else x)) for x in r) for r in out]
    out = wdb_sql._apply_order(out, proj, tree.args.get('order'))
    lim = tree.args.get('limit')
    if lim is not None: out = out[:int(lim.expression.this)]
    return out, names9



def _topk_rows_door(db, tree):
    """THE TOP-K ROWS DOOR: SELECT cols FROM t [WHERE] ORDER BY keys LIMIT k on a
    single table with no group/aggregate/window -- argpartition on the first
    order key over survivors (ties at the k-th value kept), a full sort only
    over the candidates, then gather the projections at the winners. The row
    select used to materialise 10M Python tuples and sort them (26.9s)."""
    if not isinstance(tree, E.Select): return None
    if tree.args.get('group') is not None or tree.args.get('joins') or tree.args.get('distinct') is not None: return None
    order = tree.args.get('order'); lim = tree.args.get('limit')
    if order is None or lim is None: return None
    try:
        k = int(lim.expression.this)
    except Exception:
        return None
    off = tree.args.get('offset')
    off_n = int(off.expression.this) if off is not None else 0
    if k <= 0 or k + off_n > 1_000_000: return None
    frm = tree.args.get('from') or tree.args.get('from_')
    if frm is None or not isinstance(frm.this, E.Table): return None
    if tree.find(E.AggFunc) is not None or tree.find(E.Window) is not None or tree.find(E.Select) is not tree: return None
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        if not isinstance(nd, E.Column): return None
    okeys = list(order.expressions)
    if not all(isinstance(o.this, E.Column) for o in okeys): return None
    tname = frm.this.name
    try:
        seg, _sp = _solo_segment(db, tname)
    except _FastUnsupported:
        return None
    pm = db.cat.phys_map(tname)
    try:
        cm9 = seg.cluster_meta()
    except Exception:
        cm9 = None
    if cm9 is not None and pm.get(okeys[0].this.name, okeys[0].this.name) == cm9.get('key'):
        return None                       # ordered by the CLUSTER KEY: the clustertopk door is faster
    w = tree.args.get('where')
    if w is not None:
        m = wdb_sql._eval_pred(seg, w.this, lambda nm: pm.get(nm, nm))
        rows = np.flatnonzero(np.asarray(m, dtype=bool))
    else:
        rows = np.arange(int(seg.N), dtype=np.int64)
    if rows.size == 0: return [], [wdb_sql._alias(p) for p in tree.expressions]
    def key_of(col, rr):
        pc = pm.get(col.name, col.name); cd = seg.cols.get(pc, {})
        if cd.get('has_null') or cd.get('mode') not in (0, 1, 2, 4): raise _FastUnsupported
        if cd.get('mode') in (0, 1, 2):
            # SORTED DICTIONARIES (np.unique at encode; front-coding requires order):
            # code order IS value order for strings and numbers alike -- the key is
            # the codes, no base gather
            return np.asarray(seg.codes_at(pc, rr)).astype(np.int64)
        raw = wdb_sql.raw_dict_col(seg, pc, want_codes=False)
        if raw is None: raise _FastUnsupported
        return raw[0][np.asarray(seg.codes_at(pc, rr))]
    need = k + off_n
    k0 = key_of(okeys[0].this, rows)
    desc0 = bool(okeys[0].args.get('desc'))
    kk = -k0 if desc0 else k0
    if need < rows.size:
        part = np.argpartition(kk, need - 1)[:need]
        thresh = kk[part].max()
        cand = np.flatnonzero(kk <= thresh)                           # every row tied at the k-th value
    else:
        cand = np.arange(rows.size)
    # full ORDER BY over the candidates only
    keys = []
    for o in reversed(okeys):
        kv = key_of(o.this, rows[cand])
        keys.append(-kv if o.args.get('desc') else kv)
    order_idx = np.lexsort(tuple(keys)) if len(keys) > 1 else np.argsort(keys[0], kind='stable')
    win = rows[cand][order_idx][off_n:off_n + k]
    cols = []
    for p in tree.expressions:
        nd = p.this if isinstance(p, E.Alias) else p
        pc = pm.get(nd.name, nd.name)
        if seg.cols.get(pc, {}).get('mode') == 5:
            cols.append(list(seg.values_at_rows(pc, win)))
        else:
            cols.append(_bulk_keyvals(seg, pc, np.asarray(seg.codes_at(pc, win))))
    return [tuple(c[i] for c in cols) for i in range(win.size)], [wdb_sql._alias(p) for p in tree.expressions]



def _from_door(db, tree):
    """THE FROM DOOR (Q7/Q13/Q22): FROM (SELECT ...) alias -- run the INNER
    through the engine (it rides every fast path: trees, cascades, courts),
    then evaluate the outer over the inner's small output frame. Returns rows
    or None (not this shape)."""
    frm = tree.args.get('from') or tree.args.get('from_')   # sqlglot renamed the key
    if frm is None or not isinstance(frm.this, E.Subquery): return None
    if tree.args.get('joins'): return None
    if tree.args.get('having') is not None: return None
    inner = frm.this.this
    if not isinstance(inner, E.Select): return None
    if inner.find(E.Window) is not None:
        r9 = _window_topk_door(db, tree)
        if r9 is not None:
            return r9
        raise NotImplementedError('window functions beyond ROW_NUMBER top-k-per-group are not supported')
    import pandas as pd
    cols = [e.alias_or_name for e in inner.expressions]
    if any(not c for c in cols): return None
    _AGG9 = (E.Sum, E.Count, E.Avg, E.Min, E.Max)
    _nested9 = next((x for x in inner.find_all(E.Select) if x is not inner), None)
    if inner.args.get('group') is None and _nested9 is None and not any(
            e.find(*_AGG9) is not None for e in inner.expressions):
        # FLATTEN: an aggregate-less inner folds into the outer -- substitute
        # the inner expressions for their aliases and run ONE tree query.
        sub_map = {e.alias_or_name: (e.this if isinstance(e, E.Alias) else e)
                   for e in inner.expressions}
        flat = tree.copy()
        for key9 in ('from', 'from_'):
            if inner.args.get(key9) is not None:
                flat.set(key9, inner.args[key9].copy())
        flat.set('joins', [j.copy() for j in (inner.args.get('joins') or [])] or None)
        names9 = [p.alias_or_name for p in flat.expressions]
        order9 = flat.args.get('order')
        if order9 is not None:
            for o9 in order9.expressions:
                if isinstance(o9.this, E.Column) and o9.this.name in names9:
                    o9.set('this', E.Literal(this=str(names9.index(o9.this.name) + 1), is_string=False))
        for c9 in list(flat.find_all(E.Column)):
            if not c9.table and c9.name in sub_map:
                c9.replace(sub_map[c9.name].copy())
        iw9 = inner.args.get('where')
        if iw9 is not None:
            fw9 = flat.args.get('where')
            merged9 = iw9.this.copy() if fw9 is None else E.And(
                this=E.Paren(this=iw9.this.copy()), expression=E.Paren(this=fw9.this.copy()))
            flat.set('where', E.Where(this=merged9))
        return join_query(db, flat.sql(dialect='duckdb'))
    df = _left_count_frame(db, inner)
    if df is None:
        df = _rich_lonely_frame(db, inner)
    if df is None:
        rows = db.run(inner.sql(dialect='duckdb'))
        rows = rows[0] if isinstance(rows, tuple) else rows
        df = pd.DataFrame(list(rows), columns=cols)
    def ev(nd):
        if isinstance(nd, E.Paren): return ev(nd.this)
        if isinstance(nd, E.Column): return df[nd.name]
        if isinstance(nd, E.Literal):
            return (nd.this if nd.is_string else (float(nd.this) if '.' in str(nd.this) else int(nd.this)))
        if isinstance(nd, E.Neg): return -ev(nd.this)
        if isinstance(nd, E.Mul): return ev(nd.this) * ev(nd.expression)
        if isinstance(nd, E.Div): return ev(nd.this) / ev(nd.expression)
        if isinstance(nd, E.Add): return ev(nd.this) + ev(nd.expression)
        if isinstance(nd, E.Sub): return ev(nd.this) - ev(nd.expression)
        raise _FastUnsupported
    def mask(nd):
        if isinstance(nd, E.Paren): return mask(nd.this)
        if isinstance(nd, E.And): return mask(nd.this) & mask(nd.expression)
        if isinstance(nd, E.Or): return mask(nd.this) | mask(nd.expression)
        if isinstance(nd, E.Not): return ~mask(nd.this)
        if isinstance(nd, E.In):
            lits = [ev(x) for x in (nd.args.get('expressions') or [])]
            if nd.args.get('query') is not None: raise _FastUnsupported
            return ev(nd.this).isin(lits)
        ops = {E.GT: '__gt__', E.GTE: '__ge__', E.LT: '__lt__', E.LTE: '__le__',
               E.EQ: '__eq__', E.NEQ: '__ne__'}
        if type(nd) in ops:
            return getattr(ev(nd.this), ops[type(nd)])(ev(nd.expression))
        raise _FastUnsupported
    try:
        w = tree.args.get('where')
        if w is not None:
            df = df[mask(w.this)]
        grp = tree.args.get('group')
        proj = list(tree.expressions)
        _AGG = (E.Sum, E.Count, E.Avg, E.Min, E.Max)
        def eval_on(nd, sub):
            nonlocal df
            keep = df; df = sub
            try: return ev(nd)
            finally: df = keep
        def agg_val(node, sub):
            if isinstance(node, E.Count) and (node.this is None or isinstance(node.this, E.Star)):
                return int(len(sub))
            arg = eval_on(node.this, sub)
            if isinstance(node, E.Sum): return float(arg.sum())
            if isinstance(node, E.Avg): return float(arg.mean())
            if isinstance(node, E.Min): return arg.min()
            if isinstance(node, E.Max): return arg.max()
            if isinstance(node, E.Count): return int(arg.notna().sum())
            raise _FastUnsupported
        def proj_val(p, sub, gvals):
            nd = p.this if isinstance(p, E.Alias) else p
            if isinstance(nd, E.Column) and nd.name in gvals:
                return gvals[nd.name]
            if isinstance(nd, _AGG):
                return agg_val(nd, sub)
            if isinstance(nd, (E.Mul, E.Div, E.Add, E.Sub, E.Paren)) and nd.find(*_AGG) is not None:
                def num(x):
                    if isinstance(x, E.Paren): return num(x.this)
                    if isinstance(x, _AGG): return agg_val(x, sub)
                    if isinstance(x, E.Literal): return float(x.this) if not x.is_string else x.this
                    if isinstance(x, E.Mul): return num(x.this) * num(x.expression)
                    if isinstance(x, E.Div): return num(x.this) / num(x.expression)
                    if isinstance(x, E.Add): return num(x.this) + num(x.expression)
                    if isinstance(x, E.Sub): return num(x.this) - num(x.expression)
                    raise _FastUnsupported
                return num(nd)
            raise _FastUnsupported
        out = []
        if grp is not None:
            gnames = [g.name for g in grp.expressions if isinstance(g, E.Column)]
            if len(gnames) != len(grp.expressions): raise _FastUnsupported
            for kv, sub in df.groupby(gnames, sort=False, dropna=False):
                kv = kv if isinstance(kv, tuple) else (kv,)
                gvals = dict(zip(gnames, kv))
                out.append(tuple(proj_val(p, sub, gvals) for p in proj))
        else:
            out.append(tuple(proj_val(p, df, {}) for p in proj))
        order = tree.args.get('order')
        if order is not None:
            names = [(p.alias if isinstance(p, E.Alias) else (p.name if isinstance(p, E.Column) else None)) for p in proj]
            keys = []
            for o in order.expressions:
                cn = o.this
                if isinstance(cn, E.Column) and cn.name in names:
                    keys.append((names.index(cn.name), bool(o.args.get('desc'))))
                elif isinstance(cn, E.Literal):
                    keys.append((int(cn.this) - 1, bool(o.args.get('desc'))))
                else:
                    raise _FastUnsupported
            import functools
            def cmp(a, b):
                for idx, desc in keys:
                    x, y = a[idx], b[idx]
                    if x == y: continue
                    lt = (x is None) or (y is not None and x < y)
                    return (1 if lt else -1) if desc else (-1 if lt else 1)
                return 0
            out.sort(key=functools.cmp_to_key(cmp))
        lim = tree.args.get('limit')
        if lim is not None:
            out = out[:int(lim.expression.this)]
        return out
    except _FastUnsupported:
        raise
    except Exception:
        raise _FastUnsupported



def _route9(name):
    if os.environ.get('WDB_ROUTE_DEBUG'):
        print('ROUTE: %s' % name, flush=True)


def join_query(db, sql, columnar=False):
    import time as _t8
    _jq_t0 = _t8.perf_counter()
    tree = sqlglot.parse_one(sql, read='duckdb')
    for _jn9 in (tree.args.get('joins') or []):
        # USING (c) -> ON l.c = r.c: the chain builder and the road organ read ON
        if _jn9.args.get('using') and _jn9.args.get('on') is None:
            _frm9 = tree.args.get('from') or tree.args.get('from_')
            if _frm9 is not None and isinstance(_frm9.this, E.Table) and isinstance(_jn9.this, E.Table):
                _la9 = _frm9.this.alias or _frm9.this.name; _ra9 = _jn9.this.alias or _jn9.this.name
                _cond9 = None
                for _u9 in _jn9.args['using']:
                    _nm9 = _u9.name if hasattr(_u9, 'name') else str(_u9)
                    _eq9 = E.EQ(this=E.Column(this=E.Identifier(this=_nm9, quoted=False), table=E.Identifier(this=_la9, quoted=False)),
                                expression=E.Column(this=E.Identifier(this=_nm9, quoted=False), table=E.Identifier(this=_ra9, quoted=False)))
                    _cond9 = _eq9 if _cond9 is None else E.And(this=_cond9, expression=_eq9)
                _jn9.set('on', _cond9); _jn9.set('using', None)

    if tree.find(E.Window) is not None and not (tree.args.get('from') or tree.args.get('from_')).this.__class__.__name__ == 'Subquery':
        try:
            _wd9 = _window_door(db, tree)
        except _FastUnsupported:
            _wd9 = None
        if _wd9 is not None:
            _route9('window-door')
            return _wd9
        raise NotImplementedError('window shape beyond the window door')
    try:
        _ae9 = _agg_expr_rewrite(db, tree)
    except _FastUnsupported:
        _ae9 = None
    if _ae9 is not None:
        _route9('agg-arith')
        return _ae9
    _factor_or_rewrite(tree)
    _lonely_rewrite(db, tree)
    _grouped_in_rewrite(db, tree)
    try:
        door9 = _from_door(db, tree)
    except _FastUnsupported:
        door9 = None
    if door9 is not None:
        _route9('from-door')
        return door9
    joins = tree.args.get('joins')
    import wdb_sql as _ws
    has_aggs = any(_ws._agg_kind(p) is not None for p in tree.expressions)
    if has_aggs and not columnar:
        _dc9j = _descent_court(db, tree.copy())   # pristine: doors below MUTATE the
        if _dc9j is not None:                     # tree (chain build pops join eqs)
            return _dc9j
    chain = None
    # FK-pointer fast path: handles 1..N joins as a chain/star of pre-resolved pointers.
    # Building a chain can RESOLVE POINTERS ON THE FLY (an O(N) hash over the fact) -- so
    # it only runs eagerly for aggregates, where the fused pointer path owns precedence.
    # Plain dumps try the dict-space route first: a one-block walk must not pay a
    # full-table probe as routing overhead.
    if has_aggs:
        tree1 = tree.copy()          # each attempt gets its OWN tree: the chain
        try:                         # builder strips consumed equalities, and a
                                     # discarded attempt must not poison the retry
            chain = _build_chain(db, tree1, allow_hash=False)    # STORED pointers only:
        except _FastUnsupported:                                 # pre-resolved and free
            chain = None
        if chain is not None:
            try:
                _route9('fpa-stored')
                return _fast_pointer_agg(db, tree1, chain, columnar)  # fully fused
            except _FastUnsupported:
                chain = None   # stored-only chain's plan declined (e.g. Q5's tree
                               # needs hash edges): release it so the retry rebuilds
    if joins and len(joins) == 1 and not has_aggs:
        # THE ROAD JOIN goes BEFORE the lazy chain: a row-emitting equi-join
        # the chain accepts (INNER ... ON) would otherwise land in the pandas
        # tail (j1: 20s -> 63s the day INNER was accepted).
        try:
            _rj9 = _road_join_emit(db, tree)
        except _FastUnsupported:
            _rj9 = None
        if _rj9 is not None:
            _route9('road-join')
            return _rj9
    import wdb_fastjoin
    fj = wdb_fastjoin.try_execute(db, tree)      # dict-space: semi-joins, cell post-maps,
    if fj is not None:                           # streaming dumps
        _route9('fastjoin')
        return fj
    if chain is None:
        try:
            tree2 = tree.copy()
            chain = _build_chain(db, tree2)      # lazy: only when the tail will use it
            if chain is not None:
                tree = tree2                     # the stripped copy is the one to execute
        except _FastUnsupported:
            chain = None
    if chain is not None:
        try:
            if __import__('os').environ.get('WDB_JOIN_BILL'):
                print('JOIN BILL: pre-work(parse+chain)=%.0fms'
                      % ((_t8.perf_counter() - _jq_t0) * 1000), flush=True)
            _route9('fpa-tree')
            return _fast_pointer_agg(db, tree, chain, columnar)  # hashed chain, fused agg
        except _FastUnsupported:
            _route9('chain-pandas')
            return _chain_pandas(db, tree, chain)    # same chain, pandas agg/predicate tail
    if not joins or len(joins) != 1:
        raise NotImplementedError("join: non-FK multi-join needs a hash join (not yet supported)")
    jn = joins[0]
    _side9 = (jn.args.get('side') or '').upper(); _kind9 = (jn.args.get('kind') or '').upper()
    if _kind9 not in ('', 'INNER', 'CROSS', 'OUTER') or _side9 not in ('', 'LEFT', 'RIGHT', 'FULL'):
        raise NotImplementedError("join: only INNER/LEFT/RIGHT/FULL/CROSS (step 1)")
    _how9 = {'': 'inner', 'LEFT': 'left', 'RIGHT': 'right', 'FULL': 'outer'}[_side9]
    frm = tree.find(E.From).this
    lt, la = frm.name, (frm.alias or frm.name)
    rt, ra = jn.this.name, (jn.this.alias or jn.this.name)
    lcols = set(db.cat.column_names(lt)); rcols = set(db.cat.column_names(rt))
    a2t = {la: lt, ra: rt}
    def side_of(col):
        if col.table: return 'L' if col.table == la else ('R' if col.table == ra else None)
        if col.name in lcols and col.name not in rcols: return 'L'
        if col.name in rcols and col.name not in lcols: return 'R'
        return None
    def sides_in(node):
        return {side_of(c) for c in node.find_all(E.Column)}
    # ---- the ON: equalities (keys), extras (filters), USING, CROSS
    eqs, extras = [], []
    on = jn.args.get('on')
    if jn.args.get('using'):
        for u in jn.args['using']:
            nm = u.name if hasattr(u, 'name') else str(u)
            eqs.append((E.Column(this=E.Identifier(this=nm, quoted=False), table=E.Identifier(this=la, quoted=False)),
                        E.Column(this=E.Identifier(this=nm, quoted=False), table=E.Identifier(this=ra, quoted=False))))
    elif on is not None:
        def _fl(x):
            if isinstance(x, E.Paren): return _fl(x.this)
            if isinstance(x, E.And): return _fl(x.this) + _fl(x.expression)
            return [x]
        for cj in _fl(on):
            if isinstance(cj, E.EQ) and sides_in(cj.this) and sides_in(cj.expression) \
                    and sides_in(cj.this) != sides_in(cj.expression) and None not in sides_in(cj.this) | sides_in(cj.expression) \
                    and len(sides_in(cj.this)) == 1 and len(sides_in(cj.expression)) == 1:
                a, b = cj.this, cj.expression
                if sides_in(a) == {'R'}: a, b = b, a
                eqs.append((a, b))
            else:
                extras.append(cj)
    elif _kind9 != 'CROSS':
        raise NotImplementedError("join: no ON/USING")
    if any(None in sides_in(x) for x in extras): raise NotImplementedError("join: ambiguous column in ON")
    # ---- needed columns
    need_l, need_r = set(), set()
    def _need(node):
        for talias, cname in _all_columns(node):
            if talias == la or (not talias and cname in lcols and cname not in rcols): need_l.add(cname)
            elif talias == ra or (not talias and cname in rcols and cname not in lcols): need_r.add(cname)
            elif not talias and cname in lcols and cname in rcols:
                raise NotImplementedError(f"ambiguous column {cname!r}")
    for part in (tree.expressions, [tree.args.get('where')], (tree.args.get('group').expressions if tree.args.get('group') else []),
                 (tree.args.get('order').expressions if tree.args.get('order') else []), [x for pr in eqs for x in pr], extras):
        for node in part:
            if node is not None: _need(node)
    if not need_l: need_l.add(next(iter(lcols)))
    if not need_r: need_r.add(next(iter(rcols)))
    def resolve(tbl_alias, name):
        if tbl_alias: return f"{tbl_alias}.{name}"
        if name in lcols and name in rcols: raise NotImplementedError(f"ambiguous column {name!r}")
        return f"{la}.{name}" if name in lcols else f"{ra}.{name}"
    lc = _materialize(db, lt, need_l); rc = _materialize(db, rt, need_r)
    ldf = pd.DataFrame({f"{la}.{c}": lc[c] for c in lc})
    rdf = pd.DataFrame({f"{ra}.{c}": rc[c] for c in rc})
    R = lambda colnode: resolve(colnode.table, colnode.name)
    def _series(df, node):
        """a join-key side as a Series: a column, or arithmetic over ONE column (a.id4 + 1)."""
        if isinstance(node, E.Column): return df[R(node)]
        if isinstance(node, E.Paren): return _series(df, node.this)
        if isinstance(node, E.Literal): return float(node.this) if '.' in str(node.this) else int(node.this)
        if isinstance(node, E.Neg): return -_series(df, node.this)
        ops = {E.Add: lambda a, b: a + b, E.Sub: lambda a, b: a - b, E.Mul: lambda a, b: a * b, E.Div: lambda a, b: a / b, E.Mod: lambda a, b: a % b}
        if type(node) in ops: return ops[type(node)](_series(df, node.this), _series(df, node.expression))
        raise NotImplementedError("join: ON key expression %s" % node.sql()[:40])
    # extras that belong to one side filter that side BEFORE the merge (outer-join semantics)
    def _mask_df(df, node):
        return _mask(df, node, R)
    pre_r = [x for x in extras if sides_in(x) == {'R'}]
    pre_l = [x for x in extras if sides_in(x) == {'L'}]
    mixed = [x for x in extras if len(sides_in(x)) == 2]
    # PREDICATE PUSHDOWN: single-sided WHERE conjuncts filter their side BEFORE the
    # merge -- exact for INNER; for an outer join only the preserved side (the
    # other side's WHERE stays post-merge, where NULLs from unmatched rows count).
    # Without it a 10M x 10M self-join on a 1000-rows-per-key pair built 10 BILLION
    # pairs before the WHERE (scope speed board, multi_col_on).
    _wh9 = tree.args.get('where')
    _post_where9 = []
    if _wh9 is not None:
        def _fl2(x):
            if isinstance(x, E.Paren): return _fl2(x.this)
            if isinstance(x, E.And): return _fl2(x.this) + _fl2(x.expression)
            return [x]
        for cj in _fl2(_wh9.this):
            sd = sides_in(cj)
            if cj.find(E.Subquery) is not None or None in sd:
                _post_where9.append(cj); continue
            if sd == {'L'} and _how9 in ('inner', 'left'):
                ldf = ldf[_mask_df(ldf, cj)]
            elif sd == {'R'} and _how9 in ('inner', 'right'):
                rdf = rdf[_mask_df(rdf, cj)]
            else:
                _post_where9.append(cj)
    if _how9 in ('inner', 'left') and pre_r:
        for x in pre_r: rdf = rdf[_mask_df(rdf, x)]
    if _how9 in ('inner', 'right') and pre_l:
        for x in pre_l: ldf = ldf[_mask_df(ldf, x)]
    if _how9 == 'inner':
        pass                                            # inner: pre-filters already applied on both sides
    elif _how9 == 'left' and pre_l:
        mixed = mixed + pre_l          # a left-side ON filter on a LEFT join only unmatches (post, as NULL-safe filter below)
    elif _how9 == 'right' and pre_r:
        mixed = mixed + pre_r
    _full_extras = []
    if _how9 == 'outer' and (pre_l or pre_r or mixed):
        # FULL OUTER with extra ON conjuncts: the ON decides MATCHING only -- match
        # as INNER under every conjunct, then append the unmatched rows of BOTH sides
        _full_extras = pre_l + pre_r + mixed; pre_l = []; pre_r = []; mixed = []
    if (_how9 in ('left', 'right')) and mixed:
        raise NotImplementedError("join: OUTER join with a two-sided ON conjunct")
    _route9('step1-merge')
    if eqs:
        lkeys, rkeys = [], []
        for j9, (a, b) in enumerate(eqs):
            ka, kb = f"__lk{j9}", f"__rk{j9}"
            ldf[ka] = _series(ldf, a).values; rdf[kb] = _series(rdf, b).values
            lkeys.append(ka); rkeys.append(kb)
        if _full_extras:
            ldf = ldf.copy(); rdf = rdf.copy()
            ldf['__lid'] = np.arange(len(ldf)); rdf['__rid'] = np.arange(len(rdf))
            inner9 = ldf.merge(rdf, left_on=lkeys, right_on=rkeys, how='inner')
            for x in _full_extras:
                inner9 = inner9[_mask_df(inner9, x)]
            ul = ldf[~ldf['__lid'].isin(inner9['__lid'])]
            ur = rdf[~rdf['__rid'].isin(inner9['__rid'])]
            merged = pd.concat([inner9, ul, ur], ignore_index=True, sort=False)
            merged = merged.drop(columns=['__lid', '__rid'])
        else:
            merged = ldf.merge(rdf, left_on=lkeys, right_on=rkeys, how=_how9)
        merged = merged.drop(columns=lkeys + rkeys)
    else:                                                   # CROSS / non-equi: bounded cross product
        if len(ldf) * len(rdf) > 20_000_000:
            raise NotImplementedError("join: cross product too large (%d x %d)" % (len(ldf), len(rdf)))
        merged = ldf.merge(rdf, how='cross')
    for x in mixed:
        merged = merged[_mask_df(merged, x)]

    R = lambda colnode: resolve(colnode.table, colnode.name)
    where = tree.args.get('where')
    if where is not None:
        _rest9 = None
        for cj in _post_where9:
            _rest9 = cj if _rest9 is None else E.And(this=_rest9, expression=cj)
        if _rest9 is not None:
            merged = merged[_mask(merged, _rest9, R)]

    proj = tree.expressions
    group = tree.args.get('group')
    has_agg = any(wdb_sql._agg_kind(p) for p in proj)
    if group is not None or has_agg:
        rows = _aggregate(merged, proj, group, R)
    else:
        keys = [R(p.this if isinstance(p, E.Alias) else p) for p in proj]
        rows = [tuple(_render(v) for v in t) for t in merged[keys].itertuples(index=False, name=None)]

    having = tree.args.get('having')
    if having is not None:
        rows = wdb_sql._apply_having(rows, proj, having.this, None)   # fused path must filter too
    rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))
    lim = wdb_sql._limit(tree)
    if lim is not None: rows = rows[:lim]
    return rows, [wdb_sql._alias(p) for p in proj]


def _render(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    if isinstance(v, pd.Timestamp): v = v.to_numpy()
    return wdb_sql._pyval(v)


def _eval_compose(node, amap, r):
    """Evaluate an arithmetic tree over already-aggregated values: agg nodes
    resolve via amap into the result row; literals and arithmetic recurse."""
    if id(node) in amap:
        return r[amap[id(node)]]
    if isinstance(node, E.Paren):
        return _eval_compose(node.this, amap, r)
    if isinstance(node, E.Literal):
        v = node.this
        return float(v) if '.' in str(v) else int(v)
    if isinstance(node, E.Neg):
        return -_eval_compose(node.this, amap, r)
    import operator as _o9
    for tp9, op9 in ((E.Mul, _o9.mul), (E.Add, _o9.add),
                     (E.Sub, _o9.sub), (E.Div, _o9.truediv)):
        if isinstance(node, tp9):
            return op9(_eval_compose(node.this, amap, r),
                       _eval_compose(node.expression, amap, r))
    raise NotImplementedError('compose: %s' % type(node).__name__)


def _eval_expr(df, node, R):
    """Pandas-path expression evaluator for aggregate ARGUMENTS: Columns,
    literals, arithmetic, and CASE WHEN (via _mask + np.select)."""
    if isinstance(node, E.Paren):
        return _eval_expr(df, node.this, R)
    if isinstance(node, E.Column):
        return df[R(node)]
    if isinstance(node, E.Literal):
        v = node.this
        if node.is_string:
            return v.encode() if isinstance(v, str) else v
        return float(v) if '.' in str(v) else int(v)
    if isinstance(node, E.Neg):
        return -_eval_expr(df, node.this, R)
    if isinstance(node, E.Case):
        conds, vals = [], []
        for br in node.args.get('ifs', []):
            c9 = _mask(df, br.this, R)
            conds.append(c9.to_numpy() if hasattr(c9, 'to_numpy') else np.asarray(c9))
            v9 = _eval_expr(df, br.args['true'], R)
            vals.append(v9.to_numpy() if hasattr(v9, 'to_numpy') else v9)
        d9 = node.args.get('default')
        dv = _eval_expr(df, d9, R) if d9 is not None else 0
        if hasattr(dv, 'to_numpy'):
            dv = dv.to_numpy()
        return pd.Series(np.select(conds, vals, default=dv), index=df.index)
    import operator as _op9
    for tp9, op9 in ((E.Mul, _op9.mul), (E.Add, _op9.add),
                     (E.Sub, _op9.sub), (E.Div, _op9.truediv)):
        if isinstance(node, tp9):
            return op9(_eval_expr(df, node.this, R), _eval_expr(df, node.expression, R))
    raise NotImplementedError('pandas expr: %s' % type(node).__name__)


def _coerce_lit(series, lit):
    if isinstance(lit, E.Neg):
        return -_coerce_lit(series, lit.this)
    k = series.dtype.kind
    if k == 'M':  # datetime
        return pd.Timestamp(str(lit.this))
    if k in 'iuf':
        return float(lit.this) if (k == 'f' or '.' in str(lit.this)) else int(lit.this)
    if k == 'O':
        # an OBJECT column carries whatever the row path put there: ints, floats, bytes or str --
        # coerce the literal to the ELEMENT type, not to bytes by default (a column of Python ints
        # compared to b'7' answered 0 for 25,669 on the multi-segment fallback, 2026-09-13)
        first = next((v for v in series.values[:1000] if v is not None), None)
        if isinstance(first, bool): return bool(lit.this)
        if isinstance(first, (int, np.integer)) and not lit.is_string: return int(float(lit.this))
        if isinstance(first, (float, np.floating)) and not lit.is_string: return float(lit.this)
        if isinstance(first, str): return str(lit.this)
    s = lit.this
    return s.encode() if isinstance(s, str) else s


def _mask(df, node, R):
    import operator
    if isinstance(node, E.Paren): return _mask(df, node.this, R)
    if isinstance(node, E.And): return _mask(df, node.this, R) & _mask(df, node.expression, R)
    if isinstance(node, E.Or): return _mask(df, node.this, R) | _mask(df, node.expression, R)
    if isinstance(node, E.Not): return ~_mask(df, node.this, R)
    if type(node) in _CMP:
        s = df[R(node.this)]
        v = df[R(node.expression)] if isinstance(node.expression, E.Column) else _coerce_lit(s, node.expression)
        op = {E.EQ: operator.eq, E.NEQ: operator.ne, E.GT: operator.gt,
              E.LT: operator.lt, E.GTE: operator.ge, E.LTE: operator.le}[type(node)]
        return op(s, v)
    if isinstance(node, E.Between):
        s = df[R(node.this)]; lo = _coerce_lit(s, node.args['low']); hi = _coerce_lit(s, node.args['high'])
        return (s >= lo) & (s <= hi)
    if isinstance(node, E.In):
        if node.args.get('query') is not None or not (node.args.get('expressions') or []):
            raise NotImplementedError('join WHERE: IN (subquery) -- fail loud, never empty')
        s = df[R(node.this)]; vals = [_coerce_lit(s, L) for L in (node.args.get('expressions') or [])]
        return s.isin(vals)
    if isinstance(node, E.Is):                         # IS NULL (IS NOT NULL arrives as Not(Is))
        if isinstance(node.expression, E.Null): return df[R(node.this)].isna()
        raise NotImplementedError(f"join WHERE: Is {type(node.expression).__name__}")
    if isinstance(node, (E.Like, E.ILike)):            # LIKE that didn't fuse (e.g. high-card column)
        s = df[R(node.this)]
        rx = '^' + re.escape(str(node.expression.this)).replace('%', '.*').replace('_', '.') + '$'
        fl9 = re.DOTALL | (re.IGNORECASE if isinstance(node, E.ILike) else 0)
        rxc9 = re.compile(rx.encode() if s.dtype == object else rx, fl9)
        arr9 = s.to_numpy()
        m = pd.Series([v is not None and bool(rxc9.match(v)) for v in arr9], index=s.index)
        return (~m) if node.args.get('negate') else m
    raise NotImplementedError(f"join WHERE: {type(node).__name__}")


def _aggregate(merged, proj, group, R):
    _PF = {'SUM': 'sum', 'AVG': 'mean', 'MIN': 'min', 'MAX': 'max', 'COUNT': 'count'}
    specs = []  # per projection: ('key', col) | ('size',) | ('agg', fn, col)
    for i, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        kind = wdb_sql._agg_kind(p)
        if kind is None and not isinstance(inner, E.Column) \
                and any(True for _ in inner.find_all(E.Sum, E.Avg, E.Min, E.Max, E.Count)):
            # arithmetic OVER aggregates (Q14's 100*SUM/SUM): each inner agg
            # becomes its own synth spec; the tree composes on the RESULT row.
            amap9 = {}
            for an9 in inner.find_all(E.Sum, E.Avg, E.Min, E.Max, E.Count):
                k9 = wdb_sql._agg_kind(an9)
                if k9 is None:
                    raise NotImplementedError('compose: agg kind')
                syn9 = '_pc%d_%d' % (i, len(amap9))
                arg9 = an9.this
                if isinstance(arg9, E.Column):
                    col9 = R(arg9)
                else:
                    col9 = syn9 + '_x'
                    merged[col9] = _eval_expr(merged, arg9, R)
                specs.append(('agg', k9[0], col9, syn9))
                amap9[id(an9)] = syn9
            specs.append(('compose', inner, amap9))
        elif kind is None:
            specs.append(('key', R(inner)))
        elif kind[0] == 'COUNT_STAR':
            specs.append(('size',))
        else:
            arg9 = inner.this
            if isinstance(arg9, E.Column):
                specs.append(('agg', kind[0], R(arg9)))
            else:                                  # SUM(CASE...), SUM(a*b), ...
                syn9 = '_xpr%d' % i
                merged[syn9] = _eval_expr(merged, arg9, R)
                specs.append(('agg', kind[0], syn9))
    if group is not None:
        key_cols = [R(g) for g in group.expressions]
        g = merged.groupby(key_cols, sort=False, dropna=False)
        named = {(s[3] if len(s) > 3 else f"_a{i}"): pd.NamedAgg(column=s[2], aggfunc=_PF[s[1]])
                 for i, s in enumerate(specs) if s[0] == 'agg'}
        agg = g.agg(**named) if named else g.size().to_frame('_dummy')
        if any(s[0] == 'size' for s in specs):
            agg['_size'] = g.size()
        agg = agg.reset_index()
        out = []
        for _, r in agg.iterrows():
            row = []
            for i, s in enumerate(specs):
                if s[0] == 'key': row.append(_render(r[s[1]]))
                elif s[0] == 'size': row.append(int(r['_size']))
                elif s[0] == 'compose':
                    row.append(_render(_eval_compose(s[1], s[2], r)))
                elif len(s) > 3:
                    continue                       # consumed by a compose
                else: row.append(_render(r[f"_a{i}"]))
            out.append(tuple(row))
        return out
    # whole-table aggregate -> single row
    row = []
    sc9 = {}
    for i, s in enumerate(specs):
        if s[0] == 'agg':
            col = merged[s[2]]
            sc9[s[3] if len(s) > 3 else f"_a{i}"] = {'SUM': col.sum(), 'AVG': col.mean(),
                'MIN': col.min(), 'MAX': col.max(), 'COUNT': col.count()}[s[1]]
    for i, s in enumerate(specs):
        if s[0] == 'size': row.append(int(len(merged)))
        elif s[0] == 'compose':
            row.append(_render(_eval_compose(s[1], s[2], sc9)))
        elif s[0] == 'agg':
            if len(s) > 3:
                continue                           # consumed by a compose
            row.append(_render(sc9[f"_a{i}"]))
        else:
            raise NotImplementedError("bare column with aggregates but no GROUP BY")
    return [tuple(row)]


# ── FK-pointer gather fast path ──────────────────────────────────────────────
# When a join's ON matches a stored foreign-key pointer (child.fk = parent.key), the join is already
# resolved: we gather parent columns by the pointer instead of hash-merging, and aggregate on WaveDB's
# integer codes with the bincount kernel. Narrow by design -- single join, one group key, GROUP BY +
# aggregates -- and raises _FastUnsupported for anything else so join_query falls back to pandas.

# Dense/tally ceilings (multi-group, grouped COUNT(DISTINCT), value-frequency tally) live in
# wdb_measure_runtime: RT.dense_multigroup_fits / RT.grouped_cdist_fits / RT.tally_worth_it.

class _FastUnsupported(Exception):
    pass

_FAST_HITS = 0   # diagnostic: how many queries took the gather fast path
_TOPK_HITS = 0   # diagnostic: how many queries had `present` pruned by the bounded top-K prefilter
_SLICE_SCALAR_HITS = 0   # diagnostic: how many queries took the per-slice scalar agg (cluster-key GROUP BY + predicate)
def _bump_fast():
    global _FAST_HITS
    _FAST_HITS += 1
# code-LUT cardinality cap lives in wdb_measure_runtime: RT.code_lut_fits(ncodes).
FUSE_STR_PRED = True   # string '='/'!=' -> inline code comparison (codes[i]==target). Measured to
                       # beat both the identity-base trick and the materialised-mask path at every
                       # cardinality, fact AND gathered-parent, at sf=1 -- so no runtime switch is
                       # warranted yet. This flag is where a parent-cache-thrash threshold would go
                       # if a large-parent (sf>=10) workload ever shows the gather losing to a mask.


def _fast_detect(db, lt, la, rt, ra, lk, rk):
    fkl = db.cat.fk_pointers(lt)
    if lk in fkl and fkl[lk]['parent'] == rt and fkl[lk]['parent_key'] == rk:
        return dict(child=lt, parent=rt, child_alias=la, parent_alias=ra, fk_col=lk)
    fkr = db.cat.fk_pointers(rt)
    if rk in fkr and fkr[rk]['parent'] == lt and fkr[rk]['parent_key'] == lk:
        return dict(child=rt, parent=lt, child_alias=ra, parent_alias=la, fk_col=rk)
    return None


_SOLO_MEMO = __import__('wdb_qmem').register({})


def _solo_segment(db, name):
    """THE CLEAN VERDICT, MEMOISED: 'this table is one clean segment (or a clean union)' asked the
    filesystem ~53 times per JOB query (hot buffer, overrides, tombstones, per table). The verdict
    can only change when DML or compaction writes, and both move the catalog stamp now."""
    stamp = db._catalog_stamp() if hasattr(db, '_catalog_stamp') else None
    # THE PIN IS PART OF THE VERDICT: segment partials pin a table to ONE member (_seg_override) while
    # they iterate a union; a verdict memoised under the pin and served after it answered the whole
    # table with one segment -- four join families WRONG on the 5-segment realm (sums low by a fifth)
    _pin9 = getattr(db.cat, '_seg_override', None)
    _pk9 = tuple(sorted((k, tuple(v)) for k, v in _pin9.items())) if _pin9 else ()
    mk = (id(db), name, stamp, _pk9)
    hit = _SOLO_MEMO.get(mk) if stamp is not None else None
    if hit is not None:
        if hit == 'decline': raise _FastUnsupported
        seg9, p9 = hit
        # the segment object may have been dropped by refresh(): re-open from the cache by path
        try:
            if getattr(seg9, 'segs', None) is None: seg9 = db.open_segment(p9, name)
        except Exception:
            _SOLO_MEMO.pop(mk, None); return _solo_segment(db, name)
        return seg9, p9
    try:
        r = _solo_segment_uncached(db, name)
    except _FastUnsupported:
        if stamp is not None:
            _SOLO_MEMO[mk] = 'decline'
            if len(_SOLO_MEMO) > 4096: _SOLO_MEMO.clear()
        raise
    if stamp is not None:
        _SOLO_MEMO[mk] = r
        if len(_SOLO_MEMO) > 4096: _SOLO_MEMO.clear()
    return r


def _solo_segment_uncached(db, name):
    paths = db.cat.segment_paths(name)
    if os.path.exists(wdb_dml.hot_path(db.cat, name)): raise _FastUnsupported
    import wdb_override
    if len(paths) > 1:
        # THE MERGED-DICTIONARY VIEW: a multi-segment table is one table to the join engine and
        # every scope-stage door (this was the nineteenth `_solo_segment` gate); dirty segments
        # (tombstones, overrides) still decline -- the presence gate keeps them on the general scan
        segs9 = [db.open_segment(p, name) for p in paths]
        for sg9 in segs9:
            if sg9.presence_mask() is not None or wdb_override.load(sg9.path) or any(c.get('mode') == 6 for c in sg9.cols.values()):
                raise _FastUnsupported
        u9 = db._union(name, segs9, paths)
        return u9, u9.path
    if len(paths) != 1: raise _FastUnsupported
    seg = db.open_segment(paths[0], name)
    if seg.presence_mask() is not None: raise _FastUnsupported
    import wdb_override
    if wdb_override.load(seg.path): raise _FastUnsupported                  # column overrides (post-UPDATE)
    if any(c.get('mode') == 6 for c in seg.cols.values()): raise _FastUnsupported   # synthetic ADD COLUMN
    return seg, paths[0]


def _code_val(seg, pcol, code):
    c = seg.cols[pcol]
    if c.get('has_null') and int(code) == c['V'] - 1: return None
    return seg.fetch(pcol, int(code))


def _bulk_keyvals(seg, pcol, codes):
    """Vectorised decode of an array of group-key dictionary codes -> list of python values. Group keys are
    value-identity (mode 0/2/5/6 -- mode-4 is gated out), so a single dict index replaces a per-row fetch()."""
    c = seg.cols[pcol]; dt = c['dt']
    codes = np.asarray(codes)
    nc = (c['V'] - 1) if c['has_null'] else None
    if dt == 1 and not c['has_null']:
        # THE DECODED-DICT SHELF: a V-scale object array of Python strings
        # survives the per-query flush (the flush drops N-scale residue; a
        # dictionary is the column's vocabulary) -- consult it BEFORE asking
        # for the typed dict, or a 6.3M-string dict re-decodes every query
        cache = getattr(seg, '_str_dict_cache', None)
        if cache is not None:
            tds = cache.get(pcol)
            if tds is not None and len(tds) == int(c['V']):
                return tds[codes].tolist()
    td = seg._typed_dict(pcol)
    if not isinstance(td, np.ndarray):
        td = np.array(td, dtype=object)
    if len(td) == 0:                                  # all-null column -> every key is NULL
        return [None] * len(codes)
    safe = np.where(codes == nc, 0, codes) if nc is not None else codes   # null code -> dummy idx (fixed below)
    picked = td[safe]
    if dt == 3:                                       # int64 epochs -> datetime64 -> _pyval string
        unit = seg.unit(pcol)
        out = [wdb_sql._pyval(x) for x in picked.astype(np.int64).view(f'datetime64[{unit}]')]
    elif dt == 1:                                     # bytes -> str: decode the DICT once (V), gather at C speed
        cache = getattr(seg, '_str_dict_cache', None)
        if cache is None:
            cache = seg._str_dict_cache = {}
        tds = cache.get(pcol)
        if tds is None or len(tds) != len(td):
            tds = np.array([wdb_sql._pyval(x) for x in td], dtype=object)
            cache[pcol] = tds
        out = tds[safe].tolist()
    else:                                             # int / float -> python scalars (C-level tolist)
        out = picked.tolist()
    if nc is not None:
        cl = codes.tolist()
        out = [None if cl[i] == nc else out[i] for i in range(len(out))]
    return out


def _mode4_group(seg, pcol):
    """Affine (mode-4) GROUP BY key -> (dense gids per row, K, gid->value labels), memoised on the
    immutable segment. Factorising the column's values is invariant for a static segment, so do it
    once per (segment, column) instead of on every query (the cost the affine-key fix introduced)."""
    cache = getattr(seg, '_mode4_group_cache', None)
    if cache is None:
        cache = {}
        try: seg._mode4_group_cache = cache
        except Exception: pass
    hit = cache.get(pcol)
    if hit is not None: return hit
    vals = np.asarray(seg.values(pcol))
    gids, uniq = pd.factorize(vals, sort=False)
    full = np.ascontiguousarray(gids.astype(np.int64))
    if seg.cols[pcol]['dt'] == 3:
        u = np.asarray(uniq).astype(np.int64).view(f"datetime64[{seg.unit(pcol)}]")
        labels = [wdb_sql._pyval(x) for x in u]
    else:
        labels = [wdb_sql._pyval(x) for x in np.asarray(uniq).tolist()]
    res = (full, len(uniq), labels)
    if isinstance(cache, dict): cache[pcol] = res
    return res


def _topk_prefilter(tree, proj, col_results, counts, present, gkeys):
    """ORDER BY <projected aggregate>[DESC] LIMIT k over a GROUP BY: shrink `present` to a provable
    superset of the top-k groups with ONE numpy partition on the primary order array, so the row
    assembly materialises ~k rows instead of every group. The downstream _apply_order over the shrunk
    set stays the source of truth for exact ordering (tie-breaks, null handling), so the result is
    identical to the full path -- this only drops groups that provably cannot enter the top-k.
    Engages only when the PRIMARY order key maps to a finite numeric aggregate/COUNT column and
    k < #groups; otherwise returns `present` unchanged (full path). Pruning by primary key alone is
    a valid superset: every true top-k row has a primary value at least as good as the k-th best, so
    `a >= thresh` (desc) / `a <= thresh` (asc), with all boundary ties kept, can never exclude one."""
    if not gkeys:
        return present
    if tree.args.get('having') is not None:
        return present                   # HAVING can disqualify winners: the top-k-by-order
                                         # superset is no longer provable -- keep every group
    order = tree.args.get('order')
    lim = wdb_sql._limit(tree)
    if order is None or lim is None or lim <= 0:
        return present
    n = len(present)
    if lim >= n:
        return present
    o0 = order.expressions[0]
    desc = bool(o0.args.get('desc'))
    target = o0.this
    idx = None
    for i, p in enumerate(proj):                       # same match rule as _apply_order
        inner = p.this if isinstance(p, E.Alias) else p
        tname = target.name if isinstance(target, E.Column) else None
        if inner.sql() == target.sql() or wdb_sql._alias(p) == tname:
            idx = i; break
    if idx is None:
        return present
    r = col_results[idx]
    if r[0] == 'count':
        a = counts[present]
    elif r[0] == 'arr' and not r[2]:                   # numeric aggregate array (not datetime)
        a = r[1][present]
    else:
        return present                                 # key column / datetime primary -> full path
    a = np.asarray(a)
    if a.dtype.kind == 'O':                            # Decimal/object agg array -> float proxy for the
        try:                                           # partition only; equal values map to identical floats
            a = a.astype(np.float64)                   # and distinct sums differ far more than float error,
        except (TypeError, ValueError):                # so the >=thresh superset stays exact. None -> bail.
            return present
    if a.dtype.kind not in 'iuf':
        return present
    if a.dtype.kind == 'f' and not np.isfinite(a).all():   # NaN/inf -> null-ordering risk, full path
        return present
    k = int(lim)
    if desc:
        thresh = np.partition(a, n - k)[n - k]         # k-th largest value
        sel = np.nonzero(a >= thresh)[0]
    else:
        thresh = np.partition(a, k - 1)[k - 1]         # k-th smallest value
        sel = np.nonzero(a <= thresh)[0]
    global _TOPK_HITS; _TOPK_HITS += 1
    return present[sel]


def _radix_plan(group_keys, slot_list, exprs, no_mm, pred_body, mask):
    """Return (key_codes, K, measure) if this GROUP BY matches the radix fast-path, else None.
    Eligible: single non-gathered key, no MIN/MAX, no filter, >=1M-ish groups (replicated accumulator
    would overflow LLC), and aggregates limited to COUNT and at most one SUM/AVG over a bare non-
    gathered numeric slot. measure is None (COUNT-only) or (dict_values, codes) for the SUM payload.
    The cluster-key slice path is checked before this, so a key reaching here is non-cluster-ordered."""
    if len(group_keys) != 1: return None
    kcodes, K, kptr = group_keys[0]
    if kptr is not None or not no_mm or pred_body or mask is not None: return None
    if len(exprs) > 1: return None
    if not wdb_radix.should_use(K, wdb_exprjit._NT): return None
    if not exprs:
        return (kcodes, K, None)
    mb = re.fullmatch(r'v(\d+)', exprs[0][0])           # the SUM/AVG body must be a bare slot
    if not mb: return None
    sb, sc, sp = slot_list[int(mb.group(1))]
    if sp is not None or sb is None: return None        # need a non-gathered numeric dict slot
    return (kcodes, K, (sb, sc))


def _slice_scalar_agg(group_keys, inputs, exprs, mask, n, pred, offsets):
    """GROUP BY the cluster key (single, direct fact column) with a fused or materialised predicate:
    each cluster range is exactly one group, so run the fast register-accumulator scalar kernel once
    per slice instead of grouped_multi's per-group indexed-write accumulators -- measured ~5x on a
    predicated SUM (33.9ms -> 6.7ms) because the scalar kernel keeps the accumulators in registers and
    SIMD-reduces. Returns (counts[K], results) in grouped_multi's shape (SUM/AVG/COUNT only; the caller
    gates out MIN/MAX, which scalar_multi does not handle). Bit-identical group sums to grouped_multi."""
    codes, K, _ = group_keys[0]
    global _SLICE_SCALAR_HITS; _SLICE_SCALAR_HITS += 1
    counts = np.zeros(K, dtype=np.int64)
    sums = [np.zeros(K, dtype=np.float64) for _ in exprs]
    for gi in range(len(offsets) - 1):
        lo, hi = int(offsets[gi]), int(offsets[gi + 1])
        if hi <= lo: continue
        g = int(codes[lo])                               # cluster range is constant in the key code
        si = [(b, c[lo:hi], (None if p is None else np.ascontiguousarray(p[lo:hi]))) for (b, c, p) in inputs]
        cnt, out = wdb_exprjit.scalar_multi(si, exprs, (None if mask is None else mask[lo:hi]), hi - lo, pred)
        counts[g] += int(cnt[0])                         # += (not =) so a key split across runs still sums
        for e in range(len(exprs)):
            sums[e][g] += out[e][0][0]
    return counts, [(sums[e], None, None) for e in range(len(exprs))]


def _fast_pointer_agg(db, tree, ctx, columnar=False):
    import operator
    import time as _t9
    _tk9 = _t9.perf_counter
    _fpa_t0 = _tk9()
    _bill9 = [] if __import__('os').environ.get('WDB_JOIN_BILL') else None
    if not tree.args.get('joins'):
        # MIN/MAX OF A STRING COLUMN belongs to the dictionary (wdb_sql reads the
        # extreme PRESENT value at V-scale); the fused path would decode N strings
        frm9 = tree.args.get('from') or tree.args.get('from_')
        if frm9 is not None and isinstance(frm9.this, E.Table):
            try:
                seg9, _ = _solo_segment(db, frm9.this.name)
                pm9 = db.cat.phys_map(frm9.this.name)
                for p9 in tree.expressions:
                    nd9 = p9.this if isinstance(p9, E.Alias) else p9
                    if isinstance(nd9, (E.Min, E.Max)) and isinstance(nd9.this, E.Column):
                        if seg9.cols.get(pm9.get(nd9.this.name, nd9.this.name), {}).get('dt') == 1:
                            raise _FastUnsupported
                    # A FUNCTION OF A STRING COLUMN under an aggregate (AVG(length(URL))): the fused
                    # operands take plain numeric columns only, so it would decline -- but only after
                    # the WHERE literal and the group census were paid (Q27: ~320 ms thrown away).
                    # Decline FIRST; the dictionary's V-table reads (wdb_lenagg) serve it.
                    if isinstance(nd9, (E.Avg, E.Sum, E.Min, E.Max)) and not isinstance(nd9.this, E.Column) \
                            and any(seg9.cols.get(pm9.get(c9.name, c9.name), {}).get('dt') == 1
                                    for c9 in nd9.this.find_all(E.Column)):
                        raise _FastUnsupported
            except _FastUnsupported:
                raise
            except Exception:
                pass
    rows9 = None
    _where_spent = False
    _stage0 = _fpa_t0
    _b1 = _fpa_t0
    proj = tree.expressions
    group = tree.args.get('group')
    has_agg = any(wdb_sql._agg_kind(p) for p in proj)
    cd_col = None                                                            # COUNT(DISTINCT col), sole, no GROUP BY
    if group is None and len(proj) == 1:
        _i0 = proj[0].this if isinstance(proj[0], E.Alias) else proj[0]
        if isinstance(_i0, E.Count) and isinstance(_i0.this, E.Distinct):
            _dx = _i0.this.expressions
            if len(_dx) == 1 and isinstance(_dx[0], E.Column): cd_col = _dx[0]
            else: raise _FastUnsupported                                     # COUNT(DISTINCT expr / multi) -> fallback
    if cd_col is not None:
        gnodes = []                                                          # computed directly after mask setup
    elif tree.args.get('distinct') is not None and not has_agg and group is None:
        cols = [(p.this if isinstance(p, E.Alias) else p) for p in proj]      # SELECT DISTINCT cols == GROUP BY cols
        if not all(isinstance(c, E.Column) for c in cols): raise _FastUnsupported  # DISTINCT * / over expr
        gnodes = cols
    elif not has_agg:
        raise _FastUnsupported                                                # plain projection -> fallback
    else:
        gnodes = group.expressions if group is not None else []

    fact = ctx['fact']; alias2t = ctx['alias2t']; seg_of = ctx['seg_of']; composed = ctx['composed']
    cols_of = {a: set(db.cat.column_names(t)) for a, t in alias2t.items()}
    phys_of = {a: db.cat.phys_map(t) for a, t in alias2t.items()}

    _colmemo = {}
    def _col_cached(seg, pcol):                 # decode each column once per query (multi-agg reuse)
        k = (id(seg), pcol)
        if k not in _colmemo: _colmemo[k] = wdb_sql._col(seg, pcol)
        v9m = _colmemo[k]
        if rows9 is None:
            return v9m
        return (_rw9(v9m[0]), _rw9(v9m[1]))     # materialised operands ride survivor space

    def resolve(node):
        if not isinstance(node, E.Column):
            raise _FastUnsupported            # scalar expressions etc.: not fusable, fall back
        a, nm = node.table, node.name
        if not a:                               # unqualified: find the unique table owning the column
            owners = [al for al, cs in cols_of.items() if nm in cs]
            if len(owners) != 1: raise _FastUnsupported
            a = owners[0]
        if a not in alias2t or nm not in cols_of[a]: raise _FastUnsupported
        if a not in composed: raise _FastUnsupported   # alias outside THIS chain's tree
        return seg_of[a], phys_of[a].get(nm, nm), composed[a]   # composed[a] is None for the fact table
    def col_operand(node):
        # parent columns become ('g', arr, composed_ptr) so the gather happens per-chunk inside the
        # threaded kernel instead of materialising the full gathered array here.
        if not isinstance(node, E.Column):
            raise _FastUnsupported                  # THE BARE-COLUMN LAW: CAST(ts AS DATE) resolved through .name to ts
        seg, pcol, cptr = resolve(node)
        if cptr is None:
            raw = wdb_sql.raw_dict_col(seg, pcol)         # plain dict numeric col -> defer/fuse the decode
            if raw is not None:
                return ('raw', raw[0], raw[1]), None, seg, pcol
        arr, nm = _col_cached(seg, pcol)
        if seg.cols[pcol]['dt'] == 3 and getattr(arr, 'dtype', None) is not None and arr.dtype.kind == 'M':
            arr = arr.view('int64')                       # datetime64 -> epoch ints for the numba kernel
        if cptr is not None:
            return ('g', arr, cptr), (('g', nm, cptr) if nm is not None else None), seg, pcol
        return ('d', arr), (('d', nm) if nm is not None else None), seg, pcol

    _ARITH = {E.Add: operator.add, E.Sub: operator.sub, E.Mul: operator.mul, E.Div: operator.truediv}
    def eval_arith(node):
        # Materialise an arithmetic expression to a per-fact-row value array (+ combined null mask).
        # Each column is resolved through its composed pointer, so expressions may mix tables in the chain.
        if isinstance(node, E.Cast) and any(k in node.to.sql().upper() for k in ('DATE', 'TIME', 'CHAR', 'TEXT', 'STRING', 'BOOL')):
            raise _FastUnsupported                  # a TYPE-CHANGING cast is not transparent
        if isinstance(node, (E.Paren, E.Cast)): return eval_arith(node.this)
        if isinstance(node, E.Neg):
            a, na = eval_arith(node.this); return -a, na
        if isinstance(node, E.Column):
            seg, pcol, cptr = resolve(node)
            arr, nm = _col_cached(seg, pcol)
            if cptr is not None:
                arr = arr[cptr]; nm = nm[cptr] if nm is not None else None
            return arr, nm
        if isinstance(node, E.Literal):
            if node.is_string: raise _FastUnsupported
            v = node.this
            return (float(v) if ('.' in v or 'e' in v.lower()) else int(v)), None
        if type(node) in _ARITH:
            a, na = eval_arith(node.this); b, nb = eval_arith(node.expression)
            out = _ARITH[type(node)](a, b)
            nm = na if nb is None else (nb if na is None else (na | nb))   # NULL if any operand is NULL
            return out, nm
        raise _FastUnsupported
    _ARITH_STR = {E.Add: '+', E.Sub: '-', E.Mul: '*', E.Div: '/'}
    def fused_expr_build(argnode):
        # Compile an arithmetic tree to (numba-source body over slot vars v0.., [(base, codes) per slot]) so
        # wdb_exprjit can fuse decode+expression+aggregate into one pass (no materialised array). Fact (direct)
        # dict-numeric columns only; raises _FastUnsupported on anything else so the caller materialises instead.
        if not wdb_exprjit.HAS_NUMBA: raise _FastUnsupported
        inputs = []; slot = {}
        def emit(node):
            if isinstance(node, E.Paren): return emit(node.this)
            if isinstance(node, E.Cast):
                _t9 = node.to.sql().upper()
                if any(k in _t9 for k in ('DATE', 'TIME', 'CHAR', 'TEXT', 'STRING', 'BOOL')):
                    raise _FastUnsupported            # a TYPE-CHANGING cast is not transparent (CAST(ts AS DATE) = d compared us to days)
                return emit(node.this)
            if isinstance(node, E.Neg): return f"(-{emit(node.this)})"
            if isinstance(node, E.Column):
                seg, pcol, cptr = resolve(node)               # cptr: fact->parent pointer (None for the fact)
                raw = wdb_sql.raw_dict_col(seg, pcol)
                if raw is None: raise _FastUnsupported               # nullable / string / computed -> fallback
                key = (id(seg), pcol, id(cptr) if cptr is not None else None)
                if key not in slot:
                    slot[key] = len(inputs)
                    _c9f = (np.asarray(seg.codes_at(pcol, rows9))
                            if (cptr is None and rows9 is not None) else
                            (_rw9(raw[1]) if cptr is None else raw[1]))
                    inputs.append((np.ascontiguousarray(raw[0]), np.ascontiguousarray(_c9f),
                                   None if cptr is None else np.ascontiguousarray(_rw9(cptr))))
                return f"v{slot[key]}"
            if isinstance(node, E.Literal):
                if node.is_string: raise _FastUnsupported
                v = node.this
                return f"({float(v)})" if ('.' in v or 'e' in v.lower()) else f"({int(v)})"
            if type(node) in _ARITH_STR:
                return f"({emit(node.this)} {_ARITH_STR[type(node)]} {emit(node.expression)})"
            raise _FastUnsupported
        body = emit(argnode)
        if not inputs: raise _FastUnsupported                        # pure constant -> not an aggregation input
        return body, inputs

    def agg_arg_operand(argnode):
        # SUM/AVG/MIN/MAX/COUNT argument: a bare column keeps its seg/pcol (for datetime MIN/MAX); an
        # arithmetic expression is materialised to a direct ('d', arr) operand the kernel slices per chunk.
        if isinstance(argnode, E.Column):
            return col_operand(argnode)
        arr, nm = eval_arith(argnode)
        if not hasattr(arr, 'shape'): raise _FastUnsupported          # need a per-row array, not a constant
        return ('d', arr), (('d', nm) if nm is not None else None), None, None

    # ---- WHERE -> boolean mask over child rows ----
    # Each predicate is evaluated on the UN-gathered column (the small parent side when it is a parent
    # column) and the resulting bool is gathered to child rows -- and string =, !=, IN compare integer
    # CODES, never materialised strings. Both avoid touching a 6M-row object array (measured 76 -> ~3 ms).
    _OPS = {E.EQ: operator.eq, E.NEQ: operator.ne, E.GT: operator.gt, E.LT: operator.lt,
            E.GTE: operator.ge, E.LTE: operator.le}
    def _str_codes(seg, pcol):
        c = seg.cols[pcol]
        if c['dt'] != 1 or c['mode'] == 4: return None        # only value-identity string dicts
        if c['mode'] == 5 and int(seg.N) >= 200_000:
            # AN INLINE COLUMN HAS NO USEFUL CODES: V ~ N, and factorising 4.2M names to build a
            # code lookup for a LIKE cost 4s before the door DECLINED anyway (name LIKE 'Downey%').
            # Decline by name here, before the work, not after it.
            raise _FastUnsupported
        # code_of ({value_bytes: code}) is expensive to build for a high-card dict (e.g. 18M URLs ~8s).
        # It depends only on the column's dictionary, so cache it on the segment -- built at most once
        # per column, not once per query.
        cache = getattr(seg, '_strcode_cache', None)
        if cache is None: cache = seg._strcode_cache = {}
        if pcol in cache:
            code_of = cache[pcol]
        else:
            if c['mode'] == 5: seg._raw_codes(pcol); dv = seg.cols[pcol].get('_idict')
            else:
                try: dv = seg.dict_vals(pcol)
                except Exception: return None
            if dv is None: return None
            code_of = {(v if isinstance(v, (bytes, bytearray)) else str(v).encode()): i for i, v in enumerate(dv)}
            cache[pcol] = code_of
        return seg.codes(pcol), code_of, (c['V'] - 1 if c['has_null'] else None)
    def _code_of_literal(seg, pcol, lit_bytes):
        """Resolve ONE literal to its dictionary code without building the full {value:code} dict.
        DuckDB-style: a filter like URL = 'x' only needs x's code, not a dict of all 18M values.
        Vectorized search over the dict values (object-array ==), result cached per (col, literal).
        Returns the code (int) or -1 if the literal is absent from the dictionary."""
        c = seg.cols[pcol]
        litcache = getattr(seg, '_litcode_cache', None)
        if litcache is None: litcache = seg._litcode_cache = {}
        key = (pcol, bytes(lit_bytes))
        if key in litcache: return litcache[key]
        # if the full code_of is already cached, just use it (no rebuild)
        full = getattr(seg, '_strcode_cache', {}).get(pcol)
        if full is not None:
            code = full.get(bytes(lit_bytes), -1)
            litcache[key] = code; return code
        if c.get('dt') == 1 and c.get('mode') in (0, 1) and int(c.get('V', 0)) >= 50_000:
            # A SORTED DICTIONARY IS A BINARY SEARCH: ~22 probes through fetch (one chunk each),
            # never the full decode (dict_vals of 6M URLs: 12s, to answer `URL <> ''`)
            try:
                import wdb_wherescan
                _cd = wdb_wherescan._code_of(seg, pcol, bytes(lit_bytes))
                code = -1 if _cd is None else int(_cd)
                litcache[key] = code; return code
            except NotImplementedError:
                raise
            except Exception:
                pass
        try:
            if c['mode'] == 5: seg._raw_codes(pcol); dv = seg.cols[pcol].get('_idict')
            else: dv = seg.dict_vals(pcol)
        except NotImplementedError:
            raise                                       # a DECLINE BY NAME is never a missing literal (the union: 0 rows for 533)
        except Exception:
            dv = None
        if dv is None:
            litcache[key] = -1; return -1
        arr = dv if isinstance(dv, np.ndarray) else np.asarray(dv, dtype=object)
        hits = np.nonzero(arr == lit_bytes)[0]          # vectorized; no per-value Python dict
        code = int(hits[0]) if len(hits) else -1
        litcache[key] = code; return code
    def _lit_bytes(seg, pcol, e):
        b = wdb_sql._lit_for_col(seg, pcol, e, 'O')
        return b if isinstance(b, (bytes, bytearray)) else str(b).encode()
    def leaf(colnode, make_bool):
        if not isinstance(colnode, E.Column):
            # NEVER a silent column: a function/expression LHS (SUBSTRING(id3,3,2) = '10')
            # used to resolve through .name to the inner column and evaluate id3 = '10'
            raise _FastUnsupported
        seg, pcol, cptr = resolve(colnode)       # evaluate un-gathered, then gather the bool via composed ptr
        b = make_bool(seg, pcol)
        return b if cptr is None else b[cptr]
    def mask_eval(node):
        if isinstance(node, E.Paren): return mask_eval(node.this)
        if isinstance(node, E.And): return mask_eval(node.this) & mask_eval(node.expression)
        if isinstance(node, E.Or):  return mask_eval(node.this) | mask_eval(node.expression)
        if isinstance(node, E.Not): return ~mask_eval(node.this)
        if (type(node) in (E.GT, E.LT) and isinstance(node.this, E.Column)
                and isinstance(node.expression, E.Column)):
            # THE DECLARED CLOCK in the mask layer: strict pair compares on an
            # enc-16 dressed pair serve from the bit+delta streams (Q21's
            # lateness reached get_mask through the lonely rewrite).
            sgA9, pcA9, cpA9 = resolve(node.this)
            sgB9, pcB9, cpB9 = resolve(node.expression)
            if sgA9 is sgB9 and cpA9 is None and cpB9 is None:
                big9, small9 = (pcA9, pcB9) if isinstance(node, E.GT) else (pcB9, pcA9)
                cB9 = sgA9.cols.get(big9, {})
                if cB9.get('code_enc') == 16 and cB9.get('e16_partner') == small9:
                    b9m, d9m = sgA9.pair_bits(big9)
                    return np.asarray(b9m) & (np.asarray(d9m) > 0)
                cS9 = sgA9.cols.get(small9, {})
                if cS9.get('code_enc') == 16 and cS9.get('e16_partner') == big9:
                    b9m, d9m = sgA9.pair_bits(small9)
                    return (~np.asarray(b9m)) & (np.asarray(d9m) > 0)
            raise _FastUnsupported

        if type(node) in _OPS:
            op = type(node)
            def mk(seg, pcol):
                if op in (E.EQ, E.NEQ):
                    c = seg.cols[pcol]
                    if c['dt'] == 1 and c['mode'] != 4:      # value-identity string dict: compare codes
                        codes = seg.codes(pcol)
                        nullcode = (c['V'] - 1) if c['has_null'] else None
                        tc = _code_of_literal(seg, pcol, _lit_bytes(seg, pcol, node.expression))
                        if op is E.EQ: return codes == tc
                        res = codes != tc
                        if nullcode is not None: res &= (codes != nullcode)   # SQL: NULL != x is not TRUE
                        return res
                c9 = seg.cols[pcol]
                if c9.get('mode') in (0, 1, 2) and c9.get('mode') != 4:
                    # THE JOIN FUNNEL, tier 1: predicate -> V-sized bool over
                    # the dictionary (_dict_keep), gathered through codes. No
                    # value decode, no astype -- the mask costs one u-int
                    # gather. Null codes read False (SQL: NULL op x not TRUE).
                    try:
                        td9 = np.asarray(seg._typed_dict(pcol))
                    except Exception:
                        td9 = None
                    if td9 is not None and td9.dtype.kind in 'iuf' and len(td9):
                        keep9 = _dict_keep(node, seg, pcol, td9)
                        if keep9 is not None:
                            kx9 = np.zeros(int(c9['V']), bool)
                            kx9[:len(keep9)] = keep9
                            return kx9[np.asarray(seg.codes(pcol))]
                arr, _ = _col_cached(seg, pcol)
                v = wdb_sql._lit_for_col(seg, pcol, node.expression, arr.dtype.kind)
                if arr.dtype.kind == 'M': arr = arr.view('int64')   # datetime: compare as epoch ints
                return _OPS[op](arr, v)
            return leaf(node.this, mk)
        if isinstance(node, E.Between):
            def mk(seg, pcol):
                arr, _ = _col_cached(seg, pcol)
                lo = wdb_sql._lit_for_col(seg, pcol, node.args['low'], arr.dtype.kind)
                hi = wdb_sql._lit_for_col(seg, pcol, node.args['high'], arr.dtype.kind)
                if arr.dtype.kind == 'M': arr = arr.view('int64')   # datetime: compare as epoch ints
                return (arr >= lo) & (arr <= hi)
            return leaf(node.this, mk)
        if isinstance(node, E.In):
            def mk(seg, pcol):
                kcs = node.args.get('_codes')
                if kcs is not None:              # same-column subquery pre-resolved to a
                    V0 = int(seg.cols[pcol]['V'])                   # CODE SET: a V-flag
                    fl = np.zeros(V0, bool)                         # + one gather, never
                    fl[np.asarray(kcs, dtype=np.int64)] = True      # np.isin's 500ms sort
                    pl = seg.e8_planes(pcol) if hasattr(seg, 'e8_planes') else None
                    if pl is not None:           # sparse dress: paint ONCE per query
                        ckm = '_e8mk_' + pcol
                        hit = seg._codes.get(ckm)
                        if hit is not None and hit[0] == id(kcs):
                            return hit[1]
                        pos8, lits8, d8 = pl
                        out = np.full(int(seg.N), bool(fl[d8]))
                        out[pos8] = fl[lits8]
                        seg._codes[ckm] = (id(kcs), out)
                        return out
                    return fl[np.asarray(seg.codes(pcol))]          # native width
                if node.args.get('query') is not None:
                    raise _FastUnsupported       # unresolved subquery: not a literal list
                exprs = node.args.get('expressions') or []
                c = seg.cols[pcol]
                if c['dt'] == 1 and c['mode'] != 4:
                    tcs = [t for t in (_code_of_literal(seg, pcol, _lit_bytes(seg, pcol, e)) for e in exprs) if t >= 0]
                    V0 = int(c['V'])
                    fl = np.zeros(V0 + 1, bool)
                    tcs2 = np.asarray(tcs, dtype=np.int64)
                    fl[tcs2[(tcs2 >= 0) & (tcs2 < V0)]] = True
                    pl = seg.e8_planes(pcol) if hasattr(seg, 'e8_planes') else None
                    if pl is not None:
                        pos8, lits8, d8 = pl
                        out = np.full(int(seg.N), bool(fl[d8]))
                        out[pos8] = fl[lits8]
                        return out
                    return fl[np.asarray(seg.codes(pcol))]
                arr, _ = _col_cached(seg, pcol)
                vals = [wdb_sql._lit_for_col(seg, pcol, e, arr.dtype.kind) for e in exprs]
                return np.isin(arr, vals)
            return leaf(node.this, mk)
        if isinstance(node, E.Is) and isinstance(node.expression, E.Null):   # IS NULL (Not(Is) = IS NOT NULL)
            def mk(seg, pcol):
                c = seg.cols[pcol]
                if not c['has_null']: return np.zeros(int(seg.N), dtype=bool)  # non-nullable -> nothing, WITHOUT reading the column
                codes = seg.codes(pcol)
                return codes == (c['V'] - 1)                                   # null is reserved code V-1
            return leaf(node.this, mk)
        if isinstance(node, (E.Like, E.ILike)):              # LIKE -> code-LUT over the dict, gathered
            negate = bool(node.args.get('negate')); ci = isinstance(node, E.ILike); pnode = node.expression
            def mk(seg, pcol):
                sc = _str_codes(seg, pcol)
                if sc is None: raise _FastUnsupported       # non-value-identity dict -> can't map codes
                codes, code_of, nullcode = sc
                pb = _lit_bytes(seg, pcol, pnode)
                patt = pb.decode('utf-8', 'replace') if isinstance(pb, (bytes, bytearray)) else str(pb)
                rx = re.compile('^' + re.escape(patt).replace('%', '.*').replace('_', '.') + '$',
                                re.DOTALL | (re.IGNORECASE if ci else 0))
                ncodes = max(max(code_of.values(), default=-1),
                             nullcode if nullcode is not None else -1) + 1
                keep = np.zeros(ncodes, dtype=bool)
                for vb, cd in code_of.items():
                    v = vb.decode('utf-8', 'replace') if isinstance(vb, (bytes, bytearray)) else str(vb)
                    if rx.match(v): keep[cd] = True
                m = keep[codes]
                return ~m if negate else m
            return leaf(node.this, mk)
        raise _FastUnsupported
    where = tree.args.get('where')
    _maskc = {}
    def get_mask():                # materialise the WHERE bool mask lazily -- only non-fused paths need it
        if _where_spent:
            return None                # the cascade already applied WHERE in row space
        if 'm' not in _maskc:
            _maskc['m'] = mask_eval(where.this) if where is not None else None
        return _maskc['m']
    def _mask_op():
        m = get_mask(); return ('d', m) if m is not None else None

    def _dict_keep(node, seg, pcol, td):
        # Evaluate a WHERE predicate over the (tiny) dictionary value array td -> bool[len(td)].
        # Single column only; returns None on any shape not reducible to td (caller falls back).
        if isinstance(node, E.Paren): return _dict_keep(node.this, seg, pcol, td)
        if isinstance(node, E.And):
            a = _dict_keep(node.this, seg, pcol, td); b = _dict_keep(node.expression, seg, pcol, td)
            return None if a is None or b is None else (a & b)
        if isinstance(node, E.Or):
            a = _dict_keep(node.this, seg, pcol, td); b = _dict_keep(node.expression, seg, pcol, td)
            return None if a is None or b is None else (a | b)
        if isinstance(node, E.Not):
            a = _dict_keep(node.this, seg, pcol, td); return None if a is None else ~a
        if type(node) in _OPS:
            if not (isinstance(node.this, E.Column) and isinstance(node.expression, (E.Literal, E.Neg))):
                return None                                    # col OP literal only (not expr/flip)
            v = wdb_sql._lit_for_col(seg, pcol, node.expression, td.dtype.kind)
            return _OPS[type(node)](td, v)
        if isinstance(node, E.Between):
            if not isinstance(node.this, E.Column): return None
            lo = wdb_sql._lit_for_col(seg, pcol, node.args['low'], td.dtype.kind)
            hi = wdb_sql._lit_for_col(seg, pcol, node.args['high'], td.dtype.kind)
            return (td >= lo) & (td <= hi)
        if isinstance(node, E.In):
            if not isinstance(node.this, E.Column): return None
            exprs = node.args.get('expressions') or []
            if not exprs: return None
            vals = [wdb_sql._lit_for_col(seg, pcol, e, td.dtype.kind) for e in exprs]
            return np.isin(td, vals)
        return None

    def _dict_count(wnode):
        # COUNT(*) WHERE wnode, when wnode is a predicate on a single value-identity numeric dict
        # FACT column -> sum cached per-code counts over qualifying dict entries. None -> normal path.
        try:
            colnodes = list(wnode.find_all(E.Column))
            if not colnodes: return None
            seg0 = pcol0 = None
            for cn in colnodes:
                s, p, cp = resolve(cn)
                if cp is not None: return None                 # parent/join column -> not a solo fact col
                if seg0 is None: seg0, pcol0 = s, p
                elif id(s) != id(seg0) or p != pcol0: return None   # more than one column
            c = seg0.cols[pcol0]
            if c['mode'] == 4 or c['dt'] not in (0, 2): return None  # value-identity int/float only (v1)
            td = np.asarray(seg0._typed_dict(pcol0))
            if td.dtype.kind not in 'iuf' or len(td) == 0: return None
            keep = _dict_keep(wnode, seg0, pcol0, td)
            if keep is None: return None
            counts = seg0.code_counts(pcol0)                   # cached bincount, length V
            return int(counts[:len(td)][keep].sum())           # [:len(td)] excludes the null bin
        except Exception:
            return None

    # ---- group codes (as an operand; gathered per-chunk in the threaded kernel) ----
    n = ctx['n']
    if n == 0 and gnodes: return [], [wdb_sql._alias(p) for p in proj]   # GROUP BY over 0 rows -> no groups
    # (no GROUP BY over 0 rows falls through: SQL still emits one grand-total row -- COUNT=0, SUM/MIN/MAX=NULL)
    if cd_col is not None:                       # COUNT(DISTINCT col) == # distinct non-null codes among matches
        cseg, cpcol, ccptr = resolve(cd_col)
        cc = cseg.cols[cpcol]
        if cc['mode'] == 4: raise _FastUnsupported                         # codes not value-identity -> fallback
        m = get_mask()
        if m is None and ccptr is None:           # no filter: distinct count == dictionary cardinality (O(1))
            _bump_fast()
            return [(int(cc['V'] - cc['has_null']),)], [wdb_sql._alias(proj[0])]
        codes = cseg.codes(cpcol); codes = codes if ccptr is None else codes[ccptr]
        if m is not None: codes = codes[m]
        uniq = np.unique(codes) if codes.size else np.empty(0, dtype=np.int64)
        if cc['has_null']: uniq = uniq[uniq != cc['V'] - 1]                # COUNT(DISTINCT) ignores NULL
        _bump_fast()
        return [(int(uniq.size),)], [wdb_sql._alias(proj[0])]

    # ---- pure COUNT(*) with a single value-identity dict-column predicate -------------------------
    # COUNT(*) WHERE P(col) == sum of per-code row counts over the codes whose dict value satisfies P.
    # O(distinct) over the (tiny) dictionary instead of materialising + scanning N values. One bincount
    # per column, cached on the segment. Numeric int/float dict columns; any other shape returns None
    # and falls through to the normal scan. Fail-safe: any unexpected node -> None -> normal path.
    if (group is None and where is not None and cd_col is None and len(proj) == 1
            and not gnodes):
        _p0 = proj[0].this if isinstance(proj[0], E.Alias) else proj[0]
        if (isinstance(_p0, E.Count) and not isinstance(_p0.this, E.Distinct)
                and (_p0.this is None or isinstance(_p0.this, E.Star))):
            _cnt = _dict_count(where.this)
            if _cnt is not None:
                _bump_fast()
                return [(int(_cnt),)], [wdb_sql._alias(proj[0])]

    # ---- whole-table SUM/AVG/MIN/MAX (+COUNT*) from the per-value tally (no GROUP BY, no WHERE) ------
    # Every reduction over a value-identity numeric dict column is a reduction over (value, frequency):
    # SUM = dict . code_counts (one BLAS dot), AVG = SUM / total, MIN/MAX = extreme value with a nonzero
    # count. O(distinct), no row scan and no gather. Self-gated: engaged only when the column actually
    # compresses (n_dict < N * TALLY_MAX_RATIO); a near-all-distinct column has no repeats to exploit and
    # falls through to the scan kernel below with identical results. Floats only in v1 (dt==2).
    def _whole_tally():
        plan = []                                          # ('sum'|'avg'|'min'|'max', seg, pcol) | ('cnt',_,_)
        has_red = False
        for p in proj:
            node = p.this if isinstance(p, E.Alias) else p
            if (isinstance(node, E.Count) and not isinstance(node.this, E.Distinct)
                    and (node.this is None or isinstance(node.this, E.Star))):
                plan.append(('cnt', None, None)); continue
            if not isinstance(node, (E.Sum, E.Avg, E.Min, E.Max)): return None
            col = node.this
            if not isinstance(col, E.Column): return None              # SUM(a*b) etc -> needs the scan
            s, pc, cp = resolve(col)
            if cp is not None: return None                             # gathered/joined column -> not whole-table
            cc = s.cols[pc]
            if cc['mode'] != 0 or cc['dt'] != 2: return None           # value-identity float dict only (v1)
            if not RT.tally_worth_it(cc['n_dict'], n): return None    # ~no repeats -> scan instead
            plan.append((node.key, s, pc)); has_red = True
        return plan if has_red else None                              # leave pure COUNT(*) to the scalar path

    if group is None and where is None and cd_col is None and not gnodes and proj and n > 0:
        _tp = _whole_tally()
        if _tp is not None:
            row = []
            for op, s, pc in _tp:
                if op == 'cnt': row.append(int(n)); continue
                nd = s.cols[pc]['n_dict']
                counts = s.code_counts(pc)[:nd]
                dv = np.asarray(s._typed_dict(pc), dtype=np.float64)[:nd]
                if op == 'sum':   row.append(float(np.dot(dv, counts)))
                elif op == 'avg':
                    tot = float(counts.sum()); row.append(float(np.dot(dv, counts) / tot) if tot else None)
                else:
                    pres = counts > 0
                    row.append(None if not pres.any() else
                               (float(dv[pres].min()) if op == 'min' else float(dv[pres].max())))
            _bump_fast()
            return [tuple(row)], [wdb_sql._alias(p) for p in proj]

    # ============ THE SURVIVOR CASCADE (Jackson's PEMDAS at pipeline scope):
    # filter FIRST, in code space, potency-descending -- everything downstream
    # (group keys, gids, slots, kernel, emit) exists only at survivor scale.
    # Atomic: every WHERE conjunct must be a single-column dict predicate the
    # cascade can serve (numeric LUT via _dict_keep; string EQ/IN via
    # literal->code), else the whole cascade declines and the old path runs.
    rows9 = None
    _where_spent = False
    _w9c = tree.args.get('where')
    if _w9c is not None and len(gnodes) == 1:
        try:
            _g0s, _g0p, _g0c = resolve(gnodes[0])
            if (_g0c is None and _g0s.cluster_meta() is not None
                    and _g0s.cluster_meta().get('key') == _g0p
                    and _g0s.presence_mask() is None):
                _w9c = None            # cluster slice-scalar lane is faster: cascade stands down
        except Exception:
            pass
    if _w9c is not None and cd_col is None:           # grouped OR scalar (Q19)
        try:
            def _cflat(x):
                if isinstance(x, E.Paren): return _cflat(x.this)
                if isinstance(x, E.And):
                    return _cflat(x.this) + _cflat(x.expression)
                return [x]
            _cb9 = [] if _bill9 is not None else None
            _ct9 = _tk9()
            plan9 = []
            _cres9 = []                      # conjuncts the cascade cannot judge: the
                                             # pred re-applies the WHERE at survivors
            for cn in _cflat(_w9c.this):
                if cn.find(E.Select) is not None:
                    raise _FastUnsupported          # subqueries: never the cascade's to judge
                cols9 = list(cn.find_all(E.Column))
                if len(cols9) != 1:
                    _cres9.append(cn); continue     # multi-column (Q7's OR): residual
                # THE BARE-COLUMN LAW: a keep is judged over ONE column's dictionary,
                # so every comparison in the conjunct must stand on that bare column.
                # SUBSTRING(id3,3,2) = '10' has one column and was judged as id3 = '10'
                # (an empty keep, a silent zero) -- a function LHS is residual.
                _CMP9 = (E.EQ, E.NEQ, E.GT, E.GTE, E.LT, E.LTE, E.Between, E.In, E.Like, E.ILike, E.Is)
                _cmps9 = [cn] if isinstance(cn, _CMP9) else []
                _cmps9 += [x for x in cn.find_all(*_CMP9) if x is not cn]
                if not _cmps9 or any(not isinstance(x.this, E.Column) for x in _cmps9):
                    _cres9.append(cn); continue
                cs9, cp9, cc9p = resolve(cols9[0])
                c9 = cs9.cols.get(cp9) or {}
                kx9 = None
                if isinstance(cn, E.In) and cn.args.get('_codes') is not None:
                    # A PRE-RESOLVED CODE SET (the lonely rewrite, exists roads):
                    # the keep IS the code list -- paint it, no judging needed.
                    V9 = int(c9.get('V') or cs9.N)
                    kx9 = np.zeros(V9 + 1, bool)
                    _cd9x = np.asarray(cn.args['_codes'], dtype=np.int64)
                    kx9[_cd9x[(_cd9x >= 0) & (_cd9x < V9)]] = True
                    td9 = None
                elif c9.get('mode') == 5 and int(cs9.N) <= 4096 and c9.get('dt') == 1 and not c9.get('has_null'):
                    # TINY MODE-5 DIM (nation names): the sorted inline dict is
                    # the rank space the codes index -- keep by literal match.
                    lits9 = None
                    if isinstance(cn, E.EQ) and isinstance(cn.expression, E.Literal) and cn.expression.is_string:
                        lits9 = {str(cn.expression.this)}
                    elif (isinstance(cn, E.In) and cn.expressions
                          and all(isinstance(e9, E.Literal) and e9.is_string for e9 in cn.expressions)):
                        lits9 = {str(e9.this) for e9 in cn.expressions}
                    if lits9 is None:
                        _cres9.append(cn); continue
                    cs9._raw_codes(cp9)
                    idict9 = list(c9['_idict'])
                    V9 = len(idict9)
                    kx9 = np.zeros(V9 + 1, bool)
                    for r9k, v9k in enumerate(idict9):
                        sv9k = v9k.decode('utf-8', 'replace') if isinstance(v9k, (bytes, bytearray)) else str(v9k)
                        kx9[r9k] = sv9k in lits9
                    td9 = None
                elif c9.get('mode') == 4 and c9.get('dt') in (0, 3) and not c9.get('has_null'):
                    # MODE-4 SEQUENCE KEY (o_orderkey): V == N, codes are the
                    # identity; a literal keep marks positions (Q18's 100 monster
                    # orders) -- searchsorted when the sequence is sorted.
                    _rm4 = wdb_sql.raw_dict_col(cs9, cp9, want_codes=False)
                    if _rm4 is None:
                        _cres9.append(cn); continue
                    _bv4 = _rm4[0]
                    lits4 = None
                    if isinstance(cn, E.EQ) and isinstance(cn.expression, E.Literal) and not cn.expression.is_string:
                        lits4 = [int(cn.expression.this)]
                    elif (isinstance(cn, E.In) and cn.expressions
                          and all(isinstance(e9, E.Literal) and not e9.is_string for e9 in cn.expressions)):
                        lits4 = [int(e9.this) for e9 in cn.expressions]
                    if lits4 is None:
                        _cres9.append(cn); continue
                    V9 = int(_bv4.shape[0])
                    kx9 = np.zeros(V9 + 1, bool)
                    _sorted4 = c9.get('_m4_sorted')
                    if _sorted4 is None:
                        _sorted4 = c9['_m4_sorted'] = bool(V9 < 2 or (np.diff(_bv4) > 0).all())
                    _la4 = np.asarray(sorted(lits4), dtype=np.int64)
                    if _sorted4:
                        _pos4 = np.searchsorted(_bv4, _la4)
                        _ok4 = (_pos4 < V9)
                        _ok4[_ok4] = _bv4[_pos4[_ok4]] == _la4[_ok4]
                        kx9[_pos4[_ok4]] = True
                    else:
                        # UNSORTED SEQUENCE: a direct compare per literal beats np.isin's sort
                        # (fjdb encodes AdvEngineID as a 100M-value sequence: 130ms -> 40ms)
                        if len(_la4) <= 8:
                            _m4 = _bv4 == _la4[0]
                            for _lv4 in _la4[1:]: _m4 |= (_bv4 == _lv4)
                            kx9[:V9] = _m4
                        else:
                            kx9[:V9] = np.isin(_bv4, _la4)
                    td9 = None
                elif c9.get('mode') not in (0, 1, 2):
                    _cres9.append(cn); continue
                if kx9 is None:
                    V9 = int(c9['V'])
                    kx9 = np.zeros(V9 + 1, bool)   # +1: null sentinel bin, False by law
                    if (c9.get('dt') == 1 and V9 >= 200_000 and type(cn) in (E.EQ, E.NEQ)
                            and isinstance(cn.expression, E.Literal) and isinstance(cn.this, E.Column)):
                        # THE BIG-DICTIONARY LITERAL: one binary search, never the whole dictionary --
                        # the fused door spent 18s decoding 6M URLs for `URL <> ''` and then DECLINED
                        # the query (Q27 cold), leaving the general scan to answer in 1.5s
                        _c9l = _code_of_literal(cs9, cp9, _lit_bytes(cs9, cp9, cn.expression))
                        if isinstance(cn, E.EQ):
                            if _c9l is not None and _c9l >= 0: kx9[_c9l] = True
                        else:
                            kx9[:V9] = True
                            if _c9l is not None and _c9l >= 0: kx9[_c9l] = False
                            if c9.get('has_null'): kx9[V9 - 1] = False
                        td9 = None
                    else:
                        td9 = np.asarray(cs9._typed_dict(cp9))
                if td9 is None:
                    pass
                elif td9.dtype.kind in 'iuf' and len(td9):
                    keep9 = _dict_keep(cn, cs9, cp9, td9)
                    if keep9 is None:
                        _cres9.append(cn); continue
                    kx9[:len(keep9)] = keep9
                elif isinstance(cn, E.EQ) and isinstance(cn.expression, E.Literal):
                    code9 = _code_of_literal(cs9, cp9, _lit_bytes(cs9, cp9, cn.expression))
                    if code9 is not None: kx9[code9] = True
                elif isinstance(cn, E.In):
                    if not cn.expressions or any(not isinstance(e9, E.Literal) for e9 in cn.expressions):
                        _cres9.append(cn); continue
                    for e9 in cn.expressions:
                        code9 = _code_of_literal(cs9, cp9, _lit_bytes(cs9, cp9, e9))
                        if code9 is not None: kx9[code9] = True
                else:
                    _cres9.append(cn); continue
                try:
                    cnt9 = np.asarray(cs9.code_counts(cp9))[:V9]
                except Exception:
                    cnt9 = np.bincount(np.asarray(cs9.codes(cp9)), minlength=V9)[:V9]
                t9s = cnt9.sum()
                prune9 = 1.0 - float(cnt9[kx9[:cnt9.size]].sum()) / t9s if t9s else 0.5
                cost9 = 1.0 if cc9p is None else 6.0
                plan9.append((prune9 / cost9, prune9, kx9, cs9, cp9, cc9p))
            if _cb9 is not None:
                _cb9.append(('plan-build(metadata+censuses)', _tk9() - _ct9)); _ct9 = _tk9()
            if not plan9: raise _FastUnsupported        # nothing servable: no cascade
            # CONJUNCT FUSION: same-column keeps AND at dict scale into ONE
            # serve (Q7's l_shipdate >= a / <= b paid two full plane_tests).
            _fused9 = {}
            for _pt9 in plan9:
                _kk9 = (id(_pt9[3]), _pt9[4], id(_pt9[5]) if _pt9[5] is not None else None)  # per ALIAS (n1/n2)
                if _kk9 in _fused9:
                    _o9 = _fused9[_kk9]
                    _kx9 = _o9[2] & _pt9[2]
                    try:
                        _cn9f = np.asarray(_pt9[3].code_counts(_pt9[4]))[:_kx9.size - 1]
                        _t9f = _cn9f.sum()
                        _pr9 = 1.0 - float(_cn9f[_kx9[:_cn9f.size]].sum()) / _t9f if _t9f else 0.5
                    except Exception:
                        _pr9 = 1.0 - (1.0 - _o9[1]) * (1.0 - _pt9[1])
                    _cost9 = 1.0 if _pt9[5] is None else 6.0
                    _fused9[_kk9] = (_pr9 / _cost9, _pr9, _kx9, _pt9[3], _pt9[4], _pt9[5])
                else:
                    _fused9[_kk9] = _pt9
            plan9 = list(_fused9.values())
            # THE ARBITER: leftovers are tolerated only when the plan holds a
            # POTENT PARENT KEEP through a road (Q7's nation INs: expected keep
            # < 5%). Fact-only plans with leftovers decline to the partition,
            # whose shadow scheduling serves clock-shaped queries better
            # (Q12 regressed 265 -> 358 under the cascade, 2026-09-03).
            if _cres9:
                _pk9 = [p for p in plan9 if p[5] is not None]
                _keep9 = 1.0
                for p in _pk9:
                    _keep9 *= max(0.0, 1.0 - float(p[1]))
                # ...and only when the per-survivor work is HEAVY enough that
                # shrinking survivors pays for the cascade's own passes: two or
                # more group keys, or a computed/dressed key (Q7). A single
                # cheap key (Q5) aggregates faster in one fused kernel pass.
                _keys_heavy9 = (group is not None and (len(group.expressions) >= 2
                                or any(not isinstance(g9, E.Column) for g9 in group.expressions)))
                if _bill9 is not None:
                    print('JOIN BILL: cascade ARBITER leftovers=%d parent_keep=%.4f keys_heavy=%s'
                          % (len(_cres9), _keep9, _keys_heavy9), flush=True)
                # ...or when the parent keep alone is tiny (< 1%): at that
                # scale even a light key aggregates for nothing (Q21: Saudi x
                # status-F x the lonely qualification).
                _keep_all9 = 1.0
                for p in plan9:
                    _keep_all9 *= max(0.0, 1.0 - float(p[1]))
                if not gnodes:
                    # SCALAR: aggregation at survivors is trivial -- any potent
                    # total keep (fact + parent, < 5%) pays for the cascade (Q19).
                    if _keep_all9 > 0.05:
                        raise _FastUnsupported
                elif not _pk9 or _keep9 > 0.05 or (not _keys_heavy9 and _keep9 > 0.01):
                    raise _FastUnsupported
            plan9.sort(key=lambda x: -x[0])
            # JACKSON'S RUNNING RULE: potency picks WHICH filter is next;
            # (next filter's cost) < (aggregating the current survivors)
            # decides WHETHER to filter at all. Leftover conjuncts apply
            # later AT SURVIVOR SCALE. Constants from measured passes:
            # seq scan ~2ns/row of N, road gather ~3ns/survivor,
            # aggregate ~7ns/survivor + 2ms fixed.
            # THE MARGINAL-BOUND GATE: expected survivors, EXACT from the
            # censuses (product of keep fractions), known BEFORE any pass
            # runs. A high-survivor cascade turns cheap sequential reads
            # into N-scale gathers (Q1: 98.6% survive -> +430ms regression),
            # so the cascade fires only when the prune pays.
            _exp9 = 1.0
            for _, pr9, *_r9 in plan9:
                _exp9 *= (1.0 - pr9)
            if _exp9 > 0.5:
                raise _FastUnsupported
            # THE HOME-TABLE LAW (Jackson): a predicate is judged AT ITS HOME
            # table, once per home row; verdicts flow DOWN the foreign-key
            # roads as keep-flags; the fact pays one gather per road. Homes
            # are causally independent, so their native keeps build IN
            # PARALLEL; only the (cheap, parent-scale) downhill is serial.
            seg2alias9 = {id(ctx['seg_of'][a9]): a9 for a9 in ctx['seg_of']}
            fact9 = ctx['fact']
            homes9 = {}                                   # alias -> [(kx, seg, pcol)]
            _resid9 = []
            _n_est9 = float(n)
            _spent9 = 0
            for pot9, _pr9, kx9, cs9, cp9, cc9p in plan9:
                a9 = seg2alias9.get(id(cs9))
                if a9 is None: raise _FastUnsupported
                if _spent9 >= 1:
                    _c_filt = (2e-9 * n) if cc9p is None else (3e-9 * _n_est9)
                    _c_agg = 7e-9 * _n_est9 + 0.002
                    if _c_filt >= _c_agg:
                        _resid9.append((kx9, cs9, cp9, cc9p))
                        continue
                homes9.setdefault(a9, []).append((kx9, cs9, cp9))
                _n_est9 *= (1.0 - _pr9)
                _spent9 += 1
            def _plane_serve9(cs9p, cp9p, kx9p):
                # enc-14 columns serve contiguous keep-bands from the PLANES:
                # kx over a sorted dict is a code band [lo, hi) -> day bounds
                # from the dict ends -> the lexicographic plane test. Any
                # non-contiguous keep (or non-14 dress) returns None and the
                # ordinary reads serve.
                c9p = cs9p.cols.get(cp9p)
                if c9p is None or c9p.get('code_enc') not in (14, 15, 16):
                    return None
                nz9 = np.flatnonzero(kx9p)
                if nz9.size == 0:
                    return np.zeros(int(cs9p.N), dtype=bool)
                lo9, hi9 = int(nz9[0]), int(nz9[-1]) + 1
                if hi9 - lo9 != nz9.size:
                    return None                       # holes: not a band
                td9p = np.asarray(cs9p._typed_dict(cp9p))
                V9p = int(c9p['V'])
                if hi9 > V9p: return None
                dlo9 = int(td9p[lo9])
                dhi9 = int(td9p[hi9]) if hi9 < V9p else int(td9p[V9p - 1]) + 1
                return cs9p.plane_test(cp9p, dlo9, dhi9)
            def _native9(items9):
                # Within a home, conjuncts CASCADE by potency: the first pays
                # one sequential pass; every later one reads codes ONLY at
                # the survivors (codes_at: touched frames decompress, the
                # rest never open). One byte straight down, then small.
                m9 = None
                for kx9, cs9, cp9 in items9:
                    pm9 = _plane_serve9(cs9, cp9, kx9)
                    if m9 is None:
                        if pm9 is not None:
                            m9 = pm9
                        else:
                            _cd9 = np.asarray(cs9.codes(cp9))
                            m9 = np.empty(_cd9.shape[0], dtype=np.bool_)
                            wdb_kernels.plut_u8(_cd9, np.ascontiguousarray(kx9, dtype=np.bool_), m9)
                    else:
                        r9i = np.flatnonzero(m9)
                        if r9i.size == 0: return m9
                        if pm9 is not None:
                            m9[r9i] = pm9[r9i]
                        else:
                            m9[r9i] = kx9[np.asarray(cs9.codes_at(cp9, r9i))]
                return m9
            from concurrent.futures import ThreadPoolExecutor as _TPE9
            with _TPE9(max_workers=max(1, len(homes9))) as _ex9:
                futs9 = {a9: _ex9.submit(_native9, its9) for a9, its9 in homes9.items()}
                keeps9 = {a9: f9.result() for a9, f9 in futs9.items()}
            if _cb9 is not None:
                _cb9.append(('parallel-homes(native keeps)', _tk9() - _ct9)); _ct9 = _tk9()
            eps9 = ctx.get('edge_ptrs') or {}
            depth9 = {fact9: 0}
            _ch9 = True
            while _ch9:
                _ch9 = False
                for pa9, (ca9, _p9) in eps9.items():
                    if ca9 in depth9 and pa9 not in depth9:
                        depth9[pa9] = depth9[ca9] + 1; _ch9 = True
            _done9 = set()
            while True:
                # FIXPOINT: a keep flowing down creates a keep on the next alias
                # (Q7: nation -> supplier -> fact; nation -> customer -> orders ->
                # fact); a snapshot loop dropped every keep past the first hop.
                _cand9 = [a for a in keeps9 if a != fact9 and a not in _done9 and keeps9[a] is not None]
                if not _cand9: break
                pa9 = max(_cand9, key=lambda a: depth9.get(a, 0))
                _done9.add(pa9)
                if pa9 not in eps9: raise _FastUnsupported
                ca9, p9 = eps9[pa9]
                if ca9 == fact9 and keeps9.get(fact9) is not None and depth9.get(pa9) == 1:
                    continue        # RUNNING RULE: a depth-1 parent keep applies AT THE
                                    # FACT KEEP'S SURVIVORS below (15M rows, 5ms), never
                                    # flowed to fact scale (60M gather + AND, 120ms)
                flow9 = keeps9[pa9][np.asarray(p9)]       # verdict rides the road down
                keeps9[ca9] = flow9 if keeps9.get(ca9) is None else (keeps9.get(ca9) & flow9)                     if ca9 in keeps9 else flow9
                if ca9 not in homes9 and ca9 != fact9:
                    homes9[ca9] = []                      # transit alias now carries a keep
            fm9 = keeps9.get(fact9)
            import wdb_engine as _WEc
            rows9 = _WEc.Segment.mask_rows(fm9) if fm9 is not None else None
            for a9, (ca9, p9) in eps9.items():
                pass
            # any keep left on a fact-adjacent alias applies through its road
            for pa9, (ca9, p9) in eps9.items():
                if ca9 == fact9 and pa9 in keeps9 and keeps9[pa9] is not None                         and depth9.get(pa9) == 1:
                    k9 = keeps9[pa9]
                    pt9 = np.asarray(p9)
                    if rows9 is None:
                        _kr9 = np.empty(pt9.shape[0], dtype=np.bool_)
                        wdb_kernels.plut_u8(pt9, np.ascontiguousarray(k9, dtype=np.bool_), _kr9)
                        rows9 = _WEc.Segment.mask_rows(_kr9)
                    else:
                        _kk9 = np.empty(rows9.shape[0], dtype=np.bool_)
                        wdb_kernels.pkeep_via_ptr(rows9, pt9, np.ascontiguousarray(k9, dtype=np.bool_), _kk9)
                        rows9 = rows9[_WEc.Segment.mask_rows(_kk9)]
            if rows9 is None: raise _FastUnsupported
            if _cb9 is not None:
                _cb9.append(('roads+fnz', _tk9() - _ct9)); _ct9 = _tk9()
            for kx9, cs9, cp9, cc9p in _resid9:      # leftovers at survivor scale
                if rows9.size == 0: break
                if cc9p is None:
                    rows9 = rows9[kx9[np.asarray(cs9.codes_at(cp9, rows9))]]
                else:
                    pc9r = np.asarray(cs9.codes(cp9))
                    rows9 = rows9[kx9[pc9r[np.asarray(cc9p)[rows9]]]]
            _where_spent = not _cres9        # leftovers: the pred re-applies the full
                                             # WHERE at survivor scale (cheap, exact)
            _cres_left9 = list(_cres9)
            if _cb9 is not None:
                _cb9.append(('residual@survivors', _tk9() - _ct9))
                print('JOIN BILL: CASCADE-SUB ' + ' | '.join('%s=%.0fms' % (n9, v9 * 1000) for n9, v9 in _cb9), flush=True)
            if _bill9 is not None:
                _bill9.append(('cascade %d->%d' % (n, rows9.size), _tk9() - _fpa_t0))
            n = int(rows9.size)
            _stage0 = _tk9()
            if n == 0 and gnodes:            # GROUP BY over zero survivors: no groups.
                _bump_fast()                 # A SCALAR still emits exactly one row
                return [], [wdb_sql._alias(p) for p in proj]   # (COUNT=0, SUM=NULL) below.
        except Exception as _cex9:
            if os.environ.get('WDB_CASCADE_DEBUG'):
                import traceback as _tbc9
                print('CASCADE-DECLINED:', repr(_cex9)[:100], flush=True); _tbc9.print_exc()
            rows9 = None; _where_spent = False
    _plane_mask9 = None
    _resid_c9X = None
    _cres_left9 = locals().get('_cres_left9', [])
    # THE HOISTED PARTITION (Jackson's structural ruling): serves and the
    # survivor handoff run BEFORE anything row-aligned exists, so keys,
    # slots and roads are BORN at survivor scale -- no retrofits.
    if where is not None and not _where_spent and rows9 is None:
        try:
            def _flatH(nH):
                if isinstance(nH, E.Paren): return _flatH(nH.this)
                if isinstance(nH, E.And):
                    return _flatH(nH.this) + _flatH(nH.expression)
                return [nH]
            conj9 = _flatH(where.this)
            # THE PLANE-TEST SERVE (the field-plane dress's primary consumer):
            # date-range conjuncts on enc-14 fact columns become plane masks
            # -- three u8 compares per row, no reconstruction -- ANDed into
            # the kernel's mask; only the residual conjuncts compile to pred.
            _pp_t9 = _tk9()
            _resid_c9 = []
            _pl_iv9 = {}
            _pl_req9 = []
            for cn9 in conj9:
                served9 = None
                try:
                    cols9 = list(cn9.find_all(E.Column))
                    if (len(cols9) == 2 and isinstance(cn9, (E.GT, E.GTE, E.LT, E.LTE, E.EQ, E.NEQ))
                            and isinstance(cn9.this, E.Column) and isinstance(cn9.expression, E.Column)):
                        sga9, cpa9, ptra9 = resolve(cn9.this)
                        sgb9, cpb9, ptrb9 = resolve(cn9.expression)
                        ca9x, cb9x = sga9.cols.get(cpa9), sgb9.cols.get(cpb9)
                        if (sga9 is sgb9 and ptra9 is None and ptrb9 is None
                                and ca9x is not None and cb9x is not None
                                and {ca9x.get('code_enc'), cb9x.get('code_enc')} == {15, 16}
                                and (ca9x.get('e16_partner') in (cpb9, None))
                                and (cb9x.get('e16_partner') in (cpa9, None))):
                            # THE PAIR BIT ANSWERS IN THE PRED PARTITION too
                            # (the clock's operator table); banked as a
                            # request, executed concurrently below.
                            t9y = type(cn9).__name__
                            if ca9x.get('code_enc') != 15:   # left is partner: flip
                                t9y = {'GT': 'LT', 'LT': 'GT', 'GTE': 'LTE',
                                       'LTE': 'GTE', 'EQ': 'EQ', 'NEQ': 'NEQ'}[t9y]
                            def _pb_run9(sg9=sga9, cp9r=cpa9, t9z=t9y):
                                bit9x, dl9x = sg9.pair_bits(cp9r)
                                if t9z == 'LTE':   return bit9x
                                if t9z == 'GT':    return ~bit9x
                                if t9z == 'LT':    return bit9x & (dl9x > 0)
                                if t9z == 'GTE':   return ~(bit9x & (dl9x > 0))
                                if t9z == 'EQ':    return (dl9x == 0)
                                return (dl9x > 0)
                            _pl_req9.append(_pb_run9)
                            served9 = True
                    if served9 is None and len(cols9) == 1 and isinstance(cn9, (E.GT, E.GTE, E.LT, E.LTE, E.EQ, E.Between)):
                        cseg9, cp9x, cptr9x = resolve(cols9[0])
                        c9x = cseg9.cols.get(cp9x)
                        if cptr9x is None and c9x is not None and c9x.get('code_enc') in (14, 15, 16):
                            td9x = np.asarray(cseg9._typed_dict(cp9x))
                            dmin9x = int(td9x[0]); dmax9x = int(td9x[-1])
                            kind9x = 'f' if c9x['dt'] == 2 else 'i'
                            if isinstance(cn9, E.Between):
                                lo9x = int(wdb_sql._lit_for_col(cseg9, cp9x, cn9.args['low'], kind9x))
                                hi9x = int(wdb_sql._lit_for_col(cseg9, cp9x, cn9.args['high'], kind9x)) + 1
                            else:
                                left9x = isinstance(cn9.this, E.Column)
                                lit9x = cn9.args.get('expression') if left9x else cn9.this
                                v9x = int(wdb_sql._lit_for_col(cseg9, cp9x, lit9x, kind9x))
                                t9x = type(cn9) if left9x else {E.GT: E.LT, E.LT: E.GT,
                                                                E.GTE: E.LTE, E.LTE: E.GTE,
                                                                E.EQ: E.EQ}[type(cn9)]
                                if t9x is E.GTE: lo9x, hi9x = v9x, dmax9x + 1
                                elif t9x is E.GT: lo9x, hi9x = v9x + 1, dmax9x + 1
                                elif t9x is E.LT: lo9x, hi9x = dmin9x, v9x
                                elif t9x is E.LTE: lo9x, hi9x = dmin9x, v9x + 1
                                else: lo9x, hi9x = v9x, v9x + 1
                            if lo9x < dmin9x: lo9x = dmin9x
                            if hi9x > dmax9x + 1: hi9x = dmax9x + 1
                            k9iv = (id(cseg9), cp9x)
                            if k9iv in _pl_iv9:
                                s9o, l9o, h9o = _pl_iv9[k9iv]
                                _pl_iv9[k9iv] = (s9o, max(l9o, lo9x), min(h9o, hi9x))
                            else:
                                _pl_iv9[k9iv] = (cseg9, lo9x, hi9x)
                            served9 = True                 # interval banked; tested fused below
                except _FastUnsupported:
                    served9 = None
                except Exception:
                    served9 = None
                if served9 is None:
                    _resid_c9.append(cn9)
            for (_sg9, _cp9), (cseg9f, lo9f, hi9f) in list(_pl_iv9.items()):
                if lo9f >= hi9f:
                    _pl_req9.append(lambda n9=int(cseg9f.N): np.zeros(n9, dtype=bool))
                else:
                    _pl_req9.append(lambda c9z=cseg9f, p9z=_cp9, a9z=lo9f, b9z=hi9f:
                                    c9z.plane_test(p9z, a9z, b9z))
            if _pl_req9:
                # ADMISSION OF LOADS (Jackson's law applied to reads): residual
                # conjuncts on dressed fact columns will need their streams at
                # survivor scale -- decompress them NOW, inside the wall's shadow,
                # where they cost ~nothing. Returns None (not a mask).
                for cnr9 in ([] if os.environ.get('WDB_NO_SHADOW_WARMS') else list(_resid_c9) + list(proj)):
                    for colr9 in cnr9.find_all(E.Column):
                        try:
                            sgr9, cpr9, ptrr9 = resolve(colr9)
                        except Exception:
                            continue
                        if ptrr9 is not None:            # parent column: warm its codes
                            if cpr9 not in sgr9._codes:
                                _pl_req9.append(lambda sg=sgr9, cp=cpr9: (sg.codes(cp), None)[1])
                            continue
                        cr9 = sgr9.cols.get(cpr9)
                        if cr9 is None or cpr9 in sgr9._codes:
                            continue
                        er9 = cr9.get('code_enc')
                        if er9 == 14:
                            def _warm14(sg=sgr9, cp=cpr9):
                                cache = getattr(sg, '_e14_pl', None)
                                if cache is None: cache = sg._e14_pl = {}
                                if cp not in cache: cache[cp] = sg._e14_planes(cp)
                                return None
                            _pl_req9.append(_warm14)
                        elif er9 in (15, 16):
                            pass                    # a clock band/bit in the shadow loads these
                # SHADOW SCHEDULING (Jackson's law): fixed-cost serves cannot
                # be reduced, so they run FIRST and TOGETHER -- the cheap
                # serves finish inside the heaviest serve's shadow.
                if len(_pl_req9) > 1:
                    from concurrent.futures import ThreadPoolExecutor as _TPq
                    with _TPq(max_workers=min(len(_pl_req9), 16)) as exq9:
                        for m9f in exq9.map(lambda f9: f9(), _pl_req9):
                            if m9f is None: continue            # a warm-up, not a verdict
                            (_plane_mask9 is None and (m9f is not None)) and None; _plane_mask9 = m9f if _plane_mask9 is None else (wdb_kernels.pand(_plane_mask9, m9f), _plane_mask9)[1]
                else:
                    m9f = _pl_req9[0]()
                    if m9f is not None:
                        (_plane_mask9 is None and (m9f is not None)) and None; _plane_mask9 = m9f if _plane_mask9 is None else (wdb_kernels.pand(_plane_mask9, m9f), _plane_mask9)[1]
            if (_plane_mask9 is not None and group is not None and rows9 is None
                    and not _where_spent):
                import wdb_engine as _WE9
                _sv9 = _WE9.Segment.mask_rows(_plane_mask9)
                # RUNNING RULE on the handoff: a DRESSED group key (enc 14/15/16)
                # costs a full 60M reconstruct at fact scale (Q7 paid 1.9s) but a
                # cheap codes_at at survivors -- so any keep under ~60% hands off.
                _dressed_key9 = False
                for _g9 in (group.expressions if group is not None else []):
                    for _gc9 in ([_g9] if isinstance(_g9, E.Column) else list(_g9.find_all(E.Column))):
                        try:
                            _gs9, _gp9, _gcp9 = resolve(_gc9)
                            if _gcp9 is None and _gs9.cols.get(_gp9, {}).get('code_enc') in (14, 15, 16):
                                _dressed_key9 = True
                        except Exception:
                            pass
                if _sv9.size * 4 < _plane_mask9.size or (_dressed_key9 and _sv9.size * 5 < _plane_mask9.size * 3):
                    # SURVIVOR HANDOFF: the mask is selective enough that the
                    # per-row leftovers (slots, keys, pred, kernel) all run at
                    # survivor scale through the existing rows9 plumbing.
                    rows9 = _sv9
                    n = int(rows9.size)
                    _plane_mask9 = None
            if _bill9 is not None:
                print('JOIN BILL: PARTITION pre-pass=%.1fms' % ((_tk9() - _pp_t9) * 1000), flush=True)
            _resid_c9X = _resid_c9
        except Exception as _hx9:
            if os.environ.get('WDB_HOIST_DEBUG'):
                import traceback as _tb9; _tb9.print_exc()
            _plane_mask9 = None; _resid_c9X = None; rows9 = None
    def _rw9(a):
        return a if (rows9 is None or a is None) else np.asarray(a)[rows9]
    # ---- THE KEY-DEPENDENT TOP-K DOOR (Q10's shape, 2026-09-01) --------------
    # GROUP BY a parent's unique key plus that parent's attributes (and its
    # own parents' attributes), ORDER BY one aggregate, LIMIT k: group by the
    # PARENT ROW alone (one int key), select the top k, and decode every
    # attribute for k rows only. Duck decodes them for every group.
    def _topk_attr_door():
        order9 = tree.args.get('order'); limit9 = tree.args.get('limit')
        if group is None or order9 is None or limit9 is None: return None
        if _plane_mask9 is not None or (_resid_c9X and rows9 is None): return None
        try:
            k9 = int(limit9.expression.this)
        except Exception:
            return None
        if k9 <= 0 or k9 > 10000: return None
        ords9 = list(order9.expressions)
        if not ords9 or any(not isinstance(o9.this, E.Column) for o9 in ords9): return None
        okey9 = ords9[0].this.name; desc9 = bool(ords9[0].args.get('desc'))
        attr_order9 = len(ords9) > 1 or True   # resolved after plan9: agg alias or attributes
        alias2t9 = ctx['alias2t']; edge_ptrs9 = ctx.get('edge_ptrs') or {}
        # group columns: which alias is the key holder?
        gcols9 = []
        for g in gnodes:
            if not isinstance(g, E.Column): return None
            sg, pc, cp = resolve(g)
            gcols9.append((g, sg, pc, cp))
        def _alias_of(g):
            return g.table or next((al for al in alias2t9 if g.name in cols_of[al]), None)
        # THE KEY IS THE ALIAS EVERY OTHER GROUP COLUMN ROUTES TO (Q18: both
        # c_name and o_orderkey are unique keys; only orders can host customer's
        # attributes -- customer is orders' PARENT, not the reverse).
        cands9 = []
        for g, sg, pc, cp in gcols9:
            if cp is None: continue
            a9 = _alias_of(g)
            if not a9: return None
            try:
                if _key_is_unique(db, alias2t9.get(a9, a9), pc):
                    cands9.append((a9, cp, sg))
            except Exception:
                continue
        if not cands9: return None
        def _route_for(A9c):
            def _r(g):
                al = _alias_of(g)
                if al == A9c: return ('A', None)
                ep = edge_ptrs9.get(al)
                if ep is not None and ep[0] == A9c: return ('B', ep[1])
                return None
            rs = []
            for g, sg, pc, cp in gcols9:
                r9 = _r(g)
                if r9 is None: return None
                rs.append(r9)
            return rs
        A9 = None; routes9 = None
        for a9c, cpc, sgc in cands9:
            rs9 = _route_for(a9c)
            if rs9 is not None:
                A9 = a9c; keyptr9 = cpc; kseg9 = sgc; routes9 = rs9; break
        if A9 is None: return None
        def _attr_route(g, sg, cp):
            al = _alias_of(g)
            if al == A9: return ('A', None)
            ep = edge_ptrs9.get(al)
            if ep is not None and ep[0] == A9: return ('B', ep[1])
            return None
        # projections: group columns or SUM/COUNT/AVG aggregates
        aggs9 = {}
        def _ev9(nd, rows):
            if isinstance(nd, E.Paren): return _ev9(nd.this, rows)
            if isinstance(nd, E.Literal):
                if nd.is_string: raise _FastUnsupported
                return float(nd.this)
            if isinstance(nd, E.Column):
                sg, pc, cp = resolve(nd)
                raw9 = wdb_sql.raw_dict_col(sg, pc, want_codes=False)
                if raw9 is None: raise _FastUnsupported
                rr = rows if cp is None else np.asarray(cp)[rows]
                return raw9[0][np.asarray(sg.codes_at(pc, rr))].astype(np.float64)
            if isinstance(nd, E.Neg): return -_ev9(nd.this, rows)
            if isinstance(nd, E.Mul): return _ev9(nd.this, rows) * _ev9(nd.expression, rows)
            if isinstance(nd, E.Add): return _ev9(nd.this, rows) + _ev9(nd.expression, rows)
            if isinstance(nd, E.Sub): return _ev9(nd.this, rows) - _ev9(nd.expression, rows)
            if isinstance(nd, E.Div): return _ev9(nd.this, rows) / _ev9(nd.expression, rows)
            raise _FastUnsupported
        plan9 = []
        gnames9 = {(_alias_of(g), g.name) for g, _s, _p, _c in gcols9}
        for i, p in enumerate(proj):
            inner = p.this if isinstance(p, E.Alias) else p
            al9 = p.alias if isinstance(p, E.Alias) else (inner.name if isinstance(inner, E.Column) else None)
            if isinstance(inner, E.Column):
                if (_alias_of(inner), inner.name) not in gnames9: return None
                plan9.append(('col', inner, al9))
            elif isinstance(inner, (E.Sum, E.Count, E.Avg)):
                plan9.append(('agg', inner, al9))
            else:
                return None
        _pcols9 = {n9.name for kind, n9, _a in plan9 if kind == 'col'}
        if not (any(kind == 'agg' and al9 == okey9 for kind, _n, al9 in plan9)
                or all(o9.this.name in _pcols9 for o9 in ords9)):    # ORDER BY attributes (Q18)
            return None
        # survivors -- WITH the cascade's leftovers re-applied AT ROWS (the door
        # once consumed rows9 raw and shipped Q21 wrong, 2026-09-03)
        if rows9 is not None:
            rows = rows9
            for _lc9 in (_cres_left9 or []):
                _mk9 = None
                if (type(_lc9) in (E.GT, E.LT) and isinstance(_lc9.this, E.Column)
                        and isinstance(_lc9.expression, E.Column)):
                    _sA, _pA, _cA = resolve(_lc9.this); _sB, _pB, _cB = resolve(_lc9.expression)
                    if _sA is _sB and _cA is None and _cB is None:
                        _big, _small = (_pA, _pB) if isinstance(_lc9, E.GT) else (_pB, _pA)
                        _cbig = _sA.cols.get(_big, {}); _csm = _sA.cols.get(_small, {})
                        if _cbig.get('code_enc') == 16 and _cbig.get('e16_partner') == _small:
                            _b9, _d9 = _sA.pair_bits(_big)
                            _mk9 = np.asarray(_b9)[rows] & (np.asarray(_d9)[rows] > 0)
                        elif _csm.get('code_enc') == 16 and _csm.get('e16_partner') == _big:
                            _b9, _d9 = _sA.pair_bits(_small)
                            _mk9 = (~np.asarray(_b9)[rows]) & (np.asarray(_d9)[rows] > 0)
                else:
                    _lcols = list(_lc9.find_all(E.Column))
                    if len(_lcols) == 1:
                        _sL, _pL, _cL = resolve(_lcols[0])
                        _cLd = _sL.cols.get(_pL, {})
                        if _cLd.get('mode') in (0, 1, 2):
                            _tdL = np.asarray(_sL._typed_dict(_pL))
                            _kpL = _dict_keep(_lc9, _sL, _pL, _tdL) if _tdL.dtype.kind in 'iuf' else None
                            if _kpL is not None:
                                _rr = rows if _cL is None else np.asarray(_cL)[rows]
                                _cdL = np.asarray(_sL.codes_at(_pL, _rr))
                                _kxL = np.zeros(int(_cLd['V']) + 1, bool); _kxL[:len(_kpL)] = _kpL
                                _mk9 = _kxL[_cdL]
                if _mk9 is None:
                    return None                    # a leftover the door can't judge
                rows = rows[_mk9]
        else:
            m9 = get_mask()
            rows = np.flatnonzero(m9) if m9 is not None else np.arange(n, dtype=np.int64)
        key = np.asarray(keyptr9)[rows]
        NK = int(kseg9.N)
        cnt = np.bincount(key, minlength=NK)
        vals9 = {}
        try:
            for kind, nd, al9 in plan9:
                if kind != 'agg': continue
                if isinstance(nd, E.Count):
                    vals9[al9] = cnt.astype(np.float64)
                else:
                    w9 = _ev9(nd.this, rows)
                    s9 = np.bincount(key, weights=w9, minlength=NK)
                    vals9[al9] = (s9 / np.maximum(cnt, 1)) if isinstance(nd, E.Avg) else s9
        except _FastUnsupported:
            return None
        present = np.flatnonzero(cnt > 0)
        kk = min(k9, present.size)
        if kk == 0:
            _bump_fast(); return [], [wdb_sql._alias(p) for p in proj]
        if len(ords9) == 1 and okey9 in vals9:
            score = vals9[okey9][present]
            sel = np.argpartition(-score if desc9 else score, kk - 1)[:kk]
            sel = sel[np.argsort(-score[sel] if desc9 else score[sel], kind='stable')]
        else:
            # ORDER BY ATTRIBUTES (Jackson's Q18: totalprice DESC, date):
            # decode the order columns for the PRESENT groups (already tiny
            # after the mask) and lexsort; the aggregate needed no ordering.
            if present.size > 200000: return None
            karr9 = []
            for o9 in ords9:
                if isinstance(o9.this, E.Column) and o9.this.name in vals9:
                    a9v = vals9[o9.this.name][present]
                else:
                    sg9o, pc9o, cp9o = resolve(o9.this)
                    rt9o = _attr_route(o9.this, sg9o, cp9o)
                    rr9o = present if rt9o[0] == 'A' else np.asarray(rt9o[1])[present]
                    v9o = sg9o.values_at_rows(pc9o, rr9o)
                    a9v = np.asarray(v9o)
                d9o = bool(o9.args.get('desc'))
                if a9v.dtype.kind in 'ifu':
                    karr9.append(-a9v if d9o else a9v)
                else:
                    _, inv9o = np.unique(a9v, return_inverse=True)
                    karr9.append(-inv9o if d9o else inv9o)
            sel = np.lexsort(tuple(reversed(karr9)))[:kk]
        top = present[sel]
        # decode attributes at k rows
        cols_out = []
        for kind, nd, al9 in plan9:
            if kind == 'agg':
                v = vals9[al9][top]
                cols_out.append([int(x) for x in v] if isinstance(nd, E.Count) else [float(x) for x in v])
            else:
                sg, pc, cp = resolve(nd)
                rt = _attr_route(nd, sg, cp)
                rr = top if rt[0] == 'A' else np.asarray(rt[1])[top]
                cols_out.append(list(sg.values_at_rows(pc, rr)))
        rows_out = [tuple(c[i] for c in cols_out) for i in range(kk)]
        if _bill9 is not None:
            _bill9.append(('topk-attr door (%d survivors, %d groups)' % (rows.size, present.size), _tk9() - _fpa_t0))
            print('JOIN BILL: ' + ' | '.join('%s=%.0fms' % (nm9, v9 * 1000) for nm9, v9 in _bill9), flush=True)
        _bump_fast()
        return rows_out, [wdb_sql._alias(p) for p in proj]
    try:
        _door9 = _topk_attr_door()
    except _FastUnsupported:
        _door9 = None
    if _door9 is not None:
        return _door9
    gkeys = []                                           # one per GROUP BY column
    for g in gnodes:
        if not isinstance(g, E.Column):
            # EXPRESSION GROUP KEY RIDES THE DICT: transform the V dictionary
            # values through the expression, collapse equal results, remap the
            # codes -- O(V), never O(N) (Q7's l_shipdate/365 folds ~2500 days
            # into 7 years at dict scale).
            cols9g = list(g.find_all(E.Column))
            if len(cols9g) != 1: raise _FastUnsupported            # declines RAISE (a None from fpa is 'no answer')
            gseg, gpcol, gcptr = resolve(cols9g[0])
            rawx = wdb_sql.raw_dict_col(gseg, gpcol, want_codes=False)
            if rawx is None: raise _FastUnsupported
            def _dx9(nd, bv):
                if isinstance(nd, E.Paren): return _dx9(nd.this, bv)
                if isinstance(nd, E.Column): return bv
                if isinstance(nd, E.Literal):
                    if nd.is_string: raise _FastUnsupported
                    return float(nd.this)
                if isinstance(nd, E.Neg): return -_dx9(nd.this, bv)
                if isinstance(nd, E.Mul): return _dx9(nd.this, bv) * _dx9(nd.expression, bv)
                if isinstance(nd, E.Div): return _dx9(nd.this, bv) / _dx9(nd.expression, bv)
                if isinstance(nd, E.Add): return _dx9(nd.this, bv) + _dx9(nd.expression, bv)
                if isinstance(nd, E.Sub): return _dx9(nd.this, bv) - _dx9(nd.expression, bv)
                if isinstance(nd, E.Mod): return np.mod(_dx9(nd.this, bv), _dx9(nd.expression, bv))
                if isinstance(nd, E.IntDiv): return np.floor_divide(_dx9(nd.this, bv), _dx9(nd.expression, bv))
                raise _FastUnsupported
            dv9 = _dx9(g, np.asarray(rawx[0], dtype=np.float64))
            if gseg.cols[gpcol].get('dt') == 0 and (dv9 == np.floor(dv9)).all():
                dv9 = dv9.astype(np.int64)             # integer keys stay integers (labels emit as ints)
            u9x, inv9x = np.unique(dv9, return_inverse=True)
            inv9x = np.ascontiguousarray(inv9x, dtype=np.int64)
            if gcptr is None and rows9 is not None:
                full = inv9x[np.asarray(gseg.codes_at(gpcol, rows9))]
            else:
                full = inv9x[np.asarray(gseg.codes(gpcol))]
                if full.size == 0: return [], [wdb_sql._alias(p) for p in proj]
                if gcptr is None: full = _rw9(full)
                else: gcptr = _rw9(gcptr)
            gkeys.append({'seg': gseg, 'pcol': gpcol, 'cptr': gcptr, 'full': full,
                          'K': int(u9x.size), 'labels': [wdb_sql._pyval(x) for x in u9x.tolist()],
                          'expr': True})
            continue
        gseg, gpcol, gcptr = resolve(g)
        if gseg.cols[gpcol]['mode'] == 4:
            # Affine-coded (mode 4): dense gids + labels, memoised on the immutable segment.
            full, _K, _labels = _mode4_group(gseg, gpcol)
            if full.size == 0: return [], [wdb_sql._alias(p) for p in proj]
            if gcptr is None: full = _rw9(full)
            else: gcptr = _rw9(gcptr)
            gkeys.append({'seg': gseg, 'pcol': gpcol, 'cptr': gcptr, 'full': full,
                          'K': _K, 'labels': _labels})
            continue
        K9full = int(gseg.cols[gpcol]['V'])              # dict-wide, no full read needed
        if gcptr is None and rows9 is not None:
            full = np.asarray(gseg.codes_at(gpcol, rows9))   # survivor-scale fetch
        else:
            full = gseg.codes(gpcol)
            if full.size == 0: return [], [wdb_sql._alias(p) for p in proj]
            if gcptr is None: full = _rw9(full)
            else: gcptr = _rw9(gcptr)
        if full.size == 0 and n > 0: return [], [wdb_sql._alias(p) for p in proj]
        gkeys.append({'seg': gseg, 'pcol': gpcol, 'cptr': gcptr, 'full': full, 'K': K9full})
    gid_to_comp = None                                    # set when a high-card composite is hash-factorised
    _gid_words9 = None                                    # set by THE HASHED COMPOSITE
    if len(gkeys) == 0:
        K = 1; group_op = None; group_keys = []
    elif len(gkeys) == 1:                                 # single key: keep the gather-fused operand
        k0 = gkeys[0]; K = k0['K']
        group_op = ('g', k0['full'], k0['cptr']) if k0['cptr'] is not None else ('d', k0['full'])
        group_keys = [(k0['full'], K, k0['cptr'])]
    else:                                                 # multi-key: mixed-radix composite
        K = 1
        for k in gkeys: K *= k['K']
        if RT.dense_multigroup_fits(K):                    # dense: codegen composes the code INLINE (no array)
            group_op = None
            group_keys = [(k['full'], k['K'], k['cptr']) for k in gkeys]
        else:                                              # high-card: hash-factorise the composite to dense ids
            prod = 1
            for k in gkeys: prod *= k['K']
            _gid_words9 = None
            if prod > (1 << 62):
                # THE HASHED COMPOSITE: pack the key codes by bit width into two
                # 64-bit words and factorise with an open-addressing kernel --
                # group ids at survivor scale, each group's key words kept for
                # emission (H2O q10: 50M rows, 50M groups; the pandas tail was a
                # 37GB hang, the mixed-radix code overflows int64).
                widths9 = [max(1, int(k['K'] - 1).bit_length()) for k in gkeys]
                if sum(widths9) > 126: raise _FastUnsupported
                shifts9 = np.zeros(len(gkeys), np.int64); words9 = np.zeros(len(gkeys), np.int64)
                _bit = 0; _wd = 0
                for j9, wdt in enumerate(widths9):
                    if _bit + wdt > 63:
                        _wd += 1; _bit = 0
                    shifts9[j9] = _bit; words9[j9] = _wd; _bit += wdt
                cm9 = np.empty((len(gkeys), n), np.int64)
                for j9, k in enumerate(gkeys):
                    cm9[j9] = (k['full'][k['cptr']] if k['cptr'] is not None else k['full']).astype(np.int64, copy=False)
                w0 = np.empty(n, np.int64); w1 = np.empty(n, np.int64)
                wdb_kernels.pack2(cm9, shifts9, words9, w0, w1)
                del cm9
                tsz = 1 << max(4, int(2 * n - 1).bit_length())
                tk0 = np.empty(tsz, np.int64); tk1 = np.empty(tsz, np.int64); tg = np.full(tsz, -1, np.int64)
                gids = np.empty(n, np.int64)
                ng9 = int(wdb_kernels.hcomposite2(w0, w1, gids, tk0, tk1, tg))
                g0 = np.empty(ng9, np.int64); g1 = np.empty(ng9, np.int64)
                wdb_kernels.hcomposite_rep(tg, tk0, tk1, g0, g1)
                del tk0, tk1, tg, w0, w1
                _gid_words9 = (g0, g1, shifts9, words9, widths9)
                gid_to_comp = None
                gids = np.ascontiguousarray(gids)
                K = ng9; group_op = ('d', gids); group_keys = [(gids, K, None)]
                _mono_gids = False
                _hashed_done9 = True
            else:
                _hashed_done9 = False
            # THE LEADING-RUN COURT (decode-spec law: group keys are IDENTITY
            # class -- never gather 60M co-key codes to label 1.1M groups).
            # Gate, proven exactly and gather-free: leading key fact-direct and
            # monotone; every co-key's POINTER constant within leading runs
            # (a sequential O(N) compare -- if a pointer never changes inside
            # a run, its gathered codes can't either). Then boundaries come
            # from the leading key alone and the composite is computed AT THE
            # STARTS ONLY, mirroring the full radix exactly.
            _lead_ok = (not _hashed_done9 and gkeys[0]['cptr'] is None
                        and all(k['cptr'] is not None for k in gkeys[1:]))
            if _lead_ok:
                f0 = gkeys[0]['full'].astype(np.int64, copy=False)
                d0 = np.diff(f0)
                _lead_ok = bool(n and (d0 >= 0).all())
                if _lead_ok:
                    nb9 = d0 > 0
                    for k in gkeys[1:]:
                        cp9 = np.asarray(k['cptr'])
                        if not bool(((cp9[1:] == cp9[:-1]) | nb9).all()):
                            _lead_ok = False
                            break
            if _hashed_done9:
                pass
            elif _lead_ok:
                _mono_gids = True
                starts = np.concatenate([[0], np.flatnonzero(nb9) + 1])
                gids = np.zeros(n, np.int64)
                gids[starts[1:]] = 1
                gids = np.cumsum(gids)
                comp_s = np.zeros(starts.size, dtype=np.int64)
                for k in gkeys:                        # same radix order as the full build
                    ck9 = (k['full'][np.asarray(k['cptr'])[starts]]
                           if k['cptr'] is not None else k['full'][starts])
                    comp_s = comp_s * k['K'] + ck9.astype(np.int64, copy=False)
                gid_to_comp = comp_s
            else:
                comp = np.zeros(n, dtype=np.int64)
                for k in gkeys:
                    codes = k['full'][k['cptr']] if k['cptr'] is not None else k['full']
                    comp = comp * k['K'] + codes.astype(np.int64, copy=False)
                d9 = np.diff(comp)
                _mono_gids = bool(n and (d9 >= 0).all())
                if _mono_gids:
                    # THE SORTED-RUN COURT: a monotone composite factorises by
                    # boundary -- no hash, one diff.
                    starts = np.concatenate([[0], np.flatnonzero(d9 > 0) + 1])
                    gids = np.zeros(n, np.int64)
                    gids[starts[1:]] = 1
                    gids = np.cumsum(gids)
                    gid_to_comp = comp[starts]
                else:
                    gids, gid_to_comp = pd.factorize(comp, sort=False)   # hash-factorise -> only groups present
            if not _hashed_done9:
                gids = np.ascontiguousarray(gids.astype(np.int64))
                K = len(gid_to_comp); group_op = ('d', gids); group_keys = [(gids, K, None)]

    # The codegen value path composes the composite group code INLINE (no comp array). The non-fused
    # plain / numpy / counts-only paths get a materialised group operand on demand via _group_op().
    _go = {}
    def _group_op():
        if group_op is not None or not gkeys: return group_op
        if 'op' not in _go:
            comp = np.zeros(n, dtype=np.int64)
            for k in gkeys:
                codes = k['full'][k['cptr']] if k['cptr'] is not None else k['full']
                comp = comp * k['K'] + codes.astype(np.int64, copy=False)
            _go['op'] = ('d', np.ascontiguousarray(comp))
        return _go['op']

    # ---- per-projection results ----
    # COUNT/SUM/AVG become operand specs computed in one pass (threaded + per-chunk gather above a row
    # threshold); MIN/MAX run on the serial kernel; bare key columns map straight to the group value.
    _CLS = {E.Sum: 'SUM', E.Avg: 'AVG', E.Min: 'MIN', E.Max: 'MAX'}
    col_results = {}
    int9 = {}
    def _int_expr9(nd):
        """True iff the aggregate ARGUMENT is integer-typed end to end --
        duck emits BIGINT for such SUM/MIN/MAX and the board's referee
        string-compares non-floats, so wave owes true ints (Q12's
        62071.0 vs 62071 was flagged WRONG on values that matched)."""
        if isinstance(nd, E.Paren): return _int_expr9(nd.this)
        if isinstance(nd, E.Literal):
            return (not nd.is_string) and ('.' not in str(nd.this))
        if isinstance(nd, E.Column):
            sg9i, pc9i, _c9i = resolve(nd)
            return sg9i.cols[pc9i].get('dt') == 0
        if isinstance(nd, E.Case):
            br9 = [b.args['true'] for b in nd.args.get('ifs', [])]
            d9i = nd.args.get('default')
            if d9i is not None: br9.append(d9i)
            return bool(br9) and all(_int_expr9(b) for b in br9)
        if type(nd) in (E.Add, E.Sub, E.Mul):
            return _int_expr9(nd.this) and _int_expr9(nd.expression)
        if isinstance(nd, E.Neg): return _int_expr9(nd.this)
        return False


    # ---- UNIFIED fused path -------------------------------------------------------------------------
    # Build every value aggregate as an expression over shared dict-column slots, then a single codegen
    # kernel computes the composite group (inline), decodes each slot once, and accumulates COUNT + every
    # expression's SUM (and MIN/MAX where needed) in ONE pass -- no comp array, no (n,V) value matrix.
    # Falls back wholesale to the per-spec machinery below if any operand is not a numeric dict column or
    # arithmetic over them (string / nullable / computed), or numba is unavailable.
    slots = {}; slot_list = []                       # (id(seg), pcol, id(cptr)) -> global slot index
    def build_fused(node):
        if isinstance(node, E.Cast) and any(k in node.to.sql().upper() for k in ('DATE', 'TIME', 'CHAR', 'TEXT', 'STRING', 'BOOL')):
            raise _FastUnsupported                  # a TYPE-CHANGING cast is not transparent
        if isinstance(node, (E.Paren, E.Cast)): return build_fused(node.this)
        if isinstance(node, E.Neg): return f"(-{build_fused(node.this)})"
        if isinstance(node, E.Column):
            bseg, bpcol, bcptr = resolve(node)
            _sv14 = (bcptr is None and rows9 is not None)   # survivor slot:
            raw = wdb_sql.raw_dict_col(bseg, bpcol, want_codes=not _sv14)   # never decode full
            if raw is None: raise _FastUnsupported
            bkey = (id(bseg), bpcol, id(bcptr) if bcptr is not None else None)
            if bkey not in slots:
                slots[bkey] = len(slot_list)
                _c9s = (np.asarray(bseg.codes_at(bpcol, rows9))
                        if _sv14 else
                        (_rw9(raw[1]) if bcptr is None else raw[1]))
                slot_list.append((np.ascontiguousarray(raw[0]), np.ascontiguousarray(_c9s),
                                  None if bcptr is None else np.ascontiguousarray(_rw9(bcptr))))
            return f"v{slots[bkey]}"
        if isinstance(node, E.Literal):
            if node.is_string: raise _FastUnsupported
            v = node.this
            return f"({float(v)})" if ('.' in v or 'e' in v.lower()) else f"({int(v)})"
        if type(node) in _ARITH_STR:
            return f"({build_fused(node.this)} {_ARITH_STR[type(node)]} {build_fused(node.expression)})"
        if isinstance(node, E.Case):
            # CASE in the FUSED KERNEL (Q12's home): the condition compiles to
            # a truthy sub-expression (string trees ride _code_lut as a 0/1
            # value slot; numeric compares inline), branches are ordinary
            # fused exprs, and the whole thing is a numba conditional.
            ifs9 = node.args.get('ifs', [])
            if node.this is not None or not ifs9:
                raise _FastUnsupported             # simple CASE x WHEN: not yet
            d9 = node.args.get('default')
            out9 = build_fused(d9) if d9 is not None else '(0)'
            for br9 in reversed(ifs9):
                cnd9 = _cond_fused(br9.this)
                tv9 = build_fused(br9.args['true'])
                out9 = f"(({tv9}) if ({cnd9}) else ({out9}))"
            return out9
        raise _FastUnsupported

    def _cond_fused(cond):
        """Compile a CASE condition to a fused truthy expression."""
        if isinstance(cond, E.Paren):
            return _cond_fused(cond.this)
        if isinstance(cond, E.And):
            return f"(({_cond_fused(cond.this)}) and ({_cond_fused(cond.expression)}))"
        if isinstance(cond, E.Or):
            return f"(({_cond_fused(cond.this)}) or ({_cond_fused(cond.expression)}))"
        if isinstance(cond, E.Not):
            return f"(not ({_cond_fused(cond.this)}))"
        cols9 = list(cond.find_all(E.Column))
        if cols9 and len({(c.table, c.name) for c in cols9}) == 1:
            cseg9, cp9, cptr9 = resolve(cols9[0])
            if cseg9.cols[cp9].get('dt') == 1:     # one string column: the LUT road
                def _ev9(nd, vb):
                    if isinstance(nd, E.Paren): return _ev9(nd.this, vb)
                    if isinstance(nd, E.Or): return _ev9(nd.this, vb) or _ev9(nd.expression, vb)
                    if isinstance(nd, E.And): return _ev9(nd.this, vb) and _ev9(nd.expression, vb)
                    if isinstance(nd, E.Not): return not _ev9(nd.this, vb)
                    if isinstance(nd, (E.EQ, E.NEQ)):
                        lit9 = nd.expression if isinstance(nd.this, E.Column) else nd.this
                        r9 = (vb == _lit_bytes(cseg9, cp9, lit9))
                        return r9 if isinstance(nd, E.EQ) else (not r9)
                    if isinstance(nd, E.In):
                        return vb in [_lit_bytes(cseg9, cp9, x) for x in nd.expressions]
                    if isinstance(nd, (E.Like, E.ILike)):
                        return _like_fn(cseg9, cp9, nd.expression, isinstance(nd, E.ILike))(vb)
                    raise _FastUnsupported
                vk9 = _code_lut(cseg9, cp9, cptr9, lambda vb: _ev9(cond, vb))
                return f"({vk9} != 0)"
        if type(cond) in _CMP_STR:                 # numeric compare inline
            return f"({build_fused(cond.this)} {_CMP_STR[type(cond)]} {build_fused(cond.expression)})"
        raise _FastUnsupported

    _CMP_STR = {E.GT: '>', E.LT: '<', E.GTE: '>=', E.LTE: '<=', E.EQ: '==', E.NEQ: '!='}
    def _is_lit(nd):
        if isinstance(nd, E.Literal): return True
        if isinstance(nd, (E.Neg, E.Cast, E.Paren)): return _is_lit(nd.this)
        return False
    def _str_slot(cseg, cpcol, cptr):                  # raw code slot for a string column (base=None)
        c = cseg.cols[cpcol]
        if c['dt'] != 1 or c['mode'] == 4: raise _FastUnsupported   # value-identity string dicts only
        try: codes = cseg.codes(cpcol)
        except Exception: raise _FastUnsupported
        nullcode = (c['V'] - 1) if c['has_null'] else None
        ckey = (id(cseg), cpcol, id(cptr) if cptr is not None else None)
        if ckey not in slots:
            slots[ckey] = len(slot_list)
            _c9t = (np.asarray(cseg.codes_at(cpcol, rows9))
                    if (cptr is None and rows9 is not None) else
                    (_rw9(codes) if cptr is None else codes))
            slot_list.append((None, np.ascontiguousarray(_c9t),
                              None if cptr is None else np.ascontiguousarray(_rw9(cptr))))
        return f"v{slots[ckey]}", nullcode      # NO code_of -- literals resolved via _code_of_literal
    def _code_lut(cseg, cpcol, cptr, fn, mark_null=False):
        # ARBITRARY single-column string predicate -> code-LUT: precompute keep[code]=fn(dict_value) over
        # the (small) dictionary, add it as a value slot whose base IS that bool table, predicate -> 'vk!=0'.
        # Reuses the value-slot kernel verbatim (base[codes[i]]); gated on dict cardinality (precompute cost).
        sc = _str_codes(cseg, cpcol)
        if sc is None: raise _FastUnsupported
        codes, code_of, nullcode = sc
        ncodes = max(max(code_of.values(), default=-1),
                     nullcode if nullcode is not None else -1) + 1
        if not RT.code_lut_fits(ncodes): raise _FastUnsupported
        keep = np.zeros(ncodes, dtype=np.int8)
        if mark_null:
            if nullcode is not None: keep[nullcode] = 1            # IS NULL
        else:
            for vb, cd in code_of.items():
                if fn(vb): keep[cd] = 1
        _c9k = (np.asarray(cseg.codes_at(cpcol, rows9))
                if (cptr is None and rows9 is not None) else
                (_rw9(codes) if cptr is None else codes))
        slot_list.append((keep, np.ascontiguousarray(_c9k),
                          None if cptr is None else np.ascontiguousarray(_rw9(cptr))))
        return f"v{len(slot_list) - 1}"
    def _like_fn(cseg, cpcol, pat_node, ci):
        pb = _lit_bytes(cseg, cpcol, pat_node)
        patt = pb.decode('utf-8', 'replace') if isinstance(pb, (bytes, bytearray)) else str(pb)
        rxsrc = '^' + re.escape(patt).replace('%', '.*').replace('_', '.') + '$'   # SQL LIKE -> regex
        rx = re.compile(rxsrc, re.DOTALL | (re.IGNORECASE if ci else 0))
        def _f(vb):
            v = vb.decode('utf-8', 'replace') if isinstance(vb, (bytes, bytearray)) else str(vb)
            return rx.match(v) is not None
        return _f
    def build_pred(node):
        if isinstance(node, E.Not) and isinstance(node.this, E.Is) \
                and isinstance(node.this.expression, E.Null):
            cs9n, cp9n, _p9n = resolve(node.this.this)
            if not cs9n.cols[cp9n].get('has_null'):
                return '(True)'                 # IS NOT NULL on a no-null column
            raise _FastUnsupported
        if isinstance(node, E.Is) and isinstance(node.expression, E.Null):
            cs9n, cp9n, _p9n = resolve(node.this)
            if not cs9n.cols[cp9n].get('has_null'):
                return '(False)'                # IS NULL on a no-null column
            raise _FastUnsupported
        # Compile a WHERE predicate to a numba boolean over the shared slots. Covers AND/OR/NOT, numeric &
        # datetime comparisons (incl. column-vs-column and arithmetic sides), BETWEEN, string '='/'!='/IN
        # via inline code comparison, and arbitrary single-string-column predicates (LIKE, ordering, IS NULL)
        # via a precomputed code-LUT. Raises _FastUnsupported otherwise -> materialised mask.
        if isinstance(node, E.Paren): return build_pred(node.this)
        if isinstance(node, E.And): return f"({build_pred(node.this)} and {build_pred(node.expression)})"
        if isinstance(node, E.Or):  return f"({build_pred(node.this)} or {build_pred(node.expression)})"
        if isinstance(node, E.Not): return f"(not {build_pred(node.this)})"
        if isinstance(node, (E.Like, E.ILike)):           # LIKE / ILIKE -> code-LUT over the dictionary
            if not FUSE_STR_PRED: raise _FastUnsupported
            col = node.this
            if not isinstance(col, E.Column): raise _FastUnsupported
            cseg, cpcol, cptr = resolve(col)
            if cseg.cols[cpcol]['dt'] != 1: raise _FastUnsupported
            pat = node.expression
            if not isinstance(pat, E.Literal) or not pat.is_string: raise _FastUnsupported
            vk = _code_lut(cseg, cpcol, cptr, _like_fn(cseg, cpcol, pat, isinstance(node, E.ILike)))
            return f"({vk} == 0)" if node.args.get('negate') else f"({vk} != 0)"   # NOT LIKE -> negate=True
        if isinstance(node, E.Is):                        # IS NULL (IS NOT NULL handled via E.Not)
            col = node.this
            if not isinstance(col, E.Column) or not isinstance(node.expression, E.Null):
                raise _FastUnsupported
            cseg, cpcol, cptr = resolve(col)
            if cseg.cols[cpcol]['dt'] != 1: raise _FastUnsupported   # numeric IS NULL: presence -> later
            vk = _code_lut(cseg, cpcol, cptr, None, mark_null=True)
            return f"({vk} != 0)"
        if type(node) in _CMP_STR:
            op = _CMP_STR[type(node)]; a, b = node.this, node.expression
            if not _is_lit(a) and not _is_lit(b):          # value-expr vs value-expr (e.g. col < col)
                return f"({build_fused(a)} {op} {build_fused(b)})"   # numeric/datetime; string -> raises
            if _is_lit(a) and _is_lit(b): raise _FastUnsupported
            col, lit, left = (a, b, True) if not _is_lit(a) else (b, a, False)
            if not isinstance(col, E.Column):          # computed expr vs numeric literal -> fuse both
                if not (isinstance(lit, E.Literal) and not lit.is_string): raise _FastUnsupported
                v = build_fused(col)
                lv = float(lit.this) if ('.' in lit.this or 'e' in lit.this.lower()) else int(lit.this)
                return f"({v} {op} {lv})" if left else f"({lv} {op} {v})"
            cseg, cpcol, cptr = resolve(col)
            if cseg.cols[cpcol]['dt'] == 1:                # string
                if not FUSE_STR_PRED or not lit.is_string: raise _FastUnsupported
                if type(node) in (E.EQ, E.NEQ):            # equality -> direct code comparison
                    vk, nullcode = _str_slot(cseg, cpcol, cptr)
                    target = _code_of_literal(cseg, cpcol, _lit_bytes(cseg, cpcol, lit))
                    if type(node) is E.EQ: return f"({vk} == {target})"
                    if nullcode is None:   return f"({vk} != {target})"
                    return f"(({vk} != {target}) and ({vk} != {nullcode}))"   # SQL: NULL != x not TRUE
                litb = _lit_bytes(cseg, cpcol, lit)        # ordering -> code-LUT (dict not order-preserving)
                opf = _OPS[type(node)]
                if not left:
                    opf = _OPS[{E.GT: E.LT, E.LT: E.GT, E.GTE: E.LTE, E.LTE: E.GTE}[type(node)]]
                vk = _code_lut(cseg, cpcol, cptr, lambda vb, opf=opf, litb=litb: bool(opf(vb, litb)))
                return f"({vk} != 0)"
            v = build_fused(col)
            kind = 'f' if cseg.cols[cpcol]['dt'] == 2 else 'i'
            lv = wdb_sql._lit_for_col(cseg, cpcol, lit, kind)
            if not isinstance(lv, (int, float, np.integer, np.floating)): raise _FastUnsupported
            lv = float(lv) if kind == 'f' else int(lv)
            return f"({v} {op} {lv})" if left else f"({lv} {op} {v})"
        if isinstance(node, E.In):
            col = node.this; exprs = node.args.get('expressions') or []
            if node.args.get('query') is not None: raise _FastUnsupported   # IN (subquery) -> mask
            if not isinstance(col, E.Column) or not exprs or len(exprs) > 256: raise _FastUnsupported
            cseg, cpcol, cptr = resolve(col)
            if cseg.cols[cpcol]['dt'] == 1:                # string IN -> OR of code equalities
                if not FUSE_STR_PRED: raise _FastUnsupported
                vk, _nc = _str_slot(cseg, cpcol, cptr)
                tgts = []
                for e in exprs:
                    if not e.is_string: raise _FastUnsupported
                    tgts.append(_code_of_literal(cseg, cpcol, _lit_bytes(cseg, cpcol, e)))
                return "(" + " or ".join(f"({vk} == {t})" for t in tgts) + ")"
            v = build_fused(col); kind = 'f' if cseg.cols[cpcol]['dt'] == 2 else 'i'  # numeric IN
            vals = []
            for e in exprs:
                lv = wdb_sql._lit_for_col(cseg, cpcol, e, kind)
                if not isinstance(lv, (int, float, np.integer, np.floating)): raise _FastUnsupported
                vals.append(float(lv) if kind == 'f' else int(lv))
            return "(" + " or ".join(f"({v} == {x})" for x in vals) + ")"
        if isinstance(node, E.Between):
            col = node.this
            if not isinstance(col, E.Column): raise _FastUnsupported
            cseg, cpcol, _c = resolve(col)
            if cseg.cols[cpcol]['dt'] == 1: raise _FastUnsupported
            v = build_fused(col); kind = 'f' if cseg.cols[cpcol]['dt'] == 2 else 'i'
            lo = wdb_sql._lit_for_col(cseg, cpcol, node.args['low'], kind)
            hi = wdb_sql._lit_for_col(cseg, cpcol, node.args['high'], kind)
            lo = float(lo) if kind == 'f' else int(lo); hi = float(hi) if kind == 'f' else int(hi)
            return f"(({lo} <= {v}) and ({v} <= {hi}))"
        raise _FastUnsupported

    _b0 = _tk9()
    if _bill9 is not None:
        _bill9.append(('setup(proj+gkeys+resolve)', _b0 - _stage0))
    pred_body = None
    if where is not None and not _where_spent:
        try:
            # THE POTENCY LAW (Jackson's ratio): conjuncts fire in descending
            # prune_fraction / access_cost -- the cheapest, deadliest test
            # first, so failed rows never pay a gather. Prune is EXACT from
            # the dictionary censuses where readable; cost by access class.
            def _flat9(n):
                if isinstance(n, E.Paren): return _flat9(n.this)
                if isinstance(n, E.And):
                    return _flat9(n.this) + _flat9(n.expression)
                return [n]
            def _potency9(n):
                cost = 1.0; prune = 0.5
                try:
                    cols9 = list(n.find_all(E.Column))
                    if any(resolve(c)[2] is not None for c in cols9):
                        cost = 6.0
                    if len(cols9) == 1:
                        cs9, cp9, _ = resolve(cols9[0])
                        td9 = np.asarray(cs9._typed_dict(cp9))
                        if td9.dtype.kind in 'iuf' and len(td9):
                            keep9 = _dict_keep(n, cs9, cp9, td9)
                            if keep9 is not None:
                                cc9 = np.asarray(cs9.code_counts(cp9))[:len(keep9)]
                                t9 = cc9.sum()
                                if t9:
                                    prune = 1.0 - float(cc9[keep9].sum()) / t9
                except Exception:
                    pass
                return prune / cost
            conj9 = _flat9(where.this)
            if _resid_c9X is not None:
                conj9 = _resid_c9X
            if _plane_mask9 is not None and not conj9:
                pred_body = None
            elif len(conj9) > 1:
                scored9 = [(_potency9(cn), build_pred(cn)) for cn in conj9]
                scored9.sort(key=lambda x: -x[0])
                pred_body = tuple(p for _, p in scored9)   # ordered conjuncts: the law rides to the codegen
            else:
                pred_body = build_pred(conj9[0])
        except _FastUnsupported:
            # FAIL LOUD: an unservable residual conjunct means THIS PLAN cannot
            # answer -- proceeding predicate-less silently dropped WHERE clauses
            # (Q21's EXISTS pair vanished and a confident wrong top-100 shipped
            # to the board, 2026-08-30). ONE explicit exception: an IN whose
            # subquery was resolved to codes (query arg kept for codes-unaware
            # consumers) is the MASK layer's by design -- the pred skips exactly
            # those and nothing else (Q4's EXISTS road). Anything else declines.
            slots.clear(); slot_list.clear()
            _fu9, _ok9 = [], []
            for _cn9 in (conj9 if isinstance(conj9, list) else []):
                try:
                    build_pred(_cn9); _ok9.append(_cn9)
                except _FastUnsupported:
                    _fu9.append(_cn9)
            slots.clear(); slot_list.clear()
            if not _fu9 or not all(isinstance(_c9, E.In)
                                    and ((_c9.args.get('query') is not None and (_c9.args.get('expressions') or []))
                                         or _c9.args.get('_codes') is not None) for _c9 in _fu9):
                raise
            # The mask layer serves: the FULL WHERE mask (mask_eval knows the
            # resolved codes) ANDs into the plan's mask; the pred stands down.
            _mfull9 = get_mask()
            if _mfull9 is None:
                raise
            _plane_mask9 = _mfull9 if _plane_mask9 is None else (_plane_mask9 & _mfull9)
            pred_body = None
        # CONSTANT FOLDING: a predicate that compiled to (False)/(True) -- IS [NOT] NULL on a
        # no-null column -- must not drag its column into a kernel slot (a 100M-row mode-4
        # decode cost 614ms to evaluate a constant on the megaboard's t-null/f-isnotnull)
        _parts9 = list(pred_body) if isinstance(pred_body, tuple) else ([pred_body] if pred_body is not None else [])
        if _parts9 and all(p9 in ('(False)', '(True)') for p9 in _parts9):
            slots.clear(); slot_list.clear()
            if any(p9 == '(False)' for p9 in _parts9):
                _plane_mask9 = np.zeros(int(seg_of[fact].N), bool)      # no mask evaluation: the constant IS the mask
                _maskc['m'] = _plane_mask9
            else:
                _maskc['m'] = None                                       # (True): as if there were no WHERE
            pred_body = None
        elif _parts9 and any(p9 == '(True)' for p9 in _parts9):
            _rest9 = [p9 for p9 in _parts9 if p9 != '(True)']
            pred_body = tuple(_rest9) if len(_rest9) > 1 else _rest9[0]

    plan = []; fully = wdb_exprjit.HAS_NUMBA
    for i, p in enumerate(proj):
        inner = p.this if isinstance(p, E.Alias) else p
        if isinstance(inner, E.Count) and (isinstance(inner.this, E.Star) or inner.this is None):
            plan.append((i, 'count', None, False, None))
        elif type(inner) in _CLS:
            try:
                body = build_fused(inner.this)
            except _FastUnsupported:
                fully = False; break
            is_dt = False; unit = None
            if isinstance(inner.this, E.Column):
                cseg, cpcol, _c = resolve(inner.this)
                if cseg.cols[cpcol]['dt'] == 3: is_dt = True; unit = cseg.unit(cpcol)
            plan.append((i, _CLS[type(inner)], body, is_dt, unit))
            if _CLS[type(inner)] in ('SUM', 'MIN', 'MAX'):
                try: int9[i] = _int_expr9(inner.this)
                except _FastUnsupported: pass
        elif isinstance(inner, E.Count):                 # COUNT(col): == group count only if non-nullable
            if not isinstance(inner.this, E.Column): fully = False; break
            cseg, cpcol, _c = resolve(inner.this)
            if cseg.cols[cpcol].get('has_null'): fully = False; break
            plan.append((i, 'count', None, False, None))
        elif gkeys and isinstance(inner, E.Column):      # bare GROUP BY key column
            pseg, ppcol, _r = resolve(inner)
            ki = next((j for j, k in enumerate(gkeys)
                       if k['seg'] is pseg and k['pcol'] == ppcol and not k.get('expr')
                       and (k['cptr'] is _r or (k['cptr'] is not None and _r is not None
                                                and k['cptr'] is not _r and False))), None)
            if ki is None:
                ki = next((j for j, k in enumerate(gkeys)
                           if k['seg'] is pseg and k['pcol'] == ppcol and not k.get('expr')), None)
                # two aliases of one table (Q7's n1/n2): (seg,pcol) is ambiguous,
                # the composed pointer is the alias's identity -- no match means
                # the pointers got rewritten (_rw9); match by group-node text then
                gsql9c = inner.sql()
                ki2 = next((j for j, gg in enumerate(gnodes) if gg.sql() == gsql9c), None)
                if ki2 is not None: ki = ki2
            if ki is None: fully = False; break
            plan.append((i, 'key', ki, False, None))
        elif gkeys and any(gg.sql() == inner.sql() for gg in gnodes):
            # EXPRESSION PROJECTION == EXPRESSION GROUP KEY (matched by text)
            ki = next(j for j, gg in enumerate(gnodes) if gg.sql() == inner.sql())
            plan.append((i, 'key', ki, False, None))
        else:
            fully = False; break

    def _grouped_cd():
        # Grouped COUNT(DISTINCT vcol): when the group key(s) and vcol are value-identity dict columns
        # whose dense (groups x value-cardinality) table is small, count distinct value codes per group
        # in ONE vectorised pass (a 2-D keep-table) instead of the sort-based fallback -- O(N). Returns
        # (counts, col_results) in the same group-id space the row assembler expects, else None.
        cd_idx = vnode = None
        for idx, p in enumerate(proj):
            inner = p.this if isinstance(p, E.Alias) else p
            if isinstance(inner, E.Count) and isinstance(inner.this, E.Distinct):
                dcols = list(inner.this.find_all(E.Column))
                if cd_idx is not None or len(dcols) != 1: return None      # one COUNT(DISTINCT col) only
                cd_idx = idx; vnode = dcols[0]
            elif isinstance(inner, E.Column):
                continue                                                   # bare group key (matched below)
            else:
                return None
        if cd_idx is None: return None
        if any(k['cptr'] is not None for k in gkeys): return None          # v1: fact-only group keys
        vseg, vpcol, vcptr = resolve(vnode)
        vc = vseg.cols[vpcol]
        if vc['mode'] == 4 or vcptr is not None: return None               # need value-identity fact codes
        nv = vc['V']
        if not RT.grouped_cdist_fits(K, nv): return None                   # dense table too big -> fallback
        op = _group_op()
        if op is None or op[0] != 'd': return None
        gid = np.asarray(op[1], dtype=np.int64)
        vcodes = (np.asarray(vseg.codes_at(vpcol, rows9)) if rows9 is not None
                  else vseg.codes(vpcol))   # fact-space by this lane's own gate
        m = get_mask()
        if m is not None: gid = gid[m]; vcodes = vcodes[m]
        # ONE pass: a (groups x value) cell count. Row totals give presence; nonzero non-null
        # columns per row give the distinct count. The null code is the last dict slot (V-1).
        table = np.bincount(gid * nv + vcodes, minlength=K * nv).reshape(K, nv)
        counts_l = table.sum(axis=1)                                       # rows per group (presence)
        nn = nv - 1 if vc['has_null'] else nv                              # COUNT(DISTINCT) drops NULL
        dist = (table[:, :nn] > 0).sum(axis=1).astype(object)
        cr = {}
        for idx, p in enumerate(proj):
            if idx == cd_idx:
                cr[idx] = ('arr', dist, False, None)
            else:
                pseg, ppcol, _r = resolve(p.this if isinstance(p, E.Alias) else p)
                ki = next((j for j, k in enumerate(gkeys) if k['seg'] is pseg and k['pcol'] == ppcol), None)
                if ki is None: return None
                cr[idx] = ('key', ki)
        return counts_l, cr

    cdist = _grouped_cd() if gkeys else None
    if cdist is not None:
        counts, col_results = cdist
    elif fully:
        ex_index = {}; exprs = []                        # dedup identical expressions; share one pass
        for (i, fn, body, is_dt, unit) in plan:
            if fn in ('SUM', 'AVG', 'MIN', 'MAX'):
                if body not in ex_index: ex_index[body] = len(exprs); exprs.append([body, False])
                if fn in ('MIN', 'MAX'): exprs[ex_index[body]][1] = True
        _mask = None if (pred_body or _where_spent
                         or _plane_mask9 is not None) else get_mask()
        if _plane_mask9 is not None and not _where_spent:
            _mask = _plane_mask9 if _mask is None else (_mask & _plane_mask9)
        _cm = gkeys[0]['seg'].cluster_meta() if len(gkeys) == 1 else None
        _no_mm = not any(e[1] for e in exprs)
        _rdx = _radix_plan(group_keys, slot_list, exprs, _no_mm, pred_body, _mask)
        if not group_keys and _no_mm:     # no GROUP BY, no MIN/MAX -> lean scalar kernel
            if _bill9 is not None:
                print('JOIN BILL: SCALAR pre-kernel=%.0fms' % ((_tk9() - _fpa_t0) * 1000), flush=True)
                _sk9 = _tk9()
            counts, results = wdb_exprjit.scalar_multi(slot_list, exprs, _mask, n, pred_body)
            if _bill9 is not None:
                print('JOIN BILL: SCALAR kernel=%.0fms' % ((_tk9() - _sk9) * 1000), flush=True)
        elif (len(gkeys) == 1 and _cm is not None and gkeys[0]['cptr'] is None and _no_mm
              and rows9 is None
              and _cm.get('key') == gkeys[0]['pcol'] and gkeys[0]['seg'].presence_mask() is None):
            counts, results = _slice_scalar_agg(group_keys, slot_list, exprs,   # cluster-key GROUP BY +
                                                _mask, n, pred_body, _cm['offsets'])  # predicate -> per-slice scalar
        elif _rdx is not None:            # high-card non-cluster key -> radix-partitioned aggregation
            _rk, _rK, _rmeas = _rdx
            counts, _rsum = wdb_radix.radix_grouped(_rk, _rK, _rmeas, NT=wdb_exprjit._NT)
            results = [] if _rmeas is None else [(_rsum[0], None, None)]
        else:
            if _bill9 is not None:
                _bill9.append(('assemble(pred+slots+keys+comp)', _tk9() - _b0)); _b1 = _tk9()
            counts, results = wdb_exprjit.grouped_multi(group_keys, slot_list, exprs,
                                                        _mask, n, pred_body,
                                                        mono=bool(locals().get('_mono_gids')))
            if _bill9 is not None:
                _bill9.append(('grouped_multi kernel', _tk9() - _b1)); _b1 = _tk9()
        nz = counts > 0
        for (i, fn, body, is_dt, unit) in plan:
            if fn == 'count':
                col_results[i] = ('count',)
            elif fn == 'key':
                col_results[i] = ('key', body)           # body field carries the group-key index
            else:
                s, mn, mx = results[ex_index[body]]
                if gkeys:
                    # GROUP BY: numeric always -- every consumer slices
                    # [present], and present IS the nonzero set (pred-killed
                    # groups never reach a row). The object-dtype NULL
                    # ceremony served nothing here and blocked the top-k
                    # gate with an object 'revenue' array.
                    if   fn == 'SUM':
                        o = np.rint(s).astype(np.int64) if int9.get(i) else s
                    elif fn in ('MIN', 'MAX') and int9.get(i):
                        o = np.rint(mn if fn == 'MIN' else mx).astype(np.int64)
                    elif fn == 'AVG':
                        o = np.divide(s, counts, out=np.zeros_like(s, dtype=np.float64),
                                      where=counts > 0)
                    elif fn == 'MIN': o = mn
                    else:             o = mx
                else:
                    # no GROUP BY: SQL demands ONE row even over zero rows,
                    # with NULL aggregates -- the ceremony earns its keep.
                    o = np.full(K, None, dtype=object)
                    # INTEGER EMISSION for the whole-table row too: an integer-typed SUM/MIN/MAX is
                    # an int (the union exposed floats here -- blockstats had always answered the
                    # single segment first: 1.0, 500000.0, 29998773.0 for 1, 500000, 29998773)
                    if   fn == 'SUM':
                        o = (np.rint(s).astype(np.int64) if int9.get(i) else s).astype(object); o[~nz] = None
                    elif fn == 'AVG': o[nz] = s[nz] / counts[nz]
                    elif fn == 'MIN': o[nz] = (np.rint(mn[nz]).astype(np.int64) if int9.get(i) else mn[nz])
                    else:             o[nz] = (np.rint(mx[nz]).astype(np.int64) if int9.get(i) else mx[nz])
                col_results[i] = ('arr', o, is_dt, unit)
    else:
        specs = []          # (i, fn, value_op, nullmask_op) for COUNT/SUM/AVG
        minmax = []         # (i, fn, value_op, nullmask_op) for MIN/MAX
        expr_aggs = []      # (i, fn, body, inputs) -- arithmetic aggregates fused via wdb_exprjit codegen
        numba_ok = wdb_agg.HAS_NUMBA   # cleared below if any value operand is string or nullable
        for i, p in enumerate(proj):
            inner = p.this if isinstance(p, E.Alias) else p
            if isinstance(inner, E.Count) and (isinstance(inner.this, E.Star) or inner.this is None):
                col_results[i] = ('count',)
            elif isinstance(inner, E.Count):
                vop, nop, seg, pcol = agg_arg_operand(inner.this)
                if nop is not None: numba_ok = False
                specs.append((i, 'COUNT', vop, nop)); col_results[i] = ('arr', None, False, None)
            elif type(inner) in _CLS:
                fn = _CLS[type(inner)]
                if not isinstance(inner.this, E.Column):
                    try:
                        _body, _inputs = fused_expr_build(inner.this)
                        expr_aggs.append((i, fn, _body, _inputs)); col_results[i] = ('arr', None, False, None)
                        if fn in ('SUM', 'MIN', 'MAX'):
                            try: int9[i] = _int_expr9(inner.this)
                            except _FastUnsupported: pass
                        continue
                    except _FastUnsupported:
                        pass
                vop, nop, seg, pcol = agg_arg_operand(inner.this)
                if nop is not None or (seg is not None and seg.cols[pcol]['dt'] == 1):
                    numba_ok = False     # nullable or string operand -> stay on the numpy reduction paths
                if fn in ('MIN', 'MAX'):
                    is_dt = (seg is not None and seg.cols[pcol]['dt'] == 3)
                    minmax.append((i, fn, vop, nop)); col_results[i] = ('arr', None, is_dt,
                                                                         (seg.unit(pcol) if is_dt else None))
                else:
                    specs.append((i, fn, vop, nop)); col_results[i] = ('arr', None, False, None)
            else:
                if not gkeys or not isinstance(inner, E.Column): raise _FastUnsupported  # bare col w/o GROUP BY
                pseg, ppcol, _ = resolve(inner)
                ki = next((j for j, k in enumerate(gkeys) if k['seg'] is pseg and k['pcol'] == ppcol), None)
                if ki is None: raise _FastUnsupported          # projected column is not a GROUP BY key
                col_results[i] = ('key', ki)

        # codegen-fused arithmetic aggregates: one pass each (decode + expression + accumulate), no array.
        # SUM and AVG of the SAME expression share a single pass; MIN/MAX trigger the min/max accumulators.
        expr_counts = None
        if expr_aggs:
            groups = {}
            for (i, fn, body, inputs) in expr_aggs:
                ek = (body, tuple(id(b) for b, _, _ in inputs))
                g = groups.setdefault(ek, {'body': body, 'inputs': inputs, 'aggs': [], 'mm': False})
                g['aggs'].append((i, fn))
                if fn in ('MIN', 'MAX'): g['mm'] = True
            for g in groups.values():
                cE, sE, mnE, mxE = wdb_exprjit.grouped_expr(group_keys, g['body'], g['inputs'], get_mask(), n, g['mm'])
                expr_counts = cE; nz = cE > 0
                for (i, fn) in g['aggs']:
                    o = np.full(K, None, dtype=object)
                    if   fn == 'SUM': o = sE.astype(object); o[~nz] = None
                    elif fn == 'AVG': o[nz] = sE[nz] / cE[nz]
                    elif fn == 'MIN': o[nz] = mnE[nz]
                    else:             o[nz] = mxE[nz]
                    cr = col_results[i]; col_results[i] = ('arr', o, cr[2], cr[3])

        # plain (column / star) aggregates via the existing fused/numpy machinery
        if specs or minmax:
            if numba_ok:
                counts, agg_arrays = wdb_agg.fused_numba(_group_op(), K, specs + minmax, _mask_op(), n)
            elif RT.parallel_worth_it(n) and not minmax:
                counts, agg_arrays = wdb_agg.fused_counts_and_aggs(_group_op(), K, specs, _mask_op(), n)
            else:
                gc = wdb_agg._slice(_group_op(), 0, n)
                gcodes = np.zeros(n, dtype=np.int64) if gc is None else gc.astype(np.int64, copy=False)
                _m = get_mask()
                if _m is not None: gcodes = gcodes[_m]
                counts = wdb_agg.group_counts(gcodes, K)
                def _materialize(vop, nop):
                    _m = get_mask()
                    v = wdb_agg._slice(vop, 0, n); v = v[_m] if _m is not None else v
                    nm = wdb_agg._slice(nop, 0, n)
                    if nm is not None and _m is not None: nm = nm[_m]
                    return v, nm
                agg_arrays = {}
                for (i, fn, vop, nop) in specs + minmax:
                    v, nm = _materialize(vop, nop); agg_arrays[i] = wdb_agg.group_agg(gcodes, K, fn, v, nm)
            for i, arr in agg_arrays.items():
                cr = col_results[i]; col_results[i] = ('arr', arr, cr[2], cr[3])
        elif expr_counts is not None:
            counts = expr_counts                                  # only arithmetic aggregates -> counts from codegen
        else:                                                     # only COUNT(*) / key columns -> counts-only pass
            gc = wdb_agg._slice(_group_op(), 0, n)
            gcodes = np.zeros(n, dtype=np.int64) if gc is None else gc.astype(np.int64, copy=False)
            _m = get_mask()
            if _m is not None: gcodes = gcodes[_m]
            counts = wdb_agg.group_counts(gcodes, K)
    # No GROUP BY -> exactly one output row (the grand total), even over zero rows (COUNT=0, SUM/MIN/MAX=NULL,
    # matching SQL). With a GROUP BY, empty groups are dropped.
    present = np.array([0]) if not gkeys else np.nonzero(counts > 0)[0]
    present = _topk_prefilter(tree, proj, col_results, counts, present, gkeys)   # bounded top-K: drop non-winners pre-assembly

    # ---- assemble rows (vectorised) ----
    # Decode every present group's composite code into per-key code arrays in one shot, bulk-decode each key
    # column through its dict, and bulk-pull each aggregate -- then zip columns into row tuples. The old path
    # was a python loop over present groups calling fetch() per cell, which dominated high-card output.
    radices = [k['K'] for k in gkeys]
    if not columnar:
        import wdb_govern
        wdb_govern.ask(int(present.size), len(proj), 'grouped result')    # THE GOVERNOR: rows as Python cells
    kc_arr = []
    if gkeys and _gid_words9 is not None:
        g0, g1, shifts9, words9, widths9 = _gid_words9
        kc_arr = [None] * len(gkeys)
        for j in range(len(gkeys)):
            src9 = (g0 if words9[j] == 0 else g1)[present]
            kc_arr[j] = (src9 >> int(shifts9[j])) & ((1 << widths9[j]) - 1)
    elif gkeys:
        comp = (present.astype(np.int64, copy=True) if gid_to_comp is None
                else np.asarray(gid_to_comp, dtype=np.int64)[present].copy())   # factorised id -> composite
        kc_arr = [None] * len(gkeys)
        for j in range(len(gkeys) - 1, -1, -1):
            kc_arr[j] = comp % radices[j]; comp //= radices[j]

    # Native columnar result: hand back {name: ndarray} built from the already-native group arrays,
    # skipping the per-cell _pyval decode and the row-tuple zip. Only when no HAVING/ORDER/LIMIT needs
    # row materialisation (those stay on the row path). This is the big lever at high cardinality, where
    # the per-row assembly -- not the gather -- dominated db.run.
    if (columnar and tree.args.get('having') is None and tree.args.get('order') is None
            and wdb_sql._limit(tree) is None):
        names = [wdb_sql._alias(p) for p in proj]
        out = {}
        for i, p in enumerate(proj):
            r = col_results[i]; nm = names[i]
            if r[0] == 'key':
                gk = gkeys[r[1]]
                if 'labels' in gk:
                    out[nm] = np.asarray(gk['labels'], dtype=object)[kc_arr[r[1]]]
                else:
                    out[nm] = np.asarray(_bulk_keyvals(gk['seg'], gk['pcol'], kc_arr[r[1]]), dtype=object)
            elif r[0] == 'count':
                out[nm] = counts[present]
            else:
                picked = r[1][present]
                out[nm] = picked.astype(np.int64).view(f"datetime64[{r[3]}]") if r[2] else picked
        _bump_fast()
        return out, names

    # THE TOP-K EMIT (the realm's oldest move, arriving at the join court):
    # with ORDER BY + LIMIT and no HAVING, select the k winners in ARRAY
    # space -- agg results are arrays already, and numeric-dict key codes are
    # order-isomorphic to their values (dict-locked ordered encode) -- then
    # materialise only k python rows instead of every group.
    _sel9 = None
    _ord9 = tree.args.get('order')
    _lim9 = wdb_sql._limit(tree)
    if _ord9 is not None and _lim9 is not None and tree.args.get('having') is None:
        try:
            keys9 = []
            aliases9 = [wdb_sql._alias(p) for p in proj]
            for oe in _ord9.expressions:
                tgt = oe.this
                nm9 = tgt.name if hasattr(tgt, 'name') else str(tgt)
                if nm9 not in aliases9:
                    raise ValueError('order term %r not in %r' % (nm9, aliases9))
                i9 = aliases9.index(nm9)
                r9 = col_results[i9]
                if r9[0] == 'count':
                    a9 = counts[present].astype(np.float64)
                elif r9[0] == 'key':
                    gk9 = gkeys[r9[1]]
                    if 'labels' in gk9:
                        a9 = np.asarray(gk9['labels'])[kc_arr[r9[1]]].astype(np.float64)
                    else:
                        if gk9['seg'].cols[gk9['pcol']].get('dt') == 1:
                            raise ValueError('string key %s' % gk9['pcol'])
                        a9 = kc_arr[r9[1]].astype(np.float64)
                else:
                    if r9[1].dtype == object:
                        raise ValueError('object-dtype agg %r' % nm9)
                    a9 = r9[1][present].astype(np.float64)
                keys9.append(-a9 if oe.args.get('desc') else a9)
            _sel9 = np.lexsort(tuple(reversed(keys9)))[:int(_lim9)]
        except Exception as _e9:
            if _bill9 is not None:
                print('JOIN BILL: topk-declined %s: %s' % (type(_e9).__name__, str(_e9)[:80]), flush=True)
            _sel9 = None
    if _sel9 is not None:
        present = np.flatnonzero(present)[_sel9] if present.dtype == bool else present[_sel9]
        counts_present = counts[present]
        kc_arr = [k9[_sel9] for k9 in kc_arr]
    col_lists = []
    for i, p in enumerate(proj):
        r = col_results[i]
        if r[0] == 'key':
            gk = gkeys[r[1]]
            if 'labels' in gk:                            # affine/factorised key: gid -> value via label table
                _labo = gk.get('_labels_obj')
                if _labo is None:
                    _labo = gk['_labels_obj'] = np.asarray(gk['labels'], dtype=object)
                col_lists.append(_labo[kc_arr[r[1]]].tolist())     # C-speed gather, not a list comp
            else:
                col_lists.append(_bulk_keyvals(gk['seg'], gk['pcol'], kc_arr[r[1]]))
        elif r[0] == 'count':
            col_lists.append(counts[present].tolist())
        else:
            picked = r[1][present]                       # present is already narrowed by _sel9
            if r[2]:                                      # datetime epoch -> datetime64 -> _pyval string
                unit = r[3]
                col_lists.append([(wdb_sql._pyval(np.int64(v).view(f'datetime64[{unit}]')) if v is not None
                                   else None) for v in picked.tolist()])
            elif picked.dtype.kind in 'iuf':
                col_lists.append(picked.tolist())          # tolist() already yields Python scalars
            else:
                col_lists.append([wdb_sql._pyval(v) for v in picked.tolist()])
    rows = list(zip(*col_lists)) if col_lists else [() for _ in present]

    global _FAST_HITS; _FAST_HITS += 1
    having = tree.args.get('having')
    if having is not None:
        rows = wdb_sql._apply_having(rows, proj, having.this, None)   # fused path must filter too
    if _sel9 is None:
        rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))
        lim = wdb_sql._limit(tree)
        if lim is not None: rows = rows[:lim]
    if _bill9 is not None:
        _bill9.append(('emit+having+order', _tk9() - _b1))
        print('JOIN BILL: ' + ' | '.join('%s=%.0fms' % (k9, v9 * 1000) for k9, v9 in _bill9), flush=True)
    return rows, [wdb_sql._alias(p) for p in proj]


# ── Multi-table FK-pointer chain (rung: breadth) ─────────────────────────────
# Generalises the single FK pointer to a walk: lineitem -> orders -> customer -> nation -> region.
# Each join's ON must match a stored FK pointer (child -> parent). The "fact" table is the one that is
# a child but never a parent; from it every other table is reached by composing the per-edge pointers
# (compose by gathering the next pointer through the current one). composed[alias] maps each fact row to
# that table's row (None for the fact itself). Raises _FastUnsupported if the join graph is not an
# FK-pointer-rooted tree, so the caller can fall back.
def _key_is_unique(db, table, col):
    """True if `col` in `table` is a single clean segment with all-distinct
    values (a candidate join parent). The dictionary already knows: V == N is
    uniqueness, read from metadata in O(1); mode-4 identity codes are unique by
    construction. Memoized; the old materialize-and-count survives only as the
    fallback for dressless columns."""
    memo = db.__dict__.setdefault('_uniq_memo', {})
    mk = (table, col)
    if mk in memo:
        return memo[mk]
    try: seg, _ = _solo_segment(db, table)
    except _FastUnsupported: return False
    pc = db.cat.phys_map(table).get(col, col)
    if pc not in seg.cols:
        memo[mk] = False
        return False
    c9 = seg.cols[pc]
    r = None
    if c9.get('mode') == 4:
        r = True                                   # identity: unique by law
    elif c9.get('mode') == 6:
        r = seg.N <= 1                             # constant column
    else:
        V9 = c9.get('V')
        hn9 = c9.get('has_null')
        if V9 is not None and not hn9 and c9.get('mode') in (0, 1, 2):
            r = int(V9) == int(seg.N)              # dict distincts vs rows: O(1)
    if r is None:
        v = wdb_sql._col(seg, pc)[0]
        r = v is not None and len(v) == len(np.unique(np.asarray(v)))
    memo[mk] = bool(r)
    return memo[mk]


_JPTR_ASKED = set()
_JPTR_NOT = set()


def _hash_pointer(db, ctbl, ckey, cseg, ptbl, pkey, pseg):
    """Build a child->parent gather pointer at query time via a hash probe (the parent key must be unique).
    For each child row, the parent row whose key matches. INNER + row-preserving only: a partial match would
    drop child rows, so that raises _FastUnsupported and the query falls back. A hash join becomes a pointer,
    and every downstream gather/predicate/aggregate stays on the same fused chain."""
    cp = db.cat.phys_map(ctbl).get(ckey, ckey); pp = db.cat.phys_map(ptbl).get(pkey, pkey)
    # THE POINTER SIDECAR (the plists' cousin): a resolved child->parent
    # pointer is derived data, so it is born once beside the child segment
    # and mmap'd forever after -- the N-RAM law satisfied on disk. A stale
    # or foreign sidecar can only misroute rows, so it carries a birthmark
    # (parent segment path + N) checked before trust.
    import os as _os, json as _js
    sc9 = getattr(cseg, 'path', None)
    pp9 = getattr(pseg, 'path', None)
    side = None
    mark = None
    if sc9 and pp9:
        side = '%s.%s__%s.%s.jptr.npy' % (sc9, cp, ptbl, pp)
        try:
            st_c = _os.stat(sc9); st_p = _os.stat(pp9)
            mark = {'c': [_os.path.realpath(sc9), st_c.st_size, st_c.st_mtime_ns],
                    'p': [_os.path.realpath(pp9), st_p.st_size, st_p.st_mtime_ns],
                    'ck': cp, 'pk': pp}
        except Exception:
            side = None
        import wdb_sidecar as _wsc
        if side and mark:
            # THE ROAD ON THE SHELF: a pointer that was NOT persisted (the switch off, or the first ask)
            # still lives for the process under the shelf's ceiling, keyed by both parents' identity --
            # sidecars off means nothing on disk, not a 36M-row hash per query
            import wdb_shelf as _wsh9
            _hit9 = _wsh9.SHELF.get(('jptr', side, _js.dumps(mark, sort_keys=True)))
            if _hit9 is not None and _hit9.shape[0] == cseg.N:
                return _hit9
        if side and _wsc.exists(side + '.mark') and not _wsc.exists(side):
            # FAIL LOUD: a birthmark without its body is a lie -- remove it so the
            # rebirth below persists cleanly (a 480MB road was rehashing every
            # query for a day behind an orphan mark, 2026-08-30).
            print('ROAD: orphan birthmark without body, removing: %s' % side, flush=True)
            try: _os.remove(side + '.mark')
            except Exception: pass
        if side and _wsc.exists(side) and _wsc.exists(side + '.mark'):
            try:
                # THE BIRTHMARK (plist regime): the sidecar names BOTH parents'
                # identity (path+size+mtime) and the key pair; any mismatch is a
                # hard refusal, never a guess -- a stale pointer misroutes rows.
                if _js.load(open(side + '.mark')) == mark:
                    ram9 = getattr(db, '_road_ram', None)
                    if ram9 is None:
                        ram9 = db._road_ram = {}
                    ptr9 = ram9.get(side)
                    if ptr9 is None:
                        # ROADS RIDE IN RAM: on a network mount the mmap re-pages
                        # ~450ms per query (client cache too small for 480MB);
                        # read once per process, narrowed to int32 where it fits.
                        ptr9 = np.load(side, mmap_mode='r')
                        ptr9 = (np.asarray(ptr9, dtype=np.int32) if pseg.N < (1 << 31)
                                else np.ascontiguousarray(ptr9))
                        ram9[side] = ptr9
                    if ptr9.shape[0] == cseg.N:
                        return ptr9
            except Exception:
                pass
    # A REFUSAL IS A FACT ABOUT THE TWO SEGMENTS: 'not a pointer' (many-to-many, or orphan child
    # rows -- cast_info.person_id has people missing from name) is remembered under the same
    # birthmark as a road, so it is decided once -- every JOB-COUNT query had rebuilt a 36M-row
    # pointer (0.55s) to re-discover the same refusal
    if side and mark:
        if side in _JPTR_NOT: raise _FastUnsupported
        try:
            if _wsc.exists(side + '.no') and _js.load(open(side + '.no')) == mark:
                _JPTR_NOT.add(side); raise _FastUnsupported
        except _FastUnsupported:
            raise
        except Exception:
            pass
    def _refuse():
        if side and mark:
            _JPTR_NOT.add(side)
            try:
                if _wsc.births_on(_os.path.dirname(side)):                    # THE SWITCH: the memo is in RAM regardless
                    _js.dump(mark, open(side + '.no', 'w'))
            except Exception: pass
        raise _FastUnsupported
    ck = np.asarray(wdb_sql._col(cseg, cp)[0]); pk = np.asarray(wdb_sql._col(pseg, pp)[0])
    pidx = pd.Index(pk)
    if not pidx.is_unique: _refuse()                              # many-to-many -> not a pointer
    ptr = pidx.get_indexer(ck)
    if (ptr < 0).any(): _refuse()                                 # unmatched child rows -> would drop -> fall back
    ptr = ptr.astype(np.int64)
    if side and mark:
        try:
            import wdb_shelf as _wsh9
            _wsh9.SHELF.put(('jptr', side, _js.dumps(mark, sort_keys=True)), ptr, int(ptr.nbytes), kind='road')
        except Exception:
            pass                                                      # the shelf declined: this query still has it
    # THE BIRTH GATE (plist spirit: born where reads justify): the first
    # qualifying join per (child,key,parent,key) pays its hash in RAM only;
    # the sidecar is born on the SECOND ask, so one-off exploratory joins
    # never cost the realm disk. Operators may force births with
    # WDB_JPTR_EAGER=1 or forbid them with WDB_JPTR_OFF=1.
    if side and mark and not _os.environ.get('WDB_JPTR_OFF') and _wsc.births_on(_os.path.dirname(side)):   # THE SWITCH
        # THE SECOND ASK must be remembered across queries: this set lived in db.__dict__, which the
        # per-query flush wiped, so every ask was the first -- cast_info's roads were never born
        # and every JOB-COUNT query rebuilt a 36M-row pointer (0.55s) and threw it away
        asked = _JPTR_ASKED
        key9 = (side,)
        if key9 in asked or _os.environ.get('WDB_JPTR_EAGER'):
            try:
                # Width stays intp/int64 BY MEASUREMENT: numpy indexes with
                # intp, so narrow pointer arrays pay a cast-copy at every
                # gather (u32 cost Q3 +50ms/query for 240MB disk saved --
                # the verdict law ruled disk loses).
                np.save(side + '.tmp.npy', ptr)
                _os.replace(side + '.tmp.npy', side)
                _js.dump(mark, open(side + '.mark', 'w'))
            except Exception as _e9:
                if _os.environ.get('WDB_JOIN_BILL'):
                    print('JOIN: road save failed for %s: %s' % (side, str(_e9)[:100]), flush=True)
        else:
            asked.add(key9)
            if _os.environ.get('WDB_JOIN_BILL'):
                print('JOIN: road first ask %s' % side, flush=True)
    elif _os.environ.get('WDB_JOIN_BILL'):
        print('JOIN: road not persistable (side=%s mark=%s)' % (bool(side), bool(mark)), flush=True)
    return ptr


def _build_chain(db, tree, allow_hash=True):
    frm = tree.find(E.From).this
    tables = [(frm.name, frm.alias or frm.name)]
    for jn in (tree.args.get('joins') or []):
        if jn.args.get('side'): raise _FastUnsupported                           # INNER only
        if jn.args.get('kind') and str(jn.args.get('kind')).upper() not in ('CROSS', 'INNER'):
            raise _FastUnsupported                                               # comma-dialect rides as CROSS; INNER ... ON is the same tree
        if jn.args.get('kind') == 'CROSS' and jn.args.get('on') is not None:
            raise _FastUnsupported
        if not isinstance(jn.this, E.Table): raise _FastUnsupported              # no subqueries
        tables.append((jn.this.name, jn.this.alias or jn.this.name))
    alias2t = {a: t for t, a in tables}
    if len(alias2t) != len(tables): raise _FastUnsupported                       # duplicate/self alias

    # THE DIALECT MINER: equalities come from ON clauses (residual conjuncts
    # shifted into WHERE -- lawful for INNER) and, for comma-dialect joins,
    # from WHERE's own col=col conjuncts. The edge court below is unchanged.
    def _conjuncts(x):
        if isinstance(x, E.And):
            return _conjuncts(x.this) + _conjuncts(x.expression)
        return [x]
    col_owner = {}                                  # bare name -> alias (None if ambiguous)
    for t9, a9 in tables:
        try:
            for cn9 in db.cat.column_names(t9):
                col_owner[cn9] = None if cn9 in col_owner else a9
        except Exception:
            pass
    def _own(c9):
        return c9.table or col_owner.get(c9.name)   # TPC-H speaks bare names
    eq_pool = []
    residuals = []
    for jn in (tree.args.get('joins') or []):
        on = jn.args.get('on')
        if on is None:
            continue
        got = False
        for cj in _conjuncts(on):
            if (not got and isinstance(cj, E.EQ)
                    and isinstance(cj.this, E.Column)
                    and isinstance(cj.expression, E.Column)):
                eq_pool.append(cj)
                got = True
            else:
                residuals.append(cj)
        if not got: raise _FastUnsupported
    w9 = tree.args.get('where')
    w_eqs = []
    if w9 is not None:
        for cj in _conjuncts(w9.this):
            if (isinstance(cj, E.EQ) and isinstance(cj.this, E.Column)
                    and isinstance(cj.expression, E.Column)
                    and _own(cj.this) and _own(cj.expression)
                    and _own(cj.this) != _own(cj.expression)):
                eq_pool.append(cj)
                w_eqs.append(cj)
    if residuals and not tree.args.get('_wdb_onshift'):
        import sqlglot as _sg
        merged = residuals[0]
        for r9 in residuals[1:]:
            merged = E.And(this=merged, expression=r9)
        if w9 is not None:
            w9.set('this', E.And(this=w9.this, expression=merged))
        else:
            tree.set('where', E.Where(this=merged))
        tree.set('_wdb_onshift', True)

    edges = {}            # child_alias -> (parent_alias, fk_col)
    parents = set()
    _edge_w = []
    for on in eq_pool:
        le, re = on.this, on.expression
        aA, kA, aB, kB = _own(le), le.name, _own(re), re.name
        tA, tB = alias2t.get(aA), alias2t.get(aB)
        if tA is None or tB is None: raise _FastUnsupported
        fkA, fkB = db.cat.fk_pointers(tA), db.cat.fk_pointers(tB)
        if kA in fkA and fkA[kA]['parent'] == tB and fkA[kA]['parent_key'] == kB:
            child_a, parent_a, fk_col = aA, aB, kA
        elif kB in fkB and fkB[kB]['parent'] == tA and fkB[kB]['parent_key'] == kA:
            child_a, parent_a, fk_col = aB, aA, kB
        elif allow_hash and _key_is_unique(db, tB, kB):                          # non-FK: parent = unique-key side
            child_a, parent_a, fk_col = aA, aB, ('hash', kA, kB)                 # built as a runtime hash pointer
        elif allow_hash and _key_is_unique(db, tA, kA):
            child_a, parent_a, fk_col = aB, aA, ('hash', kB, kA)
        else:
            continue                                             # neither key unique: rides WHERE as a filter
        ee9 = edges.setdefault(child_a, [])
        if any(pa9 == parent_a for pa9, _f9 in ee9):
            continue                                 # duplicate eq for the same hop stays a filter
        ee9.append((parent_a, fk_col)); parents.add(parent_a)
        if id(on) in {id(x) for x in w_eqs}:
            _edge_w.append(on)

    if not edges:                                                                # 0 joins: single-table query
        if len(tables) != 1: raise _FastUnsupported                              # multiple tables, no FK edge
        fact = tables[0][1]
    else:
        fact_candidates = [a for a in edges if a not in parents]
        if len(fact_candidates) != 1: raise _FastUnsupported                     # need a single rooted fact
        fact = fact_candidates[0]

    # THE TREE (Jackson's Q5): a child may have MANY parents. BFS from the
    # fact records each alias's route; pruning keeps exactly the aliases on
    # paths from the fact to anything the query actually reads.
    prev9 = {}
    bfs9 = [fact]
    seen9 = {fact}
    while bfs9:
        c9t = bfs9.pop()
        for pa9, _f9 in edges.get(c9t, []):
            if pa9 not in seen9:
                seen9.add(pa9); prev9[pa9] = c9t; bfs9.append(pa9)

    # JOIN PRUNING. An FK pointer is built only after verifying referential integrity (every child maps to
    # exactly one parent, parent key unique), so each child->parent INNER join is row-preserving -- joining
    # a table the query never reads cannot change the result. So we only need to compose pointers up to the
    # furthest table actually referenced by the projection / WHERE / GROUP BY / ORDER BY (NOT the join ON
    # columns, which are join plumbing). Everything beyond it is dropped, saving a gather per pruned hop.
    cols_of = {a: set(db.cat.column_names(t)) for a, t in alias2t.items()}
    ref = {fact}
    scan = list(tree.expressions)
    for key in ('where', 'group', 'order'):
        node = tree.args.get(key)
        if node is not None: scan.append(node)
    for rootn in scan:
        for col in rootn.find_all(E.Column):
            a = col.table
            if a and a in alias2t:
                ref.add(a)
            elif not a:                                          # unqualified: keep every candidate owner
                ref.update(al for al, cs in cols_of.items() if col.name in cs)
    keep = {fact}
    for r9t in ref:
        cur9t = r9t
        while cur9t in prev9:
            keep.add(cur9t); cur9t = prev9[cur9t]

    seg_of, sp_of = {}, {}
    for _, a in tables:
        seg_of[a], sp_of[a] = _solo_segment(db, alias2t[a])

    composed = {fact: None}                                                      # None = identity (fact rows)
    edge_ptrs = {}
    progress = True
    while progress:
        progress = False
        for child_a, ee9 in edges.items():
          for parent_a, fk_col in ee9:
            if parent_a not in keep: continue                                    # pruned hop: skip the gather
            if child_a in composed and parent_a not in composed:
                if isinstance(fk_col, tuple) and fk_col and fk_col[0] == 'hash':
                    if not allow_hash:
                        raise _FastUnsupported   # O(N) hash as ROUTING is forbidden: the
                                                 # dict-space route gets its turn first
                    p = _hash_pointer(db, alias2t[child_a], fk_col[1], seg_of[child_a],
                                      alias2t[parent_a], fk_col[2], seg_of[parent_a])
                else:
                    # THE CANONICAL ROAD FIRST (fk_pointer/path_for retired as
                    # the routing default): the legacy zstd .fkptr sidecar costs
                    # 480ms of decompress+cumsum after every qmem flush; the
                    # birthmarked .jptr road is mmap'd (and RAM-pinned) once.
                    p = None
                    try:
                        fkm9 = (db.cat.fk_pointers(alias2t[child_a]) or {}).get(fk_col)
                        if isinstance(fkm9, dict) and fkm9.get('parent_key'):
                            p = _hash_pointer(db, alias2t[child_a], fk_col, seg_of[child_a],
                                              alias2t[parent_a], fkm9['parent_key'], seg_of[parent_a])
                    except Exception:
                        p = None
                    if p is None:
                        p = db.fk_pointer(sp_of[child_a], fk_col)
                if p is None: raise _FastUnsupported
                cc = composed[child_a]
                edge_ptrs[parent_a] = (child_a, p)       # child-scale road, kept for the downhill flow
                if cc is None:
                    composed[parent_a] = p
                else:
                    _pc9 = np.asarray(p); _cc9 = np.asarray(cc)
                    _o9 = np.empty(_cc9.shape[0], dtype=_pc9.dtype)
                    wdb_kernels.pgather_ptr(_pc9, _cc9, _o9)     # parallel compose
                    composed[parent_a] = _o9                  # compose by gather
                progress = True
    if any(a not in composed for a in keep): raise _FastUnsupported              # kept tables must connect
    # EARLY DECLINE: an alias the query reads that this chain never composed
    # (the stored-only attempt on a tree needing hash edges) fails HERE, not
    # after running a 125ms cascade and dying in resolve (Q10's pre-work=256).
    if any((a in alias2t) and (a not in composed) for a in ref):
        raise _FastUnsupported
    # THE STRIP RUNS ONLY ON SUCCESS: the allow_hash=False attempt used to
    # consume edge equalities from WHERE and then raise, leaving the retry a
    # gutted tree (Q5's nation vanished this way).
    if w9 is not None and _edge_w:
        consumed = {id(x) for x in _edge_w}
        kept = [cj for cj in _conjuncts(w9.this) if id(cj) not in consumed]
        # col=col survivors (Q5's c_nationkey = s_nationkey): neither side is
        # unique, so it is not an edge -- it RIDES WHERE as a pred conjunct.
        # build_pred compares the two parent VALUE streams through their
        # pointer slots (dict-independent); if a side cannot resolve, the
        # pred raises and the pandas tail catches it. The old blanket
        # decline predates parent-vs-parent compare in the fused pred.
        if kept:
            merged = kept[0]
            for cj in kept[1:]:
                merged = E.And(this=merged, expression=cj)
            w9.set('this', merged)
        else:
            tree.set('where', None)
    return dict(fact=fact, alias2t=alias2t, seg_of=seg_of, composed=composed, n=seg_of[fact].N,
                edge_ptrs=edge_ptrs)
