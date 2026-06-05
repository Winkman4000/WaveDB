#!/usr/bin/env python3
"""
wdb_plan -- turn measured artifacts (host calibration + column profile) into
physical-design decisions. Pure policy over stats; touches no data.

Per column:
  code_lut         per-code boolean LUT for range/equality (value-identity, V<=lut_cliff)
  content_key      near-unique -> free content-address / primary key
  range_sliceable  sorted dict -> a range filter becomes a contiguous slice IF cluster key
  groupable        value-identity codes -> usable as GROUP BY / DISTINCT key (fused path)
  code_width_bits  entropy-minimal in-memory code width

Table:
  cluster.key      the single column to spend the one free permutation on. Workload-driven
                   when a workload is given; structural prior (datetime > high-card dim) else.
"""
import os, json, math

VALUE_IDENTITY_MODES = {0, 1, 2, 5, 6}     # codes index a real value dict (mode 4 = positional)
UNIQ_THRESH = 0.9                          # content-address eligibility


def plan_column(s, calib):
    vid = int(s['mode']) in VALUE_IDENTITY_MODES
    lut_cliff = int(calib.get('lut_cliff', 65536))
    return {
        'code_lut':        bool(vid and int(s['V']) <= lut_cliff),
        'content_key':     bool(float(s['uniq']) >= UNIQ_THRESH),
        'range_sliceable': (s.get('sorted') is True),
        'groupable':       bool(vid),
        'code_width_bits': int(s['code_bits']),
    }


def _cluster_candidates(stats, plans):
    # viable cluster key: a range/eq filter on it can become a slice (sorted) and the
    # codes are value-identity so the slice is well-defined.
    return [nm for nm, s in stats.items()
            if plans[nm]['range_sliceable'] and plans[nm]['groupable']]


def choose_cluster_key(stats, plans, calib, workload=None):
    cands = _cluster_candidates(stats, plans)
    if not cands:
        return {'key': None, 'reason': 'no sorted value-identity column', 'ranked': []}
    if workload:
        # workload: list of {'col': name, 'kind': 'range'|'eq'|'group', 'freq': float}
        score = {c: 0.0 for c in cands}
        for q in workload:
            c = q.get('col')
            if c in score and q.get('kind') in ('range', 'eq', 'group'):
                score[c] += float(q.get('freq', 1.0))
        ranked = sorted(cands, key=lambda c: score[c], reverse=True)
        best = ranked[0] if score[ranked[0]] > 0 else None
        return {'key': best, 'reason': 'workload-weighted',
                'ranked': [(c, round(score[c], 3)) for c in ranked]}
    # no workload: structural prior. datetime = canonical analytic cluster dimension;
    # otherwise prefer higher-cardinality NON-unique dims (more selective slices).
    def sscore(nm):
        s = stats[nm]
        if plans[nm]['content_key']:
            return -1.0                                   # a unique key is a poor dimension
        return (1000.0 if int(s['dt']) == 3 else 0.0) + math.log2(max(int(s['V']), 2))
    ranked = sorted(cands, key=sscore, reverse=True)
    return {'key': ranked[0], 'reason': 'structural prior (no workload)',
            'ranked': [(c, round(sscore(c), 3)) for c in ranked]}


def plan_segment(stats, calib, workload=None):
    plans = {nm: plan_column(s, calib) for nm, s in stats.items()}
    return {
        'columns': plans,
        'cluster': choose_cluster_key(stats, plans, calib, workload),
        'host': {'locality_penalty': calib.get('locality_penalty'),
                 'lut_cliff': calib.get('lut_cliff'), 'n_cores': calib.get('n_cores')},
    }


def plan_for_segment(segment_path, workload=None, calib=None):
    """Convenience: load the sidecar profile + host calibration, return the plan."""
    import wdb_profile, wdb_calibrate
    stats = wdb_profile.load_profile(segment_path)
    if stats is None:
        raise FileNotFoundError(f"no profile sidecar for {segment_path}; encode first")
    if calib is None:
        calib = wdb_calibrate.get()
    return plan_segment(stats, calib, workload)


# --- measured cluster-key selection (rests on selectivity + compounding + workload) -----
# group-locality: clustering the GROUP key turns hash-grouping into sequential run-grouping.
# That is a rate gain, not a work cut, so it is weighted below a filter row-save. It is the
# one constant here not yet measured end-to-end; treat as calibration (override via calib
# key 'group_locality_factor').
GROUP_LOCALITY_FACTOR = 0.25

_PRED_PAT = (r"([a-z_]+\s+BETWEEN\s+[\d.]+\s+AND\s+[\d.]+"
             r"|[a-z_]+\s+IN\s*\([^)]*\)"
             r"|[a-z_]+\s*(?:>=|<=|<>|!=|>|<|=)\s*(?:DATE\s*'[^']*'|'[^']*'|[\d.]+))")


def derive_workload(queries):
    """Query log -> {col: {'filter': n, 'group': n}}. Pure text policy; the 6th, only
    non-data parameter. `queries` is a list of SQL strings (one statement each)."""
    import re
    wl = {}
    def bump(c, k):
        wl.setdefault(c, {'filter': 0, 'group': 0})[k] += 1
    for sql in queries:
        s = ' '.join(sql.split())
        mw = re.search(r'\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)', s, re.I)
        if mw:
            seen = set()
            for p in re.findall(_PRED_PAT, mw.group(1), re.I):
                c = re.match(r'([a-z_]+)', p.strip(), re.I).group(1).lower()
                if c not in seen:
                    bump(c, 'filter'); seen.add(c)
        mg = re.search(r'\bGROUP\s+BY\b(.+?)(?:\bORDER\b|\bLIMIT\b|$)', s, re.I)
        if mg:
            for c in re.findall(r'[a-z_][a-z_0-9]*', mg.group(1), re.I):
                cl = c.lower()
                if cl[:2] in ('l_', 'o_', 'c_', 'p_', 's_', 'n_', 'r_'):
                    bump(cl, 'group')
    return wl


def score_cluster_keys(candidates, workload, selectivity, coverage,
                       group_locality=GROUP_LOCALITY_FACTOR):
    """Pure objective over MEASURED inputs. Higher score = better cluster key.

      score(K) = sum_C wf(C)*(1-sel(C))   for C sliceable under K   [filter work saved]
               + g * sum_C wg(C)          for C sliceable under K   [group locality]

    candidates  : column names eligible as the single cluster key
    workload    : {col: {'filter': nf, 'group': ng}}
    selectivity : {col: fraction of rows a typical filter on col keeps}  (lower = more selective)
    coverage    : {K: iterable of cols made sliceable by clustering on K (incl. K itself)}
    Returns ranked [(K, score, detail)] high-to-low."""
    out = []
    for K in candidates:
        cov = set(coverage.get(K, {K}))
        f = sum(workload.get(C, {}).get('filter', 0) * (1.0 - float(selectivity.get(C, 1.0)))
                for C in cov)
        g = group_locality * sum(workload.get(C, {}).get('group', 0) for C in cov)
        out.append((K, round(f + g, 4),
                    {'filter_saved': round(f, 4), 'group_saved': round(g, 4),
                     'covers': sorted(cov)}))
    return sorted(out, key=lambda x: x[1], reverse=True)


def _count_where(db, table, where):
    rows, _cols = db.run(f"SELECT COUNT(*) FROM {table} WHERE {where}")
    return float(rows[0][0])


def _measure_selectivity(db, table, queries, want, N):
    """Exact per-column selectivity via WaveDB's OWN executor (no external oracle).
    Averages the log's predicate-groups on each wanted column."""
    import re
    acc = {}
    for sql in queries:
        s = ' '.join(sql.split())
        mw = re.search(r'\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)', s, re.I)
        if not mw:
            continue
        bycol = {}
        for p in re.findall(_PRED_PAT, mw.group(1), re.I):
            c = re.match(r'([a-z_]+)', p.strip(), re.I).group(1).lower()
            bycol.setdefault(c, []).append(p)
        for c, ps in bycol.items():
            if c not in want:
                continue
            try:
                acc.setdefault(c, []).append(_count_where(db, table, ' AND '.join(ps)) / N)
            except Exception:
                pass
    return {c: sum(v) / len(v) for c, v in acc.items() if v}


def _measure_coverage(seg, candidates, cols, sample=400_000):
    """Compounding: clustering by K makes C sliceable iff sorting by K leaves C nearly
    monotone (descents < 0.10). Measured on a sample of the segment's own data."""
    import numpy as np
    N = seg.N
    n = min(sample, N)
    idx = np.linspace(0, N - 1, n).astype(np.int64) if n < N else np.arange(N)
    def num(nm):
        v = seg.values(nm)
        dt = getattr(v, 'dtype', None)
        if dt is not None and dt.kind == 'M':
            return v.view('int64').astype('float64')[idx]
        if dt is not None and dt.kind in 'iuf':
            return v.astype('float64')[idx]
        return seg.codes(nm).astype('float64')[idx]
    allc = sorted(set(candidates) | set(cols))
    samp = {}
    for c in allc:
        try:
            samp[c] = num(c)
        except Exception:
            pass
    cov = {}
    for K in candidates:
        if K not in samp:
            cov[K] = {K}; continue
        o = np.argsort(samp[K], kind='stable')
        s = {K}
        for C in cols:
            if C == K or C not in samp:
                continue
            d = float(np.mean(samp[C][o][1:] < samp[C][o][:-1]))
            if d < 0.10:
                s.add(C)
        cov[K] = s
    return cov


def plan_cluster_key(db, table, queries, calib=None):
    """Integrated, self-contained measured pick. Loads the table's first segment + profile,
    measures selectivity (via db.run -> WaveDB counts) + compounding coverage, derives the
    workload from the query log, and scores. Falls back to the structural prior when the
    workload does not discriminate."""
    import wdb_profile, wdb_calibrate
    from wdb_engine import Segment
    if calib is None:
        calib = wdb_calibrate.get()
    sp = db.cat.segment_paths(table)[0]
    stats = wdb_profile.load_profile(sp)
    if stats is None:
        raise FileNotFoundError(f"no profile for {table}; encode first")
    plans = {nm: plan_column(s, calib) for nm, s in stats.items()}
    cands = [c for c in _cluster_candidates(stats, plans) if not c.endswith('_ptr')]
    if not cands:
        return {'key': None, 'reason': 'no sorted value-identity column', 'ranked': []}
    workload = derive_workload(queries)
    filtered = [c for c, w in workload.items() if w['filter'] and c in stats]
    seg = Segment(sp)
    N = seg.N
    want = set(cands) | set(filtered)
    selectivity = _measure_selectivity(db, table, queries, want, N)
    cov = _measure_coverage(seg, cands, sorted(want))
    g = float(calib.get('group_locality_factor', GROUP_LOCALITY_FACTOR))
    ranked = score_cluster_keys(cands, workload, selectivity, cov, g)
    best, score, _ = ranked[0]
    if score <= 0.0:
        sp_pick = choose_cluster_key(stats, plans, calib)
        sp_pick['reason'] = 'structural prior (workload does not discriminate)'
        sp_pick['ranked'] = ranked
        return sp_pick
    return {'key': best, 'reason': 'measured (selectivity + compounding + workload)',
            'ranked': ranked,
            'selectivity': {c: round(s, 4) for c, s in selectivity.items()}}


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wdb_db import Database
    import wdb_calibrate
    db = Database.open(sys.argv[1]); table = sys.argv[2]
    calib = wdb_calibrate.get()

    # workload / query log: explicit file (;-separated SQL) if given, else the canonical
    # catalog if reachable, else none (structural prior).
    queries = []
    if len(sys.argv) > 3:
        raw = open(sys.argv[3]).read()
        queries = [q.strip() for q in raw.split(';') if q.strip()]
    else:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, os.path.join(here, '..', 'bench'))
            from catalog import QUERIES
            queries = [sql for _, _, sql, _ in QUERIES]
        except Exception:
            queries = []

    if queries:
        pick = plan_cluster_key(db, table, queries, calib=calib)
        print(f"\nMEASURED cluster key for {table}: {pick['key']}  ({pick['reason']})")
        if pick.get('selectivity'):
            sels = ', '.join(f"{c}={s}" for c, s in sorted(pick['selectivity'].items()))
            print(f"  measured selectivity: {sels}")
        for k, sc, d in pick.get('ranked', []):
            print(f"    {k:16s} score={sc:7.3f}  filter={d['filter_saved']:6.3f} "
                  f"group={d['group_saved']:6.3f}  covers={','.join(d['covers'])}")
    else:
        print("\n(no workload provided; structural prior only)")

    for sp in db.cat.segment_paths(table):
        plan = plan_for_segment(sp, calib=calib)
        print(f"\n{sp}")
        print(f"  cluster_key: {plan['cluster']['key']}  ({plan['cluster']['reason']})")
        print(f"  {'column':18s} {'lut':>4s} {'key':>4s} {'slice':>6s} {'group':>6s} {'cbits':>5s}")
        for nm, p in plan['columns'].items():
            print(f"  {nm:18s} {str(p['code_lut'])[:1]:>4s} {str(p['content_key'])[:1]:>4s} "
                  f"{str(p['range_sliceable'])[:1]:>6s} {str(p['groupable'])[:1]:>6s} "
                  f"{p['code_width_bits']:>5d}")
