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


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wdb_db import Database
    db = Database.open(sys.argv[1]); table = sys.argv[2]
    import wdb_calibrate
    calib = wdb_calibrate.get()
    for sp in db.cat.segment_paths(table):
        plan = plan_for_segment(sp, calib=calib)
        print(f"\n{sp}")
        print(f"  cluster_key: {plan['cluster']['key']}  ({plan['cluster']['reason']})")
        print(f"  {'column':18s} {'lut':>4s} {'key':>4s} {'slice':>6s} {'group':>6s} {'cbits':>5s}")
        for nm, p in plan['columns'].items():
            print(f"  {nm:18s} {str(p['code_lut'])[:1]:>4s} {str(p['content_key'])[:1]:>4s} "
                  f"{str(p['range_sliceable'])[:1]:>6s} {str(p['groupable'])[:1]:>6s} "
                  f"{p['code_width_bits']:>5d}")
