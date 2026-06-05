"""Tests for wdb_plan -- pure policy over measured stats (no data touched).
Confirms each lever gates on the right measured threshold, and that mode-4 columns
are content-keys but NOT groupable (must match the engine's fused-path gate)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import wdb_plan as P

CALIB = {'lut_cliff': 65536, 'locality_penalty': 4.17, 'n_cores': 16}

def _s(V, uniq, sorted_, mode, dt, code_bits, H=0.0):
    return {'V': V, 'H_bits': H, 'uniq': uniq, 'sorted': sorted_,
            'code_bits': code_bits, 'mode': mode, 'dt': dt}

STATS = {
    'returnflag': _s(3, 0.0, True, 0, 1, 2),
    'quantity':   _s(50, 0.0, True, 0, 2, 6),
    'partkey':    _s(200000, 0.033, True, 2, 0, 18),
    'price':      _s(933900, 0.156, True, 0, 2, 20),
    'shipdate':   _s(2526, 0.0004, True, 0, 3, 12),
    'orderkey':   _s(1500000, 0.9999, True, 4, 0, 21),
    'comment':    _s(3610733, 0.60, None, 5, 1, 22),
}

def test_code_lut_gated_by_cliff():
    pl = P.plan_segment(STATS, CALIB)['columns']
    assert pl['returnflag']['code_lut'] and pl['quantity']['code_lut'] and pl['shipdate']['code_lut']
    assert not pl['partkey']['code_lut'] and not pl['price']['code_lut']   # V > lut_cliff

def test_content_key_by_uniqueness():
    pl = P.plan_segment(STATS, CALIB)['columns']
    assert pl['orderkey']['content_key']                 # uniq ~1.0
    assert not pl['price']['content_key'] and not pl['comment']['content_key']

def test_mode4_not_groupable_but_keyable():
    pl = P.plan_segment(STATS, CALIB)['columns']
    assert pl['orderkey']['content_key'] and not pl['orderkey']['groupable']

def test_groupable_is_value_identity():
    pl = P.plan_segment(STATS, CALIB)['columns']
    assert pl['quantity']['groupable'] and pl['comment']['groupable']  # mode 0 / 5
    assert not pl['orderkey']['groupable']                              # mode 4

def test_range_sliceable_follows_sorted():
    pl = P.plan_segment(STATS, CALIB)['columns']
    assert pl['shipdate']['range_sliceable'] and not pl['comment']['range_sliceable']

def test_code_width_is_entropy_minimal():
    pl = P.plan_segment(STATS, CALIB)['columns']
    assert pl['returnflag']['code_width_bits'] == 2 and pl['quantity']['code_width_bits'] == 6

def test_cluster_structural_prefers_datetime():
    c = P.plan_segment(STATS, CALIB)['cluster']
    assert c['key'] == 'shipdate'                        # dt==3 wins the structural prior

def test_cluster_workload_overrides_prior():
    wl = [{'col': 'quantity', 'kind': 'range', 'freq': 100},
          {'col': 'shipdate', 'kind': 'range', 'freq': 5}]
    c = P.plan_segment(STATS, CALIB, workload=wl)['cluster']
    assert c['key'] == 'quantity' and c['reason'] == 'workload-weighted'

def test_cluster_none_when_no_candidate():
    only_unsorted = {'x': _s(10, 0.0, None, 0, 0, 4)}
    c = P.plan_segment(only_unsorted, CALIB)['cluster']
    assert c['key'] is None
