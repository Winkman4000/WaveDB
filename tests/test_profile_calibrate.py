"""Tests for the two sensory organs:
  wdb_profile   -- per-column stats (V, H, uniq, sorted, code_bits); piggybacks encode.
  wdb_calibrate -- host analog constants (seq/rand rate, locality penalty, lut_cliff).
Key correctness case: mode-4 (affine/seq) cardinality must come from VALUES, not the
positional codes -- else a sequential id falsely reads as the table's row count."""
import sys, os, tempfile, uuid, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, wdb_profile, wdb_calibrate
from wdb_engine import Segment

_SEG = None
def _seg():
    global _SEG
    if _SEG is not None: return _SEG
    con = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), f'prof_{uuid.uuid4().hex[:8]}'); os.makedirs(d, exist_ok=True)
    con.execute("CREATE TABLE t AS SELECT i AS id, 'G'||(i%6) AS g, (i%9) AS k, "
                "((i%50)+1)*1.5 AS amt FROM range(30000) t(i)")
    pq = os.path.join(d, 't.parquet')
    con.execute(f"COPY (SELECT * FROM t ORDER BY id) TO '{pq}' (FORMAT parquet)")
    path = os.path.join(d, 't_0.wdb'); wdb_encode.encode(pq, path)
    _SEG = (Segment(path), path); return _SEG


def test_profile_lowcard_string():
    seg, _ = _seg(); st = wdb_profile.profile_col(seg, 'g')
    assert st['V'] == 6, st
    assert st['code_bits'] == 3
    assert st['uniq'] < 0.01

def test_profile_mode4_truecard():            # the bug this organ had to fix
    seg, _ = _seg()
    assert seg.cols['k']['mode'] == 4, "expected k=i%9 to encode mode-4"
    st = wdb_profile.profile_col(seg, 'k')
    assert st['V'] == 9, f"mode-4 true cardinality should be 9, got {st['V']}"
    assert st['uniq'] < 0.01

def test_profile_nearunique_id():             # sequential id -> content-address eligible
    seg, _ = _seg(); st = wdb_profile.profile_col(seg, 'id')
    assert st['uniq'] > 0.99, st

def test_profile_entropy_uniform():
    seg, _ = _seg(); st = wdb_profile.profile_col(seg, 'g')
    assert abs(st['H_bits'] - math.log2(6)) < 0.05    # 6 uniform values

def test_profile_sidecar_written_by_encode():
    seg, path = _seg()
    assert os.path.exists(wdb_profile.stats_path(path)), "encode hook must auto-write sidecar"
    loaded = wdb_profile.load_profile(path)
    assert loaded is not None and loaded['g']['V'] == 6

def test_profile_save_load_roundtrip():
    seg, path = _seg(); st = wdb_profile.profile_segment(seg)
    wdb_profile.save_profile(path, st)
    assert wdb_profile.load_profile(path) == st


def test_calibrate_structure():               # fast: skip the heavy cliff scan
    d = wdb_calibrate.calibrate(n_elem=300_000, measure_cliff=False)
    for kk in ('seq_rate_ns', 'rand_rate_ns', 'locality_penalty', 'lut_cliff', 'engine', 'n_cores'):
        assert kk in d, (kk, d)
    assert d['seq_rate_ns'] > 0 and d['lut_cliff'] > 0

def test_calibrate_save_load_get():
    d = {'seq_rate_ns': 0.6, 'rand_rate_ns': 2.5, 'locality_penalty': 4.2,
         'lut_cliff': 65536, 'n_cores': 8, 'engine': 'test'}
    p = os.path.join(tempfile.gettempdir(), f'calib_{uuid.uuid4().hex[:8]}.json')
    wdb_calibrate.save_calibration(d, p)
    assert wdb_calibrate.load_calibration(p) == d
    assert wdb_calibrate.get(p) == d          # lazy load hits the cache, no recompute
    os.remove(p)
