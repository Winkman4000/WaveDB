"""Materialized group-wise COUNT(DISTINCT) sidecar (wdb_gdsidecar).

The sidecar precomputes the per-group distinct counts (one exact walk) and stores them indexed by
group code; a matching GROUP BY..COUNT(DISTINCT) query then READS the stored answer instead of
re-walking. These tests assert: (a) build counts are bit-exact vs DuckDB, (b) the served rows are
identical to the live walk and to DuckDB (full + deterministic top-K), (c) the serve path actually
fires (not the walk), (d) save/load round-trips, (e) per-group trim falls back to the walk and stays
correct, (f) the materialized registration survives a catalog reopen, (g) non-value-identity pairs
are declined. All synthetic + tiny; no bench DB."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, numpy as np
from wdb_db import Database
import wdb_groupdistinct as gd
import wdb_gdsidecar as scm

_DB = None; _CON = None; _DIR = None
def _fixture():
    global _DB, _CON, _DIR
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    _DIR = os.path.join(tempfile.gettempdir(), f'gdsc_{uuid.uuid4().hex[:8]}'); os.makedirs(_DIR, exist_ok=True)
    # g: 6 groups; s: value-identity string target; k: sequential -> mode-4 (non-value-identity)
    _CON.execute("CREATE TABLE t AS SELECT i AS id, 'G'||(i%6) AS g, 'S'||(i%13) AS s, "
                 "i AS k FROM range(60000) t(i)")
    _DB = Database.create(_DIR)
    wt = {'BIGINT':'int','INTEGER':'int','VARCHAR':'string','DOUBLE':'float'}
    desc = _CON.execute("DESCRIBE t").fetchall()
    pq = os.path.join(_DIR,'t.parquet'); _CON.execute(f"COPY (SELECT * FROM t ORDER BY id) TO '{pq}' (FORMAT parquet)")
    _DB.cat.add_table('t', [[c[0], wt[c[1]]] for c in desc])
    wdb_encode.encode(pq, os.path.join(_DIR,'t_0.wdb')); _DB.cat.add_segment('t','t_0.wdb')
    return _DB, _CON

def _norm(rows):
    return sorted(tuple((c if isinstance(c,int) and not isinstance(c,bool) else str(c)) for c in r) for r in rows)
def _duck(q):
    _, con = _fixture(); return _norm([tuple(r) for r in con.execute(q).fetchall()])
def _ordered(rows):
    return [tuple((c if isinstance(c,int) and not isinstance(c,bool) else str(c)) for c in r) for r in rows]

Q  = "SELECT g, COUNT(DISTINCT s) FROM t GROUP BY g"
QL = "SELECT g, COUNT(DISTINCT s) AS u FROM t GROUP BY g ORDER BY u DESC, g LIMIT 3"  # 2nd key g => deterministic


def _ensure_materialized():
    db, _ = _fixture()
    if db.cat.gd_entry('t','g','s') is None:
        db.materialize_gd('t','g','s')


def test_build_counts_bitexact_vs_duck():
    db, con = _fixture()
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    s = scm.build(seg, 'g', 's')
    dec = gd._ids(seg, 'g')[2]
    got = {}
    for code in s['present'].tolist():
        lab = dec[code]; lab = lab.decode() if isinstance(lab,(bytes,bytearray)) else str(lab)
        got[lab] = int(s['counts'][code])
    truth = {k:int(v) for k,v in con.execute("SELECT g, COUNT(DISTINCT s) FROM t GROUP BY g").fetchall()}
    assert got == truth, (got, truth)


def test_save_load_roundtrip():
    db, _ = _fixture()
    sp = db.cat.segment_paths('t')[0]; seg = db.open_segment(sp, 't')
    s = scm.build(seg, 'g', 's'); scm.save(sp, s); r = scm.load(sp, 'g', 's')
    assert np.array_equal(r['counts'], s['counts']) and np.array_equal(r['present'], s['present'])
    assert r['meta'] == s['meta']


def test_walk_before_materialize():
    db, _ = _fixture()
    # a pair that is NOT materialized must go through the walk, not the serve path
    assert db.cat.gd_entry('t','s','g') is None
    s0, w0 = scm._SERVE_HITS, gd._HITS
    r = db.run("SELECT s, COUNT(DISTINCT g) FROM t GROUP BY s")[0]
    assert scm._SERVE_HITS == s0 and gd._HITS == w0 + 1                 # walk fired, serve did not
    assert _norm(r) == _duck("SELECT s, COUNT(DISTINCT g) FROM t GROUP BY s")


def test_serve_full_matches_duck_and_fires():
    db, _ = _fixture(); _ensure_materialized()
    s0, w0 = scm._SERVE_HITS, gd._HITS
    r = db.run(Q)[0]
    assert scm._SERVE_HITS == s0 + 1 and gd._HITS == w0               # serve fired, walk did not
    assert _norm(r) == _duck(Q)


def test_serve_topk_matches_duck_and_fires():
    db, _ = _fixture(); _ensure_materialized()
    s0 = scm._SERVE_HITS
    r = db.run(QL)[0]
    assert scm._SERVE_HITS == s0 + 1
    assert _ordered(r) == [tuple((c if isinstance(c,int) else str(c)) for c in row)
                           for row in _fixture()[1].execute(QL).fetchall()]


def test_serve_matches_walk_exactly():
    """The strongest invariant: the served rows equal the live walk's rows, byte for byte and in order.
    (The walk is itself validated vs DuckDB in the suite; matching it pins the sidecar to ground truth
    independent of DuckDB's tie-break.)"""
    db, _ = _fixture(); _ensure_materialized()
    walk = gd.try_groupdistinct(db.open_segment(db.cat.segment_paths('t')[0],'t'),
                                __import__('wdb_db')._parse_sql_cached(QL), None)[0]
    served = db.run(QL)[0]
    assert _ordered(served) == _ordered(walk)


def test_trim_falls_back_to_walk_and_stays_correct():
    db, _ = _fixture(); _ensure_materialized()
    db.gd_trim('t','g','s', ['G2'])                                    # leave G2 to the walk
    s0, w0 = scm._SERVE_HITS, gd._HITS
    r = db.run(Q)[0]
    assert scm._SERVE_HITS == s0 and gd._HITS == w0 + 1               # v1: any trim -> walk (needs all groups)
    assert _norm(r) == _duck(Q)                                       # still exact (walk computes every group)
    db.gd_trim('t','g','s', [])                                       # un-trim
    s0 = scm._SERVE_HITS
    assert _norm(db.run(Q)[0]) == _duck(Q) and scm._SERVE_HITS == s0 + 1   # serves again


def test_inspect_lists_groups_with_counts():
    db, con = _fixture(); _ensure_materialized()
    rows = db.gd_inspect('t','g','s')
    assert rows is not None and len(rows) == 6
    assert all(len(r) == 3 for r in rows)                            # (value, count, is_trimmed)
    counts = sorted(c for _, c, _ in rows)
    truth = sorted(int(v) for _, v in con.execute("SELECT g, COUNT(DISTINCT s) FROM t GROUP BY g").fetchall())
    assert counts == truth
    assert all(t is False for _, _, t in rows)                       # nothing trimmed by default


def test_registration_survives_reopen():
    db, _ = _fixture(); _ensure_materialized()
    db2 = Database.open(_DIR)
    assert db2.cat.gd_entry('t','g','s') == {'excluded': []}
    s0 = scm._SERVE_HITS
    assert _norm(db2.run(Q)[0]) == _duck(Q) and scm._SERVE_HITS == s0 + 1   # serves on a fresh handle


def test_build_declines_non_value_identity():
    db, _ = _fixture()
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    assert scm.build(seg, 'g', 'k') is None                          # k is sequential -> mode-4 target
    try:
        db.materialize_gd('t','g','k'); assert False, "should have raised"
    except ValueError:
        pass


# ── v2: WHERE confined to the group key (filter just removes whole groups) ──────────────────────
QF_NEQ = "SELECT g, COUNT(DISTINCT s) AS u FROM t WHERE g <> 'G2' GROUP BY g"
QF_EQ  = "SELECT g, COUNT(DISTINCT s) AS u FROM t WHERE g = 'G3' GROUP BY g"
QF_IN  = "SELECT g, COUNT(DISTINCT s) AS u FROM t WHERE g IN ('G1','G4') GROUP BY g"
QF_NIN = "SELECT g, COUNT(DISTINCT s) AS u FROM t WHERE g NOT IN ('G0','G5') GROUP BY g"
QF_AND = "SELECT g, COUNT(DISTINCT s) AS u FROM t WHERE g <> 'G2' AND g <> 'G4' GROUP BY g"
QF_OTHER = "SELECT g, COUNT(DISTINCT s) AS u FROM t WHERE s <> 'S0' GROUP BY g"  # filter on TARGET -> must decline


def test_serve_filter_on_group_key_neq():
    db, _ = _fixture(); _ensure_materialized()
    s0, w0 = scm._SERVE_HITS, gd._HITS
    r = db.run(QF_NEQ)[0]
    assert scm._SERVE_HITS == s0 + 1 and gd._HITS == w0          # served from sidecar, no walk
    assert _norm(r) == _duck(QF_NEQ)
    assert all(str(row[0]) != 'G2' for row in r)                 # the excluded group is gone


def test_serve_filter_eq_in_notin_and():
    db, _ = _fixture(); _ensure_materialized()
    for q in (QF_EQ, QF_IN, QF_NIN, QF_AND):
        s0 = scm._SERVE_HITS
        r = db.run(q)[0]
        assert scm._SERVE_HITS == s0 + 1, q                      # each served from the sidecar
        assert _norm(r) == _duck(q), q


def test_filter_on_nongroup_column_declines_serve():
    db, _ = _fixture(); _ensure_materialized()
    s0 = scm._SERVE_HITS
    r = db.run(QF_OTHER)[0]
    assert scm._SERVE_HITS == s0                                 # sidecar must NOT serve (counts would be wrong)
    assert _norm(r) == _duck(QF_OTHER)                           # answer still exact via the fallback path


def test_walk_path_still_rejects_where():
    db, _ = _fixture()
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    tree = __import__('wdb_db')._parse_sql_cached(QF_NEQ)
    assert gd.try_groupdistinct(seg, tree, None) is None         # walk contract unchanged: declines WHERE
    assert gd.detect(seg, tree, None) is None                    # default gate still rejects WHERE
    assert gd.detect(seg, tree, None, _allow_group_filter=True) is not None  # sidecar opt-in lets it through
