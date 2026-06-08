"""Phase 4: BSI exec accounting (footprint) + RAM-budget guard.

The lazy filter-index is workload-driven by construction (only a filtered column
gets built) and bounded by BSI_RAM_BUDGET; past the budget a column simply falls
back to the fused scan with an identical answer. These tests lock both."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb
import wdb_encode, wdb_bsi_exec as BX
from wdb_db import Database

_WT = {'BIGINT': 'int', 'INTEGER': 'int', 'DOUBLE': 'float'}


def _db():
    con = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), f'bsiacct_{uuid.uuid4().hex[:8]}')
    db = Database.create(d)
    con.execute("CREATE TABLE t AS SELECT CAST(i AS BIGINT) id, "
                "CAST(((i%97)+1)*1.5 AS DOUBLE) AS n, "
                "CAST(i%11 AS DOUBLE)*CAST(0.01 AS DOUBLE) AS disc FROM range(20000) t(i)")
    desc = con.execute("DESCRIBE t").fetchall()
    pq = os.path.join(d, 't.parquet')
    con.execute(f"COPY (SELECT * FROM t ORDER BY id) TO '{pq}' (FORMAT parquet)")
    db.cat.add_table('t', [[c[0], 'float' if c[1].startswith('DECIMAL') else _WT[c[1]]] for c in desc])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb')
    return db


def _seg(db):
    return db.open_segment(db.cat.segment_paths('t')[0], 't')


def test_footprint_zero_before_use():
    db = _db()
    assert BX.footprint(_seg(db)) == (0, [])


def test_footprint_after_filter():
    db = _db()
    db.run("SELECT SUM(n) FROM t WHERE disc < 0.03")     # 18% selective -> BSI builds on n
    b, cols = BX.footprint(_seg(db))
    assert 'disc' in cols and b > 0


def test_budget_guard_falls_back_but_correct():
    db = _db()
    ans_bsi = db.run("SELECT SUM(n) FROM t WHERE disc < 0.03")[0]
    old = BX.BSI_RAM_BUDGET
    BX.BSI_RAM_BUDGET = 1                              # 1 byte -> no column can be indexed
    try:
        db2 = _db()
        ans_fb = db2.run("SELECT SUM(n) FROM t WHERE disc < 0.03")[0]
        assert BX.footprint(_seg(db2)) == (0, [])     # nothing built under the budget
    finally:
        BX.BSI_RAM_BUDGET = old
    assert abs(float(ans_bsi[0][0]) - float(ans_fb[0][0])) < 1e-6   # path-independent answer


def test_escalate_knob_routes_and_agrees():
    # default is throughput (non-escalated): BSI engages
    db = _db()
    assert db.escalate is False
    a_thru = db.run("SELECT SUM(n) FROM t WHERE disc < 0.03", escalate=False)[0]
    assert 'disc' in BX.footprint(_seg(db))[1]          # throughput mode built the index
    # escalated (latency): skips BSI entirely -> fused parallel scan, index never built
    db2 = _db()
    a_lat = db2.run("SELECT SUM(n) FROM t WHERE disc < 0.03", escalate=True)[0]
    assert BX.footprint(_seg(db2)) == (0, [])           # escalated never touched the BSI path
    # identical answer either way (path-independent)
    assert abs(float(a_thru[0][0]) - float(a_lat[0][0])) < 1e-6


def test_db_level_escalate_default():
    db = _db(); db.escalate = True                      # operator sets the deployment default
    db.run("SELECT SUM(n) FROM t WHERE disc < 0.03")    # no per-call override -> uses db default
    assert BX.footprint(_seg(db)) == (0, [])            # escalated default -> no BSI
