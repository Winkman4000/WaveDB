"""Single-pass GROUP BY with foldable co-aggregates + one COUNT(DISTINCT) (wdb_groupmix) -- the
ClickBench Q09 shape. The foldables (COUNT(*), SUM, AVG) are bincount reductions; the distinct rides
the wdb_groupdistinct walk; all in one pass. Asserts results match DuckDB (full + deterministic top-K,
SUM(int)->int, AVG->float), the operator actually fires, and out-of-scope shapes are declined (so they
fall through to the correct slower path). Synthetic + tiny; no bench DB."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode
from wdb_db import Database
import wdb_groupmix as gm

_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), f'gmix_{uuid.uuid4().hex[:8]}'); os.makedirs(d, exist_ok=True)
    _CON.execute("CREATE TABLE t AS SELECT i AS id, 'G'||(i%6) AS g, 'S'||(i%13) AS s, "
                 "(i%4) AS adv, CAST((i%50)+1 AS DOUBLE) AS amt FROM range(30000) t(i)")
    _DB = Database.create(d)
    wt = {'BIGINT':'int','INTEGER':'int','VARCHAR':'string','DOUBLE':'float'}
    desc = _CON.execute("DESCRIBE t").fetchall()
    pq = os.path.join(d,'t.parquet'); _CON.execute(f"COPY (SELECT * FROM t ORDER BY id) TO '{pq}' (FORMAT parquet)")
    _DB.cat.add_table('t', [[c[0], wt[c[1]]] for c in desc])
    wdb_encode.encode(pq, os.path.join(d,'t_0.wdb')); _DB.cat.add_segment('t','t_0.wdb')
    return _DB, _CON

def _norm(rows):
    out=[]
    for r in rows:
        row=[]
        for c in r:
            if isinstance(c,float): row.append(round(c,3))
            elif isinstance(c,int) and not isinstance(c,bool): row.append(c)
            else: row.append(None if c is None else str(c))
        out.append(tuple(row))
    return out
def _duck(q):
    _, con = _fixture(); return [tuple(r) for r in con.execute(q).fetchall()]

def _match(q, ordered):
    db, _ = _fixture()
    h0 = gm._HITS
    got = db.run(q)[0]
    assert gm._HITS == h0 + 1, f"groupmix did not fire: {q}"
    g = _norm(got); e = _norm(_duck(q))
    if ordered: assert g == e, f"mismatch {q}\n got {g[:4]}\n exp {e[:4]}"
    else:       assert sorted(g) == sorted(e), f"mismatch {q}\n got {sorted(g)[:4]}\n exp {sorted(e)[:4]}"

def _declined(q):
    db, _ = _fixture()
    h0 = gm._HITS
    db.run(q)                                  # must still be correct via some other path
    assert gm._HITS == h0, f"groupmix fired but should have declined: {q}"


# the Q09 shape: key + foldables + one COUNT(DISTINCT), ORDER BY a foldable
def test_q09_shape_topk():
    _match("SELECT g, COUNT(*) AS c, SUM(adv), AVG(amt), COUNT(DISTINCT s) FROM t "
           "GROUP BY g ORDER BY c DESC, g LIMIT 3", ordered=True)

def test_full_no_limit():
    _match("SELECT g, COUNT(*), SUM(amt), COUNT(DISTINCT s) FROM t GROUP BY g", ordered=False)

def test_order_by_avg():
    _match("SELECT g, AVG(amt) AS a, SUM(adv) AS sa, COUNT(DISTINCT s) AS u FROM t "
           "GROUP BY g ORDER BY a DESC, g", ordered=True)

def test_sum_int_is_int():
    db, _ = _fixture()
    rows = db.run("SELECT g, SUM(adv) AS sa, COUNT(DISTINCT s) FROM t GROUP BY g")[0]
    assert all(isinstance(r[1], int) and not isinstance(r[1], bool) for r in rows)   # SUM(int) -> int

def test_avg_is_float():
    db, _ = _fixture()
    rows = db.run("SELECT g, AVG(amt) AS a, COUNT(DISTINCT s) FROM t GROUP BY g")[0]
    assert all(isinstance(r[1], float) for r in rows)                                # AVG -> float

def test_count_star_and_two_sums():
    _match("SELECT g, COUNT(*) AS c, SUM(adv) AS sa, SUM(amt) AS sm, COUNT(DISTINCT s) AS u "
           "FROM t GROUP BY g ORDER BY c DESC, g", ordered=True)

# out-of-scope shapes must DECLINE (fall through to the correct path), not fire
def test_decline_pure_distinct():        # len-2 {key, COUNT(DISTINCT)} -> wdb_groupdistinct owns it
    _declined("SELECT g, COUNT(DISTINCT s) FROM t GROUP BY g")

def test_decline_minmax():               # MIN/MAX foldable not in v1 scope -> fall through
    _declined("SELECT g, MIN(amt), COUNT(DISTINCT s) FROM t GROUP BY g")

def test_decline_where():                # WHERE is v2 -> fall through
    _declined("SELECT g, COUNT(*), COUNT(DISTINCT s) FROM t WHERE adv > 0 GROUP BY g")

def test_decline_no_distinct():          # no COUNT(DISTINCT) -> cube/fused path, not groupmix
    _declined("SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g")


def test_groupmix_consumes_materialized_sidecar():
    """When the (group, distinct-target) pair is materialized, groupmix reads the sidecar's per-group
    counts instead of walking -- same answer, no walk. Uses its own db so the shared fixture is untouched."""
    import shutil
    con = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), f'gmixsc_{uuid.uuid4().hex[:8]}'); os.makedirs(d, exist_ok=True)
    con.execute("CREATE TABLE t AS SELECT i AS id, 'G'||(i%6) AS g, 'S'||(i%13) AS s, "
                "(i%4) AS adv, CAST((i%50)+1 AS DOUBLE) AS amt FROM range(30000) t(i)")
    db = Database.create(d)
    wt = {'BIGINT':'int','INTEGER':'int','VARCHAR':'string','DOUBLE':'float'}
    desc = con.execute("DESCRIBE t").fetchall()
    pq = os.path.join(d,'t.parquet'); con.execute(f"COPY (SELECT * FROM t ORDER BY id) TO '{pq}' (FORMAT parquet)")
    db.cat.add_table('t', [[c[0], wt[c[1]]] for c in desc])
    wdb_encode.encode(pq, os.path.join(d,'t_0.wdb')); db.cat.add_segment('t','t_0.wdb')
    q = ("SELECT g, COUNT(*) AS c, SUM(adv), AVG(amt), COUNT(DISTINCT s) FROM t "
         "GROUP BY g ORDER BY c DESC, g LIMIT 3")
    exp = _norm([tuple(r) for r in con.execute(q).fetchall()])
    try:
        s0 = gm._SIDECAR_HITS
        assert _norm(db.run(q)[0]) == exp                       # correct via walk
        assert gm._SIDECAR_HITS == s0, "should walk before materialize"
        db.materialize_gd('t', 'g', 's')
        s1 = gm._SIDECAR_HITS
        assert _norm(db.run(q)[0]) == exp                       # correct via sidecar
        assert gm._SIDECAR_HITS == s1 + 1, "groupmix must consume the materialized sidecar"
    finally:
        shutil.rmtree(d, ignore_errors=True)
