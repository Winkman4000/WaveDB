"""Non-FK joins resolved by a runtime hash pointer: when a join edge has no stored FK pointer but the parent
key is unique, _build_chain hash-probes the parent key to synthesise the child->parent gather pointer, then
the whole fused chain runs unchanged. Verified vs DuckDB and asserted to take the fast path."""
import sys, os, tempfile, uuid, math, datetime
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, wdb_join
from wdb_db import Database

_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), f'hashjoin_{uuid.uuid4().hex[:8]}'); os.makedirs(d, exist_ok=True)
    # dim has a UNIQUE key 'did'; fact references it (all match). NO fk pointer is created on purpose.
    _CON.execute("CREATE TABLE dim AS SELECT i AS did, 'L'||(i%5) AS label, (i*1.5) AS w FROM range(400) t(i)")
    _CON.execute("CREATE TABLE fact AS SELECT i AS fid, (i*7)%400 AS dimid, ((i%50)+1)*10.0 AS amt FROM range(40000) t(i)")
    _DB = Database.create(d)
    wt = {'BIGINT':'int','INTEGER':'int','VARCHAR':'string','DOUBLE':'float'}
    for tbl, order in (('dim','did'), ('fact','fid')):
        desc = _CON.execute(f"DESCRIBE {tbl}").fetchall()
        sel = ", ".join((f"CAST({c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else c[0]) for c in desc)
        pq = os.path.join(d, f'{tbl}.parquet')
        _CON.execute(f"COPY (SELECT {sel} FROM {tbl} ORDER BY {order}) TO '{pq}' (FORMAT parquet)")
        _DB.cat.add_table(tbl, [[c[0], ('float' if c[1].startswith('DECIMAL') else wt[c[1]])] for c in desc])
        seg = f'{tbl}_0.wdb'; wdb_encode.encode(pq, os.path.join(d, seg)); _DB.cat.add_segment(tbl, seg)
    # second, SMALLER dim (only 0..199) -> some fact rows have no parent -> partial match (must fall back)
    _CON.execute("CREATE TABLE dim2 AS SELECT i AS did, 'M'||(i%4) AS label FROM range(200) t(i)")
    pq = os.path.join(d, 'dim2.parquet'); _CON.execute(f"COPY (SELECT * FROM dim2 ORDER BY did) TO '{pq}' (FORMAT parquet)")
    _DB.cat.add_table('dim2', [['did','int'], ['label','string']])
    wdb_encode.encode(pq, os.path.join(d, 'dim2_0.wdb')); _DB.cat.add_segment('dim2', 'dim2_0.wdb')
    return _DB, _CON

def _norm(rows):
    out = []
    for r in rows:
        out.append(tuple(round(float(c), 3) if isinstance(c, (float, Decimal)) else
                         (c if isinstance(c, int) and not isinstance(c, bool) else str(c)) for c in r))
    return sorted(out)

def _match(q, expect_fast=True):
    db, con = _fixture()
    before = wdb_join._FAST_HITS
    g = _norm(db.run(q)[0]); e = _norm([tuple(r) for r in con.execute(q).fetchall()])
    if expect_fast:
        assert wdb_join._FAST_HITS == before + 1, f"hash-join fast path NOT taken: {q}"
    assert g == e, f"mismatch {q}\n got {g[:3]}\n exp {e[:3]}"

_J = "FROM fact f JOIN dim d ON f.dimid = d.did "

def test_hashjoin_agg_parent_groupkey():
    _match("SELECT d.label, SUM(f.amt) " + _J + "GROUP BY d.label")
def test_hashjoin_count_parent_gather():
    _match("SELECT d.label, COUNT(*) " + _J + "GROUP BY d.label")
def test_hashjoin_parent_column_aggregate():
    _match("SELECT d.label, SUM(d.w) " + _J + "GROUP BY d.label")
def test_hashjoin_where_parent_and_child():
    _match("SELECT d.label, SUM(f.amt) " + _J + "WHERE d.w > 200 AND f.amt > 100 GROUP BY d.label")
def test_hashjoin_whole_table_aggregate():
    _match("SELECT SUM(f.amt) " + _J)
def test_hashjoin_plain_projection_via_chain_pandas():
    # plain projection is agg-only-fast, so it routes through _chain_pandas using the SAME hash pointer
    _match("SELECT f.fid, d.label " + _J + "WHERE f.fid < 6 ORDER BY f.fid", expect_fast=False)
def test_hashjoin_partial_match_falls_back():
    # some fact.dimid (>=200) have no parent in dim2 -> hash pointer is partial -> graceful fallback, still correct
    _match("SELECT d.label, COUNT(*) FROM fact f JOIN dim2 d ON f.dimid = d.did GROUP BY d.label", expect_fast=False)
