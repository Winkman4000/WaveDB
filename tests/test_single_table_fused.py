"""Single-table aggregates route through the SAME fused engine as joins (a 0-join chain). Verified vs DuckDB,
asserted to take the fast path, and including the empty-table edge cases (no-GROUP-BY aggregate over 0 rows
must still emit one grand-total row: COUNT=0, SUM/MIN/MAX=NULL)."""
import sys, os, tempfile, uuid, math
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, wdb_join
from wdb_db import Database

_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), f'stf_{uuid.uuid4().hex[:8]}'); os.makedirs(d, exist_ok=True)
    _CON.execute("CREATE TABLE t AS SELECT i AS id, 'G'||(i%6) AS g, 'S'||(i%4) AS s, (i%9) AS k, ((i%50)+1)*1.5 AS amt FROM range(30000) t(i)")
    _DB = Database.create(d)
    wt = {'BIGINT':'int','INTEGER':'int','VARCHAR':'string','DOUBLE':'float'}
    desc = _CON.execute("DESCRIBE t").fetchall()
    sel = ", ".join((f"CAST({c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else c[0]) for c in desc)
    pq = os.path.join(d, 't.parquet'); _CON.execute(f"COPY (SELECT {sel} FROM t ORDER BY id) TO '{pq}' (FORMAT parquet)")
    _DB.cat.add_table('t', [[c[0], ('float' if c[1].startswith('DECIMAL') else wt[c[1]])] for c in desc])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON

def _norm(rows):
    return sorted(tuple(round(float(c), 3) if isinstance(c, (float, Decimal)) else
                        (c if isinstance(c, int) and not isinstance(c, bool) else (None if c is None else str(c)))
                        for c in r) for r in rows)

def _match(q, expect_fast=True):
    db, con = _fixture()
    before = wdb_join._FAST_HITS
    g = _norm(db.run(q)[0]); e = _norm([tuple(r) for r in con.execute(q).fetchall()])
    if expect_fast:
        assert wdb_join._FAST_HITS == before + 1, f"single-table fast path NOT taken: {q}"
    assert g == e, f"mismatch {q}\n got {g[:4]}\n exp {e[:4]}"

def test_st_group_sum():        _match("SELECT g, SUM(amt) FROM t GROUP BY g")
def test_st_group_multi_agg():  _match("SELECT g, COUNT(*), SUM(amt), AVG(amt), MIN(amt), MAX(amt) FROM t GROUP BY g")
def test_st_where_sum():        _match("SELECT SUM(amt) FROM t WHERE k > 4 AND amt < 50")
def test_st_two_col_group():    _match("SELECT g, s, COUNT(*) FROM t GROUP BY g, s")
def test_st_whole_table():      _match("SELECT COUNT(*), SUM(amt) FROM t")
def test_st_arith_in_agg():     _match("SELECT g, SUM(amt * k) FROM t GROUP BY g")

# empty-table edge cases (route through the fused path; no-group aggregate still emits one row)
def test_st_empty_count():
    db, con = _fixture()
    assert db.run("SELECT COUNT(*) FROM t WHERE k > 999")[0] == [(0,)]
def test_st_empty_sum_null():
    db, con = _fixture()
    assert db.run("SELECT SUM(amt) FROM t WHERE k > 999")[0] == [(None,)]
def test_st_empty_group_is_empty():
    db, con = _fixture()
    assert db.run("SELECT g, COUNT(*) FROM t WHERE k > 999 GROUP BY g")[0] == []

# DISTINCT (fused == GROUP BY over value-identity cols; mode-4 col k falls back to wdb_sql dedup)
def test_st_distinct_two_cols():    _match("SELECT DISTINCT g, s FROM t")
def test_st_distinct_one_col():     _match("SELECT DISTINCT g FROM t")
def test_st_distinct_where():       _match("SELECT DISTINCT g FROM t WHERE k > 4")
def test_st_distinct_order_limit(): _match("SELECT DISTINCT s FROM t ORDER BY s LIMIT 2", expect_fast=True)
def test_st_distinct_mode4_fallback(): _match("SELECT DISTINCT k FROM t", expect_fast=False)  # k is mode-4 -> wdb_sql dedup

# COUNT(DISTINCT) (single fused; grouped via wdb_sql)
def test_st_count_distinct():       _match("SELECT COUNT(DISTINCT g) FROM t")
def test_st_count_distinct_where(): _match("SELECT COUNT(DISTINCT g) FROM t WHERE k > 3")
def test_st_count_distinct_mode4_fallback(): _match("SELECT COUNT(DISTINCT id) FROM t", expect_fast=False)  # id sequential -> mode-4 -> wdb_sql
def test_st_grouped_count_distinct(): _match("SELECT g, COUNT(DISTINCT s) FROM t GROUP BY g", expect_fast=False)
def test_st_grouped_count_distinct_mixed(): _match("SELECT g, COUNT(*), COUNT(DISTINCT s) FROM t GROUP BY g", expect_fast=False)


# ── bounded top-K: ORDER BY <projected aggregate/COUNT> [DESC] LIMIT k over a GROUP BY ────────────
# _topk_prefilter shrinks `present` to a provable superset of the winners before row assembly; the
# unchanged _apply_order over that subset is the source of truth for exact ordering, so results match
# DuckDB. These assert BOTH correctness and that the prefilter actually engages (_TOPK_HITS bumps),
# except the skip case where lim >= #groups so pruning cannot help.
def _match_topk(q, expect_topk=True):
    db, con = _fixture()
    f0 = wdb_join._FAST_HITS; t0 = wdb_join._TOPK_HITS
    g = _norm(db.run(q)[0]); e = _norm([tuple(r) for r in con.execute(q).fetchall()])
    assert wdb_join._FAST_HITS == f0 + 1, f"single-table fast path NOT taken: {q}"
    assert (wdb_join._TOPK_HITS == t0 + 1) == expect_topk, \
        f"top-K prefilter {'did not engage' if expect_topk else 'engaged unexpectedly'}: {q}"
    assert g == e, f"mismatch {q}\n got {g[:4]}\n exp {e[:4]}"

def test_st_topk_sum_desc():   _match_topk("SELECT g, SUM(amt) AS s FROM t GROUP BY g ORDER BY s DESC, g LIMIT 3")
def test_st_topk_sum_asc():    _match_topk("SELECT g, SUM(amt) AS s FROM t GROUP BY g ORDER BY s ASC, g LIMIT 3")
def test_st_topk_count_ties(): _match_topk("SELECT g, COUNT(*) AS c FROM t GROUP BY g ORDER BY c DESC, g LIMIT 3")  # all groups tie on count -> tie-break by g; superset keeps all ties
def test_st_topk_k1():         _match_topk("SELECT g, SUM(amt) AS m FROM t GROUP BY g ORDER BY m DESC, g LIMIT 1")
def test_st_topk_unaliased():  _match_topk("SELECT g, SUM(amt) FROM t GROUP BY g ORDER BY SUM(amt) DESC, g LIMIT 3")
def test_st_topk_skip_lim_ge_n(): _match_topk("SELECT g, SUM(amt) AS s FROM t GROUP BY g ORDER BY s DESC LIMIT 10", expect_topk=False)  # lim(10) >= 6 groups -> no prune
