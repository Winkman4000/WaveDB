"""Path-coverage hardening: run every supported query SHAPE through db.run and record whether it engaged
the fused/gather FAST path or fell back to the generic executor (wdb_sql / pandas). Each case has a
locked-in expected path. A 'fast' case that regresses to a fallback FAILS (a silent perf cliff -- the bug
this file exists to catch); a 'fallback' case that starts fusing also FAILS (good news -- promote it to
'fast' so the ratchet stays honest). Every case is additionally checked against DuckDB, because a wrong
fast answer is worse than a slow right one.

Path is read from wdb_join._FAST_HITS -- the same signal the bench scoreboard trusts -- so no production
code changes. Run `python3 tests/run.py path_coverage` to print the per-query path report."""
import sys, os, tempfile, uuid, math, datetime, re
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, sqlglot, sqlglot.expressions as E
import wdb_encode, wdb_join, wdb_sql, wdb_bsi_exec, wdb_groupdistinct, wdb_blockstats
from wdb_db import Database

_DB = None; _CON = None
_WT = {'BIGINT': 'int', 'INTEGER': 'int', 'VARCHAR': 'string', 'DATE': 'datetime'}
def _wt(t): return 'float' if t.startswith('DECIMAL') else _WT[t]

def _fixture():
    """TPC-H sf=0.01 star schema (region<-nation<-customer<-orders<-lineitem) with FK pointers, so both
    single-table aggregates and multi-hop joins can be exercised on the fast path."""
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect(); _CON.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=0.01)")
    d = os.path.join(tempfile.gettempdir(), f'pathcov_{uuid.uuid4().hex[:8]}')
    _DB = Database.create(d)
    order = {'region': 'r_regionkey', 'nation': 'n_nationkey', 'customer': 'c_custkey',
             'orders': 'o_orderkey', 'lineitem': 'l_orderkey'}
    for tbl in ('region', 'nation', 'customer', 'orders', 'lineitem'):
        desc = _CON.execute(f"DESCRIBE {tbl}").fetchall()
        sel = ", ".join((f"CAST({c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else c[0]) for c in desc)
        pq = os.path.join(d, f'{tbl}.parquet')
        _CON.execute(f"COPY (SELECT {sel} FROM {tbl} ORDER BY {order[tbl]}) TO '{pq}' (FORMAT parquet)")
        _DB.cat.add_table(tbl, [[c[0], _wt(c[1])] for c in desc])
        seg = f'{tbl}_0.wdb'; wdb_encode.encode(pq, os.path.join(d, seg)); _DB.cat.add_segment(tbl, seg)
    _DB.create_fk_pointer('orders', 'o_custkey', 'customer', 'c_custkey')
    _DB.create_fk_pointer('nation', 'n_regionkey', 'region', 'r_regionkey')
    _DB.create_fk_pointer('customer', 'c_nationkey', 'nation', 'n_nationkey')
    _DB.create_fk_pointer('lineitem', 'l_orderkey', 'orders', 'o_orderkey')
    return _DB, _CON

# (label, expected_path, sql).  expected_path: 'fast' = fused/gather kernel MUST engage;
# 'fallback' = an aggregate/join/distinct that does NOT currently fuse (generic executor) -- the
# candidates to fix; 'rows' = a plain row projection that legitimately uses the row path (never fuses).
CASES = [
    # ---- single-table aggregate / group / distinct: must fuse ----
    ('st_whole_sum',       'fast', "SELECT SUM(l_extendedprice) FROM lineitem"),
    ('st_count_star',      'fast', "SELECT COUNT(*) FROM lineitem"),
    ('st_group_sum',       'fast', "SELECT l_returnflag, SUM(l_quantity) FROM lineitem GROUP BY l_returnflag"),
    ('st_group_multi_agg', 'fast', "SELECT l_returnflag, COUNT(*), SUM(l_quantity), AVG(l_discount), "
                                   "MIN(l_tax), MAX(l_tax) FROM lineitem GROUP BY l_returnflag"),
    ('st_two_col_group',   'fast', "SELECT l_returnflag, l_linestatus, COUNT(*) FROM lineitem "
                                   "GROUP BY l_returnflag, l_linestatus"),
    ('st_where_sum',       'fast', "SELECT SUM(l_extendedprice) FROM lineitem WHERE l_quantity > 25"),
    ('st_arith_agg',       'fast', "SELECT l_returnflag, SUM(l_extendedprice * l_discount) FROM lineitem "
                                   "GROUP BY l_returnflag"),
    ('st_pred_count',      'fast', "SELECT COUNT(*) FROM lineitem WHERE l_quantity > 30"),
    ('st_count_distinct',  'fast', "SELECT COUNT(DISTINCT l_shipmode) FROM lineitem"),
    ('st_grouped_cdist',   'fast', "SELECT l_returnflag, COUNT(DISTINCT l_shipmode) FROM lineitem "
                                   "GROUP BY l_returnflag"),                                       # #23
    ('st_cdist_highcard',  'fast', "SELECT l_returnflag, COUNT(DISTINCT l_partkey) FROM lineitem "
                                   "GROUP BY l_returnflag"),
    ('st_cdist_where',     'fast', "SELECT l_shipmode, COUNT(DISTINCT l_returnflag) FROM lineitem "
                                   "WHERE l_quantity > 25 GROUP BY l_shipmode"),
    ('st_order_limit',     'fast', "SELECT l_partkey, SUM(l_quantity) AS s FROM lineitem "
                                   "GROUP BY l_partkey ORDER BY s DESC LIMIT 10"),                 # #24
    # ---- BSI filter-aggregate path: selective off-key numeric/range predicates ----
    ('bsi_discount_btw',   'bsi',  "SELECT SUM(l_extendedprice) FROM lineitem "
                                   "WHERE l_discount BETWEEN 0.05 AND 0.07"),
    ('bsi_q6_multi',       'bsi',  "SELECT SUM(l_extendedprice * l_discount) FROM lineitem "
                                   "WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' "
                                   "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24"),
    # ---- FK-pointer joins: must gather (not hash / pandas) ----
    ('jn_group_sum',       'fast', "SELECT c.c_mktsegment, SUM(o.o_totalprice) FROM orders o "
                                   "JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment"),
    ('jn_multi_hop',       'fast', "SELECT r.r_name, COUNT(*) FROM customer c "
                                   "JOIN nation n ON c.c_nationkey=n.n_nationkey "
                                   "JOIN region r ON n.n_regionkey=r.r_regionkey GROUP BY r.r_name"),
    ('jn_fact_to_region',  'fast', "SELECT r.r_name, SUM(l.l_extendedprice) FROM lineitem l "
                                   "JOIN orders o ON l.l_orderkey=o.o_orderkey "
                                   "JOIN customer c ON o.o_custkey=c.c_custkey "
                                   "JOIN nation n ON c.c_nationkey=n.n_nationkey "
                                   "JOIN region r ON n.n_regionkey=r.r_regionkey GROUP BY r.r_name"),
    # ---- documented non-fused shapes (locked in; promote if one starts fusing) ----
    ('rows_projection',    'rows', "SELECT l_orderkey, l_quantity FROM lineitem LIMIT 5"),
]

_TS = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(\.\d+)?$')
def _cell(c):
    if isinstance(c, Decimal): c = float(c)
    if isinstance(c, datetime.datetime): c = c.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(c, datetime.date): return c.strftime('%Y-%m-%d')
    if isinstance(c, str):
        m = _TS.match(c)
        if m: c = m.group(1)
        if len(c) == 19 and c.endswith(' 00:00:00'): c = c[:10]
    return c
def _norm(rows): return [tuple(_cell(c) for c in r) for r in rows]
def _eq(g, e):
    if len(g) != len(e): return False
    for ra, rb in zip(sorted(g, key=repr), sorted(e, key=repr)):   # group key leads the repr -> aligns rows
        if len(ra) != len(rb): return False
        for a, b in zip(ra, rb):
            if isinstance(a, float) or isinstance(b, float):
                if a is None or b is None:
                    if a is not b: return False
                elif not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6): return False
            elif a != b: return False
    return True

def _is_fusion_candidate(tree):
    """True if the query has joins / GROUP BY / DISTINCT / an aggregate -- i.e. it COULD take the fused
    path. A non-candidate that doesn't fuse is 'rows' (expected), a candidate that doesn't is 'fallback'."""
    return (bool(tree.args.get('joins')) or tree.args.get('group') is not None
            or tree.args.get('distinct') is not None
            or any(wdb_sql._agg_kind(x) for x in tree.expressions))

def _classify(db, q):
    """Run q and report ('bsi' | 'fast' | 'fallback' | 'rows', result_rows)."""
    tree = sqlglot.parse_one(q, read='duckdb')
    cand = _is_fusion_candidate(tree)
    fb = wdb_join._FAST_HITS; bb = wdb_bsi_exec._BSI_HITS; gd = wdb_groupdistinct._HITS
    import wdb_smallk; sk = wdb_smallk._HITS
    bs = wdb_blockstats._HITS
    rows = db.run(q)[0]
    if wdb_bsi_exec._BSI_HITS > bb: return 'bsi', rows
    if wdb_blockstats._HITS > bs: return 'fast', rows      # per-block stats: aggregates, no row data
    if wdb_groupdistinct._HITS > gd: return 'fast', rows   # group-wise COUNT(DISTINCT) code-hash kernel
    if wdb_join._FAST_HITS > fb: return 'fast', rows
    import wdb_smallk
    if wdb_smallk._HITS > sk: return 'fast', rows   # narrow-key fused composite board
    return ('fallback' if cand else 'rows'), rows

def _evaluate():
    """Run every case once; return (report_rows, failures). report_rows: (label, expected, got, ans)."""
    db, con = _fixture()
    report, fails = [], []
    for label, expected, q in CASES:
        got, g = _classify(db, q)
        try:
            e = con.execute(q).fetchall()
            ans = 'OK' if _eq(_norm(g), _norm([tuple(r) for r in e])) else 'WRONG'
        except Exception as ex:
            ans = f'ERR:{ex}'
        report.append((label, expected, got, ans))
        if ans != 'OK':
            fails.append(f"{label}: answer {ans} vs DuckDB  ({q})")
        elif got != expected:
            perf = ('fast', 'bsi')
            if expected in perf and got not in perf:
                fails.append(f"{label}: REGRESSION -- expected {expected!r} path, got {got!r}. Investigate "
                             f"why it stopped using the accelerated path.  ({q})")
            elif expected not in perf and got in perf:
                fails.append(f"{label}: now takes the {got!r} path (expected {expected!r}). Good news -- "
                             f"promote it in CASES so the ratchet stays honest.  ({q})")
            else:
                fails.append(f"{label}: path changed {expected!r} -> {got!r}; update CASES if intended.  ({q})")
    return report, fails

def _print_report(report):
    counts = {}
    for _, _, got, _a in report: counts[got] = counts.get(got, 0) + 1
    print("\n  path coverage  (" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "):")
    for label, exp, got, ans in report:
        flag = '' if (exp == got and ans == 'OK') else '   <-- CHECK'
        print(f"    {label:20s} expect={exp:8s} got={got:8s} {ans}{flag}")

def test_path_coverage():
    """Every supported shape engages its expected path AND returns DuckDB-correct rows."""
    report, fails = _evaluate()
    _print_report(report)
    assert not fails, "PATH-COVERAGE ISSUES:\n  " + "\n  ".join(fails)

def test_fast_cases_are_correct():
    """Defense in depth: the fast-path cases must match DuckDB (a wrong fast answer beats no test)."""
    db, con = _fixture()
    for label, expected, q in CASES:
        if expected != 'fast': continue
        g = _norm(db.run(q)[0]); e = _norm([tuple(r) for r in con.execute(q).fetchall()])
        assert _eq(g, e), f"{label}: fast-path answer != DuckDB ({q})"
