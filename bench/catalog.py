"""Canonical catalog of the query shapes WaveDB supports. SINGLE SOURCE OF TRUTH:
- bench/query_matrix.py imports QUERIES from here (no second copy)
- docs/queries.md is generated from here (the running list)
- workload() derives the per-column filter/group frequency the planner needs to choose a cluster key

Each entry: (category, name, sql, exercises).  Add a shape here and it shows up everywhere.
"""
QUERIES = [
 ("agg",      "whole COUNT(*)",            "SELECT COUNT(*) FROM lineitem", "row count, zero column bits"),
 ("agg",      "whole SUM",                 "SELECT SUM(l_extendedprice) FROM lineitem", "single-column reduction"),
 ("agg",      "whole multi-agg",           "SELECT COUNT(*),SUM(l_extendedprice),AVG(l_discount),MIN(l_quantity),MAX(l_quantity) FROM lineitem", "5 aggregates, one pass"),
 ("group",    "GROUP BY K3 count",         "SELECT l_returnflag,COUNT(*) FROM lineitem GROUP BY l_returnflag", "low-card dense tally"),
 ("group",    "GROUP BY K3 sum",           "SELECT l_returnflag,SUM(l_extendedprice) FROM lineitem GROUP BY l_returnflag", "low-card grouped reduction"),
 ("group",    "GROUP BY K7 avg",           "SELECT l_shipmode,AVG(l_quantity) FROM lineitem GROUP BY l_shipmode", "avg = sum/count per group"),
 ("group",    "GROUP BY 2-col (Q1)",       "SELECT l_returnflag,l_linestatus,COUNT(*),SUM(l_quantity),AVG(l_extendedprice) FROM lineitem GROUP BY l_returnflag,l_linestatus", "composite key, TPC-H Q1"),
 ("group",    "GROUP BY datetime K2.5k",   "SELECT l_shipdate,COUNT(*) FROM lineitem GROUP BY l_shipdate", "datetime grouping"),
 ("group",    "GROUP BY high-card K200k",  "SELECT l_partkey,SUM(l_quantity) FROM lineitem GROUP BY l_partkey", "hash-factorise high card"),
 ("group",    "GROUP BY vhigh-card K1.5M", "SELECT l_orderkey,COUNT(*) FROM lineitem GROUP BY l_orderkey", "near-unique grouping"),
 ("filter",   "WHERE numeric >",           "SELECT COUNT(*) FROM lineitem WHERE l_quantity > 30", "scalar predicate count"),
 ("filter",   "WHERE BETWEEN + agg",       "SELECT SUM(l_extendedprice) FROM lineitem WHERE l_discount BETWEEN 0.05 AND 0.07", "range predicate"),
 ("filter",   "WHERE date-range (Q6)",     "SELECT SUM(l_extendedprice*l_discount) FROM lineitem WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24", "TPC-H Q6, multi-predicate + arithmetic"),
 ("filter",   "WHERE string =",            "SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'", "dictionary-code equality"),
 ("filter",   "WHERE IN (3)",              "SELECT COUNT(*) FROM lineitem WHERE l_shipmode IN ('AIR','RAIL','SHIP')", "code set membership"),
 ("filter",   "WHERE AND/OR",              "SELECT SUM(l_quantity) FROM lineitem WHERE l_quantity > 30 AND (l_returnflag='R' OR l_linestatus='F')", "boolean predicate tree"),
 ("filter",   "WHERE + GROUP BY",          "SELECT l_returnflag,SUM(l_extendedprice) FROM lineitem WHERE l_quantity > 25 GROUP BY l_returnflag", "filter then group"),
 ("distinct", "DISTINCT 1-col",            "SELECT DISTINCT l_returnflag FROM lineitem", "distinct = dictionary"),
 ("distinct", "DISTINCT 2-col",            "SELECT DISTINCT l_returnflag,l_linestatus FROM lineitem", "composite distinct"),
 ("distinct", "DISTINCT high-card",        "SELECT DISTINCT l_partkey FROM lineitem", "high-card distinct"),
 ("distinct", "COUNT(DISTINCT) low",       "SELECT COUNT(DISTINCT l_shipmode) FROM lineitem", "distinct count"),
 ("distinct", "COUNT(DISTINCT) high",      "SELECT COUNT(DISTINCT l_partkey) FROM lineitem", "high-card distinct count"),
 ("distinct", "grouped COUNT(DISTINCT)",   "SELECT l_returnflag,COUNT(DISTINCT l_shipmode) FROM lineitem GROUP BY l_returnflag", "distinct count per group"),
 ("order",    "ORDER BY + LIMIT",          "SELECT l_partkey,SUM(l_quantity) s FROM lineitem GROUP BY l_partkey ORDER BY s DESC,l_partkey LIMIT 10", "top-K"),
 ("order",    "HAVING",                    "SELECT l_shipmode,COUNT(*) c FROM lineitem GROUP BY l_shipmode HAVING COUNT(*) > 800000", "group filter"),
 ("join",     "JOIN group parent-key",     "SELECT c.c_mktsegment,COUNT(*),SUM(o.o_totalprice) FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment", "FK gather, group on parent dim"),
 ("join",     "JOIN group child-key",      "SELECT l.l_returnflag,SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY l.l_returnflag", "FK gather, group on child"),
 ("join",     "JOIN group parent-date",    "SELECT o.o_orderdate,SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY o.o_orderdate", "FK gather, high-card parent group"),
 ("join",     "JOIN + WHERE",              "SELECT o.o_orderpriority,SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey WHERE l.l_quantity > 30 GROUP BY o.o_orderpriority", "filtered join"),
 ("join",     "3-table JOIN",              "SELECT c.c_mktsegment,SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment", "two-hop FK chain"),
]


import re
def workload():
    """Per-column filter/group frequency across the catalog -- the planner's 6th parameter
    (the one term that cannot be read from the data). col -> {'filter': n, 'group': n}."""
    wl = {}
    for _cat, _name, sql, _ex in QUERIES:
        gm = re.search(r'GROUP BY (.+?)(?: ORDER| HAVING| LIMIT|$)', sql)
        gcols = set(re.findall(r'[loc]?_?[a-z]+_[a-z_]+', gm.group(1))) if gm else set()
        wm = re.search(r'WHERE (.+?)(?: GROUP| ORDER| LIMIT|$)', sql)
        wcols = set(re.findall(r'[loc]_[a-z_]+', wm.group(1))) if wm else set()
        for c in gcols:
            c = c.split('.')[-1]
            wl.setdefault(c, {'filter': 0, 'group': 0})['group'] += 1
        for c in wcols:
            c = c.split('.')[-1]
            wl.setdefault(c, {'filter': 0, 'group': 0})['filter'] += 1
    return wl


if __name__ == '__main__':
    wl = workload()
    print("workload signal (per-column filter/group frequency across the catalog):")
    for c, d in sorted(wl.items(), key=lambda kv: -(kv[1]['filter'] + kv[1]['group'])):
        print(f"  {c:18s} filter={d['filter']}  group={d['group']}")
