# WaveDB query catalog — the running list

Every query shape the engine supports today. **Generated from `bench/catalog.py`** (single source of truth — also drives `bench/query_matrix.py` and the planner's workload signal). Don't edit this file by hand; add a shape to the catalog and regenerate.

**30 shapes across 6 categories.** Measured correctness + speed vs DuckDB live in `examples/query_matrix.md`. Multi-segment variants of every shape are exercised by `tests/test_matrix.py`.


## Whole-table aggregates

| query | what it exercises |
|---|---|
| `whole COUNT(*)` | row count, zero column bits |
| `whole SUM` | single-column reduction |
| `whole multi-agg` | 5 aggregates, one pass |

<details><summary>SQL</summary>

- **whole COUNT(*)** — `SELECT COUNT(*) FROM lineitem`
- **whole SUM** — `SELECT SUM(l_extendedprice) FROM lineitem`
- **whole multi-agg** — `SELECT COUNT(*),SUM(l_extendedprice),AVG(l_discount),MIN(l_quantity),MAX(l_quantity) FROM lineitem`

</details>

## GROUP BY

| query | what it exercises |
|---|---|
| `GROUP BY K3 count` | low-card dense tally |
| `GROUP BY K3 sum` | low-card grouped reduction |
| `GROUP BY K7 avg` | avg = sum/count per group |
| `GROUP BY 2-col (Q1)` | composite key, TPC-H Q1 |
| `GROUP BY datetime K2.5k` | datetime grouping |
| `GROUP BY high-card K200k` | hash-factorise high card |
| `GROUP BY vhigh-card K1.5M` | near-unique grouping |

<details><summary>SQL</summary>

- **GROUP BY K3 count** — `SELECT l_returnflag,COUNT(*) FROM lineitem GROUP BY l_returnflag`
- **GROUP BY K3 sum** — `SELECT l_returnflag,SUM(l_extendedprice) FROM lineitem GROUP BY l_returnflag`
- **GROUP BY K7 avg** — `SELECT l_shipmode,AVG(l_quantity) FROM lineitem GROUP BY l_shipmode`
- **GROUP BY 2-col (Q1)** — `SELECT l_returnflag,l_linestatus,COUNT(*),SUM(l_quantity),AVG(l_extendedprice) FROM lineitem GROUP BY l_returnflag,l_linestatus`
- **GROUP BY datetime K2.5k** — `SELECT l_shipdate,COUNT(*) FROM lineitem GROUP BY l_shipdate`
- **GROUP BY high-card K200k** — `SELECT l_partkey,SUM(l_quantity) FROM lineitem GROUP BY l_partkey`
- **GROUP BY vhigh-card K1.5M** — `SELECT l_orderkey,COUNT(*) FROM lineitem GROUP BY l_orderkey`

</details>

## Filtering (WHERE)

| query | what it exercises |
|---|---|
| `WHERE numeric >` | scalar predicate count |
| `WHERE BETWEEN + agg` | range predicate |
| `WHERE date-range (Q6)` | TPC-H Q6, multi-predicate + arithmetic |
| `WHERE string =` | dictionary-code equality |
| `WHERE IN (3)` | code set membership |
| `WHERE AND/OR` | boolean predicate tree |
| `WHERE + GROUP BY` | filter then group |

<details><summary>SQL</summary>

- **WHERE numeric >** — `SELECT COUNT(*) FROM lineitem WHERE l_quantity > 30`
- **WHERE BETWEEN + agg** — `SELECT SUM(l_extendedprice) FROM lineitem WHERE l_discount BETWEEN 0.05 AND 0.07`
- **WHERE date-range (Q6)** — `SELECT SUM(l_extendedprice*l_discount) FROM lineitem WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24`
- **WHERE string =** — `SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'`
- **WHERE IN (3)** — `SELECT COUNT(*) FROM lineitem WHERE l_shipmode IN ('AIR','RAIL','SHIP')`
- **WHERE AND/OR** — `SELECT SUM(l_quantity) FROM lineitem WHERE l_quantity > 30 AND (l_returnflag='R' OR l_linestatus='F')`
- **WHERE + GROUP BY** — `SELECT l_returnflag,SUM(l_extendedprice) FROM lineitem WHERE l_quantity > 25 GROUP BY l_returnflag`

</details>

## DISTINCT / COUNT(DISTINCT)

| query | what it exercises |
|---|---|
| `DISTINCT 1-col` | distinct = dictionary |
| `DISTINCT 2-col` | composite distinct |
| `DISTINCT high-card` | high-card distinct |
| `COUNT(DISTINCT) low` | distinct count |
| `COUNT(DISTINCT) high` | high-card distinct count |
| `grouped COUNT(DISTINCT)` | distinct count per group |

<details><summary>SQL</summary>

- **DISTINCT 1-col** — `SELECT DISTINCT l_returnflag FROM lineitem`
- **DISTINCT 2-col** — `SELECT DISTINCT l_returnflag,l_linestatus FROM lineitem`
- **DISTINCT high-card** — `SELECT DISTINCT l_partkey FROM lineitem`
- **COUNT(DISTINCT) low** — `SELECT COUNT(DISTINCT l_shipmode) FROM lineitem`
- **COUNT(DISTINCT) high** — `SELECT COUNT(DISTINCT l_partkey) FROM lineitem`
- **grouped COUNT(DISTINCT)** — `SELECT l_returnflag,COUNT(DISTINCT l_shipmode) FROM lineitem GROUP BY l_returnflag`

</details>

## ORDER BY · LIMIT · HAVING

| query | what it exercises |
|---|---|
| `ORDER BY + LIMIT` | top-K |
| `HAVING` | group filter |

<details><summary>SQL</summary>

- **ORDER BY + LIMIT** — `SELECT l_partkey,SUM(l_quantity) s FROM lineitem GROUP BY l_partkey ORDER BY s DESC,l_partkey LIMIT 10`
- **HAVING** — `SELECT l_shipmode,COUNT(*) c FROM lineitem GROUP BY l_shipmode HAVING COUNT(*) > 800000`

</details>

## Joins

| query | what it exercises |
|---|---|
| `JOIN group parent-key` | FK gather, group on parent dim |
| `JOIN group child-key` | FK gather, group on child |
| `JOIN group parent-date` | FK gather, high-card parent group |
| `JOIN + WHERE` | filtered join |
| `3-table JOIN` | two-hop FK chain |

<details><summary>SQL</summary>

- **JOIN group parent-key** — `SELECT c.c_mktsegment,COUNT(*),SUM(o.o_totalprice) FROM orders o JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment`
- **JOIN group child-key** — `SELECT l.l_returnflag,SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY l.l_returnflag`
- **JOIN group parent-date** — `SELECT o.o_orderdate,SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY o.o_orderdate`
- **JOIN + WHERE** — `SELECT o.o_orderpriority,SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey WHERE l.l_quantity > 30 GROUP BY o.o_orderpriority`
- **3-table JOIN** — `SELECT c.c_mktsegment,SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment`

</details>
