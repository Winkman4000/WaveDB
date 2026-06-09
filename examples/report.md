# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `f1fd4d9` - 2026-06-09 - generated in 201s_

## Storage  (compression total)

| store | size | vs DuckDB |
|---|--:|--:|
| WaveDB lineitem | 121.2 MB | |
| WaveDB orders | 38.9 MB | |
| WaveDB customer | 7.2 MB | |
| WaveDB FK pointers | 7.0 MB | _(join index, like a sort key)_ |
| **WaveDB total** | **174.3 MB** | **1.23x smaller** |
| DuckDB native (3 tables) | 213.5 MB | 1.00x |

WaveDB stores the same data in **1.23x less space** than DuckDB (174.3 MB vs 213.5 MB), FK-pointer join index included. Column data alone is 167.3 MB (1.28x).

_Throughput mode (`escalate=False`, the default) additionally builds a BSI filter-index: 15.7 MB in RAM across 3 column(s) (l_discount, l_quantity, l_shipdate), built lazily only for filtered columns, capped at 64 MB/segment. The per-query table below is the **escalated** (latency) path -- fully parallel fused scan, no BSI -- so it does not include this._

## Per-query  (speed - memory - bits)

Each engine is measured ALONE in a fresh process per query (best-of-5 latency + peak RAM) -- the production scenario, since WaveDB and DuckDB never run together in deployment. Load floor (open + COUNT) = 369 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.1 ms | 0.2 ms | 0.95x | 713 MB | 55 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 0.9 ms | 0.8 ms | 1.17x | 783 MB | 84 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 2.9 ms | 1.0 ms | 3.04x | 784 MB | 110 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 2.3 ms | 0.1 ms | 15.97x | 713 MB | 71 MB | - | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 2.7 ms | 1.9 ms | 1.42x | 831 MB | 99 MB | - | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 2.6 ms | 2.5 ms | 1.04x | 713 MB | 104 MB | Y | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 5.2 ms | 5.7 ms | 0.90x | 924 MB | 108 MB | Y | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.2 ms | 2.0 ms | 1.09x | 713 MB | 73 MB | Y | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 213.6 ms | 68.1 ms | 3.14x | 782 MB | 543 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 291.8 ms | 203.8 ms | 1.43x | 798 MB | 405 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 1.2 ms | 0.2 ms | 6.58x | 713 MB | 71 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 1.9 ms | 1.1 ms | 1.66x | 832 MB | 96 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 3.0 ms | 3.4 ms | 0.86x | 931 MB | 124 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 2.3 ms | 0.3 ms | 7.93x | 713 MB | 70 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 8.2 ms | 0.3 ms | 23.84x | 713 MB | 72 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 4.4 ms | 1.8 ms | 2.48x | 723 MB | 88 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 3.1 ms | 2.5 ms | 1.26x | 878 MB | 100 MB | Y | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 1.2 ms | 0.6 ms | 2.19x | 713 MB | 86 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 12.7 ms | 2.1 ms | 6.00x | 713 MB | 100 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 50.2 ms | 17.4 ms | 2.89x | 782 MB | 322 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 1.3 ms | 0.2 ms | 7.91x | 713 MB | 87 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 27.3 ms | 0.2 ms | 176.32x | 713 MB | 327 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 13.1 ms | 10.7 ms | 1.23x | 713 MB | 114 MB | Y | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 33.1 ms | 13.2 ms | 2.52x | 781 MB | 516 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 2.0 ms | 1.5 ms | 1.31x | 713 MB | 89 MB | Y | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 8.3 ms | 1.3 ms | 6.53x | 713 MB | 167 MB | Y | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 15.1 ms | 1.7 ms | 9.09x | 829 MB | 376 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 17.2 ms | 4.4 ms | 3.94x | 839 MB | 410 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 21.7 ms | 4.9 ms | 4.46x | 889 MB | 657 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 36.6 ms | 9.5 ms | 3.85x | 936 MB | 299 MB | Y | ok |

**30/30 correct - 28/30 fused - 27/30 faster than DuckDB - median 2.89x - peak RAM median 781 MB / max 936 MB**

