# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `955579c` - 2026-06-08 - generated in 204s_

## Storage  (compression total)

| store | size | vs DuckDB |
|---|--:|--:|
| WaveDB lineitem | 121.0 MB | |
| WaveDB orders | 38.9 MB | |
| WaveDB customer | 7.2 MB | |
| WaveDB FK pointers | 7.0 MB | _(join index, like a sort key)_ |
| **WaveDB total** | **174.1 MB** | **1.23x smaller** |
| DuckDB native (3 tables) | 214.3 MB | 1.00x |

WaveDB stores the same data in **1.23x less space** than DuckDB (174.1 MB vs 214.3 MB), FK-pointer join index included. Column data alone is 167.0 MB (1.28x).

_Throughput mode (`escalate=False`, the default) additionally builds a BSI filter-index: 15.7 MB in RAM across 3 column(s) (l_discount, l_quantity, l_shipdate), built lazily only for filtered columns, capped at 64 MB/segment. The per-query table below is the **escalated** (latency) path -- fully parallel fused scan, no BSI -- so it does not include this._

## Per-query  (speed - memory - bits)

Each engine is measured ALONE in a fresh process per query (best-of-5 latency + peak RAM) -- the production scenario, since WaveDB and DuckDB never run together in deployment. Load floor (open + COUNT) = 370 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.1 ms | 0.2 ms | 0.97x | 715 MB | 56 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 0.9 ms | 3.0 ms | 0.30x | 785 MB | 86 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 2.9 ms | 1.0 ms | 2.96x | 784 MB | 110 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 2.4 ms | 0.1 ms | 16.62x | 715 MB | 73 MB | - | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 3.0 ms | 2.0 ms | 1.54x | 832 MB | 100 MB | - | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 2.9 ms | 2.6 ms | 1.13x | 714 MB | 105 MB | Y | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 5.5 ms | 6.3 ms | 0.87x | 924 MB | 108 MB | Y | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.2 ms | 2.6 ms | 0.85x | 714 MB | 74 MB | Y | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 221.1 ms | 71.0 ms | 3.11x | 783 MB | 534 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 295.8 ms | 201.9 ms | 1.47x | 799 MB | 406 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 1.2 ms | 0.2 ms | 6.85x | 715 MB | 72 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 2.0 ms | 1.8 ms | 1.09x | 833 MB | 95 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 3.0 ms | 4.2 ms | 0.71x | 933 MB | 124 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 2.4 ms | 0.3 ms | 6.94x | 714 MB | 72 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 8.3 ms | 0.3 ms | 24.74x | 714 MB | 73 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 4.4 ms | 1.8 ms | 2.51x | 725 MB | 89 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 3.2 ms | 5.7 ms | 0.55x | 879 MB | 102 MB | Y | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 1.3 ms | 1.3 ms | 1.01x | 715 MB | 88 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 14.3 ms | 2.3 ms | 6.30x | 715 MB | 102 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 51.0 ms | 16.0 ms | 3.18x | 783 MB | 339 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 1.3 ms | 0.1 ms | 8.86x | 715 MB | 89 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 26.8 ms | 0.2 ms | 177.64x | 715 MB | 325 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 13.2 ms | 10.4 ms | 1.27x | 715 MB | 116 MB | Y | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 34.2 ms | 162.7 ms | 0.21x | 783 MB | 504 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 2.1 ms | 1.9 ms | 1.09x | 715 MB | 90 MB | Y | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 8.5 ms | 1.5 ms | 5.84x | 715 MB | 170 MB | Y | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 15.4 ms | 4.2 ms | 3.65x | 832 MB | 344 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 17.9 ms | 6.2 ms | 2.89x | 839 MB | 340 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 20.0 ms | 5.0 ms | 4.04x | 891 MB | 468 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 36.7 ms | 10.5 ms | 3.48x | 937 MB | 336 MB | Y | ok |

**30/30 correct - 28/30 fused - 23/30 faster than DuckDB - median 2.89x - peak RAM median 783 MB / max 937 MB**

