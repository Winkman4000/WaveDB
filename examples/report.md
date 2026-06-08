# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `8b8de92` - 2026-06-07 - generated in 197s_

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

## Per-query  (speed - memory - bits)

Each engine is measured ALONE in a fresh process per query (best-of-5 latency + peak RAM) -- the production scenario, since WaveDB and DuckDB never run together in deployment. Load floor (open + COUNT) = 368 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.2 ms | 0.2 ms | 1.05x | 657 MB | 56 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 0.9 ms | 0.7 ms | 1.26x | 728 MB | 85 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 2.9 ms | 1.0 ms | 2.89x | 728 MB | 110 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 2.3 ms | 0.1 ms | 15.32x | 657 MB | 73 MB | - | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 2.7 ms | 2.0 ms | 1.38x | 773 MB | 100 MB | - | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 2.6 ms | 2.6 ms | 1.00x | 668 MB | 104 MB | Y | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 5.1 ms | 8.8 ms | 0.58x | 867 MB | 107 MB | Y | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.1 ms | 2.0 ms | 1.06x | 658 MB | 74 MB | Y | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 214.5 ms | 69.3 ms | 3.09x | 729 MB | 530 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 290.7 ms | 202.5 ms | 1.44x | 796 MB | 397 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 1.1 ms | 0.2 ms | 6.48x | 658 MB | 72 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 1.8 ms | 1.8 ms | 1.00x | 777 MB | 95 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 2.9 ms | 3.3 ms | 0.89x | 874 MB | 123 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 2.3 ms | 0.3 ms | 6.97x | 657 MB | 72 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 8.2 ms | 0.3 ms | 24.89x | 657 MB | 73 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 4.5 ms | 1.8 ms | 2.57x | 714 MB | 89 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 3.1 ms | 5.5 ms | 0.56x | 821 MB | 101 MB | Y | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 1.3 ms | 0.9 ms | 1.33x | 657 MB | 87 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 13.8 ms | 2.0 ms | 6.95x | 665 MB | 101 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 48.6 ms | 15.3 ms | 3.19x | 726 MB | 324 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 1.2 ms | 0.2 ms | 8.14x | 657 MB | 88 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 27.1 ms | 0.2 ms | 180.24x | 657 MB | 317 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 13.1 ms | 10.6 ms | 1.24x | 657 MB | 115 MB | Y | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 33.5 ms | 162.6 ms | 0.21x | 741 MB | 515 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 2.0 ms | 1.5 ms | 1.31x | 658 MB | 90 MB | Y | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 8.5 ms | 1.7 ms | 4.99x | 658 MB | 149 MB | Y | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 14.9 ms | 4.2 ms | 3.56x | 775 MB | 342 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 16.3 ms | 4.2 ms | 3.88x | 783 MB | 344 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 21.2 ms | 5.0 ms | 4.27x | 834 MB | 783 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 36.4 ms | 11.2 ms | 3.26x | 925 MB | 376 MB | Y | ok |

**30/30 correct - 28/30 fused - 25/30 faster than DuckDB - median 2.89x - peak RAM median 726 MB / max 925 MB**

