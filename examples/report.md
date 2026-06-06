# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `6583245` - 2026-06-06 - generated in 202s_

## Storage  (compression total)

| store | size | vs DuckDB |
|---|--:|--:|
| WaveDB lineitem | 121.2 MB | |
| WaveDB orders | 38.9 MB | |
| WaveDB customer | 7.2 MB | |
| WaveDB FK pointers | 5.9 MB | _(join index, like a sort key)_ |
| **WaveDB total** | **173.1 MB** | **1.23x smaller** |
| DuckDB native (3 tables) | 213.8 MB | 1.00x |

WaveDB stores the same data in **1.23x less space** than DuckDB (173.1 MB vs 213.8 MB), FK-pointer join index included. Column data alone is 167.3 MB (1.28x).

## Per-query  (speed - memory - bits)

Each engine is measured ALONE in a fresh process per query (best-of-5 latency + peak RAM) -- the production scenario, since WaveDB and DuckDB never run together in deployment. Load floor (open + COUNT) = 367 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.2 ms | 0.2 ms | 0.91x | 656 MB | 55 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 1.0 ms | 0.8 ms | 1.18x | 724 MB | 84 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 2.9 ms | 3.0 ms | 0.97x | 735 MB | 109 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 2.3 ms | 1.3 ms | 1.80x | 656 MB | 71 MB | Y | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 2.8 ms | 4.0 ms | 0.69x | 772 MB | 99 MB | Y | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 2.8 ms | 1.7 ms | 1.62x | 667 MB | 104 MB | Y | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 5.4 ms | 6.1 ms | 0.89x | 865 MB | 109 MB | Y | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.2 ms | 2.0 ms | 1.11x | 655 MB | 73 MB | Y | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 214.9 ms | 70.0 ms | 3.07x | 730 MB | 534 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 326.9 ms | 200.6 ms | 1.63x | 795 MB | 403 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 1.1 ms | 0.2 ms | 6.62x | 655 MB | 71 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 1.9 ms | 1.6 ms | 1.14x | 771 MB | 94 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 2.8 ms | 2.8 ms | 1.00x | 870 MB | 123 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 2.3 ms | 0.4 ms | 6.64x | 656 MB | 71 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 8.3 ms | 0.4 ms | 19.57x | 655 MB | 72 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 4.6 ms | 1.6 ms | 2.83x | 713 MB | 88 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 3.0 ms | 5.5 ms | 0.55x | 818 MB | 100 MB | Y | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 1.3 ms | 1.2 ms | 1.06x | 656 MB | 87 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 13.1 ms | 2.5 ms | 5.32x | 664 MB | 101 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 50.2 ms | 16.8 ms | 2.98x | 728 MB | 360 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 1.3 ms | 0.2 ms | 8.45x | 655 MB | 88 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 25.9 ms | 0.1 ms | 173.37x | 656 MB | 315 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 13.4 ms | 257.4 ms | 0.05x | 841 MB | 114 MB | - | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 34.5 ms | 163.4 ms | 0.21x | 743 MB | 513 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 2.0 ms | 1.9 ms | 1.03x | 655 MB | 89 MB | Y | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 8.0 ms | 1.7 ms | 4.68x | 655 MB | 155 MB | Y | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 15.3 ms | 3.3 ms | 4.63x | 772 MB | 313 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 18.3 ms | 4.3 ms | 4.30x | 780 MB | 404 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 19.9 ms | 5.2 ms | 3.84x | 831 MB | 592 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 37.1 ms | 10.4 ms | 3.58x | 925 MB | 365 MB | Y | ok |

**30/30 correct - 29/30 fused - 23/30 faster than DuckDB - median 1.80x - peak RAM median 728 MB / max 925 MB**

