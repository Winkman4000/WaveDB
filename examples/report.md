# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `0b2ed15` - 2026-06-07 - generated in 199s_

## Storage  (compression total)

| store | size | vs DuckDB |
|---|--:|--:|
| WaveDB lineitem | 121.1 MB | |
| WaveDB orders | 38.9 MB | |
| WaveDB customer | 7.2 MB | |
| WaveDB FK pointers | 0.0 MB | _(join index, like a sort key)_ |
| **WaveDB total** | **167.2 MB** | **1.28x smaller** |
| DuckDB native (3 tables) | 214.3 MB | 1.00x |

WaveDB stores the same data in **1.28x less space** than DuckDB (167.2 MB vs 214.3 MB), FK-pointer join index included. Column data alone is 167.2 MB (1.28x).

## Per-query  (speed - memory - bits)

Each engine is measured ALONE in a fresh process per query (best-of-5 latency + peak RAM) -- the production scenario, since WaveDB and DuckDB never run together in deployment. Load floor (open + COUNT) = 366 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.2 ms | 0.2 ms | 0.93x | 655 MB | 54 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 0.9 ms | 1.3 ms | 0.70x | 724 MB | 83 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 2.9 ms | 2.5 ms | 1.16x | 734 MB | 108 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 2.3 ms | 1.2 ms | 1.83x | 655 MB | 71 MB | Y | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 2.8 ms | 2.8 ms | 1.00x | 771 MB | 98 MB | Y | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 2.7 ms | 1.7 ms | 1.53x | 666 MB | 103 MB | Y | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 5.2 ms | 4.6 ms | 1.15x | 864 MB | 105 MB | Y | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.2 ms | 2.0 ms | 1.11x | 655 MB | 73 MB | Y | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 214.8 ms | 67.5 ms | 3.18x | 730 MB | 539 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 289.2 ms | 198.0 ms | 1.46x | 794 MB | 394 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 1.2 ms | 0.2 ms | 6.64x | 655 MB | 70 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 1.9 ms | 1.8 ms | 1.07x | 770 MB | 93 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 3.0 ms | 3.2 ms | 0.93x | 870 MB | 122 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 2.3 ms | 0.3 ms | 8.36x | 655 MB | 70 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 8.2 ms | 0.4 ms | 19.58x | 655 MB | 71 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 4.4 ms | 1.8 ms | 2.46x | 712 MB | 87 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 3.1 ms | 5.5 ms | 0.56x | 818 MB | 100 MB | Y | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 1.3 ms | 1.5 ms | 0.86x | 655 MB | 85 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 13.3 ms | 2.7 ms | 4.99x | 664 MB | 101 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 50.0 ms | 15.9 ms | 3.15x | 725 MB | 343 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 1.3 ms | 0.2 ms | 8.46x | 655 MB | 86 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 25.4 ms | 0.1 ms | 172.91x | 654 MB | 321 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 12.5 ms | 10.3 ms | 1.21x | 655 MB | 113 MB | Y | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 32.7 ms | 159.0 ms | 0.21x | 742 MB | 500 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 2.0 ms | 1.5 ms | 1.34x | 655 MB | 88 MB | Y | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 8.6 ms | 1.3 ms | 6.69x | 655 MB | 168 MB | Y | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 14.6 ms | 2.9 ms | 5.01x | 771 MB | 310 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 16.7 ms | 5.0 ms | 3.32x | 780 MB | 373 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 20.1 ms | 4.9 ms | 4.07x | 830 MB | 721 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 36.3 ms | 9.6 ms | 3.79x | 924 MB | 316 MB | Y | ok |

**30/30 correct - 30/30 fused - 23/30 faster than DuckDB - median 1.83x - peak RAM median 724 MB / max 924 MB**

