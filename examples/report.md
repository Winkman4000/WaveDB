# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `e4f1d2b` - 2026-06-06 - generated in 186s_

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

Peak RAM = VmHWM in a fresh process per query. Load floor (open + COUNT) = 367 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.3 ms | 0.2 ms | 1.86x | 656 MB | 54 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 0.6 ms | 1.0 ms | 0.61x | 725 MB | 84 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 2.8 ms | 2.8 ms | 1.00x | 735 MB | 107 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 4.7 ms | 1.3 ms | 3.63x | 656 MB | 71 MB | Y | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 4.9 ms | 4.1 ms | 1.19x | 771 MB | 98 MB | Y | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 15.4 ms | 2.9 ms | 5.31x | 667 MB | 94 MB | Y | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 10.0 ms | 7.3 ms | 1.38x | 865 MB | 108 MB | Y | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.0 ms | 2.1 ms | 0.96x | 656 MB | 72 MB | Y | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 211.6 ms | 71.9 ms | 2.95x | 730 MB | 431 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 288.0 ms | 178.9 ms | 1.61x | 795 MB | 363 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 0.8 ms | 0.2 ms | 4.73x | 656 MB | 70 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 1.6 ms | 1.8 ms | 0.86x | 771 MB | 94 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 2.3 ms | 3.7 ms | 0.60x | 870 MB | 121 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 4.0 ms | 0.3 ms | 14.53x | 656 MB | 70 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 9.3 ms | 0.4 ms | 21.95x | 656 MB | 71 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 8.4 ms | 2.0 ms | 4.12x | 713 MB | 86 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 5.0 ms | 5.6 ms | 0.89x | 819 MB | 99 MB | Y | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 13.6 ms | 1.5 ms | 9.12x | 656 MB | 81 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 18.1 ms | 4.3 ms | 4.20x | 664 MB | 93 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 47.2 ms | 16.6 ms | 2.85x | 727 MB | 264 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 17.0 ms | 0.2 ms | 112.84x | 656 MB | 83 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 27.1 ms | 0.2 ms | 174.22x | 656 MB | 263 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 21.9 ms | 258.5 ms | 0.08x | 841 MB | 104 MB | - | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 33.9 ms | 163.0 ms | 0.21x | 742 MB | 398 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 14.7 ms | 1.9 ms | 7.68x | 656 MB | 81 MB | Y | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 8.6 ms | 1.8 ms | 4.88x | 656 MB | 106 MB | Y | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 19.2 ms | 4.4 ms | 4.35x | 772 MB | 171 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 17.5 ms | 4.5 ms | 3.91x | 780 MB | 200 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 30.0 ms | 5.2 ms | 5.83x | 830 MB | 310 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 36.4 ms | 9.9 ms | 3.68x | 851 MB | 211 MB | Y | ok |

**30/30 correct - 29/30 fused - 23/30 faster than DuckDB - median 3.68x - peak RAM median 727 MB / max 870 MB**

