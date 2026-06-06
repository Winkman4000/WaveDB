# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `80468fe` - 2026-06-05 - generated in 188s_

## Storage  (compression total)

| store | size | vs DuckDB |
|---|--:|--:|
| WaveDB lineitem | 121.2 MB | |
| WaveDB orders | 38.9 MB | |
| WaveDB customer | 7.2 MB | |
| WaveDB FK pointers | 5.9 MB | _(join index, like a sort key)_ |
| **WaveDB total** | **173.1 MB** | **1.24x smaller** |
| DuckDB native (3 tables) | 214.0 MB | 1.00x |

WaveDB stores the same data in **1.24x less space** than DuckDB (173.1 MB vs 214.0 MB), FK-pointer join index included. Column data alone is 167.3 MB (1.28x).

## Per-query  (speed - memory - bits)

Peak RAM = VmHWM in a fresh process per query. Load floor (open + COUNT) = 374 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.3 ms | 0.2 ms | 1.42x | 657 MB | 57 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 0.6 ms | 1.9 ms | 0.34x | 726 MB | 86 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 2.6 ms | 3.7 ms | 0.70x | 736 MB | 109 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 4.9 ms | 1.6 ms | 3.15x | 657 MB | 73 MB | Y | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 5.2 ms | 3.1 ms | 1.67x | 773 MB | 100 MB | Y | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 15.3 ms | 2.1 ms | 7.40x | 669 MB | 96 MB | Y | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 10.3 ms | 6.1 ms | 1.69x | 866 MB | 107 MB | Y | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.1 ms | 2.1 ms | 1.00x | 657 MB | 74 MB | Y | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 210.9 ms | 68.2 ms | 3.09x | 732 MB | 438 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 285.9 ms | 176.9 ms | 1.62x | 796 MB | 364 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 0.9 ms | 3.7 ms | 0.24x | 657 MB | 73 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 1.7 ms | 5.8 ms | 0.29x | 773 MB | 96 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 2.4 ms | 4.3 ms | 0.57x | 871 MB | 124 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 4.0 ms | 1.9 ms | 2.13x | 657 MB | 72 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 9.4 ms | 3.2 ms | 3.00x | 657 MB | 73 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 8.7 ms | 5.3 ms | 1.66x | 715 MB | 89 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 5.1 ms | 5.7 ms | 0.89x | 820 MB | 101 MB | Y | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 13.2 ms | 1.5 ms | 8.90x | 657 MB | 84 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 17.0 ms | 2.5 ms | 6.78x | 665 MB | 95 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 50.3 ms | 18.1 ms | 2.79x | 729 MB | 269 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 16.5 ms | 0.1 ms | 111.92x | 657 MB | 84 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 26.7 ms | 0.2 ms | 176.91x | 657 MB | 263 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 21.6 ms | 271.6 ms | 0.08x | 842 MB | 106 MB | - | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 34.4 ms | 161.2 ms | 0.21x | 744 MB | 400 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 14.9 ms | 1.9 ms | 7.71x | 657 MB | 83 MB | Y | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 9.5 ms | 1.8 ms | 5.39x | 657 MB | 108 MB | Y | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 20.7 ms | 4.8 ms | 4.30x | 773 MB | 175 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 20.0 ms | 6.6 ms | 3.04x | 782 MB | 206 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 29.7 ms | 5.3 ms | 5.63x | 831 MB | 311 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 38.4 ms | 9.8 ms | 3.93x | 852 MB | 220 MB | Y | ok |

**30/30 correct - 29/30 fused - 22/30 faster than DuckDB - median 2.79x - peak RAM median 729 MB / max 871 MB**

