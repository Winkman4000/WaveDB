# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `ce5c504` - 2026-06-07 - generated in 199s_

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
| 1 | agg | whole COUNT(*) | 1 | - | 0.2 ms | 0.2 ms | 0.94x | 657 MB | 56 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 0.9 ms | 0.7 ms | 1.25x | 727 MB | 85 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 2.9 ms | 1.0 ms | 2.96x | 728 MB | 109 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 2.3 ms | 0.1 ms | 15.81x | 657 MB | 72 MB | - | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 2.8 ms | 2.0 ms | 1.36x | 773 MB | 99 MB | - | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 2.7 ms | 2.1 ms | 1.25x | 668 MB | 105 MB | Y | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 5.3 ms | 5.9 ms | 0.90x | 867 MB | 107 MB | Y | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.2 ms | 2.0 ms | 1.09x | 657 MB | 74 MB | Y | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 219.0 ms | 73.4 ms | 2.98x | 729 MB | 530 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 293.3 ms | 203.4 ms | 1.44x | 796 MB | 372 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 1.2 ms | 0.2 ms | 6.93x | 658 MB | 71 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 2.0 ms | 1.7 ms | 1.14x | 775 MB | 95 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 2.9 ms | 4.5 ms | 0.65x | 875 MB | 123 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 2.4 ms | 0.3 ms | 8.37x | 657 MB | 72 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 8.4 ms | 0.3 ms | 25.43x | 657 MB | 73 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 4.7 ms | 1.8 ms | 2.58x | 713 MB | 89 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 3.4 ms | 5.6 ms | 0.61x | 822 MB | 101 MB | Y | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 1.3 ms | 1.3 ms | 1.00x | 657 MB | 87 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 13.6 ms | 2.7 ms | 4.97x | 665 MB | 102 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 51.4 ms | 16.6 ms | 3.09x | 725 MB | 339 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 1.3 ms | 0.2 ms | 8.72x | 657 MB | 88 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 27.1 ms | 0.2 ms | 172.39x | 657 MB | 319 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 13.3 ms | 57.9 ms | 0.23x | 657 MB | 116 MB | - | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 34.2 ms | 175.6 ms | 0.20x | 741 MB | 519 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 2.1 ms | 1.6 ms | 1.31x | 657 MB | 90 MB | Y | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 8.7 ms | 1.7 ms | 5.13x | 657 MB | 193 MB | Y | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 14.8 ms | 4.2 ms | 3.48x | 774 MB | 376 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 17.2 ms | 4.2 ms | 4.06x | 784 MB | 407 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 20.9 ms | 4.9 ms | 4.26x | 833 MB | 659 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 37.0 ms | 11.2 ms | 3.32x | 925 MB | 335 MB | Y | ok |

**30/30 correct - 27/30 fused - 24/30 faster than DuckDB - median 2.96x - peak RAM median 725 MB / max 925 MB**

