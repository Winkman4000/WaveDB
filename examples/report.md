# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `625b666` - 2026-06-05 - generated in 130s_

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

Peak RAM = VmHWM in a fresh process per query. Load floor (open + COUNT) = 483 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | peak RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.3 ms | 0.2 ms | 1.44x | 899 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 0.6 ms | 1.9 ms | 0.34x | 899 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 2.8 ms | 3.4 ms | 0.83x | 899 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 5.0 ms | 1.6 ms | 3.15x | 899 MB | Y | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 5.1 ms | 4.4 ms | 1.16x | 899 MB | Y | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 14.9 ms | 2.9 ms | 5.14x | 899 MB | Y | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 10.2 ms | 6.5 ms | 1.56x | 899 MB | Y | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.1 ms | 2.3 ms | 0.92x | 899 MB | Y | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 220.6 ms | 72.5 ms | 3.04x | 899 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 286.6 ms | 179.0 ms | 1.60x | 899 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 0.8 ms | 3.7 ms | 0.22x | 899 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 1.7 ms | 5.8 ms | 0.30x | 899 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 2.5 ms | 4.0 ms | 0.61x | 899 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 4.0 ms | 1.9 ms | 2.10x | 899 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 9.3 ms | 3.2 ms | 2.97x | 899 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 8.5 ms | 5.3 ms | 1.61x | 899 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 5.0 ms | 5.6 ms | 0.89x | 899 MB | Y | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 12.5 ms | 1.5 ms | 8.37x | 899 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 17.6 ms | 2.7 ms | 6.40x | 899 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 50.6 ms | 15.8 ms | 3.21x | 899 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 15.5 ms | 0.1 ms | 105.94x | 899 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 26.8 ms | 0.2 ms | 177.98x | 899 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 20.4 ms | 265.8 ms | 0.08x | 899 MB | - | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 33.7 ms | 160.1 ms | 0.21x | 899 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 14.0 ms | 1.9 ms | 7.33x | 899 MB | Y | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 9.8 ms | 1.9 ms | 5.27x | 899 MB | Y | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 19.8 ms | 2.8 ms | 6.99x | 899 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 17.0 ms | 4.3 ms | 3.92x | 899 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 30.5 ms | 4.8 ms | 6.34x | 899 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 38.6 ms | 10.0 ms | 3.85x | 911 MB | Y | ok |

**30/30 correct - 29/30 fused - 21/30 faster than DuckDB - median 2.97x - peak RAM median 899 MB / max 911 MB**

