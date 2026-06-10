# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `4c4512b` - 2026-06-10 - generated in 193s_

## Storage  (compression total)

| store | size | vs DuckDB |
|---|--:|--:|
| WaveDB lineitem | 120.9 MB | |
| WaveDB orders | 40.2 MB | |
| WaveDB customer | 7.2 MB | |
| WaveDB FK pointers | 7.0 MB | _(join index, like a sort key)_ |
| **WaveDB total** | **180.4 MB** | **1.17x smaller** |
| DuckDB native (3 tables) | 210.5 MB | 1.00x |

WaveDB stores the same data in **1.17x less space** than DuckDB (180.4 MB vs 210.5 MB), FK-pointer join index included. Column data alone is 168.3 MB (1.25x).

_Throughput mode (`escalate=False`, the default) additionally builds a BSI filter-index: 15.7 MB in RAM across 3 column(s) (l_discount, l_quantity, l_shipdate), built lazily only for filtered columns, capped at 64 MB/segment. The per-query table below is the **escalated** (latency) path -- fully parallel fused scan, no BSI -- so it does not include this._

## Per-query  (speed - memory - bits)

Each engine is measured ALONE in a fresh process per query (best-of-5 latency + peak RAM) -- the production scenario, since WaveDB and DuckDB never run together in deployment. Load floor (open + COUNT) = 371 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.2 ms | 0.1 ms | 1.99x | 708 MB | 60 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 1.4 ms | 0.6 ms | 2.44x | 780 MB | 89 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 7.7 ms | 0.7 ms | 10.74x | 779 MB | 105 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 2.2 ms | 0.0 ms | 59.07x | 709 MB | 76 MB | - | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 3.5 ms | 0.0 ms | 90.49x | 709 MB | 99 MB | - | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 2.9 ms | 0.0 ms | 73.95x | 709 MB | 109 MB | - | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 6.0 ms | 0.1 ms | 82.52x | 709 MB | 111 MB | - | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.2 ms | 0.3 ms | 8.03x | 709 MB | 77 MB | - | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 66.8 ms | 68.4 ms | 0.98x | 788 MB | 550 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 287.9 ms | 204.2 ms | 1.41x | 807 MB | 397 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 4.2 ms | 0.1 ms | 54.44x | 709 MB | 75 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 6.6 ms | 1.5 ms | 4.57x | 828 MB | 97 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 5.5 ms | 3.4 ms | 1.60x | 926 MB | 118 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 2.5 ms | 0.2 ms | 14.65x | 709 MB | 75 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 8.5 ms | 0.2 ms | 41.93x | 709 MB | 76 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 7.7 ms | 1.3 ms | 6.02x | 719 MB | 92 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 6.8 ms | 0.1 ms | 71.89x | 709 MB | 110 MB | - | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 1.3 ms | 0.5 ms | 2.64x | 709 MB | 90 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 14.7 ms | 2.9 ms | 5.14x | 709 MB | 93 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 49.2 ms | 15.1 ms | 3.26x | 777 MB | 341 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 1.3 ms | 0.1 ms | 20.65x | 709 MB | 91 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 27.3 ms | 0.1 ms | 426.68x | 708 MB | 327 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 13.4 ms | 0.0 ms | 307.86x | 709 MB | 119 MB | - | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 34.3 ms | 12.8 ms | 2.68x | 787 MB | 492 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 2.1 ms | 0.3 ms | 5.89x | 708 MB | 94 MB | - | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 8.3 ms | 0.1 ms | 127.86x | 709 MB | 164 MB | - | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 18.3 ms | 2.0 ms | 9.28x | 825 MB | 444 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 14.9 ms | 6.3 ms | 2.37x | 835 MB | 411 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 24.6 ms | 4.7 ms | 5.21x | 885 MB | 596 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 37.9 ms | 10.1 ms | 3.74x | 1023 MB | 399 MB | Y | ok |

**30/30 correct - 21/30 fused - 29/30 faster than DuckDB - median 8.03x - peak RAM median 709 MB / max 1023 MB**

_Some filter-free low-card GROUP BY queries are answered from a materialised aggregate cube (a precomputed [count, sums] per cell built at load time) rather than a scan, so they show as non-fused here; the cube is a materialised view (a different class than a faster scan), gated to filter-free low-card group-bys with COUNT/SUM/AVG and falling back to the scan otherwise._

