# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `a93e44b` - 2026-06-09 - generated in 194s_

## Storage  (compression total)

| store | size | vs DuckDB |
|---|--:|--:|
| WaveDB lineitem | 120.9 MB | |
| WaveDB orders | 38.9 MB | |
| WaveDB customer | 7.2 MB | |
| WaveDB FK pointers | 7.0 MB | _(join index, like a sort key)_ |
| **WaveDB total** | **174.2 MB** | **1.23x smaller** |
| DuckDB native (3 tables) | 214.3 MB | 1.00x |

WaveDB stores the same data in **1.23x less space** than DuckDB (174.2 MB vs 214.3 MB), FK-pointer join index included. Column data alone is 167.0 MB (1.28x).

_Throughput mode (`escalate=False`, the default) additionally builds a BSI filter-index: 15.7 MB in RAM across 3 column(s) (l_discount, l_quantity, l_shipdate), built lazily only for filtered columns, capped at 64 MB/segment. The per-query table below is the **escalated** (latency) path -- fully parallel fused scan, no BSI -- so it does not include this._

## Per-query  (speed - memory - bits)

Each engine is measured ALONE in a fresh process per query (best-of-5 latency + peak RAM) -- the production scenario, since WaveDB and DuckDB never run together in deployment. Load floor (open + COUNT) = 369 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.2 ms | 0.1 ms | 2.27x | 714 MB | 57 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 1.0 ms | 0.6 ms | 1.71x | 784 MB | 86 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 3.0 ms | 0.7 ms | 4.04x | 785 MB | 111 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 2.6 ms | 0.0 ms | 79.69x | 715 MB | 73 MB | - | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 3.1 ms | 0.0 ms | 91.32x | 714 MB | 100 MB | - | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 2.9 ms | 0.0 ms | 87.13x | 714 MB | 105 MB | - | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 5.4 ms | 0.1 ms | 82.04x | 714 MB | 110 MB | - | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.1 ms | 0.3 ms | 7.93x | 714 MB | 74 MB | - | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 224.0 ms | 69.0 ms | 3.25x | 783 MB | 530 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 293.3 ms | 200.3 ms | 1.46x | 799 MB | 394 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 1.2 ms | 0.1 ms | 16.40x | 714 MB | 72 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 1.9 ms | 1.1 ms | 1.77x | 832 MB | 97 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 2.8 ms | 3.0 ms | 0.94x | 932 MB | 125 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 2.4 ms | 0.2 ms | 13.47x | 714 MB | 72 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 8.4 ms | 0.2 ms | 40.34x | 714 MB | 73 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 4.4 ms | 1.4 ms | 3.15x | 725 MB | 89 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 3.1 ms | 2.4 ms | 1.25x | 879 MB | 102 MB | Y | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 1.2 ms | 0.5 ms | 2.60x | 714 MB | 87 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 13.8 ms | 1.9 ms | 7.33x | 714 MB | 102 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 51.3 ms | 16.1 ms | 3.19x | 783 MB | 331 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 1.3 ms | 0.1 ms | 20.21x | 714 MB | 89 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 27.5 ms | 0.1 ms | 424.85x | 714 MB | 329 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 13.3 ms | 0.0 ms | 317.25x | 714 MB | 116 MB | - | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 34.6 ms | 12.0 ms | 2.89x | 783 MB | 510 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 2.0 ms | 0.4 ms | 5.68x | 714 MB | 90 MB | - | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 8.5 ms | 1.1 ms | 7.90x | 714 MB | 176 MB | Y | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 15.9 ms | 1.8 ms | 9.03x | 831 MB | 312 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 19.1 ms | 4.2 ms | 4.55x | 840 MB | 376 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 20.3 ms | 4.7 ms | 4.34x | 890 MB | 660 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 36.4 ms | 11.0 ms | 3.32x | 1028 MB | 355 MB | Y | ok |

**30/30 correct - 23/30 fused - 29/30 faster than DuckDB - median 5.68x - peak RAM median 715 MB / max 1028 MB**

_Some filter-free low-card GROUP BY queries are answered from a materialised aggregate cube (a precomputed [count, sums] per cell built at load time) rather than a scan, so they show as non-fused here; the cube is a materialised view (a different class than a faster scan), gated to filter-free low-card group-bys with COUNT/SUM/AVG and falling back to the scan otherwise._

