# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `259f5dd` - 2026-06-20 - generated in 195s_

## Storage  (compression total)

| store | size | vs DuckDB |
|---|--:|--:|
| WaveDB lineitem | 120.9 MB | |
| WaveDB orders | 40.2 MB | |
| WaveDB customer | 7.2 MB | |
| WaveDB FK pointers | 7.0 MB | _(join index, like a sort key)_ |
| **WaveDB total** | **180.3 MB** | **1.19x smaller** |
| DuckDB native (3 tables) | 213.8 MB | 1.00x |

WaveDB stores the same data in **1.19x less space** than DuckDB (180.3 MB vs 213.8 MB), FK-pointer join index included. Column data alone is 168.2 MB (1.27x).

_Throughput mode (`escalate=False`, the default) additionally builds a BSI filter-index: 15.7 MB in RAM across 3 column(s) (l_discount, l_quantity, l_shipdate), built lazily only for filtered columns, capped at 64 MB/segment. The per-query table below is the **escalated** (latency) path -- fully parallel fused scan, no BSI -- so it does not include this._

## Per-query  (speed - memory - bits)

Each engine is measured ALONE in a fresh process per query (best-of-5 latency + peak RAM) -- the production scenario, since WaveDB and DuckDB never run together in deployment. Load floor (open + COUNT) = 368 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.2 ms | 0.1 ms | 1.95x | 693 MB | 55 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 0.9 ms | 0.6 ms | 1.65x | 733 MB | 84 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 2.9 ms | 2.8 ms | 1.06x | 733 MB | 109 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 2.2 ms | 0.0 ms | 57.21x | 693 MB | 71 MB | - | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 2.8 ms | 0.0 ms | 71.41x | 693 MB | 98 MB | - | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 2.8 ms | 0.0 ms | 69.16x | 693 MB | 103 MB | - | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 5.5 ms | 0.1 ms | 76.09x | 693 MB | 107 MB | - | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.2 ms | 0.3 ms | 8.28x | 693 MB | 73 MB | - | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 224.9 ms | 64.7 ms | 3.47x | 742 MB | 524 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 299.4 ms | 181.8 ms | 1.65x | 780 MB | 404 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 1.1 ms | 0.1 ms | 13.44x | 693 MB | 70 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 1.8 ms | 1.3 ms | 1.41x | 743 MB | 94 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 2.9 ms | 1.8 ms | 1.61x | 771 MB | 123 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 2.4 ms | 0.2 ms | 13.29x | 693 MB | 70 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 8.5 ms | 0.2 ms | 35.99x | 692 MB | 71 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 4.7 ms | 0.7 ms | 6.55x | 693 MB | 88 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 3.2 ms | 0.1 ms | 33.67x | 693 MB | 100 MB | - | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 1.3 ms | 0.2 ms | 8.43x | 692 MB | 86 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 13.9 ms | 1.2 ms | 11.81x | 693 MB | 101 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 50.7 ms | 14.7 ms | 3.45x | 732 MB | 336 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 1.3 ms | 0.1 ms | 18.81x | 693 MB | 87 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 27.4 ms | 0.1 ms | 404.22x | 693 MB | 325 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 13.5 ms | 0.0 ms | 293.35x | 693 MB | 114 MB | - | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 35.0 ms | 12.3 ms | 2.85x | 743 MB | 514 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 2.0 ms | 0.4 ms | 5.82x | 693 MB | 89 MB | - | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 8.1 ms | 0.1 ms | 127.45x | 693 MB | 160 MB | - | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 14.7 ms | 1.4 ms | 10.31x | 741 MB | 342 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 17.4 ms | 3.5 ms | 4.99x | 781 MB | 311 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 21.2 ms | 4.4 ms | 4.86x | 787 MB | 532 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 38.5 ms | 10.0 ms | 3.84x | 938 MB | 394 MB | Y | ok |

**30/30 correct - 21/30 fused - 30/30 faster than DuckDB - median 8.43x - peak RAM median 693 MB / max 938 MB**

_Some filter-free low-card GROUP BY queries are answered from a materialised aggregate cube (a precomputed [count, sums] per cell built at load time) rather than a scan, so they show as non-fused here; the cube is a materialised view (a different class than a faster scan), gated to filter-free low-card group-bys with COUNT/SUM/AVG and falling back to the scan otherwise._

