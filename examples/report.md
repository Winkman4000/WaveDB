# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `9494d44` - 2026-06-10 - generated in 193s_

## Storage  (compression total)

| store | size | vs DuckDB |
|---|--:|--:|
| WaveDB lineitem | 120.9 MB | |
| WaveDB orders | 38.9 MB | |
| WaveDB customer | 7.2 MB | |
| WaveDB FK pointers | 7.0 MB | _(join index, like a sort key)_ |
| **WaveDB total** | **178.8 MB** | **1.20x smaller** |
| DuckDB native (3 tables) | 214.3 MB | 1.00x |

WaveDB stores the same data in **1.20x less space** than DuckDB (178.8 MB vs 214.3 MB), FK-pointer join index included. Column data alone is 167.0 MB (1.28x).

_Throughput mode (`escalate=False`, the default) additionally builds a BSI filter-index: 15.7 MB in RAM across 3 column(s) (l_discount, l_quantity, l_shipdate), built lazily only for filtered columns, capped at 64 MB/segment. The per-query table below is the **escalated** (latency) path -- fully parallel fused scan, no BSI -- so it does not include this._

## Per-query  (speed - memory - bits)

Each engine is measured ALONE in a fresh process per query (best-of-5 latency + peak RAM) -- the production scenario, since WaveDB and DuckDB never run together in deployment. Load floor (open + COUNT) = 370 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.2 ms | 0.1 ms | 2.10x | 715 MB | 57 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 0.9 ms | 3.0 ms | 0.30x | 785 MB | 86 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 2.9 ms | 0.7 ms | 3.98x | 785 MB | 112 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 2.3 ms | 0.0 ms | 61.49x | 714 MB | 74 MB | - | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 2.7 ms | 0.0 ms | 71.48x | 715 MB | 101 MB | - | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 2.7 ms | 0.0 ms | 69.62x | 715 MB | 106 MB | - | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 5.1 ms | 0.1 ms | 70.78x | 715 MB | 110 MB | - | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.2 ms | 0.3 ms | 8.31x | 715 MB | 75 MB | - | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 217.5 ms | 67.6 ms | 3.22x | 795 MB | 525 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 292.5 ms | 200.8 ms | 1.46x | 813 MB | 396 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 1.2 ms | 0.1 ms | 16.12x | 715 MB | 73 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 2.0 ms | 1.5 ms | 1.38x | 833 MB | 98 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 3.0 ms | 3.1 ms | 0.98x | 932 MB | 126 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 2.5 ms | 0.2 ms | 10.40x | 715 MB | 73 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 8.5 ms | 0.2 ms | 39.35x | 715 MB | 74 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 4.7 ms | 1.4 ms | 3.42x | 725 MB | 90 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 3.4 ms | 0.1 ms | 36.40x | 714 MB | 103 MB | - | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 1.3 ms | 0.5 ms | 2.83x | 715 MB | 88 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 14.5 ms | 1.9 ms | 7.65x | 714 MB | 103 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 48.8 ms | 15.0 ms | 3.25x | 783 MB | 331 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 1.3 ms | 0.1 ms | 19.45x | 715 MB | 89 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 26.4 ms | 0.1 ms | 409.60x | 715 MB | 319 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 12.0 ms | 0.0 ms | 258.72x | 715 MB | 116 MB | - | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 34.3 ms | 11.7 ms | 2.93x | 795 MB | 500 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 2.0 ms | 0.3 ms | 5.81x | 715 MB | 91 MB | - | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 8.4 ms | 1.1 ms | 7.80x | 714 MB | 173 MB | Y | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 14.9 ms | 1.6 ms | 9.25x | 832 MB | 409 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 16.8 ms | 4.1 ms | 4.09x | 840 MB | 409 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 21.3 ms | 4.7 ms | 4.58x | 890 MB | 528 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 35.4 ms | 9.7 ms | 3.65x | 1029 MB | 363 MB | Y | ok |

**30/30 correct - 22/30 fused - 28/30 faster than DuckDB - median 7.65x - peak RAM median 715 MB / max 1029 MB**

_Some filter-free low-card GROUP BY queries are answered from a materialised aggregate cube (a precomputed [count, sums] per cell built at load time) rather than a scan, so they show as non-fused here; the cube is a materialised view (a different class than a faster scan), gated to filter-free low-card group-bys with COUNT/SUM/AVG and falling back to the scan otherwise._

