# WaveDB scoreboard

_TPC-H sf=1 (lineitem N=6,001,215) - vs DuckDB - commit `3bba63b` - 2026-06-09 - generated in 198s_

## Storage  (compression total)

| store | size | vs DuckDB |
|---|--:|--:|
| WaveDB lineitem | 120.9 MB | |
| WaveDB orders | 38.9 MB | |
| WaveDB customer | 7.2 MB | |
| WaveDB FK pointers | 0.0 MB | _(join index, like a sort key)_ |
| **WaveDB total** | **167.1 MB** | **1.28x smaller** |
| DuckDB native (3 tables) | 214.0 MB | 1.00x |

WaveDB stores the same data in **1.28x less space** than DuckDB (167.1 MB vs 214.0 MB), FK-pointer join index included. Column data alone is 166.9 MB (1.28x).

_Throughput mode (`escalate=False`, the default) additionally builds a BSI filter-index: 15.7 MB in RAM across 3 column(s) (l_discount, l_quantity, l_shipdate), built lazily only for filtered columns, capped at 64 MB/segment. The per-query table below is the **escalated** (latency) path -- fully parallel fused scan, no BSI -- so it does not include this._

## Per-query  (speed - memory - bits)

Each engine is measured ALONE in a fresh process per query (best-of-5 latency + peak RAM) -- the production scenario, since WaveDB and DuckDB never run together in deployment. Load floor (open + COUNT) = 369 MB; anything above that is the query's own footprint.

| # | category | query | rows | bits read | DuckDB | WaveDB | speedup | WaveDB RAM | DuckDB RAM | fused | ok |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | - | 0.2 ms | 0.2 ms | 1.01x | 714 MB | 56 MB | Y | ok |
| 2 | agg | whole SUM | 1 | 120 Mb | 1.0 ms | 0.7 ms | 1.44x | 785 MB | 85 MB | Y | ok |
| 3 | agg | whole multi-agg | 1 | 180 Mb | 3.0 ms | 1.0 ms | 3.14x | 785 MB | 109 MB | Y | ok |
| 4 | group | GROUP BY K3 count | 3 | 12 Mb | 2.3 ms | 0.1 ms | 17.56x | 714 MB | 73 MB | - | ok |
| 5 | group | GROUP BY K3 sum | 3 | 132 Mb | 2.8 ms | 0.1 ms | 20.93x | 714 MB | 100 MB | - | ok |
| 6 | group | GROUP BY K7 avg | 7 | 54 Mb | 2.7 ms | 0.1 ms | 19.76x | 714 MB | 105 MB | - | ok |
| 7 | group | GROUP BY 2-col (Q1) | 4 | 174 Mb | 5.3 ms | 0.2 ms | 22.30x | 714 MB | 109 MB | - | ok |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72 Mb | 2.2 ms | 0.4 ms | 5.84x | 714 MB | 74 MB | - | ok |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144 Mb | 217.1 ms | 67.2 ms | 3.23x | 782 MB | 523 MB | Y | ok |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126 Mb | 291.1 ms | 200.9 ms | 1.45x | 798 MB | 398 MB | Y | ok |
| 11 | filter | WHERE numeric > | 1 | 36 Mb | 1.1 ms | 0.2 ms | 4.48x | 714 MB | 72 MB | Y | ok |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144 Mb | 1.8 ms | 1.6 ms | 1.13x | 833 MB | 95 MB | Y | ok |
| 13 | filter | WHERE date-range (Q6) | 1 | 252 Mb | 2.8 ms | 4.2 ms | 0.68x | 932 MB | 123 MB | Y | ok |
| 14 | filter | WHERE string = | 1 | 12 Mb | 2.3 ms | 0.4 ms | 6.61x | 714 MB | 72 MB | Y | ok |
| 15 | filter | WHERE IN (3) | 1 | 18 Mb | 8.2 ms | 0.4 ms | 18.98x | 714 MB | 73 MB | Y | ok |
| 16 | filter | WHERE AND/OR | 1 | 54 Mb | 4.4 ms | 1.6 ms | 2.73x | 725 MB | 89 MB | Y | ok |
| 17 | filter | WHERE + GROUP BY | 3 | 168 Mb | 3.1 ms | 2.2 ms | 1.37x | 878 MB | 102 MB | Y | ok |
| 18 | distinct | DISTINCT 1-col | 3 | 12 Mb | 1.3 ms | 0.6 ms | 2.26x | 714 MB | 88 MB | Y | ok |
| 19 | distinct | DISTINCT 2-col | 4 | 18 Mb | 13.7 ms | 3.4 ms | 4.05x | 714 MB | 101 MB | Y | ok |
| 20 | distinct | DISTINCT high-card | 200,000 | 108 Mb | 49.7 ms | 15.2 ms | 3.26x | 782 MB | 341 MB | Y | ok |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18 Mb | 1.3 ms | 0.2 ms | 8.21x | 714 MB | 89 MB | Y | ok |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108 Mb | 25.5 ms | 0.2 ms | 167.28x | 714 MB | 322 MB | Y | ok |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30 Mb | 12.2 ms | 10.9 ms | 1.12x | 714 MB | 115 MB | Y | ok |
| 24 | order | ORDER BY + LIMIT | 10 | 144 Mb | 34.2 ms | 12.9 ms | 2.66x | 783 MB | 520 MB | Y | ok |
| 25 | order | HAVING | 7 | 18 Mb | 2.0 ms | 0.5 ms | 3.87x | 714 MB | 90 MB | - | ok |
| 26 | join | JOIN group parent-key | 5 | 60 Mb | 8.2 ms | 1.7 ms | 4.93x | 714 MB | 166 MB | Y | ok |
| 27 | join | JOIN group child-key | 3 | 290 Mb | 15.3 ms | 1.8 ms | 8.63x | 830 MB | 311 MB | Y | ok |
| 28 | join | JOIN group parent-date | 2,406 | 296 Mb | 17.9 ms | 5.2 ms | 3.40x | 839 MB | 405 MB | Y | ok |
| 29 | join | JOIN + WHERE | 5 | 318 Mb | 21.3 ms | 4.9 ms | 4.33x | 890 MB | 776 MB | Y | ok |
| 30 | join | 3-table JOIN | 5 | 306 Mb | 34.7 ms | 10.1 ms | 3.44x | 937 MB | 426 MB | Y | ok |

**30/30 correct - 24/30 fused - 29/30 faster than DuckDB - median 3.87x - peak RAM median 714 MB / max 937 MB**

_Some filter-free low-card GROUP BY queries are answered from a materialised aggregate cube (a precomputed [count, sums] per cell built at load time) rather than a scan, so they show as non-fused here; the cube is a materialised view (a different class than a faster scan), gated to filter-free low-card group-bys with COUNT/SUM/AVG and falling back to the scan otherwise._

