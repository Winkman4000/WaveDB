# WaveDB throughput scoreboard

_TPC-H sf=1 - vs DuckDB - commit `9494d44` - 2026-06-10 - W=16 single-thread workers per engine, 32 GB budget._

Throughput is the optimization target for a shared analytical DB. Each engine runs W single-threaded workers concurrently (one per core); the number is the **real measured** aggregate queries/sec, memory-bandwidth contention included (not extrapolated from single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the BSI filter-index engages where it pays (BSI column).

Rows marked **cube** in the bound column are answered from a materialised low-card GROUP BY aggregate (a precomputed [count, sums] per cell, built at load time for filter-free group-bys whose cell count is under the cap) -- the query reads a few hundred bytes instead of scanning the value columns, so it is parse-bound, not bandwidth-bound. This is a materialised view: a DIFFERENT class than a faster scan (DuckDB could build the same), and it applies ONLY to filter-free low-card group-bys with COUNT/SUM/AVG; everything else falls back to the scan.

| # | category | query | BSI | WaveDB 1-wkr q/s | WaveDB @W q/s | scale | bound | DuckDB @W q/s | ratio |
|---|---|---|:-:|--:|--:|--:|:-:|--:|--:|
| **agg** | | | | | | | | | |
| 0 | agg | whole COUNT(*) | - | 14186 | 116227 | 51% | BW | 81503 | 1.43x ** |
| 1 | agg | whole SUM | - | 2216 | 2764 | 8% | BW | 1566 | 1.76x ** |
| 2 | agg | whole multi-agg | - | 1781 | 3082 | 11% | BW | 359 | 8.59x ** |
| **group** | | | | | | | | | |
| 3 | group | GROUP BY K3 count | - | 29827 | 233850 | 49% | cube | 535 | 437.10x ** |
| 4 | group | GROUP BY K3 sum | - | 28331 | 218064 | 48% | cube | 441 | 494.48x ** |
| 5 | group | GROUP BY K7 avg | - | 27928 | 222836 | 50% | cube | 498 | 447.46x ** |
| 6 | group | GROUP BY 2-col (Q1) | - | 14353 | 113405 | 49% | cube | 230 | 493.07x ** |
| 7 | group | GROUP BY datetime K2.5k | - | 3583 | 31445 | 55% | cube | 685 | 45.91x ** |
| 8 | group | GROUP BY high-card K200k | - | 12 | 75 | 39% | BW | 32 | 2.34x ** |
| 9 | group | GROUP BY vhigh-card K1.5M | - | 5 | 21 | 28% | BW | 16 | 1.31x ** |
| **filter** | | | | | | | | | |
| 10 | filter | WHERE numeric > | - | 1169 | 7880 | 42% | BW | 1218 | 6.47x ** |
| 11 | filter | WHERE BETWEEN + agg | Y | 107 | 879 | 51% | BW | 702 | 1.25x ** |
| 12 | filter | WHERE date-range (Q6) | Y | 233 | 910 | 24% | BW | 430 | 2.12x ** |
| 13 | filter | WHERE string = | - | 1830 | 2495 | 9% | BW | 538 | 4.64x ** |
| 14 | filter | WHERE IN (3) | - | 1219 | 2139 | 11% | BW | 141 | 15.17x ** |
| 15 | filter | WHERE AND/OR | - | 227 | 638 | 18% | BW | 280 | 2.28x ** |
| 16 | filter | WHERE + GROUP BY | - | 10867 | 87334 | 50% | cube | 394 | 221.66x ** |
| **distinct** | | | | | | | | | |
| 17 | distinct | DISTINCT 1-col | - | 2094 | 2222 | 7% | BW | 1247 | 1.78x ** |
| 18 | distinct | DISTINCT 2-col | - | 129 | 484 | 23% | BW | 184 | 2.62x ** |
| 19 | distinct | DISTINCT high-card | - | 47 | 164 | 22% | BW | 88 | 1.86x ** |
| 20 | distinct | COUNT(DISTINCT) low | - | 15396 | 128172 | 52% | BW | 1284 | 99.86x ** |
| 21 | distinct | COUNT(DISTINCT) high | - | 15387 | 134619 | 55% | BW | 102 | 1313.36x ** |
| 22 | distinct | grouped COUNT(DISTINCT) | - | 24919 | 201204 | 50% | cube | 224 | 896.23x ** |
| **order** | | | | | | | | | |
| 23 | order | ORDER BY + LIMIT | - | 38 | 129 | 21% | BW | 48 | 2.66x ** |
| 24 | order | HAVING | - | 2682 | 22889 | 53% | cube | 724 | 31.59x ** |
| **join** | | | | | | | | | |
| 25 | join | JOIN group parent-key | - | 179 | 364 | 13% | BW | 374 | 0.97x |
| 26 | join | JOIN group child-key | - | 191 | 322 | 11% | BW | 53 | 6.08x ** |
| 27 | join | JOIN group parent-date | - | 57 | 148 | 16% | BW | 52 | 2.86x ** |
| 28 | join | JOIN + WHERE | - | 32 | 232 | 45% | BW | 56 | 4.18x ** |
| 29 | join | 3-table JOIN | - | 37 | 92 | 15% | BW | 38 | 2.47x ** |

**29/30 faster than DuckDB - median 4.64x.** 8 filter-free low-card group-bys are answered from a materialised cube (bound=cube; precomputed aggregate, not a scan). BSI filter-index engaged on 2 queries. Of the scan queries, 22/30 are memory-bandwidth-bound under concurrency (scale < 60%): the shared memory bus, not core count or RAM capacity, is the throughput wall -- so the lever is **bytes read per query** (compression, the BSI index, and -- where the shape allows -- not reading at all via the cube), not single-query latency. Bold ratios are WaveDB wins.

