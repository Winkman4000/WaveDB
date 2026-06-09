# WaveDB throughput scoreboard

_TPC-H sf=1 - vs DuckDB - commit `2e212ab` - 2026-06-09 - W=16 single-thread workers per engine, 32 GB budget._

Throughput is the optimization target for a shared analytical DB. Each engine runs W single-threaded workers concurrently (one per core); the number is the **real measured** aggregate queries/sec, memory-bandwidth contention included (not extrapolated from single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the BSI filter-index engages where it pays (BSI column).

Rows marked **cube** in the bound column are answered from a materialised low-card GROUP BY aggregate (a precomputed [count, sums] per cell, built at load time for filter-free group-bys whose cell count is under the cap) -- the query reads a few hundred bytes instead of scanning the value columns, so it is parse-bound, not bandwidth-bound. This is a materialised view: a DIFFERENT class than a faster scan (DuckDB could build the same), and it applies ONLY to filter-free low-card group-bys with COUNT/SUM/AVG; everything else falls back to the scan.

| # | category | query | BSI | WaveDB 1-wkr q/s | WaveDB @W q/s | scale | bound | DuckDB @W q/s | ratio |
|---|---|---|:-:|--:|--:|--:|:-:|--:|--:|
| **agg** | | | | | | | | | |
| 0 | agg | whole COUNT(*) | - | 6480 | 58256 | 56% | BW | 73166 | 0.80x |
| 1 | agg | whole SUM | - | 1495 | 3365 | 14% | BW | 1482 | 2.27x ** |
| 2 | agg | whole multi-agg | - | 1271 | 2598 | 13% | BW | 403 | 6.45x ** |
| **group** | | | | | | | | | |
| 3 | group | GROUP BY K3 count | - | 7919 | 73314 | 58% | cube | 536 | 136.65x ** |
| 4 | group | GROUP BY K3 sum | - | 7779 | 71096 | 57% | cube | 446 | 159.41x ** |
| 5 | group | GROUP BY K7 avg | - | 7759 | 70204 | 57% | cube | 504 | 139.43x ** |
| 6 | group | GROUP BY 2-col (Q1) | - | 4233 | 37848 | 56% | cube | 232 | 162.79x ** |
| 7 | group | GROUP BY datetime K2.5k | - | 201 | 636 | 20% | BW | 705 | 0.90x |
| 8 | group | GROUP BY high-card K200k | - | 12 | 76 | 39% | BW | 34 | 2.25x ** |
| 9 | group | GROUP BY vhigh-card K1.5M | - | 5 | 22 | 29% | BW | 16 | 1.38x ** |
| **filter** | | | | | | | | | |
| 10 | filter | WHERE numeric > | - | 1025 | 7198 | 44% | BW | 1224 | 5.88x ** |
| 11 | filter | WHERE BETWEEN + agg | Y | 105 | 890 | 53% | BW | 681 | 1.31x ** |
| 12 | filter | WHERE date-range (Q6) | Y | 229 | 900 | 25% | BW | 420 | 2.15x ** |
| 13 | filter | WHERE string = | - | 1521 | 2908 | 12% | BW | 528 | 5.51x ** |
| 14 | filter | WHERE IN (3) | - | 1036 | 1904 | 11% | BW | 140 | 13.60x ** |
| 15 | filter | WHERE AND/OR | - | 214 | 468 | 14% | BW | 279 | 1.68x ** |
| 16 | filter | WHERE + GROUP BY | - | 117 | 292 | 16% | BW | 402 | 0.73x |
| **distinct** | | | | | | | | | |
| 17 | distinct | DISTINCT 1-col | - | 1820 | 1766 | 6% | BW | 1272 | 1.39x ** |
| 18 | distinct | DISTINCT 2-col | - | 141 | 480 | 21% | BW | 189 | 2.54x ** |
| 19 | distinct | DISTINCT high-card | - | 48 | 159 | 21% | BW | 90 | 1.77x ** |
| 20 | distinct | COUNT(DISTINCT) low | - | 6663 | 60196 | 56% | BW | 1299 | 46.34x ** |
| 21 | distinct | COUNT(DISTINCT) high | - | 6573 | 60114 | 57% | BW | 104 | 578.02x ** |
| 22 | distinct | grouped COUNT(DISTINCT) | - | 91 | 135 | 9% | BW | 218 | 0.62x |
| **order** | | | | | | | | | |
| 23 | order | ORDER BY + LIMIT | - | 35 | 130 | 23% | BW | 53 | 2.46x ** |
| 24 | order | HAVING | - | 1895 | 16753 | 55% | cube | 726 | 23.06x ** |
| **join** | | | | | | | | | |
| 25 | join | JOIN group parent-key | - | 172 | 354 | 13% | BW | 381 | 0.93x |
| 26 | join | JOIN group child-key | - | 183 | 352 | 12% | BW | 53 | 6.64x ** |
| 27 | join | JOIN group parent-date | - | 57 | 144 | 16% | BW | 55 | 2.63x ** |
| 28 | join | JOIN + WHERE | - | 31 | 214 | 43% | BW | 55 | 3.90x ** |
| 29 | join | 3-table JOIN | - | 40 | 86 | 13% | BW | 39 | 2.21x ** |

**25/30 faster than DuckDB - median 2.54x.** 5 filter-free low-card group-bys are answered from a materialised cube (bound=cube; precomputed aggregate, not a scan). BSI filter-index engaged on 2 queries. Of the scan queries, 25/30 are memory-bandwidth-bound under concurrency (scale < 60%): the shared memory bus, not core count or RAM capacity, is the throughput wall -- so the lever is **bytes read per query** (compression, the BSI index, and -- where the shape allows -- not reading at all via the cube), not single-query latency. Bold ratios are WaveDB wins.

