# WaveDB throughput scoreboard

_TPC-H sf=1 - vs DuckDB - commit `a93e44b` - 2026-06-09 - W=16 single-thread workers per engine, 32 GB budget._

Throughput is the optimization target for a shared analytical DB. Each engine runs W single-threaded workers concurrently (one per core); the number is the **real measured** aggregate queries/sec, memory-bandwidth contention included (not extrapolated from single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the BSI filter-index engages where it pays (BSI column).

Rows marked **cube** in the bound column are answered from a materialised low-card GROUP BY aggregate (a precomputed [count, sums] per cell, built at load time for filter-free group-bys whose cell count is under the cap) -- the query reads a few hundred bytes instead of scanning the value columns, so it is parse-bound, not bandwidth-bound. This is a materialised view: a DIFFERENT class than a faster scan (DuckDB could build the same), and it applies ONLY to filter-free low-card group-bys with COUNT/SUM/AVG; everything else falls back to the scan.

| # | category | query | BSI | WaveDB 1-wkr q/s | WaveDB @W q/s | scale | bound | DuckDB @W q/s | ratio |
|---|---|---|:-:|--:|--:|--:|:-:|--:|--:|
| **agg** | | | | | | | | | |
| 0 | agg | whole COUNT(*) | - | 14193 | 121290 | 53% | BW | 76280 | 1.59x ** |
| 1 | agg | whole SUM | - | 2235 | 2802 | 8% | BW | 1524 | 1.84x ** |
| 2 | agg | whole multi-agg | - | 1775 | 3050 | 11% | BW | 398 | 7.67x ** |
| **group** | | | | | | | | | |
| 3 | group | GROUP BY K3 count | - | 32977 | 266048 | 50% | cube | 532 | 499.62x ** |
| 4 | group | GROUP BY K3 sum | - | 31245 | 247414 | 49% | cube | 446 | 554.12x ** |
| 5 | group | GROUP BY K7 avg | - | 30738 | 244285 | 50% | cube | 502 | 486.14x ** |
| 6 | group | GROUP BY 2-col (Q1) | - | 14937 | 120056 | 50% | cube | 233 | 515.26x ** |
| 7 | group | GROUP BY datetime K2.5k | - | 3775 | 32921 | 55% | cube | 702 | 46.93x ** |
| 8 | group | GROUP BY high-card K200k | - | 12 | 72 | 38% | BW | 34 | 2.10x ** |
| 9 | group | GROUP BY vhigh-card K1.5M | - | 5 | 23 | 31% | BW | 16 | 1.44x ** |
| **filter** | | | | | | | | | |
| 10 | filter | WHERE numeric > | - | 1155 | 7929 | 43% | BW | 1209 | 6.56x ** |
| 11 | filter | WHERE BETWEEN + agg | Y | 106 | 916 | 54% | BW | 692 | 1.33x ** |
| 12 | filter | WHERE date-range (Q6) | Y | 248 | 908 | 23% | BW | 422 | 2.15x ** |
| 13 | filter | WHERE string = | - | 1843 | 2344 | 8% | BW | 514 | 4.56x ** |
| 14 | filter | WHERE IN (3) | - | 1225 | 2093 | 11% | BW | 141 | 14.84x ** |
| 15 | filter | WHERE AND/OR | - | 224 | 418 | 12% | BW | 283 | 1.48x ** |
| 16 | filter | WHERE + GROUP BY | - | 121 | 296 | 15% | BW | 400 | 0.74x |
| **distinct** | | | | | | | | | |
| 17 | distinct | DISTINCT 1-col | - | 2108 | 2458 | 7% | BW | 1258 | 1.95x ** |
| 18 | distinct | DISTINCT 2-col | - | 143 | 390 | 17% | BW | 190 | 2.06x ** |
| 19 | distinct | DISTINCT high-card | - | 49 | 157 | 20% | BW | 91 | 1.73x ** |
| 20 | distinct | COUNT(DISTINCT) low | - | 15220 | 123572 | 51% | BW | 1292 | 95.64x ** |
| 21 | distinct | COUNT(DISTINCT) high | - | 15299 | 130414 | 53% | BW | 106 | 1230.32x ** |
| 22 | distinct | grouped COUNT(DISTINCT) | - | 27456 | 216266 | 49% | cube | 218 | 989.78x ** |
| **order** | | | | | | | | | |
| 23 | order | ORDER BY + LIMIT | - | 36 | 129 | 22% | BW | 48 | 2.66x ** |
| 24 | order | HAVING | - | 2741 | 23534 | 54% | cube | 726 | 32.42x ** |
| **join** | | | | | | | | | |
| 25 | join | JOIN group parent-key | - | 177 | 366 | 13% | BW | 375 | 0.98x |
| 26 | join | JOIN group child-key | - | 186 | 326 | 11% | BW | 54 | 6.08x ** |
| 27 | join | JOIN group parent-date | - | 57 | 150 | 17% | BW | 54 | 2.79x ** |
| 28 | join | JOIN + WHERE | - | 31 | 226 | 45% | BW | 56 | 4.04x ** |
| 29 | join | 3-table JOIN | - | 37 | 83 | 14% | BW | 38 | 2.16x ** |

**28/30 faster than DuckDB - median 4.04x.** 7 filter-free low-card group-bys are answered from a materialised cube (bound=cube; precomputed aggregate, not a scan). BSI filter-index engaged on 2 queries. Of the scan queries, 23/30 are memory-bandwidth-bound under concurrency (scale < 60%): the shared memory bus, not core count or RAM capacity, is the throughput wall -- so the lever is **bytes read per query** (compression, the BSI index, and -- where the shape allows -- not reading at all via the cube), not single-query latency. Bold ratios are WaveDB wins.

