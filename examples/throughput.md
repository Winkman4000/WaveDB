# WaveDB throughput scoreboard

_TPC-H sf=1 - vs DuckDB - commit `3bba63b` - 2026-06-09 - W=16 single-thread workers per engine, 32 GB budget._

Throughput is the optimization target for a shared analytical DB. Each engine runs W single-threaded workers concurrently (one per core); the number is the **real measured** aggregate queries/sec, memory-bandwidth contention included (not extrapolated from single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the BSI filter-index engages where it pays (BSI column).

Rows marked **cube** in the bound column are answered from a materialised low-card GROUP BY aggregate (a precomputed [count, sums] per cell, built at load time for filter-free group-bys whose cell count is under the cap) -- the query reads a few hundred bytes instead of scanning the value columns, so it is parse-bound, not bandwidth-bound. This is a materialised view: a DIFFERENT class than a faster scan (DuckDB could build the same), and it applies ONLY to filter-free low-card group-bys with COUNT/SUM/AVG; everything else falls back to the scan.

| # | category | query | BSI | WaveDB 1-wkr q/s | WaveDB @W q/s | scale | bound | DuckDB @W q/s | ratio |
|---|---|---|:-:|--:|--:|--:|:-:|--:|--:|
| **agg** | | | | | | | | | |
| 0 | agg | whole COUNT(*) | - | 6707 | 61437 | 57% | BW | 79914 | 0.77x |
| 1 | agg | whole SUM | - | 1848 | 2938 | 10% | BW | 1586 | 1.85x ** |
| 2 | agg | whole multi-agg | - | 1276 | 2663 | 13% | BW | 403 | 6.61x ** |
| **group** | | | | | | | | | |
| 3 | group | GROUP BY K3 count | - | 7958 | 71625 | 56% | cube | 542 | 132.27x ** |
| 4 | group | GROUP BY K3 sum | - | 7800 | 70166 | 56% | cube | 445 | 157.68x ** |
| 5 | group | GROUP BY K7 avg | - | 7745 | 70913 | 57% | cube | 500 | 141.68x ** |
| 6 | group | GROUP BY 2-col (Q1) | - | 4207 | 37830 | 56% | cube | 231 | 163.77x ** |
| 7 | group | GROUP BY datetime K2.5k | - | 2615 | 23574 | 56% | cube | 700 | 33.70x ** |
| 8 | group | GROUP BY high-card K200k | - | 12 | 76 | 39% | BW | 32 | 2.36x ** |
| 9 | group | GROUP BY vhigh-card K1.5M | - | 5 | 23 | 31% | BW | 16 | 1.44x ** |
| **filter** | | | | | | | | | |
| 10 | filter | WHERE numeric > | - | 1027 | 6926 | 42% | BW | 1212 | 5.71x ** |
| 11 | filter | WHERE BETWEEN + agg | Y | 105 | 903 | 54% | BW | 697 | 1.30x ** |
| 12 | filter | WHERE date-range (Q6) | Y | 223 | 846 | 24% | BW | 426 | 1.99x ** |
| 13 | filter | WHERE string = | - | 1471 | 2028 | 9% | BW | 530 | 3.82x ** |
| 14 | filter | WHERE IN (3) | - | 1049 | 2062 | 12% | BW | 143 | 14.42x ** |
| 15 | filter | WHERE AND/OR | - | 217 | 664 | 19% | BW | 280 | 2.37x ** |
| 16 | filter | WHERE + GROUP BY | - | 115 | 295 | 16% | BW | 396 | 0.75x |
| **distinct** | | | | | | | | | |
| 17 | distinct | DISTINCT 1-col | - | 1812 | 2184 | 8% | BW | 1267 | 1.72x ** |
| 18 | distinct | DISTINCT 2-col | - | 133 | 475 | 22% | BW | 188 | 2.53x ** |
| 19 | distinct | DISTINCT high-card | - | 47 | 149 | 20% | BW | 93 | 1.60x ** |
| 20 | distinct | COUNT(DISTINCT) low | - | 6501 | 60281 | 58% | BW | 1292 | 46.66x ** |
| 21 | distinct | COUNT(DISTINCT) high | - | 6653 | 60488 | 57% | BW | 106 | 570.64x ** |
| 22 | distinct | grouped COUNT(DISTINCT) | - | 91 | 151 | 10% | BW | 220 | 0.68x |
| **order** | | | | | | | | | |
| 23 | order | ORDER BY + LIMIT | - | 36 | 136 | 24% | BW | 49 | 2.77x ** |
| 24 | order | HAVING | - | 1880 | 16708 | 56% | cube | 731 | 22.86x ** |
| **join** | | | | | | | | | |
| 25 | join | JOIN group parent-key | - | 179 | 358 | 12% | BW | 382 | 0.94x |
| 26 | join | JOIN group child-key | - | 181 | 324 | 11% | BW | 54 | 6.05x ** |
| 27 | join | JOIN group parent-date | - | 57 | 142 | 16% | BW | 54 | 2.60x ** |
| 28 | join | JOIN + WHERE | - | 31 | 235 | 47% | BW | 56 | 4.16x ** |
| 29 | join | 3-table JOIN | - | 37 | 88 | 15% | BW | 38 | 2.32x ** |

**26/30 faster than DuckDB - median 2.77x.** 6 filter-free low-card group-bys are answered from a materialised cube (bound=cube; precomputed aggregate, not a scan). BSI filter-index engaged on 2 queries. Of the scan queries, 24/30 are memory-bandwidth-bound under concurrency (scale < 60%): the shared memory bus, not core count or RAM capacity, is the throughput wall -- so the lever is **bytes read per query** (compression, the BSI index, and -- where the shape allows -- not reading at all via the cube), not single-query latency. Bold ratios are WaveDB wins.

