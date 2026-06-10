# WaveDB throughput scoreboard

_TPC-H sf=1 - vs DuckDB - commit `b44ed48` - 2026-06-09 - W=16 single-thread workers per engine, 32 GB budget._

Throughput is the optimization target for a shared analytical DB. Each engine runs W single-threaded workers concurrently (one per core); the number is the **real measured** aggregate queries/sec, memory-bandwidth contention included (not extrapolated from single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the BSI filter-index engages where it pays (BSI column).

Rows marked **cube** in the bound column are answered from a materialised low-card GROUP BY aggregate (a precomputed [count, sums] per cell, built at load time for filter-free group-bys whose cell count is under the cap) -- the query reads a few hundred bytes instead of scanning the value columns, so it is parse-bound, not bandwidth-bound. This is a materialised view: a DIFFERENT class than a faster scan (DuckDB could build the same), and it applies ONLY to filter-free low-card group-bys with COUNT/SUM/AVG; everything else falls back to the scan.

| # | category | query | BSI | WaveDB 1-wkr q/s | WaveDB @W q/s | scale | bound | DuckDB @W q/s | ratio |
|---|---|---|:-:|--:|--:|--:|:-:|--:|--:|
| **agg** | | | | | | | | | |
| 0 | agg | whole COUNT(*) | - | 6763 | 61945 | 57% | BW | 80492 | 0.77x |
| 1 | agg | whole SUM | - | 1841 | 2822 | 10% | BW | 1572 | 1.80x ** |
| 2 | agg | whole multi-agg | - | 1267 | 2438 | 12% | BW | 402 | 6.07x ** |
| **group** | | | | | | | | | |
| 3 | group | GROUP BY K3 count | - | 7867 | 71487 | 57% | cube | 537 | 133.12x ** |
| 4 | group | GROUP BY K3 sum | - | 7680 | 69750 | 57% | cube | 446 | 156.57x ** |
| 5 | group | GROUP BY K7 avg | - | 7739 | 68992 | 56% | cube | 500 | 137.98x ** |
| 6 | group | GROUP BY 2-col (Q1) | - | 4231 | 37652 | 56% | cube | 230 | 163.70x ** |
| 7 | group | GROUP BY datetime K2.5k | - | 2577 | 23354 | 57% | cube | 697 | 33.51x ** |
| 8 | group | GROUP BY high-card K200k | - | 11 | 72 | 39% | BW | 32 | 2.23x ** |
| 9 | group | GROUP BY vhigh-card K1.5M | - | 5 | 20 | 27% | BW | 16 | 1.28x ** |
| **filter** | | | | | | | | | |
| 10 | filter | WHERE numeric > | - | 1031 | 6791 | 41% | BW | 1212 | 5.61x ** |
| 11 | filter | WHERE BETWEEN + agg | Y | 105 | 900 | 54% | BW | 696 | 1.29x ** |
| 12 | filter | WHERE date-range (Q6) | Y | 236 | 895 | 24% | BW | 426 | 2.10x ** |
| 13 | filter | WHERE string = | - | 1503 | 1802 | 7% | BW | 527 | 3.42x ** |
| 14 | filter | WHERE IN (3) | - | 1022 | 1878 | 11% | BW | 141 | 13.32x ** |
| 15 | filter | WHERE AND/OR | - | 217 | 433 | 12% | BW | 283 | 1.53x ** |
| 16 | filter | WHERE + GROUP BY | - | 115 | 295 | 16% | BW | 398 | 0.74x |
| **distinct** | | | | | | | | | |
| 17 | distinct | DISTINCT 1-col | - | 1815 | 3072 | 11% | BW | 1254 | 2.45x ** |
| 18 | distinct | DISTINCT 2-col | - | 134 | 414 | 19% | BW | 188 | 2.19x ** |
| 19 | distinct | DISTINCT high-card | - | 47 | 160 | 21% | BW | 93 | 1.72x ** |
| 20 | distinct | COUNT(DISTINCT) low | - | 6585 | 59377 | 56% | BW | 1291 | 45.99x ** |
| 21 | distinct | COUNT(DISTINCT) high | - | 6595 | 60825 | 58% | BW | 108 | 563.19x ** |
| 22 | distinct | grouped COUNT(DISTINCT) | - | 6879 | 61872 | 56% | cube | 218 | 283.17x ** |
| **order** | | | | | | | | | |
| 23 | order | ORDER BY + LIMIT | - | 34 | 133 | 24% | BW | 52 | 2.53x ** |
| 24 | order | HAVING | - | 1887 | 16964 | 56% | cube | 738 | 23.00x ** |
| **join** | | | | | | | | | |
| 25 | join | JOIN group parent-key | - | 170 | 350 | 13% | BW | 382 | 0.92x |
| 26 | join | JOIN group child-key | - | 186 | 326 | 11% | BW | 50 | 6.46x ** |
| 27 | join | JOIN group parent-date | - | 55 | 145 | 16% | BW | 54 | 2.66x ** |
| 28 | join | JOIN + WHERE | - | 31 | 216 | 43% | BW | 56 | 3.81x ** |
| 29 | join | 3-table JOIN | - | 37 | 90 | 15% | BW | 38 | 2.32x ** |

**27/30 faster than DuckDB - median 3.42x.** 7 filter-free low-card group-bys are answered from a materialised cube (bound=cube; precomputed aggregate, not a scan). BSI filter-index engaged on 2 queries. Of the scan queries, 23/30 are memory-bandwidth-bound under concurrency (scale < 60%): the shared memory bus, not core count or RAM capacity, is the throughput wall -- so the lever is **bytes read per query** (compression, the BSI index, and -- where the shape allows -- not reading at all via the cube), not single-query latency. Bold ratios are WaveDB wins.

