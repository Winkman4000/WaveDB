# WaveDB throughput scoreboard

_TPC-H sf=1 - vs DuckDB - commit `24c2c45` - 2026-06-08 - W=16 single-thread workers per engine, 32 GB budget._

Throughput is the optimization target for a shared analytical DB. Each engine runs W single-threaded workers concurrently (one per core); the number is the **real measured** aggregate queries/sec, memory-bandwidth contention included (not extrapolated from single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the BSI filter-index engages where it pays (BSI column).

| # | category | query | BSI | WaveDB 1-wkr q/s | WaveDB @W q/s | scale | bound | DuckDB @W q/s | ratio |
|---|---|---|:-:|--:|--:|--:|:-:|--:|--:|
| **agg** | | | | | | | | | |
| 0 | agg | whole COUNT(*) | - | 6449 | 61462 | 60% | BW | 75806 | 0.81x |
| 1 | agg | whole SUM | - | 1831 | 2956 | 10% | BW | 1546 | 1.91x ** |
| 2 | agg | whole multi-agg | - | 1251 | 2995 | 15% | BW | 405 | 7.40x ** |
| **group** | | | | | | | | | |
| 3 | group | GROUP BY K3 count | - | 6969 | 63594 | 57% | BW | 518 | 122.89x ** |
| 4 | group | GROUP BY K3 sum | - | 536 | 846 | 10% | BW | 428 | 1.97x ** |
| 5 | group | GROUP BY K7 avg | - | 113 | 442 | 25% | BW | 498 | 0.89x |
| 6 | group | GROUP BY 2-col (Q1) | - | 44 | 100 | 14% | BW | 236 | 0.42x |
| 7 | group | GROUP BY datetime K2.5k | - | 202 | 612 | 19% | BW | 679 | 0.90x |
| 8 | group | GROUP BY high-card K200k | - | 12 | 74 | 39% | BW | 36 | 2.10x ** |
| 9 | group | GROUP BY vhigh-card K1.5M | - | 5 | 22 | 29% | BW | 16 | 1.34x ** |
| **filter** | | | | | | | | | |
| 10 | filter | WHERE numeric > | - | 1044 | 7074 | 42% | BW | 1233 | 5.74x ** |
| 11 | filter | WHERE BETWEEN + agg | Y | 106 | 881 | 52% | BW | 708 | 1.24x ** |
| 12 | filter | WHERE date-range (Q6) | Y | 244 | 866 | 22% | BW | 414 | 2.09x ** |
| 13 | filter | WHERE string = | - | 1449 | 1856 | 8% | BW | 526 | 3.53x ** |
| 14 | filter | WHERE IN (3) | - | 1018 | 1883 | 12% | BW | 141 | 13.35x ** |
| 15 | filter | WHERE AND/OR | - | 216 | 460 | 13% | BW | 281 | 1.64x ** |
| 16 | filter | WHERE + GROUP BY | - | 28 | 173 | 39% | BW | 393 | 0.44x |
| **distinct** | | | | | | | | | |
| 17 | distinct | DISTINCT 1-col | - | 281 | 1058 | 24% | BW | 1254 | 0.84x |
| 18 | distinct | DISTINCT 2-col | - | 79 | 471 | 37% | BW | 188 | 2.51x ** |
| 19 | distinct | DISTINCT high-card | - | 44 | 152 | 22% | BW | 92 | 1.65x ** |
| 20 | distinct | COUNT(DISTINCT) low | - | 6626 | 60224 | 57% | BW | 1300 | 46.33x ** |
| 21 | distinct | COUNT(DISTINCT) high | - | 6583 | 60232 | 57% | BW | 108 | 560.30x ** |
| 22 | distinct | grouped COUNT(DISTINCT) | - | 96 | 134 | 9% | BW | 215 | 0.63x |
| **order** | | | | | | | | | |
| 23 | order | ORDER BY + LIMIT | - | 33 | 134 | 26% | BW | 50 | 2.72x ** |
| 24 | order | HAVING | - | 196 | 857 | 27% | BW | 742 | 1.16x ** |
| **join** | | | | | | | | | |
| 25 | join | JOIN group parent-key | - | 179 | 378 | 13% | BW | 393 | 0.96x |
| 26 | join | JOIN group child-key | - | 63 | 156 | 15% | BW | 56 | 2.78x ** |
| 27 | join | JOIN group parent-date | - | 59 | 148 | 16% | BW | 56 | 2.64x ** |
| 28 | join | JOIN + WHERE | - | 32 | 228 | 44% | BW | 56 | 4.06x ** |
| 29 | join | 3-table JOIN | - | 41 | 88 | 13% | BW | 40 | 2.19x ** |

**22/30 faster than DuckDB - median 2.09x.** BSI filter-index engaged on 2 queries. 30/30 queries are memory-bandwidth-bound under concurrency (scale < 60%): the shared memory bus, not core count or RAM capacity, is the throughput wall -- so the lever is **bytes read per query** (compression + the BSI index), not single-query latency. Bold ratios are WaveDB wins.

