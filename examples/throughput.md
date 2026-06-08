# WaveDB throughput scoreboard

_TPC-H sf=1 - vs DuckDB - commit `978c356` - 2026-06-08 - W=16 single-thread workers per engine, 32 GB budget._

Throughput is the optimization target for a shared analytical DB. Each engine runs W single-threaded workers concurrently (one per core); the number is the **real measured** aggregate queries/sec, memory-bandwidth contention included (not extrapolated from single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the BSI filter-index engages where it pays (BSI column).

| # | category | query | BSI | WaveDB 1-wkr q/s | WaveDB @W q/s | scale | bound | DuckDB @W q/s | ratio |
|---|---|---|:-:|--:|--:|--:|:-:|--:|--:|
| **agg** | | | | | | | | | |
| 0 | agg | whole COUNT(*) | - | 6637 | 61648 | 58% | BW | 80013 | 0.77x |
| 1 | agg | whole SUM | - | 1820 | 2576 | 9% | BW | 1558 | 1.65x ** |
| 2 | agg | whole multi-agg | - | 1277 | 2696 | 13% | BW | 402 | 6.71x ** |
| **group** | | | | | | | | | |
| 3 | group | GROUP BY K3 count | - | 7159 | 63360 | 55% | BW | 537 | 117.99x ** |
| 4 | group | GROUP BY K3 sum | - | 521 | 695 | 8% | BW | 442 | 1.57x ** |
| 5 | group | GROUP BY K7 avg | - | 59 | 496 | 52% | BW | 505 | 0.98x |
| 6 | group | GROUP BY 2-col (Q1) | - | 43 | 94 | 14% | BW | 232 | 0.41x |
| 7 | group | GROUP BY datetime K2.5k | - | 198 | 690 | 22% | BW | 700 | 0.99x |
| 8 | group | GROUP BY high-card K200k | - | 12 | 76 | 39% | BW | 32 | 2.36x ** |
| 9 | group | GROUP BY vhigh-card K1.5M | - | 5 | 22 | 29% | BW | 16 | 1.30x ** |
| **filter** | | | | | | | | | |
| 10 | filter | WHERE numeric > | - | 1014 | 6864 | 42% | BW | 1202 | 5.71x ** |
| 11 | filter | WHERE BETWEEN + agg | Y | 104 | 898 | 54% | BW | 696 | 1.29x ** |
| 12 | filter | WHERE date-range (Q6) | Y | 225 | 876 | 24% | BW | 423 | 2.07x ** |
| 13 | filter | WHERE string = | - | 1495 | 2368 | 10% | BW | 532 | 4.45x ** |
| 14 | filter | WHERE IN (3) | - | 1057 | 1912 | 11% | BW | 141 | 13.56x ** |
| 15 | filter | WHERE AND/OR | - | 214 | 484 | 14% | BW | 282 | 1.72x ** |
| 16 | filter | WHERE + GROUP BY | - | 26 | 171 | 41% | BW | 399 | 0.43x |
| **distinct** | | | | | | | | | |
| 17 | distinct | DISTINCT 1-col | - | 285 | 1005 | 22% | BW | 1245 | 0.81x |
| 18 | distinct | DISTINCT 2-col | - | 136 | 540 | 25% | BW | 187 | 2.89x ** |
| 19 | distinct | DISTINCT high-card | - | 47 | 158 | 21% | BW | 92 | 1.73x ** |
| 20 | distinct | COUNT(DISTINCT) low | - | 6569 | 59692 | 57% | BW | 1284 | 46.51x ** |
| 21 | distinct | COUNT(DISTINCT) high | - | 6520 | 59864 | 57% | BW | 104 | 572.87x ** |
| 22 | distinct | grouped COUNT(DISTINCT) | - | 91 | 144 | 10% | BW | 222 | 0.65x |
| **order** | | | | | | | | | |
| 23 | order | ORDER BY + LIMIT | - | 6 | 32 | 33% | BW | 52 | 0.62x |
| 24 | order | HAVING | - | 201 | 872 | 27% | BW | 733 | 1.19x ** |
| **join** | | | | | | | | | |
| 25 | join | JOIN group parent-key | - | 157 | 359 | 14% | BW | 379 | 0.95x |
| 26 | join | JOIN group child-key | - | 60 | 186 | 19% | BW | 52 | 3.54x ** |
| 27 | join | JOIN group parent-date | - | 56 | 163 | 18% | BW | 54 | 3.05x ** |
| 28 | join | JOIN + WHERE | - | 31 | 218 | 44% | BW | 56 | 3.94x ** |
| 29 | join | 3-table JOIN | - | 37 | 100 | 17% | BW | 38 | 2.65x ** |

**21/30 faster than DuckDB - median 1.73x.** BSI filter-index engaged on 2 queries. 30/30 queries are memory-bandwidth-bound under concurrency (scale < 60%): the shared memory bus, not core count or RAM capacity, is the throughput wall -- so the lever is **bytes read per query** (compression + the BSI index), not single-query latency. Bold ratios are WaveDB wins.

