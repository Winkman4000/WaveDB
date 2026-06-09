# WaveDB throughput scoreboard

_TPC-H sf=1 - vs DuckDB - commit `f1fd4d9` - 2026-06-09 - W=16 single-thread workers per engine, 32 GB budget._

Throughput is the optimization target for a shared analytical DB. Each engine runs W single-threaded workers concurrently (one per core); the number is the **real measured** aggregate queries/sec, memory-bandwidth contention included (not extrapolated from single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the BSI filter-index engages where it pays (BSI column).

| # | category | query | BSI | WaveDB 1-wkr q/s | WaveDB @W q/s | scale | bound | DuckDB @W q/s | ratio |
|---|---|---|:-:|--:|--:|--:|:-:|--:|--:|
| **agg** | | | | | | | | | |
| 0 | agg | whole COUNT(*) | - | 6639 | 62041 | 58% | BW | 81604 | 0.76x |
| 1 | agg | whole SUM | - | 1842 | 2918 | 10% | BW | 1574 | 1.85x ** |
| 2 | agg | whole multi-agg | - | 1277 | 2398 | 12% | BW | 408 | 5.89x ** |
| **group** | | | | | | | | | |
| 3 | group | GROUP BY K3 count | - | 7151 | 62316 | 54% | BW | 537 | 116.04x ** |
| 4 | group | GROUP BY K3 sum | - | 533 | 688 | 8% | BW | 446 | 1.55x ** |
| 5 | group | GROUP BY K7 avg | - | 121 | 546 | 28% | BW | 504 | 1.08x ** |
| 6 | group | GROUP BY 2-col (Q1) | - | 43 | 110 | 16% | BW | 232 | 0.48x |
| 7 | group | GROUP BY datetime K2.5k | - | 200 | 708 | 22% | BW | 703 | 1.01x ** |
| 8 | group | GROUP BY high-card K200k | - | 12 | 76 | 39% | BW | 34 | 2.22x ** |
| 9 | group | GROUP BY vhigh-card K1.5M | - | 5 | 22 | 29% | BW | 16 | 1.38x ** |
| **filter** | | | | | | | | | |
| 10 | filter | WHERE numeric > | - | 1034 | 6999 | 42% | BW | 1218 | 5.74x ** |
| 11 | filter | WHERE BETWEEN + agg | Y | 106 | 884 | 52% | BW | 706 | 1.25x ** |
| 12 | filter | WHERE date-range (Q6) | Y | 238 | 898 | 24% | BW | 400 | 2.24x ** |
| 13 | filter | WHERE string = | - | 1523 | 2084 | 9% | BW | 536 | 3.89x ** |
| 14 | filter | WHERE IN (3) | - | 1045 | 2392 | 14% | BW | 142 | 16.90x ** |
| 15 | filter | WHERE AND/OR | - | 218 | 528 | 15% | BW | 286 | 1.85x ** |
| 16 | filter | WHERE + GROUP BY | - | 118 | 296 | 16% | BW | 399 | 0.74x |
| **distinct** | | | | | | | | | |
| 17 | distinct | DISTINCT 1-col | - | 1817 | 3263 | 11% | BW | 1236 | 2.64x ** |
| 18 | distinct | DISTINCT 2-col | - | 141 | 444 | 20% | BW | 188 | 2.36x ** |
| 19 | distinct | DISTINCT high-card | - | 47 | 162 | 22% | BW | 94 | 1.73x ** |
| 20 | distinct | COUNT(DISTINCT) low | - | 6613 | 60923 | 58% | BW | 1303 | 46.76x ** |
| 21 | distinct | COUNT(DISTINCT) high | - | 6580 | 60256 | 57% | BW | 106 | 568.46x ** |
| 22 | distinct | grouped COUNT(DISTINCT) | - | 93 | 146 | 10% | BW | 222 | 0.66x |
| **order** | | | | | | | | | |
| 23 | order | ORDER BY + LIMIT | - | 34 | 130 | 24% | BW | 52 | 2.49x ** |
| 24 | order | HAVING | - | 204 | 718 | 22% | BW | 734 | 0.98x |
| **join** | | | | | | | | | |
| 25 | join | JOIN group parent-key | - | 171 | 364 | 13% | BW | 386 | 0.94x |
| 26 | join | JOIN group child-key | - | 183 | 314 | 11% | BW | 54 | 5.81x ** |
| 27 | join | JOIN group parent-date | - | 57 | 148 | 16% | BW | 55 | 2.69x ** |
| 28 | join | JOIN + WHERE | - | 31 | 227 | 45% | BW | 52 | 4.37x ** |
| 29 | join | 3-table JOIN | - | 39 | 86 | 14% | BW | 40 | 2.19x ** |

**24/30 faster than DuckDB - median 2.22x.** BSI filter-index engaged on 2 queries. 30/30 queries are memory-bandwidth-bound under concurrency (scale < 60%): the shared memory bus, not core count or RAM capacity, is the throughput wall -- so the lever is **bytes read per query** (compression + the BSI index), not single-query latency. Bold ratios are WaveDB wins.

