# WaveDB throughput scoreboard

_TPC-H sf=1 - vs DuckDB - commit `4077ac7` - 2026-06-08 - W=16 single-thread workers per engine, 32 GB budget._

Throughput is the optimization target for a shared analytical DB. Each engine runs W single-threaded workers concurrently (one per core); the number is the **real measured** aggregate queries/sec, memory-bandwidth contention included (not extrapolated from single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the BSI filter-index engages where it pays (BSI column).

| # | category | query | BSI | WaveDB @W q/s | DuckDB @W q/s | ratio | W |
|---|---|---|:-:|--:|--:|--:|--:|
| **agg** | | | | | | | |
| 0 | agg | whole COUNT(*) | - | 60741 | 80274 | 0.76x | 16 |
| 1 | agg | whole SUM | - | 2560 | 1548 | 1.65x ** | 16 |
| 2 | agg | whole multi-agg | - | 3460 | 394 | 8.78x ** | 16 |
| **group** | | | | | | | |
| 3 | group | GROUP BY K3 count | - | 67200 | 537 | 125.14x ** | 16 |
| 4 | group | GROUP BY K3 sum | - | 700 | 444 | 1.58x ** | 16 |
| 5 | group | GROUP BY K7 avg | - | 569 | 504 | 1.13x ** | 16 |
| 6 | group | GROUP BY 2-col (Q1) | - | 97 | 227 | 0.43x | 16 |
| 7 | group | GROUP BY datetime K2.5k | - | 683 | 694 | 0.98x | 16 |
| 8 | group | GROUP BY high-card K200k | - | 76 | 32 | 2.36x ** | 16 |
| 9 | group | GROUP BY vhigh-card K1.5M | - | 24 | 16 | 1.45x ** | 16 |
| **filter** | | | | | | | |
| 10 | filter | WHERE numeric > | - | 6720 | 1190 | 5.65x ** | 16 |
| 11 | filter | WHERE BETWEEN + agg | Y | 888 | 676 | 1.31x ** | 16 |
| 12 | filter | WHERE date-range (Q6) | Y | 887 | 402 | 2.21x ** | 16 |
| 13 | filter | WHERE string = | - | 2138 | 531 | 4.03x ** | 16 |
| 14 | filter | WHERE IN (3) | - | 1842 | 142 | 12.98x ** | 16 |
| 15 | filter | WHERE AND/OR | - | 564 | 279 | 2.02x ** | 16 |
| 16 | filter | WHERE + GROUP BY | - | 179 | 398 | 0.45x | 16 |
| **distinct** | | | | | | | |
| 17 | distinct | DISTINCT 1-col | - | 1071 | 1250 | 0.86x | 16 |
| 18 | distinct | DISTINCT 2-col | - | 498 | 186 | 2.67x ** | 16 |
| 19 | distinct | DISTINCT high-card | - | 154 | 91 | 1.70x ** | 16 |
| 20 | distinct | COUNT(DISTINCT) low | - | 61374 | 1298 | 47.30x ** | 16 |
| 21 | distinct | COUNT(DISTINCT) high | - | 60056 | 105 | 571.96x ** | 16 |
| 22 | distinct | grouped COUNT(DISTINCT) | - | 150 | 223 | 0.67x | 16 |
| **order** | | | | | | | |
| 23 | order | ORDER BY + LIMIT | - | 32 | 50 | 0.64x | 16 |
| 24 | order | HAVING | - | 892 | 728 | 1.22x ** | 16 |
| **join** | | | | | | | |
| 25 | join | JOIN group parent-key | - | 353 | 383 | 0.92x | 16 |
| 26 | join | JOIN group child-key | - | 167 | 52 | 3.18x ** | 16 |
| 27 | join | JOIN group parent-date | - | 136 | 52 | 2.60x ** | 16 |
| 28 | join | JOIN + WHERE | - | 226 | 56 | 4.04x ** | 16 |
| 29 | join | 3-table JOIN | - | 90 | 38 | 2.36x ** | 16 |

**22/30 faster than DuckDB - median 2.02x - BSI filter-index engaged on 2 queries.** Bold ratios are WaveDB wins.

