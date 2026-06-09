# WaveDB throughput scoreboard

_TPC-H sf=1 - vs DuckDB - commit `2d2be69` - 2026-06-09 - W=16 single-thread workers per engine, 32 GB budget._

Throughput is the optimization target for a shared analytical DB. Each engine runs W single-threaded workers concurrently (one per core); the number is the **real measured** aggregate queries/sec, memory-bandwidth contention included (not extrapolated from single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the BSI filter-index engages where it pays (BSI column).

Rows marked **cube** in the bound column are answered from a materialised low-card GROUP BY aggregate (a precomputed [count, sums] per cell, built at load time for filter-free group-bys whose cell count is under the cap) -- the query reads a few hundred bytes instead of scanning the value columns, so it is parse-bound, not bandwidth-bound. This is a materialised view: a DIFFERENT class than a faster scan (DuckDB could build the same), and it applies ONLY to filter-free low-card group-bys with COUNT/SUM/AVG; everything else falls back to the scan.

| # | category | query | BSI | WaveDB 1-wkr q/s | WaveDB @W q/s | scale | bound | DuckDB @W q/s | ratio |
|---|---|---|:-:|--:|--:|--:|:-:|--:|--:|
| **agg** | | | | | | | | | |
| 0 | agg | whole COUNT(*) | - | 6783 | 61949 | 57% | BW | 81434 | 0.76x |
| 1 | agg | whole SUM | - | 1857 | 2600 | 9% | BW | 1578 | 1.65x ** |
| 2 | agg | whole multi-agg | - | 1284 | 2406 | 12% | BW | 406 | 5.93x ** |
| **group** | | | | | | | | | |
| 3 | group | GROUP BY K3 count | - | 7866 | 72786 | 58% | cube | 542 | 134.42x ** |
| 4 | group | GROUP BY K3 sum | - | 7797 | 69953 | 56% | cube | 446 | 156.85x ** |
| 5 | group | GROUP BY K7 avg | - | 7715 | 70576 | 57% | cube | 504 | 140.03x ** |
| 6 | group | GROUP BY 2-col (Q1) | - | 4251 | 37996 | 56% | cube | 235 | 161.68x ** |
| 7 | group | GROUP BY datetime K2.5k | - | 199 | 670 | 21% | BW | 687 | 0.98x |
| 8 | group | GROUP BY high-card K200k | - | 12 | 73 | 38% | BW | 33 | 2.21x ** |
| 9 | group | GROUP BY vhigh-card K1.5M | - | 5 | 21 | 28% | BW | 16 | 1.27x ** |
| **filter** | | | | | | | | | |
| 10 | filter | WHERE numeric > | - | 1012 | 6874 | 42% | BW | 1222 | 5.63x ** |
| 11 | filter | WHERE BETWEEN + agg | Y | 105 | 885 | 53% | BW | 706 | 1.25x ** |
| 12 | filter | WHERE date-range (Q6) | Y | 237 | 804 | 21% | BW | 426 | 1.89x ** |
| 13 | filter | WHERE string = | - | 1525 | 2318 | 9% | BW | 535 | 4.33x ** |
| 14 | filter | WHERE IN (3) | - | 1042 | 1767 | 11% | BW | 142 | 12.40x ** |
| 15 | filter | WHERE AND/OR | - | 216 | 470 | 14% | BW | 284 | 1.65x ** |
| 16 | filter | WHERE + GROUP BY | - | 116 | 292 | 16% | BW | 394 | 0.74x |
| **distinct** | | | | | | | | | |
| 17 | distinct | DISTINCT 1-col | - | 1796 | 2384 | 8% | BW | 1244 | 1.92x ** |
| 18 | distinct | DISTINCT 2-col | - | 363 | 544 | 9% | BW | 184 | 2.95x ** |
| 19 | distinct | DISTINCT high-card | - | 47 | 161 | 21% | BW | 92 | 1.75x ** |
| 20 | distinct | COUNT(DISTINCT) low | - | 6601 | 60586 | 57% | BW | 1300 | 46.59x ** |
| 21 | distinct | COUNT(DISTINCT) high | - | 6532 | 59188 | 57% | BW | 103 | 574.64x ** |
| 22 | distinct | grouped COUNT(DISTINCT) | - | 91 | 170 | 12% | BW | 225 | 0.76x |
| **order** | | | | | | | | | |
| 23 | order | ORDER BY + LIMIT | - | 36 | 129 | 22% | BW | 50 | 2.55x ** |
| 24 | order | HAVING | - | 1892 | 17022 | 56% | cube | 743 | 22.91x ** |
| **join** | | | | | | | | | |
| 25 | join | JOIN group parent-key | - | 173 | 369 | 13% | BW | 345 | 1.07x ** |
| 26 | join | JOIN group child-key | - | 184 | 341 | 12% | BW | 54 | 6.31x ** |
| 27 | join | JOIN group parent-date | - | 56 | 150 | 17% | BW | 54 | 2.77x ** |
| 28 | join | JOIN + WHERE | - | 31 | 220 | 45% | BW | 57 | 3.85x ** |
| 29 | join | 3-table JOIN | - | 38 | 94 | 15% | BW | 40 | 2.37x ** |

**26/30 faster than DuckDB - median 2.77x.** 5 filter-free low-card group-bys are answered from a materialised cube (bound=cube; precomputed aggregate, not a scan). BSI filter-index engaged on 2 queries. Of the scan queries, 25/30 are memory-bandwidth-bound under concurrency (scale < 60%): the shared memory bus, not core count or RAM capacity, is the throughput wall -- so the lever is **bytes read per query** (compression, the BSI index, and -- where the shape allows -- not reading at all via the cube), not single-query latency. Bold ratios are WaveDB wins.

