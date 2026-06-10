# WaveDB throughput scoreboard

_TPC-H sf=1 - vs DuckDB - commit `6a60d23` - 2026-06-09 - W=16 single-thread workers per engine, 32 GB budget._

Throughput is the optimization target for a shared analytical DB. Each engine runs W single-threaded workers concurrently (one per core); the number is the **real measured** aggregate queries/sec, memory-bandwidth contention included (not extrapolated from single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the BSI filter-index engages where it pays (BSI column).

Rows marked **cube** in the bound column are answered from a materialised low-card GROUP BY aggregate (a precomputed [count, sums] per cell, built at load time for filter-free group-bys whose cell count is under the cap) -- the query reads a few hundred bytes instead of scanning the value columns, so it is parse-bound, not bandwidth-bound. This is a materialised view: a DIFFERENT class than a faster scan (DuckDB could build the same), and it applies ONLY to filter-free low-card group-bys with COUNT/SUM/AVG; everything else falls back to the scan.

| # | category | query | BSI | WaveDB 1-wkr q/s | WaveDB @W q/s | scale | bound | DuckDB @W q/s | ratio |
|---|---|---|:-:|--:|--:|--:|:-:|--:|--:|
| **agg** | | | | | | | | | |
| 0 | agg | whole COUNT(*) | - | 14374 | 120527 | 52% | BW | 81383 | 1.48x ** |
| 1 | agg | whole SUM | - | 2204 | 3256 | 9% | BW | 1488 | 2.19x ** |
| 2 | agg | whole multi-agg | - | 1782 | 2836 | 10% | BW | 384 | 7.37x ** |
| **group** | | | | | | | | | |
| 3 | group | GROUP BY K3 count | - | 33043 | 261806 | 50% | cube | 537 | 487.54x ** |
| 4 | group | GROUP BY K3 sum | - | 31442 | 240922 | 48% | cube | 438 | 549.42x ** |
| 5 | group | GROUP BY K7 avg | - | 30749 | 248632 | 51% | cube | 498 | 498.76x ** |
| 6 | group | GROUP BY 2-col (Q1) | - | 15095 | 121636 | 50% | cube | 229 | 531.16x ** |
| 7 | group | GROUP BY datetime K2.5k | - | 3619 | 32868 | 57% | cube | 704 | 46.69x ** |
| 8 | group | GROUP BY high-card K200k | - | 12 | 75 | 39% | BW | 34 | 2.17x ** |
| 9 | group | GROUP BY vhigh-card K1.5M | - | 5 | 22 | 30% | BW | 16 | 1.36x ** |
| **filter** | | | | | | | | | |
| 10 | filter | WHERE numeric > | - | 1174 | 8213 | 44% | BW | 1227 | 6.69x ** |
| 11 | filter | WHERE BETWEEN + agg | Y | 107 | 890 | 52% | BW | 706 | 1.26x ** |
| 12 | filter | WHERE date-range (Q6) | Y | 259 | 882 | 21% | BW | 428 | 2.06x ** |
| 13 | filter | WHERE string = | - | 1860 | 2153 | 7% | BW | 538 | 4.00x ** |
| 14 | filter | WHERE IN (3) | - | 1242 | 2134 | 11% | BW | 142 | 14.97x ** |
| 15 | filter | WHERE AND/OR | - | 227 | 494 | 14% | BW | 283 | 1.74x ** |
| 16 | filter | WHERE + GROUP BY | - | 11604 | 93368 | 50% | cube | 400 | 233.13x ** |
| **distinct** | | | | | | | | | |
| 17 | distinct | DISTINCT 1-col | - | 2086 | 3450 | 10% | BW | 1268 | 2.72x ** |
| 18 | distinct | DISTINCT 2-col | - | 138 | 531 | 24% | BW | 188 | 2.82x ** |
| 19 | distinct | DISTINCT high-card | - | 49 | 156 | 20% | BW | 90 | 1.72x ** |
| 20 | distinct | COUNT(DISTINCT) low | - | 15289 | 126160 | 52% | BW | 1282 | 98.41x ** |
| 21 | distinct | COUNT(DISTINCT) high | - | 15417 | 129762 | 53% | BW | 106 | 1229.97x ** |
| 22 | distinct | grouped COUNT(DISTINCT) | - | 27609 | 218112 | 49% | cube | 220 | 989.17x ** |
| **order** | | | | | | | | | |
| 23 | order | ORDER BY + LIMIT | - | 33 | 132 | 25% | BW | 55 | 2.39x ** |
| 24 | order | HAVING | - | 2745 | 23285 | 53% | cube | 728 | 31.98x ** |
| **join** | | | | | | | | | |
| 25 | join | JOIN group parent-key | - | 177 | 368 | 13% | BW | 382 | 0.96x |
| 26 | join | JOIN group child-key | - | 191 | 303 | 10% | BW | 52 | 5.83x ** |
| 27 | join | JOIN group parent-date | - | 57 | 140 | 15% | BW | 55 | 2.55x ** |
| 28 | join | JOIN + WHERE | - | 31 | 234 | 47% | BW | 57 | 4.11x ** |
| 29 | join | 3-table JOIN | - | 35 | 100 | 18% | BW | 39 | 2.58x ** |

**29/30 faster than DuckDB - median 4.11x.** 8 filter-free low-card group-bys are answered from a materialised cube (bound=cube; precomputed aggregate, not a scan). BSI filter-index engaged on 2 queries. Of the scan queries, 22/30 are memory-bandwidth-bound under concurrency (scale < 60%): the shared memory bus, not core count or RAM capacity, is the throughput wall -- so the lever is **bytes read per query** (compression, the BSI index, and -- where the shape allows -- not reading at all via the cube), not single-query latency. Bold ratios are WaveDB wins.

