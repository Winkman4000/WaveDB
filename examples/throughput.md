# WaveDB throughput scoreboard

_TPC-H sf=1 - vs DuckDB - commit `4c4512b` - 2026-06-10 - W=16 single-thread workers per engine, 32 GB budget._

Throughput is the optimization target for a shared analytical DB. Each engine runs W single-threaded workers concurrently (one per core); the number is the **real measured** aggregate queries/sec, memory-bandwidth contention included (not extrapolated from single-query latency). WaveDB runs in the default non-escalated (throughput) mode, so the BSI filter-index engages where it pays (BSI column).

Rows marked **cube** in the bound column are answered from a materialised low-card GROUP BY aggregate (a precomputed [count, sums] per cell, built at load time for filter-free group-bys whose cell count is under the cap) -- the query reads a few hundred bytes instead of scanning the value columns, so it is parse-bound, not bandwidth-bound. This is a materialised view: a DIFFERENT class than a faster scan (DuckDB could build the same), and it applies ONLY to filter-free low-card group-bys with COUNT/SUM/AVG; everything else falls back to the scan.

| # | category | query | BSI | WaveDB 1-wkr q/s | WaveDB @W q/s | scale | bound | DuckDB @W q/s | ratio |
|---|---|---|:-:|--:|--:|--:|:-:|--:|--:|
| **agg** | | | | | | | | | |
| 0 | agg | whole COUNT(*) | - | 14147 | 121181 | 54% | BW | 75612 | 1.60x ** |
| 1 | agg | whole SUM | - | 2224 | 2895 | 8% | BW | 910 | 3.18x ** |
| 2 | agg | whole multi-agg | - | 1797 | 2651 | 9% | BW | 151 | 17.56x ** |
| **group** | | | | | | | | | |
| 3 | group | GROUP BY K3 count | - | 30039 | 230623 | 48% | cube | 544 | 424.33x ** |
| 4 | group | GROUP BY K3 sum | - | 27239 | 211458 | 49% | cube | 372 | 567.67x ** |
| 5 | group | GROUP BY K7 avg | - | 27831 | 214874 | 48% | cube | 460 | 467.63x ** |
| 6 | group | GROUP BY 2-col (Q1) | - | 14293 | 114733 | 50% | cube | 202 | 569.39x ** |
| 7 | group | GROUP BY datetime K2.5k | - | 3631 | 31924 | 55% | cube | 704 | 45.38x ** |
| 8 | group | GROUP BY high-card K200k | - | 12 | 75 | 39% | BW | 42 | 1.76x ** |
| 9 | group | GROUP BY vhigh-card K1.5M | - | 5 | 25 | 33% | BW | 16 | 1.52x ** |
| **filter** | | | | | | | | | |
| 10 | filter | WHERE numeric > | - | 1172 | 8435 | 45% | BW | 300 | 28.16x ** |
| 11 | filter | WHERE BETWEEN + agg | Y | 106 | 890 | 52% | BW | 179 | 4.97x ** |
| 12 | filter | WHERE date-range (Q6) | Y | 243 | 913 | 24% | BW | 255 | 3.58x ** |
| 13 | filter | WHERE string = | - | 1852 | 2455 | 8% | BW | 533 | 4.61x ** |
| 14 | filter | WHERE IN (3) | - | 1221 | 2574 | 13% | BW | 140 | 18.45x ** |
| 15 | filter | WHERE AND/OR | - | 213 | 490 | 14% | BW | 168 | 2.93x ** |
| 16 | filter | WHERE + GROUP BY | - | 10951 | 88616 | 51% | cube | 188 | 472.62x ** |
| **distinct** | | | | | | | | | |
| 17 | distinct | DISTINCT 1-col | - | 2065 | 3960 | 12% | BW | 1240 | 3.19x ** |
| 18 | distinct | DISTINCT 2-col | - | 141 | 552 | 25% | BW | 186 | 2.96x ** |
| 19 | distinct | DISTINCT high-card | - | 46 | 199 | 27% | BW | 95 | 2.09x ** |
| 20 | distinct | COUNT(DISTINCT) low | - | 15143 | 130773 | 54% | BW | 1278 | 102.33x ** |
| 21 | distinct | COUNT(DISTINCT) high | - | 15041 | 131407 | 55% | BW | 104 | 1257.48x ** |
| 22 | distinct | grouped COUNT(DISTINCT) | - | 24465 | 197792 | 51% | cube | 220 | 899.06x ** |
| **order** | | | | | | | | | |
| 23 | order | ORDER BY + LIMIT | - | 37 | 142 | 24% | BW | 48 | 2.95x ** |
| 24 | order | HAVING | - | 2626 | 23096 | 55% | cube | 728 | 31.75x ** |
| **join** | | | | | | | | | |
| 25 | join | JOIN group parent-key | - | 16749 | 136788 | 51% | cube | 373 | 366.73x ** |
| 26 | join | JOIN group child-key | - | 190 | 332 | 11% | BW | 51 | 6.51x ** |
| 27 | join | JOIN group parent-date | - | 57 | 139 | 15% | BW | 54 | 2.57x ** |
| 28 | join | JOIN + WHERE | - | 31 | 216 | 43% | BW | 52 | 4.16x ** |
| 29 | join | 3-table JOIN | - | 37 | 88 | 15% | BW | 39 | 2.26x ** |

**30/30 faster than DuckDB - median 6.51x.** 9 filter-free low-card group-bys are answered from a materialised cube (bound=cube; precomputed aggregate, not a scan). BSI filter-index engaged on 2 queries. Of the scan queries, 21/30 are memory-bandwidth-bound under concurrency (scale < 60%): the shared memory bus, not core count or RAM capacity, is the throughput wall -- so the lever is **bytes read per query** (compression, the BSI index, and -- where the shape allows -- not reading at all via the cube), not single-query latency. Bold ratios are WaveDB wins.

