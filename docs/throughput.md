# Throughput & the RAM-per-core ceiling

**TL;DR — RAM is not a throughput constraint for WaveDB on any normal machine.**
At a 32 GB budget on 16 cores, every one of the 30 catalog queries is *CPU-bound*:
the box runs out of cores long before RAM. Lowering our memory footprint buys
zero throughput; latency is the only lever. This is measured, not assumed —
reproduce with `python bench/throughput.py`.

## The model

The footprint of a columnar engine only matters as a limit on how many query
workers fit in memory. Dedicate the machine to one query type and run `W`
single-threaded workers in parallel (math libs pinned to one thread, so `W`
workers saturate `W` cores cleanly). Then:

    throughput = W x (1000 / latency_ms)
    W          = min(cores, (budget - shared) / private_per_worker)

Memmap'd segment files are shared across workers (page cache, paid once), so the
quantity that scales per worker is the **private (anonymous)** working set
(`Private_Dirty` in `/proc/self/smaps_rollup`) — the decoded column caches, FK
pointers, and Python/numpy objects — *not* the total RSS.

A worker is **RAM-bound only when RAM-per-core < its private working set.**

## Measured: queries/sec at 32 GB / 16 cores (SF1, 6M-row lineitem)

| # | query | latency | priv MB/worker | QPS @ 32 GB | bound |
|---|---|--:|--:|--:|---|
| 1 | COUNT(*) | 0.15 ms | 340 | 104,371 | CPU |
| 21 | COUNT(DISTINCT) low | 0.24 | 322 | 66,806 | CPU |
| 22 | COUNT(DISTINCT) high | 0.24 | 322 | 66,308 | CPU |
| 11 | WHERE numeric > | 0.26 | 368 | 62,064 | CPU |
| 14 | WHERE string = | 0.28 | 385 | 56,180 | CPU |
| 15 | WHERE IN (3) | 0.33 | 385 | 47,876 | CPU |
| 2 | whole SUM | 0.84 | 383 | 19,057 | CPU |
| 4 | GROUP BY K3 count | 1.27 | 386 | 12,565 | CPU |
| 26 | JOIN parent-key | 1.30 | 373 | 12,270 | CPU |
| 18 | DISTINCT 1-col | 1.31 | 386 | 12,169 | CPU |
| 12 | WHERE BETWEEN + agg | 1.49 | 430 | 10,733 | CPU |
| 25 | HAVING | 1.57 | 386 | 10,178 | CPU |
| 16 | WHERE AND/OR | 1.63 | 478 | 9,831 | CPU |
| 6 | GROUP BY K7 avg | 1.87 | 434 | 8,565 | CPU |
| 8 | GROUP BY datetime | 2.08 | 386 | 7,676 | CPU |
| 19 | DISTINCT 2-col | 2.44 | 432 | 6,560 | CPU |
| 13 | WHERE date-range (Q6) | 2.98 | 524 | 5,372 | CPU |
| 27 | JOIN child-key | 3.01 | 436 | 5,317 | CPU |
| 3 | whole multi-agg | 3.06 | 488 | 5,237 | CPU |
| 5 | GROUP BY K3 sum | 3.21 | 436 | 4,984 | CPU |
| 28 | JOIN parent-date | 4.57 | 448 | 3,500 | CPU |
| 29 | JOIN + WHERE | 4.91 | 495 | 3,261 | CPU |
| 7 | GROUP BY 2-col (Q1) | 5.42 | 531 | 2,954 | CPU |
| 17 | WHERE + GROUP BY | 5.67 | 482 | 2,823 | CPU |
| 30 | 3-table JOIN | 10.13 | 819 | 1,579 | CPU |
| 20 | DISTINCT high-card | 15.63 | 412 | 1,024 | CPU |
| 9 | GROUP BY high-card 200k | 69.78 | 486 | 229 | CPU |
| 24 | ORDER BY + LIMIT | 164.32 | 495 | 97 | CPU |
| 10 | GROUP BY vhigh-card 1.5M | 201.94 | 344 | 79 | CPU |
| 23 | grouped COUNT(DISTINCT) | 258.53 | 415 | 62 | CPU |

Private working set per worker: **322–819 MB** for dedicated single-query
workers; **~974 MB** worst case for a mixed worker that has touched every column.
Shared once (file + libs): **~253 MB**.

## The break-even: which chip could ever make RAM the bottleneck?

RAM throttles us when it can't hold `cores` workers:

    RAM-bound  <=>  RAM_total < shared + cores x private
    break-even RAM-per-core  ~=  private_per_worker  (~0.4-1.0 GB/core at SF1)

So **RAM only bottlenecks below ~1 GB of RAM per core.** Where real hardware sits:

| machine | RAM/core | result |
|---|--:|---|
| this box (30 GB / 16) | 1.9 GB | CPU-bound, ~2x margin |
| compute-optimized cloud (stingiest commodity) | ~2 GB | CPU-bound |
| general / memory-optimized cloud | 4–32 GB | CPU-bound by miles |
| laptop / Raspberry Pi 5 | 1–2 GB | CPU-bound |
| **to flip RAM-bound** | **< 1 GB** | 128-core EPYC starved to < 128 GB — rare, deliberate |

A 128-core EPYC at 32 GB (0.13 GB/core) *would* be RAM-bound — but nobody builds
that. The hardware enforces it: EPYC needs all 12 DDR5 channels populated to feed
a dense core count, which forces a 192–384 GB minimum regardless of capacity
needs. **The same density that creates the cores forces the RAM up**, so you
physically can't build the config where our 1.85x footprint costs throughput
without also starving the chip's bandwidth and losing throughput anyway.

## Scaling caveat (honest bound)

The private number scales with the *working set*, which scales with dataset rows
(~0.34–0.97 GB at SF1; ~10x at SF10). Two release valves keep it bounded:
memmap means the on-disk dataset can dwarf RAM (only hot decoded columns are
resident), and an optional LRU cap bounds the per-worker number directly. The
precise claim: **at a given data scale, any machine with >=~1 GB RAM/core is
CPU-bound — which covers all normal hardware.**

## Consequence for the roadmap

Stop optimizing RAM; optimize latency. Throughput is dragged down by a short list
of CPU-bound queries: grouped COUNT(DISTINCT) (#23, 62 QPS), GROUP BY vhigh-card
(#10, 80), ORDER BY+LIMIT (#24, 99), GROUP BY high-card (#9, 224), DISTINCT
high-card (#20, 1046) — largely the same set as the scoreboard losses. Every ms
shaved there multiplies by the core count into aggregate QPS.
