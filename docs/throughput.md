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
| 1 | COUNT(*) | 0.17 ms | 340 | 95,012 | CPU |
| 21 | COUNT(DISTINCT) low | 0.24 | 322 | 65,708 | CPU |
| 22 | COUNT(DISTINCT) high | 0.24 | 322 | 67,086 | CPU |
| 2 | whole SUM | 1.04 | 389 | 15,352 | CPU |
| 18 | DISTINCT 1-col | 1.21 | 386 | 13,244 | CPU |
| 4 | GROUP BY K3 count | 1.30 | 386 | 12,306 | CPU |
| 26 | JOIN parent-key | 1.40 | 373 | 11,451 | CPU |
| 25 | HAVING | 1.56 | 386 | 10,287 | CPU |
| 14 | WHERE string = | 1.68 | 386 | 9,501 | CPU |
| 6 | GROUP BY K7 avg | 1.82 | 433 | 8,795 | CPU |
| 8 | GROUP BY datetime | 2.08 | 386 | 7,680 | CPU |
| 19 | DISTINCT 2-col | 2.16 | 432 | 7,411 | CPU |
| 3 | whole multi-agg | 2.79 | 488 | 5,734 | CPU |
| 5 | GROUP BY K3 sum | 2.80 | 435 | 5,709 | CPU |
| 15 | WHERE IN | 2.90 | 386 | 5,519 | CPU |
| 27 | JOIN child-key | 3.04 | 436 | 5,266 | CPU |
| 13 | WHERE date-range (Q6) | 3.28 | 530 | 4,877 | CPU |
| 11 | WHERE numeric > | 3.67 | 387 | 4,365 | CPU |
| 16 | WHERE AND/OR | 4.89 | 480 | 3,272 | CPU |
| 7 | GROUP BY 2-col (Q1) | 5.12 | 531 | 3,124 | CPU |
| 29 | JOIN + WHERE | 5.15 | 495 | 3,105 | CPU |
| 28 | JOIN parent-date | 5.37 | 448 | 2,978 | CPU |
| 17 | WHERE + GROUP BY | 5.53 | 482 | 2,893 | CPU |
| 12 | WHERE BETWEEN | 6.00 | 436 | 2,666 | CPU |
| 30 | 3-table JOIN | 11.40 | 727 | 1,404 | CPU |
| 20 | DISTINCT high-card | 15.30 | 413 | 1,046 | CPU |
| 9 | GROUP BY high-card 200k | 71.55 | 485 | 224 | CPU |
| 24 | ORDER BY + LIMIT | 161.83 | 490 | 99 | CPU |
| 23 | grouped COUNT(DISTINCT) | 257.17 | 415 | 62 | CPU |
| 10 | GROUP BY vhigh-card 1.5M | 199.47 | 344 | 80 | CPU |

Private working set per worker: **322–727 MB** for dedicated single-query
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
