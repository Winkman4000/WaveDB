# Throughput: the memory-bandwidth wall

**TL;DR — under concurrency, every catalog query is *memory-bandwidth-bound*, not
CPU-bound and not RAM-capacity-bound.** Sixteen workers deliver only 8–58 % of
16× the single-worker rate; the missing throughput is contention on the shared
~50 GB/s memory bus. Scaling tracks *bytes read per query* — `COUNT(*)` reads
almost nothing and scales to 58 %, a whole-column `SUM` streams ~48 MB and scales
to 9 %. So the lever for throughput is **bytes read per query** (compression and
the BSI filter-index), *not* single-query latency. RAM capacity never binds on a
machine with ≥ ~1 GB/core. Live numbers: [`examples/throughput.md`](../examples/throughput.md);
reproduce with `python bench/throughput.py`.

## What this corrects

An earlier version of this doc computed throughput as `W × (1000 / latency_ms)`
— it *extrapolated* aggregate queries/sec from a single query's latency, assuming
`W` workers scale perfectly. They do not. Measuring `W` workers running **at the
same time** shows 8–58 % scaling, and that gap *is* the bandwidth wall. The
extrapolation inverted the conclusion: it reported every query "CPU-bound" and
recommended optimizing latency. The real concurrent measurement says the
opposite — bandwidth-bound, optimize bytes per query.

## The model

Dedicate the machine to one query type and run `W` single-threaded workers in
parallel (math libs and numba pinned to one thread, so `W` workers saturate `W`
cores cleanly). The reported number is the **real aggregate queries/sec with all
`W` running together** — bandwidth contention included, not extrapolated.

    throughput = measured aggregate q/s across W workers running concurrently
    W          = min(cores, budget_MB / private_per_worker_MB)

Memmap'd segment files are shared across workers (page cache, paid once), so the
quantity that scales per worker is the **private (anonymous)** working set
(`Private_Dirty` in `/proc/self/smaps_rollup`) — decoded column caches, FK
pointers, the BSI index, Python/numpy objects — not the total RSS. WaveDB workers
run in the default **non-escalated (throughput) mode**, so the BSI filter-index
engages where it pays.

## Measured: SF1, W = 16, 32 GB budget

The full per-query table is the committed scoreboard,
[`examples/throughput.md`](../examples/throughput.md). The shape of the result:

- **All 30 queries are bandwidth-bound** (scale < 60 %). At 32 GB / 16 cores the
  worker count is `W = cores = 16` for every query — RAM capacity never caps it
  (the `bound` column reads `BW` everywhere, never `RAM`).
- **Scaling inversely tracks bytes read per query.** The byte-light queries scale
  best; the byte-heavy ones scale worst:

  | query | reads | scale @16 |
  |---|---|--:|
  | `COUNT(*)` | ~0 column bytes (row count only) | 58 % |
  | `COUNT(DISTINCT)` (code-only) | dictionary codes, no value decode | 57 % |
  | `GROUP BY K3 count` | one code column, tally | 55 % |
  | `WHERE l_quantity > 30` count | one code column | 42 % |
  | whole `SUM(l_extendedprice)` | full 48 MB value column | 9 % |
  | `GROUP BY returnflag, SUM(...)` | key codes + full value column | 8 % |

  More bytes streamed per query → more pressure on the bus → worse concurrent
  scaling. That inverse relationship is the whole argument: throughput is gated
  by how many bytes each query pulls through the ~50 GB/s bus, not by how many
  cores or how much RAM the box has.

## Why bandwidth, not cores or RAM capacity

- **Cores:** 16 workers fill 16 cores, but an analytical worker spends most of
  its cycles *waiting* on column bytes from DRAM, not computing. The bus
  saturates before the ALUs do, so the 17th core (or hyperthread) buys little.
- **RAM capacity:** still not the binding constraint. The per-worker private set
  is well under the 2 GB/worker that a 32 GB / 16-core box allows, so `W` is
  pinned by core count, never by capacity. Capacity headroom does not buy
  throughput; bus bandwidth does. (The RAM-per-core ceiling is detailed below —
  it is real but no normal machine hits it.)

## The lever: fewer bytes per query

Because aggregate throughput is gated by the memory bus, it improves by reading
**fewer bytes per query**:

- **Compression.** WaveDB stores the same data ~1.23× smaller than DuckDB, so
  every scan streams fewer bytes off the bus.
- **The BSI filter-index (throughput mode).** For selective off-key filters it
  evaluates predicates on compact bit-planes and walks only the set bits,
  touching ~16 MB of index instead of scanning the full predicate columns. This
  flips the two off-key filters it engages — `WHERE BETWEEN` (#11) and TPC-H Q6
  (#12) — from losses to wins **under concurrency**, even though the same path
  loses single-query latency (see modes below).
- **Narrow, code-only paths.** `COUNT`, `GROUP BY`, and `COUNT(DISTINCT)` over
  dictionary codes never decode values; they read the fewest bytes and scale
  best (55–58 %).

Single-query latency is a *different* axis, optimized by a *different* mode.

## Two modes — operator-selected (escalation)

The same query has two honest execution strategies; the operator declares intent
per call (`db.run(sql, escalate=...)`) or as a deployment default (`db.escalate`).

| mode | `escalate` | strategy | wins when | scoreboard |
|---|---|---|---|---|
| **throughput** (default) | `False` | BSI filter-index, fewest bytes | many concurrent queries split the cores | [`examples/throughput.md`](../examples/throughput.md) |
| **latency** | `True` | fully parallel fused scan | one query owns all the cores | [`examples/report.md`](../examples/report.md) |

The BSI bitmap-walk kernel is single-threaded, so it wins when cores are split
across concurrent workers (bytes saved dominates) and loses when one query can
use all 16 cores in a parallel fused scan. Rather than have the engine guess
intent, the mode is the operator's explicit choice; both return identical
results.

## The RAM-per-core ceiling (capacity is not the wall)

RAM throttles throughput only when it cannot hold `cores` workers:

    RAM-bound  ⇔  RAM_total < shared + cores × private_per_worker
    break-even RAM-per-core  ≈  private_per_worker  (well under 1 GB/core at SF1)

So RAM only becomes the bottleneck **below ~1 GB of RAM per core** — a regime no
normal machine occupies:

| machine | RAM/core | result |
|---|--:|---|
| this box (30 GB / 16) | 1.9 GB | bandwidth-bound, capacity to spare |
| compute-optimized cloud (stingiest commodity) | ~2 GB | capacity to spare |
| general / memory-optimized cloud | 4–32 GB | capacity to spare by miles |
| laptop / Raspberry Pi 5 | 1–2 GB | capacity to spare |
| **to flip RAM-bound** | **< 1 GB** | a 128-core chip starved to < 128 GB — rare, deliberate |

And the hardware resists even that: a dense core count needs all the DDR channels
populated to be fed, which forces a large RAM minimum regardless of capacity
need. The same density that creates the cores forces the RAM up — so you can't
easily build the config where footprint costs throughput without also starving
the bus (and losing throughput to bandwidth anyway). Capacity is not the lever;
bytes-per-query is.

## Reproduce

    python bench/throughput.py [budget_gb] [cores]      # defaults: 32, nproc

Writes [`examples/throughput.md`](../examples/throughput.md) (WaveDB vs DuckDB,
per-query 1-worker and @W aggregate q/s, scaling %, bound, BSI flag) and prints
the same table to stdout. WaveDB is measured in throughput mode (BSI on); both
engines run as `W` single-thread workers concurrently.
