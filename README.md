# WaveDB

A compressed, lossless, columnar analytical store with a fast query kernel.

WaveDB dictionary-encodes each column, bit-packs the codes, and runs GROUP BY /
COUNT / per-group aggregates directly over the packed codes with a parallel,
cache-resident tally. It is a **specialist**: it pays a real cost up front at
encode time, and in exchange wins on reads of data whose shape it knows in
advance. It is not a general-purpose database and not a fast bulk loader.

## What it is good at (and what it isn't)

Wins and losses come from one property: **cardinality** (distinct values per column).

- **Low / medium cardinality** (up to ~tens of thousands distinct): GROUP BY and
  filtered counts run in **single-digit milliseconds** via a parallel counter-fan
  tally that stays inside CPU cache.
- **High cardinality** (millions distinct): GROUP BY / COUNT / aggregates still
  **win 3-37x over DuckDB** (measured at 100M rows: 37x at 1M distinct, 8x at 10M,
  3x at 50M). The dense-code array tally beats hash aggregation at every scale
  tested; DuckDB never overtakes it. The lead narrows as values approach all-unique
  but stays a multi-x win.
- **The one tie**: using a high-cardinality *string* column in a filter or scalar
  pass that decodes the raw values (e.g. `WHERE url <> ...`) is a scan — but DuckDB
  and ClickHouse scan here too, because pulling back literal high-entropy values has
  no structure left to exploit. Everyone is memory-bandwidth bound, so it's a tie,
  not a loss. A store could beat it only by sacrificing compression (keep the column
  uncompressed/byte-aligned) or adding a secondary index — a different trade, not a
  free win.
- **Encode** does more work than a general engine (it builds dictionaries and packs
  codes), but does it efficiently — a hash-dictionary + 8-core parallel encoder did
  50 columns in ~3s vs DuckDB's ~6s (105 cols) and ClickHouse's ~10s. Roughly a
  tie, edging ahead per-column. The up-front work is what makes reads cheap later.

Stable schema + mostly low/mid cardinality → fast. Mostly unique high-cardinality
values (random IDs, free text) → correct but not dramatically faster.

## Usage

```bash
make                                         # build the C kernels

./wavedb build   mydata.parquet seg.wdb      # build a segment (schema read from the file)
./wavedb verify  seg.wdb mydata.parquet      # prove byte-exact lossless
./wavedb stats   seg.wdb                      # per-column cardinality, bits, speed-class
./wavedb groupby seg.wdb mycolumn             # fast GROUP BY count (C kernel)
./wavedb groupby seg.wdb colA colB            # multi-key GROUP BY
./wavedb agg     seg.wdb groupcol valcol avg  # per-group sum/min/max/avg in one pass
```

Requires Python 3 with `duckdb`, `numpy`, `pyarrow`, and `gcc`.

## How it works

- **Encode**: each column -> sorted dictionary of distinct values + bit-packed
  codes (lossless; reconstruction is byte-exact).
- **GROUP BY**: codes are dense (0..V-1), so we tally directly into an array
  indexed by the code — no hashing. Parallel per-core histograms ("counter fan")
  merged at the end. Beats hash aggregation while the histogram fits in cache.
- **Per-group aggregates**: one pass, each row's value added to its group's
  accumulator (value is the weight) — SUM/MIN/MAX/AVG from a single pass.
- **COUNT(DISTINCT)**: exact and free — it is the dictionary size.
- **High-cardinality strings**: front-coded (shared-prefix delta vs the previous
  sorted value) + zstd, with restart points for random access. On URL this is
  ~7x smaller than the plain dictionary and beats ClickHouse's best ZSTD mode,
  at a competitive build time (~2s in C with an AVX2 prefix-match loop).

## Benchmark (worked example: ClickBench, 10M rows)

Measured on a Ryzen 7 7800X3D (8 cores), all 55 columns the 43 ClickBench queries
touch, compared against DuckDB on identical data. See `examples/clickbench.md`.

- **Lossless:** 55/55 columns byte-perfect.
- **Correctness:** 43/43 queries match DuckDB.
- **Segment size:** 448 MB (down from 942 MB) vs 916 MB source parquet — high-card
  string columns (URL, Title) are front-coded + zstd. On the URL column alone:
  68 MB vs ClickHouse-ZSTD 110 MB, ClickHouse-LZ4 231 MB, DuckDB-FSST 412 MB.
- **Speed:** low/mid-card count GROUP BYs 1.4–6 ms (~20x faster than DuckDB);
  high-cardinality GROUP BY 3–37x faster than DuckDB (measured to 100M rows);
  per-group SUM/MIN/MAX/AVG ~21 ms; exact COUNT(DISTINCT) ~6 ms. The only slow
  path is decoding raw high-card *string* values in a filter/scalar pass.

These numbers are reproducible with the CLI above on the ClickBench `hits`
parquet; they are not hardcoded.

## Format (`WVDB2`)

```
magic "WVDB2" | u16 n_cols | u32 n_rows
per column: u16 name_len | name | u32 V | u8 bits | u8 dtype(0=int,1=bytes)
            | dict: V x (u32 len | bytes)
            | packed codes (n_rows x bits, MSB-first)
```

## Status

Research prototype. The query CLI exposes primitives (build / verify / stats /
groupby / agg), not a full SQL parser. The ClickBench SQL runner that produced
the 43-query result is a worked example, not a general SQL engine.
