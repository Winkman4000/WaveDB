# WaveDB

A columnar analytical database written in Python, with its hot loops compiled by numba. One developer's
research engine, fully functional on the read path: single-table analytics, joins, subqueries, CTEs and window
functions, checked against DuckDB as the correctness oracle.

## The idea

WaveDB spends its effort at load time. Every column is measured several ways while it is encoded -- its
cardinality, how its values cluster, how long its strings are, how its blocks are bounded -- and stored in the
encoding that fits it. At query time those measurements steer the plan: which filter runs first, which blocks
can be skipped, which read serves the shape. The closest comparison is a query planner choosing an order of
operations, except that the facts it chooses from were measured once, when the data arrived.

The measurements steer; they never answer. A query's result is always computed from the stored data.

## Using it

```bash
pip install numpy numba pyarrow zstandard sqlglot pandas
bin/wdb load  mydb hits hits.parquet           # encode a parquet (or csv) file into a table
bin/wdb sql   mydb "SELECT COUNT(*) FROM hits"
bin/wdb shell mydb                              # interactive prompt
bin/wdb serve mydb --port 8765                  # a long-lived server (what the ClickBench scripts use)
```

`bin/wdb --help` lists every command. Tests: `PYTHONPATH=src python tests/run.py` (DuckDB is the oracle).

## What is stored

- **The data.** Each column is dictionary-encoded with a sorted dictionary and bit-packed codes, or stored with
  a codec chosen from its measurements (front-coded text, block dictionaries, sequences, inline values).
  Lossless: every column decodes byte-exact.
- **Load statistics** (`<segment>.stats.npz`): per block of rows, the smallest and largest code and the
  number of non-NULL rows (block skipping, and MIN/MAX/COUNT the way zone maps answer them); sampled entries
  of block dictionaries, used to seek inside a column.
- **Text lengths**: for every large text column, the character length of each dictionary entry and of each
  row (`--row-lengths COL` adds them for other text columns).

Nothing else. **Sidecars** -- indexes and pre-aggregates the engine can build to make repeated queries faster --
are an extension that is **off by default**: with it off no derived file is ever written, by a load or by a
query (`wdb sidecars DB on|off|status|drop`). Block sums, per-value counts and repeat lists -- statistics that
could answer a query rather than steer it -- are neither written nor read by default (`WDB_LOAD_ANSWERS=1`, at
load and at query, turns them on for experiments only). No query result is cached:
between queries the engine keeps decoded source data in memory (dictionaries, codes), like a buffer pool, and
drops everything derived when the query ends.

## Benchmarks

`benchmark/clickbench/` is the ClickBench entry (install, load, start, query scripts; `README.md` there says
exactly what is loaded). `bench/` holds the boards used during development (ClickBench, TPC-H, JOB, h2o and
others) and `docs/CODEX.md` is the running engineering log: every change, with the measurement that motivated
it. Numbers in older docs are point-in-time and may be stale; the code is the source of truth.

## Status

Research engine. The analytics read path is the focus; DML and DDL are basic, and there are no transactions,
constraints or views. See `CAPABILITIES.md` for the SQL surface (its measured numbers date from June 2026).
