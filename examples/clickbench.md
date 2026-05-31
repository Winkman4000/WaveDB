# ClickBench worked example

[ClickBench](https://github.com/ClickHouse/ClickBench) is a standard analytical
benchmark (web-traffic `hits` table, 43 queries). This is WaveDB's measured
result on a 10M-row slice, vs DuckDB on identical data, same machine
(Ryzen 7 7800X3D, 8 cores).

## Reproduce

```bash
# 1. get a 10M-row hits.parquet (from the ClickBench repo)
# 2. build a segment over the columns the 43 queries touch:
./wavedb build hits.parquet hits.wdb
# 3. confirm lossless:
./wavedb verify hits.wdb hits.parquet      # -> 55/55 byte-perfect
# 4. inspect:
./wavedb stats hits.wdb
# 5. time individual operations:
./wavedb groupby hits.wdb OS               # ~2 ms
./wavedb agg     hits.wdb OS ResolutionWidth avg
```

## Result

| metric | WaveDB | notes |
|---|---|---|
| lossless | 55/55 columns | byte-exact vs parquet |
| correctness | 43/43 queries | match DuckDB exactly |
| segment size | 942 MB | vs 916 MB source parquet |
| count GROUP BY | 1.4–6 ms | ~20x faster than DuckDB (~14 ms) |
| per-group SUM/MIN/MAX/AVG | ~21 ms | single value-weighted pass |
| COUNT(DISTINCT) | ~6 ms | exact, free from dictionary |
| high-card GROUP BY (1M-50M distinct) | 3-37x faster | measured to 100M rows; DuckDB never overtakes |
| high-card string *filter/scalar decode* | seconds | the one slow path (raw value decode) |

## Honest notes

- The 702 MB of the 942 MB segment is three high-cardinality string columns
  (URL ~2.6M distinct, Title ~1.6M, UserID ~1.5M). The other 51 columns are
  240 MB total. Compression is strong where data is structured, weak where it is
  high-entropy — as expected.
- The win is **pre-paid**: dense-code array tally beats hashing, but the
  densification happens at encode time, which is the slow side of the trade.
- Encode is the one place WaveDB does not beat a general engine; it does more
  work up front so reads and updates are cheap.
