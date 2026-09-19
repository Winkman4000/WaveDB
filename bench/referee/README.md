# THE REFEREES: other engines, measured once on our pod, saved so they are never re-run

Every file here is one engine on one board, run on the same pod as the WaveDB boards (16 cores,
128 GB cgroup, RunPod network volume), from the same source data (`/workspace/data/hits.parquet`,
`/workspace/data/job/pq/*.parquet`). Three runs per query: the first after load (page cache warm
from the load itself, so not truly cold), then two more; `best` is the min of three, `warm` the min
of the last two. Sums are in seconds. Timings are the engine's own elapsed (ClickHouse `--time`)
or the client round trip (Umbra, like ClickBench's psql `\timing`).

| file | engine | board | load | on disk | sum(warm) | recorded |
|---|---|---|---|---|---|---|
| clickhouse_clickbench.json | ClickHouse 26.10.1.129 | ClickBench 43 (their queries.sql) | 32.6 s | 9.41 GB | 12.40 s | 2026-09-19 |
| clickhouse_job.json | ClickHouse 26.10.1.129 | JOB 113 (job_queries.sql) | 0.9 s (parquet) | 1.94 GB | 34.90 s | 2026-09-19 |
| umbra_clickbench.json | Umbra 26.09 (umbradb/umbra image, run without docker) | ClickBench 43 (their queries.sql) | 272.8 s | 8.01 GB | 5.58 s | 2026-09-19 |
| umbra_job.json | Umbra 26.09 | JOB 113 (job_queries.sql) | 11.0 s (parquet) | 2.66 GB | 2.78 s | 2026-09-19 |

WaveDB on the same day, same pod: ClickBench warm 6.5 s (per-process harness) / 8.78 GB loaded,
11.3 GB with sidecars; JOB warm 6.3 s / 0.77 GB loaded, 1.66 GB with sidecars; DuckDB in-process:
ClickBench 33.7 s / 20.5 GB, JOB 11.6 s.

Caveats recorded with the numbers: Umbra ran from the unpacked image via its own loader (no docker
on the pod) and the pod refuses to raise memlock, so Umbra logged "performance may degrade" for its
writeback buffers; ClickHouse's IMDB tables were created `ORDER BY tuple()` from the parquet (no
sort key), the JOB queries ran unmodified except `AS at` -> `AS at1`.

Runners: `bench/referee_clickhouse.py` (clickhouse client), `bench/referee_pg.py` (any
PostgreSQL-protocol engine). Re-run only when an engine version or the machine changes.
