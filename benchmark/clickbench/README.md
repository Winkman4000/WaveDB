# WaveDB

WaveDB (https://github.com/Winkman4000/WaveDB) is a column-oriented analytical database written in Python,
with its hot loops compiled by numba. It runs here as a small HTTP server (`wdb serve`) and every query goes
through `./query` (curl measures the round trip).

**Load.** `./load` reads the single `hits.parquet` file: the rows ordered by `EventTime`, `EventDate` and
`EventTime` cast to a date and a timestamp, `URLHash`/`RefererHash` kept as hashes. Stored with it: the data,
its load statistics (per-block min/max and counts) and the character lengths of the text columns (row lengths
for `URL`). No index, projection, materialized view or pre-aggregate is built (`WDB_SIDECARS=0`), and none is
built by a query. `create.sql` documents the table; the loader takes the schema from the Parquet file with the
casts above.

**Install.** `./install` checks out the engine at `WAVEDB_REF` with pinned Python packages, then compiles every
numba kernel signature the engine uses (`tools/kernel_build.py build`, from `src/kernels.manifest`), as a C++
engine is compiled at install.

**Start.** The server loads every compiled kernel the engine ships (`src/wdb_preload.py`) and opens the
database before it answers `./check` -- program code only; nothing from the data, and no result, is cached
across the restart.

**Queries.** `queries.sql` is the standard set; Q28/Q29 use `length()` (characters).
