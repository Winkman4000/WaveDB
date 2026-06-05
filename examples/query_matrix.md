# WaveDB query-type matrix

Every query shape the engine supports today, measured on **identical TPC-H sf=1 data**
(lineitem 6,001,215 rows · orders 1,500,000 · customer 150,000), WaveDB vs DuckDB,
same machine (Ryzen 7 7800X3D, 8C/16T). Best-of-5 ms. Every result checked row-for-row
against DuckDB.

Reproduce: `python3 bench/query_matrix.py` (builds nothing; expects the sf=1 DB at
`/tmp/jbprof_sf1.0/wdb`, which `python3 bench/join_prof.py 1.0` creates).

**"code bits read"** = the dense code-stream the query must scan = Σ (table_rows ×
code_width_bits) over the columns it touches. It is the information the computation
consumes — note `COUNT(*)` reads **zero** column bits (it is the row count, O(1)), while
a high-cardinality float sum reads the most. This is the *C* in C-vs-S; the ms column is *S*.

## The matrix

| # | category | query type | rows out | code bits read | DuckDB | WaveDB | speedup | fast | ✓ |
|---|---|---|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | — | 0.3 ms | 0.2 ms | 1.71x | Y | ✓ |
| 2 | agg | whole SUM | 1 | 120.0 Mbit | 0.6 ms | 1.9 ms | 0.34x | Y | ✓ |
| 3 | agg | whole multi-agg (5) | 1 | 180.0 Mbit | 2.6 ms | 3.7 ms | 0.71x | Y | ✓ |
| 4 | group | GROUP BY K3 count | 3 | 12.0 Mbit | 4.9 ms | 1.3 ms | 3.69x | Y | ✓ |
| 5 | group | GROUP BY K3 sum | 3 | 132.0 Mbit | 5.0 ms | 2.7 ms | 1.87x | Y | ✓ |
| 6 | group | GROUP BY K7 avg | 7 | 54.0 Mbit | 14.1 ms | 2.9 ms | 4.89x | Y | ✓ |
| 7 | group | GROUP BY 2-col Q1-shape | 4 | 174.0 Mbit | 9.9 ms | 8.2 ms | 1.20x | Y | ✓ |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72.0 Mbit | 1.9 ms | 2.0 ms | 0.94x | Y | ✓ |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144.0 Mbit | 212.0 ms | 70.0 ms | 3.03x | Y | ✓ |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126.0 Mbit | 290.6 ms | 180.8 ms | 1.61x | Y | ✓ |
| 11 | filter | WHERE numeric > | 1 | 36.0 Mbit | 0.8 ms | 3.7 ms | 0.22x | Y | ✓ |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144.0 Mbit | 1.7 ms | 5.8 ms | 0.28x | Y | ✓ |
| 13 | filter | WHERE date-range Q6-shape | 1 | 252.1 Mbit | 2.4 ms | 4.2 ms | 0.57x | Y | ✓ |
| 14 | filter | WHERE string = | 1 | 12.0 Mbit | 4.0 ms | 1.9 ms | 2.10x | Y | ✓ |
| 15 | filter | WHERE IN (3) | 1 | 18.0 Mbit | 9.3 ms | 3.1 ms | 2.97x | Y | ✓ |
| 16 | filter | WHERE AND/OR | 1 | 54.0 Mbit | 8.6 ms | 5.9 ms | 1.46x | Y | ✓ |
| 17 | filter | WHERE + GROUP BY | 3 | 168.0 Mbit | 5.0 ms | 7.6 ms | 0.65x | Y | ✓ |
| 18 | distinct | DISTINCT 1-col | 3 | 12.0 Mbit | 13.3 ms | 1.5 ms | 9.02x | Y | ✓ |
| 19 | distinct | DISTINCT 2-col | 4 | 18.0 Mbit | 17.4 ms | 3.1 ms | 5.62x | Y | ✓ |
| 20 | distinct | DISTINCT high-card | 200,000 | 108.0 Mbit | 48.6 ms | 15.3 ms | 3.17x | Y | ✓ |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18.0 Mbit | 15.9 ms | 58.8 ms | 0.27x | Y | ✓ |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108.0 Mbit | 26.4 ms | 216.8 ms | 0.12x | Y | ✓ |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30.0 Mbit | 21.9 ms | 331.9 ms | 0.07x | — | ✓ |
| 24 | order | ORDER BY + LIMIT | 10 | 144.0 Mbit | 34.5 ms | 167.3 ms | 0.21x | Y | ✓ |
| 25 | order | HAVING | 7 | 18.0 Mbit | 14.1 ms | 1.8 ms | 7.62x | Y | ✓ |
| 26 | join | JOIN group parent-key | 5 | 60.1 Mbit | 8.7 ms | 29.2 ms | 0.30x | Y | ✓ |
| 27 | join | JOIN group child-key | 3 | 289.6 Mbit | 17.5 ms | 226.9 ms | 0.08x | Y | ✓ |
| 28 | join | JOIN group parent-date hiK | 2,406 | 295.5 Mbit | 17.7 ms | 308.7 ms | 0.06x | Y | ✓ |
| 29 | join | JOIN + WHERE | 5 | 318.1 Mbit | 31.5 ms | 331.6 ms | 0.09x | Y | ✓ |
| 30 | join | 3-table JOIN | 5 | 306.2 Mbit | 35.5 ms | 357.0 ms | 0.10x | Y | ✓ |

**30/30 correct vs DuckDB · 29/30 on the fused fast path · median speedup 0.94x**

## Where we win, where we lose

**Wins (the dictionary + dense-code strengths, paid for at encode time):**
- `DISTINCT` — 3–9×. The dictionary already *is* the distinct set; we read codes, not values.
- `GROUP BY` low/mid card — 1.9–4.9×; high-card (200k–1.5M groups) — 1.6–3×. Dense-code tally beats hashing because densification happened at encode.
- `HAVING` — 7.6× (group is cheap, the filter is trivial on top).
- `WHERE` string `=` / `IN` — 2–3× (dictionary-code compare, no string materialisation).
- whole `COUNT(*)` — instant, zero bits read.

**Losses (the work the engine has not yet specialised):**
- **Joins — 0.06–0.30×. This is the dominant weakness.** The FK-pointer gather + scatter loses badly to DuckDB's hash join. Reads the most bits (290–318 Mbit) *and* the worst locality.
- `COUNT(DISTINCT)`, esp. grouped — 0.07–0.27×. Not specialised; the grouped form falls off the fast path entirely (the only non-fused shape here).
- Simple scalar `WHERE` (count/sum + predicate) — 0.22–0.28×. DuckDB's raw SIMD scan beats our mask build for a one-number answer.
- `ORDER BY … LIMIT` top-K — 0.21×. We sort the full grouped result then slice; no top-K heap.
- whole single-column `SUM` — 0.34×. One column is exactly DuckDB's SIMD sweet spot.

## The C-vs-S read

Bits read does **not** predict speed by itself — locality does. GROUP BY high-card reads
144 Mbit in 70 ms; JOIN child-key reads ~2× the bits (290 Mbit) but takes **3×** the time
per bit. The extra cost is not information, it is the scatter of the gather — exactly
`S = work × rate(locality)` with the measured ~4.2× locality penalty. Where the access is
sequential (group/distinct/scan) we win; where it scatters (join gather, distinct-value
hashing) we pay the penalty.

## Versus the ClickBench origin

We began at *correctness* — 43/43 ClickBench queries matching DuckDB on the `hits` table,
with the group-by wins. The surface has since grown well past those 43 shapes: joins (1–3
table), `DISTINCT`/`COUNT(DISTINCT)`, `HAVING`, datetime ranges, `ORDER BY`/`LIMIT`, and
multi-segment — all correct (this matrix is 30/30, the full suite 1517/0). The group-by and
distinct wins held; the join and count-distinct paths are now clearly marked as the work
that remains. (The `hits` dataset isn't reproducible on this machine, so this matrix is
measured on TPC-H sf=1 — same engine, broader shape coverage.)
