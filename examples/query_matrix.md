# WaveDB query-type matrix

> **Canonical corpus:** these are our TPC-H **matrix** queries — the `#` column maps to
> **M#** in [`../docs/crosswalk.md`](../docs/crosswalk.md). For the unified, deduplicated
> corpus across both datasets (matrix + ClickBench), see [`../docs/corpus.md`](../docs/corpus.md).
> Kept here for its measured ms results and the honesty notes below.


Every query shape the engine supports today, measured on **identical TPC-H sf=1 data**
(lineitem 6,001,215 rows · orders 1,500,000 · customer 150,000), WaveDB vs DuckDB,
same machine (Ryzen 7 7800X3D, 8C/16T). Best-of-5 ms. Every result checked row-for-row
against DuckDB.

Reproduce: `python3 bench/join_prof.py 1.0` (builds the sf=1 DB), then
`python3 bench/query_matrix.py`. The matrix declares the two FK relationships up front
(`create_fk_pointer`) — see the join note below.

**"code bits read"** = the dense code-stream the query must scan = Σ (table_rows ×
code_width_bits) over the columns it touches. `COUNT(*)` reads **zero** column bits (it is
the row count, O(1)); a high-cardinality float sum reads the most. That is the *C* in
C-vs-S; the ms column is *S*.

## The matrix

| # | category | query type | rows out | code bits read | DuckDB | WaveDB | speedup | fast | ✓ |
|---|---|---|--:|--:|--:|--:|--:|:-:|:-:|
| 1 | agg | whole COUNT(*) | 1 | — | 0.3 ms | 0.2 ms | 1.94x | Y | ✓ |
| 2 | agg | whole SUM | 1 | 120.0 Mbit | 0.6 ms | 1.7 ms | 0.35x | Y | ✓ |
| 3 | agg | whole multi-agg (5) | 1 | 180.0 Mbit | 2.6 ms | 2.6 ms | 1.00x | Y | ✓ |
| 4 | group | GROUP BY K3 count | 3 | 12.0 Mbit | 4.6 ms | 1.3 ms | 3.66x | Y | ✓ |
| 5 | group | GROUP BY K3 sum | 3 | 132.0 Mbit | 5.0 ms | 3.3 ms | 1.51x | Y | ✓ |
| 6 | group | GROUP BY K7 avg | 7 | 54.0 Mbit | 14.9 ms | 4.3 ms | 3.48x | Y | ✓ |
| 7 | group | GROUP BY 2-col Q1-shape | 4 | 174.0 Mbit | 10.0 ms | 5.4 ms | 1.85x | Y | ✓ |
| 8 | group | GROUP BY datetime K2.5k | 2,526 | 72.0 Mbit | 2.0 ms | 2.0 ms | 0.97x | Y | ✓ |
| 9 | group | GROUP BY high-card K200k | 200,000 | 144.0 Mbit | 212.9 ms | 76.4 ms | 2.79x | Y | ✓ |
| 10 | group | GROUP BY vhigh-card K1.5M | 1,500,000 | 126.0 Mbit | 284.5 ms | 182.4 ms | 1.56x | Y | ✓ |
| 11 | filter | WHERE numeric > | 1 | 36.0 Mbit | 0.9 ms | 3.6 ms | 0.24x | Y | ✓ |
| 12 | filter | WHERE BETWEEN + agg | 1 | 144.0 Mbit | 1.6 ms | 5.8 ms | 0.27x | Y | ✓ |
| 13 | filter | WHERE date-range Q6-shape | 1 | 252.1 Mbit | 2.4 ms | 3.5 ms | 0.67x | Y | ✓ |
| 14 | filter | WHERE string = | 1 | 12.0 Mbit | 4.0 ms | 1.9 ms | 2.11x | Y | ✓ |
| 15 | filter | WHERE IN (3) | 1 | 18.0 Mbit | 9.3 ms | 2.8 ms | 3.26x | Y | ✓ |
| 16 | filter | WHERE AND/OR | 1 | 54.0 Mbit | 8.5 ms | 4.8 ms | 1.76x | Y | ✓ |
| 17 | filter | WHERE + GROUP BY | 3 | 168.0 Mbit | 5.0 ms | 5.7 ms | 0.89x | Y | ✓ |
| 18 | distinct | DISTINCT 1-col | 3 | 12.0 Mbit | 12.6 ms | 1.5 ms | 8.57x | Y | ✓ |
| 19 | distinct | DISTINCT 2-col | 4 | 18.0 Mbit | 17.1 ms | 2.0 ms | 8.44x | Y | ✓ |
| 20 | distinct | DISTINCT high-card | 200,000 | 108.0 Mbit | 48.2 ms | 16.7 ms | 2.89x | Y | ✓ |
| 21 | distinct | COUNT(DISTINCT) low | 1 | 18.0 Mbit | 15.5 ms | 59.0 ms | 0.26x | Y | ✓ |
| 22 | distinct | COUNT(DISTINCT) high | 1 | 108.0 Mbit | 26.2 ms | 216.4 ms | 0.12x | Y | ✓ |
| 23 | distinct | grouped COUNT(DISTINCT) | 3 | 30.0 Mbit | 20.5 ms | 341.9 ms | 0.06x | — | ✓ |
| 24 | order | ORDER BY + LIMIT | 10 | 144.0 Mbit | 33.0 ms | 164.8 ms | 0.20x | Y | ✓ |
| 25 | order | HAVING | 7 | 18.0 Mbit | 14.3 ms | 1.5 ms | 9.33x | Y | ✓ |
| 26 | join | JOIN group parent-key | 5 | 60.1 Mbit | 8.5 ms | 1.3 ms | 6.51x | Y | ✓ |
| 27 | join | JOIN group child-key | 3 | 289.6 Mbit | 19.2 ms | 4.6 ms | 4.18x | Y | ✓ |
| 28 | join | JOIN group parent-date hiK | 2,406 | 295.5 Mbit | 17.8 ms | 4.4 ms | 4.07x | Y | ✓ |
| 29 | join | JOIN + WHERE | 5 | 318.1 Mbit | 30.7 ms | 5.2 ms | 5.88x | Y | ✓ |
| 30 | join | 3-table JOIN | 5 | 306.2 Mbit | 40.2 ms | 11.6 ms | 3.47x | Y | ✓ |

**30/30 correct vs DuckDB · 29/30 on the fused fast path · median speedup 1.94x**

## Where we win, where we lose

**Wins:**
- **Joins — 3.5–6.5×.** FK relationships are pre-resolved once into a parent-row pointer
  (a one-time structure, like declaring an index), so an equi-join becomes a gather +
  fused group, while DuckDB rebuilds a hash table every query. 1-, 2- and 3-table all win.
- `DISTINCT` — 2.9–8.6×. The dictionary already *is* the distinct set; we read codes.
- `GROUP BY` — 1.5–3.7× (low/mid card), 1.6–2.8× (high card 200k–1.5M groups).
- `HAVING` — 9.3×; string `=`/`IN` — 2–3×; whole `COUNT(*)` — instant, zero bits.

**Losses (the work still done at query time, with poor locality):**
- `COUNT(DISTINCT)` — 0.06–0.26×. The real weak spot now. Grouped count-distinct is the
  **only** shape that falls off the fused fast path. Highest-value thing to fix next.
- `ORDER BY … LIMIT` top-K — 0.20×. We sort the full grouped result then slice; no heap.
- one-number scalar `WHERE` (count/sum + predicate) — 0.24–0.27×. DuckDB's SIMD scan wins
  when the answer is a single value and there's no group to amortise our mask build over.
- whole single-column `SUM` — 0.35×; the `Q6` date-range filter — 0.67× (known).

**Join caveat (full honesty):** the join wins use FK pointers created up front. *Without*
them the same joins fall back to a runtime hash join and run **0.06–0.30×** (slower than
DuckDB). The pointer is the intended path for a declared FK relationship — pre-pay the
resolution once, gather many times — but it is a pre-built structure DuckDB doesn't get
here, so the fair statement is: *with FK relationships declared, WaveDB wins joins 3.5–6.5×;
on cold ad-hoc joins it currently loses.*

## The C-vs-S read

Bits read do not predict speed; **where the work happens** does. Every WaveDB win is
*pre-paid* — the dictionary (distinct/group), the FK pointer (joins), the dense codes — so
query time is a sequential sweep. Every loss is work done *at query time with scatter*:
count-distinct hashes values into a set, top-K sorts, a cold join builds a hash table.
That is `S = work × rate(locality)` directly: move the work to encode/declare time and the
rate is sequential; leave it to query time and you pay the ~4.2× locality penalty.

## Versus the ClickBench origin

We began at *correctness* — 43/43 ClickBench queries matching DuckDB on `hits`. The surface
has grown well past those shapes: joins (1–3 table), `DISTINCT`/`COUNT(DISTINCT)`, `HAVING`,
datetime ranges, `ORDER BY`/`LIMIT`, multi-segment — all correct (this matrix 30/30, full
suite 1517/0). Joins and group-by/distinct are now wins; count-distinct and top-K are the
marked remaining work. (`hits` isn't reproducible on this machine, so this is TPC-H sf=1 —
same engine, broader coverage.)
