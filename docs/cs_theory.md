# WaveDB — The (C, S) Theory of Storage and Speed

*Design note — session 2026-06-05. Status: validated by prototype + microbenchmark on the dev host; NOT yet wired into the engine. All numbers are sf=1 lineitem unless noted.*

## 0. One-line summary

A query's cost splits into two independent ledgers:

- **C** — the information in the *data*. Shannon floor. Abstract, fixed, unbeatable.
- **S** — the speed, where **S = work × rate(locality)**.
  - **work** = number of operations the *question* forces. Abstract. The only term we freely engineer.
  - **rate** = the medium (ns per operation). Analog, fixed by hardware.
  - **locality** = modulates the rate: sequential access is cheap, random ~4.2x dearer on this host.

Every physical-design lever (clustering, indexes, code-LUTs, posting lists, pre-aggregation)
is a **work reduction** and/or a **locality repair**. None of them changes C.

The practical thesis: the database should **measure its own hardware and its own data**, then
apply each lever from a calibrated threshold — not from heuristics hand-fit to one benchmark.

## 1. The reframing (C is bits, S is medium — with one correction)

The starting intuition: "C is bits (information, abstract); S is analog (hardware, reality)."
Mostly right. The correction that makes it complete: **S has a hidden abstract half too.**

S = work x rate. The *rate* is analog (the silicon). The *work* is as abstract as C — it is
the number of dependent operations the question forces, measured in operations, not bits.
That is the "sequential steps" clause, promoted from footnote to half the theory. It is the
half we control: indexes/clustering/tabs/pre-agg are all work reductions. None adds information;
none changes the hardware; each reshapes the question so the medium runs fewer steps.

### Orthogonality verdict
C and S **do** separate. S's only informational dependence is on the *question's* work, never
the *data's* content. You can compress C to the floor and independently engineer work down —
two separate ledgers.

### The one entanglement that survives
The rate is not a pure constant; it is `rate(access_pattern)`. Sequential ~0.6 ns/elem; random
gather ~2.5 ns/elem (4.2x penalty) on this host. So a plan that reduces work but scatters memory
access pays the penalty back. Clustering wins because it reduces work AND keeps access sequential.

## 2. The three measurements that established S = work x rate(locality)

All on the dev host, single-threaded numpy/numba unless noted, same data, same machine.

**(1) S tracks work at a fixed rate.** Summing K elements for growing K:

| K (elements) | ns/element |
|---|---|
| 10,000 | 237 (overhead-dominated) |
| 1,000,000 | 132 |
| 6,000,000 | 142 |

At scale the rate is flat (~135 ns/elem here for this op). S is linear in work; the slope is the medium.

**(2) S is blind to C.** Sum a high-entropy column vs a low-entropy one, same size:

| column | info (bits/elem) | time |
|---|---|---|
| extendedprice | 19.65 | 788 us |
| discount | 3.46 | 783 us |

5.7x the information, 1.01x the time. Speed cannot see information content.

**(3) Work is the lever.** Same op, clustered so it touches fewer rows:

| scope | elements | time |
|---|---|---|
| full | 6,001,215 | 792 us |
| 1994 block | 909,455 | 125 us |

Work dropped 6.60x -> time dropped 6.33x. Near-perfect proportionality. C and rate untouched.

**Locality penalty (calibration).** numba sum, 16M float64, sequential vs random index:
sequential 0.60 ns/elem; random 2.53 ns/elem => **4.2x penalty**. (Absolute ns were measured;
the ratio is the load-bearing constant and is unit-robust.)

## 3. Case study: TPC-H Q6 — from a loss to a 5x win

Q6 = `SUM(l_extendedprice * l_discount)` filtered by a 1-year ship-date range, a discount band,
and a quantity bound. It is DuckDB's home turf: a selective scan + scalar sum.

Starting point (current engine): **0.57x** vs DuckDB (we lose).

Root cause (measured, not guessed):
- NOT pruning — DuckDB is *fastest* on the broad/whole-table cases, so zone-map skipping is not its edge.
- The fused kernel already short-circuits and is multi-threaded.
- The leak: range predicates (`<`, BETWEEN) decoded the whole column to values; equality used the
  fast code path. (Per-predicate: equality +0.3ms; range +3-4ms each.)

Prototype fix chain (all correct, bit-exact 123,141,078.23):

| approach | time | vs DuckDB 2.4ms |
|---|---|---|
| fused full scan, 1 thread | 18.78 ms | loss |
| + clustering (read only 909k block) | 3.11 ms | ~par (6.0x from clustering) |
| + parallelism | **0.48 ms** | **5x win** |

The decomposition exactly matches the theory: clustering = work reduction (6.0x); parallelism =
better use of the rate across cores (~6.5x); C never moved (identical answer/bits).

Caveats (honest): spends the one permutation on date (other-column filters do not benefit);
excludes one-time ~341ms cluster-build; prototype kernel, not yet wired.

## 4. Latent structure already sitting in what we store

**Counts = free exact selectivity.** The frequency histogram (pointers per dict value) gives the
exact selectivity of any filter with zero row-scanning (matched measured 15.2 / 27.3 / 46.0%).
Encoding already runs `np.unique`, so persisting counts is ~free. This is a cost-free query planner.

**The "tab" = sorted-dict column -> range filter is a contiguous slice.** Because the date dict is
sorted, the 1994 codes are a contiguous range (730-1094). With positions stored sorted by date-code,
"find all 1994 rows" is a searchsorted + slice: **0.003 ms vs 5.19 ms to scan — 1709x.** The tab was
latent in the structure. (But finding rows != reading them fast; see locality.)

**Content-as-address = free primary key.** 99.99% of rows are uniquely pinned by their column values
(5,999,935 / 6,001,215). Identity recovered from content alone in ~8-60 us, zero stored id, zero scan:
start from the smallest posting list (the near-unique price -> 6-19 candidates), intersect the rest -> 1.
The high-entropy column *is* the primary key it always secretly was. Cost in extra bits: ~0.

**Stride / runs = free identity for the indistinguishable.** Where content collides (e.g. only
disc+qty: biggest group 11,325 rows), cluster by that content and each group becomes a contiguous run.
Identity = run_start + offset; offset is a pure stride (stride=1), zero content bits. Verified:
recovered row 3,000,000 as (group 227, offset 5509).

**The convergence (the key unification).** In a clustered low-card column the *run* is one object that
answers all three: **run-length = the frequency count; run-start = the "tab"; offset-in-run = the stride
ordinal.** The three intuitions are the same structure. 550 runs x ~3 numbers identify all 6M rows.

## 5. The one-axis law (the limit that did NOT crack)

A table is N rows; a physical layout is one linear order; one order sorts by one key. To make two
independent columns both runny you would need the rows in two orders at once — impossible for one
sequence. This is arithmetic, not observation. **Permutation is exactly one free `s`-currency:** it
buys speed for one access pattern at zero C-cost (reordering preserves the multiset = zero information).

Measured proof: after clustering by date, the other low-card columns stay scattered.

| column | runs in date-order | runs if it were the sort key |
|---|---|---|
| shipdate | 2,526 (perfect) | 2,526 |
| discount | 5,455,213 (~random) | 11 |
| quantity | 5,881,656 (~random) | 50 |

So one clustering buys **one** good spoke. TPC-H columns are near-independent, so there is no free ring.
Additional fast access patterns must be *paid*: a second physical copy (projection), a composite key
that helps a chosen pair, or lightweight per-block zone maps. The "circle" intuition is real only where
columns are genuinely dependent (a region that determines tax, a customer that determines nation); there,
clustering on the determiner makes dependents runny for free. That is measurable per dataset, in advance.

## 6. Two regimes for free identity

- **High-entropy column present** -> content IS the address (price posting list). Random-access identity
  for free, WITHOUT spending the permutation. Possibly composes with clustering on a different key
  (untested — see open questions).
- **Low-entropy table** -> content cannot address; cluster -> runs -> stride identity. Free, but it costs
  the one permutation.

## 7. Generalization: two sensory organs + measured thresholds

The insight must fire from measurement, never from TPC-H-specific tuning. Two artifacts to build.

### A. Calibrator (`wdb_calibrate.py`, run once per install)
Microbenchmarks the host and writes its analog constants:
- `seq_rate` — sequential ns/elem (~0.60 here)
- `rand_rate` — random-gather ns/elem (~2.53 here)
- `locality_penalty = rand_rate / seq_rate` (~4.2x here)
- `lut_cliff` — dict/LUT size where gather rate degrades (~2^16-2^18; validates LUT_MAX_CARD=65536)
- (later) per-core scaling for the parallel rate

### B. Profiler (`wdb_profile.py`, at encode — piggybacks on existing np.unique)
Per column, persist:
- `V` cardinality, `H` entropy (bits), `uniq = V/N`, `sorted` (dict monotonic?),
  `code_bits = ceil(log2 V)`, `mode`, `dt`.

These two are the eyes. Every downstream lever reads from them; nothing is hand-set.

## 8. Calibrated constants (dev host, 2026-06-05)

| constant | value | gates |
|---|---|---|
| sequential rate | ~0.60 ns/elem | the base medium rate |
| random gather rate | ~2.53 ns/elem | scattered-access cost |
| **locality penalty** | **4.2x** | cascade-vs-scan, scatter-vs-cluster |
| LUT cache cliff | ~2^16-2^18 entries | code-LUT eligibility (`LUT_MAX_CARD`) |

**Decision rule from the penalty:** a scattered plan must cut work by **> 4.2x** to beat a sequential
scan; otherwise scan, or cluster to make the access sequential and erase the penalty.

## 9. Column profile (lineitem, sf=1) — produced by the prototype profiler

| column | mode | V | H bits | uniq | code_bits | sorted |
|---|---|---|---|---|---|---|
| l_orderkey | 2 | 1,500,000 | 20.32 | 0.2499 | 21 | yes |
| l_partkey | 2 | 200,000 | 17.59 | 0.0333 | 18 | yes |
| l_suppkey | 0 | 10,000 | 13.29 | 0.0017 | 14 | yes |
| l_linenumber | 0 | 7 | 2.61 | 0.0000 | 3 | yes |
| l_quantity | 0 | 50 | 5.64 | 0.0000 | 6 | yes |
| l_extendedprice | 0 | 933,900 | 19.65 | 0.1556 | 20 | yes |
| l_discount | 0 | 11 | 3.46 | 0.0000 | 4 | yes |
| l_tax | 0 | 9 | 3.17 | 0.0000 | 4 | yes |
| l_returnflag | 0 | 3 | 1.49 | 0.0000 | 2 | (str) |
| l_linestatus | 0 | 2 | 1.00 | 0.0000 | 1 | (str) |
| l_shipdate | 0 | 2,526 | 11.27 | 0.0004 | 12 | yes |
| l_commitdate | 0 | 2,466 | 11.25 | 0.0004 | 12 | yes |
| l_receiptdate | 0 | 2,554 | 11.27 | 0.0004 | 12 | yes |
| l_shipinstruct | 0 | 4 | 2.00 | 0.0000 | 2 | (str) |
| l_shipmode | 0 | 7 | 2.81 | 0.0000 | 3 | (str) |
| l_comment | 5 | 3,610,733 | 21.22 | 0.6017 | 22 | (str) |
| l_ord_ptr | 2 | 1,500,000 | 20.32 | 0.2499 | 21 | yes |

Read-off: code-LUT for every V<=~2500 column; content-address from l_comment (0.60) and
l_extendedprice (0.156); narrow in-memory codes (discount 4 bits vs stored 64).
"(str)" = string dict; sortedness needs the bytes-comparison check (TODO).

## 10. Decision rules (the auto-physical-design optimizer)

Each lever is a measured gate, not a heuristic:

1. **Code-LUT for range/equality** -> fires when `V <= lut_cliff`. Pure cardinality test.
2. **Content-addressing (free primary key)** -> fires when `max column uniq >= u_thresh` (e.g. >= ~0.5).
   No structure built; it already exists as the posting list.
3. **Range-as-slice / inverted tab** -> fires when `sorted == true` AND the column is range-filtered.
4. **Cluster on a key (spend the one permutation)** -> choose the column with the highest
   `workload_weight x selectivity_benefit`, only if `work_saved x rate > amortized build_cost`.
   This is the ONLY lever that needs the workload, because permutation is scarce.
5. **Stride/run identity** -> automatic once clustered + low-card (collision groups become runs).
6. **Narrow in-memory codes to `code_bits`** -> always (kernel reads fewer bytes; needs a numba
   kernel that consumes narrow dtypes directly — numpy fancy-indexing upcasts and hides the win).
7. **Cascade vs bitmap-AND** -> bitmap by default; cascade/slice only when clustered on the lead
   predicate (else the 4.2x penalty eats the work savings). Measured: cascade lost 12.85 vs 11.47ms.

## 11. Open questions / next steps

- **Wire the two organs first**: `wdb_calibrate.py` (host constants) + `wdb_profile.py` (column stats at
  encode). Everything downstream reads from these. Lowest-risk, highest-leverage commit.
- Cleaner cache-cliff measurement via a numba kernel (numpy fancy-index confounds it).
- String-dict sortedness check (bytes comparison) so string range/ordering can slice too.
- **Composition test**: content-address (price) WHILE clustered by date — does free random-access
  identity survive spending the permutation elsewhere? Cleanest evidence yet for c/s separation.
- Multiple projections (Vertica-style) for >1 fast access pattern; cost model for when to pay.
- Build-cost amortization model (cluster build ~341ms): break-even in #queries.
- Float bit-exactness vs DuckDB (per-group naive sum differs ~1e-13; matters for strict Q1).
- Wire the fused clustered+parallel kernel into the engine and re-run the 9-shape board.
