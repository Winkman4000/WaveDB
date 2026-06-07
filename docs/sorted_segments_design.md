# Sorted Segments + Boundary Tree — design spec

**Status:** proposal, prototyped + measured, not yet built.
**Goal:** flip the grouped / filtered-aggregate throughput losers by laying rows out in
organizing-power order so a filter or group becomes a *range walk* over contiguous data
instead of a *scan + mask* over the whole column.

**One line:** the only lever against the memory-bandwidth wall is reading fewer bytes; sorting
the rows by a low-cardinality "organizing" column turns "touch 6M rows" into "jump to a range,"
and the savings are intrinsic to the data — chosen with zero assumptions about the workload.

---

## 1. The problem this solves

Under concurrency every column scan is **memory-bandwidth-bound**: 16 cores starve on the
~50 GB/s bus, so throughput = bandwidth / bytes-per-query. The whole-table aggregates were
fixed by the value-frequency tally (committed: #2 590->2606 q/s, #3 168->2659 q/s @16). What
remains are the **grouped / filtered** losers, where the measure is high-card so there is no
tally shortcut — you must read the rows to bucket them:

| # | query | WaveDB @16 | DuckDB @16 |
|---|---|--:|--:|
| 5 | grouped SUM by returnflag | 166 | 452 |
| 6 | grouped AVG by shipmode | 472 | 508 |
| 12 | filtered SUM, discount range | 346 | 704 |
| 17 | filtered SUM + group | 195 | 402 |

These are O(N)-at-runtime: there is no precomputed per-group structure, so the engine must
read every row to know which group/filter bucket it belongs to. Sorting removes that — the
buckets become contiguous ranges.

## 2. The result (measured on bench lineitem, sf=1, 6,001,215 rows)

Filter target = returnflag's smallest group (24.6% of rows). Sorting by returnflag makes that
group **one contiguous range** (natural order: 2,099,592 runs; sorted: 3 runs).

| operation | current | sorted | gain |
|---|--:|--:|--:|
| filtered SUM (latency) | 6.00 ms (scan+dot-mask) | 0.28 ms (range read) | **21×** |
| grouped SUM, all groups (latency) | 10.32 ms (scatter-add) | 0.92 ms (3 slice-sums) | **11×** |
| cross-cut leaf-harvest (latency) | 6.00 ms | 0.18 ms (3 leaf ranges) | **33×** |
| **filtered SUM @16 (throughput)** | **212 q/s** | **10,874 q/s** | **51×** |

Why it wins three ways at once: the range read (a) touches only the selected rows' bytes,
(b) is pure sequential `.sum()` (prefetcher-friendly), and (c) needs no mask computation and
no group-code reads. Even the all-groups case (must touch every row) wins 11× because three
sequential slice-sums beat a scatter-add that reads group codes and does random accumulator
writes.

**Honest scope of the measurement:** these are kernel-level (latency) plus one concurrent
@16 kernel run. End-to-end @16 through `db.run` will be lower because SQL-parse/setup is the
new ceiling (the same reason #2 hit 2606 and not 30k). Parse caching is a complementary,
separate lever; it does not change the byte-traffic argument here.

## 3. Core principle — organizing power (zero assumptions)

A column earns a place in the row order by **how much each of its values narrows the rows** —
its organizing power, which is intrinsic and equals N / cardinality. returnflag (3 values) is a
strong divider; partkey (200k) is a weak one; orderkey (1.5M) is none.

- **Low cardinality -> high organizing power -> belongs in the sort order** (a row-organizer).
- **Near-unique -> no organizing power -> stays a plain column** (read off a row, never searched by).

This is the field-role decision: a field that organizes other fields should be a *row-organizer*,
not just a column. It is computed from the segment's own measured cardinality — **never from the
workload.** Under the engine's zero-assumption stance (it cannot see traffic, only data), uniform
query likelihood is the only honest prior, and under uniform priors the optimal order is exactly
organizing-power order. Traffic-based reweighting is an *optional* operator refinement layered on
top, not the foundation.

## 4. The structure — sorted segment + boundary tree

At segment seal, choose a compound sort key (ordered list of organizing columns, §6), compute
one argsort permutation over that key, and reorder **every column** through it. The rows are now
physically grouped. Alongside, store a tiny **boundary tree**: per sort level, the (value, start,
end) markers.

```
level 0 (returnflag):   R:[0,1478493)  N:[1478493,4522345)  A:[4522345,6001215)
level 1 (shipmode, within each L0 block):
        R/AIR:[0,..) R/RAIL:[..) ...   N/AIR:[..) ...   A/AIR:[..) ...
level 2 (...): ...
```

- **Size:** boundaries only, not row lists. A few levels of low-card dimensions = kilobytes.
  (returnflag x shipmode x linestatus = 3x7x2 = 42 leaf ranges.) This is the "trade compression
  headroom for a tiny index" move, spent on *order* not on decoded data.
- **Build cost:** one argsort + one gather per column, at seal time, sub-second; never per query.
- **Format:** a sidecar next to the .wdb (like the FK-pointer and override sidecars), or a section
  in the segment header. Stores: sort-key column ids, and the nested (value_code, start, end) tree.

## 5. The reads — three dispatch shapes

All three are gated on the query's filter/group columns being (a prefix-compatible subset of) the
segment's sort key. Otherwise fall through to the existing scan/tally path unchanged.

1. **Range walk** — filter on the root (or a root prefix): look up the value's [start,end] in the
   tree, run the aggregate over that contiguous slice. (#12, #17, #22-style.) 21× / 51× measured.
2. **Grouped slice-sums** — GROUP BY the root with no filter: walk the level-0 ranges, one
   sequential reduction per block, no group-code reads, no scatter. (#5, #6.) 11× measured.
3. **Leaf harvest** — filter/group on a *non-root* level ("all AIR regardless of returnflag"):
   skip the upper levels and collect that value's sub-range from each parent block. Costs
   (product of cardinalities above it) range-reads — small while upper levels are low-card.
   (33× measured for shipmode under returnflag.)

Multi-predicate queries intersect ranges down the tree (root range, then sub-range within it).
The dispatch is itself self-gated: it engages only when the tree covers the query's columns.

## 6. Choosing the sort order (the math)

Objective, computable from the segment alone: minimize expected query cost
**Σ_c  freq(c) × pieces(c)**, where `pieces(c)` = product of cardinalities of the levels above
c (how many ranges c shatters into when queried alone), and `freq(c)` defaults to uniform under
zero assumptions.

Consequences:
- **Low-card, high-organizing columns go near the root** (Huffman-like: strong dividers get the
  short path); **high-card columns go to the leaves or out of the tree** (they shatter everything
  below them — never put orderkey above anything).
- **Eligibility gate:** a column joins the tree only if its cardinality is below a cell budget
  (so the tree stays KB and `pieces` stays small). Self-referential — a table of all-high-card
  columns gets an empty tree and we scan, identically and correctly.
- **Refinement (design-pass, not v1): conditional organizing power.** Correlated columns divide
  redundantly (if linestatus ~ returnflag, stacking both buys less than their cardinalities
  imply). The exact metric is "narrowing *given the levels already chosen*"; first-order
  (rank by cardinality) is the backbone, correlation is a second-order correction.
- **Optional operator input:** real `freq(c)` from an observed query stream re-weights the
  objective. This is the *only* place workload enters, it is optional, and it sits above the
  zero-assumption foundation.

## 7. The write path — tree-guided placement + the hot tail

The boundary tree that makes reads fast also tells writes exactly where they belong: a new/changed
row's sort-key value -> walk the tree -> the block it joins (and, for an update, the block it
leaves). **Placement is free.** The residual cost is making physical room without shifting the
array. Two strategies, and the first is already our architecture:

- **(A) Unsorted hot tail + compaction (reuse what exists).** WaveDB is already sealed cold
  segments + an unsorted hot parquet buffer; `flush()` folds the buffer into a new cold segment.
  So: inserts land in the hot tail (unsorted, cheap); reads hit the sorted segments (ranged) plus
  the small hot tail (cheap scan because it's small); **compaction sorts the tail into a sealed
  segment with its boundary tree.** Minimal new machinery — sorting happens at flush/compact.
- **(B) Packed-memory slack (later, if always-sorted reads matter).** Leave a gap at each block's
  end; inserts drop into the gap (no shift); only a full block re-packs, and the tree already knows
  when a block is full. Write cost scales with **block size, not table size** — the thing that
  makes in-place sorted storage viable rather than a read-only trick.

Recommended v1: (A). It is the existing model; sorted-on-seal is the only delta.

## 8. The seams (honest hard parts)

- **FK pointer stability.** Pointers are a per-child-row sidecar holding *absolute parent row
  positions*. So:
  - Sorting a **fact/child** table (lineitem) reorders its pointer sidecar *alongside* the rows
    (the sidecar is indexed by child position); pointer *values* (parent positions) are unchanged.
    **Joins keep working, zero remap.** This is the safe, recommended case — sort facts freely.
  - Sorting a **parent** (orders) changes parent positions, so every child's pointer values must be
    remapped: `new_ptr = inv_perm[old_ptr]` — one gather per referencing child, done once at the
    parent's seal/sort. Bounded, but it couples tables; defer parent-sorting past v1.
  - Cleanest long-term: pointers reference a **stable row-id** that maps through the sort, decoupling
    join identity from physical slot. Design-pass question.
- **Mutability / correctness.** A boundary tree is valid only for its sealed segment. The hot tail
  is never claimed to be sorted (it's scanned). Overrides (UPDATE sidecars) that change a sort-key
  value invalidate that row's placement -> treat as tail until compaction, or re-pack the block.
  Deletes (presence mask) are fine (ranges still valid, masked rows skipped).
- **One physical order.** A segment has one sort key; leaf-harvest (§5.3) recovers non-root columns
  at `pieces(c)` cost, so the one order serves many query shapes — but a high-card column queried
  alone is still a scan. That's correct (its organizing power was zero).
- **MIN/MAX** don't compose from sums; per-range they're a sequential min/max over the slice (still
  cheap) — handle alongside SUM/COUNT in the slice kernels.
- **Parse-overhead ceiling.** End-to-end @16 is capped by SQL parse/setup, not the kernel; the byte
  win is real but the headline 51× is kernel-level. Parse caching is the complementary follow-up.

## 9. The law (self-referential, non-negotiable)

Every fast path here is gated on facts the segment measured about itself (cardinality, sort-key
coverage), and **degrades to the existing scan with identical results** when the gate fails. No
optimization depends on a column being a particular thing or on the workload being a particular
shape. Sorting is chosen by organizing power; the tree is built only for eligible low-card columns;
the read paths engage only when the tree covers the query. Hostile data (all high-card) -> empty
tree -> scan, no regression.

## 10. Phased implementation (small verified increments)

Each step is independently verifiable, suite-green, committed before the next. Prototype kernels
standalone before wiring (as we did for the tally).

1. **Read-only proof on a hand-sorted segment.** Build a sorted copy of the bench lineitem offline
   (no encoder change), write a boundary-tree sidecar, add a standalone range/harvest/slice-sum
   kernel, and measure @16 throughput end-to-end vs the scan. Decision gate: does the end-to-end
   win survive parse overhead? (If not, parse-caching moves ahead of this.)
2. **Boundary-tree format + reader.** Define the sidecar; `Segment` exposes `sort_key`,
   `range_of(level, value)`, `leaves(level, value)`. Tests: tree matches the data's runs.
3. **Seal-time sort.** Encoder sorts by a chosen key, reorders all columns + the FK-pointer
   sidecar, writes the tree. Verify joins still pass (fact sorted) and all correctness tests hold.
4. **Order selection.** Implement organizing-power ranking + eligibility gate (cardinality-only,
   v1). `test_profile_calibrate`-style calibration of the cell budget.
5. **Query dispatch.** Planner detects sort-key-covered filter/group -> range/harvest/slice-sum;
   scan fallback otherwise. Path-coverage ratchet extended with the new path tag.
6. **Compaction sorts the tail.** `flush`/compact produces sorted sealed segments + trees.
7. **(Later) packed-memory slack; conditional/correlation-aware ordering; parent-sort + pointer remap.**

## 11. Open questions to poke holes in

- Does the end-to-end @16 win survive parse overhead, or must parse-caching come first? (Step 1 decides.)
- Stable row-id vs physical-slot pointers — worth it now, or defer parent-sorting?
- Cell budget for tree eligibility — one global ratio, or per-segment calibrated?
- Multi-segment: each segment independently sorted (simpler) vs a global order (better ranges, harder)?
- Float MIN/MAX and datetime handling in the slice kernels.
- Interaction with existing mode-4 / override / presence-mask paths (likely: tree only on clean sealed segments).

## 12. Expected impact on the losers (projected from kernels; confirm in step 1)

| # | loser | now @16 | Duck @16 | mechanism | projection |
|---|---|--:|--:|---|---|
| 5 | grouped SUM by returnflag | 166 | 452 | slice-sums (11× kernel) | win |
| 6 | grouped AVG by shipmode | 472 | 508 | slice-sums | win |
| 12 | filtered SUM, discount | 346 | 704 | range walk (sort by discount) | win |
| 17 | filtered SUM + group | 195 | 402 | range walk (51× kernel) | win |
| 7 | Q1 (mixed) | 94 | 234 | grouping via tree; high-card AVG still scans | partial |

Caveat: end-to-end gains are capped below kernel gains by parse overhead until that's addressed.
The byte-traffic reduction (and thus the concurrency scaling) is the robust, measured claim.

---

*Provenance: this design is the synthesis of a long reasoning thread — organizing power, the
boundary tree, range-walk + leaf-harvest, tree-guided writes — validated against the bench data
(§2). The reads are proven; the write-maintenance reuses the existing hot/cold model; the seams
(pointer stability, parse ceiling) are the parts to settle before building.*
