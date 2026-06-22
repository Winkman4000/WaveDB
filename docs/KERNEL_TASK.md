# Task: native-dtype AVX-512 hash-aggregation kernel for `wdb_scanpair`

You are working in the **WaveDB** repository (an indie analytical database benchmarked against
DuckDB on 100M-row ClickBench data). Build an optional, compiled AVX-512 kernel that accelerates the
high-match regime of the filtered two-key `COUNT(*)` top-K path, and wire it into `wdb_scanpair` with
a numpy fallback. This document is self-contained: the design is already decided and validated by
measurement — **do not re-derive it or substitute a different approach.** Your job is correct,
fast, well-tested implementation, plus an honest benchmark.

Current `main` HEAD when this was written: `3bbe2f7`.

---

## 1. The problem, precisely

The operator in `src/wdb_scanpair.py` answers:

```sql
SELECT a, b, COUNT(*) AS c FROM t WHERE C = v GROUP BY a, b ORDER BY c DESC LIMIT k
```

where `a`, `b` are two dictionary-coded group keys and `C` is a **non-key** dictionary-coded filter
column. It works by: scan the C code array for rows where `C == v` → gather the `a` and `b` codes at
those rows → fuse each pair into one integer key → count occurrences of each distinct pair → take
the top-k by count → decode only the k winners.

The current committed implementation (read `execute()` in `src/wdb_scanpair.py`) does the count with
`np.unique(key, return_counts=True)`, i.e. **a sort**. That sort is the bottleneck at high match
fractions, and it is why this query *loses* to DuckDB when the filter value `v` covers a large
fraction of the table.

### Measured evidence (already collected — trust it, do not re-litigate)

- The **scan** (`C == v` boolean sweep over 100M rows) is cheap and flat: ~13–25 ms regardless of
  match size. It is **not** the bottleneck.
- The **aggregation** is the bottleneck, and it scales with match size. At high match the current
  path loses to DuckDB: e.g. `RegionID = <dominant>` (≈18% of rows) ran ≈0.84×, `ResolutionWidth =
  <dominant>` (≈24%) ran ≈0.61×.
- **Counting instead of sorting wins.** On the real lopsided shape (one pair owning ~87% of matched
  rows, ~1.2M distinct pairs out of 18M matched rows), a count-based tally beat `np.unique` by
  **~6.2×** (191 ms → 31 ms); on uniform-random keys it was ~32×. The sort does work (ordering) that
  this query does not need — we only need counts, and order is irrelevant.
- A C-prototype proved the full kernel correct and showed the two-stage "read fewer bytes" structure
  wins: scanning C alone for survivor positions, then gathering a/b only at survivors, beat the
  current numpy path **1.10–1.14×** on the high-match losers and **1.38×** on a selective case — all
  bit-exact vs DuckDB. The split profile showed: stage 1 (filter+fuse, reading all three columns)
  was 67% of time; the hash tally itself was only 33% (~98 ms) and already beat the sort.

### The trap that killed the previous integration attempt (MUST avoid)

The prototype kernel required **uint32** code arrays. But WaveDB stores codes in their **natural
narrow width**: `seg.codes(col)` returns `uint8` / `uint16` / `uint32` depending on the column's
cardinality (e.g. `SearchEngineID` is uint8, `RegionID`/`ResolutionWidth` are uint16, `SearchPhrase`
is uint32). Feeding the uint32-only kernel meant `np.ascontiguousarray(seg.codes(col),
dtype=np.uint32)` — a full-column **upcast/copy of 100M elements per call** (~28 ms for a uint8
column alone). That copy erased the kernel's win when measured end-to-end through `db.run`, even
though the kernel won in isolation with pre-converted inputs.

**Therefore the kernel MUST read codes in their native width (uint8/uint16/uint32) with no upcast.**
This is the single most important requirement. A kernel that only accepts uint32 is a known failure.

---

## 2. The design to implement (decided — implement exactly this)

A two-stage, read-less, hash-aggregating kernel:

**Stage 1 — scan C alone, native width.** Read only the filter column's code array (in its native
dtype) and produce the survivor row positions (`int32`) where `C == v`. Use AVX-512 to compare 16/32/64
lanes at a time (lane count depends on code width) and `compressstore` the surviving indices. Reading
only C (one column) instead of all three is the "read fewer bytes" win — A and B are never touched for
the rows the filter discards.

**Stage 2 — gather a/b at survivors, native width.** Using the survivor positions, gather the `a`
and `b` codes (each in its native width) and fuse each pair into one key `a*Vb + b`. Choose the
fused-key integer width as the narrowest that holds `Va*Vb`: `uint32` if `Va*Vb < 2^32` (common),
else `int64`. (Narrow keys → less memory moved in the tally; this matches the committed narrowing.)

**Stage 3 — hash tally (NOT sort, NOT flat bincount).** Count occurrences of each fused key with an
open-addressing hash table sized to ~2× the number of *distinct* pairs that occur (a power of two,
load factor < 0.5). **Do not** `np.unique`/sort (that is the slow path we are replacing). **Do not**
flat `bincount` over the full key space: `Va*Vb` can be ~577M, so a dense counter would be ~4.3 GB
of mostly-zero slots and blow up memory. The hash table holds only pairs that actually occur
(~1.2M → ~32 MB), giving the same counts. This is standard hash aggregation.

**Stage 4 — top-K + decode.** From the distinct (key, count) pairs, take the top-k by count
(`argpartition`, no full sort) and decode **only the k winners** back to `(a_code, b_code)` and the
group-key values via `seg.fetch(col, code)` (random-access, never decode the whole dictionary — a
full high-card dict decode is ~1 s).

### Why each "don't" matters (so you don't optimize them away)
- Don't accept only uint32 → see §1 trap (upcast cost).
- Don't sort → that is the current slow path; counting is ~6× faster and order is not needed.
- Don't flat-bincount the full keyspace → 4.3 GB blowup; hash table is ~32 MB.
- Don't decode the full dictionary → ~1 s; decode only k winners.

---

## 3. Files to create / change

### Create `src/kernels/scanpair_kernel.c`
The C kernel. Native-width variants. Suggested entry points (you may adjust names/shape, but keep the
two-stage structure and native widths):

- `wdb_scan_positions_u8 / _u16 / _u32(const void* cC, int64_t N, uint32_t vC, int32_t* out_pos)`
  → vectorized `C == vC`, compressstore surviving indices, return survivor count `m`.
- `wdb_gather_fuse(const void* cA, int aw, const void* cB, int bw, const int32_t* pos, int64_t m,
  int64_t Vb, void* out_keys, int kw)` → gather a/b (native widths `aw`,`bw` in bytes) at positions,
  fuse `a*Vb+b` into out_keys of width `kw` (4 or 8 bytes). (Or split into per-width variants — your
  call, but no upcast of the full column.)
- `wdb_tally(const void* keys, int kw, int64_t m, int capbits, ... , int64_t* out_k, int64_t* out_c)`
  → open-addressing hash tally, return distinct count `d` and fill `out_k`/`out_c`.

Use AVX-512 intrinsics (`<immintrin.h>`). Provide scalar tails for the remainder. The reference
prototype used: `_mm512_cmpeq_epu32_mask`, `_mm512_mask_compressstoreu_epi32`, widen via
`_mm512_cvtepu32_epi64`, `_mm512_mullo_epi64`. For uint8/uint16 lanes use the appropriate
`epu8`/`epu16` compares (`avx512bw`) and widen accordingly. Hash: open addressing, multiplicative
hash `(k * 2654435761) & mask`, linear probe, slot value `-1` = empty.

### Create `src/wdb_kernel.py`
Lazy compile + load + safe fallback. Requirements:
- Compile on first use with `gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vl -mavx512dq
  -shared -fPIC`. Cache the `.so` next to the source, keyed by a hash of the `.c` so a changed source
  rebuilds. Compile with a timeout.
- Detect AVX-512 (read `/proc/cpuinfo` for `avx512f`). If absent, or `gcc` missing, or compile fails,
  return `None` from a `load()`-style function so callers fall back.
- Expose one function, e.g. `scanpair_topk(cC, cA, cB, vC, Vb, k) -> (a_codes, b_codes, counts)` for
  the top-k pairs, or `None` to decline (kernel unavailable / inputs out of supported shape, e.g.
  Va*Vb ≥ 2^64, or N ≥ 2^31 so int32 positions overflow). It must accept the code arrays **in their
  native dtype** and branch to the right C entry point by `dtype.itemsize` — never upcast the whole
  array.
- Everything in a `try/except` that degrades to `None`. The kernel is an accelerator, never a
  correctness dependency.

### Modify `src/wdb_scanpair.py` (`execute()`)
Add a fast path: when the filter is a single code (`fcodes.size == 1`) and `Va*Vb < 2^32`, call
`wdb_kernel.scanpair_topk(...)`. If it returns a result, build `winners = [(a_code, b_code, count),
...]` from it. If it returns `None`, fall through to the **existing** numpy path unchanged. Then the
existing decode loop (decode only the k winners via `seg.fetch`) runs on `winners` for both paths.
Keep the multi-code (`np.isin`) and `Va*Vb >= 2^32` cases on the numpy path. Do not change `detect()`
or the reader registration in `src/read_methods.py` / `src/controller.py`.

### Relevant existing API (in `src/wdb_engine.py`, class `Segment`)
- `seg.codes(nm)` → native-dtype code array (uint8/uint16/uint32), length N. **This is your input.**
- `seg.cols[nm]['V']` → cardinality (use for Va, Vb). `seg.cols[nm]['mode']`, `['has_null']` exist.
- `seg.code_counts(nm)` → cached per-code bincount (length V). Useful if you want match size cheaply.
- `seg.fetch(nm, code)` → decode a single code to its Python value (use for the k winners only).
- `seg.N` → row count.

---

## 4. Correctness requirements (non-negotiable)

- Results must be **bit-exact** vs the numpy path and vs DuckDB, including tie handling at the LIMIT
  boundary. (The numpy path's tie behavior is the reference; match it. If ties make the top-k
  ambiguous in a way the numpy path resolves a particular way, match that, or if it's genuinely
  ambiguous, ensure the *set* and *counts* match what DuckDB returns for the verified queries.)
- NULL semantics: the filter excludes the null code (`V-1` when `has_null`) — preserve whatever the
  current `_eq_on_nonkey` / `execute` does. Read the current code; don't change observable behavior.
- The full test suite must stay green: `PYTHONPATH=src python3 tests/run.py` (currently **1673
  passed, 0 failed**). On a machine with AVX-512 + gcc the kernel path will actually execute during
  the suite (good — that's coverage). On a machine without, the fallback runs. Both must pass.

---

## 5. Verification (do this; report the numbers honestly)

There is a 100M-row ClickBench segment and a DuckDB parquet oracle. Benchmark warm (median of ≥4
runs after a warmup call) end-to-end through `db.run` — NOT just the kernel in isolation, because the
last attempt looked great in isolation and regressed through `db.run` due to the dtype upcast.

Group by `(SearchEngineID, SearchPhrase)`; filter on a non-key column. Test at least:

| filter | match % | current numpy ratio vs Duck | goal |
|---|---|---|---|
| `CounterID = 62` | small/selective | ~1.3–1.5× (win) | stay a win, ideally faster |
| `RegionID = <dominant>` | ~18% | ~0.84× (loss) | flip to ≥1.0× |
| `ResolutionWidth = <dominant>` | ~24% | ~0.61× (loss) | improve toward/above 1.0× |
| `RegionID = 2` | ~7% | ~1.0× | stay ≥1.0× |

For each: assert **exact** match vs DuckDB, record `wave_ms`, `duck_ms`, `ratio`, and whether the
kernel fired (not the fallback). DuckDB reads the parquet with this retype CTE (EventDate/EventTime
are stored as offsets):

```sql
WITH hits AS (
  SELECT * REPLACE (
    (DATE '1970-01-01' + EventDate) AS EventDate,
    (TIMESTAMP '1970-01-01' + to_seconds(EventTime)) AS EventTime
  ) FROM read_parquet('<parquet path>')
)
```

Pick the dominant value for a column self-referentially, e.g.
`v = int(seg._typed_dict(col)[np.bincount(seg.codes(col)).argmax()])` — do not hardcode a value read
from peeking at the data.

### Honesty requirement
If the kernel does **not** flip a loss to a win, say so plainly with the numbers. Do not claim a win
that the measurement doesn't show. A correct kernel that ties is still useful (less memory traffic);
report what is true. If it regresses anywhere vs the committed numpy path, that is a blocker — find
out why (profile stage 1 vs stage 2 vs tally) before claiming done.

---

## 6. Acceptance criteria (all required)

1. `src/kernels/scanpair_kernel.c`, `src/wdb_kernel.py`, and the `execute()` fast-path exist.
2. Kernel reads **native-width** codes (uint8/uint16/uint32) with **no full-column upcast**.
3. Tally is a **hash table** (not sort, not flat bincount); memory scales with distinct pairs, not
   keyspace.
4. Decodes only the **k winners** via `seg.fetch`.
5. Kernel is **optional**: missing AVX-512 / missing gcc / compile failure / out-of-shape input →
   returns `None` → numpy fallback → identical results.
6. Full suite green (`PYTHONPATH=src python3 tests/run.py`): 1673+ passed, 0 failed.
7. End-to-end-through-`db.run` benchmark vs DuckDB on 100M reported honestly for the four cases
   above, all bit-exact, with the kernel firing on the single-code cases.
8. No regression vs the committed numpy path on any tested case.

---

## 7. Process notes

- Work on a branch; keep commits focused. Don't commit a `.so` binary — it's built lazily at runtime
  and is arch-specific. Add `src/kernels/*.so` to `.gitignore`.
- Build C against small hand-checked arrays first (unit-test the C via a tiny ctypes harness:
  feed known uint8/uint16/uint32 inputs, assert the counts match a Python reference) BEFORE running
  on 100M. Get correctness on tiny data, then benchmark on big data.
- If you discover the design as specified cannot beat DuckDB at very high match (e.g. >50%), report
  the profile and the numbers rather than forcing a win — that is a real finding, not a failure to
  hide. The bar is: flip the ~18% case, don't regress anything, stay exact, keep the kernel optional.
- The numpy fallback path already in `execute()` is correct and committed — preserve it exactly as
  the fallback. Your fast path sits in front of it, never replaces it.
