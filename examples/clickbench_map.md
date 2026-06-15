# WaveDB — ClickBench Coverage Map

> **Canonical corpus:** the `Q#` ids here are **ClickBench Q#** in [`../docs/crosswalk.md`](../docs/crosswalk.md).
> For the unified corpus across both datasets, see [`../docs/corpus.md`](../docs/corpus.md).
> Kept here for its shape-class roadmap and correctness/precision notes.

*The finite target. Drawn 2026-06-13 on pod yabbering_sapphire_nightingale, cb25db (100M rows, 25 cols), vs DuckDB 16-thread over hits.parquet.*

## Core truth
On 100M single-query latency, WaveDB **wins the cells it turns into a READ; loses the cells where it must SCAN.**
- Warm reads fly: COUNT(*) 1.4ms, global COUNT(DISTINCT) 2.3ms (140-274x), GROUP BY URL top-K 1.3ms, small group 71ms.
- Pure 100M scalar scan: 461ms vs DuckDB 201ms (LOSE). Python-parallel scan does not beat 16-core C++.
- High-card filtered group: 4280ms vs 423ms (LOSE). Filtered high-card group-distinct (Q13): **108,889ms** (LOSE hard).
WaveDB's two structural wins remain: (a) precompute-and-read, (b) concurrent throughput (many single-thread workers, ~14x measured prior).

## The 43 queries collapse to ~8 shape-classes. Closing them all = ~6 structures, 4 already built.

| # | shape-class | example Qs | status now | structure that makes it a READ | have it? |
|---|---|---|---|---|---|
| A | COUNT(*) global | Q0 | WIN 1.4ms (stored N) | stored row count | YES |
| B | global COUNT(DISTINCT) | Q4,Q5 | WIN 140-274x (dict V) | dictionary cardinality | YES |
| C | GROUP BY low-card + agg | Q7, URL-topK | WIN 1-71ms | cube / gbcount | YES |
| D | per-key COUNT(DISTINCT), no filter | Q8,Q9 | materializable -> 0.58ms | gd-sidecar / groupmix | YES (built) |
| E | scalar agg over full scan | Q1,Q2,Q3,Q30 | LOSE 461 vs 201 | stored per-col SUM/COUNT/MIN/MAX | trivial, not wired |
| F | per-key COUNT(DISTINCT) + WHERE | Q10,Q11,Q13 | **BANKED**: Q13 108s->800ms (beats duck 967), Q10 213x | sidecar + filter-on-group-key | YES (single key) |
| G | high-card compound group | Q14,Q15,Q31,Q32,Q33 | LOSE 4-6s | compound-key group-count cube | partial (compound/survgroup) |
| H | date-needle filter -> group top-K | Q37-Q42 | scan / date-cast gap | filter zone-map/BSI + group cube | BSI exists, date typing gap |
| I | point lookup WHERE col = literal | Q20 | LOSE 35s | inverted index (value->rows) | NO |
| J | sort-key row top-K (ORDER BY EventTime LIMIT) | Q24,Q25,Q26,Q27 | LOSE / unsupported | sorted-index top-K read | NO |
| K | unsupported functions | Q18,Q23,Q28,Q29,Q35,Q36,Q43 | ERROR | parser/operator coverage (orthogonal) | NO |

## Finite open cells (the work)
1. **F: sidecar + filter-on-group-key** — when WHERE is confined to the group key (col <> '' etc), filter served groups. Closes Q13 (108s->read), Q10, Q11. *(BANKED FIRST — cheapest, highest value.)*
2. **E: stored per-column scalars** — SUM/COUNT/MIN/MAX per col. Closes Q2/Q3/Q30 aggregate cells; Q1/Q6 with a low-card value-count.
3. **G: compound-key group-count cube** — Q14/Q15/Q31/Q32/Q33. Storage gated by compound cardinality.
4. **H: filter-zone + group cube (date-needle)** — Q37-Q42. Needs EventDate integer-range typing + derived-column (EventTime->EventDate) discovery.
5. **I: inverted index** — Q20 point lookup. value->rows postings.
6. **J: sorted-index top-K** — Q24-Q27 row projection by sort key.
7. **K: function coverage** — orthogonal feature work, NOT part of the materialize-relationships thesis.

## How the relationship-discovery engine maps onto this
The fan-out / approximate-FD / derived-column detector is the GENERAL mechanism that auto-picks which structure (D/F sidecar, G cube, H derived-column) to build per dataset, gated by cheap isolation measurements (cardinality, fan-out, FD-strength) then sample-probe then commit-on-survivor. Not infinite relationships — it chooses among the ~6 structure-types above. Every materialization falls through to a proven scan if it declines: worst case wasted storage/time, never a wrong answer. Main long-term cost = staleness as materializations multiply (sidestepped today: immutable segments).

## Correctness/precision notes seen in the run
- Q3 (AVG(UserID)) ok=False: float precision on huge ids.
- Q17 ok=False: top-K tie ordering.
- Q20/Q22 ok=False: under investigation (point-lookup / LIKE result ordering).

---

## Session log — 2026-06-14 (correctness audit)

Commits (pushed):
- `984151f` — gd-sidecar serves COUNT(DISTINCT) with a group-key-only WHERE. Q13 108,889ms -> ~800ms (beats DuckDB), Q10 0.99ms (213x), bit-exact. (cell F, single group key.)
- `421bab8` — SQL LIKE missing `re.DOTALL` in the fused wdb_join paths. Q20 `URL LIKE '%google%'` 15908 -> 15911 (bit-exact vs DuckDB). +2 regression tests. Suite 1645 green.

Audited the 6 board `ok=False` flags. Verdict: only ONE was a real bug.
- **Q20** — REAL: LIKE->regex compiled `^.*pat.*$` without DOTALL, so `.` couldn't cross embedded `\r\n`; 3 long URLs had "google" after a newline. Root-caused on 100M (dict-code frequency proved data = 15911 exact; storage flawless). FIXED + verified.
- **Q17** — false alarm: unordered `LIMIT` (nondeterministic; matches DuckDB when ordered).
- **Q38 / Q40 / Q41** — false alarms: full grouped aggregations are BIT-EXACT vs DuckDB (13299 / 41194 / 10948 groups all match). The `ok=False` was `OFFSET`-into-tie-bands + EventDate int-vs-date display in the harness. **The earlier board OVERSTATED the date-needle class as "broken/wrong" — it computes correctly, just slowly.**
- **Q03** — `AVG(UserID)` sums uint64 in float64, exact only to ~13 sig figs (float-inherent; DuckDB uses int128). Not "wrong", but the lone non-bit-exact result. Optional fix: hi/lo 32-bit split integer accumulator in wdb_exprjit.

Net: everything WaveDB computes is now **bit-exact except Q03**.

Persisted on volume (survive pod stop/start): cb25db + both gd sidecars (SearchPhrase 96MB, MobilePhoneModel 3.5KB), queries.sql, this map.

Open threads for next session:
1. Q03 exact-integer-sum (optional, to claim literal 100% bit-exact).
2. Q29 hard crash — the 90-column `SUM(ResolutionWidth + k)` query kills the process (stability bug).
3. Build the RegionID gd-sidecar to flip Q08/Q09 from walk (~0.16x) to the ~200x read class (proven by Q10).
