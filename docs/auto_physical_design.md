# Auto Physical Design — salvaged design notes

**Status: NOT IMPLEMENTED.** Captured from the retired `wdb_plan.py` prototype
(built 2026-06-05, two commits same day, never touched again; superseded by the
in-operator policy/measurement layer and deleted 2026-06-14). These are the two
ideas worth keeping *if* WaveDB ever pursues automatic physical design.

Everything else in `wdb_plan` — the per-column "levers"
(`code_lut`/`content_key`/`range_sliceable`/`groupable`) — is now re-derived live
by `wdb_policies` + `wdb_measure_runtime` and was **not** salvaged (redundant).

---

## 1. Workload-driven cluster-key recommender

WaveDB can physically sort a segment by one column at encode time
(`encode(..., cluster_by=K)`). Today K is chosen **by hand**. The prototype chose
K automatically by scoring candidates against a query workload.

**Eligible candidate K:** value-identity codes (mode in {0,1,2,5,6}) AND
sorted-storable, excluding `*_ptr` columns.

**Objective** — score(K) = filter work saved + group-locality bonus, summed over
the columns K makes sliceable:

    score(K) = Σ_C  wf(C) · (1 − sel(C))     [filter rows skipped]
             + g · Σ_C wg(C)                  [hash-group → run-group]
    for C in coverage(K)

where:
  - `wf(C)`, `wg(C)` = how often column C appears as a filter / GROUP BY key in the
    workload (counted by regex over the query log).
  - `sel(C)` = measured fraction of rows a typical filter on C keeps (lower = more
    selective), measured via WaveDB's OWN `COUNT(*) WHERE` on real data — no
    external oracle.
  - `g` = group-locality factor ≈ 0.25. Clustering the group key turns
    hash-grouping into sequential run-grouping: a *rate* gain, not a work cut, so
    weighted below a filter row-save. NEVER measured end-to-end — needs calibration.

### Compounding coverage (the non-obvious bit worth keeping)

Clustering by K also makes a *different* column C sliceable IF sorting by K
happens to leave C nearly monotone. Measured on a data sample: sort by K, then
check the fraction of adjacent descents in C; if < 0.10, C is "covered" by K.
This captures correlated columns (e.g. sorting by ship-date also roughly orders
receipt-date), so one permutation can pay off for several filters.

Fall back to a structural prior when the workload doesn't discriminate (score ≤ 0):
datetime dimension > higher-cardinality non-unique dim; unique keys are poor
dimensions.

---

## 2. Host calibration → runtime constants (the MACHINE measurement pillar)

`wdb_measure_runtime` currently **hand-sets** its gate constants (BSI RAM budget,
LUT cardinality cap, parallel threshold, locality factor, etc.). The retired
`wdb_calibrate` measured the host's real analog constants once (sequential vs
random read rate, locality penalty, cache cliff = `lut_cliff`) and cached them to
`~/.wavedb/calibration.json`.

The intended-but-never-wired step: feed those measured host constants into the
runtime gates so thresholds **auto-tune per machine** instead of being hand-set.
See the "hand-set for now" comments in `wdb_measure_runtime`.

If pursued: measure once on first run, persist, have the RT gates read calibrated
values with the current hand-set numbers as fallback defaults.
