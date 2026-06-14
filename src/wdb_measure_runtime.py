"""
wdb_measure_runtime -- runtime / query-derived measurements and the cost thresholds that read them.

The THIRD measurement file. The other two already exist:
    wdb_profile    -- DATA measurements (per-column: cardinality, entropy, sortedness) [at rest]
    wdb_calibrate  -- MACHINE measurements (host read rates, locality penalty, cache cliff)
This one holds quantities that are only knowable per query, against the live data + predicate, plus
the calibrated thresholds that turn a measured quantity into a path choice. Centralized so that
"why did the router take path A over path B?" is answerable from ONE auditable function instead of a
magic number buried in an operator body.

Honesty note: these are NOT the cheap static guards (those live in wdb_policies). A selectivity is
only known after the predicate is evaluated, so the operator computes it mid-flight and then asks a
threshold function here whether its specialized path still wins; if not, it falls through to the
(correct) general path. These gate a path CHOICE, not eligibility.
"""

# --- survivor-set vs dense GROUP BY COUNT(*) crossover (used by wdb_survgroup) ---
# Crossover calibration (from the selectivity sweep on this class of machine). survivor-set cost
# ~ c*survivors; K-wide cost ~ a*N + b*K, roughly flat in survivors -> crossover survivors* grows
# with K. Empirically survivors*/N ~ 0.30 at K/N ~ 0.18, so gate ~ 1.6 * K/N, clamped to a safe band.
# The gate only affects SPEED -- a wrong gate just defers to the (correct) dense path -- so the band
# is deliberately conservative.
SURVIVOR_GATE_SLOPE = 1.6
SURVIVOR_GATE_LO, SURVIVOR_GATE_HI = 0.05, 0.35


def survivor_gate(N, K):
    """Max survivor fraction at which the survivor-set group-by beats the dense K-wide accumulator,
    for N rows and key cardinality K. Returns a fraction in [LO, HI]; the operator takes the survivor
    path iff (measured survivors) <= survivor_gate(N, K) * N."""
    if N <= 0:
        return SURVIVOR_GATE_LO
    return min(SURVIVOR_GATE_HI, max(SURVIVOR_GATE_LO, SURVIVOR_GATE_SLOPE * K / N))


# --- structural-pushdown eligibility (used by wdb_survgroup) ---
# Worth pushing `col <op> const` into the filter column's mode-4 exception structure only when the
# exceptions are few relative to N (processing them must beat reading the whole column). Calibrated
# from the exception-count survey (AdvEngineID 0.8% wins; SearchEngineID 17% does not).
SEQ_EXC_FRAC = 0.05


def structural_pushdown_worth_it(nexc, N):
    """True iff the filter column's mode-4 exception count `nexc` is small enough (<= SEQ_EXC_FRAC * N)
    that processing the exceptions beats reading the whole column. None nexc -> not eligible."""
    return nexc is not None and nexc <= SEQ_EXC_FRAC * N



# --- BSI (bit-sliced index) path gates (used by wdb_bsi_exec) ---
# The bitmap walk pays off only when the predicate is selective enough; above this fraction of rows
# surviving, the fused scan is cheaper. And a prospective index must fit a per-segment RAM budget.
BSI_SEL_CEIL = 0.35
BSI_RAM_BUDGET = 1 << 26   # 64 MB of built index per segment


def bsi_too_unselective(cnt, N):
    """True iff `cnt` surviving rows is too large a fraction of N (> BSI_SEL_CEIL) for the bitmap walk
    to beat a fused scan -- the operator then falls back to fused."""
    return cnt > BSI_SEL_CEIL * N


def bsi_index_fits(current_bytes, add_bytes):
    """True iff adding `add_bytes` of BSI planes keeps the segment's index within BSI_RAM_BUDGET."""
    return current_bytes + add_bytes <= BSI_RAM_BUDGET



# --- compound-AND range-path eligibility (used by wdb_compound) ---
# A conjunct `col <op> const` takes the structural range path only when its filter column has few
# enough mode-4 exceptions to process directly; above this ABSOLUTE cap it is left as a residual mask.
# (Absolute cap here, unlike the survivor gate's fraction of N -- the calibration was done that way.)
COMPOUND_RANGE_MAX_EXC = 50000


def compound_range_worth_it(nexc):
    """True iff a conjunct's filter column has few enough exceptions (<= COMPOUND_RANGE_MAX_EXC) to
    take the structural range path; None nexc (not a mode-4 sequence column) -> not eligible."""
    return nexc is not None and nexc <= COMPOUND_RANGE_MAX_EXC



# --- cube materialization cap (used by wdb_cube) ---
CUBE_MAX_CELLS = 4096          # prod(dim cardinalities) cap; above this storage cost outweighs the win.
                               # Measured: a 2,526-cell l_shipdate cube is ~114 KB and answers GROUP BY date
                               # in 0.37ms vs DuckDB 12.4ms (33x/worker). 4096 keeps that in, stays tiny.


def cube_worth_materializing(cell_count, cap=CUBE_MAX_CELLS):
    """True iff a cube of `cell_count` cells (the product of its dims' cardinalities) is small enough
    to be worth materializing -- below the cap it is a few KB and parse-bound; far above it the cube
    costs MB for a query that must emit ~cell_count rows anyway (output-bound)."""
    return 0 < cell_count <= cap
