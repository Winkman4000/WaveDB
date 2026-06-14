"""Unit tests for wdb_measure_runtime -- the runtime/derived measurement thresholds.

Anti-cheat for a measurement migration: these PIN the exact calibrated values. The gate only affects
which path runs (not the answer), so a correctness test can't catch a drifted constant -- only pinning
the numbers proves the move out of wdb_survgroup changed nothing."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import wdb_measure_runtime as RT


def test_calibration_constants_pinned():
    assert RT.SURVIVOR_GATE_SLOPE == 1.6
    assert (RT.SURVIVOR_GATE_LO, RT.SURVIVOR_GATE_HI) == (0.05, 0.35)
    assert RT.SEQ_EXC_FRAC == 0.05


def test_survivor_gate_band():
    # mid-band: slope * K / N
    assert abs(RT.survivor_gate(1000, 100) - 0.16) < 1e-12   # 1.6 * 100/1000
    # below the band clamps up to LO; above clamps down to HI
    assert RT.survivor_gate(1_000_000, 1) == RT.SURVIVOR_GATE_LO
    assert RT.survivor_gate(10, 1_000_000) == RT.SURVIVOR_GATE_HI
    # degenerate N
    assert RT.survivor_gate(0, 5) == RT.SURVIVOR_GATE_LO
    assert RT.survivor_gate(-1, 5) == RT.SURVIVOR_GATE_LO


def test_structural_pushdown_worth_it():
    assert RT.structural_pushdown_worth_it(None, 1000) is False
    assert RT.structural_pushdown_worth_it(50, 1000) is True    # boundary: 50 <= 0.05*1000 inclusive
    assert RT.structural_pushdown_worth_it(51, 1000) is False


def test_bsi_gates():
    assert RT.BSI_SEL_CEIL == 0.35
    assert RT.BSI_RAM_BUDGET == (1 << 26)
    # selectivity ceiling: > 35% of rows -> too unselective
    assert RT.bsi_too_unselective(36, 100) is True
    assert RT.bsi_too_unselective(35, 100) is False        # boundary: 35 <= 0.35*100 inclusive
    # RAM budget: current + add must stay within budget
    assert RT.bsi_index_fits(0, 1 << 26) is True           # exactly the budget fits
    assert RT.bsi_index_fits(1, 1 << 26) is False          # one byte over
