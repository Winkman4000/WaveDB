# WaveDB test suite

Zero-dependency. Run all tests:

    python3 tests/run.py

Run a subset (substring filter on module.testname):

    python3 tests/run.py roundtrip
    python3 tests/run.py correctness.test_groupby

Exit code is nonzero if any test fails (CI-friendly).

## Layout
- `helpers.py`         synthetic-data -> encode -> decode round-trip + comparison
- `run.py`             discovers and runs every `test_*` function in `test_*.py`
- `test_roundtrip.py`  lossless round-trip: every dtype, every mode (0/1/2), nulls, edges
- `test_correctness.py` same SQL through WaveDB and DuckDB must agree (oracle)
- `test_memory.py`     regression guard: codes() decodes in bounded memory (not N x bits)

## Adding tests
Write a `test_*` function in any `test_*.py`; it passes if it returns without
raising. Use `helpers.roundtrip(df)` to get a Segment from a pandas DataFrame.
Run the suite before every commit.
