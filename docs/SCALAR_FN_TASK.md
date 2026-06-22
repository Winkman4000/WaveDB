# Task: scalar function on a dict column (general), starting with length()

## Goal
Make `length(col)` work anywhere a plain int column works: as an aggregate argument
(`AVG/SUM/MIN/MAX/COUNT(length(col))`) and as a GROUP BY key (`GROUP BY length(col)`). Build it as a
GENERAL "scalar function evaluated over a column's dictionary" mechanism, with `length` as the first
registered function -- so adding `lower`, `octet_length`, etc. later is just registering a function.

## Why this is cheap on WaveDB (the design)
Dict columns store each distinct value once. A scalar function `f(value)` that depends only on the
value is a property of the DICTIONARY ENTRY, not the row. Compute `f` over the V distinct values once
-> `fval_by_code[V]`; per-row value is then `fval_by_code[codes]` (a gather). This is the SAME shape
the date-coarsening group key already uses (`_group_key` -> `('fn', col, unit)` evaluated over
`_typed_dict`). We mirror it.

## Correctness (matches DuckDB)
- DuckDB `length(str)` = CHARACTER count (not bytes). Compute `len(decoded_str)`. For ASCII this
  equals byte length, but use char length to be correct for multi-byte UTF-8.
- `length(NULL)` = NULL (SQL). Preserve null mask: a null code maps to a null result, and SQL
  aggregates skip nulls (existing `_col` null handling already does this once we route through it).
- Empty string '' has length 0 (it is a normal value, not null).

## Registration
Add a small table `_SCALAR_FNS = { 'LENGTH': _fn_length, ... }` where each entry maps a dict of
typed values -> an int64 array of results (length V). `_fn_length(td)` returns
`np.array([0 if v is None else len(v.decode() if bytes else str(v)) for v in td], np.int64)`.

## Touch points in src/wdb_sql.py
1. `_scalar_fn(node)` NEW: classify an expression as `('sfn', fname, colname)` if it is
   `FUNC(column)` with FUNC in `_SCALAR_FNS`, else None. (sqlglot: `E.Length`, or generic
   `E.Anonymous`/`E.Func` with `.this` an `E.Column`.)
2. `_group_key`: before the final `raise`, try `_scalar_fn(g)`; if it matches, return
   `('sfn', col, fname)`. Then in the group-value builder (the `for gn, gk in zip(gnames, gkeys)`
   loop), handle `gk[0]=='sfn'` exactly like the `'fn'` date branch: evaluate `fval_by_code` over
   the dict, gather per row, `np.unique` -> distinct group values + inverse. (Group values are
   plain ints, so ordering/typing is trivial.)
3. Aggregates: `_agg_kind(p)` currently returns `(FN, colname)` for `FN(col)`. Extend so it can
   return `(FN, ('sfn', fname, colname))`; and `_agg_scalar` / `_agg_scalar_range`, where they call
   `_col(seg, cn)`, detect the sfn form and instead build the derived array
   `fval_by_code[codes]` with the correct null mask (null code -> null), then run the existing
   SUM/AVG/MIN/MAX/COUNT logic unchanged.
4. Helper `_sfn_array(seg, fname, cn, mask)` NEW: returns `(arr_int64, nmask_or_None)` =
   `fval_by_code[codes]` (+ null mask from the column), applying `mask` if given. Used by both the
   group path and the agg path so there is ONE implementation.
5. Cache `fval_by_code` per (col, fname) on the segment (like `code_counts`) so repeat use is free.

## Out of scope (for now)
Functions of MULTIPLE columns, functions whose result is not value-identity-per-code (e.g. random),
nested functions `f(g(col))`. Registry makes these addable later; not needed for Q27.

## Acceptance
- Q27 (`AVG(length(URL)) ... GROUP BY CounterID HAVING COUNT(*)>k ORDER BY l DESC LIMIT`) returns
  bit-exact vs DuckDB on the synthetic table, AND `GROUP BY length(col)` returns bit-exact.
- Existing suite stays green (no regression in plain-column agg/group paths).
- Then (pod) confirm on 100M hits Q27 exact vs DuckDB + a timing baseline.
