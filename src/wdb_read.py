"""Column reader for the encoder. Produces, per column, numpy matching what the encoder
expects: numpy.ma.MaskedArray (original dtype) for nullable columns, plain ndarray else.

Two backends:
  - arrow:  pyarrow.parquet (parquet only). Builds masks from Arrow's validity bitmap and
            fill_null to preserve dtype (Arrow's to_numpy would promote nullable int->float
            and drop the mask). Avoids the DuckDB dependency on the read path.
  - duckdb: DuckDB fetchnumpy (any format DuckDB reads: parquet, csv, ...). The fallback.

'auto' uses arrow for .parquet inputs, duckdb otherwise. Both produce byte-identical
encoder output (verified across all dtypes incl. nulls).
"""
import numpy as np, numpy.ma as ma

def _arrow_default(t):
    import pyarrow as pa
    if pa.types.is_integer(t):   return pa.scalar(0, type=t)
    if pa.types.is_floating(t):  return pa.scalar(0.0, type=t)
    if pa.types.is_timestamp(t): return pa.scalar(0, type=t)
    if pa.types.is_boolean(t):   return pa.scalar(False, type=t)
    if pa.types.is_string(t) or pa.types.is_large_string(t): return pa.scalar('', type=t)
    return pa.scalar(0, type=t)

def _arrow_col_to_numpy(col):
    import pyarrow as pa
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if col.null_count == 0:
        return col.to_numpy(zero_copy_only=False)
    mask = col.is_null().to_numpy(zero_copy_only=False)
    vals = col.fill_null(_arrow_default(col.type)).to_numpy(zero_copy_only=False)
    return ma.MaskedArray(vals, mask=mask)

def _read_arrow(path, columns):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=columns)
    cols = columns if columns else t.column_names
    coldata = {nm: _arrow_col_to_numpy(t.column(nm)) for nm in cols}
    return coldata, t.num_rows, cols

def _read_duckdb(path, columns):
    import duckdb
    con = duckdb.connect(); con.execute("PRAGMA threads=8")
    src = f"'{path}'"
    schema = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
    cols = columns if columns else [r[0] for r in schema]
    N = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    coldata = {nm: con.execute(f'SELECT "{nm}" FROM {src}').fetchnumpy()[nm] for nm in cols}
    return coldata, N, cols

def read_columns(path, columns=None, reader='auto'):
    """Return (coldata, n_rows, cols). reader: 'auto' | 'arrow' | 'duckdb'."""
    if reader == 'arrow' or (reader == 'auto' and str(path).lower().endswith('.parquet')):
        try:
            return _read_arrow(path, columns)
        except Exception:
            if reader == 'arrow':
                raise
            return _read_duckdb(path, columns)   # auto: fall back on any arrow failure
    return _read_duckdb(path, columns)
