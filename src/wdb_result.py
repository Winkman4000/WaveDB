"""wdb_result -- THE CURSOR (2026-10-22, Q7's huge answers).

The result interface the users of DuckDB's and ClickHouse's Python clients expect:
  - PEP 249's cursor: description, rowcount, arraysize, fetchone / fetchmany / fetchall, iteration;
  - columnar fetches beside it: fetchnumpy (a dict of arrays), arrow / to_arrow_table, fetch_record_batch (Arrow
    record batches, 131,072 rows each by default, as DuckDB's reader), df (pandas).
A result stays in columns inside; Python values and row tuples are built only for the rows a caller fetches, a batch at
a time -- ClickHouse streams a result as blocks of columns ("only loads a single block at a time"), DuckDB hands back
Arrow batches. db.execute(sql) returns a Cursor; db.run(sql) is unchanged (it is fetchall).

Reads that can answer in columns hand back a Columns object -- only when the outermost db.execute asked (WANT), never
to a nested run (a subquery's caller expects rows). Every other read's rows are wrapped as they are.
A lazily decoded column (text from the dictionary) decodes outside the query that made it: the cursor ends each such
batch the way a query ends (wdb_qmem.flush) when no query is running, so no decoded state outlives it (Jackson's law)."""
import numpy as np

WANT = [False]                    # Database.execute sets it for its outermost run: a read may answer with Columns
DEFAULT_BATCH = 131072            # rows per Arrow record batch (DuckDB's to_arrow_reader default)


class ArrayColumn:
    """a column held as one array; py(slice) -> Python values (default: tolist)"""
    lazy = False

    def __init__(self, arr, py=None):
        self.a = arr
        self.n = len(arr)
        self.py = py

    def take(self, lo, hi):
        s = self.a[lo:hi]
        return self.py(s) if self.py is not None else s.tolist()

    def array(self, lo=0, hi=None):
        return self.a[lo:hi]


class ConstColumn:
    """one value n times (an echo: SELECT c WHERE c = lit) -- no array, no rows until fetched"""
    lazy = False

    def __init__(self, value, n):
        self.v = value
        self.n = int(n)

    def take(self, lo, hi):
        return [self.v] * max(0, min(hi, self.n) - lo)

    def array(self, lo=0, hi=None):
        hi = self.n if hi is None else min(hi, self.n)
        k = max(0, hi - lo)
        if isinstance(self.v, (int, float)) and not isinstance(self.v, bool):
            return np.full(k, self.v)
        out = np.empty(k, dtype=object)
        out[:] = [self.v] * k
        return out


class LazyColumn:
    """a column decoded per batch: decode(lo, hi) -> Python values of rows lo..hi-1"""
    lazy = True

    def __init__(self, n, decode):
        self.n = n
        self.decode = decode

    def take(self, lo, hi):
        return self.decode(lo, hi)

    def array(self, lo=0, hi=None):
        hi = self.n if hi is None else hi
        out = np.empty(max(0, hi - lo), dtype=object)
        out[:] = self.decode(lo, hi)
        return out


class Columns:
    """a columnar result: one column object per name, all of the same length"""

    def __init__(self, names, cols):
        self.names = list(names)
        self.cols = list(cols)
        self.n = self.cols[0].n if self.cols else 0

    def __len__(self):
        return self.n

    def rows(self, lo, hi):
        return list(zip(*[c.take(lo, hi) for c in self.cols]))


class Cursor:
    """PEP 249 cursor over one result (Columns, or rows already built)"""
    arraysize = 1

    def __init__(self, result, names, db=None):
        self._db = db
        if isinstance(result, Columns):
            self._c = result
            self._r = None
            names = names or result.names
            n = len(result)
        else:
            self._c = None
            self._r = result if isinstance(result, list) else list(result)
            n = len(self._r)
        self.description = None if names is None else [(str(nm), None, None, None, None, None, None) for nm in names]
        self._names = None if names is None else [str(nm) for nm in names]
        self.rowcount = n
        self._pos = 0

    # ---------------------------------------------------------------- rows
    def _end_batch(self):
        """a lazy column decoded outside its query: end the batch as a query ends (nothing decoded outlives it)"""
        db = self._db
        if db is not None and self._c is not None and any(c.lazy for c in self._c.cols) \
                and getattr(db, '_qdepth', 0) == 0:
            import wdb_qmem
            wdb_qmem.flush(db)

    def _take(self, k):
        lo = self._pos
        hi = min(self.rowcount, lo + max(0, int(k)))
        self._pos = hi
        if hi <= lo:
            return []
        if self._r is not None:
            return self._r[lo:hi]
        try:
            return self._c.rows(lo, hi)
        finally:
            self._end_batch()

    def fetchone(self):
        r = self._take(1)
        return r[0] if r else None

    def fetchmany(self, size=None):
        return self._take(self.arraysize if size is None else size)

    def fetchall(self):
        return self._take(self.rowcount - self._pos)

    def __iter__(self):
        while self._pos < self.rowcount:
            for r in self._take(2048):
                yield r

    def close(self):
        self._c = None
        self._r = []
        self.rowcount = 0
        self._pos = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ---------------------------------------------------------------- columns
    def _column_arrays(self, lo, hi):
        """the rows lo..hi-1 as one array per column"""
        if self._c is not None:
            try:
                return [c.array(lo, hi) for c in self._c.cols]
            finally:
                self._end_batch()
        rows = self._r[lo:hi]
        ncol = len(self._names) if self._names else (len(rows[0]) if rows else 0)
        out = []
        for j in range(ncol):
            vals = [r[j] for r in rows]
            try:
                a = np.asarray(vals)
                if a.dtype.kind in 'USV':                  # text stays as Python objects (no fixed-width copy)
                    raise ValueError
            except Exception:
                a = np.empty(len(vals), dtype=object)
                a[:] = vals
            out.append(a)
        return out

    def _names_or_default(self, ncol):
        return self._names if self._names else ['column%d' % j for j in range(ncol)]

    def fetchnumpy(self):
        """the remaining rows as {name: array}"""
        lo, hi = self._pos, self.rowcount
        arrs = self._column_arrays(lo, hi)
        self._pos = hi
        return dict(zip(self._names_or_default(len(arrs)), arrs))

    def df(self):
        import pandas as pd
        return pd.DataFrame(self.fetchnumpy())

    fetchdf = df

    def fetch_record_batch(self, rows_per_batch=DEFAULT_BATCH):
        """the remaining rows as a pyarrow RecordBatchReader, rows_per_batch rows a batch"""
        import pyarrow as pa
        rows_per_batch = max(1, int(rows_per_batch))

        def _batch(lo, hi):
            arrs = self._column_arrays(lo, hi)
            names = self._names_or_default(len(arrs))
            return pa.RecordBatch.from_arrays([pa.array(a) for a in arrs], names=names)

        lo0, n = self._pos, self.rowcount
        first = _batch(lo0, min(n, lo0 + rows_per_batch))
        schema = first.schema

        def _gen():
            yield first
            lo = lo0 + rows_per_batch
            while lo < n:
                hi = min(n, lo + rows_per_batch)
                yield _batch(lo, hi).cast(schema)
                lo = hi
        self._pos = n
        return pa.RecordBatchReader.from_batches(schema, _gen())

    to_arrow_reader = fetch_record_batch

    def arrow(self, batch_size=DEFAULT_BATCH):
        """the remaining rows as a pyarrow Table"""
        return self.fetch_record_batch(batch_size).read_all()

    to_arrow_table = arrow
    fetch_arrow_table = arrow
