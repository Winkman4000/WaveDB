"""WaveDB DML: INSERT. Step 3a - single segment per table.

Each table keeps a buffer parquet (canonical rows); the segment is derived from it.
INSERT appends rows to the buffer and re-encodes the ONE segment. This buffer is also
the precursor to the operator's hot-tier option (3b): high-traffic tables can keep
appending without re-encoding every time.
"""
import os, sqlglot, sqlglot.expressions as E
import pandas as pd, numpy as np
import wdb_encode

_PD = {'int': 'Int64', 'float': 'float64', 'string': 'object', 'datetime': 'datetime64[ns]'}

def _litval(node):
    if isinstance(node, E.Null): return None
    if isinstance(node, E.Boolean): return 1 if node.this else 0
    if isinstance(node, E.Neg): 
        v = _litval(node.this); return None if v is None else -v
    if isinstance(node, E.Literal):
        if node.args.get('is_string'): return node.this
        s = node.this
        return float(s) if ('.' in s or 'e' in s.lower()) else int(s)
    raise NotImplementedError(f"unsupported literal: {node.sql()}")

def _coerce(val, wtype):
    if val is None: return None
    if wtype == 'int':      return int(val)
    if wtype == 'float':    return float(val)
    if wtype == 'string':   return str(val)
    if wtype == 'datetime': return pd.Timestamp(val)
    raise NotImplementedError(f"type {wtype}")

def parse_insert(sql):
    """Return (table_name, columns_or_None, list_of_rows)."""
    t = sqlglot.parse_one(sql, read='duckdb')
    if not isinstance(t, E.Insert): raise NotImplementedError("not an INSERT")
    if isinstance(t.this, E.Schema):
        name = t.this.this.name; cols = [c.name for c in t.this.expressions]
    else:
        name = t.this.name; cols = None
    vals = t.expression
    if not isinstance(vals, E.Values): raise NotImplementedError("INSERT ... SELECT not supported yet")
    rows = [[_litval(x) for x in tup.expressions] for tup in vals.expressions]
    return name, cols, rows

def _buffer_path(catalog, name): return os.path.join(catalog.dbdir, f"{name}__buffer.parquet")
def _segment_name(name): return f"{name}_0.wdb"

def insert(catalog, sql):
    name, cols, rows = parse_insert(sql)
    tinfo = catalog.get_table(name)
    schema = tinfo['schema']                       # [[col, wtype], ...]
    scol = [c[0] for c in schema]; stype = {c[0]: c[1] for c in schema}
    order = cols if cols else scol
    if cols and set(cols) != set(scol):
        raise ValueError(f"INSERT columns {cols} don't match table columns {scol}")
    for r in rows:
        if len(r) != len(order):
            raise ValueError(f"row has {len(r)} values, expected {len(order)}")
    # build new-rows dict in schema order, coerced
    newdata = {c: [] for c in scol}
    for r in rows:
        rowmap = dict(zip(order, r))
        for c in scol:
            newdata[c].append(_coerce(rowmap.get(c), stype[c]))
    df_new = pd.DataFrame({c: pd.array(newdata[c], dtype=_PD[stype[c]]) for c in scol})
    # append to buffer
    bpath = _buffer_path(catalog, name)
    if os.path.exists(bpath):
        df_old = pd.read_parquet(bpath)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(bpath, index=False)
    # re-encode the single segment from the buffer
    seg_file = _segment_name(name); seg_path = os.path.join(catalog.dbdir, seg_file)
    wdb_encode.encode(bpath, seg_path)
    if seg_file not in tinfo['segments']:
        tinfo['segments'] = [seg_file]; catalog.save()
    return len(rows)
