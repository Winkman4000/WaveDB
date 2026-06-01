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
def hot_path(catalog, name):    return os.path.join(catalog.dbdir, f"{name}__hot.parquet")
def _segment_name(name): return f"{name}_0.wdb"

def _rows_to_df(schema, order, rows):
    scol = [c[0] for c in schema]; stype = {c[0]: c[1] for c in schema}
    if order and set(order) != set(scol):
        raise ValueError(f"INSERT columns {order} don't match table columns {scol}")
    order = order if order else scol
    for r in rows:
        if len(r) != len(order):
            raise ValueError(f"row has {len(r)} values, expected {len(order)}")
    data = {c: [] for c in scol}
    for r in rows:
        rowmap = dict(zip(order, r))
        for c in scol:
            data[c].append(_coerce(rowmap.get(c), stype[c]))
    return pd.DataFrame({c: pd.array(data[c], dtype=_PD[stype[c]]) for c in scol})

def _append_parquet(path, df_new):
    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(path, index=False)
    return df

def _encode_segment(catalog, name, src_parquet):
    seg_file = _segment_name(name); seg_path = os.path.join(catalog.dbdir, seg_file)
    wdb_encode.encode(src_parquet, seg_path)
    tinfo = catalog.get_table(name)
    if tinfo['segments'] != [seg_file]:
        tinfo['segments'] = [seg_file]; catalog.save()

def flush(catalog, name):
    """Fold the hot buffer into the cold segment, then clear hot. (buffered tables)"""
    hp = hot_path(catalog, name)
    if not os.path.exists(hp):
        return 0
    hot_df = pd.read_parquet(hp)
    bpath = _buffer_path(catalog, name)
    cold_df = _append_parquet(bpath, hot_df)      # buffer becomes full canonical set
    _encode_segment(catalog, name, bpath)
    os.remove(hp)
    return len(hot_df)

def insert(catalog, sql):
    name, cols, rows = parse_insert(sql)
    tinfo = catalog.get_table(name)
    schema = tinfo['schema']
    df_new = _rows_to_df(schema, cols, rows)
    mode = tinfo.get('mode', 'segment')
    if mode == 'buffered':
        # operator opted in: append to the hot buffer only, no re-encode (the win)
        _append_parquet(hot_path(catalog, name), df_new)
    else:
        # default: buffer is canonical; re-encode the single segment each time
        bpath = _buffer_path(catalog, name)
        _append_parquet(bpath, df_new)
        _encode_segment(catalog, name, bpath)
    return len(rows)
