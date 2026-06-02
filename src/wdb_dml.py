"""WaveDB DML: INSERT. Step 3a - single segment per table.

Each table keeps a buffer parquet (canonical rows); the segment is derived from it.
INSERT appends rows to the buffer and re-encodes the ONE segment. This buffer is also
the precursor to the operator's hot-tier option (3b): high-traffic tables can keep
appending without re-encoding every time.
"""
import os, sqlglot, sqlglot.expressions as E
import pandas as pd, numpy as np
import wdb_encode, wdb_labels

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

def _next_segment_index(catalog, name):
    idxs = []
    for sfile in catalog.get_table(name)['segments']:
        base = sfile[:-4] if sfile.endswith('.wdb') else sfile
        try: idxs.append(int(base.rsplit('_', 1)[1]))
        except (IndexError, ValueError): pass
    return max(idxs) + 1 if idxs else 0

def flush(catalog, name):
    """Encode the hot buffer into a NEW cold segment and append it; clear hot.
    O(hot) — existing segments are untouched (this is what produces multi-segment tables)."""
    hp = hot_path(catalog, name)
    if not os.path.exists(hp):
        return 0
    hot_df = pd.read_parquet(hp)
    idx = _next_segment_index(catalog, name)
    seg_file = f"{name}_{idx}.wdb"
    wdb_encode.encode(hp, os.path.join(catalog.dbdir, seg_file))
    # discover FD labels for this segment (sampled, ~40ms) so the compactor can later
    # VERIFY rather than rediscover. Labels are hints, never trusted without verification.
    labels = wdb_labels.discover_fds(hot_df)
    catalog.add_segment(name, seg_file, labels=labels)
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


def _filter_parquet_keep_complement(path, pred_sql):
    """Rewrite a parquet keeping only rows the DELETE should NOT remove.
    DELETE removes rows where the predicate is definitely TRUE; rows where it is FALSE or
    NULL stay (SQL semantics) -> keep `(pred) IS NOT TRUE`. pred_sql None means delete all."""
    import duckdb
    if pred_sql is None:
        df = pd.read_parquet(path).iloc[0:0]
    else:
        con = duckdb.connect()
        df = con.execute(
            f"SELECT * FROM read_parquet('{path}') WHERE ({pred_sql}) IS NOT TRUE"
        ).fetch_df()
    df.to_parquet(path, index=False)

def delete(catalog, sql):
    """DELETE FROM t [WHERE ...]. Strategy matches the table's storage class:
      buffered -> tombstone matching rows in each cold segment's presence sidecar (no rewrite)
                  + filter the hot buffer parquet;
      segment  -> filter the canonical buffer and re-encode the single segment (buffer is
                  source of truth; re-encoding is already how segment-mode works).
    Returns the number of rows deleted."""
    import wdb_sql, wdb_presence
    from wdb_engine import Segment
    tree = sqlglot.parse_one(sql, read='duckdb')
    name = tree.find(E.Table).name
    wnode = tree.args.get('where')
    pred = wnode.this if wnode is not None else None
    pred_sql = pred.sql(dialect='duckdb') if pred is not None else None
    mode = catalog.table_mode(name)
    deleted = 0

    if mode == 'buffered':
        segs = catalog.get_table(name)['segments']
        for sf, sp in zip(segs, catalog.segment_paths(name)):
            seg = Segment(sp)
            if pred is None:
                idx = np.arange(seg.N)
            else:
                m = wdb_sql._eval_pred(seg, pred, lambda x: x)
                idx = np.nonzero(m)[0]
            if len(idx):
                deleted += wdb_presence.mark_deleted(sp, seg.N, idx)
        hp = hot_path(catalog, name)
        if os.path.exists(hp):
            before = len(pd.read_parquet(hp))
            _filter_parquet_keep_complement(hp, pred_sql)
            deleted += before - len(pd.read_parquet(hp))
    else:
        bpath = _buffer_path(catalog, name)
        if os.path.exists(bpath):
            before = len(pd.read_parquet(bpath))
            _filter_parquet_keep_complement(bpath, pred_sql)
            deleted += before - len(pd.read_parquet(bpath))
            _encode_segment(catalog, name, bpath)
    return deleted
