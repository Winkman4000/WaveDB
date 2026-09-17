"""WaveDB DML: INSERT. Step 3a - single segment per table.

Each table keeps a buffer parquet (canonical rows); the segment is derived from it.
INSERT appends rows to the buffer and re-encodes the ONE segment. This buffer is also
the precursor to the operator's hot-tier option (3b): high-traffic tables can keep
appending without re-encoding every time.
"""
import os, sqlglot, sqlglot.expressions as E
import pandas as pd, numpy as np
import wdb_encode, wdb_labels
from wdb_encode import _crash_point

_PD = {'int': 'Int64', 'float': 'float64', 'string': 'object', 'datetime': 'datetime64[ns]'}
_DT_CODE = {'int': 0, 'string': 1, 'float': 2, 'datetime': 3}
_CANON = {'int': 'int', 'integer': 'int', 'bigint': 'int', 'int64': 'int', 'int32': 'int', 'smallint': 'int',
          'float': 'float', 'double': 'float', 'real': 'float', 'float64': 'float', 'decimal': 'float', 'numeric': 'float',
          'string': 'string', 'str': 'string', 'varchar': 'string', 'text': 'string', 'char': 'string', 'utf8': 'string',
          'datetime': 'datetime', 'timestamp': 'datetime', 'date': 'datetime', 'datetime64': 'datetime', 'bool': 'int', 'boolean': 'int'}

def _touch_stamp(dbdir):
    """A DML WRITE MOVES THE CATALOG STAMP: every per-process verdict about a segment's cleanness
    (hot buffer, overrides, tombstones) is memoised under that stamp, so it must move when the
    truth does; the catalog's contents are unchanged, its mtime is the signal"""
    import os as _o
    try: _o.utime(_o.path.join(dbdir, 'catalog.json'), None)
    except Exception: pass
    try:
        import wdb_sidecar; wdb_sidecar.stamp_moved(dbdir)
    except Exception: pass

def _wtype(t):
    """catalog type names come in many spellings (str, varchar, int64...): one canonical name"""
    return _CANON.get(str(t).lower(), str(t).lower())

def _synth_value(default, wtype):
    """Coerce a catalog default into the engine's per-dtype constant form (strings -> bytes)."""
    if default is None: return None
    if _wtype(wtype) == 'string': return default.encode() if isinstance(default, str) else bytes(default)
    if wtype == 'int':    return int(default)
    if wtype == 'float':  return float(default)
    return default                                  # datetime: engine parses string/epoch

def register_synth(catalog, seg, name):
    """Give `seg` a synthetic constant column for any logical column it physically PREDATES
    (ADD COLUMN): the default is materialized at read instead of rewriting the segment. No-op for
    segments that already carry every column (segment-mode tables materialize eagerly)."""
    tab = catalog.get_table(name); phys = catalog.phys_map(name)
    dmap = tab.get('defaults', {})
    for c, wt in [(x[0], x[1]) for x in tab['schema']]:
        pcol = phys.get(c, c)
        if pcol not in seg.cols:
            seg.add_const_column(pcol, _synth_value(dmap.get(c), _wtype(wt)), _DT_CODE[_wtype(wt)], 0)

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
    w = str(wtype).lower()
    if w in ('int', 'integer', 'bigint', 'int64', 'int32', 'smallint'): return int(val)
    if w in ('float', 'double', 'real', 'float64', 'decimal', 'numeric'): return float(val)
    if w in ('string', 'str', 'varchar', 'text', 'char', 'utf8'): return str(val)
    if w in ('datetime', 'timestamp', 'date', 'datetime64'): return pd.Timestamp(val)
    if w in ('bool', 'boolean'): return bool(val)
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

def _has_cold_segment(catalog, name):
    """a table encoded straight from parquet has a segment and no canonical buffer"""
    try:
        return any(os.path.exists(p) for p in catalog.segment_paths(name))
    except Exception:
        return False

def _buffer_path(catalog, name): return os.path.join(catalog.dbdir, f"{name}__buffer.parquet")
def hot_path(catalog, name):    return os.path.join(catalog.dbdir, f"{name}__hot.parquet")
def _segment_name(name): return f"{name}_0.wdb"

def _rows_to_df(schema, order, rows, phys=None, defaults=None):
    phys = phys or {}; defaults = defaults or {}
    scol = [c[0] for c in schema]; stype = {c[0]: _wtype(c[1]) for c in schema}
    if order and not set(order).issubset(set(scol)):
        raise ValueError(f"INSERT columns {order} include names not in table columns {scol}")
    # omitted columns fall back to their default (ADD COLUMN ... DEFAULT) or NULL
    order = order if order else scol
    for r in rows:
        if len(r) != len(order):
            raise ValueError(f"row has {len(r)} values, expected {len(order)}")
    data = {c: [] for c in scol}
    for r in rows:
        rowmap = dict(zip(order, r))
        for c in scol:
            data[c].append(_coerce(rowmap.get(c, defaults.get(c)), stype[c]))
    df = pd.DataFrame({c: pd.array(data[c], dtype=_PD[stype[c]]) for c in scol})
    # store under PHYSICAL names so buffer/hot parquet and the encoded segment all agree on one
    # stable storage name per column (logical names live in the catalog, resolved at read time)
    if phys:
        df = df.rename(columns={c: phys.get(c, c) for c in scol})
    return df

def _append_parquet(path, df_new):
    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), df_new], ignore_index=True)
    else:
        df = df_new
    tmp = path + '.partial'
    df.to_parquet(tmp, index=False)
    with open(tmp, 'rb') as f: os.fsync(f.fileno())
    os.replace(tmp, path)                              # THE RENAME LAW: never a half-written buffer
    _touch_stamp(os.path.dirname(path))
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
    df_new = _rows_to_df(schema, cols, rows, phys=catalog.phys_map(name),
                         defaults=tinfo.get('defaults', {}))
    mode = tinfo.get('mode', 'segment')
    if mode == 'buffered':
        # operator opted in: append to the hot buffer only, no re-encode (the win)
        _append_parquet(hot_path(catalog, name), df_new)
    else:
        bpath = _buffer_path(catalog, name)
        if os.path.exists(bpath) or not _has_cold_segment(catalog, name):
            # the buffer IS canonical (a table born through the DML path, with or without a
            # segment yet): re-encode the single segment
            _append_parquet(bpath, df_new)
            _encode_segment(catalog, name, bpath)
        else:
            # NO CANONICAL BUFFER (a realm encoded straight from parquet): the segment is the
            # truth. Re-encoding from a fresh buffer holding only the new rows REPLACED a
            # 200,000-row segment with a 1-row one (2026-09-13). Append to the hot buffer;
            # merge-read serves both tiers; flush() promotes the hot rows to a new segment.
            _append_parquet(hot_path(catalog, name), df_new)
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
    phys = catalog.phys_map(name)
    wnode = tree.args.get('where')
    pred = wnode.this if wnode is not None else None
    # parquet (hot/canonical buffer) carries physical names -> translate predicate for DuckDB
    pred_sql = (wdb_sql._to_physical(pred, phys).sql(dialect='duckdb')
                if pred is not None else None)
    mode = catalog.table_mode(name)
    deleted = 0

    if mode == 'buffered':
        segs = catalog.get_table(name)['segments']
        _batch9 = {}                                  # THE ATOMIC DELETE: every segment in ONE write
        for sf, sp in zip(segs, catalog.segment_paths(name)):
            seg = Segment(sp); register_synth(catalog, seg, name)
            if pred is None:
                idx = np.arange(seg.N)
            else:
                m = wdb_sql._eval_pred(seg, pred, lambda x: phys.get(x, x))
                idx = np.nonzero(m)[0]
            if len(idx):
                _batch9[sp] = (seg.N, idx)
        deleted += wdb_presence.mark_deleted_many(_batch9)
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
        elif _has_cold_segment(catalog, name):
            # NO CANONICAL BUFFER (a realm encoded straight from parquet): tombstone the cold
            # segments' presence sidecars -- the buffered strategy. A DELETE that silently
            # removes nothing is a lie (found 2026-09-12: deleted 0 on every parquet realm).
            _batch9b = {}
            for sf, sp in zip(catalog.get_table(name)['segments'], catalog.segment_paths(name)):
                seg = Segment(sp); register_synth(catalog, seg, name)
                if pred is None:
                    idx = np.arange(seg.N)
                else:
                    m = wdb_sql._eval_pred(seg, pred, lambda x: phys.get(x, x))
                    idx = np.nonzero(m)[0]
                if len(idx):
                    _batch9b[sp] = (seg.N, idx)
            deleted += wdb_presence.mark_deleted_many(_batch9b)
            hp = hot_path(catalog, name)
            if os.path.exists(hp):
                before = len(pd.read_parquet(hp))
                _filter_parquet_keep_complement(hp, pred_sql)
                deleted += before - len(pd.read_parquet(hp))
    return deleted


def _literal_rhs(node):
    """Typed python value for a SET RHS that is a literal/null/bool/negative-literal.
    Raises NotImplementedError for column references or arithmetic (deferred to step 1d)."""
    import sqlglot.expressions as E
    if isinstance(node, (E.Literal, E.Null, E.Boolean, E.Neg)):
        return _litval(node)
    raise NotImplementedError("UPDATE supports literal assignment only "
                              "(column/expression RHS is step 1d)")

def _update_parquet(path, sets, pred_sql):
    """Apply SET assignments to matching rows of a parquet (hot buffer or canonical buffer),
    keeping column order. sets: list of (col, typed_value, val_sql). Returns rows updated."""
    import duckdb
    con = duckdb.connect()
    cond = "TRUE" if pred_sql is None else f"({pred_sql}) IS TRUE"
    n = con.execute(f"SELECT count(*) FROM read_parquet('{path}') WHERE {cond}").fetchone()[0]
    if n:
        repl = ", ".join(f'CASE WHEN {cond} THEN {vsql} ELSE "{col}" END AS "{col}"'
                         for col, _, vsql in sets)
        df = con.execute(f"SELECT * REPLACE ({repl}) FROM read_parquet('{path}')").fetch_df()
        df.to_parquet(path, index=False)
    return n

def update(catalog, sql):
    """UPDATE t SET col = <expr> [, ...] [WHERE ...]. <expr> is a literal, a column
    reference, or arithmetic over columns/literals (+ - * / %, unary minus, parens).
    Strategy matches storage class:
      buffered -> evaluate each RHS per-row against the segment's (override-aware) values and
                  write an override sidecar per cold segment for matching rows (no rewrite)
                  + update the hot buffer parquet (DuckDB evaluates the same expression SQL);
      segment  -> update the canonical buffer (DuckDB) and re-encode the single segment.
    Multi-column SET uses simultaneous semantics (all RHS evaluated against the pre-update
    row). Returns the number of rows updated."""
    import wdb_sql, wdb_override
    from wdb_engine import Segment
    tree = sqlglot.parse_one(sql, read='duckdb')
    name = tree.find(E.Table).name
    phys = catalog.phys_map(name)
    assigns = tree.args.get('expressions') or []
    if not assigns:
        return 0
    schema = {c[0]: c[1] for c in catalog.get_table(name)['schema']}
    sets = []   # (col, rhs_node, val_sql)
    for a in assigns:
        if not isinstance(a, E.EQ):
            raise NotImplementedError(f"unsupported SET clause: {a.sql()}")
        col = a.this.name
        if col not in schema:
            raise KeyError(f"no such column {col!r} in {name!r}")
        sets.append((col, a.expression, a.expression.sql(dialect='duckdb')))
    wnode = tree.args.get('where'); pred = wnode.this if wnode is not None else None
    # physical-name versions for the DuckDB/parquet path (parquet carries physical names)
    psets = [(phys.get(col, col), rhs,
              wdb_sql._to_physical(rhs, phys).sql(dialect='duckdb')) for col, rhs, _ in sets]
    pred_sql = (wdb_sql._to_physical(pred, phys).sql(dialect='duckdb')
                if pred is not None else None)
    mode = catalog.table_mode(name)
    updated = 0

    if mode == 'buffered':
        segs = catalog.get_table(name)['segments']
        for sf, sp in zip(segs, catalog.segment_paths(name)):
            seg = Segment(sp); register_synth(catalog, seg, name)
            if pred is None:
                idx = np.arange(seg.N)
            else:
                idx = np.nonzero(wdb_sql._eval_pred(seg, pred, lambda x: phys.get(x, x)))[0]
            if len(idx):
                # simultaneous semantics: evaluate every RHS against the pre-update segment
                # BEFORE writing any override (so SET a=b, b=a swaps correctly)
                newvals = [(phys.get(col, col), wdb_sql._eval_expr(seg, rhs, col_map=phys)[idx])
                           for col, rhs, _ in sets]
                for pcol, vals in newvals:
                    wdb_override.set_override(sp, pcol, idx, np.asarray(vals, dtype=object))
                updated += len(idx)
        hp = hot_path(catalog, name)
        if os.path.exists(hp):
            updated += _update_parquet(hp, psets, pred_sql)
    else:
        bpath = _buffer_path(catalog, name)
        if os.path.exists(bpath):
            updated += _update_parquet(bpath, psets, pred_sql)
            _encode_segment(catalog, name, bpath)
    return updated
