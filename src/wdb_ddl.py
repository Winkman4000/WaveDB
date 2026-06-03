"""WaveDB DDL: parse CREATE TABLE (borrowed SQL syntax via sqlglot) into the catalog.

We borrow SQL's language and sqlglot's parser; execution/storage is all WaveDB's.
Maps SQL column types -> WaveDB's internal type tags (int/float/string/datetime),
which line up with the encoder's dtype codes (0/2/1/3).
"""
import sqlglot, sqlglot.expressions as E
from sqlglot.expressions import DataType

def _wdb_type(dt):
    """Classify a sqlglot DataType into WaveDB's internal type tag."""
    k = dt.this
    if k in DataType.INTEGER_TYPES: return 'int'
    if k in DataType.REAL_TYPES:    return 'float'      # float/double/decimal
    if k in DataType.TEXT_TYPES:    return 'string'
    if k in DataType.TEMPORAL_TYPES:return 'datetime'
    if k == DataType.Type.BOOLEAN:  return 'int'        # bools stored as small ints
    raise NotImplementedError(f"unsupported column type: {dt.sql()}")

def parse_create_table(sql):
    """Return (table_name, schema) where schema = [[col, wdb_type], ...]."""
    tree = sqlglot.parse_one(sql, read='duckdb')
    if not isinstance(tree, E.Create) or (tree.args.get('kind') or '').upper() != 'TABLE':
        raise NotImplementedError("only CREATE TABLE is supported here")
    schema_node = tree.this
    if not isinstance(schema_node, E.Schema):
        raise ValueError("CREATE TABLE requires a column list, e.g. (id INT, name TEXT)")
    name = schema_node.this.name
    cols = []
    for cdef in schema_node.expressions:
        if not isinstance(cdef, E.ColumnDef):
            raise NotImplementedError(f"unsupported element in CREATE TABLE: {cdef.sql()}")
        cols.append([cdef.name, _wdb_type(cdef.args['kind'])])
    if not cols:
        raise ValueError("CREATE TABLE needs at least one column")
    return name, cols

def create_table(catalog, sql):
    """Parse a CREATE TABLE statement and register it in the catalog."""
    name, schema = parse_create_table(sql)
    catalog.add_table(name, schema)
    return name, schema


def alter_table(catalog, sql, db=None):
    """Execute an ALTER TABLE statement against the catalog (metadata-only; segments are immutable
    and reconciled at read via col_map / at compaction). Column-altering actions flush the hot
    buffer first, so the buffer always holds current logical names and only cold segments need
    name translation. Returns the (possibly new) table name."""
    tree = sqlglot.parse_one(sql, read='duckdb')
    if not isinstance(tree, E.Alter):
        raise NotImplementedError(f"not an ALTER statement: {type(tree).__name__}")
    name = tree.this.name
    actions = tree.args.get('actions') or []
    if db is not None:
        try: db.flush(name)            # fold any hot buffer into cold before names diverge
        except Exception: pass         # no hot buffer / nothing to flush -> fine
    import os, wdb_dml
    for act in actions:
        if isinstance(act, E.AlterRename):
            new = act.this.name
            # the canonical buffer file's name derives from the table name and is NOT tracked in
            # the catalog, so it must move with the table. Segment files keep their tracked
            # filenames (so presence/override sidecars are never orphaned); the hot buffer was
            # already folded in by the flush above.
            src = wdb_dml._buffer_path(catalog, name); dst = wdb_dml._buffer_path(catalog, new)
            if os.path.exists(src): os.rename(src, dst)
            catalog.rename_table(name, new); name = new
        elif isinstance(act, E.RenameColumn):
            catalog.rename_column(name, act.this.name, act.args['to'].name)
        elif isinstance(act, E.ColumnDef):
            import wdb_dml
            col = act.name; wt = _wdb_type(act.args['kind']); default = None
            for con in (act.args.get('constraints') or []):
                k = getattr(con, 'kind', None)
                if isinstance(k, E.DefaultColumnConstraint):
                    default = wdb_dml._litval(k.this)
            pcol = _fresh_physical(catalog, name, col)   # dodge stale bytes from a prior DROP
            catalog.add_column(name, col, wt, default)
            if pcol != col:
                tab = catalog.get_table(name); tab.setdefault('phys', {})[col] = pcol; catalog.save()
            _materialize_added_column(catalog, name, pcol, wt, default)
        elif isinstance(act, E.Drop):
            dcol = act.this.name
            pcol = catalog.phys_map(name).get(dcol, dcol)
            catalog.drop_column(name, dcol)          # refuses the last column; cleans phys/defaults
            _drop_column_from_buffer(catalog, name, pcol)
        else:
            raise NotImplementedError(f"unsupported ALTER action: {type(act).__name__}")
    return name


def _materialize_added_column(catalog, name, pcol, wt, default):
    """Segment-mode tables keep a canonical buffer that is re-encoded on every write; the cheapest
    correct thing is to fill the new column there (with its default) and re-encode now, so the
    single segment carries it. Buffered tables have no canonical buffer -- their cold segments
    synthesize the default at read instead (and the next flush writes the column for real)."""
    import os, wdb_dml
    import pandas as pd
    bpath = wdb_dml._buffer_path(catalog, name)
    if not os.path.exists(bpath):
        return
    df = pd.read_parquet(bpath); n = len(df)
    if wt == 'datetime':
        df[pcol] = pd.to_datetime(pd.Series([default] * n), errors='coerce')
    else:
        df[pcol] = pd.array([default] * n, dtype=wdb_dml._PD[wt])
    df.to_parquet(bpath, index=False)
    wdb_dml._encode_segment(catalog, name, bpath)


def _fresh_physical(catalog, name, col):
    """A physical (storage) name for a new column that does not collide with any column still
    present in a segment -- so re-adding a previously dropped name reads its DEFAULT (synth),
    not the dropped column's stale bytes."""
    from wdb_engine import Segment
    used = set(catalog.phys_map(name).values())
    for sp in catalog.segment_paths(name):
        try: used |= set(Segment(sp).cols.keys())
        except Exception: pass
    if col not in used: return col
    k = 1
    while f"{col}__v{k}" in used: k += 1
    return f"{col}__v{k}"


def _drop_column_from_buffer(catalog, name, pcol):
    """Segment-mode tables keep a canonical buffer re-encoded on every write; drop the column there
    and re-encode so the single segment sheds it immediately. Buffered tables have no canonical
    buffer -- the dead bytes in their cold segments are reclaimed at the next compaction."""
    import os, wdb_dml
    import pandas as pd
    bpath = wdb_dml._buffer_path(catalog, name)
    if not os.path.exists(bpath): return
    df = pd.read_parquet(bpath)
    if pcol in df.columns:
        df = df.drop(columns=[pcol]); df.to_parquet(bpath, index=False)
        wdb_dml._encode_segment(catalog, name, bpath)
