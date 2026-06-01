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
