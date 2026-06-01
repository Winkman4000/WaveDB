"""WaveDB Database: a directory-backed database you build with SQL.

    db = Database.create('/path/mydb')
    db.run("CREATE TABLE users (id INT, name VARCHAR)")
    db.run("INSERT INTO users VALUES (1,'alice')")
    rows, header = db.run("SELECT * FROM users WHERE id = 1")

Borrowed SQL syntax + sqlglot parser; catalog, storage, and execution are WaveDB's.
Step 3a: single segment per table.
"""
import sqlglot, sqlglot.expressions as E
from wdb_catalog import Catalog
from wdb_engine import Segment
import wdb_ddl, wdb_dml, wdb_sql

class Database:
    def __init__(self, catalog): self.cat = catalog
    @classmethod
    def create(cls, dbdir): return cls(Catalog.create(dbdir))
    @classmethod
    def open(cls, dbdir): return cls(Catalog.open(dbdir))

    def _table_in(self, tree):
        f = tree.find(E.From)
        if f is None: raise NotImplementedError("SELECT without FROM")
        return f.this.name

    def run(self, sql):
        tree = sqlglot.parse_one(sql, read='duckdb')
        if isinstance(tree, E.Create):
            return wdb_ddl.create_table(self.cat, sql)
        if isinstance(tree, E.Insert):
            return wdb_dml.insert(self.cat, sql)
        if isinstance(tree, E.Drop):
            self.cat.drop_table(tree.this.this.name); return None
        if isinstance(tree, E.Select):
            name = self._table_in(tree)
            paths = self.cat.segment_paths(name)
            if not paths:
                raise ValueError(f"table {name!r} has no data yet")
            if len(paths) > 1:
                raise NotImplementedError("multi-segment read (step 3c)")
            return wdb_sql.execute(Segment(paths[0]), sql)
        raise NotImplementedError(f"unsupported statement: {type(tree).__name__}")

    def tables(self): return self.cat.list_tables()
