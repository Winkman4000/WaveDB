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
import os
import wdb_ddl, wdb_dml, wdb_sql, wdb_merge, wdb_compact

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
        if isinstance(tree, E.Alter):
            return wdb_ddl.alter_table(self.cat, sql, self)
        if isinstance(tree, E.Insert):
            return wdb_dml.insert(self.cat, sql)
        if isinstance(tree, E.Drop):
            self.cat.drop_table(tree.this.this.name); return None
        if isinstance(tree, E.Delete):
            return wdb_dml.delete(self.cat, sql)
        if isinstance(tree, E.Update):
            return wdb_dml.update(self.cat, sql)
        if isinstance(tree, E.Select):
            name = self._table_in(tree)
            phys = self.cat.phys_map(name)
            cmap = {c: phys.get(c, c) for c in self.cat.column_names(name)}  # complete logical->physical
            paths = self.cat.segment_paths(name)
            segs = [Segment(p) for p in paths]
            hp = wdb_dml.hot_path(self.cat, name)
            hot = hp if os.path.exists(hp) else None
            if hot is None and len(segs) == 1:
                return wdb_sql.execute(segs[0], sql, col_map=cmap)
            if hot is None and not segs:
                raise ValueError(f"table {name!r} has no data yet")
            return wdb_merge.merge_query(segs, hot, sql, col_map=cmap)
        raise NotImplementedError(f"unsupported statement: {type(tree).__name__}")

    def set_table_mode(self, name, mode):
        """Operator control: 'buffered' = high-traffic, INSERT appends to hot buffer
        without re-encoding; SELECT merges hot+cold; call flush() to fold in."""
        self.cat.set_table_mode(name, mode)

    def flush(self, name):
        return wdb_dml.flush(self.cat, name)

    def compact(self, name, seg_files=None):
        """Merge cold segments into one, verifying FD labels on the union."""
        return wdb_compact.compact(self.cat, name, seg_files)

    def tables(self): return self.cat.list_tables()
