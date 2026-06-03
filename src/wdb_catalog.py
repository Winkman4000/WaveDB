"""WaveDB catalog: a database is a directory; catalog.json registers tables.

Minimal first slice: persistent registry of table name -> schema + segment files.
No SQL, no storage classes yet - those are later, tested increments.
"""
import os, json

CATALOG_NAME = 'catalog.json'

class Catalog:
    def __init__(self, dbdir, data):
        self.dbdir = dbdir
        self.data = data            # {'tables': {name: {'schema': [...], 'segments': [...]}}}

    # ---- lifecycle ----
    @classmethod
    def create(cls, dbdir):
        if os.path.exists(os.path.join(dbdir, CATALOG_NAME)):
            raise FileExistsError(f"database already exists at {dbdir}")
        os.makedirs(dbdir, exist_ok=True)
        cat = cls(dbdir, {'tables': {}})
        cat.save()
        return cat

    @classmethod
    def open(cls, dbdir):
        path = os.path.join(dbdir, CATALOG_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(f"no database at {dbdir}")
        with open(path) as f:
            return cls(dbdir, json.load(f))

    def save(self):
        tmp = os.path.join(self.dbdir, CATALOG_NAME + '.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, os.path.join(self.dbdir, CATALOG_NAME))  # atomic

    # ---- tables ----
    def add_table(self, name, schema):
        """schema = list of [col_name, type_str]. Registers an empty table."""
        if name in self.data['tables']:
            raise ValueError(f"table {name!r} already exists")
        self.data['tables'][name] = {'schema': schema, 'segments': [], 'mode': 'segment'}
        self.save()

    def drop_table(self, name):
        if name not in self.data['tables']:
            raise KeyError(f"no such table {name!r}")
        del self.data['tables'][name]
        self.save()

    # ---- schema evolution (ALTER): logical schema lives here; segments are immutable and
    # may lag. 'phys' maps a logical column to its (fixed) physical name in segments when they
    # differ (rename); 'defaults' holds the fill value for segments written before an ADD COLUMN.
    # Both absent == identity, so pre-ALTER tables and catalogs are unaffected (no migration).
    def column_names(self, name):
        return [c[0] for c in self.get_table(name)['schema']]

    def phys_map(self, name):
        """logical -> physical column name, only where they differ (identity otherwise)."""
        return dict(self.get_table(name).get('phys', {}))

    def column_default(self, name, col):
        return self.get_table(name).get('defaults', {}).get(col)

    def rename_table(self, old, new):
        t = self.data['tables']
        if old not in t: raise KeyError(f"no such table {old!r}")
        if new in t: raise ValueError(f"table {new!r} already exists")
        t[new] = t.pop(old); self.save()

    def rename_column(self, name, old, new):
        tab = self.get_table(name); sch = tab['schema']
        i = next((k for k, c in enumerate(sch) if c[0] == old), None)
        if i is None: raise KeyError(f"no such column {old!r} in {name!r}")
        if any(c[0] == new for c in sch): raise ValueError(f"column {new!r} already exists")
        phys = tab.setdefault('phys', {})
        underlying = phys.pop(old, old)              # physical name stays put across renames
        sch[i][0] = new
        if underlying != new: phys[new] = underlying
        self.save()

    def add_column(self, name, col, type_str, default=None):
        tab = self.get_table(name); sch = tab['schema']
        if any(c[0] == col for c in sch): raise ValueError(f"column {col!r} already exists")
        sch.append([col, type_str])
        tab.setdefault('defaults', {})[col] = default
        self.save()

    def drop_column(self, name, col):
        tab = self.get_table(name); sch = tab['schema']
        if not any(c[0] == col for c in sch): raise KeyError(f"no such column {col!r}")
        if len(sch) == 1: raise ValueError("cannot drop the last column of a table")
        tab['schema'] = [c for c in sch if c[0] != col]
        tab.get('phys', {}).pop(col, None)
        tab.get('defaults', {}).pop(col, None)
        self.save()

    def get_table(self, name):
        if name not in self.data['tables']:
            raise KeyError(f"no such table {name!r}")
        return self.data['tables'][name]

    def list_tables(self):
        return sorted(self.data['tables'].keys())

    def add_segment(self, name, segment_filename, labels=None):
        t = self.get_table(name)
        t['segments'].append(segment_filename)
        if labels is not None:
            t.setdefault('fd_labels', {})[segment_filename] = labels
        self.save()

    def segment_labels(self, name, segment_filename):
        return self.get_table(name).get('fd_labels', {}).get(segment_filename, [])

    def all_labels(self, name):
        return self.get_table(name).get('fd_labels', {})

    def set_table_mode(self, name, mode):
        if mode not in ('segment', 'buffered'):
            raise ValueError(f"unknown table mode {mode!r}")
        self.get_table(name)['mode'] = mode
        self.save()

    def table_mode(self, name):
        return self.get_table(name).get('mode', 'segment')

    def segment_paths(self, name):
        t = self.get_table(name)
        return [os.path.join(self.dbdir, s) for s in t['segments']]
