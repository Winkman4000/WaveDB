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

    def get_table(self, name):
        if name not in self.data['tables']:
            raise KeyError(f"no such table {name!r}")
        return self.data['tables'][name]

    def list_tables(self):
        return sorted(self.data['tables'].keys())

    def add_segment(self, name, segment_filename):
        self.get_table(name)['segments'].append(segment_filename)
        self.save()

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
