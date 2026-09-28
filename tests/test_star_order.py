"""SELECT * ANSWERS IN THE TABLE'S DECLARED COLUMN ORDER (2026-09-28). A segment keeps its columns in
the order the load's jobs finished; the table's declared order is the catalog schema, written in the
input's order at load (wdb_encode.input_column_order). Q23 (SELECT * ... ORDER BY EventTime LIMIT 10)
had every value right and the columns in the load's order. Every * shape here is compared POSITIONALLY
with DuckDB reading the same parquet, whose column order is the input's."""
import sys, os, uuid, tempfile, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode
from wdb_db import Database
from wdb_catalog import Catalog
from wdb_engine import Segment

TMP = tempfile.gettempdir()


def _frame(n=50_000, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'id': np.arange(n, dtype=np.int64),
        'ev': rng.integers(0, 5_000, n).astype(np.int64),
        'url': np.array(['http://s%d.example/p%d' % (i % 97, i % 1013) for i in range(n)], dtype=object),
        'flag': rng.integers(0, 2, n).astype(np.int64),
        'title': np.array(['t%d' % (i % 331) for i in range(n)], dtype=object),
        'score': rng.random(n).round(3),
        'region': rng.integers(0, 40, n).astype(np.int64),
        'phrase': np.array(['' if i % 3 else 'q%d' % (i % 57) for i in range(n)], dtype=object)})


def _load(df, declared=None):
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'star_{t}'); pq = os.path.join(TMP, f'star_{t}.parquet')
    df.to_parquet(pq, index=False)
    os.makedirs(d); Catalog.create(d)
    out = os.path.join(d, 'hits_0.wdb')
    wdb_encode.encode(pq, out, stream=True)
    seg = Segment(out)
    order = declared or wdb_encode.input_column_order(pq, seg.order)
    tn = {0: 'int', 1: 'str', 2: 'float'}
    cat = Catalog.open(d)
    cat.data['tables']['hits'] = {'schema': [[c, tn.get(seg.cols[c].get('dt'), 'str')] for c in order],
                                  'segments': ['hits_0.wdb'], 'mode': 'segment'}
    cat.save()
    return d, pq, seg


def _norm(rows):
    return [tuple(x.decode() if isinstance(x, bytes) else
                  (round(float(x), 6) if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) else x)
                  for x in r) for r in rows]


STAR = ["SELECT * FROM hits WHERE url LIKE '%s7.example%' ORDER BY ev, id LIMIT 10",
        "SELECT * FROM hits ORDER BY ev DESC, id LIMIT 7",
        "SELECT * FROM hits WHERE region = 5 ORDER BY id LIMIT 12",
        "SELECT * FROM hits WHERE phrase <> '' ORDER BY ev, id LIMIT 9",
        "SELECT * FROM hits WHERE id = 4242"]


def test_input_column_order():
    df = _frame(n=100)
    t = uuid.uuid4().hex[:8]
    pq = os.path.join(TMP, f'ico_{t}.parquet'); csv = os.path.join(TMP, f'ico_{t}.csv')
    df.to_parquet(pq, index=False); df.to_csv(csv, index=False)
    try:
        shuffled = list(reversed(df.columns))
        assert wdb_encode.input_column_order(pq, shuffled) == list(df.columns)
        assert wdb_encode.input_column_order(csv, shuffled) == list(df.columns)
        assert wdb_encode.input_column_order(pq, shuffled[:-1]) == shuffled[:-1]      # not exactly these: unchanged
        assert wdb_encode.input_column_order('/no/such/file', shuffled) == shuffled
    finally:
        os.remove(pq); os.remove(csv)


def test_star_answers_in_the_declared_order():
    """the input's order (the load's default), and a deliberately different declared order: every *
    answer's names and every row equal DuckDB's, position by position"""
    df = _frame()
    for declared in (None, ['score', 'phrase', 'id', 'region', 'flag', 'url', 'title', 'ev']):
        d, pq, seg = _load(df, declared)
        try:
            db = Database.open(d)
            want_names = declared or list(df.columns)
            assert db.cat.column_names('hits') == want_names
            con = duckdb.connect()
            for sql in STAR:
                rows, names = db.run(sql)
                assert list(names) == want_names, (sql, list(names)[:4])
                q = sql.replace('SELECT *', 'SELECT ' + ', '.join(want_names)).replace('FROM hits', f"FROM '{pq}'")
                assert _norm(rows) == _norm(con.execute(q).fetchall()), sql
        finally:
            shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_wdb_load_writes_the_input_order():
    """the load command itself (bin/wdb load, the kit's path): the catalog schema in the parquet's
    order, and SELECT * answers in it"""
    import subprocess
    df = _frame(n=20_000, seed=9)
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'starload_{t}'); pq = os.path.join(TMP, f'starload_{t}.parquet')
    df.to_parquet(pq, index=False)
    root = os.path.join(os.path.dirname(__file__), '..')
    env = dict(os.environ, PYTHONPATH=os.path.join(root, 'src'))
    try:
        p = subprocess.run([sys.executable, os.path.join(root, 'bin', 'wdb'), 'load', d, 'hits', pq],
                           capture_output=True, text=True, env=env, timeout=300)
        assert p.returncode == 0, p.stderr[-400:]
        cat = json.load(open(os.path.join(d, 'catalog.json')))
        assert [c for c, t in cat['tables']['hits']['schema']] == list(df.columns)
        rows, names = Database.open(d).run("SELECT * FROM hits ORDER BY id LIMIT 3")
        assert list(names) == list(df.columns)
        want = duckdb.connect().execute(f"SELECT * FROM '{pq}' ORDER BY id LIMIT 3").fetchall()
        assert _norm(rows) == _norm(want)
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)
