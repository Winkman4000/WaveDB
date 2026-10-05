"""THE FILTER FIRST (2026-10-05, TPC-H Q4): an EXISTS over the child road judges its child conditions only on the
lines of parents the outer's own window kept -- the rest of the child's rows are never decoded. Q4's shape (a date
window on the parent, a column-against-column on the child), with a literal child condition beside it, a window on
both ends, and a window that keeps nothing answer as DuckDB does."""
import sys, os, uuid, tempfile, shutil, subprocess, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd

TMP = tempfile.gettempdir()
WDB = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
FLOOR = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_QMEM_STRICT='1')


def test_exists_window_first_equals_duck():
    import duckdb
    import pyarrow as pa, pyarrow.parquet as pq
    from wdb_db import Database
    rng = np.random.default_rng(17)
    no = 80_000
    d0 = datetime.date(1992, 1, 1)
    odate = [d0 + datetime.timedelta(days=int(x)) for x in rng.integers(0, 2400, no)]
    orders = pa.table({'o_orderkey': np.arange(1, no + 1, dtype=np.int64) * 4,
                       'o_orderdate': pa.array(odate, pa.date32()),
                       'o_orderpriority': np.array(['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT', '5-LOW'])[rng.integers(0, 5, no)]})
    per = rng.integers(1, 8, no)
    lk = np.repeat(np.arange(1, no + 1, dtype=np.int64) * 4, per)
    nl = lk.size
    cd = rng.integers(0, 2500, nl)
    lineitem = pa.table({'l_orderkey': lk,
                         'l_commitdate': pa.array([d0 + datetime.timedelta(days=int(x)) for x in cd], pa.date32()),
                         'l_receiptdate': pa.array([d0 + datetime.timedelta(days=int(x)) for x in cd + rng.integers(-30, 31, nl)], pa.date32()),
                         'l_quantity': rng.integers(1, 51, nl)})
    ex = ("EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate%s)")
    sqls = [
        "SELECT o_orderpriority, COUNT(*) AS order_count FROM orders WHERE o_orderdate >= DATE '1993-07-01' AND "
        "o_orderdate < DATE '1993-10-01' AND " + ex % '' + " GROUP BY o_orderpriority ORDER BY o_orderpriority",
        "SELECT o_orderpriority, COUNT(*) FROM orders WHERE o_orderdate >= DATE '1994-01-01' AND " + ex % ' AND l_quantity > 40'
        + " GROUP BY o_orderpriority ORDER BY o_orderpriority",
        "SELECT COUNT(*) FROM orders WHERE o_orderdate < DATE '1992-03-01' AND " + ex % '',
        "SELECT COUNT(*) FROM orders WHERE o_orderdate > DATE '2010-01-01' AND " + ex % '',
    ]
    d = os.path.join(TMP, 'xw_' + uuid.uuid4().hex[:8]); db_dir = d + '_db'; os.makedirs(d)
    old = {k: os.environ.get(k) for k in FLOOR}
    try:
        con = duckdb.connect()
        for name, t in (('orders', orders), ('lineitem', lineitem)):
            p = os.path.join(d, name + '.parquet'); pq.write_table(t, p)
            subprocess.run([sys.executable, WDB, 'load', db_dir, name, p], check=True, capture_output=True,
                           env=dict(os.environ, **FLOOR))
            con.execute("CREATE VIEW %s AS SELECT * FROM read_parquet('%s')" % (name, p))
        os.environ.update(FLOOR)
        db = Database.open(db_dir)
        for sql in sqls:
            r = db.run(sql); got = [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)]
            want = [tuple(x) for x in con.execute(sql).fetchall()]
            assert [tuple(str(v) for v in g) for g in got] == [tuple(str(v) for v in w) for w in want], (sql, got, want)
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
