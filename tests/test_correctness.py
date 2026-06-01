"""Oracle tests: same SQL through WaveDB and DuckDB must agree."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
from helpers import roundtrip
import wdb_sql
from wdb_engine import Segment

def _sample_df(n=2000):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        'region':   rng.choice(['north','south','east','west'], n),
        'category': rng.choice(['a','b','c'], n),
        'qty':      rng.integers(1, 100, n).astype(np.int64),
        'price':    np.round(rng.uniform(1, 1000, n), 2),
    })

def _norm(rows):
    # normalize to a sorted list of tuples of rounded floats / plain scalars
    out = []
    for r in rows:
        rr = tuple(round(float(x),4) if isinstance(x,(int,float,np.floating,np.integer)) else x for x in r)
        out.append(rr)
    return sorted(out, key=lambda t: tuple(str(x) for x in t))

def _check(sql_wdb, sql_duck=None):
    df = _sample_df()
    seg, pq = roundtrip(df)
    con = duckdb.connect()
    duck = con.execute((sql_duck or sql_wdb).replace('TBL', f"'{pq}'")).fetchall()
    rows, _hdr = wdb_sql.execute(seg, sql_wdb.replace('TBL', 'tbl'))
    assert _norm(rows) == _norm(duck), f"mismatch\n SQL: {sql_wdb}\n wdb:  {_norm(w)[:5]}\n duck: {_norm(duck)[:5]}"

def test_select_where_int():
    _check("SELECT qty FROM TBL WHERE qty > 50")

def test_groupby_count():
    _check("SELECT region, COUNT(*) FROM TBL GROUP BY region")

def test_groupby_sum():
    _check("SELECT category, SUM(qty) FROM TBL GROUP BY category")

def test_groupby_avg():
    _check("SELECT region, AVG(qty) FROM TBL GROUP BY region")

def test_where_and_groupby():
    _check("SELECT category, COUNT(*) FROM TBL WHERE qty > 30 GROUP BY category")

def test_min_max():
    _check("SELECT region, MIN(qty), MAX(qty) FROM TBL GROUP BY region")

def test_where_string_eq():
    _check("SELECT qty FROM TBL WHERE region = 'north'")

def test_where_in():
    _check("SELECT qty FROM TBL WHERE region IN ('north','south')")
