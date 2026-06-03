"""Broadened oracle: more query shapes (BETWEEN, !=, >=/<=, OR, NOT IN, multi-aggregate,
HAVING, deterministic ORDER BY/LIMIT) over multiple datasets -- including a SEQUENTIAL-KEY
table (mode-4's exact turf) and a wide/negative-int table. Every result is checked against
DuckDB. When mode-4 changes how integer columns decode, this battery is the front line that
proves query answers are unchanged.

Set queries are compared order-insensitively; ORDER BY queries are compared order-PRESERVING
on a unique key so there's no tie ambiguity."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
from helpers import roundtrip
import wdb_sql

def _norm(rows):
    out=[tuple(round(float(x),4) if isinstance(x,(int,float,np.floating,np.integer)) and not isinstance(x,bool)
               else x for x in r) for r in rows]
    return sorted(out, key=lambda t: tuple(str(x) for x in t))

def _seqkey_df(n=3000):
    rng=np.random.default_rng(101)
    return pd.DataFrame({
        'id':  1_000_000 + np.arange(n, dtype=np.int64),          # sequential PK (mode-4 turf)
        'grp': rng.choice(['alpha','beta','gamma','delta'], n),
        'val': rng.integers(1, 500, n).astype(np.int64),
    })

def _wideint_df(n=3000):
    rng=np.random.default_rng(202)
    return pd.DataFrame({
        'a': rng.integers(-10**14, 10**14, n).astype(np.int64),
        'b': rng.choice([-1,0,1,2,3], n).astype(np.int64),
    })

def _sales_df(n=3000):
    rng=np.random.default_rng(303)
    return pd.DataFrame({
        'region': rng.choice(['north','south','east','west'], n),
        'qty':    rng.integers(1,100,n).astype(np.int64),
        'price':  np.round(rng.uniform(1,1000,n),2),
    })

def _check_set(df, sql):
    seg, pq = roundtrip(df); con = duckdb.connect()
    duck = con.execute(sql.replace('TBL', f"'{pq}'")).fetchall()
    rows, _ = wdb_sql.execute(seg, sql.replace('TBL', 'tbl'))
    assert _norm(rows) == _norm(duck), f"mismatch\n SQL: {sql}\n wdb: {_norm(rows)[:6]}\n duck: {_norm(duck)[:6]}"

def _check_ordered(df, sql):
    """order-PRESERVING comparison (for ORDER BY ... LIMIT on a unique key)."""
    seg, pq = roundtrip(df); con = duckdb.connect()
    duck = [tuple(r) for r in con.execute(sql.replace('TBL', f"'{pq}'")).fetchall()]
    rows, _ = wdb_sql.execute(seg, sql.replace('TBL', 'tbl'))
    rows = [tuple(r) for r in rows]
    assert rows == duck, f"order mismatch\n SQL: {sql}\n wdb: {rows[:6]}\n duck: {duck[:6]}"

# ---- sequential-key table: the mode-4 front line ----
def test_seqkey_between():        _check_set(_seqkey_df(), "SELECT id FROM TBL WHERE id BETWEEN 1000100 AND 1000200")
def test_seqkey_point_eq():       _check_set(_seqkey_df(), "SELECT grp,val FROM TBL WHERE id = 1001234")
def test_seqkey_min_max():        _check_set(_seqkey_df(), "SELECT MIN(id), MAX(id), COUNT(*) FROM TBL")
def test_seqkey_groupby_sum():    _check_set(_seqkey_df(), "SELECT grp, SUM(val), COUNT(*) FROM TBL GROUP BY grp")
def test_seqkey_having():         _check_set(_seqkey_df(), "SELECT grp, COUNT(*) FROM TBL GROUP BY grp HAVING COUNT(*) > 500")
def test_seqkey_orderby_limit():  _check_ordered(_seqkey_df(), "SELECT id, val FROM TBL ORDER BY id DESC LIMIT 10")
def test_seqkey_orderby_asc():    _check_ordered(_seqkey_df(), "SELECT id FROM TBL WHERE val > 250 ORDER BY id ASC LIMIT 20")
def test_seqkey_sum_of_key():     _check_set(_seqkey_df(), "SELECT SUM(id) FROM TBL")   # large-sum reduction over the key

# ---- wide/negative ints ----
def test_wideint_filter():        _check_set(_wideint_df(), "SELECT a FROM TBL WHERE a < 0")
def test_wideint_groupby():       _check_set(_wideint_df(), "SELECT b, COUNT(*), MIN(a), MAX(a) FROM TBL GROUP BY b")
def test_wideint_between_neg():   _check_set(_wideint_df(), "SELECT a FROM TBL WHERE a BETWEEN -1000000 AND 1000000")
def test_wideint_lt_negative():   _check_set(_wideint_df(), "SELECT a FROM TBL WHERE a < -5000000")
def test_wideint_gte_negative():  _check_set(_wideint_df(), "SELECT a FROM TBL WHERE a >= -100 AND a <= 100")

# ---- broadened operators on sales ----
def test_sales_neq():             _check_set(_sales_df(), "SELECT qty FROM TBL WHERE region != 'north'")
def test_sales_gte_lte():         _check_set(_sales_df(), "SELECT qty FROM TBL WHERE qty >= 10 AND qty <= 90")
def test_sales_or():              _check_set(_sales_df(), "SELECT qty FROM TBL WHERE region='north' OR qty > 95")
def test_sales_not_in():          _check_set(_sales_df(), "SELECT qty FROM TBL WHERE region NOT IN ('north','south')")
def test_sales_multi_agg():       _check_set(_sales_df(), "SELECT region, COUNT(*), SUM(qty), MIN(qty), MAX(qty), AVG(qty) FROM TBL GROUP BY region")
