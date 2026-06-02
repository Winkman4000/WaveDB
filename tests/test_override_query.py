"""Override query-correctness (mutable layer, step 1b): the effective-code-space makes
GROUP BY / WHERE / aggregates reflect overridden values -- including grouping by and
filtering on a value the segment's dictionary never had. Verified vs DuckDB oracle on the
post-override data. No-override columns must be unaffected."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_override, wdb_sql
from wdb_engine import Segment

TMP = tempfile.gettempdir()
def _seg(df):
    base = os.path.join(TMP, f'obq_{uuid.uuid4().hex[:8]}')
    df.to_parquet(base + '.parquet', index=False)
    sp = base + '.wdb'; wdb_encode.encode(base + '.parquet', sp)
    return sp
def _norm(rows):
    return sorted([tuple(round(float(x),4) if isinstance(x,(int,float)) and not isinstance(x,bool) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))
def _oracle(rows, cols, sql):
    con = duckdb.connect()
    def lit(v): return "NULL" if v is None else ("'"+v.replace("'","''")+"'" if isinstance(v,str) else str(v))
    vals = ",".join("("+",".join(lit(v) for v in r)+")" for r in rows)
    return _norm(con.execute(sql.replace("FROM t", f"FROM (VALUES {vals}) AS t({','.join(cols)})")).fetchall())
def _run(sp, sql): return _norm(wdb_sql.execute(Segment(sp), sql)[0])

def test_groupby_and_where_reflect_overrides():
    sp = _seg(pd.DataFrame({'g':['a','b','a','c','b','a'], 'x':[10,20,30,40,50,60]}))
    wdb_override.set_override(sp, 'g', [5], np.array(['NEW'.encode()], dtype=object))  # new group
    wdb_override.set_override(sp, 'x', [1], np.array([999], dtype=np.int64))            # new value
    data = [('a',10),('b',999),('a',30),('c',40),('b',50),('NEW',60)]
    for sql in ["SELECT g,COUNT(*) FROM t GROUP BY g","SELECT g,SUM(x) FROM t GROUP BY g",
                "SELECT g,x FROM t WHERE x=999","SELECT g,x FROM t WHERE g='NEW'",
                "SELECT SUM(x) FROM t","SELECT g,AVG(x) FROM t GROUP BY g",
                "SELECT g,x FROM t WHERE x>35"]:
        assert _run(sp, sql) == _oracle(data, ['g','x'], sql), (sql, _run(sp,sql), _oracle(data,['g','x'],sql))

def test_override_to_existing_value_merges_group():
    # override row to a value already present -> it joins that existing group
    sp = _seg(pd.DataFrame({'g':['a','b','c'], 'x':[1,2,3]}))
    wdb_override.set_override(sp, 'g', [2], np.array(['a'.encode()], dtype=object))   # c -> a
    data = [('a',1),('b',2),('a',3)]
    assert _run(sp, "SELECT g,COUNT(*) FROM t GROUP BY g") == _oracle(data,['g','x'],"SELECT g,COUNT(*) FROM t GROUP BY g")
    assert _run(sp, "SELECT g,SUM(x) FROM t GROUP BY g") == _oracle(data,['g','x'],"SELECT g,SUM(x) FROM t GROUP BY g")

def test_float_override_in_queries():
    sp = _seg(pd.DataFrame({'k':['a','a','b'], 'v':[1.5,2.5,3.5]}))
    wdb_override.set_override(sp, 'v', [0], np.array([100.25], dtype=np.float64))
    data = [('a',100.25),('a',2.5),('b',3.5)]
    assert _run(sp,"SELECT k,SUM(v) FROM t GROUP BY k") == _oracle(data,['k','v'],"SELECT k,SUM(v) FROM t GROUP BY k")
    assert _run(sp,"SELECT v FROM t WHERE v>50") == _oracle(data,['k','v'],"SELECT v FROM t WHERE v>50")

def test_no_override_column_unaffected():
    sp = _seg(pd.DataFrame({'g':['a','b','a'], 'x':[1,2,3]}))
    wdb_override.set_override(sp, 'x', [0], np.array([9], dtype=np.int64))   # only x overridden
    # g queries must be unchanged
    assert _run(sp, "SELECT g,COUNT(*) FROM t GROUP BY g") == _norm([('a',2),('b',1)])

def test_override_with_nulls_present():
    sp = _seg(pd.DataFrame({'g':['a','b','c'], 'x':pd.array([10,None,30],dtype='Int64')}))
    wdb_override.set_override(sp, 'x', [0], np.array([777], dtype=np.int64))
    data = [('a',777),('b',None),('c',30)]
    assert _run(sp,"SELECT SUM(x) FROM t") == _oracle(data,['g','x'],"SELECT SUM(x) FROM t")     # 807
    assert _run(sp,"SELECT g,x FROM t WHERE x>100") == _oracle(data,['g','x'],"SELECT g,x FROM t WHERE x>100")
