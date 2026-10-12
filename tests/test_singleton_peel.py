"""THE SINGLETON PEEL (wdb_peel, through fused_agg's peel door): when one GROUP BY key is nearly unique,
nearly every group is one row -- census that key over the survivors, group only the repeaters, and pick
the k winners over repeater groups + singleton rows (2026-10-12).

The fixture's u is unique except every 997th row, which repeats its neighbour's u; k2 sometimes agrees
across such a pair (a real two-row group) and sometimes not (two one-row groups whose u repeats -- the
case that proves the peel groups repeaters on the WHOLE key, not on u alone). Answers are checked against
DuckDB: exactly where the ORDER BY is total, and as SQL defines it where ties reach the cut (k real
groups whose order values equal DuckDB's)."""
import sys, os, tempfile, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, wdb_peel
from wdb_db import Database

N = 400_000
_DB = None; _CON = None
def _fixture():
    global _DB, _CON
    if _DB is not None: return _DB, _CON
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'peel_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    _CON.execute("""CREATE TABLE t AS SELECT i AS id,
        CASE WHEN i %% 997 = 0 AND i > 0 THEN (i - 1) * 3 ELSE i * 3 END AS u,
        (i // 2) %% 5 AS k2,
        (i * 31) %% 1000 AS x,
        ((i * 17) %% 1000) / 8.0 AS f,
        'w' || CAST(CASE WHEN i %% 991 = 0 AND i > 0 THEN i - 1 ELSE i END AS VARCHAR) AS s
        FROM range(%d) t(i)""" % N)
    _DB = Database.create(d)
    pq = os.path.join(d, 't.parquet')
    _CON.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    _DB.cat.add_table('t', [['id', 'int'], ['u', 'int'], ['k2', 'int'], ['x', 'int'], ['f', 'float'], ['s', 'str']])
    # born as a real database is: the suite's toy switches would admit a repeating sequence (mode 4,
    # codes = row positions), which no database is born with
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None: os.environ[k] = v
    _DB.cat.add_segment('t', 't_0.wdb')
    return _DB, _CON


def _n(v):
    return round(float(v), 6) if isinstance(v, float) else v


def _run(q, peeled=True):
    db, con = _fixture()
    h0 = wdb_peel._HITS
    got = [tuple(_n(c) for c in r) for r in db.run(q)[0]]
    if peeled is None:
        pass                                            # an earlier read may serve it: the answer is the test
    elif peeled:
        assert wdb_peel._HITS == h0 + 1, 'the peel was not taken: ' + q
    else:
        assert wdb_peel._HITS == h0, 'the peel answered a shape it should decline: ' + q
    return got, con


def _exact(q, peeled=True):
    got, con = _run(q, peeled)
    want = [tuple(_n(c) for c in r) for r in con.execute(q).fetchall()]
    assert got == want, (q, got, want)


def _tied(q, order_cols, k, keys, peeled=True):
    """ties at the cut: k distinct real groups, with DuckDB's order values"""
    got, con = _run(q, peeled)
    full = q.split(' ORDER BY ')[0].split(' LIMIT ')[0]
    allrows = {tuple(r[i] for i in keys): tuple(_n(c) for c in r) for r in con.execute(full).fetchall()}
    assert len(got) == k, (q, got)
    assert len({tuple(r[i] for i in keys) for r in got}) == k, ('duplicate groups', q, got)
    for r in got:
        assert allrows.get(tuple(r[i] for i in keys)) == r, ('not a real group row', q, r)
    if order_cols is not None:
        want = [tuple(_n(r[i]) for i in order_cols) for r in con.execute(q).fetchall()]
        assert [tuple(r[i] for i in order_cols) for r in got] == want, (q, got, want)


def test_count_desc_then_key_is_exact():
    _exact("SELECT u, k2, COUNT(*) AS c, SUM(x), AVG(f) FROM t WHERE x <> 5 GROUP BY u, k2 ORDER BY c DESC, u LIMIT 10")

def test_count_asc_ties_any_singletons():
    _tied("SELECT u, k2, COUNT(*) AS c, SUM(x) FROM t WHERE x <> 5 GROUP BY u, k2 ORDER BY c ASC LIMIT 10", [2], 10, (0, 1))

def test_sum_desc_one_key():
    _exact("SELECT u, SUM(x) AS sx FROM t GROUP BY u ORDER BY sx DESC, u LIMIT 7")

def test_min_max_by_ordinal():
    _exact("SELECT u, k2, MIN(f), MAX(x) FROM t GROUP BY u, k2 ORDER BY 3 DESC, 1 LIMIT 5")

def test_avg_ascending_with_key_tiebreak():
    _exact("SELECT u, AVG(x) AS a, COUNT(*) FROM t WHERE k2 <> 3 GROUP BY u ORDER BY a, u DESC LIMIT 9")

def test_order_by_an_unprojected_count():
    _exact("SELECT u, SUM(x) FROM t GROUP BY u ORDER BY COUNT(*) DESC, u LIMIT 6")

def test_order_by_the_aggregate_expression():
    _exact("SELECT u, k2, SUM(x) FROM t GROUP BY u, k2 ORDER BY SUM(x) DESC, u, k2 LIMIT 8")

def test_string_key_ordered_by_itself():
    """a dictionary-coded string key peels (its codes are value ranks); an inline one (mode 5, which a
    near-unique string usually wins) has no codes to census and declines -- right either way"""
    import wdb_join
    db, _ = _fixture()
    seg, _p = wdb_join._solo_segment(db, 't')
    dict_coded = seg.cols['s'].get('mode') in (0, 1, 2)
    _exact("SELECT s, COUNT(*) FROM t GROUP BY s ORDER BY s DESC LIMIT 5", peeled=dict_coded)
    _exact("SELECT s, COUNT(*) AS c FROM t GROUP BY s ORDER BY c DESC, s LIMIT 5", peeled=dict_coded)

def test_selective_where():
    _exact("SELECT u, COUNT(*) AS c, SUM(x) FROM t WHERE x < 3 GROUP BY u ORDER BY c DESC, u LIMIT 12")

def test_where_keeps_nothing():
    """(the cascade proves the WHERE empty before any door)"""
    _exact("SELECT u, COUNT(*) AS c FROM t WHERE x > 5000 GROUP BY u ORDER BY c DESC LIMIT 3", peeled=None)

def test_limit_without_order():
    """(the narrow reads before fused_agg -- gridwalk, wherescan -- answer this shape first today)"""
    _tied("SELECT u, k2, COUNT(*), SUM(x) FROM t WHERE x <> 5 GROUP BY u, k2 LIMIT 10", None, 10, (0, 1), peeled=None)

def test_limit_past_the_group_count():
    _exact("SELECT u, COUNT(*) AS c FROM t WHERE x = 7 GROUP BY u ORDER BY u LIMIT 100000")

def test_no_near_unique_key_declines():
    _exact("SELECT k2, COUNT(*) AS c FROM t GROUP BY k2 ORDER BY c DESC, k2 LIMIT 3", peeled=False)

def test_having_declines():
    _exact("SELECT u, COUNT(*) AS c FROM t GROUP BY u HAVING COUNT(*) > 1 ORDER BY u LIMIT 3", peeled=False)

def test_peel_key_not_first():
    _exact("SELECT k2, u, COUNT(*) AS c, MAX(f) FROM t WHERE k2 <> 1 GROUP BY k2, u ORDER BY c DESC, u, k2 LIMIT 11")

def test_the_pick_without_an_order_directly():
    """no ORDER BY reaches the peel only behind the narrow reads today; call it: k real groups"""
    import numpy as np, wdb_join
    db, con = _fixture()
    seg, _p = wdb_join._solo_segment(db, 't')
    keys = [{'seg': seg, 'pcol': 'u'}, {'seg': seg, 'pcol': 'k2'}]
    out = wdb_peel.run(np.arange(seg.N, dtype=np.int64), keys, 0, [('key', 0), ('key', 1), ('count',)], [], 25)
    full = {(r[0], r[1]): r[2] for r in con.execute("SELECT u, k2, COUNT(*) FROM t GROUP BY u, k2").fetchall()}
    assert len(out) == 25 and len({(r[0], r[1]) for r in out}) == 25
    assert all(full[(r[0], r[1])] == r[2] for r in out), out

def test_the_ablation_switch():
    """WDB_PEEL=0 hands the same query to the fused kernel: same answer, peel untouched"""
    old = os.environ.get('WDB_PEEL')
    os.environ['WDB_PEEL'] = '0'
    try:
        _exact("SELECT u, k2, COUNT(*) AS c, SUM(x) FROM t WHERE x <> 5 GROUP BY u, k2 ORDER BY c DESC, u LIMIT 10", peeled=False)
    finally:
        if old is None: os.environ.pop('WDB_PEEL', None)
        else: os.environ['WDB_PEEL'] = old
