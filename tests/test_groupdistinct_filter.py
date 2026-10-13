"""GROUP BY key, COUNT(DISTINCT t) WHERE key <> '' through wdb_groupdistinct.

The read accepts the group-key filter at detect time, so EVERY execute lane
must drop the empty group: the shelf, the sparse-plane lane, the scatter lane
and the hash walk. The walk and scatter lanes once kept it (ClickBench review,
2026-10-12: Title <> '' returned the empty title with its distinct count).
Checked against DuckDB on the full answer, no LIMIT, so nothing hides."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, duckdb, sqlglot
import helpers
import wdb_groupdistinct as GD

SQL = "SELECT k, COUNT(DISTINCT u) AS c FROM t WHERE k <> '' GROUP BY k ORDER BY c DESC"


def _case(n, empty_share, nuser, null_share=0.0, seed=7):
    rng = np.random.default_rng(seed)
    k = np.array(['k%04d' % i for i in range(2000)], dtype=object)[rng.integers(0, 2000, n)]
    k[rng.random(n) < empty_share] = ''
    u = pd.array(rng.integers(0, nuser, n), dtype='Int64')
    if null_share:
        u[rng.random(n) < null_share] = pd.NA
    seg, pq = helpers.roundtrip(pd.DataFrame({'k': k, 'u': u}))
    tree = sqlglot.parse_one(SQL)
    det = GD.detect(seg, tree, None)
    assert det is not None and det[-1] is not None, 'the filtered shape must reach the read'
    got = GD.execute(seg, det, tree)
    assert got is not None
    want = duckdb.sql(SQL.replace('FROM t', "FROM '%s'" % pq)).fetchall()
    g = sorted((str(a), int(b)) for a, b in got[0])
    w = sorted((str(a), int(b)) for a, b in want)
    assert not any(a == '' for a, _ in g), 'empty group survived the filter'
    assert g == w, (g[:5], w[:5])


def test_walk_lane_drops_empty():
    _case(200_000, 0.3, 50_000)


def test_walk_lane_nullable_target_drops_empty():
    _case(200_000, 0.3, 50_000, null_share=0.1)


def test_scatter_lane_drops_empty():
    _case(4_200_000, 0.3, 200_000)


def test_no_empty_rows_present():
    _case(200_000, 0.0, 50_000)
