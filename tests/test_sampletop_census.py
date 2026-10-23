"""THE CENSUS ROAD (2026-10-23, Q17's board): GROUP BY a, b LIMIT k with no ORDER BY = any k real groups, counts exact.
The picker (the key with more values) is censused, k values with few rows are drawn, every row of them is read with the
partner at those rows only. Its switch is the road's predicted cost (frames touched x frame cost + rows x row cost, by
encoding) against the full count's. Driven through sampletop's detect + execute on test_groupself_run's fixture;
oracle: DuckDB, group by group (each row a real group, its count exact, no duplicates, k rows)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import sqlglot
import wdb_sampletop as ST
import test_groupself_run as F

PAIRS = (('u', 'v'), ('w', 'v'), ('tm', 'nk'), ('nk', 'u'), ('v', 'tm'), ('w', 'tm'))   # (id is mode 4: declined)


def _q(a, b, k=10):
    return 'SELECT %s, %s, COUNT(*) FROM t GROUP BY %s, %s LIMIT %d' % (a, b, a, b, k)


def _serve(q, full_ns=1e6):
    _db, _con, seg = F._fixture()
    spec = ST.detect(seg, sqlglot.parse_one(q, read='duckdb'), {})
    assert spec is not None and spec.get('road') == 'census', (q, spec and spec.get('road'))
    old = ST._FULL_NS[0], ST._FULL_WIDE_NS[0]
    ST._FULL_NS[0] = ST._FULL_WIDE_NS[0] = full_ns
    try:
        h0 = ST._CENSUS_HITS
        out = ST.execute(seg, spec)
        return spec, out, ST._CENSUS_HITS - h0
    finally:
        ST._FULL_NS[0], ST._FULL_WIDE_NS[0] = old


def _legal(a, b, rows, k):
    _db, con, _seg = F._fixture()
    w = pd.DataFrame({'a': [r[0] for r in rows], 'b': [r[1] for r in rows], 'c': [int(r[2]) for r in rows]})
    con.register('w', w)
    try:
        chk = con.execute('SELECT w.a, w.b, w.c, COUNT(h.rid) FROM w LEFT JOIN (SELECT %s, %s, 1 AS rid FROM tt) h '
                          'ON h.%s IS NOT DISTINCT FROM w.a AND h.%s IS NOT DISTINCT FROM w.b GROUP BY ALL'
                          % (a, b, a, b)).fetchall()
    finally:
        con.unregister('w')
    assert all(x[2] == x[3] for x in chk), (a, b, [x for x in chk if x[2] != x[3]][:3])
    assert len({(r[0], r[1]) for r in rows}) == len(rows), (a, b, 'duplicate groups')
    g = con.execute('SELECT COUNT(*) FROM (SELECT DISTINCT %s, %s FROM tt)' % (a, b)).fetchone()[0]
    assert len(rows) == min(k, g), (a, b, len(rows), g)


def test_any_keys_any_k_legal():
    for a, b in PAIRS:
        for k in (1, 10, 37):
            spec, out, hits = _serve(_q(a, b, k))
            assert out is not None and hits == 1, (a, b, k)
            rows = out[0]
            assert [h for h in out[1]] == [a, b, 'COUNT(*)'] or len(out[1]) == 3
            _legal(a, b, rows, k)
            vals = {spec['b']: int(_seg().cols[spec['b']]['V']), spec['a']: int(_seg().cols[spec['a']]['V'])}
            assert vals[spec['b']] >= vals[spec['a']]          # the picker is the key with more values


def _seg():
    return F._fixture()[2]


def test_fresh_sample_each_run():
    seen = set()
    for _ in range(4):
        _spec, out, _h = _serve(_q('u', 'v'))
        seen.add(frozenset((r[0], r[1]) for r in out[0]))
    assert len(seen) > 1                                    # a fresh lawful sample each run (Jackson's ruling)


def test_switch_declines_past_the_full_count():
    _spec, out, hits = _serve(_q('w', 'v'), full_ns=1e-6)    # a full count cheaper than any road
    assert out is None and hits == 0


def test_cost_model_frames_and_rows():
    seg = _seg()
    for col in ('u', 'v', 'w', 'tm', 'nk', 'id'):
        c = [ST._read_cost(seg, col, R) for R in (1, 10, 1000, 100000, 300000)]
        assert all(x <= y + 1e-12 for x, y in zip(c, c[1:])), (col, c)
    import types
    fake = types.SimpleNamespace(N=100_000_000, cols={'x': {'code_enc': 3, 'BR': 65536, 'cwidth': 1, 'bits': 8}})
    # 1,526 frames: 1,000 scattered rows touch 1526 (1 - e^(-1000/1526)) = 733 of them
    t1k = ST._read_cost(fake, 'x', 1000)
    assert abs(t1k - (733.4 * 75e-6 + 1000 * 2.3e-9)) < 2e-4, t1k
    tall, tmore = ST._read_cost(fake, 'x', 10_000_000), ST._read_cost(fake, 'x', 20_000_000)
    assert tmore - tall < 0.03                              # every frame touched: only the rows term grows
    fake.cols['x']['V'] = 100
    fake.cols['y'] = dict(fake.cols['x'], V=50_000_000)
    assert ST._road_pays(fake, 'x', 'x', 10, 0.0, 10) and not ST._road_pays(fake, 'x', 'x', 10 ** 8, 0.2, 10)
    # the full count's cost from the two keys' values: dense board (101 x 101 cells) vs past it (50M x 101)
    assert ST._full_cost(fake, 'x', 'x') < ST._full_cost(fake, 'y', 'x')
    assert ST._road_pays(fake, 'y', 'x', 10, 0.4, 10) and not ST._road_pays(fake, 'x', 'x', 10, 0.4, 10)


def test_switch_off_and_the_lane_untouched():
    _db, _con, seg = F._fixture()
    q = _q('u', 'v')
    old = ST._CENSUS[0]
    try:
        ST._CENSUS[0] = False
        assert ST.detect(seg, sqlglot.parse_one(q, read='duckdb'), {}) is None
    finally:
        ST._CENSUS[0] = old
    for bad in ('SELECT u, v, COUNT(*) FROM t GROUP BY u, v ORDER BY 3 DESC LIMIT 10',
                'SELECT u, v, COUNT(*) FROM t WHERE u > 3 GROUP BY u, v LIMIT 10',
                'SELECT u, u, COUNT(*) FROM t GROUP BY u, u LIMIT 10',
                'SELECT u, v, SUM(id) FROM t GROUP BY u, v LIMIT 10'):
        assert ST.detect(seg, sqlglot.parse_one(bad, read='duckdb'), {}) is None, bad
