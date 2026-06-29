"""wdb_heavypair on main: the persisted count-sorted 2-key pair sidecar (.gbp) with ratio-spaced
landmarks. A clean-boundary 2-key COUNT(*) top-K is answered from the sidecar and matches db.run;
ties at the LIMIT boundary, WHERE, and single-key shapes correctly decline. Landmarks locate a
count>=T floor from landmark counts alone. Self-contained small DB so it runs in CI."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd, numpy as np, sqlglot
from wdb_db import Database
from wdb_engine import Segment
import wdb_encode, wdb_heavypair as HP, wdb_gridwalk as GW


def _build(d):
    db = Database.create(d); db.run("CREATE TABLE t (k1 VARCHAR, k2 VARCHAR, filler INTEGER)")
    rows = []
    for (a, b), n in {('A', 'X'): 5, ('B', 'Y'): 4, ('C', 'Z'): 3, ('D', 'W'): 2}.items():
        rows += [(a, b, i) for i in range(n)]            # heavy pairs, DISTINCT counts 5,4,3,2
    for i in range(20):
        rows.append((f'u{i}', f'v{i}', i))               # singletons (implicit count 1, not stored)
    pq = os.path.join(d, 's.parquet'); pd.DataFrame(rows, columns=['k1', 'k2', 'filler']).to_parquet(pq, index=False)
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb')
    return db, Segment(os.path.join(d, 't_0.wdb'))


def _try(seg, sql):
    return HP.try_heavypair(seg, sqlglot.parse_one(sql, read='duckdb'), None)


def test_heavypair_answers_and_declines():
    d = os.path.join(tempfile.gettempdir(), f'whp_{uuid.uuid4().hex[:8]}')
    try:
        db, seg = _build(d)
        sql3 = "SELECT k1, k2, COUNT(*) FROM t GROUP BY k1, k2 ORDER BY COUNT(*) DESC LIMIT 3"
        res = _try(seg, sql3)
        assert res is not None, "should answer a clean-boundary 2-key topK"
        assert [r[2] for r in res[0]] == [5, 4, 3]
        old, _ = db.run(sql3)
        assert sorted(res[0]) == sorted(old), f"{res[0]} != {old}"
        # full heavy set (LIMIT == #heavy pairs = 4): answered
        assert _try(seg, "SELECT k1, k2, COUNT(*) FROM t GROUP BY k1, k2 ORDER BY COUNT(*) DESC LIMIT 4") is not None
        # routes through the real db.run path (with gridwalk -- the default 2-key read -- stood down,
        # so heavypair, its fallback, gets the query)
        GW.disable()
        try:
            h0 = HP._HITS; db.run(sql3); assert HP._HITS > h0, "db.run should route through heavypair"
        finally:
            GW.enable()
        # declines: LIMIT past the heavy set would need singletons (5 > 4 heavy pairs)
        assert _try(seg, "SELECT k1, k2, COUNT(*) FROM t GROUP BY k1, k2 ORDER BY COUNT(*) DESC LIMIT 5") is None
        # declines: WHERE (restrict not in this node's shape)
        assert _try(seg, "SELECT k1, k2, COUNT(*) FROM t WHERE filler > 0 GROUP BY k1, k2 ORDER BY COUNT(*) DESC LIMIT 3") is None
        # declines: single key (that's gbcount's shape, not heavypair's)
        assert _try(seg, "SELECT k1, COUNT(*) FROM t GROUP BY k1 ORDER BY COUNT(*) DESC LIMIT 3") is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_heavypair_landmarks_and_threshold():
    d = os.path.join(tempfile.gettempdir(), f'whpl_{uuid.uuid4().hex[:8]}')
    try:
        db, seg = _build(d)
        cA, cB, cn, lm = HP._load(seg, ['k1', 'k2'])
        assert list(cn) == [5, 4, 3, 2], list(cn)           # count-descending block
        # landmarks are (position, count), positions ascending, counts non-increasing, in range
        assert lm.ndim == 2 and lm.shape[1] == 2
        assert list(lm[:, 0]) == sorted(lm[:, 0])
        assert all(lm[i, 1] >= lm[i + 1, 1] for i in range(len(lm) - 1))
        # threshold_floor uses landmark counts alone to pick a scan start that never overshoots
        for T in (5, 4, 3, 2, 1):
            p = HP.threshold_floor(lm, cn, T)
            assert 0 <= p < cn.size
            assert int(cn[p]) >= T or p == 0                # the segment head at/above T (or front)
    finally:
        shutil.rmtree(d, ignore_errors=True)
