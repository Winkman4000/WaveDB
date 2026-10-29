"""THE ONE PASS (2026-10-29, Jackson's fix 3): per-thread hash tables, merged on every core. Exact against numpy's
group-by on skewed and spread keys, with and without a filter, every thread count, tables too small (the redo) and too
large to allow (None); and legal against DuckDB through SQL on both sides of the line."""
import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, duckdb
import wdb_pairfold
from wdb_db import Database
from test_pairfold import _db30, _legal5


def _truth(pos, ac, bc, xc, yc, dv1, dv2, AB):
    sel = np.arange(bc.size) if pos is None else pos
    key = (bc[sel].astype(np.int64) << AB) | ac[sel].astype(np.int64)
    u, inv = np.unique(key, return_inverse=True)
    return {int(k): (int(c), float(a), float(b)) for k, c, a, b in zip(
        u, np.bincount(inv), np.bincount(inv, weights=dv1[xc[sel]]), np.bincount(inv, weights=dv2[yc[sel]]))}


def _check(r, want, W):
    fK, ucnt, us1, us2 = r
    assert fK.size == len(want), (fK.size, len(want))
    assert len(set(fK.tolist())) == fK.size, 'a pair written twice'
    for kk, c, a, b in zip(fK.tolist(), ucnt.tolist(), us1.tolist(), us2.tolist()):
        wc, wa, wb = want[kk]
        assert c == wc
        if W > 1:
            assert a == wa and b == wb


def test_one_pass_exact_against_numpy():
    rng = np.random.default_rng(29)
    old = (wdb_pairfold.HT_LINE, wdb_pairfold.HT_SAMPLE, wdb_pairfold.HT_CBMAX, wdb_pairfold.HT_HEAVY)
    try:
        wdb_pairfold.HT_LINE = 1e18
        wdb_pairfold.HT_HEAVY = 2.0
        for n, VA, VB, T9, samp, filt, skew, W in (
                (200_000, 50, 300, 16, 1 << 18, False, True, 3),
                (200_000, 1000, 1000, 3, 64, True, False, 3),       # the estimate far short: the redo
                (50_000, 7, 9, 1, 1 << 18, False, False, 1),
                (300_000, 4000, 4000, 16, 64, False, False, 3),     # ~300K pairs, two redos
                (300_000, 255, 60_000, 5, 1 << 10, True, True, 1),
                (1_000, 3, 3, 16, 1 << 18, False, False, 3)):        # threads with a few rows each
            wdb_pairfold.HT_SAMPLE = samp
            if skew:
                ac = np.minimum(rng.zipf(1.3, n) - 1, VA - 1).astype(np.uint16)
                bc = np.minimum(rng.zipf(1.2, n) - 1, VB - 1).astype(np.uint32)
            else:
                ac = rng.integers(0, VA, n).astype(np.uint16)
                bc = rng.integers(0, VB, n).astype(np.uint32)
            xc = rng.integers(0, 40, n).astype(np.uint8)
            yc = rng.integers(0, 40, n).astype(np.uint8)
            dv1 = rng.integers(-1000, 1000, 40).astype(np.float64)
            dv2 = rng.integers(0, 5000, 40).astype(np.float64)
            pos = np.flatnonzero(rng.random(n) < 0.6).astype(np.int64) if filt else None
            AB = max(1, int(VA - 1).bit_length())
            s0 = wdb_pairfold._HT_SPILL[0]
            r = wdb_pairfold._onepass(pos, ac, bc, xc, yc, dv1, dv2, AB, W, T9)
            assert r is not None
            _check(r, _truth(pos, ac, bc, xc, yc, dv1, dv2, AB), W)
            if samp == 64 and n >= 200_000:
                assert wdb_pairfold._HT_SPILL[0] > s0, 'the short estimate must redo the pass'
        # past the largest table allowed: None (the buckets take it)
        wdb_pairfold.HT_SAMPLE = 64
        wdb_pairfold.HT_CBMAX = 12
        ac = rng.integers(0, 4000, 300_000).astype(np.uint16)
        bc = rng.integers(0, 4000, 300_000).astype(np.uint32)
        z = np.zeros(300_000, np.uint8)
        assert wdb_pairfold._onepass(None, ac, bc, z, z, np.zeros(1), np.zeros(1), 12, 1, 4) is None
        # under the line nothing runs
        wdb_pairfold.HT_LINE = 0.0
        assert wdb_pairfold._onepass(None, ac[:10], bc[:10], z[:10], z[:10], np.zeros(1), np.zeros(1), 12, 1, 4) is None
    finally:
        wdb_pairfold.HT_LINE, wdb_pairfold.HT_SAMPLE, wdb_pairfold.HT_CBMAX, wdb_pairfold.HT_HEAVY = old


def test_one_pass_legal_against_duck():
    d, pq, seg = _db30()
    try:
        db = Database.open(d)
        con = duckdb.connect()
        old = wdb_pairfold.HT_LINE, wdb_pairfold.HT_HEAVY
        wdb_pairfold.HT_HEAVY = 2.0
        ran = 0
        for where in (" WHERE p <> ''", ''):
            for ka, part in (('a', 'w'), ('m', 'n')):
                truth = {(int(a), int(b)): (int(c), int(sx), float(ay)) for a, b, c, sx, ay in con.execute(
                    f"SELECT {ka}, {part}, COUNT(*), SUM(x), AVG(y) FROM '{pq}'{where} GROUP BY {ka}, {part}").fetchall()}
                for k in (1, 10, 100):
                    sql = ("SELECT %s, %s, COUNT(*) AS c, SUM(x), AVG(y) FROM hits%s GROUP BY %s, %s "
                           "ORDER BY c DESC LIMIT %d" % (ka, part, where, ka, part, k))
                    got = {}
                    try:
                        for line in (0.0, 1e18):
                            wdb_pairfold.HT_LINE = line
                            h0 = wdb_pairfold._HT_HITS[0]
                            rows = db.run(sql)[0]
                            _legal5(rows, truth, min(k, len(truth)))
                            got[line] = ([int(r[2]) for r in rows], wdb_pairfold._HT_HITS[0] - h0)
                    finally:
                        wdb_pairfold.HT_LINE = old[0]
                    assert got[0.0][0] == got[1e18][0]
                    assert got[0.0][1] == 0, 'line 0: the one pass must never run'
                    ran += got[1e18][1]
        assert ran > 0, 'past the line the one pass must answer some of them'
    finally:
        wdb_pairfold.HT_HEAVY = old[1]
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_heavy_share_line():
    """THE HEAVY SHARE: a big key whose one value holds 99% of the rows is left to the heavy fold (None), the same key
    spread out is the one pass's -- the sample's share against HT_HEAVY"""
    rng = np.random.default_rng(64)
    old = wdb_pairfold.HT_LINE, wdb_pairfold.HT_HEAVY
    try:
        wdb_pairfold.HT_LINE = 1e18
        wdb_pairfold.HT_HEAVY = 0.9
        n = 400_000
        ac = rng.integers(0, 30, n).astype(np.uint8)
        z = np.zeros(n, np.uint8)
        heavy = np.where(rng.random(n) < 0.99, 0, rng.integers(1, 5000, n)).astype(np.uint32)
        spread = rng.integers(0, 5000, n).astype(np.uint32)
        assert wdb_pairfold._onepass(None, ac, heavy, z, z, np.zeros(1), np.zeros(1), 5, 1, 4) is None
        r = wdb_pairfold._onepass(None, ac, spread, z, z, np.zeros(1), np.zeros(1), 5, 1, 4)
        assert r is not None
        _check(r, _truth(None, ac, spread, z, z, np.zeros(1), np.zeros(1), 5), 1)
        pos = np.arange(0, n, 3, dtype=np.int64)           # under a filter the line does not apply
        r = wdb_pairfold._onepass(pos, ac, heavy, z, z, np.zeros(1), np.zeros(1), 5, 1, 4)
        assert r is not None
        _check(r, _truth(pos, ac, heavy, z, z, np.zeros(1), np.zeros(1), 5), 1)
        wdb_pairfold.HT_HEAVY = 2.0
        r = wdb_pairfold._onepass(None, ac, heavy, z, z, np.zeros(1), np.zeros(1), 5, 1, 4)
        _check(r, _truth(None, ac, heavy, z, z, np.zeros(1), np.zeros(1), 5), 1)
    finally:
        wdb_pairfold.HT_LINE, wdb_pairfold.HT_HEAVY = old
