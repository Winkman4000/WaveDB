"""THE PAIR BOUND (wdb_pairbound): SELECT a, b, COUNT(*) AS c [, SUM(x), AVG(y)] FROM t GROUP BY a, b ORDER BY c DESC
LIMIT k with two many-valued keys -- only the rows whose two values both have >= T rows are counted, certified when the
k-th exact count reaches T. Legal against DuckDB (counts, SUM, AVG; ties at the k-th count may differ), for a good and
a bad first guess (too high: lowered; 1: every row), and the shapes it must leave to other reads."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_pairbound
from wdb_db import Database
from wdb_catalog import Catalog
from wdb_engine import Segment

TMP = tempfile.gettempdir()


def _db(n=400_000, seed=32):
    """a: ~20K values, b: ~30K values, both skewed (a few heavy values, a long tail) so real pairs repeat; u: a value
    per row (pairtop's); x, y: the summed columns"""
    rng = np.random.default_rng(seed)
    a = (rng.zipf(1.3, n) % 20_000).astype(np.int64)
    b = ((a * 7 + rng.zipf(1.4, n)) % 30_000).astype(np.int64)
    u = rng.permutation(n).astype(np.int64)
    df = pd.DataFrame({'a': a, 'b': b, 'u': u, 'x': rng.integers(0, 2, n).astype(np.int64),
                       'y': rng.integers(0, 3000, n).astype(np.int64)})
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'pb_{t}'); pq = os.path.join(TMP, f'pb_{t}.parquet')
    df.to_parquet(pq, index=False)
    os.makedirs(d); Catalog.create(d)
    out = os.path.join(d, 'hits_0.wdb')
    wdb_encode.encode(pq, out, stream=True)
    seg = Segment(out); cat = Catalog.open(d)
    tn = {0: 'int', 1: 'str', 2: 'float'}
    cat.data['tables']['hits'] = {'schema': [[c, tn.get(seg.cols[c].get('dt'), 'str')]
                                             for c in wdb_encode.input_column_order(pq, seg.order)],
                                  'segments': ['hits_0.wdb'], 'mode': 'segment'}
    cat.save()
    return d, pq, seg


def _legal(rows, truth, k, with_sums):
    want = sorted((v[0] for v in truth.values()), reverse=True)[:k]
    cs = [int(r[2]) for r in rows]
    assert cs == want, (cs, want)
    for r in rows:
        tc, tsx, tay = truth[(int(r[0]), int(r[1]))]
        assert int(r[2]) == tc, (r, tc)
        if with_sums:
            assert int(r[3]) == tsx, (r, tsx)
            assert abs(float(r[4]) - tay) < 1e-9 * max(1.0, abs(tay)), (r, tay)
    assert len({(int(r[0]), int(r[1])) for r in rows}) == len(rows)


def test_pair_bound_legal_against_duck():
    d, pq, seg = _db()
    old = wdb_pairbound.LINE, wdb_pairbound._guess
    try:
        wdb_pairbound.LINE = 'all'
        db = Database.open(d)
        con = duckdb.connect()
        truth = {(int(a), int(b)): (int(c), int(sx), float(ay)) for a, b, c, sx, ay in con.execute(
            f"SELECT a, b, COUNT(*), SUM(x), AVG(y) FROM '{pq}' GROUP BY a, b").fetchall()}
        for guess in (None, lambda *z: 10 ** 9, lambda *z: 1):
            wdb_pairbound._guess = old[1] if guess is None else guess
            for k in (1, 10, 100):
                for sums in (True,):
                    sel = 'a, b, COUNT(*) AS c, SUM(x), AVG(y)' if sums else 'a, b, COUNT(*) AS c'
                    sql = 'SELECT %s FROM hits GROUP BY a, b ORDER BY c DESC LIMIT %d' % (sel, k)
                    h0 = wdb_pairbound._HITS
                    rows = db.run(sql)[0]
                    assert wdb_pairbound._HITS == h0 + 1, ('pairbound did not serve', sql)
                    _legal(rows, truth, min(k, len(truth)), sums)
            if guess is not None and guess() > 1:
                assert wdb_pairbound._ROUNDS[0] >= 2, 'a too-high guess must be lowered'
    finally:
        wdb_pairbound.LINE, wdb_pairbound._guess = old
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_pair_bound_line():
    """THE LINE on the column counts: the measured cells in, the rest out"""
    L = wdb_pairbound._on_line
    assert L(1, 1 << 24) and L(1 << 24, 5) and L(1 << 16, 1 << 22) and L(1 << 10, 89_907)
    assert not L(397, 6) and not L(1000, 1 << 21) and not L(1 << 15, 1 << 23) and not L(1 << 9, 89_907)
    assert not L(1 << 16, 1 << 21)


def test_pair_bound_leaves_other_shapes():
    d, pq, seg = _db(n=200_000, seed=7)
    old = wdb_pairbound.LINE
    try:
        wdb_pairbound.LINE = 'all'
        db = Database.open(d)
        for sql in ("SELECT a, b, COUNT(*) AS c FROM hits WHERE x = 1 GROUP BY a, b ORDER BY c DESC LIMIT 10",   # a WHERE
                    "SELECT a, u, COUNT(*) AS c FROM hits GROUP BY a, u ORDER BY c DESC LIMIT 10",               # near-unique
                    "SELECT a, b, COUNT(*) AS c FROM hits GROUP BY a, b ORDER BY c ASC LIMIT 10",                # ascending
                    "SELECT a, b, COUNT(*) AS c FROM hits GROUP BY a, b ORDER BY c DESC LIMIT 10"):              # COUNT only
            h0 = wdb_pairbound._HITS
            db.run(sql)
            assert wdb_pairbound._HITS == h0, ('pairbound took a shape it must leave', sql)
        wdb_pairbound.LINE = 'line'                    # the measured line: these keys (<= 30K values) are too small
        h0 = wdb_pairbound._HITS
        db.run("SELECT a, b, COUNT(*) AS c, SUM(x) FROM hits GROUP BY a, b ORDER BY c DESC LIMIT 10")
        assert wdb_pairbound._HITS == h0
    finally:
        wdb_pairbound.LINE = old
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)
